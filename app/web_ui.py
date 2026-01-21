#!/usr/bin/env python3
"""
Streamlit 기반 보안 모니터링 Web UI
====================================
원격 접속 가능한 웹 대시보드

기능:
- 실시간 비디오 스트림
- 이상 점수 차트
- 이벤트 타임라인
- 설정 패널
- 통계 대시보드
"""

import os
import sys
import time
import json
import logging
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from collections import deque

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 전역 스레드 안전 큐 (Streamlit 세션 외부)
_frame_update_queue = queue.Queue(maxsize=20)

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from app.e2e_system import (
    E2ESystem, SystemConfig, VideoSourceType, VADModelType, AgentFlowType,
    AnomalyEvent, SystemStats
)
from app.ui_components.video_overlay import VideoOverlayRenderer


# =============================================================================
# 세션 상태 관리
# =============================================================================

def init_session_state():
    """세션 상태 초기화"""
    if 'system' not in st.session_state:
        st.session_state.system = None
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'recent_scores' not in st.session_state:
        st.session_state.recent_scores = deque(maxlen=100)
    if 'recent_events' not in st.session_state:
        st.session_state.recent_events = []
    if 'current_frame' not in st.session_state:
        st.session_state.current_frame = None
    if 'current_score' not in st.session_state:
        st.session_state.current_score = 0.0
    if 'stats' not in st.session_state:
        st.session_state.stats = {}
    if 'frame_queue' not in st.session_state:
        st.session_state.frame_queue = queue.Queue(maxsize=10)
    if 'overlay_renderer' not in st.session_state:
        st.session_state.overlay_renderer = None
    if 'last_vlm_result' not in st.session_state:
        st.session_state.last_vlm_result = None
    if 'last_agent_actions' not in st.session_state:
        st.session_state.last_agent_actions = None
    if 'frame_number' not in st.session_state:
        st.session_state.frame_number = 0
    if 'uploaded_file_path' not in st.session_state:
        st.session_state.uploaded_file_path = None


# =============================================================================
# 파일 업로드 유틸리티
# =============================================================================

def save_uploaded_file(uploaded_file) -> Optional[str]:
    """업로드된 파일을 임시 디렉토리에 저장하고 경로 반환"""
    try:
        uploads_dir = PROJECT_ROOT / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        # 고유 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        original_name = Path(uploaded_file.name).stem
        file_extension = Path(uploaded_file.name).suffix
        saved_filename = f"{timestamp}_{original_name}{file_extension}"
        saved_path = uploads_dir / saved_filename
        
        # 파일 저장
        with open(saved_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return str(saved_path)
    except Exception as e:
        # logging이 없을 수 있으므로 print 사용
        print(f"Error saving uploaded file: {e}", file=sys.stderr)
        return None


# =============================================================================
# 콜백 함수
# =============================================================================

def on_frame_update(frame, score: float):
    """프레임 업데이트 콜백 - 스레드 안전한 전역 큐 사용"""
    # 로깅 설정 확인 (백그라운드 스레드에서도 작동하도록)
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        # 핸들러가 없으면 기본 설정
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    
    # 즉시 로그 추가 (함수 진입 확인)
    frame_counter = getattr(on_frame_update, '_frame_counter', 0) + 1
    on_frame_update._frame_counter = frame_counter
    
    # 함수 진입 로그 (항상 기록) - 여러 방법으로 로깅
    log_msg = f"[CALLBACK ENTRY] Frame {frame_counter}, score={score:.4f}, HAS_CV2={HAS_CV2}, frame is None={frame is None}"
    logger.info(log_msg)
    print(f"[CALLBACK ENTRY] {log_msg}")  # print도 사용 (백그라운드 스레드에서도 보임)
    
    try:
        # 프레임 유효성 검사 (즉시)
        if frame is None:
            logging.error(f"[CALLBACK ERROR] Frame {frame_counter} is None")
            return
        
        if not HAS_CV2:
            logging.error(f"[CALLBACK ERROR] OpenCV not available for frame {frame_counter}")
            return
        
        # 프레임 타입 및 shape 검증
        try:
            frame_shape = frame.shape if hasattr(frame, 'shape') else 'No shape'
            frame_dtype = frame.dtype if hasattr(frame, 'dtype') else 'No dtype'
            frame_size = frame.size if hasattr(frame, 'size') else 0
            logging.info(f"[CALLBACK] Frame {frame_counter} validation: shape={frame_shape}, dtype={frame_dtype}, size={frame_size}")
        except Exception as e:
            logging.error(f"[CALLBACK ERROR] Frame {frame_counter} validation failed: {e}")
            return
        
        # 로깅 강화 (처음 20프레임은 모두 로그)
        if frame_counter <= 20 or frame_counter % 10 == 0:
            queue_size_before = _frame_update_queue.qsize()
            logging.info(f"[CALLBACK] Frame {frame_counter}: score={score:.4f}, queue_size={queue_size_before}, frame_shape={frame_shape}")
        
        # 프레임을 JPEG로 인코딩 (백그라운드 스레드에서)
        if HAS_CV2 and frame is not None:
            try:
                logging.info(f"[ENCODING START] Frame {frame_counter}: Starting encoding process")
                
                # 프레임 유효성 검사
                if frame.size == 0:
                    logging.error(f"[ENCODING ERROR] Frame {frame_counter} is empty (size=0)")
                    return
                
                # 오버레이 적용 (간단한 버전 - 세션 상태 접근 최소화)
                try:
                    overlay_frame = frame.copy()
                    logging.info(f"[ENCODING] Frame {frame_counter}: Copy successful, shape={overlay_frame.shape}")
                except Exception as e:
                    logging.error(f"[ENCODING ERROR] Frame {frame_counter}: Copy failed: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
                    return
                
                # 기본 오버레이 정보만 추가 (세션 상태 접근 없이)
                h, w = overlay_frame.shape[:2]
                if h > 0 and w > 0:
                    try:
                        cv2.putText(overlay_frame, f"Frame: {frame_counter}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(overlay_frame, f"Score: {score:.4f}", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        logging.info(f"[ENCODING] Frame {frame_counter}: Overlay text added")
                    except Exception as e:
                        logging.warning(f"[ENCODING WARNING] Frame {frame_counter}: Overlay text failed: {e}")
                        # 오버레이 실패해도 계속 진행
                    
                    # JPEG로 인코딩
                    try:
                        encode_result = cv2.imencode('.jpg', overlay_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if encode_result[0]:
                            frame_bytes = encode_result[1].tobytes()
                            logging.info(f"[ENCODING SUCCESS] Frame {frame_counter}: Encoded to {len(frame_bytes)} bytes")
                            
                            # 전역 스레드 안전 큐에 추가
                            try:
                                _frame_update_queue.put_nowait({
                                    'frame_bytes': frame_bytes,
                                    'score': score,
                                    'frame_number': frame_counter
                                })
                                # 큐에 추가 성공 로그 (처음 20프레임)
                                queue_size_after = _frame_update_queue.qsize()
                                if frame_counter <= 20:
                                    logging.info(f"[QUEUE SUCCESS] Frame {frame_counter} added to queue, queue_size={queue_size_after}, bytes={len(frame_bytes)}")
                                else:
                                    logging.debug(f"[QUEUE SUCCESS] Frame {frame_counter} added to queue, queue_size={queue_size_after}")
                            except queue.Full:
                                # 큐가 가득 차면 오래된 항목 제거
                                logging.warning(f"[QUEUE FULL] Frame {frame_counter}: Queue is full, removing old frame")
                                try:
                                    old_frame = _frame_update_queue.get_nowait()
                                    _frame_update_queue.put_nowait({
                                        'frame_bytes': frame_bytes,
                                        'score': score,
                                        'frame_number': frame_counter
                                    })
                                    queue_size_after = _frame_update_queue.qsize()
                                    logging.warning(f"[QUEUE REPLACED] Frame {frame_counter}: Replaced old frame (was #{old_frame.get('frame_number', 'unknown')}), queue_size={queue_size_after}")
                                except queue.Empty:
                                    logging.error(f"[QUEUE ERROR] Frame {frame_counter}: Queue was full but now empty (race condition?)")
                                    # 다시 시도
                                    try:
                                        _frame_update_queue.put_nowait({
                                            'frame_bytes': frame_bytes,
                                            'score': score,
                                            'frame_number': frame_counter
                                        })
                                        logging.info(f"[QUEUE RETRY SUCCESS] Frame {frame_counter}: Successfully added on retry")
                                    except Exception as e2:
                                        logging.error(f"[QUEUE RETRY ERROR] Frame {frame_counter}: Retry failed: {e2}")
                            except Exception as e:
                                logging.error(f"[QUEUE ERROR] Frame {frame_counter}: Failed to add to queue: {e}")
                                import traceback
                                logging.error(traceback.format_exc())
                        else:
                            logging.error(f"[ENCODING ERROR] Frame {frame_counter}: JPEG encoding failed (encode_result[0]={encode_result[0]})")
                    except Exception as e:
                        logging.error(f"[ENCODING ERROR] Frame {frame_counter}: JPEG encoding exception: {e}")
                        import traceback
                        logging.error(traceback.format_exc())
                else:
                    logging.error(f"[ENCODING ERROR] Frame {frame_counter}: Invalid dimensions (h={h}, w={w})")
            except Exception as e:
                logging.error(f"[ENCODING ERROR] Frame {frame_counter}: Encoding process exception: {e}")
                import traceback
                logging.error(traceback.format_exc())
        else:
            if not HAS_CV2:
                logging.error(f"[CALLBACK ERROR] Frame {frame_counter}: OpenCV not available")
            if frame is None:
                logging.error(f"[CALLBACK ERROR] Frame {frame_counter}: Frame is None")
    except Exception as e:
        logging.error(f"Frame callback error: {e}")
        import traceback
        logging.error(traceback.format_exc())


def on_anomaly_update(event: AnomalyEvent):
    """이상 감지 콜백"""
    st.session_state.recent_events.insert(0, event)
    if len(st.session_state.recent_events) > 20:
        st.session_state.recent_events = st.session_state.recent_events[:20]
    
    # VLM 및 Agent 결과 저장 (오버레이용)
    if event.vlm_type and event.vlm_type != "Unknown":
        st.session_state.last_vlm_result = {
            'detected_type': event.vlm_type,
            'description': event.vlm_description or '',
            'confidence': event.vlm_confidence
        }
    
    if event.agent_actions:
        st.session_state.last_agent_actions = event.agent_actions


def on_stats_update(stats: SystemStats):
    """통계 업데이트 콜백"""
    st.session_state.stats = stats.to_dict()


# =============================================================================
# UI 컴포넌트
# =============================================================================

def render_sidebar():
    """사이드바 설정 패널 (개선된 버전)"""
    st.sidebar.title("⚙️ Settings")
    
    # 데모 모드 선택
    st.sidebar.subheader("🎬 Demo Mode")
    demo_mode = st.sidebar.checkbox("Enable Demo Mode", value=False)
    
    if demo_mode:
        try:
            from app.demo_config import DEMO_PRESETS, get_demo_videos, get_preset
            
            preset_names = list(DEMO_PRESETS.keys())
            selected_preset = st.sidebar.selectbox(
                "Demo Preset",
                preset_names,
                index=0
            )
            
            if selected_preset:
                preset = get_preset(selected_preset)
                if preset:
                    st.sidebar.info(f"**{preset.name}**\n\n{preset.description}")
                    
                    # 프리셋 적용 버튼
                    if st.sidebar.button("Apply Preset", use_container_width=True):
                        st.session_state['demo_preset'] = preset
                        st.sidebar.success("Preset applied!")
            
            # 데모 비디오 선택
            demo_videos = get_demo_videos()
            if demo_videos:
                st.sidebar.markdown("### 📁 Demo Videos")
                video_options = [f"{v['name']} ({v['type']})" for v in demo_videos]
                selected_video_idx = st.sidebar.selectbox(
                    "Select Demo Video",
                    range(len(video_options)),
                    format_func=lambda x: video_options[x] if x < len(video_options) else ""
                )
                
                if selected_video_idx < len(demo_videos):
                    st.session_state['demo_video_path'] = demo_videos[selected_video_idx]['path']
        except ImportError:
            st.sidebar.warning("Demo config not available")
    
    st.sidebar.divider()
    
    # 비디오 소스
    st.sidebar.subheader("📹 Video Source")
    source_type = st.sidebar.selectbox(
        "Source Type",
        ["file", "rtsp", "webcam"],
        index=0
    )
    
    # 데모 비디오 경로 자동 설정
    if demo_mode and 'demo_video_path' in st.session_state:
        default_path = st.session_state['demo_video_path']
    else:
        default_path = "/path/to/video.mp4"
    
    if source_type == "file":
        # 파일 입력 방식 선택
        upload_option = st.sidebar.radio(
            "File Input Method",
            ["Upload File", "File Path"],
            index=1 if st.session_state.uploaded_file_path is None else 0,
            help="Upload a local file or enter a file path"
        )
        
        if upload_option == "Upload File":
            uploaded_file = st.sidebar.file_uploader(
                "Choose video file",
                type=['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'],
                help="Upload a video file from your local computer"
            )
            
            if uploaded_file is not None:
                # 파일 저장
                with st.sidebar.spinner("Saving uploaded file..."):
                    saved_path = save_uploaded_file(uploaded_file)
                    if saved_path:
                        st.session_state.uploaded_file_path = saved_path
                        st.sidebar.success(f"File saved: {uploaded_file.name}")
                        source_path = saved_path
                    else:
                        st.sidebar.error("Failed to save uploaded file")
                        source_path = default_path
            elif st.session_state.uploaded_file_path:
                # 이전에 업로드된 파일이 있으면 사용
                source_path = st.session_state.uploaded_file_path
                st.sidebar.info(f"Using previously uploaded file")
            else:
                source_path = default_path
        else:
            # 기존 텍스트 입력 방식
            source_path = st.sidebar.text_input(
                "File Path",
                value=default_path
            )
            
            # 파일 경로 검증
            if source_path and source_path != default_path:
                if os.path.exists(source_path):
                    if os.path.isfile(source_path):
                        # 비디오 파일 확장자 확인
                        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
                        if any(source_path.lower().endswith(ext) for ext in valid_extensions):
                            st.sidebar.success("✓ Valid video file")
                        else:
                            st.sidebar.warning("⚠ File extension may not be supported")
                    else:
                        st.sidebar.error("✗ Path is not a file")
                else:
                    st.sidebar.error("✗ File does not exist")
            
            # 텍스트 입력 사용 시 업로드 파일 경로 초기화
            if st.session_state.uploaded_file_path:
                st.session_state.uploaded_file_path = None
    elif source_type == "rtsp":
        source_path = st.sidebar.text_input(
            "RTSP URL",
            value="rtsp://192.168.1.100:554/stream"
        )
    else:
        source_path = st.sidebar.selectbox(
            "Webcam Index",
            ["0", "1", "2"],
            index=0
        )
    
    # VAD 설정 (llama 환경 모델만 - attribute_based_aivad 제외)
    st.sidebar.subheader("🔍 VAD Settings")
    vad_model = st.sidebar.selectbox(
        "VAD Model",
        ["stead", "stae", "mnad", "memae"],  # llama 환경 모델만
        index=0,
        help="Select VAD model (llama environment models only)"
    )
    
    threshold = st.sidebar.slider(
        "Anomaly Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )
    
    # VLM 설정
    st.sidebar.subheader("🤖 VLM Settings")
    enable_vlm = st.sidebar.checkbox("Enable VLM Analysis", value=True)
    
    vlm_frames = 4
    optimize_vlm = True
    if enable_vlm:
        vlm_frames = st.sidebar.slider(
            "VLM Frames",
            min_value=1,
            max_value=16,
            value=4
        )
        optimize_vlm = st.sidebar.checkbox("Optimize VLM (Speed)", value=True)
    
    # Agent 설정
    st.sidebar.subheader("🤖 Agent Settings")
    enable_agent = st.sidebar.checkbox("Enable Agent", value=True)
    
    agent_flow = "sequential"
    if enable_agent:
        agent_flow = st.sidebar.selectbox(
            "Agent Flow",
            ["sequential", "hierarchical", "collaborative"],
            index=0
        )
    
    # 저장 설정
    st.sidebar.subheader("💾 Storage Settings")
    save_clips = st.sidebar.checkbox("Save Anomaly Clips", value=True)
    clip_duration = st.sidebar.slider(
        "Clip Duration (sec)",
        min_value=1.0,
        max_value=10.0,
        value=3.0
    )
    
    # GPU 설정
    st.sidebar.subheader("🖥️ GPU Settings")
    gpu_id = st.sidebar.number_input("GPU ID", min_value=0, max_value=7, value=2)
    
    return {
        "source_type": source_type,
        "source_path": source_path,
        "vad_model": vad_model,
        "threshold": threshold,
        "enable_vlm": enable_vlm,
        "vlm_frames": vlm_frames,
        "optimize_vlm": optimize_vlm,
        "enable_agent": enable_agent,
        "agent_flow": agent_flow,
        "save_clips": save_clips,
        "clip_duration": clip_duration,
        "gpu_id": gpu_id
    }


def render_video_panel():
    """비디오 스트림 패널 (개선된 버전)"""
    # 헤더 및 컨트롤
    col_header, col_controls = st.columns([3, 1])
    
    with col_header:
        st.subheader("📹 Live Video Stream")
    
    with col_controls:
        # 풀스크린 모드 (세션 상태로 관리)
        if 'fullscreen' not in st.session_state:
            st.session_state.fullscreen = False
        
        if st.button("🔍 Fullscreen", key="fullscreen_btn", use_container_width=True):
            st.session_state.fullscreen = not st.session_state.fullscreen
    
    # 비디오 플레이스홀더
    video_placeholder = st.empty()
    
    # 디버깅 정보 (개발 모드) - 사이드바에 표시
    if st.session_state.is_running:
        try:
            global_queue_size = _frame_update_queue.qsize()
            session_queue_size = st.session_state.frame_queue.qsize()
            current_frame = st.session_state.frame_number
            
            # 콜백 호출 횟수 확인 (여러 방법으로)
            callback_counter = getattr(on_frame_update, '_frame_counter', 0)
            
            # 시스템 프레임 번호 확인
            system_frame_num = 0
            if st.session_state.system and hasattr(st.session_state.system, 'stats'):
                system_frame_num = st.session_state.system.stats.total_frames
            
            # current_frame 존재 여부 확인
            has_session_frame = st.session_state.current_frame is not None
            has_system_frame = (st.session_state.system and 
                               hasattr(st.session_state.system, 'current_frame') and 
                               st.session_state.system.current_frame is not None)
            
            # 큐 상태 상세 정보
            queue_status = "🟢 OK" if global_queue_size > 0 else "🟡 Empty" if global_queue_size == 0 else "🔴 Error"
            
            # 콜백 상태
            callback_status = "🟢 Active" if callback_counter > 0 else "🟡 Not called" if callback_counter == 0 else "🔴 Error"
            
            # 항상 디버깅 정보 표시 (큐가 비어있어도)
            st.sidebar.markdown("### 🔍 Debug Info")
            st.sidebar.caption(
                f"**Queue Status:** {queue_status}\n"
                f"**Global queue:** {global_queue_size}\n"
                f"**Session queue:** {session_queue_size}\n"
                f"**Frame # (UI):** {current_frame}\n"
                f"**Frame # (System):** {system_frame_num}\n"
                f"**Callback Status:** {callback_status}\n"
                f"**Callback calls:** {callback_counter}\n"
                f"**Has session frame:** {has_session_frame}\n"
                f"**Has system frame:** {has_system_frame}\n"
                f"**System running:** {st.session_state.is_running}"
            )
            
            # 경고 표시
            if global_queue_size == 0 and callback_counter > current_frame + 5:
                st.sidebar.warning(f"⚠️ 큐가 비어있지만 콜백은 {callback_counter}번 호출됨. 프레임 인코딩 문제 가능성.")
            elif callback_counter == 0 and system_frame_num > 0:
                st.sidebar.warning(f"⚠️ 시스템은 {system_frame_num}개 프레임 처리했지만 콜백이 호출되지 않음.")
            elif has_system_frame and not has_session_frame:
                st.sidebar.info(f"ℹ️ 시스템 프레임은 있지만 세션 프레임이 없음. 대체 경로 사용 중.")
        except Exception as e:
            st.sidebar.error(f"Debug info error: {e}")
    
    # 프레임 표시
    try:
        frame_bytes = None
        frame_data = None
        
        # 전역 큐에서 최신 프레임 가져오기 (스레드 안전)
        # 오래된 프레임은 버리고 최신 것만 사용
        latest_frame_data = None
        try:
            while not _frame_update_queue.empty():
                try:
                    latest_frame_data = _frame_update_queue.get_nowait()
                except queue.Empty:
                    break
        except Exception as e:
            logging.warning(f"Error reading from queue: {e}")
            latest_frame_data = None
        
        if latest_frame_data:
            frame_data = latest_frame_data
            # 세션 상태 업데이트 (메인 스레드에서만)
            st.session_state.frame_number = frame_data.get('frame_number', 0)
            st.session_state.current_score = frame_data.get('score', 0.0)
            if 'recent_scores' not in st.session_state:
                st.session_state.recent_scores = deque(maxlen=100)
            st.session_state.recent_scores.append(frame_data.get('score', 0.0))
            frame_bytes = frame_data.get('frame_bytes')
        # 기존 세션 큐에서도 확인 (하위 호환성)
        elif not st.session_state.frame_queue.empty():
            # 모든 오래된 프레임 제거하고 최신 것만 유지
            latest_frame = None
            while not st.session_state.frame_queue.empty():
                latest_frame = st.session_state.frame_queue.get_nowait()
            if latest_frame:
                frame_bytes = latest_frame
        elif st.session_state.current_frame is not None and HAS_CV2:
            # 오버레이가 적용된 프레임 사용
            try:
                _, buffer = cv2.imencode('.jpg', st.session_state.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                logging.debug(f"Using current_frame from session_state, shape={st.session_state.current_frame.shape}")
            except Exception as e:
                logging.warning(f"Error encoding current_frame: {e}")
                frame_bytes = None
        # e2e_system의 current_frame도 확인 (콜백 실패 시 대체 경로)
        elif st.session_state.system and hasattr(st.session_state.system, 'current_frame') and st.session_state.system.current_frame is not None and HAS_CV2:
            try:
                system_frame = st.session_state.system.current_frame
                # 오버레이 추가
                overlay_frame = system_frame.copy()
                h, w = overlay_frame.shape[:2]
                if h > 0 and w > 0:
                    cv2.putText(overlay_frame, f"Frame: {st.session_state.frame_number}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(overlay_frame, f"Score: {st.session_state.current_score:.4f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    _, buffer = cv2.imencode('.jpg', overlay_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_bytes = buffer.tobytes()
                    logging.info(f"[FALLBACK] Using system.current_frame, shape={system_frame.shape}")
                    # session_state에도 저장
                    st.session_state.current_frame = system_frame
            except Exception as e:
                logging.warning(f"Error encoding system.current_frame: {e}")
                frame_bytes = None
        
        if frame_bytes:
            # 풀스크린 모드에 따라 크기 조정
            try:
                if st.session_state.fullscreen:
                    video_placeholder.image(frame_bytes, channels="BGR", use_container_width=False, width=None)
                else:
                    video_placeholder.image(frame_bytes, channels="BGR", use_container_width=True)
                
                # 비디오 정보 표시 (오버레이에 이미 포함되어 있지만 추가 정보)
                stats = st.session_state.stats if st.session_state.stats else {}
                col_info1, col_info2, col_info3 = st.columns(3)
                
                with col_info1:
                    st.caption(f"📊 Frame: {st.session_state.frame_number:,}")
                with col_info2:
                    fps = stats.get('current_fps', 0) if stats else 0
                    st.caption(f"⚡ FPS: {fps:.1f}")
                with col_info3:
                    score = st.session_state.current_score
                    threshold = st.session_state.system.config.vad_threshold if st.session_state.system else 0.5
                    score_status = "🔴 Anomaly" if score >= threshold else "🟢 Normal"
                    st.caption(f"{score_status} | Score: {score:.4f}")
            except Exception as e:
                logging.error(f"Error displaying frame: {e}")
                video_placeholder.error(f"Error displaying frame: {e}")
        else:
            # 프레임이 없을 때 상태 표시
            if st.session_state.is_running:
                # 디버깅 정보 추가
                try:
                    global_queue_size = _frame_update_queue.qsize()
                    system_running = st.session_state.system is not None
                    callback_set = st.session_state.system.on_frame_callback is not None if system_running else False
                    callback_counter = getattr(on_frame_update, '_frame_counter', 0)
                    current_frame = st.session_state.frame_number
                    
                    # 시스템 프레임 번호 확인
                    system_frame_num = 0
                    if st.session_state.system and hasattr(st.session_state.system, 'stats'):
                        system_frame_num = st.session_state.system.stats.total_frames
                    
                    # 상태 분석
                    if callback_counter == 0 and system_frame_num == 0:
                        status_msg = "시스템 시작 중..."
                    elif callback_counter == 0 and system_frame_num > 0:
                        status_msg = f"⚠️ 시스템은 {system_frame_num}개 프레임 처리했지만 콜백이 호출되지 않음"
                    elif callback_counter > current_frame + 10:
                        status_msg = f"⚠️ 콜백은 {callback_counter}번 호출되었지만 큐에 프레임이 없음 (인코딩 문제 가능성)"
                    elif global_queue_size == 0 and callback_counter > 0:
                        status_msg = f"⚠️ 콜백 {callback_counter}번 호출, 큐는 비어있음"
                    else:
                        status_msg = "프레임 처리 중..."
                    
                    debug_info = (
                        f"**상태:** {status_msg}\n\n"
                        f"**Global queue:** {global_queue_size}\n"
                        f"**Frame # (UI):** {current_frame}\n"
                        f"**Frame # (System):** {system_frame_num}\n"
                        f"**Callback calls:** {callback_counter}\n"
                        f"**System running:** {system_running}\n"
                        f"**Callback set:** {callback_set}"
                    )
                except Exception as e:
                    debug_info = f"Waiting for frames... (Frame #: {st.session_state.frame_number}, Error: {e})"
                video_placeholder.warning(f"⏳ Processing frames... Please wait.\n\n{debug_info}")
            else:
                video_placeholder.info("🎬 Video stream will appear here. Click 'Start' to begin.")
    except Exception as e:
        logging.error(f"Video panel error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        video_placeholder.error(f"Video error: {e}")
        st.error(traceback.format_exc())


def render_stats_panel():
    """통계 패널 (개선된 버전)"""
    st.subheader("📊 System Statistics")
    
    stats = st.session_state.stats
    
    if not stats:
        st.info("Waiting for data...")
        return
    
    # 기본 통계
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Frames", f"{stats.get('total_frames', 0):,}")
    
    with col2:
        fps = stats.get('current_fps', 0)
        st.metric("Current FPS", f"{fps:.1f}")
    
    with col3:
        anomaly_count = stats.get('anomaly_count', 0)
        st.metric("Anomalies", anomaly_count)
    
    with col4:
        runtime = stats.get('runtime_seconds', 0)
        st.metric("Runtime", f"{int(runtime // 60)}m {int(runtime % 60)}s")
    
    st.divider()
    
    # 성능 메트릭
    st.markdown("### ⚡ Performance Metrics")
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        vad_time = stats.get('avg_vad_time_ms', 0)
        st.metric("VAD Time", f"{vad_time:.1f}ms")
        if vad_time > 0:
            st.progress(min(vad_time / 100.0, 1.0))  # 100ms를 최대값으로 가정
    
    with col6:
        vlm_time = stats.get('avg_vlm_time_ms', 0)
        st.metric("VLM Time", f"{vlm_time:.1f}ms")
        if vlm_time > 0:
            st.progress(min(vlm_time / 1000.0, 1.0))  # 1000ms를 최대값으로 가정
    
    with col7:
        agent_time = stats.get('avg_agent_time_ms', 0)
        st.metric("Agent Time", f"{agent_time:.1f}ms")
        if agent_time > 0:
            st.progress(min(agent_time / 5000.0, 1.0))  # 5000ms를 최대값으로 가정
    
    # 추가 통계
    if stats.get('total_frames', 0) > 0:
        st.divider()
        st.markdown("### 📈 Additional Metrics")
        
        col8, col9 = st.columns(2)
        
        with col8:
            # 이상 감지율
            detection_rate = (anomaly_count / stats.get('total_frames', 1)) * 100
            st.metric("Detection Rate", f"{detection_rate:.2f}%")
        
        with col9:
            # 메모리 사용량 (있는 경우)
            memory_mb = stats.get('memory_usage_mb', 0)
            if memory_mb > 0:
                st.metric("Memory Usage", f"{memory_mb:.1f} MB")
            else:
                st.metric("Status", "🟢 Running")


def render_score_chart():
    """점수 차트 (Plotly 기반 인터랙티브)"""
    st.subheader("📈 Anomaly Score Timeline")
    
    scores = list(st.session_state.recent_scores)
    
    if not scores:
        st.info("Waiting for data...")
        return
    
    # Plotly 사용 시도, 실패하면 기본 차트
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        
        threshold = st.session_state.system.config.vad_threshold if st.session_state.system else 0.5
        
        # 데이터 준비
        frames = list(range(len(scores)))
        
        # Plotly Figure 생성
        fig = go.Figure()
        
        # 점수 라인
        fig.add_trace(go.Scatter(
            x=frames,
            y=scores,
            mode='lines',
            name='Anomaly Score',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='Frame: %{x}<br>Score: %{y:.4f}<extra></extra>'
        ))
        
        # 임계값 라인
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold ({threshold:.4f})",
            annotation_position="right"
        )
        
        # 이상 구간 하이라이트
        anomaly_frames = [i for i, s in enumerate(scores) if s >= threshold]
        if anomaly_frames:
            for frame_idx in anomaly_frames:
                if frame_idx < len(scores):
                    fig.add_vline(
                        x=frame_idx,
                        line_width=1,
                        line_dash="dot",
                        line_color="orange",
                        opacity=0.3
                    )
        
        # 레이아웃 설정
        fig.update_layout(
            height=300,
            xaxis_title="Frame",
            yaxis_title="Anomaly Score",
            hovermode='x unified',
            showlegend=True,
            margin=dict(l=40, r=20, t=20, b=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Y축 범위 설정
        if scores:
            y_min = min(min(scores), threshold * 0.5)
            y_max = max(max(scores), threshold * 1.5)
            fig.update_yaxes(range=[y_min, y_max])
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
    except ImportError:
        # Plotly가 없으면 기본 차트 사용
        import pandas as pd
        
        df = pd.DataFrame({
            "Frame": range(len(scores)),
            "Score": scores
        })
        
        threshold = st.session_state.system.config.vad_threshold if st.session_state.system else 0.5
        
        st.line_chart(df.set_index("Frame"))
        st.caption(f"Threshold: {threshold:.4f}")
    
    # 현재 점수 및 통계
    current = scores[-1] if scores else 0
    threshold = st.session_state.system.config.vad_threshold if st.session_state.system else 0.5
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if current >= threshold:
            st.error(f"⚠️ Current: {current:.4f}")
        else:
            st.success(f"✅ Current: {current:.4f}")
    
    with col2:
        if scores:
            st.metric("Min", f"{min(scores):.4f}")
    
    with col3:
        if scores:
            st.metric("Max", f"{max(scores):.4f}")


def render_events_panel():
    """이벤트 패널 (개선된 타임라인 뷰)"""
    st.subheader("⚠️ Recent Events")
    
    events = st.session_state.recent_events
    
    if not events:
        st.info("No anomalies detected yet")
        return
    
    # 최신 이벤트 알림
    if events:
        latest_event = events[0]
        threshold = st.session_state.system.config.vad_threshold if st.session_state.system else 0.5
        if latest_event.vad_score >= threshold:
            st.warning(
                f"🚨 **Latest Alert:** {latest_event.vlm_type} "
                f"at {latest_event.timestamp.strftime('%H:%M:%S')} "
                f"(Score: {latest_event.vad_score:.4f})"
            )
    
    # 이벤트 타임라인 컴포넌트 사용
    try:
        from app.ui_components.event_timeline import render_event_timeline
        render_event_timeline(events, max_events=5)
    except ImportError:
        # 폴백: 기본 이벤트 목록
        for event in events[:5]:
            with st.expander(
                f"🚨 {event.vlm_type} - {event.timestamp.strftime('%H:%M:%S')}",
                expanded=False
            ):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Score:** {event.vad_score:.3f}")
                    st.write(f"**Frame:** {event.frame_number}")
                    st.write(f"**Type:** {event.vlm_type}")
                    if event.vlm_confidence > 0:
                        st.write(f"**Confidence:** {event.vlm_confidence:.2f}")
                
                with col2:
                    st.write(f"**Description:** {event.vlm_description or 'N/A'}")
                    st.write(f"**Actions:** {len(event.agent_actions)}")
                    if event.agent_response_time > 0:
                        st.write(f"**Response Time:** {event.agent_response_time:.2f}s")
                
                if event.agent_actions:
                    st.write("**Recommended Actions:**")
                    for action in event.agent_actions:
                        if isinstance(action, dict):
                            priority = action.get('priority', 'N/A')
                            priority_color = {'high': '🔴', 'medium': '🟡', 'low': '🟢', 'critical': '🚨'}.get(priority.lower(), '⚪')
                            st.write(f"  {priority_color} {action.get('action', 'N/A')} (Priority: {priority})")
                        else:
                            st.write(f"  - {action}")
                
                if event.clip_path:
                    st.write(f"**Clip:** `{event.clip_path}`")
                    if st.button(f"View Clip", key=f"clip_{event.id}"):
                        st.video(event.clip_path)


def render_controls():
    """제어 버튼"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Start", disabled=st.session_state.is_running, use_container_width=True):
            return "start"
    
    with col2:
        if st.button("⏹️ Stop", disabled=not st.session_state.is_running, use_container_width=True):
            return "stop"
    
    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            return "reset"
    
    return None


# =============================================================================
# 시스템 제어
# =============================================================================

def start_system(settings: Dict) -> Tuple[bool, Optional[str]]:
    """시스템 시작
    
    Returns:
        (success: bool, error_message: Optional[str])
    """
    # 로깅 설정 확인 및 강화
    if not logging.getLogger().handlers:
        # 로깅 핸들러가 없으면 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(PROJECT_ROOT / "logs" / f"web_ui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
    
    # 소스 타입 변환
    source_type_map = {
        "file": VideoSourceType.FILE,
        "rtsp": VideoSourceType.RTSP,
        "webcam": VideoSourceType.WEBCAM
    }
    
    # 프레임 카운터 초기화
    on_frame_update._frame_counter = 0
    logging.info(f"[START SYSTEM] Initializing system, frame_counter reset to 0")
    
    # 전역 큐 초기화 (이전 프레임 제거)
    while not _frame_update_queue.empty():
        try:
            _frame_update_queue.get_nowait()
        except queue.Empty:
            break
    
    config = SystemConfig(
        source_type=source_type_map[settings["source_type"]],
        source_path=settings["source_path"],
        vad_model=VADModelType(settings["vad_model"]),
        vad_threshold=settings["threshold"],
        enable_vlm=settings["enable_vlm"],
        vlm_n_frames=settings["vlm_frames"],
        optimize_vlm=settings["optimize_vlm"],
        enable_agent=settings["enable_agent"],
        agent_flow=AgentFlowType(settings["agent_flow"]),
        save_clips=settings["save_clips"],
        clip_duration=settings["clip_duration"],
        clips_dir="clips",
        log_dir="logs",
        gpu_id=settings["gpu_id"],
        target_fps=30
    )
    
    system = E2ESystem(config)
    
    # 콜백 설정 (초기화 전에 설정)
    system.on_frame_callback = on_frame_update
    system.on_anomaly_callback = on_anomaly_update
    system.on_stats_callback = on_stats_update
    
    success, error_message = system.initialize()
    if success:
        # 콜백 재설정 (초기화 후에도 확인)
        system.on_frame_callback = on_frame_update
        system.on_anomaly_callback = on_anomaly_update
        system.on_stats_callback = on_stats_update
        
        # 콜백 설정 확인 및 로깅
        if system.on_frame_callback is None:
            logging.error("Frame callback is None after setting!")
            return False, "Frame callback 설정 실패"
        else:
            logging.info(f"Frame callback set successfully: {type(system.on_frame_callback).__name__}")
        
        # 세션 상태 초기화
        st.session_state.system = system
        st.session_state.is_running = True
        st.session_state.frame_number = 0
        st.session_state.current_score = 0.0
        
        # 백그라운드 스레드에서 실행
        thread = threading.Thread(target=system.start, daemon=True)
        thread.start()
        
        # 스레드 시작 확인을 위한 짧은 대기
        time.sleep(0.1)
        
        logging.info("System started in background thread")
        return True, None
    
    return False, error_message


def stop_system():
    """시스템 중지"""
    if st.session_state.system:
        st.session_state.system.stop()
        st.session_state.system = None
    
    st.session_state.is_running = False


def reset_session():
    """세션 리셋"""
    stop_system()
    st.session_state.recent_scores = deque(maxlen=100)
    st.session_state.recent_events = []
    st.session_state.current_frame = None
    st.session_state.current_score = 0.0
    st.session_state.stats = {}


# =============================================================================
# 메인 앱
# =============================================================================

def main():
    """Streamlit 메인 앱"""
    # 로깅 설정 (앱 시작 시)
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 로깅 설정 (Streamlit 환경에서도 작동하도록)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        # 파일 핸들러 추가
        file_handler = logging.FileHandler(
            log_dir / f"web_ui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # 콘솔 핸들러 추가
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter('%(levelname)s - %(message)s')
        )
        
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    st.set_page_config(
        page_title="E2E Security Monitoring",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 세션 상태 초기화
    init_session_state()
    
    # 헤더
    st.title("🔒 E2E Security Monitoring System")
    st.caption("Real-time Video Anomaly Detection with Agentic AI")
    
    # 사이드바 설정
    settings = render_sidebar()
    
    # 제어 버튼
    action = render_controls()
    
    if action == "start":
        with st.spinner("Starting system..."):
            success, error_msg = start_system(settings)
            if success:
                st.success("System started!")
            else:
                st.error(f"Failed to start system: {error_msg or 'Unknown error'}")
                # 추가 도움말 표시
                if error_msg:
                    if "File does not exist" in error_msg:
                        st.info("💡 Tip: Please check if the video file path is correct and the file exists.")
                    elif "File is not readable" in error_msg:
                        st.info("💡 Tip: Please check file permissions. The file may not be readable.")
                    elif "VAD model" in error_msg:
                        st.info("💡 Tip: The VAD model may not be properly initialized. Check GPU availability and model files.")
                    elif "Failed to open video source" in error_msg:
                        st.info("💡 Tip: Please verify the video source path and format.")
                
                # 로그 파일 경로 표시
                log_dir = PROJECT_ROOT / "logs"
                if log_dir.exists():
                    log_files = sorted(log_dir.glob("system_*.log"), key=os.path.getmtime, reverse=True)
                    if log_files:
                        st.info(f"📋 Check logs: `{log_files[0]}`")
    elif action == "stop":
        stop_system()
        st.info("System stopped")
    elif action == "reset":
        reset_session()
        st.info("Session reset")
    
    # 상태 표시
    if st.session_state.is_running:
        st.success("🟢 System Running")
    else:
        st.warning("🔴 System Stopped")
    
    st.divider()
    
    # 메인 레이아웃
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        render_video_panel()
        render_score_chart()
    
    with col_right:
        render_stats_panel()
        st.divider()
        render_events_panel()
    
    # 자동 새로고침 (실행 중일 때) - 최적화된 버전
    if st.session_state.is_running:
        # 전역 큐 확인 (우선순위 높음)
        has_global_frame = not _frame_update_queue.empty()
        has_new_frame = not st.session_state.frame_queue.empty()
        has_new_event = len(st.session_state.recent_events) > 0
        
        # 프레임 번호 추적
        last_frame_check = st.session_state.get('last_frame_check', 0)
        current_frame = st.session_state.frame_number
        frame_updated = current_frame > last_frame_check
        
        # 마지막 업데이트 시간 추적 (너무 빈번한 rerun 방지)
        last_rerun_time = st.session_state.get('last_rerun_time', 0)
        current_time = time.time()
        min_rerun_interval = 0.033  # 약 30 FPS (33ms)
        
        # 전역 큐에 프레임이 있으면 즉시 업데이트 (최소 간격 확인)
        if has_global_frame and (current_time - last_rerun_time) >= min_rerun_interval:
            st.session_state.last_rerun_time = current_time
            time.sleep(0.01)  # 매우 짧은 대기로 부하 감소
            st.rerun()
        # 세션 큐에 프레임이 있으면 즉시 업데이트
        elif has_new_frame and (current_time - last_rerun_time) >= min_rerun_interval:
            st.session_state.last_rerun_time = current_time
            time.sleep(0.01)
            st.rerun()
        # 프레임이 업데이트되었으면 업데이트
        elif frame_updated and (current_time - last_rerun_time) >= min_rerun_interval:
            st.session_state.last_frame_check = current_frame
            st.session_state.last_rerun_time = current_time
            time.sleep(0.05)  # 약간 긴 대기
            st.rerun()
        # 이벤트가 있으면 업데이트 (이벤트는 덜 빈번하므로 더 긴 간격 허용)
        elif has_new_event and (current_time - last_rerun_time) >= 0.1:
            st.session_state.last_rerun_time = current_time
            time.sleep(0.05)
            st.rerun()
        # 프레임이 없어도 시스템이 실행 중이면 주기적으로 확인 (더 긴 간격)
        elif (current_time - last_rerun_time) >= 0.5:
            st.session_state.last_rerun_time = current_time
            time.sleep(0.1)  # 시스템 상태 확인용 대기
            st.rerun()


if __name__ == "__main__":
    if not HAS_STREAMLIT:
        print("Error: Streamlit not installed. Run: pip install streamlit")
        sys.exit(1)
    
    main()

