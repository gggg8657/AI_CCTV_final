#!/usr/bin/env python3
"""
End-to-End 보안 모니터링 시스템 핵심 엔진
==========================================
VAD + VLM + Agent 통합 파이프라인

주요 기능:
- 다양한 비디오 소스 지원 (파일, RTSP, 웹캠)
- 실시간 이상 탐지 (VAD)
- VLM 기반 상황 분석
- Agent 기반 자동 대응
- 이벤트 로깅 및 클립 저장
"""

import os
import sys
import json
import time
import queue
import threading
import logging
import tempfile
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from collections import deque

# 조건부 import
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# 설정 및 상수
# =============================================================================

class VideoSourceType(Enum):
    """비디오 소스 타입"""
    FILE = "file"
    RTSP = "rtsp"
    WEBCAM = "webcam"


class VADModelType(Enum):
    """VAD 모델 타입"""
    STEAD = "stead"
    STAE = "stae"
    MNAD = "mnad"
    MEMAE = "memae"


class AgentFlowType(Enum):
    """Agent Flow 타입"""
    HIERARCHICAL = "hierarchical"
    SEQUENTIAL = "sequential"
    COLLABORATIVE = "collaborative"


@dataclass
class SystemConfig:
    """시스템 설정"""
    # 비디오 소스
    source_type: VideoSourceType = VideoSourceType.FILE
    source_path: str = ""
    
    # VAD 설정
    vad_model: VADModelType = VADModelType.STEAD
    vad_threshold: float = 0.5
    checkpoint_path: str = ""
    
    # VLM 설정
    enable_vlm: bool = True
    vlm_n_frames: int = 4
    vlm_model_path: str = ""
    optimize_vlm: bool = True
    
    # Agent 설정
    enable_agent: bool = True
    agent_flow: AgentFlowType = AgentFlowType.SEQUENTIAL
    
    # 클립 저장 설정
    save_clips: bool = True
    clip_duration: float = 3.0
    clips_dir: str = "clips"
    
    # 로깅 설정
    log_dir: str = "logs"
    log_level: str = "INFO"
    
    # GPU 설정
    gpu_id: int = 0
    
    # 처리 설정
    target_fps: int = 30
    display_width: int = 640
    display_height: int = 480


@dataclass
class AnomalyEvent:
    """이상 감지 이벤트"""
    id: str
    timestamp: datetime
    frame_number: int
    vad_score: float
    threshold: float
    
    # VLM 분석 결과
    vlm_type: str = "Unknown"
    vlm_description: str = ""
    vlm_confidence: float = 0.0
    
    # Agent 대응 결과
    agent_actions: List[Dict] = field(default_factory=list)
    agent_response_time: float = 0.0
    
    # 클립 정보
    clip_path: str = ""
    
    # 메타데이터
    metadata: Dict = field(default_factory=dict)


@dataclass
class SystemStats:
    """시스템 통계"""
    start_time: datetime = field(default_factory=datetime.now)
    total_frames: int = 0
    anomaly_count: int = 0
    current_fps: float = 0.0
    avg_vad_time: float = 0.0
    avg_vlm_time: float = 0.0
    avg_agent_time: float = 0.0
    memory_usage_mb: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "start_time": self.start_time.isoformat(),
            "runtime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "total_frames": self.total_frames,
            "anomaly_count": self.anomaly_count,
            "current_fps": round(self.current_fps, 2),
            "avg_vad_time_ms": round(self.avg_vad_time * 1000, 2),
            "avg_vlm_time_ms": round(self.avg_vlm_time * 1000, 2),
            "avg_agent_time_ms": round(self.avg_agent_time * 1000, 2),
            "memory_usage_mb": round(self.memory_usage_mb, 2)
        }


# =============================================================================
# 이벤트 로거
# =============================================================================

class EventLogger:
    """이벤트 로깅 시스템"""
    
    def __init__(self, log_dir: str, log_level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 로그 파일 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.event_log_path = self.log_dir / f"events_{timestamp}.json"
        self.system_log_path = self.log_dir / f"system_{timestamp}.log"
        
        # 이벤트 목록
        self.events: List[AnomalyEvent] = []
        
        # 시스템 로거 설정
        self.logger = logging.getLogger("E2ESystem")
        self.logger.setLevel(getattr(logging, log_level))
        
        # 파일 핸들러
        file_handler = logging.FileHandler(self.system_log_path, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(file_handler)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(console_handler)
    
    def log_event(self, event: AnomalyEvent):
        """이상 이벤트 로깅"""
        self.events.append(event)
        self._save_events()
        self.logger.warning(
            f"[ANOMALY] ID={event.id} Score={event.vad_score:.3f} "
            f"Type={event.vlm_type} Actions={len(event.agent_actions)}"
        )
    
    def log_info(self, message: str):
        self.logger.info(message)
    
    def log_warning(self, message: str):
        self.logger.warning(message)
    
    def log_error(self, message: str):
        self.logger.error(message)
    
    def _save_events(self):
        """이벤트 파일 저장"""
        events_data = []
        for event in self.events:
            event_dict = asdict(event)
            event_dict["timestamp"] = event.timestamp.isoformat()
            events_data.append(event_dict)
        
        with open(self.event_log_path, 'w', encoding='utf-8') as f:
            json.dump(events_data, f, indent=2, ensure_ascii=False)
    
    def get_recent_events(self, n: int = 10) -> List[AnomalyEvent]:
        """최근 이벤트 조회"""
        return self.events[-n:] if self.events else []


# =============================================================================
# 클립 저장기
# =============================================================================

class ClipSaver:
    """이상 클립 저장기"""
    
    def __init__(self, clips_dir: str, clip_duration: float = 3.0, fps: int = 30):
        self.clips_dir = Path(clips_dir)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.clip_duration = clip_duration
        self.fps = fps
        self.buffer_size = int(clip_duration * fps)
        
        # 프레임 버퍼
        self.frame_buffer: deque = deque(maxlen=self.buffer_size)
        self.lock = threading.Lock()
    
    def add_frame(self, frame):
        """프레임 버퍼에 추가"""
        with self.lock:
            self.frame_buffer.append(frame.copy())
    
    def save_clip(self, event_id: str) -> str:
        """클립 저장"""
        if not HAS_CV2:
            return ""
        
        with self.lock:
            if len(self.frame_buffer) < 10:
                return ""
            
            frames = list(self.frame_buffer)
        
        # 클립 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_path = self.clips_dir / f"clip_{event_id}_{timestamp}.mp4"
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(clip_path), fourcc, self.fps, (width, height))
        
        for frame in frames:
            writer.write(frame)
        
        writer.release()
        return str(clip_path)


# =============================================================================
# 비디오 소스
# =============================================================================

class VideoSource:
    """비디오 소스 관리자"""
    
    def __init__(self, source_type: VideoSourceType, source_path: str):
        self.source_type = source_type
        self.source_path = source_path
        self.cap = None
        self.fps = 30
        self.width = 640
        self.height = 480
        self.total_frames = 0
    
    def open(self) -> bool:
        """비디오 소스 열기"""
        if not HAS_CV2:
            logging.error("OpenCV not available")
            return False
        
        # 파일 타입인 경우 존재 여부 및 접근 권한 확인
        if self.source_type == VideoSourceType.FILE:
            if not os.path.exists(self.source_path):
                logging.error(f"Video file does not exist: {self.source_path}")
                return False
            if not os.path.isfile(self.source_path):
                logging.error(f"Path is not a file: {self.source_path}")
                return False
            if not os.access(self.source_path, os.R_OK):
                logging.error(f"Video file is not readable: {self.source_path}")
                return False
        
        # 소스 설정
        if self.source_type == VideoSourceType.WEBCAM:
            try:
                source = int(self.source_path) if self.source_path and self.source_path.isdigit() else 0
            except (ValueError, AttributeError):
                source = 0
        else:
            source = self.source_path
        
        # 비디오 캡처 열기
        try:
            self.cap = cv2.VideoCapture(source)
            
            # RTSP의 경우 타임아웃 설정
            if self.source_type == VideoSourceType.RTSP:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화
            
            if not self.cap.isOpened():
                logging.error(f"Failed to open video source: {source} (type: {self.source_type.value})")
                return False
            
            # 비디오 속성 읽기
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            
            # 테스트 프레임 읽기
            ret, test_frame = self.cap.read()
            if not ret:
                logging.warning(f"Could not read test frame from source: {source}")
                # 웹캠/RTSP의 경우 계속 진행 (연결 중일 수 있음)
                if self.source_type == VideoSourceType.FILE:
                    self.cap.release()
                    return False
            else:
                # 프레임을 다시 앞으로 되돌림 (파일의 경우)
                if self.source_type == VideoSourceType.FILE:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            logging.info(f"Video source opened successfully: {self.source_type.value}, fps={self.fps}, resolution={self.width}x{self.height}")
            return True
        except Exception as e:
            logging.error(f"Error opening video source: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    def read(self) -> Tuple[bool, Any]:
        """프레임 읽기"""
        if self.cap is None:
            return False, None
        return self.cap.read()
    
    def close(self):
        """비디오 소스 닫기"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def get_info(self) -> Dict:
        """비디오 정보"""
        return {
            "type": self.source_type.value,
            "path": self.source_path,
            "fps": self.fps,
            "resolution": f"{self.width}x{self.height}",
            "total_frames": self.total_frames
        }


# =============================================================================
# VAD 모델 래퍼
# =============================================================================

class VADWrapper:
    """VAD 모델 래퍼 - 실제 sci_v2 모델 사용"""
    
    def __init__(self, model_type: VADModelType, checkpoint_path: str = "", gpu_id: int = 0):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.gpu_id = gpu_id
        self.model = None
        self.device = None
        self._loaded = False
    
    def load(self) -> bool:
        """모델 로드 - sci_v2의 실제 모델 사용"""
        try:
            # sci_v2 모델 경로 추가
            sci_v2_src = PROJECT_ROOT / "sci_v2" / "src"
            if str(sci_v2_src) not in sys.path:
                sys.path.insert(0, str(sci_v2_src))
            
            from vad import create_model as create_vad_model
            
            # GPU 설정
            if HAS_TORCH:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
                # CUDA_VISIBLE_DEVICES 설정 후에는 항상 cuda:0으로 접근
                device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
                self.device = torch.device(device_str)
            else:
                device_str = "cpu"
                self.device = None
            
            # 실제 모델 생성 (llama 환경 모델만)
            model_name = self.model_type.value
            if model_name in ["stead", "stae", "mnad", "memae"]:
                self.model = create_vad_model(model_name)
                if hasattr(self.model, 'initialize'):
                    # device 파라미터 전달 (initialize()는 None 반환하므로 예외 처리)
                    try:
                        self.model.initialize(device_str)
                        self._loaded = True
                        logging.info(f"VAD model loaded: {model_name} on {device_str}")
                    except Exception as e:
                        logging.error(f"VAD model initialization error: {e}")
                        import traceback
                        logging.error(traceback.format_exc())
                        self._loaded = False
                else:
                    self._loaded = True
                    logging.info(f"VAD model created: {model_name}")
            else:
                logging.warning(f"Unsupported model for llama environment: {model_name}")
                self._loaded = False
            
            return self._loaded
        except Exception as e:
            logging.error(f"VAD model load error: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    def predict(self, frames: List) -> Tuple[float, float]:
        """
        이상 점수 예측 - 실제 모델 사용
        
        Returns:
            (anomaly_score, inference_time)
        """
        if not self._loaded or self.model is None:
            logging.warning("VAD model not loaded, returning dummy score")
            score = np.random.random() * 0.3 if HAS_NUMPY else 0.1
            return float(score), 0.001
        
        start_time = time.time()
        
        try:
            # 실제 모델 추론 - VADModel 인터페이스의 process_frame() 사용
            if len(frames) == 0:
                score = 0.1
            elif hasattr(self.model, 'process_frame'):
                # VADModel 인터페이스: process_frame 사용 (실제 모델 추론)
                latest_frame = frames[-1]
                if HAS_CV2 and isinstance(latest_frame, np.ndarray):
                    # 실제 모델의 process_frame() 호출
                    score = self.model.process_frame(latest_frame)
                    if score is None:
                        # 프레임 부족 시 기본값 (모델이 None 반환)
                        score = 0.1
                else:
                    score = 0.1
            else:
                # process_frame이 없는 경우 (구버전 호환)
                logging.warning(f"VAD model {type(self.model).__name__} does not have process_frame method")
                score = 0.5
        except Exception as e:
            logging.error(f"VAD predict error: {e}")
            import traceback
            logging.error(traceback.format_exc())
            score = np.random.random() * 0.3 if HAS_NUMPY else 0.1
        
        inference_time = time.time() - start_time
        return float(score), inference_time
    
    def _preprocess_frames(self, frames: List):
        """프레임 전처리"""
        if not HAS_TORCH or not HAS_NUMPY:
            return None
        
        processed = []
        for frame in frames:
            # 리사이즈 및 정규화
            frame_resized = cv2.resize(frame, (256, 256)) if HAS_CV2 else frame
            frame_norm = frame_resized.astype(np.float32) / 255.0
            frame_tensor = torch.from_numpy(frame_norm).permute(2, 0, 1)
            processed.append(frame_tensor)
        
        batch = torch.stack(processed).unsqueeze(0).to(self.device)
        return batch


# =============================================================================
# VLM 분석기 래퍼
# =============================================================================

class VLMWrapper:
    """VLM 분석기 래퍼 - sci_v2의 실제 VLMAnalyzer 사용"""
    
    def __init__(self, model_path: str = "", n_frames: int = 4, optimize: bool = True, gpu_id: int = 0):
        self.model_path = model_path
        self.n_frames = n_frames
        self.optimize = optimize
        self.gpu_id = gpu_id
        self.vlm_analyzer = None
        self._loaded = False
    
    def load(self) -> bool:
        """VLM 로드 - sci_v2의 실제 VLMAnalyzer 사용"""
        try:
            # sci_v2 모델 경로 추가
            sci_v2_src = PROJECT_ROOT / "sci_v2" / "src"
            if str(sci_v2_src) not in sys.path:
                sys.path.insert(0, str(sci_v2_src))
            
            from vlm import VLMAnalyzer
            
            # 기본 모델 경로 설정
            if not self.model_path:
                # sci_v2의 기본 경로 사용
                default_path = "/data/DJ/models/Qwen2.5-VL-7B-Instruct-q4_k_m.gguf"
                if not os.path.exists(default_path):
                    # 대체 경로
                    default_path = "/home/dongjukim/models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
                self.model_path = default_path if os.path.exists(default_path) else ""
            
            if not self.model_path or not os.path.exists(self.model_path):
                logging.warning(f"VLM model path not found: {self.model_path}")
                self._loaded = False
                return False
            
            # mmproj 경로 설정
            mmproj_path = self.model_path.replace(".gguf", "-mmproj-f16.gguf")
            if not os.path.exists(mmproj_path):
                # 대체 경로
                mmproj_path = "/data/DJ/models/Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf"
                if not os.path.exists(mmproj_path):
                    mmproj_path = "/home/dongjukim/models/Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf"
            
            # 실제 VLMAnalyzer 생성 및 초기화
            self.vlm_analyzer = VLMAnalyzer(
                model_path=self.model_path,
                mmproj_path=mmproj_path if os.path.exists(mmproj_path) else None,
                n_frames=self.n_frames,
                use_multiframe=True,
                optimize_speed=self.optimize,
                gpu_id=self.gpu_id
            )
            
            if self.vlm_analyzer.initialize():
                self._loaded = True
                logging.info("VLM loaded using sci_v2 VLMAnalyzer")
                return True
            else:
                logging.warning("VLM initialization failed")
                self._loaded = False
                return False
        except Exception as e:
            logging.error(f"VLM load error: {e}")
            import traceback
            logging.error(traceback.format_exc())
            self._loaded = False
            return False
    
    def analyze(self, frames: List, clip_path: str = "") -> Dict:
        """영상 분석 - sci_v2의 실제 VLMAnalyzer 사용"""
        start_time = time.time()
        
        result = {
            "detected_type": "Unknown",
            "description": "",
            "actions": [],
            "confidence": 0.0,
            "latency_ms": 0.0,
            "success": False
        }
        
        if not self._loaded or self.vlm_analyzer is None:
            result["detected_type"] = "Normal"
            result["description"] = "VLM not loaded"
            result["latency_ms"] = (time.time() - start_time) * 1000
            return result
        
        try:
            # sci_v2의 VLMAnalyzer.analyze() 호출
            if clip_path and os.path.exists(clip_path):
                vlm_result = self.vlm_analyzer.analyze(video_path=clip_path)
            elif frames:
                vlm_result = self.vlm_analyzer.analyze(frames=frames)
            else:
                result["description"] = "No frames or clip path provided"
                result["latency_ms"] = (time.time() - start_time) * 1000
                return result
            
            # VLMAnalysisResult를 Dict로 변환
            if hasattr(vlm_result, 'to_dict'):
                result = vlm_result.to_dict()
            elif isinstance(vlm_result, dict):
                result = vlm_result
            else:
                # VLMAnalysisResult 객체인 경우
                result["detected_type"] = getattr(vlm_result, 'detected_type', 'Unknown')
                result["description"] = getattr(vlm_result, 'description', '')
                result["actions"] = getattr(vlm_result, 'actions', [])
                result["confidence"] = getattr(vlm_result, 'confidence', 0.0)
                result["success"] = getattr(vlm_result, 'success', False)
                result["latency_ms"] = getattr(vlm_result, 'latency_ms', 0.0)
            
            result["success"] = True
        except Exception as e:
            result["description"] = f"VLM error: {str(e)}"
            logging.error(f"VLM analyze error: {e}")
            import traceback
            logging.error(traceback.format_exc())
        
        result["latency_ms"] = (time.time() - start_time) * 1000
        return result


# =============================================================================
# Agent 래퍼
# =============================================================================

class AgentWrapper:
    """Agent 시스템 래퍼 - sci_v2의 실제 Agent 사용"""
    
    def __init__(self, flow_type: AgentFlowType, gpu_id: int = 0):
        self.flow_type = flow_type
        self.gpu_id = gpu_id
        self.flow = None
        self._loaded = False
    
    def load(self) -> bool:
        """Agent Flow 로드 - sci_v2의 실제 Agent 사용"""
        try:
            # sci_v2 모델 경로 추가
            sci_v2_src = PROJECT_ROOT / "sci_v2" / "src"
            if str(sci_v2_src) not in sys.path:
                sys.path.insert(0, str(sci_v2_src))
            
            from agent import create_flow
            
            # Flow 이름 변환
            flow_name = self.flow_type.value  # "sequential", "hierarchical", "collaborative"
            
            # 실제 Agent Flow 생성
            self.flow = create_flow(flow_name, gpu_id=self.gpu_id)
            
            if self.flow:
                # 초기화 (llama.cpp 모델 로드 포함)
                if self.flow.initialize():
                    self._loaded = True
                    logging.info(f"Agent flow loaded: {flow_name} (using sci_v2 agent with llama.cpp)")
                    return True
                else:
                    logging.warning(f"Agent flow initialization failed: {flow_name}")
                    self._loaded = False
                    return False
        except Exception as e:
            logging.error(f"Agent load error: {e}")
            import traceback
            logging.error(traceback.format_exc())
        
        self._loaded = False
        return False
    
    def process(self, event: AnomalyEvent, vlm_result: Dict) -> Dict:
        """이상 이벤트 처리 - sci_v2의 실제 Agent 사용"""
        start_time = time.time()
        
        result = {
            "actions": [],
            "processing_time": 0.0,
            "success": False
        }
        
        if not self._loaded or self.flow is None:
            # Agent 미로드 시 기본 대응
            result["actions"] = self._get_default_actions(vlm_result.get("detected_type", "Unknown"))
            result["processing_time"] = time.time() - start_time
            return result
        
        try:
            # sci_v2의 실제 Agent Flow 실행 (llama.cpp 사용)
            # event.clip_path 또는 video_path 사용
            video_path = event.clip_path if event.clip_path and os.path.exists(event.clip_path) else None
            
            if not video_path:
                # 클립이 없으면 기본 대응
                result["actions"] = self._get_default_actions(vlm_result.get("detected_type", "Unknown"))
                result["processing_time"] = time.time() - start_time
                return result
            
            # 실제 Agent Flow 실행 (sci_v2의 Agent, llama.cpp 사용)
            agent_result = self.flow.run(video_path)
            
            # 결과 변환 (sci_v2 Agent 결과 형식 -> Web UI 형식)
            if isinstance(agent_result, dict):
                # Agent 결과에서 actions 추출
                if 'agent_plan' in agent_result:
                    plan = agent_result['agent_plan']
                    if isinstance(plan, dict) and 'actions' in plan:
                        actions = plan['actions']
                        result["actions"] = [
                            {
                                "action": action.get('action', '') if isinstance(action, dict) else str(action),
                                "priority": action.get('priority', 'medium') if isinstance(action, dict) else 'medium',
                                "description": action.get('description', '') if isinstance(action, dict) else ''
                            }
                            for action in actions
                        ]
                    else:
                        result["actions"] = self._get_default_actions(vlm_result.get("detected_type", "Unknown"))
                elif 'actions' in agent_result:
                    # 직접 actions가 있는 경우
                    actions = agent_result['actions']
                    if isinstance(actions, list):
                        result["actions"] = [
                            {
                                "action": action.get('action', '') if isinstance(action, dict) else str(action),
                                "priority": action.get('priority', 'medium') if isinstance(action, dict) else 'medium',
                                "description": action.get('description', '') if isinstance(action, dict) else ''
                            }
                            for action in actions
                        ]
                    else:
                        result["actions"] = self._get_default_actions(vlm_result.get("detected_type", "Unknown"))
                else:
                    result["actions"] = self._get_default_actions(vlm_result.get("detected_type", "Unknown"))
                
                # 처리 시간 추출
                if 'processing_times' in agent_result:
                    times = agent_result['processing_times']
                    total_time = sum(times.values()) if isinstance(times, dict) else 0.0
                    result["processing_time"] = total_time
                else:
                    result["processing_time"] = time.time() - start_time
                
                result["success"] = agent_result.get('success', False)
            else:
                # 결과 형식이 예상과 다른 경우
                result["actions"] = self._get_default_actions(vlm_result.get("detected_type", "Unknown"))
                result["processing_time"] = time.time() - start_time
        except Exception as e:
            logging.error(f"Agent process error: {e}")
            import traceback
            logging.error(traceback.format_exc())
            result["actions"] = self._get_default_actions(vlm_result.get("detected_type", "Unknown"))
        
        result["processing_time"] = time.time() - start_time
        return result
    
    def _get_default_actions(self, anomaly_type: str) -> List[Dict]:
        """기본 대응 액션"""
        actions_map = {
            "Arson": [{"action": "alert_fire_dept", "priority": "high"}, {"action": "evacuate", "priority": "high"}],
            "Fighting": [{"action": "alert_security", "priority": "high"}, {"action": "dispatch_guard", "priority": "medium"}],
            "Explosion": [{"action": "alert_emergency", "priority": "critical"}, {"action": "lockdown", "priority": "high"}],
            "Road_Accident": [{"action": "alert_ambulance", "priority": "high"}, {"action": "traffic_control", "priority": "medium"}],
            "Suspicious_Object": [{"action": "alert_security", "priority": "medium"}, {"action": "investigate", "priority": "medium"}],
            "Falling": [{"action": "alert_medical", "priority": "high"}, {"action": "check_status", "priority": "high"}],
            "Normal": []
        }
        return actions_map.get(anomaly_type, [{"action": "monitor", "priority": "low"}])


# =============================================================================
# E2E 시스템 엔진
# =============================================================================

class E2ESystem:
    """End-to-End 보안 모니터링 시스템"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = None
        self.clip_saver = None
        self.video_source = None
        self.vad = None
        self.vlm = None
        self.agent = None
        
        # 상태
        self.stats = SystemStats()
        self.is_running = False
        self._stop_event = threading.Event()
        
        # 콜백
        self.on_frame_callback: Optional[Callable] = None
        self.on_anomaly_callback: Optional[Callable] = None
        self.on_stats_callback: Optional[Callable] = None
        
        # 프레임 버퍼
        self.frame_buffer: deque = deque(maxlen=30)
        self.current_frame = None
        self.current_score = 0.0
        
        # 프레임 스킵 카운터 (부하 시 프레임 스킵)
        self.frame_skip_counter = 0
        self.frame_skip_interval = 1  # 모든 프레임 처리 (부하 시 증가)
    
    def initialize(self) -> Tuple[bool, Optional[str]]:
        """시스템 초기화
        
        Returns:
            (success: bool, error_message: Optional[str])
        """
        # GPU 설정
        if HAS_TORCH:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_id)
        
        # 로거 초기화
        self.logger = EventLogger(self.config.log_dir, self.config.log_level)
        self.logger.log_info("E2E System initializing...")
        
        # 클립 저장기 초기화
        if self.config.save_clips:
            self.clip_saver = ClipSaver(
                self.config.clips_dir,
                self.config.clip_duration,
                self.config.target_fps
            )
        
        # 비디오 소스 초기화
        self.video_source = VideoSource(self.config.source_type, self.config.source_path)
        if not self.video_source.open():
            error_msg = f"Failed to open video source: {self.config.source_path}"
            if self.config.source_type == VideoSourceType.FILE:
                if not os.path.exists(self.config.source_path):
                    error_msg += " (File does not exist)"
                elif not os.access(self.config.source_path, os.R_OK):
                    error_msg += " (File is not readable)"
            self.logger.log_error(error_msg)
            return False, error_msg
        
        self.logger.log_info(f"Video source: {self.video_source.get_info()}")
        
        # VAD 모델 초기화 (실제 모델 사용)
        try:
            self.vad = VADWrapper(
                self.config.vad_model,
                self.config.checkpoint_path,
                self.config.gpu_id
            )
            if self.vad.load():
                self.logger.log_info(f"VAD model loaded: {self.config.vad_model.value}")
            else:
                error_msg = f"VAD model '{self.config.vad_model.value}' failed to load"
                self.logger.log_error(error_msg)
                return False, error_msg
        except Exception as e:
            error_msg = f"VAD model initialization error: {str(e)}"
            self.logger.log_error(error_msg)
            import traceback
            self.logger.log_error(traceback.format_exc())
            return False, error_msg
        
        # VLM 초기화 (선택적)
        if self.config.enable_vlm:
            try:
                self.vlm = VLMWrapper(
                    self.config.vlm_model_path,
                    self.config.vlm_n_frames,
                    self.config.optimize_vlm,
                    self.config.gpu_id
                )
                if self.vlm.load():
                    self.logger.log_info("VLM loaded")
                else:
                    self.logger.log_warning("VLM not loaded (continuing without VLM)")
            except Exception as e:
                self.logger.log_warning(f"VLM initialization error: {str(e)}")
        
        # Agent 초기화 (선택적)
        if self.config.enable_agent:
            try:
                self.agent = AgentWrapper(self.config.agent_flow, self.config.gpu_id)
                if self.agent.load():
                    self.logger.log_info(f"Agent loaded: {self.config.agent_flow.value}")
                else:
                    self.logger.log_warning("Agent not loaded (continuing without Agent)")
            except Exception as e:
                self.logger.log_warning(f"Agent initialization error: {str(e)}")
        
        self.logger.log_info("E2E System initialized successfully")
        return True, None
    
    def start(self):
        """시스템 시작"""
        if self.is_running:
            return
        
        self.is_running = True
        self._stop_event.clear()
        self.stats = SystemStats()
        
        self.logger.log_info("E2E System started")
        
        # 메인 처리 루프
        self._process_loop()
    
    def stop(self):
        """시스템 중지"""
        self._stop_event.set()
        self.is_running = False
        
        if self.video_source:
            self.video_source.close()
        
        self.logger.log_info("E2E System stopped")
        self.logger.log_info(f"Final stats: {json.dumps(self.stats.to_dict())}")
    
    def _process_loop(self):
        """메인 처리 루프 (최적화된 버전)"""
        fps_counter = 0
        fps_start_time = time.time()
        vad_times = deque(maxlen=100)
        
        while not self._stop_event.is_set():
            try:
                loop_start = time.time()
                
                # 프레임 스킵 로직 (부하 시)
                self.frame_skip_counter += 1
                if self.frame_skip_counter < self.frame_skip_interval:
                    # 프레임 스킵
                    try:
                        ret, _ = self.video_source.read()
                        if not ret and self.config.source_type == VideoSourceType.FILE:
                            break
                    except:
                        pass
                    continue
                
                self.frame_skip_counter = 0
                
                # 프레임 읽기
                try:
                    ret, frame = self.video_source.read()
                except Exception as e:
                    self.logger.log_error(f"Error reading frame: {e}")
                    import traceback
                    self.logger.log_error(traceback.format_exc())
                    if self.config.source_type == VideoSourceType.FILE:
                        break
                    time.sleep(0.1)  # RTSP/웹캠의 경우 짧은 대기 후 재시도
                    continue
                
                if not ret:
                    if self.config.source_type == VideoSourceType.FILE:
                        self.logger.log_info("Video file ended")
                        break
                    # RTSP/웹캠의 경우 연결 문제일 수 있음
                    self.logger.log_warning("Failed to read frame from video source")
                    time.sleep(0.1)
                    continue
                
                self.stats.total_frames += 1
                fps_counter += 1
                
                # 디버깅: 처음 몇 프레임 로그
                if self.stats.total_frames <= 10:
                    frame_shape = frame.shape if HAS_CV2 else 'N/A'
                    self.logger.log_info(f"Frame read successfully: {self.stats.total_frames}, shape={frame_shape}, callback set: {self.on_frame_callback is not None}")
                
                # 프레임 리사이즈
                if HAS_CV2:
                    frame = cv2.resize(frame, (self.config.display_width, self.config.display_height))
                
                self.current_frame = frame
                
                # 프레임 콜백 즉시 호출 (VAD 전에) - 프레임 표시 보장
                if self.on_frame_callback:
                    try:
                        # 콜백 호출 전 로깅
                        if self.stats.total_frames <= 20:
                            self.logger.log_info(
                                f"[BEFORE CALLBACK] Frame {self.stats.total_frames}, "
                                f"callback type: {type(self.on_frame_callback).__name__}, "
                                f"callback module: {getattr(self.on_frame_callback, '__module__', 'unknown')}"
                            )
                        
                        # 콜백 호출
                        callback_result = self.on_frame_callback(frame, 0.0)  # 임시 점수 0.0
                        
                        # 콜백 호출 후 로깅
                        if self.stats.total_frames <= 20:
                            self.logger.log_info(
                                f"[AFTER CALLBACK] Frame {self.stats.total_frames}, "
                                f"callback returned: {callback_result}, "
                                f"frame shape: {frame.shape if HAS_CV2 else 'N/A'}"
                            )
                    except Exception as e:
                        self.logger.log_error(f"[CALLBACK ERROR] Immediate frame callback error for frame {self.stats.total_frames}: {e}")
                        import traceback
                        self.logger.log_error(f"[CALLBACK TRACEBACK] {traceback.format_exc()}")
                        # 콜백 실패해도 시스템은 계속 실행
                else:
                    if self.stats.total_frames <= 20:
                        self.logger.log_warning(f"[CALLBACK WARNING] Frame callback is None for frame {self.stats.total_frames}")
                
                # 클립 버퍼에 추가
                if self.clip_saver:
                    self.clip_saver.add_frame(frame)
                
                # 프레임 버퍼에 추가
                self.frame_buffer.append(frame.copy() if HAS_CV2 else frame)
                
                # VAD 추론 (에러 핸들링) - VAD 실패해도 프레임은 이미 표시됨
                try:
                    if self.vad and self.vad._loaded and len(self.frame_buffer) >= 1:
                        vad_score, vad_time = self.vad.predict(list(self.frame_buffer))
                    else:
                        # VAD 모델이 없거나 버퍼가 부족한 경우
                        vad_score = 0.0
                        vad_time = 0.0
                        if self.stats.total_frames <= 10:
                            self.logger.log_info(f"VAD skipped: loaded={self.vad._loaded if self.vad else False}, buffer_size={len(self.frame_buffer)}")
                except Exception as e:
                    self.logger.log_error(f"VAD prediction error: {e}")
                    import traceback
                    self.logger.log_error(traceback.format_exc())
                    vad_score = 0.0
                    vad_time = 0.0
                
                vad_times.append(vad_time)
                self.current_score = vad_score
                
                # 이상 감지
                if vad_score >= self.config.vad_threshold:
                    try:
                        self._handle_anomaly(frame, vad_score)
                    except Exception as e:
                        self.logger.log_error(f"Error handling anomaly: {e}")
                        # 에러가 발생해도 시스템은 계속 실행
                
                # FPS 계산
                if time.time() - fps_start_time >= 1.0:
                    self.stats.current_fps = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()
                    self.stats.avg_vad_time = sum(vad_times) / len(vad_times) if vad_times else 0
                
                # VAD 점수로 프레임 콜백 다시 호출 (점수 업데이트) - 선택적
                if self.on_frame_callback and vad_score != 0.0:
                    try:
                        self.on_frame_callback(frame, vad_score)
                        if self.stats.total_frames <= 10:
                            self.logger.log_info(f"Frame callback updated with score for frame {self.stats.total_frames}, score={vad_score:.4f}")
                    except Exception as e:
                        self.logger.log_warning(f"Frame callback update error: {e}")
                
                # 통계 콜백
                if self.on_stats_callback:
                    self.on_stats_callback(self.stats)
                
                # FPS 제한 및 부하 감지
                elapsed = time.time() - loop_start
                target_time = 1.0 / self.config.target_fps
                
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)
                elif elapsed > target_time * 2:
                    # 처리 시간이 너무 오래 걸리면 프레임 스킵 간격 증가
                    self.frame_skip_interval = min(self.frame_skip_interval + 1, 5)
                    self.logger.log_warning(
                        f"Frame processing took {elapsed:.3f}s (target: {target_time:.3f}s). "
                        f"Increasing skip interval to {self.frame_skip_interval}"
                    )
                else:
                    # 정상 처리 시 스킵 간격 감소
                    if self.frame_skip_interval > 1:
                        self.frame_skip_interval = max(1, self.frame_skip_interval - 1)
            except Exception as e:
                self.logger.log_error(f"Error in process loop: {e}")
                import traceback
                self.logger.log_error(traceback.format_exc())
                # 예외가 발생해도 루프는 계속 실행
                time.sleep(0.1)
                continue
    
    def _handle_anomaly(self, frame, vad_score: float):
        """이상 감지 처리"""
        event_id = f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # 클립 저장
        clip_path = ""
        if self.clip_saver:
            clip_path = self.clip_saver.save_clip(event_id)
        
        # 이벤트 생성
        event = AnomalyEvent(
            id=event_id,
            timestamp=datetime.now(),
            frame_number=self.stats.total_frames,
            vad_score=vad_score,
            threshold=self.config.vad_threshold,
            clip_path=clip_path
        )
        
        # VLM 분석
        vlm_result = {}
        if self.vlm:
            vlm_result = self.vlm.analyze(list(self.frame_buffer), clip_path)
            event.vlm_type = vlm_result.get("detected_type", "Unknown")
            event.vlm_description = vlm_result.get("description", "")
            event.vlm_confidence = vlm_result.get("confidence", 0.0)
            self.stats.avg_vlm_time = vlm_result.get("latency_ms", 0) / 1000
        
        # Agent 처리
        if self.agent:
            agent_result = self.agent.process(event, vlm_result)
            event.agent_actions = agent_result.get("actions", [])
            event.agent_response_time = agent_result.get("processing_time", 0)
            self.stats.avg_agent_time = event.agent_response_time
        
        # 이벤트 로깅
        self.logger.log_event(event)
        self.stats.anomaly_count += 1
        
        # 이상 콜백
        if self.on_anomaly_callback:
            self.on_anomaly_callback(event)
    
    def get_stats(self) -> Dict:
        """통계 조회"""
        return self.stats.to_dict()
    
    def get_recent_events(self, n: int = 10) -> List[AnomalyEvent]:
        """최근 이벤트 조회"""
        if self.logger:
            return self.logger.get_recent_events(n)
        return []
    
    def get_current_frame(self):
        """현재 프레임 조회"""
        return self.current_frame
    
    def get_current_score(self) -> float:
        """현재 이상 점수 조회"""
        return self.current_score


# =============================================================================
# 유틸리티 함수
# =============================================================================

def create_system_from_args(args) -> E2ESystem:
    """argparse args에서 시스템 생성"""
    # 소스 타입 결정
    if args.source.startswith("rtsp://"):
        source_type = VideoSourceType.RTSP
    elif args.source.isdigit():
        source_type = VideoSourceType.WEBCAM
    else:
        source_type = VideoSourceType.FILE
    
    config = SystemConfig(
        source_type=source_type,
        source_path=args.source,
        vad_model=VADModelType(args.vad_model),
        vad_threshold=args.threshold,
        enable_vlm=args.enable_vlm,
        vlm_n_frames=args.vlm_frames,
        optimize_vlm=args.optimize_vlm,
        enable_agent=args.enable_agent,
        agent_flow=AgentFlowType(args.agent_flow),
        save_clips=args.save_clips,
        clip_duration=args.clip_duration,
        clips_dir=args.clips_dir,
        log_dir=args.log_dir,
        gpu_id=args.gpu,
        target_fps=args.fps
    )
    
    return E2ESystem(config)


def main():
    """CLI 테스트용 메인"""
    import argparse
    
    parser = argparse.ArgumentParser(description="E2E Security Monitoring System")
    parser.add_argument("--source", type=str, required=True, help="Video source (file/rtsp/webcam)")
    parser.add_argument("--vad-model", type=str, default="mnad", choices=["mnad", "mulde", "memae", "stae"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--enable-vlm", action="store_true")
    parser.add_argument("--vlm-frames", type=int, default=4)
    parser.add_argument("--optimize-vlm", action="store_true")
    parser.add_argument("--enable-agent", action="store_true")
    parser.add_argument("--agent-flow", type=str, default="sequential", choices=["hierarchical", "sequential", "collaborative"])
    parser.add_argument("--save-clips", action="store_true", default=True)
    parser.add_argument("--clip-duration", type=float, default=3.0)
    parser.add_argument("--clips-dir", type=str, default="clips")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    
    args = parser.parse_args()
    
    system = create_system_from_args(args)
    
    success, error_msg = system.initialize()
    if success:
        try:
            system.start()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            system.stop()
    else:
        print(f"Failed to initialize system: {error_msg or 'Unknown error'}")
        sys.exit(1)


if __name__ == "__main__":
    main()

