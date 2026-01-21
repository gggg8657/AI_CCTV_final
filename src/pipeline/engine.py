"""
E2E 파이프라인 엔진
==================

VAD + VLM + Agent 통합 실시간 처리 엔진
"""

import os
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict, List, Any
from collections import deque
from enum import Enum

import cv2
import numpy as np

from ..vad import VADModel, create_model as create_vad_model
from ..vlm import VLMAnalyzer
from ..agent import create_flow as create_agent_flow
from ..utils.video import VideoSource, VideoSourceType
from ..utils.logging import EventLogger, AnomalyEvent
from .clip_saver import ClipSaver


class AgentFlowType(Enum):
    """Agent Flow 타입"""
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
    COLLABORATIVE = "collaborative"


@dataclass
class EngineConfig:
    """엔진 설정"""
    # 비디오 소스
    source_type: VideoSourceType = VideoSourceType.FILE
    source_path: str = ""
    
    # VAD 설정
    vad_model: str = "mnad"
    vad_threshold: float = 0.5
    
    # VLM 설정
    enable_vlm: bool = True
    vlm_n_frames: int = 8
    optimize_vlm: bool = False
    
    # Agent 설정
    enable_agent: bool = True
    agent_flow: AgentFlowType = AgentFlowType.SEQUENTIAL
    
    # 클립 저장 설정
    save_clips: bool = True
    clip_duration: float = 3.0
    clips_dir: str = "./clips"
    
    # 로깅 설정
    logs_dir: str = "./logs"
    
    # GPU 설정
    gpu_id: int = 2
    
    # 처리 설정
    target_fps: int = 30


@dataclass
class EngineStats:
    """엔진 통계"""
    start_time: datetime = field(default_factory=datetime.now)
    total_frames: int = 0
    anomaly_count: int = 0
    current_fps: float = 0.0
    avg_vad_time: float = 0.0
    avg_vlm_time: float = 0.0
    avg_agent_time: float = 0.0
    
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
        }


class E2EEngine:
    """
    End-to-End 파이프라인 엔진
    
    VAD → 이상 감지 → 클립 저장 → VLM 분석 → Agent 대응
    """
    
    def __init__(self, config: EngineConfig):
        self.config = config
        
        # 컴포넌트
        self.video_source: Optional[VideoSource] = None
        self.vad_model: Optional[VADModel] = None
        self.vlm_analyzer: Optional[VLMAnalyzer] = None
        self.agent_flow = None
        self.clip_saver: Optional[ClipSaver] = None
        self.event_logger: Optional[EventLogger] = None
        
        # 상태
        self.stats = EngineStats()
        self.is_running = False
        self._stop_event = threading.Event()
        
        # 콜백
        self.on_frame_callback: Optional[Callable] = None
        self.on_anomaly_callback: Optional[Callable] = None
        self.on_stats_callback: Optional[Callable] = None
        
        # 버퍼
        self.frame_buffer: deque = deque(maxlen=30)
        self.current_frame = None
        self.current_score = 0.0
        
        # 시간 측정
        self._vad_times: deque = deque(maxlen=100)
        self._vlm_times: deque = deque(maxlen=10)
        self._agent_times: deque = deque(maxlen=10)
    
    def initialize(self) -> bool:
        """엔진 초기화"""
        print(f"[E2EEngine] 초기화 중...")
        
        # 이벤트 로거
        self.event_logger = EventLogger(self.config.logs_dir)
        self.event_logger.log_info("E2E Engine initializing...")
        
        # 클립 저장기
        if self.config.save_clips:
            self.clip_saver = ClipSaver(
                output_dir=self.config.clips_dir,
                record_seconds=self.config.clip_duration,
                fps=self.config.target_fps,
                on_clip_saved=self._on_clip_saved,
            )
        
        # 비디오 소스
        self.video_source = VideoSource(
            self.config.source_type,
            self.config.source_path
        )
        if not self.video_source.open():
            self.event_logger.log_error(f"Failed to open video source: {self.config.source_path}")
            return False
        
        self.event_logger.log_info(f"Video source: {self.video_source.get_info()}")
        
        # VAD 모델
        try:
            self.vad_model = create_vad_model(self.config.vad_model)
            self.vad_model.initialize(f"cuda:{self.config.gpu_id}")
            self.event_logger.log_info(f"VAD model loaded: {self.config.vad_model}")
        except Exception as e:
            self.event_logger.log_error(f"VAD model load failed: {e}")
            return False
        
        # VLM 분석기
        if self.config.enable_vlm:
            self.vlm_analyzer = VLMAnalyzer(
                n_frames=self.config.vlm_n_frames,
                optimize_speed=self.config.optimize_vlm,
                gpu_id=self.config.gpu_id,
            )
            if self.vlm_analyzer.initialize():
                self.event_logger.log_info("VLM analyzer loaded")
            else:
                self.event_logger.log_warning("VLM analyzer load failed")
        
        # Agent Flow
        if self.config.enable_agent:
            self.agent_flow = create_agent_flow(
                self.config.agent_flow.value,
                gpu_id=self.config.gpu_id
            )
            if self.agent_flow.initialize():
                self.event_logger.log_info(f"Agent flow loaded: {self.config.agent_flow.value}")
            else:
                self.event_logger.log_warning("Agent flow load failed")
        
        self.event_logger.log_info("E2E Engine initialized successfully")
        print(f"[E2EEngine] 초기화 완료")
        return True
    
    def start(self):
        """엔진 시작 (메인 루프)"""
        if self.is_running:
            return
        
        self.is_running = True
        self._stop_event.clear()
        self.stats = EngineStats()
        
        self.event_logger.log_info("E2E Engine started")
        print("[E2EEngine] 시작")
        
        self._process_loop()
    
    def stop(self):
        """엔진 중지"""
        self._stop_event.set()
        self.is_running = False
        
        if self.video_source:
            self.video_source.close()
        
        self.event_logger.log_info("E2E Engine stopped")
        self.event_logger.log_info(f"Final stats: {self.stats.to_dict()}")
        print("[E2EEngine] 중지")
    
    def _process_loop(self):
        """메인 처리 루프"""
        fps_counter = 0
        fps_start_time = time.time()
        
        while not self._stop_event.is_set():
            loop_start = time.time()
            
            # 프레임 읽기
            ret, frame = self.video_source.read_rgb()
            if not ret:
                if self.config.source_type == VideoSourceType.FILE:
                    self.event_logger.log_info("Video file ended")
                    break
                continue
            
            self.stats.total_frames += 1
            fps_counter += 1
            self.current_frame = frame
            
            # 클립 버퍼에 추가
            if self.clip_saver:
                self.clip_saver.add_frame(frame)
            
            # 프레임 버퍼에 추가
            self.frame_buffer.append(frame)
            
            # VAD 추론
            vad_start = time.time()
            score = self.vad_model.process_frame(frame)
            if score is not None:
                self._vad_times.append(time.time() - vad_start)
                self.current_score = score
                
                # 이상 감지
                if score >= self.config.vad_threshold:
                    self._handle_anomaly(frame, score)
            
            # FPS 계산
            if time.time() - fps_start_time >= 1.0:
                self.stats.current_fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
                self.stats.avg_vad_time = sum(self._vad_times) / len(self._vad_times) if self._vad_times else 0
            
            # 프레임 콜백
            if self.on_frame_callback and score is not None:
                self.on_frame_callback(frame, score)
            
            # 통계 콜백
            if self.on_stats_callback:
                self.on_stats_callback(self.stats)
            
            # FPS 제한
            elapsed = time.time() - loop_start
            target_time = 1.0 / self.config.target_fps
            if elapsed < target_time:
                time.sleep(target_time - elapsed)
    
    def _handle_anomaly(self, frame: np.ndarray, vad_score: float):
        """이상 감지 처리"""
        event_id = f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # 클립 녹화 시작
        clip_path = ""
        if self.clip_saver and not self.clip_saver.is_recording:
            self.clip_saver.trigger_save(vad_score, self.stats.total_frames)
            print(f"[Anomaly] Frame {self.stats.total_frames}: Score={vad_score:.4f} → 클립 녹화 시작")
        
        # 이벤트 생성 (VLM/Agent는 클립 저장 완료 후 처리)
        event = AnomalyEvent(
            id=event_id,
            timestamp=datetime.now().isoformat(),
            frame_number=self.stats.total_frames,
            vad_score=vad_score,
            threshold=self.config.vad_threshold,
        )
        
        self.stats.anomaly_count += 1
        
        # 이상 콜백
        if self.on_anomaly_callback:
            self.on_anomaly_callback(event)
    
    def _on_clip_saved(self, clip_path: str, clip_meta: Dict):
        """클립 저장 완료 콜백"""
        print(f"[E2EEngine] 클립 저장 완료: {clip_path}")
        
        # VLM 분석
        vlm_result = None
        if self.vlm_analyzer and self.vlm_analyzer.is_initialized:
            print("[E2EEngine] VLM 분석 중...")
            vlm_start = time.time()
            vlm_result = self.vlm_analyzer.analyze(video_path=clip_path)
            self._vlm_times.append(time.time() - vlm_start)
            self.stats.avg_vlm_time = sum(self._vlm_times) / len(self._vlm_times)
            
            print(f"[VLM] 결과: {vlm_result.detected_type} ({vlm_result.latency_ms:.0f}ms)")
        
        # Agent 대응
        if self.agent_flow:
            print("[E2EEngine] Agent Flow 실행 중...")
            agent_start = time.time()
            agent_result = self.agent_flow.run(video_path=clip_path)
            self._agent_times.append(time.time() - agent_start)
            self.stats.avg_agent_time = sum(self._agent_times) / len(self._agent_times)
            
            print(f"[Agent] 결과: {agent_result.get('situation_type', 'Unknown')}")
        
        # 이벤트 로깅
        event = AnomalyEvent(
            id=clip_meta.get('timestamp', ''),
            timestamp=clip_meta.get('trigger_time', ''),
            frame_number=clip_meta.get('frame_number', 0),
            vad_score=clip_meta.get('score', 0.0),
            threshold=self.config.vad_threshold,
            vlm_type=vlm_result.detected_type if vlm_result else "Unknown",
            vlm_description=vlm_result.description if vlm_result else "",
            vlm_confidence=vlm_result.confidence if vlm_result else 0.0,
            clip_path=clip_path,
        )
        self.event_logger.log_event(event)
    
    def get_stats(self) -> Dict:
        """통계 조회"""
        return self.stats.to_dict()
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """현재 프레임 조회"""
        return self.current_frame
    
    def get_current_score(self) -> float:
        """현재 이상 점수 조회"""
        return self.current_score



