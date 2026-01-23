"""
이벤트 타입 정의
================

Vision-Agents 통합을 위한 이벤트 타입들
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np

from .event_bus import BaseEvent


@dataclass
class AnomalyDetectedEvent(BaseEvent):
    """이상 감지 이벤트"""
    frame_id: int = 0
    score: float = 0.0
    threshold: float = 0.0
    frame: Optional[np.ndarray] = None  # 프레임 이미지 (선택)
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"anomaly_{self.frame_id}_{datetime.now().timestamp()}"
        if not self.event_type:
            self.event_type = "AnomalyDetectedEvent"
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.source:
            self.source = "VAD"


@dataclass
class VLMAnalysisCompletedEvent(BaseEvent):
    """VLM 분석 완료 이벤트"""
    original_event_id: str = ""  # 원본 AnomalyDetectedEvent ID
    detected_type: str = ""
    description: str = ""
    actions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    clip_path: str = ""
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"vlm_{self.original_event_id}_{datetime.now().timestamp()}"
        if not self.event_type:
            self.event_type = "VLMAnalysisCompletedEvent"
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.source:
            self.source = "VLM"


@dataclass
class AgentResponseEvent(BaseEvent):
    """Agent 대응 계획 이벤트"""
    original_event_id: str = ""  # 원본 AnomalyDetectedEvent ID
    plan: List[str] = field(default_factory=list)
    priority: int = 3  # 1=높음, 5=낮음
    estimated_time: float = 0.0  # 예상 소요 시간 (초)
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"agent_{self.original_event_id}_{datetime.now().timestamp()}"
        if not self.event_type:
            self.event_type = "AgentResponseEvent"
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.source:
            self.source = "Agent"


@dataclass
class FrameProcessedEvent(BaseEvent):
    """프레임 처리 완료 이벤트"""
    frame_id: int = 0
    processing_time: float = 0.0  # 처리 시간 (초)
    vad_score: float = 0.0
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"frame_{self.frame_id}_{datetime.now().timestamp()}"
        if not self.event_type:
            self.event_type = "FrameProcessedEvent"
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.source:
            self.source = "E2ESystem"


@dataclass
class StatsUpdatedEvent(BaseEvent):
    """통계 업데이트 이벤트"""
    total_frames: int = 0
    anomaly_count: int = 0
    avg_processing_time: float = 0.0
    current_fps: float = 0.0
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"stats_{datetime.now().timestamp()}"
        if not self.event_type:
            self.event_type = "StatsUpdatedEvent"
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.source:
            self.source = "E2ESystem"


@dataclass
class PackageDetectedEvent(BaseEvent):
    """패키지 감지 이벤트 (Phase 3용)"""
    package_id: str = ""
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x1, y1, x2, y2)
    confidence: float = 0.0
    camera_id: int = 0
    frame_index: int = 0
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"package_{self.package_id}_{datetime.now().timestamp()}"
        if not self.event_type:
            self.event_type = "PackageDetectedEvent"
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.source:
            self.source = "PackageDetector"


@dataclass
class PackageDisappearedEvent(BaseEvent):
    """패키지 사라짐 이벤트 (Phase 3용)"""
    package_id: str = ""
    last_seen: str = ""
    camera_id: int = 0
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"package_disappeared_{self.package_id}_{datetime.now().timestamp()}"
        if not self.event_type:
            self.event_type = "PackageDisappearedEvent"
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.source:
            self.source = "PackageDetector"


@dataclass
class TheftDetectedEvent(BaseEvent):
    """도난 감지 이벤트 (Phase 3용)"""
    package_id: str = ""
    theft_time: str = ""
    camera_id: int = 0
    evidence_frame_paths: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"theft_{self.package_id}_{datetime.now().timestamp()}"
        if not self.event_type:
            self.event_type = "TheftDetectedEvent"
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.source:
            self.source = "TheftDetector"
