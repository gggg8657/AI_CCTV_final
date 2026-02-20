"""
CameraConfig / CameraStatus — 카메라 설정 및 상태 데이터 클래스
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict


class PipelineState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class CameraConfig:
    camera_id: int
    source_type: str = "rtsp"
    source_path: str = ""
    location: Optional[str] = None
    vad_model: str = "mnad"
    vad_threshold: float = 0.5
    enable_vlm: bool = True
    enable_agent: bool = True
    agent_flow: str = "sequential"
    gpu_id: int = 0
    target_fps: int = 30
    save_clips: bool = True
    clip_duration: float = 3.0


@dataclass
class CameraStatus:
    camera_id: int
    state: PipelineState = PipelineState.IDLE
    total_frames: int = 0
    anomaly_count: int = 0
    current_fps: float = 0.0
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    last_frame_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "camera_id": self.camera_id,
            "state": self.state.value,
            "total_frames": self.total_frames,
            "anomaly_count": self.anomaly_count,
            "current_fps": round(self.current_fps, 2),
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_frame_at": self.last_frame_at.isoformat() if self.last_frame_at else None,
        }
