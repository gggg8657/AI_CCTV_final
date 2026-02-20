"""
알림 채널 추상 인터페이스 + 페이로드 정의
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any


@dataclass
class NotificationPayload:
    """알림 페이로드"""
    camera_id: int
    camera_name: str = ""
    location: str = ""
    vlm_type: str = "Unknown"
    vlm_description: str = ""
    vad_score: float = 0.0
    confidence: float = 0.0
    severity: str = "medium"
    timestamp: datetime = field(default_factory=datetime.now)
    actions: List[str] = field(default_factory=list)
    clip_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def title(self) -> str:
        sev = self.severity.upper()
        return f"[{sev}] {self.vlm_type} — Camera {self.camera_id}"

    @property
    def body(self) -> str:
        lines = [
            f"Camera: {self.camera_name or self.camera_id} ({self.location})",
            f"Type: {self.vlm_type}",
            f"Score: {self.vad_score:.2f} (confidence: {self.confidence:.2f})",
            f"Description: {self.vlm_description}",
            f"Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        if self.actions:
            lines.append(f"Actions: {', '.join(self.actions)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "location": self.location,
            "vlm_type": self.vlm_type,
            "vlm_description": self.vlm_description,
            "vad_score": self.vad_score,
            "confidence": self.confidence,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "actions": self.actions,
            "clip_path": self.clip_path,
        }


class NotificationChannel(ABC):
    """알림 채널 추상 클래스"""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def send(self, payload: NotificationPayload) -> bool:
        ...

    @property
    def is_available(self) -> bool:
        return True
