"""
DummyVLMAnalyzer — 실제 LLM 없이 VLMAnalyzer 인터페이스를 구현
================================================================

랜덤으로 이상 유형을 분류하고 설명을 생성.
VLMAnalysisResult와 동일한 인터페이스를 반환.
"""

import random
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


ANOMALY_TYPES = [
    ("Fighting", "Two individuals are engaged in a physical altercation near the entrance."),
    ("Arson", "Smoke and fire detected in the corridor area."),
    ("Falling", "A person has collapsed on the ground and appears unresponsive."),
    ("Loitering", "An individual has been standing in the restricted area for an extended period."),
    ("Suspicious_Object", "An unattended bag has been detected near the main gate."),
    ("Road_Accident", "A vehicle collision has occurred in the parking lot."),
    ("Normal", "No anomalous activity detected. Scene appears normal."),
]


@dataclass
class DummyVLMResult:
    """VLMAnalysisResult 호환 더미 결과"""
    detected_type: str
    description: str
    actions: List[str]
    confidence: float
    response: str
    latency_ms: float
    n_frames: int
    success: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected_type": self.detected_type,
            "description": self.description,
            "actions": self.actions,
            "confidence": self.confidence,
            "response": self.response,
            "latency_ms": self.latency_ms,
            "n_frames": self.n_frames,
            "success": self.success,
        }


ACTION_MAP: Dict[str, List[str]] = {
    "Fighting": ["alert_security", "dispatch_guard"],
    "Arson": ["alert_fire_dept", "evacuate"],
    "Falling": ["alert_medical", "check_status"],
    "Loitering": ["monitor", "alert_security"],
    "Suspicious_Object": ["alert_security", "investigate"],
    "Road_Accident": ["alert_ambulance", "traffic_control"],
    "Normal": [],
}


class DummyVLMAnalyzer:
    """실제 VLM 없이 VLMAnalyzer 인터페이스를 구현하는 더미"""

    def __init__(self, latency_ms: float = 50.0, **kwargs: Any):
        self._initialized = False
        self._latency_ms = latency_ms

    def initialize(self) -> bool:
        self._initialized = True
        return True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def analyze(
        self,
        frames: Optional[List[Any]] = None,
        video_path: Optional[str] = None,
    ) -> DummyVLMResult:
        start = time.time()

        weights = [3, 1, 2, 3, 1, 1, 10]
        anomaly_type, description = random.choices(ANOMALY_TYPES, weights=weights, k=1)[0]

        confidence = round(random.uniform(0.7, 0.95), 2) if anomaly_type != "Normal" else round(random.uniform(0.3, 0.5), 2)
        actions = ACTION_MAP.get(anomaly_type, [])

        time.sleep(self._latency_ms / 1000.0)

        return DummyVLMResult(
            detected_type=anomaly_type,
            description=description,
            actions=actions,
            confidence=confidence,
            response=f"[DUMMY] {anomaly_type}: {description}",
            latency_ms=(time.time() - start) * 1000,
            n_frames=len(frames) if frames else 0,
            success=True,
        )

    def unload(self) -> None:
        self._initialized = False
