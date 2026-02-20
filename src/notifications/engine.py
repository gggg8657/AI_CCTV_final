"""
알림 엔진 — 규칙 기반 필터링 + 중복 방지 + 멀티채널 발송
"""

import logging
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Any

from .base import NotificationChannel, NotificationPayload

logger = logging.getLogger(__name__)

SEVERITY_MAP: Dict[str, str] = {
    "Fighting": "high",
    "Arson": "critical",
    "Falling": "high",
    "Loitering": "low",
    "Suspicious_Object": "medium",
    "Road_Accident": "critical",
    "Normal": "info",
}


class NotificationEngine:
    """규칙 기반 알림 발송 엔진"""

    def __init__(self, cooldown_seconds: float = 15.0):
        self._channels: List[NotificationChannel] = []
        self._cooldown = cooldown_seconds
        self._last_sent: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()
        self._stats = {"total": 0, "sent": 0, "suppressed": 0, "failed": 0}

    def add_channel(self, channel: NotificationChannel) -> None:
        self._channels.append(channel)
        logger.info("Notification channel added: %s", channel.name)

    def remove_channel(self, name: str) -> bool:
        before = len(self._channels)
        self._channels = [c for c in self._channels if c.name != name]
        return len(self._channels) < before

    @property
    def channels(self) -> List[str]:
        return [c.name for c in self._channels]

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def notify(self, event_data: Dict[str, Any], camera_name: str = "", location: str = "") -> bool:
        """
        이상 이벤트로부터 알림 발송.
        cooldown 기간 내 동일 카메라+유형 조합은 중복 방지.
        """
        self._stats["total"] += 1

        vlm_type = event_data.get("vlm_type", "Unknown")
        camera_id = event_data.get("camera_id", 0)
        dedup_key = f"{camera_id}:{vlm_type}"

        with self._lock:
            now = time.time()
            if now - self._last_sent[dedup_key] < self._cooldown:
                self._stats["suppressed"] += 1
                return False
            self._last_sent[dedup_key] = now

        severity = SEVERITY_MAP.get(vlm_type, "medium")
        if severity == "info":
            return False

        payload = NotificationPayload(
            camera_id=camera_id,
            camera_name=camera_name,
            location=location,
            vlm_type=vlm_type,
            vlm_description=event_data.get("vlm_description", ""),
            vad_score=event_data.get("vad_score", 0.0),
            confidence=event_data.get("vlm_confidence", 0.0),
            severity=severity,
            timestamp=datetime.fromisoformat(event_data["timestamp"]) if isinstance(event_data.get("timestamp"), str) else event_data.get("timestamp", datetime.now()),
            actions=event_data.get("actions", []),
            clip_path=event_data.get("clip_path"),
        )

        sent_any = False
        for channel in self._channels:
            if not channel.is_available:
                continue
            try:
                if channel.send(payload):
                    sent_any = True
            except Exception as exc:
                logger.error("Channel %s failed: %s", channel.name, exc)

        if sent_any:
            self._stats["sent"] += 1
        else:
            self._stats["failed"] += 1

        return sent_any
