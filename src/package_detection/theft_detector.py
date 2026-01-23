"""
Theft detection logic implementation.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque, List, Optional

from .base import BaseTheftDetector, EventPublisher, TrackedPackage
from ..utils.events import TheftDetectedEvent


LOGGER = logging.getLogger(__name__)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TheftDetector(BaseTheftDetector):
    """Detect package theft based on missing duration.

    Args:
        confirmation_time: Seconds a package must be missing to be theft.
        evidence_buffer_size: Max number of evidence frame paths to keep.
        event_bus: Optional event bus for emitting theft events.
        camera_id: Camera identifier for event metadata.
    """

    def __init__(
        self,
        confirmation_time: float = 3.0,
        evidence_buffer_size: int = 10,
        event_bus: Optional[EventPublisher] = None,
        camera_id: int = 0,
    ) -> None:
        self._confirmation_time = float(confirmation_time)
        self._event_bus = event_bus
        self._camera_id = int(camera_id)
        self._evidence_frames: Deque[str] = deque(maxlen=int(evidence_buffer_size))

    def add_evidence_frame(self, frame_path: str) -> None:
        """Add an evidence frame path to the ring buffer."""
        if not frame_path:
            return
        self._evidence_frames.append(frame_path)

    def check_theft(
        self,
        tracked_packages: List[TrackedPackage],
        timestamp: float,
    ) -> Optional[TheftDetectedEvent]:
        """Check theft conditions.

        Args:
            tracked_packages: Current tracked packages.
            timestamp: Monotonic timestamp for timing logic.

        Returns:
            Theft event if detected, otherwise None.
        """
        event_to_return: Optional[TheftDetectedEvent] = None
        try:
            for package in tracked_packages:
                if package.status == "stolen":
                    continue
                if package.status != "missing" or package.missing_since_monotonic is None:
                    continue

                missing_duration = timestamp - package.missing_since_monotonic
                if missing_duration < self._confirmation_time:
                    continue

                package.status = "stolen"
                theft_event = TheftDetectedEvent(
                    package_id=package.package_id,
                    theft_time=_iso_now(),
                    camera_id=package.camera_id or self._camera_id,
                    evidence_frame_paths=list(self._evidence_frames),
                )
                self._publish_event(theft_event)
                if event_to_return is None:
                    event_to_return = theft_event
        except Exception as exc:
            LOGGER.error("Theft detection error: %s", exc, exc_info=True)

        return event_to_return

    def _publish_event(self, event: TheftDetectedEvent) -> None:
        if self._event_bus is None:
            return
        try:
            self._event_bus.publish_sync(event)
        except Exception as exc:
            LOGGER.error("Failed to publish theft event: %s", exc, exc_info=True)
