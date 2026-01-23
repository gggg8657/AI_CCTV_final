"""
Base interfaces and data models for package detection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, List, Optional, Protocol, Tuple, Literal

import numpy as np

from ..utils.events import TheftDetectedEvent


PackageStatus = Literal["present", "missing", "stolen"]


class EventPublisher(Protocol):
    """Minimal event publisher protocol for dependency inversion."""

    def publish_sync(self, event: object) -> None:  # pragma: no cover - protocol
        ...

    def publish(self, event: object) -> object:  # pragma: no cover - protocol
        ...


def _iso_from_epoch(seconds: float) -> str:
    """Convert epoch seconds to ISO-8601 string (UTC)."""
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()


@dataclass
class Detection:
    """Detection result for a single frame.

    Attributes:
        bbox: Bounding box in (x1, y1, x2, y2) format.
        confidence: Detection confidence (0.0 ~ 1.0).
        class_id: COCO class ID.
        class_name: Class label name.
        timestamp: Wall-clock time in epoch seconds.
    """

    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str = "package"
    timestamp: float = 0.0

    @property
    def timestamp_iso(self) -> str:
        """ISO-8601 formatted timestamp (UTC)."""
        return _iso_from_epoch(self.timestamp)


@dataclass
class TrackedPackage:
    """Tracked package state across frames.

    Attributes:
        package_id: Unique package identifier.
        detections: Recent detections for this package.
        first_seen: Wall-clock epoch when first detected.
        last_seen: Wall-clock epoch when last detected.
        current_position: Latest bounding box.
        status: Package status (present, missing, stolen).
        last_seen_monotonic: Monotonic timestamp when last detected.
        missing_since_monotonic: Monotonic timestamp when missing started.
        camera_id: Camera identifier.
    """

    package_id: str
    detections: Deque[Detection] = field(default_factory=deque)
    first_seen: float = 0.0
    last_seen: float = 0.0
    current_position: Tuple[int, int, int, int] = (0, 0, 0, 0)
    status: PackageStatus = "present"
    last_seen_monotonic: float = 0.0
    missing_since_monotonic: Optional[float] = None
    camera_id: int = 0
    disappeared_event_emitted: bool = False

    @property
    def first_seen_iso(self) -> str:
        """ISO-8601 formatted first seen time (UTC)."""
        return _iso_from_epoch(self.first_seen)

    @property
    def last_seen_iso(self) -> str:
        """ISO-8601 formatted last seen time (UTC)."""
        return _iso_from_epoch(self.last_seen)

    def to_dict(self) -> dict:
        """Convert tracked package to a JSON-serializable dict.

        Returns:
            Dictionary representation of the tracked package.
        """
        return {
            "package_id": self.package_id,
            "first_seen": self.first_seen_iso,
            "last_seen": self.last_seen_iso,
            "status": self.status,
            "current_position": self.current_position,
            "detection_count": len(self.detections),
            "camera_id": self.camera_id,
        }


class BaseDetector(ABC):
    """Base interface for package detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect packages in a frame.

        Args:
            frame: Input frame as a numpy.ndarray.

        Returns:
            List of detections (may be empty).
        """
        raise NotImplementedError


class BaseTracker(ABC):
    """Base interface for package trackers."""

    @abstractmethod
    def track(self, detections: List[Detection], timestamp: float) -> List[TrackedPackage]:
        """Track detections across frames.

        Args:
            detections: Detections for the current frame.
            timestamp: Monotonic timestamp for timing logic.

        Returns:
            List of tracked packages after update.
        """
        raise NotImplementedError


class BaseTheftDetector(ABC):
    """Base interface for theft detection."""

    @abstractmethod
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
        raise NotImplementedError
