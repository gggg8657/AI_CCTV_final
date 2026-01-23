"""
IOU-based package tracker implementation.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

from .base import BaseTracker, Detection, EventPublisher, TrackedPackage
from ..utils.events import PackageDetectedEvent, PackageDisappearedEvent


LOGGER = logging.getLogger(__name__)


class PackageTracker(BaseTracker):
    """Track packages across frames using IOU matching.

    Args:
        iou_threshold: IOU threshold for matching.
        max_age: Maximum age (seconds) to keep unmatched packages.
        missing_threshold: Seconds before marking a package missing.
        history_size: Max detections to keep per package.
        event_bus: Optional event bus for emitting package events.
        camera_id: Camera identifier for event metadata.
        id_prefix: Prefix for generated package IDs.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: float = 30.0,
        missing_threshold: float = 1.0,
        history_size: int = 50,
        event_bus: Optional[EventPublisher] = None,
        camera_id: int = 0,
        id_prefix: str = "pkg",
    ) -> None:
        self._iou_threshold = float(iou_threshold)
        self._max_age = float(max_age)
        self._missing_threshold = float(missing_threshold)
        self._history_size = int(history_size)
        self._event_bus = event_bus
        self._camera_id = int(camera_id)
        self._id_prefix = id_prefix
        self._tracked: Dict[str, TrackedPackage] = {}
        self._next_id = 1
        self._frame_index = 0

    def track(self, detections: List[Detection], timestamp: float) -> List[TrackedPackage]:
        """Track detections across frames.

        Args:
            detections: Detections for the current frame.
            timestamp: Monotonic timestamp for timing logic.

        Returns:
            List of tracked packages after update.
        """
        self._frame_index += 1

        if not isinstance(detections, list):
            LOGGER.warning("Detections must be a list")
            return list(self._tracked.values())

        try:
            matched = self._match_detections(detections)
            matched_ids = {pkg_id for pkg_id, _ in matched}
            matched_det_indices = {det_idx for _, det_idx in matched}

            # Update matched packages
            for pkg_id, det_idx in matched:
                detection = detections[det_idx]
                self._update_package(self._tracked[pkg_id], detection, timestamp)

            # Create new packages for unmatched detections
            for det_idx, detection in enumerate(detections):
                if det_idx in matched_det_indices:
                    continue
                self._create_package(detection, timestamp)

            # Update missing packages and remove expired ones
            self._update_missing_and_expired(timestamp, matched_ids)
        except Exception as exc:
            LOGGER.error("Tracking error: %s", exc, exc_info=True)

        return list(self._tracked.values())

    def get_package(self, package_id: str) -> Optional[TrackedPackage]:
        """Get a tracked package by ID."""
        return self._tracked.get(package_id)

    def get_all_packages(self) -> List[TrackedPackage]:
        """Get all tracked packages."""
        return list(self._tracked.values())

    def _create_package(self, detection: Detection, timestamp: float) -> None:
        package_id = f"{self._id_prefix}_{self._next_id:04d}"
        self._next_id += 1

        wall_time = detection.timestamp or time.time()
        history = deque(maxlen=self._history_size)
        history.append(detection)

        package = TrackedPackage(
            package_id=package_id,
            detections=history,
            first_seen=wall_time,
            last_seen=wall_time,
            current_position=detection.bbox,
            status="present",
            last_seen_monotonic=timestamp,
            missing_since_monotonic=None,
            camera_id=self._camera_id,
            disappeared_event_emitted=False,
        )
        self._tracked[package_id] = package

        event = PackageDetectedEvent(
            package_id=package_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            camera_id=self._camera_id,
            frame_index=self._frame_index,
        )
        self._publish_event(event)

    def _update_package(
        self,
        package: TrackedPackage,
        detection: Detection,
        timestamp: float,
    ) -> None:
        wall_time = detection.timestamp or time.time()
        package.detections.append(detection)
        package.last_seen = wall_time
        package.current_position = detection.bbox
        package.last_seen_monotonic = timestamp
        if package.status != "present":
            package.status = "present"
            package.missing_since_monotonic = None
            package.disappeared_event_emitted = False

    def _update_missing_and_expired(self, timestamp: float, matched_ids: Iterable[str]) -> None:
        to_remove: List[str] = []
        for package_id, package in self._tracked.items():
            if package_id in matched_ids:
                continue

            time_since_seen = timestamp - package.last_seen_monotonic
            if package.status == "present" and time_since_seen >= self._missing_threshold:
                package.status = "missing"
                if package.missing_since_monotonic is None:
                    package.missing_since_monotonic = package.last_seen_monotonic
                if not package.disappeared_event_emitted:
                    event = PackageDisappearedEvent(
                        package_id=package.package_id,
                        last_seen=package.last_seen_iso,
                        camera_id=package.camera_id,
                    )
                    self._publish_event(event)
                    package.disappeared_event_emitted = True

            if time_since_seen >= self._max_age and package.status != "present":
                to_remove.append(package_id)

        for package_id in to_remove:
            self._tracked.pop(package_id, None)

    def _match_detections(self, detections: List[Detection]) -> List[Tuple[str, int]]:
        matches: List[Tuple[float, str, int]] = []
        for package_id, package in self._tracked.items():
            for det_idx, detection in enumerate(detections):
                iou = self._calculate_iou(package.current_position, detection.bbox)
                if iou >= self._iou_threshold:
                    matches.append((iou, package_id, det_idx))

        matches.sort(reverse=True, key=lambda item: item[0])
        assigned_packages = set()
        assigned_detections = set()
        result: List[Tuple[str, int]] = []

        for _, package_id, det_idx in matches:
            if package_id in assigned_packages or det_idx in assigned_detections:
                continue
            assigned_packages.add(package_id)
            assigned_detections.add(det_idx)
            result.append((package_id, det_idx))

        return result

    @staticmethod
    def _calculate_iou(
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
    ) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter_width = max(0, x2 - x1)
        inter_height = max(0, y2 - y1)
        inter_area = inter_width * inter_height

        area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
        area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])

        union = area1 + area2 - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _publish_event(self, event: object) -> None:
        if self._event_bus is None:
            return
        try:
            self._event_bus.publish_sync(event)
        except Exception as exc:
            LOGGER.error("Failed to publish event: %s", exc, exc_info=True)
