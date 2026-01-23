"""
YOLO-based package detector implementation.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .base import BaseDetector, Detection


LOGGER = logging.getLogger(__name__)

# Default COCO mapping for package-like objects (per requirements).
DEFAULT_PACKAGE_CLASS_IDS = (26, 27, 28)
DEFAULT_CLASS_NAMES: Dict[int, str] = {
    26: "handbag",
    27: "backpack",
    28: "suitcase",
}


class PackageDetector(BaseDetector):
    """Package detector using YOLO v12 nano via ultralytics.

    Args:
        model_path: Path or model name for YOLO.
        device: Device string ("cuda" or "cpu").
        confidence_threshold: Confidence threshold for detections.
        target_class_ids: COCO class IDs to treat as packages.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        target_class_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._confidence_threshold = float(confidence_threshold)
        self._target_class_ids = tuple(target_class_ids or DEFAULT_PACKAGE_CLASS_IDS)
        self._model = None

    def load_model(self) -> bool:
        """Load YOLO model.

        Returns:
            True if model loaded successfully, otherwise False.
        """
        try:
            from ultralytics import YOLO

            self._model = YOLO(self._model_path)
            return True
        except Exception as exc:
            LOGGER.error("Failed to load YOLO model: %s", exc, exc_info=True)
            self._model = None
            return False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect packages in a frame.

        Args:
            frame: Input frame as numpy.ndarray.

        Returns:
            List of detections.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            LOGGER.warning("Invalid frame input for detection")
            return []

        if self._model is None and not self.load_model():
            return []

        try:
            results = self._model.predict(
                source=frame,
                device=self._device,
                conf=self._confidence_threshold,
                verbose=False,
            )
        except Exception as exc:
            LOGGER.error("YOLO inference failed: %s", exc, exc_info=True)
            return []

        detections: List[Detection] = []
        timestamp = time.time()

        try:
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                for idx in range(len(boxes)):
                    class_id = int(boxes.cls[idx].item())
                    confidence = float(boxes.conf[idx].item())
                    if class_id not in self._target_class_ids:
                        continue
                    if confidence < self._confidence_threshold:
                        continue

                    xyxy = boxes.xyxy[idx].tolist()
                    bbox = (
                        int(xyxy[0]),
                        int(xyxy[1]),
                        int(xyxy[2]),
                        int(xyxy[3]),
                    )
                    class_name = DEFAULT_CLASS_NAMES.get(class_id, "package")
                    detections.append(
                        Detection(
                            bbox=bbox,
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name,
                            timestamp=timestamp,
                        )
                    )
        except Exception as exc:
            LOGGER.error("Failed to parse YOLO results: %s", exc, exc_info=True)
            return []

        return detections
