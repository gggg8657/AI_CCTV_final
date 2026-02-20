"""
DummyVideoSource — 실제 카메라/파일 없이 합성 프레임을 생성
============================================================

OpenCV 없이도 동작: numpy만으로 컬러 노이즈 프레임 생성.
OpenCV가 있으면 텍스트 오버레이도 추가.
"""

import time
from datetime import datetime
from typing import Tuple, Any, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class DummyVideoSource:
    """합성 프레임을 생성하는 더미 비디오 소스"""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        total_frames: int = 0,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.total_frames = total_frames
        self._frame_count = 0
        self._opened = False

    def open(self) -> bool:
        self._opened = True
        self._frame_count = 0
        return True

    def read(self) -> Tuple[bool, Optional[Any]]:
        if not self._opened:
            return False, None

        if self.total_frames > 0 and self._frame_count >= self.total_frames:
            return False, None

        self._frame_count += 1
        frame = self._generate_frame()
        return True, frame

    def read_rgb(self) -> Tuple[bool, Optional[Any]]:
        return self.read()

    def close(self) -> None:
        self._opened = False

    def get_info(self) -> dict:
        return {
            "type": "dummy",
            "path": "synthetic",
            "fps": self.fps,
            "resolution": f"{self.width}x{self.height}",
            "total_frames": self.total_frames,
        }

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _generate_frame(self) -> Any:
        if not HAS_NUMPY:
            return None

        t = self._frame_count / max(self.fps, 1)
        base_color = np.array([
            int(127 + 80 * np.sin(t * 0.5)),
            int(127 + 80 * np.sin(t * 0.3 + 2)),
            int(127 + 80 * np.sin(t * 0.7 + 4)),
        ], dtype=np.uint8)

        frame = np.full((self.height, self.width, 3), base_color, dtype=np.uint8)
        noise = np.random.randint(0, 20, (self.height, self.width, 3), dtype=np.uint8)
        frame = np.clip(frame.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)

        if HAS_CV2:
            ts_text = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            cv2.putText(frame, f"DUMMY #{self._frame_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, ts_text, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        return frame
