"""
DummyVADModel — 실제 체크포인트 없이 VADModel 인터페이스를 구현
================================================================

주기적으로 이상 점수를 생성하여 파이프라인 동작을 시뮬레이션.
- 기본: 0.1~0.3 사이의 정상 점수
- 100프레임마다: 0.6~0.95 사이의 이상 점수 (스파이크)
"""

import math
import random
import time
from typing import Optional, Dict, Any

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class DummyVADModel:
    """실제 모델 없이 VADModel 인터페이스를 구현하는 더미"""

    def __init__(self, spike_interval: int = 100, spike_duration: int = 10):
        self._initialized = False
        self._device: Optional[str] = None
        self._frame_count = 0
        self._spike_interval = spike_interval
        self._spike_duration = spike_duration

    def initialize(self, device: str = "cpu") -> None:
        self._device = device
        self._initialized = True
        self._frame_count = 0

    def process_frame(self, frame: Any = None) -> Optional[float]:
        self._frame_count += 1
        cycle = self._frame_count % self._spike_interval
        if cycle < self._spike_duration:
            progress = cycle / self._spike_duration
            base = 0.5 + 0.4 * math.sin(progress * math.pi)
            return round(base + random.uniform(-0.05, 0.05), 4)
        return round(random.uniform(0.05, 0.25), 4)

    def reset(self) -> None:
        self._frame_count = 0

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def required_frames(self) -> int:
        return 1

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def device(self) -> Optional[str]:
        return self._device

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "required_frames": self.required_frames,
            "initialized": self.is_initialized,
            "device": self.device,
            "frame_count": self._frame_count,
        }
