"""
VAD 모델 기본 추상 클래스
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import numpy as np


class VADModel(ABC):
    """
    VAD 모델 추상 기본 클래스
    
    모든 VAD 모델은 이 인터페이스를 구현해야 합니다.
    """
    
    def __init__(self):
        self._initialized = False
        self._device = None
    
    @abstractmethod
    def initialize(self, device: str = 'cuda:0') -> None:
        """
        모델 초기화 및 체크포인트 로드
        
        Args:
            device: 디바이스 (cuda:0, cuda:1, cpu 등)
        """
        pass
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Optional[float]:
        """
        단일 프레임 처리 및 이상 점수 반환
        
        Args:
            frame: RGB 프레임 (H, W, C)
        
        Returns:
            이상 점수 (0~1) 또는 None (프레임 부족 시)
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        내부 상태 초기화 (프레임 버퍼 등)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """모델 이름"""
        pass
    
    @property
    @abstractmethod
    def required_frames(self) -> int:
        """추론에 필요한 최소 프레임 수"""
        pass
    
    @property
    def is_initialized(self) -> bool:
        """초기화 여부"""
        return self._initialized
    
    @property
    def device(self) -> Optional[str]:
        """현재 디바이스"""
        return self._device
    
    def get_info(self) -> Dict[str, Any]:
        """모델 정보"""
        return {
            "name": self.name,
            "required_frames": self.required_frames,
            "initialized": self.is_initialized,
            "device": self.device,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(initialized={self.is_initialized}, device={self.device})"



