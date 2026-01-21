"""
MNAD (Memory-guided Normality AutoEncoder) VAD 모델
===================================================

Memory-guided Normality Autoencoder for Video Anomaly Detection

원본: models/mnad
체크포인트: experiments/mnad/model.pth + keys.pt
"""

import os
import sys
from pathlib import Path
from collections import deque
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from .base import VADModel


# 기본 체크포인트 경로 (SCI 프로젝트 루트 기준)
# sci_v2/src/vad/mnad.py -> SCI/
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "experiments" / "mnad" / "model.pth"
DEFAULT_KEYS_PATH = PROJECT_ROOT / "experiments" / "mnad" / "keys.pt"


class MNADModel(VADModel):
    """
    MNAD VAD 모델
    
    Memory-guided Normality Autoencoder
    - 입력: 4개 프레임 (12채널)
    - 출력: 다음 프레임 예측
    - 이상 점수: 재구성 오차 (MSE)
    - Keys: Memory 모듈용 attention keys
    
    실제 학습된 체크포인트 사용 (AUC 79.27%)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        keys_path: Optional[str] = None,
        t_length: int = 5,  # 4 input + 1 target
        input_size: tuple = (256, 256),
    ):
        super().__init__()
        
        self.model_path = model_path or str(DEFAULT_MODEL_PATH)
        self.keys_path = keys_path or str(DEFAULT_KEYS_PATH)
        self.t_length = t_length
        self.input_size = input_size
        
        self.model = None
        self.m_items = None  # Memory keys
        self._frame_buffer: deque = deque(maxlen=t_length)
    
    def initialize(self, device: str = 'cuda:0') -> None:
        """모델 초기화 및 체크포인트 로드"""
        self._device = device
        
        # 모델 경로를 sys.path에 추가
        mnad_path = PROJECT_ROOT / "models" / "mnad"
        if str(mnad_path) not in sys.path:
            sys.path.insert(0, str(mnad_path))
        
        # 모델 로드
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"MNAD 모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        self.model = torch.load(self.model_path, map_location=device, weights_only=False)
        self.model.to(device)
        self.model.eval()
        print(f"[MNAD] 모델 로드 완료: {self.model_path}")
        
        # Keys 로드
        if not os.path.exists(self.keys_path):
            raise FileNotFoundError(f"MNAD keys 파일을 찾을 수 없습니다: {self.keys_path}")
        
        self.m_items = torch.load(self.keys_path, map_location=device, weights_only=False)
        print(f"[MNAD] Keys 로드 완료: {self.keys_path} (shape: {self.m_items.shape})")
        
        self._initialized = True
        print(f"[MNAD] 초기화 완료 (device: {device})")
    
    def process_frame(self, frame: np.ndarray) -> Optional[float]:
        """
        프레임 처리 및 이상 점수 반환
        
        Args:
            frame: RGB 프레임 (H, W, C)
        
        Returns:
            이상 점수 (MSE) 또는 None (프레임 부족 시)
        """
        if not self._initialized:
            raise RuntimeError("모델이 초기화되지 않았습니다. initialize()를 먼저 호출하세요.")
        
        # 프레임 전처리
        processed = self._preprocess_frame(frame)
        self._frame_buffer.append(processed)
        
        # 프레임 수 확인
        if len(self._frame_buffer) < self.t_length:
            return None
        
        # 추론
        return self._compute_score()
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """프레임 전처리"""
        # RGB -> BGR (optional, depends on training)
        # 리사이즈
        if frame.shape[0] != self.input_size[0] or frame.shape[1] != self.input_size[1]:
            frame = cv2.resize(frame, self.input_size)
        
        # 정규화 [-1, 1]
        frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 127.5 - 1.0
        return frame_tensor
    
    def _compute_score(self) -> float:
        """이상 점수 계산 (재구성 오차)"""
        # MNAD prediction: 4 input frames -> predict next frame
        # Input: concat of 4 frames = 12 channels
        frames = list(self._frame_buffer)[:-1]  # 처음 4개 프레임
        target = self._frame_buffer[-1]  # 마지막 프레임 (예측 타겟)
        
        # (4, 3, H, W) -> (1, 12, H, W)
        input_tensor = torch.cat(frames, dim=0).unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            # MNAD forward (test mode) returns multiple values
            # output, fea, updated_fea, keys, softmax_query, softmax_memory, query, top1_keys, keys_ind, compactness_loss
            result = self.model.forward(input_tensor, self.m_items, False)
            output = result[0]
        
        # MSE 기반 이상 점수 (reconstruction error)
        target_tensor = target.unsqueeze(0).to(self._device)
        mse = torch.mean((output - target_tensor) ** 2).item()
        
        return mse
    
    def reset(self) -> None:
        """상태 초기화"""
        self._frame_buffer.clear()
    
    @property
    def name(self) -> str:
        return "MNAD"
    
    @property
    def required_frames(self) -> int:
        return self.t_length



