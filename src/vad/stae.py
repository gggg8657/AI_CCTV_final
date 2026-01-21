"""
STAE (Spatio-Temporal AutoEncoder) VAD 모델
============================================

Spatio-Temporal Autoencoder for Video Anomaly Detection

원본: models/STAE_original
체크포인트: experiments/stae/stae_best.pt
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
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
# 원본 LSTMAutoEncoder 모델 체크포인트
DEFAULT_MODEL_PATH = PROJECT_ROOT / "experiments" / "stae_original" / "stae_best.pt"


class STAEModel(VADModel):
    """
    STAE VAD 모델
    
    Spatio-Temporal Autoencoder (ConvLSTM based)
    - 입력: T개 프레임의 시퀀스
    - 출력: 재구성된 시퀀스
    - 이상 점수: 재구성 오차 (MSE)
    
    실제 학습된 체크포인트 사용 (AUC 64.92%)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        t_length: int = 10,
        input_size: tuple = (256, 256),
        stride: int = 1,
    ):
        super().__init__()
        
        self.model_path = model_path or str(DEFAULT_MODEL_PATH)
        self.t_length = t_length
        self.input_size = input_size
        self.stride = stride
        
        self.model = None
        self._frame_buffer: deque = deque(maxlen=t_length)
        self._frames_since_inference = 0
    
    def initialize(self, device: str = 'cuda:0') -> None:
        """모델 초기화 및 체크포인트 로드"""
        import importlib.util
        
        self._device = device
        
        # 모델 경로
        stae_path = PROJECT_ROOT / "models" / "STAE_original"
        model_file = stae_path / "model.py"
        modules_file = stae_path / "modules.py"
        
        if not model_file.exists():
            raise FileNotFoundError(f"STAE 모델 파일을 찾을 수 없습니다: {model_file}")
        
        # modules.py 먼저 로드
        try:
            spec = importlib.util.spec_from_file_location("stae_modules", modules_file)
            stae_modules = importlib.util.module_from_spec(spec)
            sys.modules["modules"] = stae_modules
            spec.loader.exec_module(stae_modules)
            
            # model.py 로드
            spec = importlib.util.spec_from_file_location("stae_model", model_file)
            stae_model = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(stae_model)
            
            LSTMAutoEncoder = stae_model.LSTMAutoEncoder
        except Exception as e:
            raise ImportError(f"STAE 모델을 import할 수 없습니다. 원인: {e}")
        
        # 모델 생성
        self.model = LSTMAutoEncoder(in_channels=3, time_steps=self.t_length)
        
        # 체크포인트 로드
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"[STAE] 체크포인트 로드 완료: {self.model_path}")
        else:
            print(f"[STAE] 경고: 체크포인트 없음, 초기화된 가중치 사용")
        
        self.model.to(device)
        self.model.eval()
        
        self._initialized = True
        print(f"[STAE] 초기화 완료 (device: {device})")
    
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
        self._frames_since_inference += 1
        
        # 프레임 수 및 stride 확인
        if len(self._frame_buffer) < self.t_length:
            return None
        
        if self._frames_since_inference < self.stride:
            return None
        
        self._frames_since_inference = 0
        return self._compute_score()
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """프레임 전처리"""
        # 리사이즈
        if frame.shape[0] != self.input_size[0] or frame.shape[1] != self.input_size[1]:
            frame = cv2.resize(frame, self.input_size)
        
        # 정규화 [0, 1]
        frame = frame.astype(np.float32) / 255.0
        return frame
    
    def _compute_score(self) -> float:
        """이상 점수 계산 (재구성 오차)"""
        # (T, H, W, C) -> (1, T, C, H, W)
        frames = np.stack(list(self._frame_buffer), axis=0)
        input_tensor = torch.from_numpy(frames).float()
        input_tensor = input_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
        input_tensor = input_tensor.unsqueeze(0).to(self._device)  # (1, T, C, H, W)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            if isinstance(output, tuple):
                recon = output[0]
            else:
                recon = output
        
        # MSE 기반 이상 점수
        mse = torch.mean((input_tensor - recon) ** 2).item()
        
        return mse
    
    def reset(self) -> None:
        """상태 초기화"""
        self._frame_buffer.clear()
        self._frames_since_inference = 0
    
    @property
    def name(self) -> str:
        return "STAE"
    
    @property
    def required_frames(self) -> int:
        return self.t_length



