"""
MemAE (Memory-augmented AutoEncoder) VAD 모델
==============================================

Memory-augmented Autoencoder for Video Anomaly Detection

원본: models/memae
체크포인트: experiments/memae/.../epoch_0029_final.pt
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
DEFAULT_MODEL_PATH = PROJECT_ROOT / "experiments" / "memae" / "model_MemAE_Conv3DSpar_custom_MemDim2000_EntW0.0002_ShrThres0.0025_Seed1_custom" / "MemAE_Conv3DSpar_custom_MemDim2000_EntW0.0002_ShrThres0.0025_Seed1_custom_epoch_0029_final.pt"


class MemAEModel(VADModel):
    """
    MemAE VAD 모델
    
    Memory-augmented 3D Convolutional Autoencoder
    - 입력: T개 프레임의 3D 볼륨
    - 출력: 재구성된 볼륨
    - 이상 점수: 재구성 오차 (MSE)
    
    실제 학습된 체크포인트 사용 (AUC 72.91%)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        t_length: int = 16,
        input_size: tuple = (256, 256),
        mem_dim: int = 2000,
    ):
        super().__init__()
        
        self.model_path = model_path or str(DEFAULT_MODEL_PATH)
        self.t_length = t_length
        self.input_size = input_size
        self.mem_dim = mem_dim
        
        self.model = None
        self._frame_buffer: deque = deque(maxlen=t_length)
    
    def initialize(self, device: str = 'cuda:0') -> None:
        """모델 초기화 및 체크포인트 로드"""
        self._device = device
        
        # 모델 경로를 sys.path에 추가
        memae_path = PROJECT_ROOT / "models" / "memae"
        memae_models_path = memae_path / "models"
        
        # models/memae/models를 sys.path에 추가하고 models 패키지를 먼저 로드
        if str(memae_models_path) not in sys.path:
            sys.path.insert(0, str(memae_models_path))
        
        # models 패키지를 먼저 로드 (memae_3dconv가 from models import MemModule을 사용)
        try:
            import importlib.util
            # memory_module을 먼저 로드
            memory_spec = importlib.util.spec_from_file_location(
                "memory_module",
                memae_models_path / "memory_module.py"
            )
            memory_module = importlib.util.module_from_spec(memory_spec)
            memory_spec.loader.exec_module(memory_module)
            
            # models 패키지 생성 및 MemModule 추가
            import types
            memae_models_pkg = types.ModuleType('models')
            memae_models_pkg.MemModule = memory_module.MemModule
            sys.modules['models'] = memae_models_pkg
            
            # 이제 memae_3dconv를 로드
            memae_spec = importlib.util.spec_from_file_location(
                "memae_3dconv",
                memae_models_path / "memae_3dconv.py"
            )
            memae_module = importlib.util.module_from_spec(memae_spec)
            memae_spec.loader.exec_module(memae_module)
            AutoEncoderCov3DMem = memae_module.AutoEncoderCov3DMem
        except Exception as e:
            raise ImportError(f"MemAE 모델을 import할 수 없습니다. models/memae 경로를 확인하세요. 원인: {e}")
        
        # 모델 생성 및 체크포인트 로드
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"MemAE 모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        # 모델 구조 생성 (체크포인트가 3채널 입력으로 학습됨)
        self.model = AutoEncoderCov3DMem(
            chnum_in=3,  # RGB
            mem_dim=self.mem_dim,
            shrink_thres=0.0025
        )
        
        # 체크포인트 로드
        checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.model.eval()
        
        self._initialized = True
        print(f"[MemAE] 초기화 완료 (device: {device})")
    
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
        
        # 프레임 전처리 (grayscale)
        processed = self._preprocess_frame(frame)
        self._frame_buffer.append(processed)
        
        # 프레임 수 확인
        if len(self._frame_buffer) < self.t_length:
            return None
        
        # 추론
        return self._compute_score()
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """프레임 전처리 (RGB)"""
        # 리사이즈
        if frame.shape[0] != self.input_size[0] or frame.shape[1] != self.input_size[1]:
            frame = cv2.resize(frame, self.input_size)
        
        # 정규화 [0, 1]
        frame = frame.astype(np.float32) / 255.0
        return frame
    
    def _compute_score(self) -> float:
        """이상 점수 계산 (재구성 오차)"""
        # (T, H, W, C) -> (1, C, T, H, W)
        frames = np.stack(list(self._frame_buffer), axis=0)  # (T, H, W, C)
        input_tensor = torch.from_numpy(frames).float()
        input_tensor = input_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
        input_tensor = input_tensor.unsqueeze(0).to(self._device)  # (1, C, T, H, W)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            # MemAE는 dict를 반환: {'output': tensor, 'att': tensor}
            if isinstance(output, dict):
                recon = output['output']
            elif isinstance(output, tuple):
                recon = output[0]
            else:
                recon = output
        
        # MSE 기반 이상 점수
        mse = torch.mean((input_tensor - recon) ** 2).item()
        
        return mse
    
    def reset(self) -> None:
        """상태 초기화"""
        self._frame_buffer.clear()
    
    @property
    def name(self) -> str:
        return "MemAE"
    
    @property
    def required_frames(self) -> int:
        return self.t_length



