"""
FlowNet2 Optical Flow 계산 래퍼
==============================

원본 리포지토리의 FlowNet2 모델을 사용하여 optical flow 계산
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn

# 원본 리포지토리 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ATTRIBUTE_VAD_ROOT = PROJECT_ROOT / "models" / "attribute_vad_original"
if str(ATTRIBUTE_VAD_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTRIBUTE_VAD_ROOT))

# FlowNet2 모델 import
try:
    from pre_processing.flownet_networks.flownet2_models import FlowNet2
    HAS_FLOWNET2 = True
except ImportError as e:
    HAS_FLOWNET2 = False
    warnings.warn(f"Failed to import FlowNet2: {e}. FlowNet2 features will not work.")


class FlowNet2Wrapper:
    """
    FlowNet2 래퍼 클래스
    
    두 프레임 간의 optical flow를 계산합니다.
    """
    
    # FlowNet2 입력 크기 (원본 리포지토리 기준)
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 640
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda:0',
        input_width: int = None,
        input_height: int = None,
    ):
        """
        FlowNet2 래퍼 초기화
        
        Args:
            checkpoint_path: FlowNet2 체크포인트 경로
            device: 디바이스 (cuda:0, cuda:1, cpu 등)
            input_width: 입력 이미지 너비 (None이면 DEFAULT_WIDTH 사용)
            input_height: 입력 이미지 높이 (None이면 DEFAULT_HEIGHT 사용)
        """
        if not HAS_FLOWNET2:
            raise RuntimeError("FlowNet2 모듈을 import할 수 없습니다. 의존성을 확인하세요.")
        
        self.device = device
        self.device_obj = torch.device(device)
        self.input_width = input_width or self.DEFAULT_WIDTH
        self.input_height = input_height or self.DEFAULT_HEIGHT
        
        # 체크포인트 경로 설정
        if checkpoint_path is None:
            # 기본 경로 시도
            default_paths = [
                ATTRIBUTE_VAD_ROOT / "pre_processing" / "checkpoints" / "FlowNet2_checkpoint.pth.tar",
                PROJECT_ROOT / "models" / "attribute_vad_original" / "pre_processing" / "checkpoints" / "FlowNet2_checkpoint.pth.tar",
            ]
            for path in default_paths:
                if path.exists():
                    checkpoint_path = str(path)
                    break
            
            if checkpoint_path is None:
                warnings.warn(
                    f"FlowNet2 체크포인트를 찾을 수 없습니다. "
                    f"다음 경로를 확인하세요: {default_paths}"
                )
        
        self.checkpoint_path = checkpoint_path
        self.model = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """FlowNet2 모델 초기화 및 체크포인트 로드"""
        if self._initialized:
            return True
        
        try:
            # FlowNet2 모델 생성
            self.model = FlowNet2()
            
            # 체크포인트 로드
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                print(f"[FlowNet2] Loading checkpoint from {self.checkpoint_path}")
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
                
                # state_dict 추출
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # 모델 state_dict와 매칭
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)
                print(f"[FlowNet2] Checkpoint loaded successfully")
            else:
                warnings.warn(
                    f"[FlowNet2] Checkpoint not found at {self.checkpoint_path}. "
                    f"Using randomly initialized weights."
                )
            
            # 모델을 디바이스로 이동
            self.model.to(self.device_obj)
            self.model.eval()
            
            self._initialized = True
            print(f"[FlowNet2] Initialized on {self.device}")
            return True
            
        except Exception as e:
            warnings.warn(f"[FlowNet2] Initialization failed: {e}")
            return False
    
    def compute_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        original_size: Optional[Tuple[int, int]] = None
    ) -> Optional[np.ndarray]:
        """
        두 프레임 간의 optical flow 계산
        
        Args:
            frame1: 첫 번째 프레임 (H, W, 3) RGB, 0-255 범위
            frame2: 두 번째 프레임 (H, W, 3) RGB, 0-255 범위
            original_size: 원본 프레임 크기 (W, H). None이면 frame1 크기 사용
        
        Returns:
            Optical flow (H, W, 2) 또는 None (실패 시)
        """
        if not self._initialized:
            if not self.initialize():
                return None
        
        if self.model is None:
            return None
        
        try:
            # 원본 크기 저장
            if original_size is None:
                original_size = (frame1.shape[1], frame1.shape[0])  # (W, H)
            
            old_size = original_size
            
            # 프레임 전처리: FlowNet2 입력 크기로 resize
            im1 = cv2.resize(frame1, (self.input_width, self.input_height))
            im2 = cv2.resize(frame2, (self.input_width, self.input_height))
            
            # numpy array로 변환 및 정규화
            ims = np.array([im1, im2]).astype(np.float32)  # [2, H, W, 3]
            
            # Tensor로 변환: [bs, 3, 2, H, W] 형식
            ims_tensor = torch.from_numpy(ims).unsqueeze(0)  # [1, 2, H, W, 3]
            ims_tensor = ims_tensor.permute(0, 4, 1, 2, 3).contiguous()  # [1, 3, 2, H, W]
            ims_tensor = ims_tensor.to(self.device_obj)
            
            # FlowNet2 추론
            with torch.no_grad():
                pred_flow = self.model(ims_tensor)  # [1, 2, H', W']
                pred_flow = pred_flow[0].cpu().data.numpy()  # [2, H', W']
                pred_flow = pred_flow.transpose((1, 2, 0))  # [H', W', 2]
            
            # 원본 크기로 resize
            flow_resized = cv2.resize(pred_flow, old_size)  # [H, W, 2]
            
            return flow_resized
            
        except Exception as e:
            warnings.warn(f"[FlowNet2] Flow computation failed: {e}")
            return None
    
    def is_initialized(self) -> bool:
        """초기화 여부 확인"""
        return self._initialized
    
    def __del__(self):
        """리소스 정리"""
        if self.model is not None:
            del self.model
            self.model = None




