"""
STEAD (Spatio-Temporal Efficient Anomaly Detection) VAD 모델
=============================================================

STEAD: Performer-based Spatio-Temporal Efficient Anomaly Detection

원본: models/STEAD_original
체크포인트: experiments/stead/stead_best.pt
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
# 원본 Performer 기반 STEAD 모델 체크포인트
DEFAULT_MODEL_PATH = PROJECT_ROOT / "experiments" / "stead_original" / "stead_final.pt"


class STEADModel(VADModel):
    """
    STEAD VAD 모델
    
    Performer-based Spatio-Temporal Efficient Anomaly Detection
    - X3D 특징 추출 필요
    - 입력: X3D 특징 (192, T, H, W)
    - 출력: 이상 점수
    
    실제 학습된 체크포인트 사용 (AUC 69.47%, FPS 320)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        clip_length: int = 13,  # 원본: x3d_s는 13 프레임
        clip_stride: int = 4,
        input_size: tuple = (182, 182),  # 원본: x3d_s는 182x182
    ):
        super().__init__()
        
        self.model_path = model_path or str(DEFAULT_MODEL_PATH)
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.input_size = input_size
        
        self.model = None
        self.x3d_model = None
        self._frame_buffer: deque = deque(maxlen=clip_length + clip_stride)
        self._frames_since_inference = 0
    
    def initialize(self, device: str = 'cuda:0') -> None:
        """모델 초기화 및 체크포인트 로드"""
        import importlib.util
        
        self._device = device
        
        # 모델 경로
        stead_path = PROJECT_ROOT / "models" / "STEAD"
        model_file = stead_path / "model.py"
        utils_file = stead_path / "utils.py"
        
        if not model_file.exists():
            raise FileNotFoundError(f"STEAD 모델 파일을 찾을 수 없습니다: {model_file}")
        
        # option 모듈 패치 (utils.py 의존성 해결)
        import types
        option_mock = types.ModuleType('option')
        option_mock.parse_args = lambda: types.SimpleNamespace()
        sys.modules['option'] = option_mock
        
        # utils.py 먼저 로드
        try:
            spec = importlib.util.spec_from_file_location("stead_utils", utils_file)
            stead_utils = importlib.util.module_from_spec(spec)
            sys.modules["utils"] = stead_utils
            spec.loader.exec_module(stead_utils)
            
            # model.py 로드
            spec = importlib.util.spec_from_file_location("stead_model", model_file)
            stead_model = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(stead_model)
            
            Model = stead_model.Model
        except Exception as e:
            raise ImportError(f"STEAD 모델을 import할 수 없습니다. 원인: {e}")
        
        # STEAD 모델 생성
        self.model = Model(
            dropout=0.2,
            attn_dropout=0.1,
            ff_mult=4,
            dims=(192, 128),
            depths=(3, 3),
            block_types=('c', 'a')
        )
        
        # 체크포인트 로드 (CPU로 먼저 로드한 후 GPU로 이동)
        # GPU 3번을 직접 지정할 수 없으므로 CPU로 로드 후 수동 이동
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"[STEAD] 체크포인트 로드 완료: {self.model_path}")
        else:
            print(f"[STEAD] 경고: 체크포인트 없음, 초기화된 가중치 사용")
        
        self.model.to(device)
        self.model.eval()
        
        # X3D 특징 추출기 로드 (선택적)
        self._load_x3d_extractor(device)
        
        self._initialized = True
        print(f"[STEAD] 초기화 완료 (device: {device})")
    
    def _load_x3d_extractor(self, device: str):
        """X3D 특징 추출기 로드"""
        try:
            self.x3d_model = torch.hub.load(
                'facebookresearch/pytorchvideo', 
                'x3d_s', 
                pretrained=True
            )
            # 마지막 분류층 제거
            self.x3d_model.blocks = self.x3d_model.blocks[:-1]
            self.x3d_model = self.x3d_model.to(device)
            self.x3d_model.eval()
            print("[STEAD] X3D 특징 추출기 로드 완료")
        except Exception as e:
            print(f"[STEAD] 경고: X3D 특징 추출기 로드 실패: {e}")
            self.x3d_model = None
    
    def process_frame(self, frame: np.ndarray) -> Optional[float]:
        """
        프레임 처리 및 이상 점수 반환
        
        Args:
            frame: RGB 프레임 (H, W, C)
        
        Returns:
            이상 점수 또는 None (프레임 부족 시)
        """
        if not self._initialized:
            raise RuntimeError("모델이 초기화되지 않았습니다. initialize()를 먼저 호출하세요.")
        
        # 프레임 전처리
        processed = self._preprocess_frame(frame)
        self._frame_buffer.append(processed)
        self._frames_since_inference += 1
        
        # 프레임 수 및 stride 확인
        if len(self._frame_buffer) < self.clip_length:
            return None
        
        if self._frames_since_inference < self.clip_stride:
            return None
        
        self._frames_since_inference = 0
        return self._compute_score()
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """프레임 전처리 (원본 STEAD 방식: ShortSideScale + CenterCrop)"""
        # 원본 STEAD는 ShortSideScale + CenterCrop 사용
        # 여기서는 간단히 리사이즈로 대체 (실시간 처리 고려)
        h, w = frame.shape[:2]
        target_size = self.input_size[0]  # 182
        
        # ShortSideScale: 짧은 변을 target_size로 조정
        if h < w:
            new_h = target_size
            new_w = int(w * target_size / h)
        else:
            new_w = target_size
            new_h = int(h * target_size / w)
        
        frame = cv2.resize(frame, (new_w, new_h))
        
        # CenterCrop: 중앙에서 target_size x target_size 크롭
        h, w = frame.shape[:2]
        start_h = (h - target_size) // 2
        start_w = (w - target_size) // 2
        frame = frame[start_h:start_h + target_size, start_w:start_w + target_size]
        
        return frame
    
    def _compute_score(self) -> float:
        """이상 점수 계산"""
        # 클립 생성
        clip = np.array(list(self._frame_buffer)[-self.clip_length:])
        
        # X3D 특징 추출
        features = self._extract_x3d_features(clip)
        
        with torch.no_grad():
            logits, _ = self.model(features)
        
        # 원본 코드 방식: sigmoid 적용하여 probability로 변환
        # (models/STEAD/test.py: scores = torch.nn.Sigmoid()(scores))
        score = torch.sigmoid(logits).item()
        return score
    
    def _extract_x3d_features(self, clip: np.ndarray) -> torch.Tensor:
        """X3D 특징 추출 (원본 STEAD 방식)"""
        if self.x3d_model is None:
            # X3D 모델이 로드되지 않음 - 에러 발생
            raise RuntimeError(
                "X3D feature extractor is not loaded. "
                "STEAD requires X3D features. Please ensure pytorchvideo is installed: "
                "pip install pytorchvideo"
            )
        
        # 원본 STEAD transform 파이프라인:
        # 1. UniformTemporalSubsample(13) - 이미 13프레임이므로 생략
        # 2. /255.0
        # 3. Permute(1, 0, 2, 3) - (T, C, H, W) -> (C, T, H, W)
        # 4. Normalize(mean, std)
        # 5. ShortSideScale(182) - 이미 전처리에서 적용
        # 6. CenterCrop(182, 182) - 이미 전처리에서 적용
        # 7. Permute(1, 0, 2, 3) - (C, T, H, W) -> (T, C, H, W)
        
        # (T, H, W, C) -> (T, C, H, W)
        clip_tensor = torch.from_numpy(clip).float().permute(0, 3, 1, 2)
        clip_tensor = clip_tensor / 255.0
        
        # 정규화 (ImageNet mean/std)
        mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1)
        clip_tensor = (clip_tensor - mean) / std
        
        # (T, C, H, W) -> (C, T, H, W) - X3D 입력 형태
        clip_tensor = clip_tensor.permute(1, 0, 2, 3)
        clip_tensor = clip_tensor.unsqueeze(0).to(self._device)  # (1, C, T, H, W)
        
        with torch.no_grad():
            features = self.x3d_model(clip_tensor)
            # 출력 형태 조정
            # 원본: (1, 192, 13, 7, 7) 형태 기대
            if features.dim() == 2:
                # Flatten된 경우: (1, 192*13*7*7) -> (1, 192, 13, 7, 7)
                features = features.view(1, 192, self.clip_length, 7, 7)
            elif features.dim() == 3:
                # (1, 192, T*H*W) 형태인 경우
                features = features.view(1, 192, self.clip_length, 7, 7)
            elif features.dim() == 4:
                # (1, 192, T, H*W) 형태인 경우
                features = features.view(1, 192, self.clip_length, 7, 7)
            # 이미 5D인 경우 그대로 사용
        
        return features
    
    def reset(self) -> None:
        """상태 초기화"""
        self._frame_buffer.clear()
        self._frames_since_inference = 0
    
    @property
    def name(self) -> str:
        return "STEAD"
    
    @property
    def required_frames(self) -> int:
        return self.clip_length



