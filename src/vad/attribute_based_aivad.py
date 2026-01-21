"""
Attribute-based VAD 모델 (AiVAD 활용)
=====================================

논문: "An Attribute-based Method for Video Anomaly Detection" (Reiss & Hoshen, TMLR 2025)
GitHub: https://github.com/talreiss/Accurate-Interpretable-VAD

AiVAD의 feature extraction 기능을 활용하고, 리포지토리의 density estimation 방식 사용.
"""

import os
import sys
from pathlib import Path
from collections import deque
from typing import Optional, List, Dict, Any
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

# 리포지토리 모듈 import
try:
    from feature_extraction import extract_velocity
    from sklearn.mixture import GaussianMixture
    import faiss
    HAS_FAISS = True
except ImportError as e:
    warnings.warn(f"Failed to import attribute VAD modules: {e}")
    HAS_FAISS = False
    faiss = None

# AiVAD import
try:
    from anomalib.models.video.ai_vad.torch_model import AiVadModel
    from anomalib.models.video.ai_vad.flow import FlowExtractor
    from anomalib.models.video.ai_vad.features import VideoRegionFeatureExtractor
    from anomalib.models.video.ai_vad.regions import RegionExtractor
    HAS_AIVAD = True
except ImportError as e:
    warnings.warn(f"Failed to import AiVAD modules: {e}")
    HAS_AIVAD = False
    AiVadModel = None

from .base import VADModel


class AttributeBasedVADAiVAD(VADModel):
    """
    Attribute-based VAD 모델 (AiVAD 활용)
    
    AiVAD의 feature extraction 기능을 활용하고,
    리포지토리의 density estimation 방식을 사용
    """
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        dataset_name: str = "custom",
        velocity_bins: int = 8,
        gmm_components: int = 5,
        k_nn: int = 1,
    ):
        super().__init__()
        
        if not HAS_AIVAD:
            raise ImportError("AiVAD modules not available. Install anomalib.")
        
        self.model_dir = Path(model_dir) if model_dir else PROJECT_ROOT / "experiments" / "attribute_vad"
        self.dataset_name = dataset_name
        self.velocity_bins = velocity_bins
        self.gmm_components = gmm_components
        self.k_nn = k_nn
        
        # AiVAD 컴포넌트
        self.flow_extractor = FlowExtractor()
        self.region_extractor = RegionExtractor(
            box_score_thresh=0.5,
            persons_only=False,
            min_bbox_area=100,
            max_bbox_overlap=0.65,
            enable_foreground_detections=True,
        )
        self.feature_extractor = VideoRegionFeatureExtractor(
            n_velocity_bins=velocity_bins,
            use_velocity_features=True,
            use_pose_features=True,
            use_deep_features=True,
        )
        
        # Density estimators (리포지토리 방식)
        self.velocity_density_estimator = None
        self.pose_index = None
        self.deep_index = None
        
        # Training features
        self.train_pose = None
        self.train_deep_features = None
        
        # Calibration parameters
        self.calibration_params = {}
        
        # 프레임 버퍼 (최소 2프레임 필요)
        self._frame_buffer: deque = deque(maxlen=2)
        
    def initialize(self, device: str = 'cuda:0') -> None:
        """모델 초기화"""
        self._device = device
        device_obj = torch.device(device)
        
        # AiVAD 컴포넌트를 device로 이동
        self.flow_extractor.eval().to(device_obj)
        self.region_extractor.eval().to(device_obj)
        self.feature_extractor.eval().to(device_obj)
        
        # Density estimators 로드
        self._load_density_estimators()
        
        # Calibration parameters 로드
        self._load_calibration_params()
        
        self._initialized = True
        print(f"[AttributeVAD-AiVAD] 초기화 완료 (device: {device})")
    
    def _load_density_estimators(self):
        """Density estimators 로드 (리포지토리 방식)"""
        model_dir = self.model_dir / self.dataset_name
        
        # Velocity GMM
        velocity_gmm_path = model_dir / "velocity_gmm.pkl"
        if velocity_gmm_path.exists():
            import pickle
            with open(velocity_gmm_path, 'rb') as f:
                self.velocity_density_estimator = pickle.load(f)
            print(f"[AttributeVAD-AiVAD] Velocity GMM loaded")
        else:
            print(f"[AttributeVAD-AiVAD] Warning: Velocity GMM not found")
        
        # Pose k-NN index (CPU 사용 - GPU CUBLAS 호환성 문제)
        pose_exemplars_path = model_dir / "pose_exemplars.npy"
        if pose_exemplars_path.exists() and HAS_FAISS:
            self.train_pose = np.load(pose_exemplars_path)
            # GPU index는 CUBLAS 오류 발생 가능하므로 CPU 사용
            self.pose_index = faiss.IndexFlatL2(self.train_pose.shape[1])
            self.pose_index.add(self.train_pose.astype(np.float32))
            print(f"[AttributeVAD-AiVAD] Pose index loaded (CPU): {len(self.train_pose)} exemplars")
        elif pose_exemplars_path.exists() and not HAS_FAISS:
            warnings.warn("faiss not available, pose index will not be loaded")
        
        # Deep features k-NN index (CPU 사용 - GPU CUBLAS 호환성 문제)
        deep_exemplars_path = model_dir / "deep_exemplars.npy"
        if deep_exemplars_path.exists() and HAS_FAISS:
            self.train_deep_features = np.load(deep_exemplars_path)
            # GPU index는 CUBLAS 오류 발생 가능하므로 CPU 사용
            self.deep_index = faiss.IndexFlatL2(self.train_deep_features.shape[1])
            self.deep_index.add(self.train_deep_features.astype(np.float32))
            print(f"[AttributeVAD-AiVAD] Deep features index loaded (CPU): {len(self.train_deep_features)} exemplars")
        elif deep_exemplars_path.exists() and not HAS_FAISS:
            warnings.warn("faiss not available, deep features index will not be loaded")
    
    def _load_calibration_params(self):
        """Calibration parameters 로드"""
        import json
        calibration_path = self.model_dir / self.dataset_name / "calibration_params.json"
        
        if calibration_path.exists():
            with open(calibration_path, 'r') as f:
                self.calibration_params = json.load(f)
            print(f"[AttributeVAD-AiVAD] Calibration parameters loaded")
        else:
            self.calibration_params = {
                "velocity": {"min": 0.0, "max": 1.0},
                "pose": {"min": 0.0, "max": 1.0},
                "deep": {"min": 0.0, "max": 1.0}
            }
    
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
        
        # 프레임 버퍼에 추가
        self._frame_buffer.append(frame)
        
        # 최소 2프레임 필요
        if len(self._frame_buffer) < 2:
            return None
        
        # 프레임을 tensor로 변환
        first_frame = self._preprocess_frame(self._frame_buffer[0])
        last_frame = self._preprocess_frame(self._frame_buffer[1])
        
        # AiVAD 방식으로 features 추출
        with torch.no_grad():
            # 1. Optical flow 추출 (AiVAD의 FlowExtractor 사용)
            flows = self.flow_extractor(
                first_frame.unsqueeze(0).to(self._device),
                last_frame.unsqueeze(0).to(self._device)
            )
            
            # 2. Region 추출 (AiVAD의 RegionExtractor 사용)
            regions = self.region_extractor(
                first_frame.unsqueeze(0).to(self._device),
                last_frame.unsqueeze(0).to(self._device)
            )
            
            # 3. Features 추출 (AiVAD의 VideoRegionFeatureExtractor 사용)
            features_per_batch = self.feature_extractor(
                last_frame.unsqueeze(0).to(self._device),
                flows,
                regions
            )
        
        # 리포지토리 방식으로 score 계산
        if len(features_per_batch) == 0:
            return 0.0
        
        features = features_per_batch[0]  # 첫 번째 이미지의 features
        
        # 각 feature type별 score 계산
        velocity_score = 0.0
        pose_score = 0.0
        deep_score = 0.0
        
        if "velocity" in features and self.velocity_density_estimator is not None:
            velocity_features = features["velocity"].cpu().numpy()
            
            # Shape 정규화 및 검증
            if velocity_features.size > 0:
                # 1D인 경우 2D로 변환 (GMM은 (n_samples, n_features) 형태 기대)
                if velocity_features.ndim == 1:
                    velocity_features = velocity_features.reshape(1, -1)
                elif velocity_features.ndim == 0:
                    velocity_features = velocity_features.reshape(1, 1)
                
                # Feature dimension 검증
                expected_dim = self.velocity_density_estimator.n_features_in_
                if velocity_features.shape[1] == expected_dim:
                    try:
                        # 리포지토리 방식: GMM negative log-likelihood
                        scores = -self.velocity_density_estimator.score_samples(velocity_features)
                        velocity_score = float(np.max(scores))
                    except Exception as e:
                        print(f"[Warning] Velocity GMM scoring failed: {e}")
                        velocity_score = 0.0
                else:
                    print(f"[Warning] Velocity feature dim mismatch: expected {expected_dim}, got {velocity_features.shape[1]}")
                    velocity_score = 0.0
            else:
                velocity_score = 0.0
        else:
            # velocity가 features에 없거나 GMM이 없는 경우
            velocity_score = 0.0
        
        if "pose" in features and self.pose_index is not None:
            pose_features = features["pose"].cpu().numpy().astype(np.float32)
            if len(pose_features) > 0:
                try:
                    D, I = self.pose_index.search(pose_features, self.k_nn)
                    pose_score = float(np.mean(D))
                except Exception as e:
                    print(f"[Warning] Pose search failed: {e}, skipping pose score")
                    pose_score = 0.0
        
        if "deep" in features and self.deep_index is not None:
            deep_features = features["deep"].cpu().numpy().astype(np.float32)
            if len(deep_features) > 0:
                try:
                    D, I = self.deep_index.search(deep_features, self.k_nn)
                    deep_score = float(np.mean(D))
                except Exception as e:
                    print(f"[Warning] Deep search failed: {e}, skipping deep score")
                    deep_score = 0.0
        
        # Score calibration
        velocity_score = self._calibrate_score(velocity_score, "velocity")
        pose_score = self._calibrate_score(pose_score, "pose")
        deep_score = self._calibrate_score(deep_score, "deep")
        
        # Feature 조합에 따라 선택적으로 사용
        # (ablation study를 위해 추가)
        if hasattr(self, '_use_velocity') and not self._use_velocity:
            velocity_score = 0.0
        if hasattr(self, '_use_pose') and not self._use_pose:
            pose_score = 0.0
        if hasattr(self, '_use_deep') and not self._use_deep:
            deep_score = 0.0
        
        # 최종 score (모든 features 합산)
        final_score = velocity_score + pose_score + deep_score
        
        return float(final_score)
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """프레임을 tensor로 변환"""
        # RGB -> Tensor [C, H, W]
        if isinstance(frame, np.ndarray):
            frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
        else:
            frame_tensor = frame
        
        return frame_tensor
    
    def _calibrate_score(self, score: float, feature_type: str) -> float:
        """Score calibration (min-max normalization)"""
        if feature_type not in self.calibration_params:
            return score
        
        params = self.calibration_params[feature_type]
        min_val = params.get("min", 0.0)
        max_val = params.get("max", 1.0)
        
        if max_val - min_val == 0:
            return score
        
        normalized = (score - min_val) / (max_val - min_val)
        return normalized
    
    def reset(self) -> None:
        """상태 초기화"""
        self._frame_buffer.clear()
    
    @property
    def name(self) -> str:
        """모델 이름"""
        return "attribute_based_aivad"
    
    @property
    def required_frames(self) -> int:
        """추론에 필요한 최소 프레임 수"""
        return 2

