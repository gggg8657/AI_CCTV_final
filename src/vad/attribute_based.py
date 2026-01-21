"""
Attribute-based VAD 모델
========================

논문: "An Attribute-based Method for Video Anomaly Detection" (Reiss & Hoshen, TMLR 2025)
GitHub: https://github.com/talreiss/Accurate-Interpretable-VAD

리포지토리 코드를 최대한 활용하여 구현.
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
    from models import CLIP
    from feature_extraction import extract_velocity
    from video_dataset import VideoDatasetWithFlows, normalize
    from sklearn.mixture import GaussianMixture
    import faiss
except ImportError as e:
    warnings.warn(f"Failed to import attribute VAD modules: {e}. Some features may not work.")

from .base import VADModel
from .flow_net2 import FlowNet2Wrapper
from .alpha_pose import AlphaPoseWrapper


class AttributeBasedVAD(VADModel):
    """
    Attribute-based VAD 모델
    
    Velocity, Pose, Deep (CLIP) features를 사용한 이상 탐지
    리포지토리 코드를 최대한 활용
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
        
        self.model_dir = Path(model_dir) if model_dir else PROJECT_ROOT / "experiments" / "attribute_vad"
        self.dataset_name = dataset_name
        self.velocity_bins = velocity_bins
        self.gmm_components = gmm_components
        self.k_nn = k_nn
        
        # 모델 컴포넌트
        self.clip_model = None
        self.velocity_density_estimator = None
        self.pose_index = None
        self.deep_index = None
        self.flownet2 = None  # FlowNet2 래퍼
        self.alpha_pose = None  # AlphaPose (MediaPipe) 래퍼
        
        # Calibration parameters
        self.calibration_params = {}
        
        # 프레임 버퍼 (velocity 계산을 위해 최소 2프레임 필요)
        self._frame_buffer: deque = deque(maxlen=2)
        self._flow_buffer: deque = deque(maxlen=2)
        
        # Training features (k-NN용)
        self.train_pose = None
        self.train_deep_features = None
        
        # FlowNet2 사용 여부 (기본값: True, 체크포인트가 없으면 False로 fallback)
        self.use_flownet2 = True
        # AlphaPose 사용 여부
        self.use_alpha_pose = True
        
    def initialize(self, device: str = 'cuda:0') -> None:
        """모델 초기화"""
        self._device = device
        device_obj = torch.device(device)
        
        # CLIP 모델 로드
        print("[AttributeVAD] Loading CLIP model...")
        self.clip_model = CLIP(device_obj)
        self.clip_model.eval()
        
        # FlowNet2 초기화
        try:
            self.flownet2 = FlowNet2Wrapper(device=device)
            if self.flownet2.initialize():
                print("[AttributeVAD] FlowNet2 initialized successfully")
                self.use_flownet2 = True
            else:
                print("[AttributeVAD] Warning: FlowNet2 initialization failed, using Farneback fallback")
                self.use_flownet2 = False
                self.flownet2 = None
        except Exception as e:
            print(f"[AttributeVAD] Warning: FlowNet2 not available ({e}), using Farneback fallback")
            self.use_flownet2 = False
            self.flownet2 = None
        
        # AlphaPose (MediaPipe) 초기화
        try:
            self.alpha_pose = AlphaPoseWrapper()
            print("[AttributeVAD] AlphaPose (MediaPipe) initialized successfully")
            self.use_alpha_pose = True
        except Exception as e:
            print(f"[AttributeVAD] Warning: AlphaPose not available ({e}), pose feature will be disabled")
            self.use_alpha_pose = False
            self.alpha_pose = None
        
        # Density estimators 로드
        self._load_density_estimators()
        
        # Calibration parameters 로드
        self._load_calibration_params()
        
        self._initialized = True
        print(f"[AttributeVAD] 초기화 완료 (device: {device})")
    
    def _load_density_estimators(self):
        """Density estimators 로드"""
        model_dir = self.model_dir / self.dataset_name
        
        # Velocity GMM
        velocity_gmm_path = model_dir / "velocity_gmm.pkl"
        if velocity_gmm_path.exists():
            import pickle
            with open(velocity_gmm_path, 'rb') as f:
                self.velocity_density_estimator = pickle.load(f)
            print(f"[AttributeVAD] Velocity GMM loaded from {velocity_gmm_path}")
        else:
            print(f"[AttributeVAD] Warning: Velocity GMM not found at {velocity_gmm_path}")
        
        # Pose k-NN index
        pose_exemplars_path = model_dir / "pose_exemplars.npy"
        if pose_exemplars_path.exists():
            self.train_pose = np.load(pose_exemplars_path)
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(self.train_pose.shape[1])
            self.pose_index = faiss.index_cpu_to_gpu(res, 0, index)
            self.pose_index.add(self.train_pose.astype(np.float32))
            print(f"[AttributeVAD] Pose index loaded: {len(self.train_pose)} exemplars")
        else:
            print(f"[AttributeVAD] Warning: Pose exemplars not found at {pose_exemplars_path}")
        
        # Deep features k-NN index
        deep_exemplars_path = model_dir / "deep_exemplars.npy"
        if deep_exemplars_path.exists():
            self.train_deep_features = np.load(deep_exemplars_path)
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(self.train_deep_features.shape[1])
            self.deep_index = faiss.index_cpu_to_gpu(res, 0, index)
            self.deep_index.add(self.train_deep_features.astype(np.float32))
            print(f"[AttributeVAD] Deep features index loaded: {len(self.train_deep_features)} exemplars")
        else:
            print(f"[AttributeVAD] Warning: Deep exemplars not found at {deep_exemplars_path}")
    
    def _load_calibration_params(self):
        """Calibration parameters 로드"""
        import json
        calibration_path = self.model_dir / self.dataset_name / "calibration_params.json"
        
        if calibration_path.exists():
            with open(calibration_path, 'r') as f:
                self.calibration_params = json.load(f)
            print(f"[AttributeVAD] Calibration parameters loaded")
        else:
            print(f"[AttributeVAD] Warning: Calibration parameters not found. Using defaults.")
            # 기본값 설정
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
        
        # Velocity 계산을 위해 최소 2프레임 필요
        if len(self._frame_buffer) < 2:
            return None
        
        # Optical flow 계산 (FlowNet2 사용, 실패 시 Farneback fallback)
        flow = self._compute_optical_flow()
        if flow is None:
            return None
        
        # Features 추출
        velocity_score = self._extract_velocity_score(flow)
        pose_score = self._extract_pose_score(frame)
        deep_score = self._extract_deep_score(frame)
        
        # Score calibration
        velocity_score = self._calibrate_score(velocity_score, "velocity")
        pose_score = self._calibrate_score(pose_score, "pose")
        deep_score = self._calibrate_score(deep_score, "deep")
        
        # 최종 score (모든 features 합산)
        final_score = velocity_score + pose_score + deep_score
        
        return float(final_score)
    
    def _compute_optical_flow(self) -> Optional[np.ndarray]:
        """
        Optical flow 계산
        
        FlowNet2를 우선 사용하고, 실패하거나 사용 불가능한 경우 Farneback fallback 사용
        """
        if len(self._frame_buffer) < 2:
            return None
        
        frame1 = self._frame_buffer[0]
        frame2 = self._frame_buffer[1]
        
        # FlowNet2 사용 시도
        if self.use_flownet2 and self.flownet2 is not None:
            try:
                flow = self.flownet2.compute_flow(frame1, frame2)
                if flow is not None:
                    return flow
            except Exception as e:
                print(f"[AttributeVAD] FlowNet2 computation failed: {e}, falling back to Farneback")
        
        # Farneback fallback (간단한 optical flow)
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        return flow
    
    def _extract_velocity_score(self, flow: np.ndarray) -> float:
        """Velocity feature 추출 및 score 계산"""
        if self.velocity_density_estimator is None:
            return 0.0
        
        # 리포지토리의 extract_velocity 함수 사용
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        orientation = np.arctan2(flow[..., 1], flow[..., 0])
        
        velocity_feature = extract_velocity(
            flow, magnitude, orientation, 
            orientations=self.velocity_bins, 
            motion_threshold=0.0
        )
        
        # GMM으로 density estimation
        score = -self.velocity_density_estimator.score_samples(velocity_feature.reshape(1, -1))[0]
        
        return float(score)
    
    def _extract_pose_score(self, frame: np.ndarray) -> float:
        """
        Pose feature 추출 및 score 계산
        
        AlphaPose (MediaPipe)로 keypoints 추출 후 k-NN search로 anomaly score 계산
        """
        if self.pose_index is None:
            return 0.0
        
        # AlphaPose로 pose feature 추출
        if not self.use_alpha_pose or self.alpha_pose is None:
            return 0.0
        
        try:
            # Pose features 추출
            pose_features = self.alpha_pose.extract_pose_features(frame, normalize=True)
            
            if pose_features is None or len(pose_features) == 0:
                # 사람이 없는 경우
                return 0.0
            
            # 여러 사람이 있는 경우, 가장 높은 score 사용
            max_score = 0.0
            
            for person_features in pose_features:
                # Feature 차원 확인 및 조정
                # 원본 리포지토리의 pose feature 차원과 맞춰야 함
                if person_features.shape[0] != self.train_pose.shape[1]:
                    # 차원이 다르면 조정 (padding 또는 truncation)
                    target_dim = self.train_pose.shape[1]
                    if person_features.shape[0] < target_dim:
                        # Padding
                        padding = np.zeros(target_dim - person_features.shape[0], dtype=np.float32)
                        person_features = np.concatenate([person_features, padding])
                    else:
                        # Truncation
                        person_features = person_features[:target_dim]
                
                # k-NN search
                person_features = person_features.reshape(1, -1).astype(np.float32)
                D, I = self.pose_index.search(person_features, self.k_nn)
                score = float(np.mean(D, axis=1)[0])
                
                max_score = max(max_score, score)
            
            return max_score
            
        except Exception as e:
            warnings.warn(f"[AttributeVAD] Pose score extraction failed: {e}")
            return 0.0
    
    def _extract_deep_score(self, frame: np.ndarray) -> float:
        """Deep (CLIP) feature 추출 및 score 계산"""
        if self.deep_index is None or self.clip_model is None:
            return 0.0
        
        # 프레임 전처리
        frame_tensor = self._preprocess_frame_for_clip(frame)
        
        # CLIP feature 추출
        with torch.no_grad():
            features = self.clip_model(frame_tensor.unsqueeze(0).to(self._device))
            features_np = features.cpu().numpy().astype(np.float32)
        
        # k-NN search
        D, I = self.deep_index.search(features_np, self.k_nn)
        score = np.mean(D, axis=1)[0]
        
        return float(score)
    
    def _preprocess_frame_for_clip(self, frame: np.ndarray) -> torch.Tensor:
        """CLIP을 위한 프레임 전처리"""
        # Resize to 224x224
        frame_resized = cv2.resize(frame, (224, 224))
        
        # RGB -> Tensor [C, H, W]
        frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1) / 255.0
        
        # Normalize (CLIP normalization)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
        frame_tensor = (frame_tensor - mean[:, None, None]) / std[:, None, None]
        
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
        self._flow_buffer.clear()
    
    @property
    def name(self) -> str:
        """모델 이름"""
        return "attribute_based"
    
    @property
    def required_frames(self) -> int:
        """추론에 필요한 최소 프레임 수"""
        return 2  # Velocity 계산을 위해 최소 2프레임 필요

