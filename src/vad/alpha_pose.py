"""
Pose 추출 래퍼 (MediaPipe Pose 사용)
===================================

실시간 pose keypoints 추출 및 feature 벡터 변환
원본 리포지토리 형식과 호환되는 pose feature 생성
"""

import warnings
from typing import Optional, List, Tuple, Dict
import numpy as np
import cv2

# MediaPipe Pose import 시도
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    warnings.warn("MediaPipe not installed. Install with: pip install mediapipe")


class AlphaPoseWrapper:
    """
    Pose 추출 래퍼 (MediaPipe Pose 사용)
    
    실시간으로 프레임에서 사람의 pose keypoints를 추출하고,
    원본 리포지토리 형식의 feature 벡터로 변환합니다.
    """
    
    # MediaPipe Pose keypoint 인덱스 (33개 keypoints)
    # 원본 리포지토리는 17개 keypoints를 사용하지만, MediaPipe는 33개를 제공
    # 주요 keypoints만 선택하여 feature 벡터 생성
    KEYPOINT_INDICES = [
        0,   # nose
        2,   # left_eye
        5,   # right_eye
        7,   # left_ear
        8,   # right_ear
        11,  # left_shoulder
        12,  # right_shoulder
        13,  # left_elbow
        14,  # right_elbow
        15,  # left_wrist
        16,  # right_wrist
        23,  # left_hip
        24,  # right_hip
        25,  # left_knee
        26,  # right_knee
        27,  # left_ankle
        28,  # right_ankle
    ]  # 총 17개 keypoints (원본 리포지토리와 동일)
    
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Pose 추출기 초기화
        
        Args:
            model_complexity: MediaPipe 모델 복잡도 (0, 1, 2)
            min_detection_confidence: 최소 감지 신뢰도
            min_tracking_confidence: 최소 추적 신뢰도
        """
        if not HAS_MEDIAPIPE:
            raise RuntimeError("MediaPipe가 설치되지 않았습니다. pip install mediapipe")
        
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # MediaPipe Pose 초기화
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        self._initialized = True
    
    def extract_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        프레임에서 pose keypoints 추출
        
        Args:
            frame: RGB 프레임 (H, W, 3), 0-255 범위
        
        Returns:
            Keypoints 배열 (N, 17, 3) - N명의 사람, 17개 keypoints, (x, y, confidence)
            또는 None (사람이 없는 경우)
        """
        if not self._initialized:
            return None
        
        try:
            # MediaPipe는 RGB 입력을 기대
            results = self.pose.process(frame)
            
            if not results.pose_landmarks:
                return None
            
            # 모든 사람의 keypoints 추출
            keypoints_list = []
            
            # MediaPipe는 한 프레임에 한 명의 사람만 감지
            # 여러 사람을 감지하려면 별도 처리 필요
            if results.pose_landmarks:
                keypoints = []
                for idx in self.KEYPOINT_INDICES:
                    if idx < len(results.pose_landmarks.landmark):
                        landmark = results.pose_landmarks.landmark[idx]
                        # (x, y, visibility) - visibility를 confidence로 사용
                        keypoints.append([
                            landmark.x,
                            landmark.y,
                            landmark.visibility
                        ])
                    else:
                        # keypoint가 없으면 0으로 채움
                        keypoints.append([0.0, 0.0, 0.0])
                
                keypoints_list.append(keypoints)
            
            if len(keypoints_list) == 0:
                return None
            
            # numpy array로 변환: (N, 17, 3)
            keypoints_array = np.array(keypoints_list, dtype=np.float32)
            
            return keypoints_array
            
        except Exception as e:
            warnings.warn(f"[AlphaPose] Keypoint extraction failed: {e}")
            return None
    
    def extract_pose_features(
        self,
        frame: np.ndarray,
        normalize: bool = True
    ) -> Optional[np.ndarray]:
        """
        Pose keypoints를 feature 벡터로 변환
        
        원본 리포지토리 형식과 호환되는 feature 벡터 생성
        
        Args:
            frame: RGB 프레임 (H, W, 3)
            normalize: 좌표를 정규화할지 여부 (프레임 크기로 나눔)
        
        Returns:
            Pose feature 벡터 (N, D) - N명의 사람, D차원 feature
            또는 None (사람이 없는 경우)
        """
        keypoints = self.extract_keypoints(frame)
        
        if keypoints is None:
            return None
        
        # Keypoints를 feature 벡터로 변환
        # 원본 리포지토리는 각 keypoint의 (x, y) 좌표를 사용
        # 또는 상대 좌표 (keypoint 간 거리, 각도 등)를 사용할 수 있음
        
        features_list = []
        H, W = frame.shape[:2]
        
        for person_keypoints in keypoints:  # (17, 3)
            # 방법 1: 모든 keypoint 좌표를 평탄화 (17 * 2 = 34차원)
            # 방법 2: 상대 좌표 사용 (더 robust)
            
            # 방법 1 사용 (간단하고 원본과 유사)
            if normalize:
                # 정규화된 좌표 사용 (이미 MediaPipe가 0-1 범위로 제공)
                feature = person_keypoints[:, :2].flatten()  # (34,)
            else:
                # 픽셀 좌표 사용
                feature = person_keypoints[:, :2].copy()
                feature[:, 0] *= W  # x 좌표
                feature[:, 1] *= H  # y 좌표
                feature = feature.flatten()  # (34,)
            
            features_list.append(feature)
        
        if len(features_list) == 0:
            return None
        
        # numpy array로 변환: (N, 34)
        features_array = np.array(features_list, dtype=np.float32)
        
        return features_array
    
    def is_initialized(self) -> bool:
        """초기화 여부 확인"""
        return self._initialized
    
    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'pose') and self.pose is not None:
            self.pose.close()
            self.pose = None




