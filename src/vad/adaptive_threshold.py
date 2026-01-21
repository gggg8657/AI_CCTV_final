#!/usr/bin/env python3
"""
Adaptive Threshold Calibration
================================

장면 특성 및 이력 성능 기반 threshold 자동 조정
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging


class AdaptiveThreshold:
    """
    적응형 Threshold 관리자
    
    VAD 모델별로 최적 threshold를 동적으로 조정
    """
    
    def __init__(
        self,
        model_name: str,
        initial_threshold: float = 0.5,
        min_threshold: float = 0.0,
        max_threshold: float = 1.0,
        history_size: int = 100
    ):
        """
        Args:
            model_name: VAD 모델 이름
            initial_threshold: 초기 threshold 값
            min_threshold: 최소 threshold 값
            max_threshold: 최대 threshold 값
            history_size: 이력 데이터 크기
        """
        self.model_name = model_name
        self.current_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        
        # 이력 데이터
        self.score_history = deque(maxlen=history_size)
        self.label_history = deque(maxlen=history_size)  # True: anomaly, False: normal
        self.scene_history = deque(maxlen=history_size)  # 장면 특성
        
        # 통계
        self.stats = {
            "total_updates": 0,
            "precision_history": [],
            "recall_history": [],
            "f1_history": []
        }
    
    def update(self, score: float, is_anomaly: bool, scene_features: Dict = None):
        """
        Threshold 업데이트를 위한 데이터 수집
        
        Args:
            score: VAD anomaly score
            is_anomaly: 실제 이상 여부 (ground truth)
            scene_features: 장면 특성 (선택적)
        """
        self.score_history.append(score)
        self.label_history.append(is_anomaly)
        if scene_features:
            self.scene_features = scene_features
    
    def optimize_threshold(self, method: str = "f1_maximize") -> float:
        """
        Threshold 최적화
        
        Args:
            method: 최적화 방법 ("f1_maximize", "precision_recall_balance", "roc_optimal")
        
        Returns:
            최적화된 threshold 값
        """
        if len(self.score_history) < 10:
            return self.current_threshold  # 데이터 부족 시 현재 값 유지
        
        scores = np.array(self.score_history)
        labels = np.array(self.label_history)
        
        if method == "f1_maximize":
            # F1-score 최대화
            best_threshold = self.current_threshold
            best_f1 = 0.0
            
            # Threshold 후보 생성
            threshold_candidates = np.linspace(
                scores.min(), scores.max(), 100
            )
            
            for threshold in threshold_candidates:
                predictions = scores >= threshold
                tp = np.sum((predictions == True) & (labels == True))
                fp = np.sum((predictions == True) & (labels == False))
                fn = np.sum((predictions == False) & (labels == True))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            self.current_threshold = np.clip(best_threshold, self.min_threshold, self.max_threshold)
            self.stats["total_updates"] += 1
            self.stats["f1_history"].append(best_f1)
            
            logging.info(f"[AdaptiveThreshold] {self.model_name}: threshold={self.current_threshold:.6f}, F1={best_f1:.4f}")
            
        elif method == "precision_recall_balance":
            # Precision-Recall 균형
            best_threshold = self.current_threshold
            best_balance = float('inf')
            
            threshold_candidates = np.linspace(scores.min(), scores.max(), 100)
            
            for threshold in threshold_candidates:
                predictions = scores >= threshold
                tp = np.sum((predictions == True) & (labels == True))
                fp = np.sum((predictions == True) & (labels == False))
                fn = np.sum((predictions == False) & (labels == True))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                # Precision과 Recall의 차이 최소화
                balance = abs(precision - recall)
                if balance < best_balance:
                    best_balance = balance
                    best_threshold = threshold
            
            self.current_threshold = np.clip(best_threshold, self.min_threshold, self.max_threshold)
            self.stats["total_updates"] += 1
        
        return self.current_threshold
    
    def get_threshold(self) -> float:
        """현재 threshold 반환"""
        return self.current_threshold
    
    def get_stats(self) -> Dict:
        """통계 정보 반환"""
        return {
            "model_name": self.model_name,
            "current_threshold": self.current_threshold,
            "total_updates": self.stats["total_updates"],
            "history_size": len(self.score_history),
            "avg_f1": np.mean(self.stats["f1_history"]) if self.stats["f1_history"] else 0.0
        }


class AdaptiveThresholdManager:
    """
    여러 VAD 모델의 Adaptive Threshold를 관리
    """
    
    def __init__(self):
        self.thresholds: Dict[str, AdaptiveThreshold] = {}
    
    def get_threshold(self, model_name: str, default: float = 0.5) -> float:
        """
        모델별 threshold 반환
        
        Args:
            model_name: VAD 모델 이름
            default: 기본 threshold 값
        
        Returns:
            threshold 값
        """
        if model_name not in self.thresholds:
            # 모델별 초기 threshold 설정
            initial_thresholds = {
                "stead": 0.5,
                "stae": 0.003,
                "mnad": 0.5,
                "memae": 0.5,
                "attribute_based_aivad": 0.5
            }
            initial = initial_thresholds.get(model_name, default)
            self.thresholds[model_name] = AdaptiveThreshold(
                model_name=model_name,
                initial_threshold=initial
            )
        
        return self.thresholds[model_name].get_threshold()
    
    def update(self, model_name: str, score: float, is_anomaly: bool, scene_features: Dict = None):
        """Threshold 업데이트 데이터 추가"""
        if model_name not in self.thresholds:
            self.get_threshold(model_name)  # 초기화
        
        self.thresholds[model_name].update(score, is_anomaly, scene_features)
    
    def optimize_all(self, method: str = "f1_maximize"):
        """모든 모델의 threshold 최적화"""
        for model_name, threshold_obj in self.thresholds.items():
            threshold_obj.optimize_threshold(method)
    
    def get_all_stats(self) -> Dict:
        """모든 모델의 통계 반환"""
        return {
            model_name: threshold_obj.get_stats()
            for model_name, threshold_obj in self.thresholds.items()
        }
