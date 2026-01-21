"""
적응형 VLM 분석기
================

상황에 따라 경량/대형 VLM 모델을 자동으로 선택하는 시스템

- 경량 모델 (Qwen2-VL-2B/4B): 일반 시나리오, 빠른 응답 필요
- 대형 모델 (Qwen2.5-VL-7B): 복잡한 상황, 높은 정확도 필요
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

import cv2
import numpy as np

from .analyzer import VLMAnalyzer, VLMAnalysisResult
from .analyzer_lightweight import LightweightVLMAnalyzer


@dataclass
class AdaptiveVLMConfig:
    """적응형 VLM 설정"""
    # 경량 모델 설정
    lightweight_model_path: str = "/data/DJ/models/Qwen2-VL-2B-Instruct-q4_k_m.gguf"
    lightweight_mmproj_path: str = "/data/DJ/models/Qwen2-VL-2B-Instruct-mmproj-f16.gguf"
    
    # 대형 모델 설정
    full_model_path: str = "/data/DJ/models/Qwen2.5-VL-7B-Instruct-q4_k_m.gguf"
    full_mmproj_path: str = "/data/DJ/models/Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf"
    
    # 선택 기준
    complexity_threshold: float = 0.7  # 이상 점수가 이 값 이상이면 대형 모델 사용
    confidence_threshold: float = 0.5  # 경량 모델 신뢰도가 이 값 미만이면 대형 모델 사용
    use_lightweight_first: bool = True  # 먼저 경량 모델 시도


class AdaptiveVLMAnalyzer:
    """적응형 VLM 분석기"""
    
    def __init__(self, config: Optional[AdaptiveVLMConfig] = None):
        self.config = config or AdaptiveVLMConfig()
        self.lightweight_analyzer: Optional[LightweightVLMAnalyzer] = None
        self.full_analyzer: Optional[VLMAnalyzer] = None
        self._lightweight_loaded = False
        self._full_loaded = False
        
        # 통계
        self.stats = {
            'lightweight_used': 0,
            'full_used': 0,
            'total_analyses': 0,
            'avg_lightweight_time': 0.0,
            'avg_full_time': 0.0,
        }
    
    def initialize(self) -> bool:
        """VLM 분석기 초기화"""
        # 경량 모델 초기화
        if self.config.use_lightweight_first:
            try:
                self.lightweight_analyzer = LightweightVLMAnalyzer(
                    model_path=self.config.lightweight_model_path,
                    mmproj_path=self.config.lightweight_mmproj_path
                )
                if self.lightweight_analyzer.initialize():
                    self._lightweight_loaded = True
                    print("[AdaptiveVLM] 경량 모델 로드 완료")
                else:
                    print("[AdaptiveVLM] 경량 모델 로드 실패")
            except Exception as e:
                print(f"[AdaptiveVLM] 경량 모델 초기화 오류: {e}")
        
        # 대형 모델 초기화
        try:
            self.full_analyzer = VLMAnalyzer()
            if self.full_analyzer.initialize():
                self._full_loaded = True
                print("[AdaptiveVLM] 대형 모델 로드 완료")
            else:
                print("[AdaptiveVLM] 대형 모델 로드 실패")
        except Exception as e:
            print(f"[AdaptiveVLM] 대형 모델 초기화 오류: {e}")
        
        return self._lightweight_loaded or self._full_loaded
    
    def analyze(
        self,
        frames: List[np.ndarray] = None,
        video_path: str = None,
        vad_score: float = 0.0,
        previous_confidence: float = 1.0,
    ) -> VLMAnalysisResult:
        """
        적응형 VLM 분석
        
        Args:
            frames: 분석할 프레임 리스트
            video_path: 비디오 파일 경로
            vad_score: VAD 이상 점수 (복잡도 판단용)
            previous_confidence: 이전 분석의 신뢰도
        
        Returns:
            VLMAnalysisResult
        """
        self.stats['total_analyses'] += 1
        
        # 모델 선택 로직
        use_full = False
        
        # 1. 복잡도 기반 선택
        if vad_score >= self.config.complexity_threshold:
            use_full = True
        
        # 2. 신뢰도 기반 선택
        if previous_confidence < self.config.confidence_threshold:
            use_full = True
        
        # 3. 경량 모델이 없으면 대형 모델 사용
        # 적응형 모델 선택 로직
        selected_model = "lightweight"  # 기본값
        
        # 1. Anomaly score 기반 선택
        if anomaly_score is not None:
            selected_model = self._select_model_by_anomaly_score(anomaly_score)
        
        # 2. 시스템 부하 기반 선택 (부하가 높으면 경량 모델 우선)
        system_load_model = self._select_model_by_system_load()
        if system_load_model == "lightweight":
            selected_model = "lightweight"
        
        # 3. 이전 confidence 기반 선택 (통계가 있으면)
        if self.stats['total_analyses'] > 0:
            avg_confidence = self.stats.get('avg_confidence', 1.0)
            confidence_model = self._select_model_by_confidence(avg_confidence)
            # Confidence가 낮으면 더 정확한 모델 사용
            if confidence_model == "full":
                selected_model = "full"
        
        if not self._lightweight_loaded:
            use_full = True
        
        # 분석 실행
        start_time = time.time()
        
        if use_full and self._full_loaded:
            # 대형 모델 사용
            result = self.full_analyzer.analyze(frames=frames, video_path=video_path)
            self.stats['full_used'] += 1
            elapsed = time.time() - start_time
            self.stats['avg_full_time'] = (
                (self.stats['avg_full_time'] * (self.stats['full_used'] - 1) + elapsed) 
                / self.stats['full_used']
            )
        elif self._lightweight_loaded:
            # 경량 모델 사용
            result = self.lightweight_analyzer.analyze(frames=frames, video_path=video_path)
            self.stats['lightweight_used'] += 1
            elapsed = time.time() - start_time
            self.stats['avg_lightweight_time'] = (
                (self.stats['avg_lightweight_time'] * (self.stats['lightweight_used'] - 1) + elapsed)
                / self.stats['lightweight_used']
            )
            
            # 신뢰도가 낮으면 대형 모델로 재분석
            if (result.confidence < self.config.confidence_threshold and 
                self._full_loaded and 
                vad_score >= self.config.complexity_threshold):
                print(f"[AdaptiveVLM] 경량 모델 신뢰도 낮음 ({result.confidence:.2f}), 대형 모델로 재분석")
                full_result = self.full_analyzer.analyze(frames=frames, video_path=video_path)
                if full_result.confidence > result.confidence:
                    result = full_result
                    self.stats['full_used'] += 1
        else:
            # 모델이 없으면 에러 결과 반환
            result = VLMAnalysisResult(
                detected_type="Error",
                description="No VLM model available",
                actions=[],
                confidence=0.0,
                response="",
                latency_ms=0.0,
                n_frames=len(frames) if frames else 0,
                success=False
            )
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 조회"""
        return {
            **self.stats,
            'lightweight_ratio': (
                self.stats['lightweight_used'] / self.stats['total_analyses']
                if self.stats['total_analyses'] > 0 else 0.0
            ),
            'full_ratio': (
                self.stats['full_used'] / self.stats['total_analyses']
                if self.stats['total_analyses'] > 0 else 0.0
            ),
        }
    
    def cleanup(self):
        """리소스 정리"""
        if self.lightweight_analyzer:
            del self.lightweight_analyzer
        if self.full_analyzer:
            del self.full_analyzer





