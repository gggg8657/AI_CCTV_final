#!/usr/bin/env python3
"""
Quantized VLM Analyzer
======================

INT8/INT4 quantization을 적용한 VLM 분석기
llama.cpp의 quantization 기능 활용
"""
import os
import time
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

import cv2
import numpy as np

from .analyzer import VLMAnalyzer, VLMResult


class QuantizedVLMAnalyzer(VLMAnalyzer):
    """
    Quantization을 적용한 VLM Analyzer
    
    INT8 또는 INT4 quantization을 통해 모델 크기 및 inference time 감소
    """
    
    def __init__(
        self,
        model_path: str = None,
        mmproj_path: str = None,
        quantization_type: str = "INT8",  # "INT8" or "INT4"
        optimize_speed: bool = True
    ):
        """
        Args:
            model_path: VLM 모델 경로 (quantized 모델)
            mmproj_path: MMProj 경로
            quantization_type: Quantization 타입 ("INT8" or "INT4")
            optimize_speed: 속도 최적화 모드
        """
        super().__init__(model_path, mmproj_path, optimize_speed)
        self.quantization_type = quantization_type
        self._quantized_model_path = None
        
    def _quantize_model(self, input_model_path: str, output_model_path: str) -> bool:
        """
        모델을 quantize하는 헬퍼 함수
        
        Note: 실제 quantization은 llama.cpp의 quantize 도구를 사용해야 함
        이 함수는 quantization된 모델 경로를 설정하는 역할만 수행
        """
        if not os.path.exists(input_model_path):
            logging.error(f"Input model not found: {input_model_path}")
            return False
        
        # Quantization은 사전에 llama.cpp의 quantize 도구로 수행되어야 함
        # 예: ./llama-quantize model.gguf model_q8_0.gguf Q8_0
        # 여기서는 quantization된 모델이 이미 존재한다고 가정
        if os.path.exists(output_model_path):
            self._quantized_model_path = output_model_path
            logging.info(f"Using quantized model: {output_model_path}")
            return True
        else:
            logging.warning(f"Quantized model not found: {output_model_path}")
            logging.warning("Falling back to original model. Please quantize the model first.")
            return False
    
    def initialize(self) -> bool:
        """Quantized VLM 모델 초기화"""
        if self._initialized:
            return True
        
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
            
            # Quantization된 모델 경로 설정
            if self.quantization_type == "INT8":
                # INT8 quantized 모델 경로 (예: model_q8_0.gguf)
                quantized_path = self.model_path.replace(".gguf", "_q8_0.gguf")
            elif self.quantization_type == "INT4":
                # INT4 quantized 모델 경로 (예: model_q4_k_m.gguf)
                quantized_path = self.model_path.replace(".gguf", "_q4_k_m.gguf")
            else:
                quantized_path = self.model_path
            
            # Quantized 모델이 있으면 사용, 없으면 원본 모델 사용
            if self._quantize_model(self.model_path, quantized_path):
                actual_model_path = quantized_path
                logging.info(f"Using {self.quantization_type} quantized model")
            else:
                actual_model_path = self.model_path
                logging.warning("Using original (non-quantized) model")
            
            if not os.path.exists(actual_model_path):
                raise FileNotFoundError(f"VLM 모델 파일을 찾을 수 없습니다: {actual_model_path}")
            
            if not os.path.exists(self.mmproj_path):
                raise FileNotFoundError(f"VLM mmproj 파일을 찾을 수 없습니다: {self.mmproj_path}")
            
            mode = "FAST" if self.optimize_speed else "DETAILED"
            logging.info(f"[QuantizedVLM] Loading {self.quantization_type} quantized model ({mode} mode)...")
            
            # Quantization된 모델은 일반적으로 더 빠르게 로드되고 실행됨
            # n_gpu_layers는 quantization 타입에 따라 조정 가능
            n_gpu_layers = -1  # 모든 레이어를 GPU에 로드
            
            try:
                self.vlm = Llama(
                    model_path=actual_model_path,
                    chat_handler=Qwen25VLChatHandler(clip_model_path=self.mmproj_path),
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=self.n_ctx,
                    main_gpu=0,
                    verbose=False
                )
            except (RuntimeError, ValueError) as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    logging.warning(f"[QuantizedVLM] GPU 메모리 부족. n_gpu_layers를 줄여서 재시도...")
                    self.vlm = Llama(
                        model_path=actual_model_path,
                        chat_handler=Qwen25VLChatHandler(clip_model_path=self.mmproj_path),
                        n_gpu_layers=20,  # 레이어 수 감소
                        n_ctx=self.n_ctx,
                        main_gpu=0,
                        verbose=False
                    )
                else:
                    raise
            
            self._initialized = True
            logging.info(f"[QuantizedVLM] {self.quantization_type} quantized model loaded successfully")
            return True
            
        except ImportError:
            logging.error("[QuantizedVLM] llama_cpp가 설치되지 않았습니다.")
            return False
        except Exception as e:
            logging.error(f"[QuantizedVLM] 초기화 실패: {e}")
            return False
    
    def analyze(self, frames: List[np.ndarray] = None, video_path: str = None) -> VLMResult:
        """
        Quantized VLM으로 분석 수행
        
        Quantization으로 인해 약간의 정확도 손실이 있을 수 있지만,
        inference time은 크게 감소함
        """
        return super().analyze(frames, video_path)
