"""
경량 VLM 분석기
==============

경량 VLM 모델을 사용한 영상 분석기
- Qwen-VL-Chat-2B/4B
- LLaVA-1.5-7B (더 작은 버전)
- 기타 경량 옵션

기존 VLMAnalyzer와 동일한 인터페이스 제공
"""

import os
import time
import base64
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import cv2
import numpy as np

from .prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_SINGLE,
    USER_PROMPT_MULTI,
    FAST_SYSTEM_PROMPT,
    FAST_USER_PROMPT,
    ANOMALY_TYPE_KEYWORDS,
    ACTION_KEYWORDS,
)
from .analyzer import VLMAnalysisResult


# 경량 모델 경로 (기본값)
DEFAULT_LIGHTWEIGHT_MODEL_PATH = "/data/DJ/models/Qwen2-VL-2B-Instruct-q4_k_m.gguf"
DEFAULT_LIGHTWEIGHT_MMPROJ_PATH = "/data/DJ/models/Qwen2-VL-2B-Instruct-mmproj-f16.gguf"

# 대체 경량 모델 옵션
ALTERNATIVE_MODELS = {
    "qwen2_vl_2b": {
        "model": "/data/DJ/models/Qwen2-VL-2B-Instruct-q4_k_m.gguf",
        "mmproj": "/data/DJ/models/Qwen2-VL-2B-Instruct-mmproj-f16.gguf",
        "description": "Qwen2-VL-2B Instruct (Q4_K_M quantized)"
    },
    "qwen2_vl_4b": {
        "model": "/data/DJ/models/Qwen2-VL-4B-Instruct-q4_k_m.gguf",
        "mmproj": "/data/DJ/models/Qwen2-VL-4B-Instruct-mmproj-f16.gguf",
        "description": "Qwen2-VL-4B Instruct (Q4_K_M quantized)"
    },
    "llava_1.5_7b": {
        "model": "/data/DJ/models/llava-v1.5-7b-q4_k_m.gguf",
        "mmproj": "/data/DJ/models/llava-v1.5-7b-mmproj-f16.gguf",
        "description": "LLaVA-1.5-7B (Q4_K_M quantized)"
    },
}


class LightweightVLMAnalyzer:
    """
    경량 VLM 기반 영상 분석기
    
    Qwen2-VL-2B/4B 또는 다른 경량 모델을 사용하여 영상 분석 수행
    
    기능:
    - 기존 VLMAnalyzer와 동일한 인터페이스
    - 더 빠른 추론 속도 (예상: 7B 대비 2-3배 빠름)
    - 메모리 사용량 감소
    """
    
    def __init__(
        self,
        model_name: str = "qwen2_vl_2b",
        model_path: str = None,
        mmproj_path: str = None,
        n_frames: int = 8,
        use_multiframe: bool = True,
        optimize_speed: bool = True,  # 경량 모델은 기본적으로 최적화 모드
        n_ctx: int = 4096,  # 경량 모델은 더 작은 컨텍스트
        gpu_id: int = 2,
    ):
        self.model_name = model_name
        
        # 모델 경로 설정
        if model_path and mmproj_path:
            self.model_path = model_path
            self.mmproj_path = mmproj_path
        elif model_name in ALTERNATIVE_MODELS:
            config = ALTERNATIVE_MODELS[model_name]
            self.model_path = config["model"]
            self.mmproj_path = config["mmproj"]
        else:
            self.model_path = DEFAULT_LIGHTWEIGHT_MODEL_PATH
            self.mmproj_path = DEFAULT_LIGHTWEIGHT_MMPROJ_PATH
        
        self.n_frames = n_frames
        self.use_multiframe = use_multiframe
        self.optimize_speed = optimize_speed
        self.n_ctx = n_ctx
        self.gpu_id = gpu_id
        
        self.vlm = None
        self._initialized = False
        
        # 속도 최적화 설정 (경량 모델은 기본적으로 최적화)
        if optimize_speed:
            self.image_size = (320, 180)
            self.max_tokens = 64
            self.n_ctx = 2048
        else:
            self.image_size = (640, 360)
            self.max_tokens = 128  # 경량 모델은 더 짧은 응답
    
    def initialize(self) -> bool:
        """경량 VLM 모델 초기화"""
        if self._initialized:
            return True
        
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
            
            # 모델 파일 존재 확인
            if not os.path.exists(self.model_path):
                print(f"[LightweightVLM] 경고: 모델 파일을 찾을 수 없습니다: {self.model_path}")
                print(f"[LightweightVLM] 사용 가능한 모델:")
                for name, config in ALTERNATIVE_MODELS.items():
                    exists = os.path.exists(config["model"])
                    print(f"  - {name}: {config['model']} {'✓' if exists else '✗'}")
                return False
            
            if not os.path.exists(self.mmproj_path):
                print(f"[LightweightVLM] 경고: mmproj 파일을 찾을 수 없습니다: {self.mmproj_path}")
                return False
            
            mode = "FAST" if self.optimize_speed else "DETAILED"
            print(f"[LightweightVLM] Loading {self.model_name} model ({mode} mode)...")
            print(f"[LightweightVLM] Model: {self.model_path}")
            print(f"[LightweightVLM] MMProj: {self.mmproj_path}")
            
            # Qwen2-VL 모델인지 확인 (파일명 기반)
            if "qwen" in self.model_path.lower() or "qwen2" in self.model_path.lower():
                chat_handler = Qwen25VLChatHandler(clip_model_path=self.mmproj_path)
            else:
                # 다른 모델 타입 (LLaVA 등)은 별도 처리 필요
                print(f"[LightweightVLM] 경고: 알 수 없는 모델 타입. Qwen25VLChatHandler 사용 시도...")
                chat_handler = Qwen25VLChatHandler(clip_model_path=self.mmproj_path)
            
            self.vlm = Llama(
                model_path=self.model_path,
                chat_handler=chat_handler,
                n_gpu_layers=-1,
                n_ctx=self.n_ctx,
                main_gpu=self.gpu_id,
                verbose=False
            )
            
            self._initialized = True
            print(f"[LightweightVLM] Model loaded (n_ctx={self.n_ctx}, max_tokens={self.max_tokens})")
            return True
            
        except ImportError:
            print("[LightweightVLM] llama_cpp가 설치되지 않았습니다. pip install llama-cpp-python")
            return False
        except Exception as e:
            print(f"[LightweightVLM] 초기화 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def analyze(
        self,
        frames: List[np.ndarray] = None,
        video_path: str = None,
    ) -> VLMAnalysisResult:
        """
        경량 VLM 분석 수행
        
        Args:
            frames: 분석할 프레임 리스트 (RGB)
            video_path: 비디오 파일 경로 (frames가 없을 경우 사용)
        
        Returns:
            VLMAnalysisResult
        """
        start_time = time.time()
        
        # 프레임 준비
        if frames is None and video_path:
            frames = self._extract_frames_from_video(video_path)
        
        if not frames:
            return VLMAnalysisResult(
                detected_type="Error",
                description="No frames available",
                actions=[],
                confidence=0.0,
                response="",
                latency_ms=0.0,
                n_frames=0,
                success=False
            )
        
        # VLM 초기화 확인
        if not self._initialized:
            if not self.initialize():
                return VLMAnalysisResult(
                    detected_type="Error",
                    description="Lightweight VLM initialization failed",
                    actions=[],
                    confidence=0.0,
                    response="",
                    latency_ms=(time.time() - start_time) * 1000,
                    n_frames=len(frames),
                    success=False
                )
        
        # 이미지 준비
        jpeg_quality = 70 if self.optimize_speed else 85
        
        if self.use_multiframe and len(frames) > 1:
            grid = self._create_frame_grid(frames)
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), 
                                     [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if self.optimize_speed:
                user_prompt = FAST_USER_PROMPT
            else:
                user_prompt = USER_PROMPT_MULTI.format(n_frames=len(frames))
        else:
            frame = frames[len(frames) // 2] if len(frames) > 1 else frames[0]
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if self.optimize_speed:
                user_prompt = FAST_USER_PROMPT
            else:
                user_prompt = USER_PROMPT_SINGLE
        
        encoded = base64.b64encode(buffer).decode('utf-8')
        
        # 프롬프트 선택
        system_prompt = FAST_SYSTEM_PROMPT if self.optimize_speed else SYSTEM_PROMPT
        
        # VLM 호출
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                {"type": "text", "text": user_prompt}
            ]}
        ]
        
        try:
            response = self.vlm.create_chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=self.max_tokens
            )
            response_text = response['choices'][0]['message']['content']
            
            # 응답 파싱
            result = self._parse_response(response_text)
            result['latency_ms'] = (time.time() - start_time) * 1000
            result['n_frames'] = len(frames)
            result['success'] = True
            result['response'] = response_text
            
            return VLMAnalysisResult(**result)
            
        except Exception as e:
            return VLMAnalysisResult(
                detected_type="Error",
                description=str(e),
                actions=[],
                confidence=0.0,
                response="",
                latency_ms=(time.time() - start_time) * 1000,
                n_frames=len(frames),
                success=False
            )
    
    def _extract_frames_from_video(self, video_path: str) -> List[np.ndarray]:
        """비디오에서 프레임 추출"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return []
        
        # 균등 간격으로 프레임 추출
        n = min(self.n_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, n, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 리사이즈
                frame = cv2.resize(frame, self.image_size)
                frames.append(frame)
        
        cap.release()
        return frames
    
    def _create_frame_grid(self, frames: List[np.ndarray]) -> np.ndarray:
        """프레임 그리드 생성 (2x4 = 8 프레임)"""
        n_frames = len(frames)
        if n_frames == 0:
            return None
        
        # 그리드 크기 결정
        if n_frames <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 4
        
        h, w = frames[0].shape[:2]
        cell_h, cell_w = h // 2, w // 2
        
        grid = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)
        
        for i, frame in enumerate(frames[:rows * cols]):
            r, c = i // cols, i % cols
            resized = cv2.resize(frame, (cell_w, cell_h))
            grid[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = resized
        
        return grid
    
    def _parse_response(self, response: str) -> Dict:
        """VLM 응답 파싱 (기존 analyzer와 동일)"""
        # 유형 추출
        detected_type = "Normal"
        response_lower = response.lower()
        
        for keyword, anomaly_type in ANOMALY_TYPE_KEYWORDS.items():
            if keyword in response_lower:
                detected_type = anomaly_type
                break
        
        # 설명 추출
        description = ""
        if "설명:" in response:
            desc_start = response.find("설명:") + 3
            desc_end = response.find("\n", desc_start)
            if desc_end == -1:
                desc_end = len(response)
            description = response[desc_start:desc_end].strip()
        
        # 액션 추출
        actions = []
        for kw in ACTION_KEYWORDS:
            if kw in response.lower():
                actions.append(kw)
        
        # 신뢰도 (이상 상황 감지 시 0.8, 정상 시 0.5)
        confidence = 0.8 if detected_type != "Normal" else 0.5
        
        return {
            'detected_type': detected_type,
            'description': description,
            'actions': actions,
            'confidence': confidence,
        }
    
    def unload(self):
        """경량 VLM 모델 언로드 (메모리 해제)"""
        if self.vlm is not None:
            del self.vlm
            self.vlm = None
            self._initialized = False
            import gc
            gc.collect()
            print("[LightweightVLM] Model unloaded")
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
