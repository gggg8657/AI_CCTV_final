"""
VLM 분석기
=========

Qwen2.5-VL-7B 기반 영상 분석기
llama.cpp를 사용한 로컬 추론

실제 VLM 모델로 추론합니다. 더미 없음.
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


# 기본 모델 경로
DEFAULT_MODEL_PATH = "/data/DJ/models/Qwen2.5-VL-7B-Instruct-q4_k_m.gguf"
DEFAULT_MMPROJ_PATH = "/data/DJ/models/Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf"


@dataclass
class VLMAnalysisResult:
    """VLM 분석 결과"""
    detected_type: str
    description: str
    actions: List[str]
    confidence: float
    response: str
    latency_ms: float
    n_frames: int
    success: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected_type": self.detected_type,
            "description": self.description,
            "actions": self.actions,
            "confidence": self.confidence,
            "response": self.response,
            "latency_ms": self.latency_ms,
            "n_frames": self.n_frames,
            "success": self.success,
        }


class VLMAnalyzer:
    """
    VLM 기반 영상 분석기
    
    Qwen2.5-VL-7B를 llama.cpp로 로드하여 영상 분석 수행
    
    기능:
    - 단일 프레임 분석
    - 멀티프레임 그리드 분석
    - 속도 최적화 모드 지원 (9.8x speedup)
    
    실제 VLM 모델로 추론합니다. 더미 없음.
    """
    
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        mmproj_path: str = DEFAULT_MMPROJ_PATH,
        n_frames: int = 8,
        use_multiframe: bool = True,
        optimize_speed: bool = False,
        n_ctx: int = 8192,
        gpu_id: int = 2,
    ):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.n_frames = n_frames
        self.use_multiframe = use_multiframe
        self.optimize_speed = optimize_speed
        self.n_ctx = n_ctx
        self.gpu_id = gpu_id
        
        self.vlm = None
        self._initialized = False
        
        # 속도 최적화 설정
        if optimize_speed:
            self.image_size = (320, 180)
            self.max_tokens = 64
            self.n_ctx = 2048
        else:
            self.image_size = (640, 360)
            self.max_tokens = 256
    
    def initialize(self) -> bool:
        """VLM 모델 초기화"""
        if self._initialized:
            return True
        
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"VLM 모델 파일을 찾을 수 없습니다: {self.model_path}")
            
            if not os.path.exists(self.mmproj_path):
                raise FileNotFoundError(f"VLM mmproj 파일을 찾을 수 없습니다: {self.mmproj_path}")
            
            mode = "FAST" if self.optimize_speed else "DETAILED"
            print(f"[VLM] Loading Qwen2.5-VL model ({mode} mode)...")
            
            # GPU 설정: VLM은 GPU 3번 사용 (거의 비어있음)
            # CUDA_VISIBLE_DEVICES=3으로 설정되어 있으면 논리적 GPU 0 = 실제 GPU 3번
            # llama.cpp는 논리적 GPU 번호를 사용하므로 main_gpu=0 사용
            print(f"[VLM] Using logical GPU 0 (actual GPU 3) for VLM model")
            
            try:
                # CUDA_VISIBLE_DEVICES=3이면 논리적 GPU 0 = 실제 GPU 3번
                self.vlm = Llama(
                    model_path=self.model_path,
                    chat_handler=Qwen25VLChatHandler(clip_model_path=self.mmproj_path),
                    n_gpu_layers=-1,
                    n_ctx=self.n_ctx,
                    main_gpu=0,  # 논리적 GPU 0 = 실제 GPU 3번
                    verbose=False
                )
            except (RuntimeError, ValueError) as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower() or "failed" in str(e).lower():
                    print(f"[VLM] GPU 3 메모리 부족 또는 로드 실패. n_gpu_layers를 줄여서 재시도...")
                    # GPU 레이어 수를 줄여서 재시도
                    self.vlm = Llama(
                        model_path=self.model_path,
                        chat_handler=Qwen25VLChatHandler(clip_model_path=self.mmproj_path),
                        n_gpu_layers=20,  # 일부 레이어만 GPU에 로드
                        n_ctx=self.n_ctx,
                        main_gpu=0,  # 논리적 GPU 0 = 실제 GPU 3번
                        verbose=False
                    )
                else:
                    raise
            
            self._initialized = True
            print(f"[VLM] Model loaded on GPU 3 (n_ctx={self.n_ctx}, max_tokens={self.max_tokens})")
            return True
            
        except ImportError as e:
            print(f"[VLM] llama_cpp가 설치되지 않았습니다: {e}")
            import traceback
            traceback.print_exc()
            return False
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg:
                print(f"[VLM] GPU 메모리 부족으로 초기화 실패: {e}")
                print(f"[VLM] n_gpu_layers를 줄여서 재시도...")
                for n_layers in [20, 10, 5]:
                    try:
                        print(f"[VLM] n_gpu_layers={n_layers}로 재시도...")
                        self.vlm = Llama(
                            model_path=self.model_path,
                            chat_handler=Qwen25VLChatHandler(clip_model_path=self.mmproj_path),
                            n_gpu_layers=n_layers,
                            n_ctx=self.n_ctx,
                            main_gpu=0,  # 논리적 GPU 0 = 실제 GPU 3번
                            verbose=False
                        )
                        self._initialized = True
                        print(f"[VLM] Model loaded on GPU 3 with {n_layers} layers (n_ctx={self.n_ctx}, max_tokens={self.max_tokens})")
                        return True
                    except Exception as e2:
                        print(f"[VLM] n_gpu_layers={n_layers} 재시도 실패: {e2}")
                        if n_layers == 5:
                            print(f"[VLM] 모든 재시도 실패")
                            import traceback
                            traceback.print_exc()
                            return False
                        continue
                return False
            else:
                print(f"[VLM] 초기화 실패: {e}")
                import traceback
                traceback.print_exc()
                return False
        except Exception as e:
            print(f"[VLM] 초기화 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def analyze(
        self,
        frames: List[np.ndarray] = None,
        video_path: str = None,
    ) -> VLMAnalysisResult:
        """
        VLM 분석 수행
        
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
                    description="VLM initialization failed",
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
        """VLM 응답 파싱"""
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
        """VLM 모델 언로드 (메모리 해제)"""
        if self.vlm is not None:
            del self.vlm
            self.vlm = None
            self._initialized = False
            import gc
            gc.collect()
            print("[VLM] Model unloaded")
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized



