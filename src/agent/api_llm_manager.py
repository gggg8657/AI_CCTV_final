"""
API-based LLM Manager
======================

OpenAI-compatible API를 사용한 LLM 관리자
로컬 모델 대신 API를 통해 LLM을 사용할 수 있음
"""

import logging
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logging.warning("openai package not installed. Install with: pip install openai")


class APILLMManager:
    """
    API 기반 LLM 관리자
    
    OpenAI-compatible API (예: vLLM, Ollama, OpenAI)를 사용
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(APILLMManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config: Dict = None):
        if not self._initialized:
            if not HAS_OPENAI:
                raise ImportError("openai package is required for API mode. Install with: pip install openai")
            
            self.text_client = None
            self.vision_client = None
            self.text_model = None
            self.vision_model = None
            self.text_loaded = False
            self.vision_loaded = False
            self.config = config or {}
            APILLMManager._initialized = True
    
    def load_text_llm(self, gpu_id: int = None) -> bool:
        """Text LLM API 클라이언트 초기화"""
        if self.text_loaded and self.text_client is not None:
            return True
        
        try:
            api_config = self.config.get("llm", {}).get("api", {})
            base_url = api_config.get("base_url", "http://localhost:8000/v1")
            api_key = api_config.get("api_key", "EMPTY")
            model_name = api_config.get("text_model", "Qwen/Qwen3-8B")
            
            logging.info(f"Initializing Text LLM API client: {base_url}")
            
            self.text_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            self.text_model = model_name
            self.text_loaded = True
            logging.info(f"Text LLM API client ready: {model_name}")
            return True
        except Exception as e:
            logging.error(f"Text LLM API client initialization failed: {e}")
            return False
    
    def load_vision_llm(self, gpu_id: int = None) -> bool:
        """Vision LLM API 클라이언트 초기화"""
        if self.vision_loaded and self.vision_client is not None:
            return True
        
        try:
            api_config = self.config.get("llm", {}).get("api", {})
            base_url = api_config.get("base_url", "http://localhost:8000/v1")
            api_key = api_config.get("api_key", "EMPTY")
            model_name = api_config.get("vision_model", "Qwen/Qwen2.5-VL-7B")
            
            logging.info(f"Initializing Vision LLM API client: {base_url}")
            
            self.vision_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            self.vision_model = model_name
            self.vision_loaded = True
            logging.info(f"Vision LLM API client ready: {model_name}")
            return True
        except Exception as e:
            logging.error(f"Vision LLM API client initialization failed: {e}")
            return False
    
    def load_all_models(self, gpu_id: int = None) -> bool:
        """모든 모델 로드"""
        text_success = self.load_text_llm(gpu_id)
        vision_success = self.load_vision_llm(gpu_id)
        return text_success and vision_success
    
    @property
    def text_llm(self):
        """Text LLM 클라이언트 반환 (호환성을 위해)"""
        if not self.text_loaded:
            self.load_text_llm()
        return self.text_client
    
    @property
    def vision_llm(self):
        """Vision LLM 클라이언트 반환 (호환성을 위해)"""
        if not self.vision_loaded:
            self.load_vision_llm()
        return self.vision_client
    
    def create_chat_completion(self, messages: List[Dict], model: str = None, **kwargs) -> Dict:
        """
        OpenAI-compatible chat completion
        
        Args:
            messages: 대화 메시지 리스트
            model: 모델 이름 (None이면 text_model 사용)
            **kwargs: 기타 파라미터 (tools, tool_choice, temperature 등)
        
        Returns:
            OpenAI-compatible 응답
        """
        if not self.text_loaded:
            if not self.load_text_llm():
                raise RuntimeError("Text LLM API client not initialized")
        
        model_name = model or self.text_model
        
        try:
            response = self.text_client.chat.completions.create(
                model=model_name,
                messages=messages,
                **kwargs
            )
            # OpenAI SDK 응답을 dict로 변환
            return {
                "choices": [{
                    "message": {
                        "role": response.choices[0].message.role,
                        "content": response.choices[0].message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in (response.choices[0].message.tool_calls or [])
                        ] if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls else None
                    }
                }],
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }
            }
        except Exception as e:
            logging.error(f"API call failed: {e}")
            raise
