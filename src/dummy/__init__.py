"""
Dummy 모듈 — 모델 파일 없이 전체 파이프라인을 실행하기 위한 목업
================================================================

실제 모델 경로를 설정하면 바로 교체 가능하도록 동일한 인터페이스를 구현.
"""

from .vad import DummyVADModel
from .vlm import DummyVLMAnalyzer, DummyVLMResult
from .agent import DummyAgentFlow
from .video import DummyVideoSource

__all__ = [
    "DummyVADModel",
    "DummyVLMAnalyzer",
    "DummyVLMResult",
    "DummyAgentFlow",
    "DummyVideoSource",
]
