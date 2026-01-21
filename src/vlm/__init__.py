"""
VLM (Vision Language Model) 분석기
==================================

Qwen2.5-VL-7B 기반 영상 분석

기능:
- 단일 프레임 분석
- 멀티프레임 그리드 분석
- 이상 상황 분류 및 설명 생성
"""

from .analyzer import VLMAnalyzer, VLMAnalysisResult
from .prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_SINGLE,
    USER_PROMPT_MULTI,
    FAST_SYSTEM_PROMPT,
    FAST_USER_PROMPT,
)


__all__ = [
    'VLMAnalyzer',
    'VLMAnalysisResult',
    'SYSTEM_PROMPT',
    'USER_PROMPT_SINGLE',
    'USER_PROMPT_MULTI',
    'FAST_SYSTEM_PROMPT',
    'FAST_USER_PROMPT',
]



