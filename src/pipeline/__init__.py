"""
E2E 파이프라인
=============

VAD + VLM + Agent 통합 파이프라인

기능:
- 다양한 비디오 소스 지원 (파일, RTSP, 웹캠)
- 실시간 이상 탐지
- VLM 기반 상황 분석
- Agent 기반 자동 대응
- 이벤트 로깅 및 클립 저장
"""

from .engine import E2EEngine, EngineConfig
from .clip_saver import ClipSaver


__all__ = [
    'E2EEngine',
    'EngineConfig',
    'ClipSaver',
]



