"""
로깅 유틸리티
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict


def setup_logger(
    name: str = "sci_v2",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    로거 설정
    
    Args:
        name: 로거 이름
        log_file: 로그 파일 경로 (None이면 콘솔만)
        level: 로그 레벨
    
    Returns:
        설정된 로거
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (옵션)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "sci_v2") -> logging.Logger:
    """기존 로거 가져오기"""
    return logging.getLogger(name)


@dataclass
class AnomalyEvent:
    """이상 감지 이벤트"""
    id: str
    timestamp: str
    frame_number: int
    vad_score: float
    threshold: float
    
    # VLM 분석 결과
    vlm_type: str = "Unknown"
    vlm_description: str = ""
    vlm_confidence: float = 0.0
    
    # Agent 대응 결과
    agent_actions: List[Dict] = None
    agent_response_time: float = 0.0
    
    # 클립 정보
    clip_path: str = ""
    
    # 메타데이터
    metadata: Dict = None
    
    def __post_init__(self):
        if self.agent_actions is None:
            self.agent_actions = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        return asdict(self)


class EventLogger:
    """이벤트 로깅 시스템"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 로그 파일 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.event_log_path = self.log_dir / f"events_{timestamp}.json"
        self.system_log_path = self.log_dir / f"system_{timestamp}.log"
        
        # 이벤트 목록
        self.events: List[AnomalyEvent] = []
        
        # 시스템 로거
        self.logger = setup_logger("event_logger", str(self.system_log_path))
    
    def log_event(self, event: AnomalyEvent):
        """이상 이벤트 로깅"""
        self.events.append(event)
        self._save_events()
        self.logger.warning(
            f"[ANOMALY] ID={event.id} Score={event.vad_score:.3f} "
            f"Type={event.vlm_type} Actions={len(event.agent_actions)}"
        )
    
    def log_info(self, message: str):
        self.logger.info(message)
    
    def log_warning(self, message: str):
        self.logger.warning(message)
    
    def log_error(self, message: str):
        self.logger.error(message)
    
    def _save_events(self):
        """이벤트 파일 저장"""
        events_data = [event.to_dict() for event in self.events]
        
        with open(self.event_log_path, 'w', encoding='utf-8') as f:
            json.dump(events_data, f, indent=2, ensure_ascii=False)
    
    def get_recent_events(self, n: int = 10) -> List[AnomalyEvent]:
        """최근 이벤트 조회"""
        return self.events[-n:] if self.events else []



