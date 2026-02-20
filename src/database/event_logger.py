"""
AsyncEventLogger — 비동기 배치 DB 저장 이벤트 로거
====================================================

메모리 버퍼에 이벤트를 모아 두었다가,
10개 또는 1초 간격으로 백그라운드 스레드에서 DB에 일괄 저장.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

from .db import SessionLocal
from .models import Event

logger = logging.getLogger(__name__)

_BATCH_SIZE = 10
_FLUSH_INTERVAL = 1.0


class AsyncEventLogger:
    """이벤트를 비동기로 DB에 배치 저장하는 로거"""

    def __init__(self) -> None:
        self._buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._flush_loop, daemon=True, name="async-event-logger")
        self._thread.start()
        logger.info("AsyncEventLogger started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._flush()
        logger.info("AsyncEventLogger stopped")

    def log(self, event_data: Dict[str, Any]) -> None:
        batch = None
        with self._lock:
            self._buffer.append(event_data)
            if len(self._buffer) >= _BATCH_SIZE:
                batch = self._buffer[:_BATCH_SIZE]
                self._buffer = self._buffer[_BATCH_SIZE:]
        if batch:
            self._save_batch(batch)

    # ── Internal ──

    def _flush_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=_FLUSH_INTERVAL)
            self._flush()

    def _flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            batch = list(self._buffer)
            self._buffer.clear()
        self._save_batch(batch)

    def _save_batch(self, batch: List[Dict[str, Any]]) -> None:
        if not batch:
            return
        db = SessionLocal()
        try:
            for data in batch:
                event = Event(
                    camera_id=data.get("camera_id"),
                    timestamp=data.get("timestamp", datetime.now()),
                    frame_number=data.get("frame_number"),
                    vad_score=data.get("vad_score", 0.0),
                    threshold=data.get("threshold"),
                    vlm_type=data.get("vlm_type"),
                    vlm_description=data.get("vlm_description"),
                    vlm_confidence=data.get("vlm_confidence"),
                    agent_response_time=data.get("agent_response_time"),
                    clip_path=data.get("clip_path"),
                )
                db.add(event)
            db.commit()
            logger.debug("AsyncEventLogger: saved %d events", len(batch))
        except Exception as exc:
            db.rollback()
            logger.error("AsyncEventLogger: batch save failed: %s", exc)
        finally:
            db.close()
