"""
파이프라인 글로벌 상태 — MultiCameraManager + NotificationEngine 싱글턴
========================================================================

FastAPI lifespan에서 초기화되며, 의존성 주입으로 라우터에 전달.
"""

import asyncio
import logging
import os
from typing import Dict, Optional

from src.pipeline.multi_camera_manager import MultiCameraManager
from src.database.event_logger import AsyncEventLogger
from src.notifications.engine import NotificationEngine
from src.notifications.console import ConsoleChannel
from src.notifications.webhook import WebhookChannel
from src.notifications.email import EmailChannel

logger = logging.getLogger(__name__)

_manager: Optional[MultiCameraManager] = None
_event_logger: Optional[AsyncEventLogger] = None
_notifier: Optional[NotificationEngine] = None
_loop: Optional[asyncio.AbstractEventLoop] = None


def init_pipeline(loop: Optional[asyncio.AbstractEventLoop] = None) -> MultiCameraManager:
    """앱 시작 시 1회 호출. loop은 async context에서 get_running_loop()으로 전달."""
    global _manager, _event_logger, _notifier, _loop
    _loop = loop or asyncio.get_event_loop()

    use_dummy = os.getenv("PIPELINE_DUMMY", "true").lower() in ("1", "true", "yes")
    use_dummy_vad = os.getenv("PIPELINE_DUMMY_VAD")
    use_dummy_vlm = os.getenv("PIPELINE_DUMMY_VLM")
    use_dummy_agent = os.getenv("PIPELINE_DUMMY_AGENT")

    def _parse_bool(val: Optional[str]) -> Optional[bool]:
        if val is None:
            return None
        return val.lower() in ("1", "true", "yes")

    _manager = MultiCameraManager(
        max_cameras=int(os.getenv("MAX_CAMERAS", "16")),
        gpu_id=int(os.getenv("GPU_ID", "0")),
        use_dummy=use_dummy,
        use_dummy_vad=_parse_bool(use_dummy_vad),
        use_dummy_vlm=_parse_bool(use_dummy_vlm),
        use_dummy_agent=_parse_bool(use_dummy_agent),
    )
    _manager.set_anomaly_callback(_on_anomaly)
    _manager.set_frame_callback(_on_frame)

    _event_logger = AsyncEventLogger()
    _event_logger.start()

    _notifier = NotificationEngine(
        cooldown_seconds=float(os.getenv("NOTIFY_COOLDOWN", "15")),
    )
    _notifier.add_channel(ConsoleChannel())

    webhook_url = os.getenv("NOTIFY_WEBHOOK_URL", "")
    if webhook_url:
        _notifier.add_channel(WebhookChannel(url=webhook_url))
        logger.info("Webhook notification channel enabled")

    smtp_host = os.getenv("SMTP_HOST", "")
    if smtp_host:
        _notifier.add_channel(EmailChannel())
        logger.info("Email notification channel enabled")

    logger.info(
        "Pipeline initialised: dummy=%s, flags=%s, notify_channels=%s",
        use_dummy, _manager.resource_pool.dummy_flags, _notifier.channels,
    )
    return _manager


def shutdown_pipeline() -> None:
    global _manager, _event_logger
    if _event_logger:
        _event_logger.stop()
    if _manager:
        _manager.shutdown()
    logger.info("Pipeline shut down")


def get_pipeline_manager() -> MultiCameraManager:
    if _manager is None:
        raise RuntimeError("Pipeline not initialised — call init_pipeline() first")
    return _manager


def get_event_logger() -> AsyncEventLogger:
    if _event_logger is None:
        raise RuntimeError("EventLogger not initialised")
    return _event_logger


def get_notifier() -> NotificationEngine:
    if _notifier is None:
        raise RuntimeError("NotificationEngine not initialised")
    return _notifier


def _on_anomaly(event_data: dict) -> None:
    """파이프라인 스레드에서 호출 — DB 저장 + WebSocket + 알림"""
    if _event_logger:
        _event_logger.log(event_data)

    if _notifier:
        _notifier.notify(event_data)

    if _loop and _loop.is_running():
        asyncio.run_coroutine_threadsafe(_ws_broadcast_anomaly(event_data), _loop)


async def _ws_broadcast_anomaly(event_data: dict) -> None:
    from app.api.websocket.manager import ws_manager
    await ws_manager.broadcast("events", {"type": "anomaly", **event_data})


_frame_log_counter: Dict[int, int] = {}


def _on_frame(camera_id: int, frame, vad_score: float, b64_jpeg: str = None) -> None:
    """파이프라인 스레드에서 호출 — base64 프레임을 WebSocket으로 전송"""
    if not b64_jpeg:
        return
    if not _loop:
        logger.warning("_on_frame: event loop not captured yet")
        return
    if not _loop.is_running():
        logger.warning("_on_frame: event loop is not running")
        return

    cnt = _frame_log_counter.get(camera_id, 0) + 1
    _frame_log_counter[camera_id] = cnt
    if cnt <= 3 or cnt % 50 == 0:
        logger.info("_on_frame cam=%d: sending frame #%d (b64 len=%d)", camera_id, cnt, len(b64_jpeg))

    asyncio.run_coroutine_threadsafe(
        _ws_broadcast_frame(camera_id, vad_score, b64_jpeg), _loop
    )


async def _ws_broadcast_frame(camera_id: int, vad_score: float, b64_jpeg: str) -> None:
    from app.api.websocket.manager import ws_manager
    channel = f"camera:{camera_id}"
    conns = ws_manager.channel_counts.get(channel, 0)
    if conns == 0:
        return
    await ws_manager.broadcast(channel, {
        "type": "frame",
        "camera_id": camera_id,
        "vad_score": round(vad_score, 4),
        "jpeg": b64_jpeg,
    })
