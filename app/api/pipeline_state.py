"""
파이프라인 글로벌 상태 — MultiCameraManager 싱글턴
==================================================

FastAPI lifespan에서 초기화되며, 의존성 주입으로 라우터에 전달.
"""

import asyncio
import logging
import os
from typing import Optional

from src.pipeline.multi_camera_manager import MultiCameraManager
from src.database.event_logger import AsyncEventLogger

logger = logging.getLogger(__name__)

_manager: Optional[MultiCameraManager] = None
_event_logger: Optional[AsyncEventLogger] = None
_loop: Optional[asyncio.AbstractEventLoop] = None


def init_pipeline() -> MultiCameraManager:
    """앱 시작 시 1회 호출"""
    global _manager, _event_logger, _loop
    _loop = asyncio.get_event_loop()

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

    _event_logger = AsyncEventLogger()
    _event_logger.start()

    logger.info(
        "Pipeline initialised: dummy=%s, flags=%s",
        use_dummy, _manager.resource_pool.dummy_flags,
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


def _on_anomaly(event_data: dict) -> None:
    """파이프라인 스레드에서 호출 — DB 저장 + WebSocket 브로드캐스트"""
    if _event_logger:
        _event_logger.log(event_data)

    if _loop and _loop.is_running():
        asyncio.run_coroutine_threadsafe(_ws_broadcast_anomaly(event_data), _loop)


async def _ws_broadcast_anomaly(event_data: dict) -> None:
    from app.api.websocket.manager import ws_manager
    await ws_manager.broadcast("events", {"type": "anomaly", **event_data})
