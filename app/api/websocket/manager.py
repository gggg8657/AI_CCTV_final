"""
WebSocket ConnectionManager — 채널 기반 연결 관리
===================================================

채널별로 WebSocket 클라이언트를 관리하고,
프레임/이벤트/통계 등을 브로드캐스트.
"""

import asyncio
import logging
from collections import defaultdict
from typing import Dict, Set, Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """채널 기반 WebSocket 연결 관리자"""

    def __init__(self) -> None:
        self._channels: Dict[str, Set[WebSocket]] = defaultdict(set)

    async def connect(self, websocket: WebSocket, channel: str) -> None:
        await websocket.accept()
        self._channels[channel].add(websocket)
        logger.info("WS connected: channel=%s (total=%d)", channel, len(self._channels[channel]))

    def disconnect(self, websocket: WebSocket, channel: str) -> None:
        self._channels[channel].discard(websocket)
        if not self._channels[channel]:
            del self._channels[channel]

    async def broadcast(self, channel: str, message: Dict[str, Any]) -> None:
        dead: list = []
        for ws in self._channels.get(channel, set()):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._channels[channel].discard(ws)

    async def broadcast_all(self, message: Dict[str, Any]) -> None:
        for channel in list(self._channels.keys()):
            await self.broadcast(channel, message)

    @property
    def channel_counts(self) -> Dict[str, int]:
        return {ch: len(conns) for ch, conns in self._channels.items()}

    @property
    def total_connections(self) -> int:
        return sum(len(c) for c in self._channels.values())


ws_manager = ConnectionManager()
