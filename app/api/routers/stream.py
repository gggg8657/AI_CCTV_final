"""
WebSocket 스트리밍 라우터
=======================

채널:
- stream/{camera_id}  : 프레임 + VAD 점수 스트리밍
- events              : 실시간 이상 이벤트 브로드캐스트
- stats               : 주기적 통계 브로드캐스트
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.websocket.manager import ws_manager

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/stream/{camera_id}")
async def websocket_camera_stream(websocket: WebSocket, camera_id: int):
    channel = f"camera:{camera_id}"
    await ws_manager.connect(websocket, channel)
    try:
        await websocket.send_json({"type": "connected", "camera_id": camera_id, "channel": channel})
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, channel)
    except Exception:
        ws_manager.disconnect(websocket, channel)


@router.websocket("/events")
async def websocket_events(websocket: WebSocket):
    channel = "events"
    await ws_manager.connect(websocket, channel)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, channel)


@router.websocket("/stats")
async def websocket_stats(websocket: WebSocket):
    channel = "stats"
    await ws_manager.connect(websocket, channel)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, channel)


# ── Helper functions for pushing data from pipeline threads ──

async def push_frame_update(camera_id: int, vad_score: float) -> None:
    await ws_manager.broadcast(f"camera:{camera_id}", {
        "type": "frame_update",
        "camera_id": camera_id,
        "vad_score": round(vad_score, 4),
    })


async def push_anomaly_event(event_data: dict) -> None:
    await ws_manager.broadcast("events", {
        "type": "anomaly",
        **event_data,
    })


async def push_stats_update(stats: dict) -> None:
    await ws_manager.broadcast("stats", {
        "type": "stats_update",
        **stats,
    })
