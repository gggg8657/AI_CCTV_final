"""
WebSocket 스트리밍 라우터
=======================

실시간 프레임 및 이벤트 스트리밍
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter()


class ConnectionManager:
    """WebSocket 연결 관리자"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        """모든 연결된 클라이언트에 메시지 브로드캐스트"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


@router.websocket("/stream/{camera_id}")
async def websocket_stream(websocket: WebSocket, camera_id: int):
    """카메라 스트림 WebSocket"""
    await manager.connect(websocket)
    try:
        while True:
            # TODO: 실제 스트리밍 로직 구현
            data = await websocket.receive_text()
            # 클라이언트로부터 메시지 수신 처리
            await websocket.send_json({
                "type": "frame",
                "camera_id": camera_id,
                "message": "Streaming endpoint - 구현 예정"
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket)
