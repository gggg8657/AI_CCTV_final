"""
이벤트 API 라우터
================

이벤트 조회, 확인 처리
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter()


@router.get("/")
async def list_events(
    camera_id: Optional[int] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    vlm_type: Optional[str] = Query(None),
    min_score: Optional[float] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """이벤트 목록 조회"""
    # TODO: 구현 예정
    return {
        "events": [],
        "total": 0,
        "limit": limit,
        "offset": offset
    }


@router.get("/{event_id}")
async def get_event(event_id: int):
    """이벤트 상세 조회"""
    # TODO: 구현 예정
    return {
        "message": f"Event {event_id} detail - 구현 예정"
    }


@router.post("/{event_id}/ack")
async def acknowledge_event(event_id: int):
    """이벤트 확인"""
    # TODO: 구현 예정
    return {
        "message": f"Event {event_id} acknowledged - 구현 예정"
    }
