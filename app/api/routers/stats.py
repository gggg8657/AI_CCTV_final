"""
통계 API 라우터
==============

통계 조회, 트렌드 분석
"""

from fastapi import APIRouter, Query
from typing import Optional
from datetime import date
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter()


@router.get("/")
async def get_stats(
    camera_id: Optional[int] = Query(None),
    date: Optional[date] = Query(None),
    period: str = Query("day", regex="^(day|week|month)$")
):
    """통계 조회"""
    # TODO: 구현 예정
    return {
        "message": "Statistics endpoint - 구현 예정"
    }


@router.get("/trends")
async def get_trends(
    camera_id: Optional[int] = Query(None),
    days: int = Query(7, ge=1, le=30)
):
    """통계 트렌드"""
    # TODO: 구현 예정
    return {
        "message": "Trends endpoint - 구현 예정"
    }
