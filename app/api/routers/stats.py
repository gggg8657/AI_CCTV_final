"""
통계 API 라우터
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional
from datetime import date, timedelta

from src.database.db import get_db
from src.database.models import DailyStatistics, Event, User
from app.api.schemas import DailyStatsOut
from app.api.dependencies import get_current_user

router = APIRouter()


@router.get("/")
def get_stats(
    camera_id: Optional[int] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    q = db.query(DailyStatistics)
    if camera_id is not None:
        q = q.filter(DailyStatistics.camera_id == camera_id)
    if start_date:
        q = q.filter(DailyStatistics.date >= start_date)
    if end_date:
        q = q.filter(DailyStatistics.date <= end_date)

    items = q.order_by(DailyStatistics.date.desc()).all()
    return {
        "items": [DailyStatsOut.model_validate(s).model_dump() for s in items],
        "total": len(items),
    }


@router.get("/summary")
def get_summary(
    camera_id: Optional[int] = Query(None),
    days: int = Query(7, ge=1, le=365),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    cutoff = date.today() - timedelta(days=days)

    q = db.query(Event)
    if camera_id is not None:
        q = q.filter(Event.camera_id == camera_id)
    q = q.filter(Event.timestamp >= cutoff.isoformat())

    total_events = q.count()
    unacked = q.filter(Event.acknowledged == False).count()  # noqa: E712

    avg_score = db.query(func.avg(Event.vad_score)).filter(
        Event.timestamp >= cutoff.isoformat()
    ).scalar()

    vlm_types = (
        db.query(Event.vlm_type, func.count(Event.id))
        .filter(Event.timestamp >= cutoff.isoformat(), Event.vlm_type.isnot(None))
        .group_by(Event.vlm_type)
        .all()
    )

    return {
        "period_days": days,
        "total_events": total_events,
        "unacknowledged": unacked,
        "avg_vad_score": round(avg_score, 4) if avg_score else 0,
        "vlm_type_distribution": {t: c for t, c in vlm_types},
    }
