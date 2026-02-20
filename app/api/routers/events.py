"""
이벤트 API 라우터
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime

from src.database.db import get_db
from src.database.models import Event, User
from app.api.schemas import EventOut, EventAck
from app.api.dependencies import get_current_user

router = APIRouter()


@router.get("/")
def list_events(
    camera_id: Optional[int] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    vlm_type: Optional[str] = Query(None),
    min_score: Optional[float] = Query(None),
    acknowledged: Optional[bool] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    q = db.query(Event)
    if camera_id is not None:
        q = q.filter(Event.camera_id == camera_id)
    if start_date:
        q = q.filter(Event.timestamp >= start_date)
    if end_date:
        q = q.filter(Event.timestamp <= end_date)
    if vlm_type:
        q = q.filter(Event.vlm_type == vlm_type)
    if min_score is not None:
        q = q.filter(Event.vad_score >= min_score)
    if acknowledged is not None:
        q = q.filter(Event.acknowledged == acknowledged)

    total = q.count()
    items = q.order_by(Event.timestamp.desc()).offset(offset).limit(limit).all()

    return {
        "items": [EventOut.model_validate(e).model_dump() for e in items],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{event_id}", response_model=EventOut)
def get_event(event_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    ev = db.query(Event).filter(Event.id == event_id).first()
    if not ev:
        raise HTTPException(status_code=404, detail="Event not found")
    return ev


@router.post("/{event_id}/ack", response_model=EventOut)
def acknowledge_event(
    event_id: int,
    body: EventAck,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ev = db.query(Event).filter(Event.id == event_id).first()
    if not ev:
        raise HTTPException(status_code=404, detail="Event not found")

    ev.acknowledged = True
    ev.acknowledged_by = user.id
    ev.acknowledged_at = datetime.now()
    if body.note:
        ev.note = body.note
    db.commit()
    db.refresh(ev)
    return ev
