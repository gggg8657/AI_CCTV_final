"""
알림 관리 API 라우터
===================

- 알림 규칙 CRUD
- 알림 채널 상태 조회
- 수동 테스트 발송
"""

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, ConfigDict
from datetime import datetime

from src.database.db import get_db
from src.database.models import NotificationRule, User
from app.api.dependencies import get_current_user, require_role
from app.api.pipeline_state import get_notifier

router = APIRouter()


# ── Schemas ──

class NotificationRuleCreate(BaseModel):
    camera_id: Optional[int] = None
    vlm_type: Optional[str] = None
    min_score: Optional[float] = None
    channels: List[str] = ["console"]
    webhook_url: Optional[str] = None
    enabled: bool = True


class NotificationRuleUpdate(BaseModel):
    vlm_type: Optional[str] = None
    min_score: Optional[float] = None
    channels: Optional[List[str]] = None
    webhook_url: Optional[str] = None
    enabled: Optional[bool] = None


class NotificationRuleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    camera_id: Optional[int]
    user_id: Optional[int]
    vlm_type: Optional[str]
    min_score: Optional[float]
    channels: List[str]
    webhook_url: Optional[str]
    enabled: bool
    created_at: datetime


# ── Rules CRUD ──

@router.get("/rules", response_model=List[NotificationRuleOut])
def list_rules(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return db.query(NotificationRule).all()


@router.post("/rules", status_code=status.HTTP_201_CREATED, response_model=NotificationRuleOut)
def create_rule(
    body: NotificationRuleCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_role("admin", "operator")),
):
    rule = NotificationRule(
        camera_id=body.camera_id,
        user_id=user.id,
        vlm_type=body.vlm_type,
        min_score=body.min_score,
        channels=body.channels,
        webhook_url=body.webhook_url,
        enabled=body.enabled,
    )
    db.add(rule)
    db.commit()
    db.refresh(rule)
    return rule


@router.put("/rules/{rule_id}", response_model=NotificationRuleOut)
def update_rule(
    rule_id: int,
    body: NotificationRuleUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(require_role("admin", "operator")),
):
    rule = db.query(NotificationRule).filter(NotificationRule.id == rule_id).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(rule, field, value)
    db.commit()
    db.refresh(rule)
    return rule


@router.delete("/rules/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_rule(
    rule_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(require_role("admin")),
):
    rule = db.query(NotificationRule).filter(NotificationRule.id == rule_id).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    db.delete(rule)
    db.commit()


# ── Engine Status ──

@router.get("/status")
def notification_status(user: User = Depends(get_current_user)):
    notifier = get_notifier()
    return {
        "channels": notifier.channels,
        "stats": notifier.stats,
    }


# ── Test ──

@router.post("/test")
def test_notification(user: User = Depends(require_role("admin"))):
    """수동 테스트 알림 발송"""
    notifier = get_notifier()
    from datetime import datetime
    ok = notifier.notify({
        "camera_id": 0,
        "vlm_type": "Fighting",
        "vlm_description": "Test alert from API",
        "vad_score": 0.99,
        "timestamp": datetime.now().isoformat(),
    }, camera_name="Test Camera", location="API Test")
    return {"sent": ok, "channels": notifier.channels}
