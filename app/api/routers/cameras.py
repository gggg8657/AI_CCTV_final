"""
카메라 관리 API 라우터
"""

from fastapi import APIRouter, HTTPException, status, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from src.database.db import get_db
from src.database.models import Camera, User
from app.api.schemas import CameraCreate, CameraUpdate, CameraOut
from app.api.dependencies import get_current_user

router = APIRouter()


@router.get("/", response_model=List[CameraOut])
def list_cameras(
    status_filter: Optional[str] = Query(None, alias="status"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    q = db.query(Camera)
    if status_filter:
        q = q.filter(Camera.status == status_filter)
    return q.order_by(Camera.id).all()


@router.get("/{camera_id}", response_model=CameraOut)
def get_camera(camera_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return cam


@router.post("/", status_code=status.HTTP_201_CREATED, response_model=CameraOut)
def create_camera(body: CameraCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    cam = Camera(**body.model_dump(), created_by=user.id)
    db.add(cam)
    db.commit()
    db.refresh(cam)
    return cam


@router.put("/{camera_id}", response_model=CameraOut)
def update_camera(camera_id: int, body: CameraUpdate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(cam, field, value)
    db.commit()
    db.refresh(cam)
    return cam


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(camera_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    db.delete(cam)
    db.commit()


@router.post("/{camera_id}/start")
def start_camera(camera_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    cam.status = "active"
    db.commit()
    return {"camera_id": camera_id, "status": "active"}


@router.post("/{camera_id}/stop")
def stop_camera(camera_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    cam.status = "inactive"
    db.commit()
    return {"camera_id": camera_id, "status": "inactive"}
