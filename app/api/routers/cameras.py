"""
카메라 관리 API 라우터
======================

DB CRUD + 실제 파이프라인 start/stop 통합
"""

from fastapi import APIRouter, HTTPException, status, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from src.database.db import get_db
from src.database.models import Camera, User
from src.pipeline.camera_config import CameraConfig
from app.api.schemas import CameraCreate, CameraUpdate, CameraOut
from app.api.dependencies import get_current_user
from app.api.pipeline_state import get_pipeline_manager

router = APIRouter()


def _db_cam_to_pipeline_config(cam: Camera) -> CameraConfig:
    return CameraConfig(
        camera_id=cam.id,
        source_type=cam.source_type,
        source_path=cam.source_path,
        location=cam.location,
        vad_model=cam.vad_model,
        vad_threshold=cam.vad_threshold,
        enable_vlm=cam.enable_vlm,
        enable_agent=cam.enable_agent,
        agent_flow=cam.agent_flow,
        gpu_id=cam.gpu_id,
    )


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

    try:
        mgr = get_pipeline_manager()
        mgr.add_camera(_db_cam_to_pipeline_config(cam))
    except Exception:
        pass

    return cam


@router.put("/{camera_id}", response_model=CameraOut)
def update_camera(camera_id: int, body: CameraUpdate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    was_active = cam.status == "active"

    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(cam, field, value)
    db.commit()
    db.refresh(cam)

    try:
        mgr = get_pipeline_manager()
        if was_active:
            mgr.stop_camera(camera_id)
            mgr.remove_camera(camera_id)
            mgr.add_camera(_db_cam_to_pipeline_config(cam))
            mgr.start_camera(camera_id)
        elif camera_id in mgr.camera_ids:
            mgr.remove_camera(camera_id)
            mgr.add_camera(_db_cam_to_pipeline_config(cam))
    except Exception:
        pass

    return cam


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(camera_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    try:
        mgr = get_pipeline_manager()
        mgr.stop_camera(camera_id)
        mgr.remove_camera(camera_id)
    except Exception:
        pass

    db.delete(cam)
    db.commit()


@router.post("/{camera_id}/start")
def start_camera(camera_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    mgr = get_pipeline_manager()

    if camera_id not in mgr.camera_ids:
        mgr.add_camera(_db_cam_to_pipeline_config(cam))

    ok = mgr.start_camera(camera_id)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to start pipeline")

    cam.status = "active"
    db.commit()
    return {"camera_id": camera_id, "status": "active"}


@router.post("/{camera_id}/stop")
def stop_camera(camera_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    mgr = get_pipeline_manager()
    mgr.stop_camera(camera_id)

    cam.status = "inactive"
    db.commit()
    return {"camera_id": camera_id, "status": "inactive"}


@router.get("/{camera_id}/pipeline-status")
def get_pipeline_status(camera_id: int, user: User = Depends(get_current_user)):
    mgr = get_pipeline_manager()
    cam_status = mgr.get_camera_status(camera_id)
    if cam_status is None:
        raise HTTPException(status_code=404, detail="Camera not registered in pipeline")
    return cam_status.to_dict()


@router.get("/pipeline/overview")
def pipeline_overview(user: User = Depends(get_current_user)):
    mgr = get_pipeline_manager()
    return {
        "active_cameras": mgr.active_count,
        "total_cameras": len(mgr.camera_ids),
        "camera_ids": mgr.camera_ids,
        "all_statuses": mgr.get_all_statuses(),
        "model_info": mgr.resource_pool.loaded_models,
        "gpu_memory": mgr.resource_pool.gpu_memory_info(),
        "dummy_flags": mgr.resource_pool.dummy_flags,
    }
