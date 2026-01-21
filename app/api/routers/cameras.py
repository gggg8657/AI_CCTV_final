"""
카메라 관리 API 라우터
====================

카메라 CRUD, 시작/중지, 상태 조회
"""

from fastapi import APIRouter, HTTPException, status
from typing import List, Optional
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter()


@router.get("/")
async def list_cameras():
    """카메라 목록 조회"""
    # TODO: 구현 예정
    return {
        "cameras": [],
        "total": 0
    }


@router.get("/{camera_id}")
async def get_camera(camera_id: int):
    """카메라 상세 조회"""
    # TODO: 구현 예정
    return {
        "message": f"Camera {camera_id} detail - 구현 예정"
    }


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_camera():
    """카메라 생성"""
    # TODO: 구현 예정
    return {
        "message": "Camera creation - 구현 예정"
    }


@router.put("/{camera_id}")
async def update_camera(camera_id: int):
    """카메라 수정"""
    # TODO: 구현 예정
    return {
        "message": f"Camera {camera_id} update - 구현 예정"
    }


@router.delete("/{camera_id}")
async def delete_camera(camera_id: int):
    """카메라 삭제"""
    # TODO: 구현 예정
    return {
        "message": f"Camera {camera_id} deletion - 구현 예정"
    }


@router.post("/{camera_id}/start")
async def start_camera(camera_id: int):
    """카메라 시작"""
    # TODO: 구현 예정
    return {
        "message": f"Camera {camera_id} start - 구현 예정"
    }


@router.post("/{camera_id}/stop")
async def stop_camera(camera_id: int):
    """카메라 중지"""
    # TODO: 구현 예정
    return {
        "message": f"Camera {camera_id} stop - 구현 예정"
    }
