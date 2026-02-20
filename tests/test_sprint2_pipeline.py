"""
Sprint 2 Pipeline 테스트
========================

ResourcePool, CameraConfig, CameraPipeline, MultiCameraManager,
AsyncEventLogger, WebSocket manager 단위 테스트
"""

import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["DATABASE_URL"] = "sqlite://"

import threading
import time
from datetime import datetime

from src.pipeline.resource_pool import ResourcePool
from src.pipeline.camera_config import CameraConfig, CameraStatus, PipelineState
from src.pipeline.multi_camera_manager import MultiCameraManager


# ── ResourcePool ──

def test_resource_pool_init():
    pool = ResourcePool(gpu_id=0)
    assert pool.gpu_id == 0
    assert pool.loaded_models == {"vad": 0, "vlm": 0, "agent": 0}


def test_resource_pool_gpu_memory_no_torch():
    pool = ResourcePool(gpu_id=0)
    info = pool.gpu_memory_info()
    assert isinstance(info, dict)


def test_resource_pool_close():
    pool = ResourcePool(gpu_id=0)
    pool.close()
    assert pool.loaded_models == {"vad": 0, "vlm": 0, "agent": 0}


# ── CameraConfig ──

def test_camera_config_defaults():
    cfg = CameraConfig(camera_id=1, source_path="rtsp://test")
    assert cfg.camera_id == 1
    assert cfg.vad_model == "mnad"
    assert cfg.vad_threshold == 0.5
    assert cfg.enable_vlm is True
    assert cfg.target_fps == 30


def test_camera_status_to_dict():
    status = CameraStatus(
        camera_id=1,
        state=PipelineState.RUNNING,
        total_frames=100,
        anomaly_count=5,
        current_fps=25.0,
        started_at=datetime(2026, 1, 1, 12, 0),
    )
    d = status.to_dict()
    assert d["camera_id"] == 1
    assert d["state"] == "running"
    assert d["total_frames"] == 100
    assert d["current_fps"] == 25.0
    assert "2026-01-01" in d["started_at"]


# ── PipelineState ──

def test_pipeline_state_values():
    assert PipelineState.IDLE == "idle"
    assert PipelineState.RUNNING == "running"
    assert PipelineState.ERROR == "error"


# ── MultiCameraManager ──

def test_manager_init():
    mgr = MultiCameraManager(max_cameras=4, gpu_id=0)
    assert mgr.max_cameras == 4
    assert mgr.active_count == 0
    assert mgr.camera_ids == []


def test_manager_add_remove():
    mgr = MultiCameraManager(max_cameras=2)
    cfg = CameraConfig(camera_id=10, source_path="rtsp://test")
    cam_id = mgr.add_camera(cfg)
    assert cam_id == 10
    assert 10 in mgr.camera_ids

    removed = mgr.remove_camera(10)
    assert removed is True
    assert 10 not in mgr.camera_ids


def test_manager_max_cameras():
    mgr = MultiCameraManager(max_cameras=1)
    mgr.add_camera(CameraConfig(camera_id=1, source_path="rtsp://a"))
    try:
        mgr.add_camera(CameraConfig(camera_id=2, source_path="rtsp://b"))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_manager_duplicate_camera():
    mgr = MultiCameraManager(max_cameras=5)
    mgr.add_camera(CameraConfig(camera_id=1, source_path="rtsp://a"))
    try:
        mgr.add_camera(CameraConfig(camera_id=1, source_path="rtsp://b"))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_manager_status():
    mgr = MultiCameraManager(max_cameras=5)
    mgr.add_camera(CameraConfig(camera_id=1, source_path="rtsp://test"))
    status = mgr.get_camera_status(1)
    assert status is not None
    assert status.state == PipelineState.IDLE

    all_statuses = mgr.get_all_statuses()
    assert len(all_statuses) == 1
    assert all_statuses[0]["state"] == "idle"


def test_manager_nonexistent_camera():
    mgr = MultiCameraManager()
    assert mgr.get_camera_status(999) is None
    assert mgr.start_camera(999) is False
    assert mgr.stop_camera(999) is False
    assert mgr.remove_camera(999) is False


def test_manager_shutdown():
    mgr = MultiCameraManager()
    mgr.add_camera(CameraConfig(camera_id=1, source_path="rtsp://test"))
    mgr.shutdown()
    assert mgr.resource_pool.loaded_models == {"vad": 0, "vlm": 0, "agent": 0}


# ── AsyncEventLogger ──

def test_async_event_logger_lifecycle():
    from src.database.event_logger import AsyncEventLogger
    from src.database.db import engine, SessionLocal
    from src.database.models import Base, Camera, Event

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    db.add(Camera(id=1, name="Test Cam", source_type="file", source_path="/dev/null"))
    db.commit()
    db.close()

    ael = AsyncEventLogger()
    ael.start()

    for i in range(3):
        ael.log({
            "camera_id": 1,
            "timestamp": datetime.now(),
            "vad_score": 0.8 + i * 0.05,
            "threshold": 0.5,
            "vlm_type": "Fighting",
        })

    time.sleep(2)
    ael.stop()

    db = SessionLocal()
    count = db.query(Event).count()
    db.close()
    assert count == 3


# ── WebSocket Manager ──

def test_ws_manager_init():
    from app.api.websocket.manager import ConnectionManager
    mgr = ConnectionManager()
    assert mgr.total_connections == 0
    assert mgr.channel_counts == {}
