"""
API ↔ Pipeline 통합 테스트
===========================

카메라 start/stop이 실제 파이프라인을 구동하는지,
pipeline-status/overview 엔드포인트가 동작하는지 검증.
"""

import sys, os, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["DATABASE_URL"] = "sqlite://"
os.environ["JWT_SECRET_KEY"] = "test-secret-key"
os.environ["PIPELINE_DUMMY"] = "true"

from fastapi.testclient import TestClient
from app.api.main import app
from src.database.db import engine
from src.database.models import Base

client = TestClient(app)
_token = None


def setup_module():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    from app.api.pipeline_state import _manager, init_pipeline
    if _manager is None:
        init_pipeline()

    client.post("/api/v1/auth/register", json={
        "username": "testadmin", "email": "t@t.com", "password": "pass123"
    })
    global _token
    r = client.post("/api/v1/auth/login", json={"username": "testadmin", "password": "pass123"})
    _token = r.json()["access_token"]


def h():
    return {"Authorization": f"Bearer {_token}"}


def test_create_dummy_camera():
    r = client.post("/api/v1/cameras/", headers=h(), json={
        "name": "Dummy Cam", "source_type": "dummy",
        "source_path": "synthetic", "location": "Test"
    })
    assert r.status_code == 201
    assert r.json()["source_type"] == "dummy"


def test_start_camera_runs_pipeline():
    r = client.post("/api/v1/cameras/1/start", headers=h())
    assert r.status_code == 200
    assert r.json()["status"] == "active"

    time.sleep(1.5)

    r = client.get("/api/v1/cameras/1/pipeline-status", headers=h())
    assert r.status_code == 200
    data = r.json()
    assert data["state"] == "running"
    assert data["total_frames"] > 0


def test_pipeline_overview():
    r = client.get("/api/v1/cameras/pipeline/overview", headers=h())
    assert r.status_code == 200
    data = r.json()
    assert data["active_cameras"] >= 1
    assert data["dummy_flags"]["vad"] is True
    assert data["dummy_flags"]["vlm"] is True


def test_stop_camera_stops_pipeline():
    r = client.post("/api/v1/cameras/1/stop", headers=h())
    assert r.status_code == 200
    assert r.json()["status"] == "inactive"

    time.sleep(0.5)

    r = client.get("/api/v1/cameras/1/pipeline-status", headers=h())
    assert r.status_code == 200
    assert r.json()["state"] == "idle"


def test_health_includes_pipeline():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "pipeline" in data
    assert "active_cameras" in data["pipeline"]


def test_delete_camera_removes_pipeline():
    r = client.delete("/api/v1/cameras/1", headers=h())
    assert r.status_code == 204

    r = client.get("/api/v1/cameras/1/pipeline-status", headers=h())
    assert r.status_code == 404
