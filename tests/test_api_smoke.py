"""
API Smoke Test — Sprint 1 기반 검증
"""

import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["DATABASE_URL"] = "sqlite://"
os.environ["JWT_SECRET_KEY"] = "test-secret-key"

from fastapi.testclient import TestClient
from app.api.main import app
from src.database.db import engine
from src.database.models import Base, Event
from sqlalchemy.orm import Session
from datetime import datetime

client = TestClient(app)

_cached_token = None


def setup_module():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def _get_token() -> str:
    global _cached_token
    if _cached_token:
        return _cached_token
    r = client.post("/api/v1/auth/login", json={
        "username": "admin", "password": "secret123"
    })
    _cached_token = r.json()["access_token"]
    return _cached_token


def _headers():
    return {"Authorization": "Bearer " + _get_token()}


# ── Auth ──

def test_register():
    r = client.post("/api/v1/auth/register", json={
        "username": "admin", "email": "admin@test.com", "password": "secret123"
    })
    assert r.status_code == 201
    assert r.json()["username"] == "admin"
    assert r.json()["role"] == "admin"


def test_register_duplicate():
    r = client.post("/api/v1/auth/register", json={
        "username": "admin", "email": "admin@test.com", "password": "secret123"
    })
    assert r.status_code == 409


def test_login():
    r = client.post("/api/v1/auth/login", json={
        "username": "admin", "password": "secret123"
    })
    assert r.status_code == 200
    data = r.json()
    assert "access_token" in data
    assert "refresh_token" in data


def test_me():
    r = client.get("/api/v1/auth/me", headers=_headers())
    assert r.status_code == 200
    assert r.json()["username"] == "admin"


def test_refresh():
    login_r = client.post("/api/v1/auth/login", json={
        "username": "admin", "password": "secret123"
    })
    refresh_tok = login_r.json()["refresh_token"]
    r = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_tok})
    assert r.status_code == 200
    assert "access_token" in r.json()


# ── Cameras ──

def test_create_camera():
    r = client.post("/api/v1/cameras/", headers=_headers(), json={
        "name": "Front Gate", "source_type": "rtsp",
        "source_path": "rtsp://192.168.1.100/stream", "location": "Main Entrance"
    })
    assert r.status_code == 201
    assert r.json()["name"] == "Front Gate"


def test_list_cameras():
    r = client.get("/api/v1/cameras/", headers=_headers())
    assert r.status_code == 200
    assert len(r.json()) >= 1


def test_get_camera():
    r = client.get("/api/v1/cameras/1", headers=_headers())
    assert r.status_code == 200
    assert r.json()["id"] == 1


def test_update_camera():
    r = client.put("/api/v1/cameras/1", headers=_headers(), json={
        "name": "Front Gate (Updated)"
    })
    assert r.status_code == 200
    assert r.json()["name"] == "Front Gate (Updated)"


def test_start_stop_camera():
    r = client.post("/api/v1/cameras/1/start", headers=_headers())
    assert r.status_code == 200
    assert r.json()["status"] == "active"

    r = client.post("/api/v1/cameras/1/stop", headers=_headers())
    assert r.status_code == 200
    assert r.json()["status"] == "inactive"


# ── Events ──

def test_list_events_empty():
    r = client.get("/api/v1/events/", headers=_headers())
    assert r.status_code == 200
    assert r.json()["total"] == 0


def _seed_events():
    from src.database.db import SessionLocal
    db = SessionLocal()
    for i in range(3):
        db.add(Event(
            camera_id=1, timestamp=datetime(2026, 2, 20, 10, i),
            vad_score=0.8 + i * 0.05, threshold=0.5,
            vlm_type="Fighting" if i % 2 == 0 else "Loitering",
            vlm_description="Test event %d" % i,
        ))
    db.commit()
    db.close()


def test_list_events_with_data():
    _seed_events()
    r = client.get("/api/v1/events/", headers=_headers())
    assert r.status_code == 200
    assert r.json()["total"] == 3


def test_get_event():
    r = client.get("/api/v1/events/1", headers=_headers())
    assert r.status_code == 200
    assert r.json()["camera_id"] == 1


def test_ack_event():
    r = client.post("/api/v1/events/1/ack", headers=_headers(), json={
        "note": "Checked by admin"
    })
    assert r.status_code == 200
    assert r.json()["acknowledged"] is True
    assert r.json()["note"] == "Checked by admin"


def test_filter_events():
    r = client.get("/api/v1/events/?vlm_type=Fighting", headers=_headers())
    assert r.status_code == 200
    assert r.json()["total"] == 2


# ── Stats ──

def test_stats_summary():
    r = client.get("/api/v1/stats/summary?days=30", headers=_headers())
    assert r.status_code == 200
    data = r.json()
    assert data["total_events"] == 3
    assert "vlm_type_distribution" in data


# ── Health & Auth guard ──

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_unauthorized():
    r = client.get("/api/v1/cameras/")
    assert r.status_code in (401, 403)
