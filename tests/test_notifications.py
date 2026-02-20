"""
알림 시스템 테스트
================

NotificationEngine + 채널 + API 엔드포인트
"""

import sys, os, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["DATABASE_URL"] = "sqlite://"
os.environ["JWT_SECRET_KEY"] = "test-secret-key"
os.environ["PIPELINE_DUMMY"] = "true"

from src.notifications.base import NotificationPayload
from src.notifications.console import ConsoleChannel
from src.notifications.engine import NotificationEngine


# ── Unit tests ──

def test_notification_payload():
    p = NotificationPayload(
        camera_id=1, camera_name="Front", location="Gate",
        vlm_type="Fighting", vad_score=0.85, severity="high",
    )
    assert "[HIGH] Fighting" in p.title
    assert "Front" in p.body
    d = p.to_dict()
    assert d["camera_id"] == 1
    assert d["vlm_type"] == "Fighting"


def test_console_channel():
    ch = ConsoleChannel()
    assert ch.name == "console"
    assert ch.is_available
    p = NotificationPayload(camera_id=1, vlm_type="Arson", severity="critical")
    assert ch.send(p) is True


def test_engine_basic():
    engine = NotificationEngine(cooldown_seconds=0.5)
    engine.add_channel(ConsoleChannel())
    assert "console" in engine.channels

    ok = engine.notify({
        "camera_id": 1, "vlm_type": "Fighting",
        "vad_score": 0.9, "timestamp": datetime.now().isoformat(),
    })
    assert ok is True
    assert engine.stats["sent"] == 1


def test_engine_cooldown():
    engine = NotificationEngine(cooldown_seconds=1.0)
    engine.add_channel(ConsoleChannel())

    event = {
        "camera_id": 1, "vlm_type": "Fighting",
        "vad_score": 0.9, "timestamp": datetime.now().isoformat(),
    }
    assert engine.notify(event) is True
    assert engine.notify(event) is False
    assert engine.stats["suppressed"] == 1

    time.sleep(1.1)
    assert engine.notify(event) is True
    assert engine.stats["sent"] == 2


def test_engine_skips_normal():
    engine = NotificationEngine(cooldown_seconds=0)
    engine.add_channel(ConsoleChannel())

    ok = engine.notify({
        "camera_id": 1, "vlm_type": "Normal",
        "vad_score": 0.1, "timestamp": datetime.now().isoformat(),
    })
    assert ok is False


def test_engine_remove_channel():
    engine = NotificationEngine()
    engine.add_channel(ConsoleChannel())
    assert engine.remove_channel("console") is True
    assert engine.channels == []


# ── API integration tests ──

from fastapi.testclient import TestClient
from app.api.main import app
from src.database.db import engine as db_engine
from src.database.models import Base

client = TestClient(app)
_token = None


def setup_module():
    Base.metadata.drop_all(bind=db_engine)
    Base.metadata.create_all(bind=db_engine)
    from app.api.pipeline_state import _manager, init_pipeline
    if _manager is None:
        init_pipeline()

    client.post("/api/v1/auth/register", json={
        "username": "notifyadmin", "email": "n@t.com", "password": "pass123"
    })
    global _token
    r = client.post("/api/v1/auth/login", json={"username": "notifyadmin", "password": "pass123"})
    _token = r.json()["access_token"]


def h():
    return {"Authorization": f"Bearer {_token}"}


def test_api_notification_status():
    r = client.get("/api/v1/notifications/status", headers=h())
    assert r.status_code == 200
    data = r.json()
    assert "channels" in data
    assert "console" in data["channels"]


def test_api_create_rule():
    r = client.post("/api/v1/notifications/rules", headers=h(), json={
        "vlm_type": "Fighting",
        "min_score": 0.7,
        "channels": ["console", "webhook"],
    })
    assert r.status_code == 201
    assert r.json()["vlm_type"] == "Fighting"


def test_api_list_rules():
    r = client.get("/api/v1/notifications/rules", headers=h())
    assert r.status_code == 200
    assert len(r.json()) >= 1


def test_api_update_rule():
    r = client.put("/api/v1/notifications/rules/1", headers=h(), json={
        "enabled": False
    })
    assert r.status_code == 200
    assert r.json()["enabled"] is False


def test_api_test_notification():
    r = client.post("/api/v1/notifications/test", headers=h())
    assert r.status_code == 200
    assert "sent" in r.json()


def test_api_delete_rule():
    r = client.delete("/api/v1/notifications/rules/1", headers=h())
    assert r.status_code == 204
