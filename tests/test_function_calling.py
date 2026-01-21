from dataclasses import dataclass
from typing import Optional

import pytest

from src.agent.function_calling import (
    FunctionRegistry,
    enable_vlm,
    get_anomaly_statistics,
    get_recent_events,
    get_system_status,
    register_core_functions,
    update_vad_threshold,
)


@dataclass
class DummyStats:
    value: int = 0

    def to_dict(self):
        return {"value": self.value}


@dataclass
class DummyConfig:
    vad_threshold: float = 0.5
    enable_vlm: bool = False


@dataclass
class DummyEvent:
    event_type: str
    timestamp: str = "ts"
    score: Optional[float] = None
    threshold: Optional[float] = None
    frame: Optional[object] = "frame"
    event_id: str = "id"


class DummyEventBus:
    def __init__(self, events):
        self.events = list(events)
        self.last_limit = None

    def get_history(self, limit=20):
        self.last_limit = limit
        return list(self.events)[:limit]


@dataclass
class DummyE2ESystem:
    stats: Optional[DummyStats] = None
    config: Optional[DummyConfig] = None
    event_bus: Optional[DummyEventBus] = None
    is_running: bool = False


@pytest.fixture
def registry():
    return FunctionRegistry()


@pytest.fixture
def e2e_system():
    return DummyE2ESystem(
        stats=DummyStats(value=3),
        config=DummyConfig(vad_threshold=0.4, enable_vlm=False),
        event_bus=DummyEventBus([]),
        is_running=True,
    )


def test_registry_initialization_empty(registry):
    assert registry.list_functions() == []


def test_registry_register_success_and_duplicate(registry):
    def sample(value: int = 0):
        return value + 1

    registry.register(
        name="sample",
        description="test",
        parameters={"type": "object", "properties": {}},
        func=sample,
    )

    with pytest.raises(ValueError):
        registry.register(
            name="sample",
            description="test",
            parameters={"type": "object", "properties": {}},
            func=sample,
        )


def test_registry_call_success_and_unknown(registry):
    def sample(value: int = 0):
        return value + 2

    registry.register(
        name="sample",
        description="test",
        parameters={"type": "object", "properties": {}},
        func=sample,
    )

    assert registry.call("sample", {"value": 3}) == 5

    with pytest.raises(KeyError):
        registry.call("missing")


def test_registry_list_functions_openai_format(registry):
    registry.register(
        name="sample",
        description="test",
        parameters={"type": "object", "properties": {}},
        func=lambda: None,
    )

    functions = registry.list_functions()
    assert functions == [
        {
            "type": "function",
            "function": {
                "name": "sample",
                "description": "test",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_registry_get_function_schema(registry):
    registry.register(
        name="sample",
        description="test",
        parameters={"type": "object", "properties": {"value": {"type": "integer"}}},
        func=lambda: None,
    )

    assert registry.get_function_schema("sample") == {
        "name": "sample",
        "description": "test",
        "parameters": {"type": "object", "properties": {"value": {"type": "integer"}}},
    }

    with pytest.raises(KeyError):
        registry.get_function_schema("missing")


@pytest.mark.asyncio
async def test_get_system_status(e2e_system):
    result = get_system_status(e2e_system)

    assert result["ok"] is True
    assert result["data"]["is_running"] is True
    assert result["data"]["stats"] == {"value": 3}
    assert result["data"]["config"]["vad_threshold"] == 0.4
    assert result["data"]["config"]["enable_vlm"] is False


def test_get_recent_events_with_event_bus():
    events = [
        DummyEvent(event_type="TestEvent", event_id="a"),
        DummyEvent(event_type="TestEvent", event_id="b"),
        DummyEvent(event_type="TestEvent", event_id="c"),
    ]
    e2e_system = DummyE2ESystem(event_bus=DummyEventBus(events))

    result = get_recent_events(e2e_system, limit=2)

    assert result["ok"] is True
    assert result["data"]["count"] == 2
    assert result["data"]["limit"] == 2
    assert result["data"]["events"][0]["event_id"] == "a"
    assert result["data"]["events"][0]["frame"] is None

    bad = get_recent_events(e2e_system, limit="nope")
    assert bad["ok"] is False


def test_get_anomaly_statistics_from_events():
    events = [
        DummyEvent(
            event_type="AnomalyDetectedEvent",
            timestamp="t1",
            score=0.5,
            threshold=0.2,
        ),
        DummyEvent(event_type="InfoEvent"),
        DummyEvent(
            event_type="AnomalyDetectedEvent",
            timestamp="t2",
            score=0.7,
            threshold=0.4,
        ),
    ]
    e2e_system = DummyE2ESystem(event_bus=DummyEventBus(events))

    result = get_anomaly_statistics(e2e_system)

    assert result["ok"] is True
    assert result["data"]["total_events"] == 3
    assert result["data"]["anomaly_count"] == 2
    assert result["data"]["avg_anomaly_score"] == 0.6
    assert result["data"]["avg_anomaly_threshold"] == 0.3
    assert result["data"]["latest_anomaly_timestamp"] == "t2"
    assert result["data"]["events_by_type"]["InfoEvent"] == 1


def test_update_vad_threshold_success_and_validation(e2e_system):
    result = update_vad_threshold(e2e_system, value=0.9)

    assert result["ok"] is True
    assert result["data"]["previous_vad_threshold"] == 0.4
    assert result["data"]["vad_threshold"] == 0.9
    assert e2e_system.config.vad_threshold == 0.9

    bad = update_vad_threshold(e2e_system, value="bad")
    assert bad["ok"] is False


def test_enable_vlm_success_and_validation(e2e_system):
    result = enable_vlm(e2e_system, enabled=True)

    assert result["ok"] is True
    assert result["data"]["previous_enable_vlm"] is False
    assert result["data"]["enable_vlm"] is True
    assert e2e_system.config.enable_vlm is True

    bad = enable_vlm(e2e_system, enabled="yes")
    assert bad["ok"] is False


def test_register_core_functions_registers_all(registry, e2e_system):
    register_core_functions(registry, e2e_system)

    names = {item["function"]["name"] for item in registry.list_functions()}
    assert names == {
        "get_system_status",
        "get_recent_events",
        "get_anomaly_statistics",
        "update_vad_threshold",
        "enable_vlm",
    }

    status = registry.call("get_system_status")
    assert status["ok"] is True
