import pytest

from app import e2e_system
from app.e2e_system import E2ESystem, SystemConfig, VideoSourceType
from src.utils.event_bus import EventBus
from src.utils.event_handlers import VADEventHandler, VLMEventHandler, AgentEventHandler
from src.utils.events import (
    AnomalyDetectedEvent,
    AgentResponseEvent,
    FrameProcessedEvent,
    StatsUpdatedEvent,
    VLMAnalysisCompletedEvent,
)


class DummyLogger:
    def __init__(self, *args, **kwargs):
        self.events = []

    def log_info(self, *args, **kwargs):
        pass

    def log_warning(self, *args, **kwargs):
        pass

    def log_error(self, *args, **kwargs):
        pass

    def log_event(self, event):
        self.events.append(event)


class DummyVideoSource:
    default_frames = []

    def __init__(self, source_type, source_path):
        self.source_type = source_type
        self.source_path = source_path
        self._frames = list(self.default_frames)

    def open(self):
        return True

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def close(self):
        pass

    def get_info(self):
        return {
            "type": self.source_type.value,
            "path": self.source_path,
            "fps": 30,
            "resolution": "640x480",
            "total_frames": len(self.default_frames),
        }


class DummyVAD:
    def __init__(self, *args, score=0.0, delay=0.0, **kwargs):
        self._loaded = True
        self._score = score
        self._delay = delay

    def load(self):
        self._loaded = True
        return True

    def predict(self, frames):
        return self._score, self._delay


class DummyVLM:
    def __init__(self, *args, **kwargs):
        self._result = {
            "detected_type": "TestType",
            "description": "Test description",
            "confidence": 0.9,
            "actions": ["notify"],
            "latency_ms": 12.0,
        }

    def load(self):
        return True

    def analyze(self, frames, clip_path):
        return dict(self._result)


class DummyAgent:
    def __init__(self, *args, **kwargs):
        self._result = {
            "actions": [{"action": "notify_security"}],
            "priority": 2,
            "processing_time": 0.05,
        }

    def load(self):
        return True

    def process(self, event, vlm_result):
        return dict(self._result)


def build_config(**overrides):
    config = SystemConfig(
        source_type=VideoSourceType.FILE,
        source_path="dummy.mp4",
        save_clips=False,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_e2e_system_initializes_with_event_bus():
    config = build_config(enable_vlm=False, enable_agent=False)
    system = E2ESystem(config)

    assert isinstance(system.event_bus, EventBus)
    assert system.get_event_bus() is system.event_bus


def test_initialize_registers_event_handlers(monkeypatch):
    config = build_config(enable_vlm=True, enable_agent=True)
    system = E2ESystem(config)

    monkeypatch.setattr(e2e_system, "EventLogger", DummyLogger)
    monkeypatch.setattr(e2e_system, "VideoSource", DummyVideoSource)
    monkeypatch.setattr(e2e_system, "VADWrapper", DummyVAD)
    monkeypatch.setattr(e2e_system, "VLMWrapper", DummyVLM)
    monkeypatch.setattr(e2e_system, "AgentWrapper", DummyAgent)

    success, error = system.initialize()
    try:
        assert success is True
        assert error is None
        assert isinstance(system.vad_event_handler, VADEventHandler)
        assert isinstance(system.vlm_event_handler, VLMEventHandler)
        assert isinstance(system.agent_event_handler, AgentEventHandler)
        assert system.event_bus.get_subscriber_count(AnomalyDetectedEvent) == 1
        assert system.event_bus.get_subscriber_count(VLMAnalysisCompletedEvent) == 1
        assert system.event_bus.get_subscriber_count(AgentResponseEvent) == 1
    finally:
        try:
            system.event_bus.stop()
        except Exception:
            # Ignore CancelledError from async task cancellation
            pass


def test_event_publishing_during_frame_processing(monkeypatch):
    config = build_config(enable_vlm=False, enable_agent=False, target_fps=30)
    system = E2ESystem(config)
    system.logger = DummyLogger()
    system.event_bus = EventBus(max_history=10)
    DummyVideoSource.default_frames = [["frame"]]
    system.video_source = DummyVideoSource(config.source_type, config.source_path)
    system.vad = DummyVAD(score=0.0)

    monkeypatch.setattr(e2e_system, "HAS_CV2", False)

    def fake_time():
        fake_time.current += 1.1
        return fake_time.current

    fake_time.current = 0.0
    monkeypatch.setattr(e2e_system.time, "time", fake_time)
    monkeypatch.setattr(e2e_system.time, "sleep", lambda *_: None)

    system._process_loop()

    history = system.event_bus.get_history(10)
    assert any(isinstance(event, FrameProcessedEvent) for event in history)
    assert any(isinstance(event, StatsUpdatedEvent) for event in history)


def test_event_flow_vad_vlm_agent(monkeypatch):
    config = build_config(enable_vlm=True, enable_agent=True)
    system = E2ESystem(config)
    system.logger = DummyLogger()
    system.event_bus = EventBus(max_history=10)
    system.vlm = DummyVLM()
    system.agent = DummyAgent()
    system.stats.total_frames = 1

    monkeypatch.setattr(e2e_system, "HAS_CV2", False)

    system._register_event_handlers()
    system._handle_anomaly(frame=["frame"], vad_score=0.9)

    history = [
        event
        for event in system.event_bus.get_history(10)
        if isinstance(
            event,
            (AnomalyDetectedEvent, VLMAnalysisCompletedEvent, AgentResponseEvent),
        )
    ]

    assert [type(event) for event in history] == [
        AnomalyDetectedEvent,
        VLMAnalysisCompletedEvent,
        AgentResponseEvent,
    ]
    assert history[1].original_event_id == history[0].event_id
    assert history[2].original_event_id == history[0].event_id


@pytest.mark.asyncio
async def test_event_handler_statistics_async():
    config = build_config(enable_vlm=True, enable_agent=True)
    system = E2ESystem(config)
    system.event_bus = EventBus(max_history=10)
    system._register_event_handlers()

    anomaly_event = AnomalyDetectedEvent(
        event_id="evt_1",
        event_type="AnomalyDetectedEvent",
        timestamp="2024-01-01T00:00:00",
        source="VAD",
        frame_id=1,
        score=0.7,
        threshold=0.5,
        frame=None,
    )
    vlm_event = VLMAnalysisCompletedEvent(
        event_id="evt_2",
        event_type="VLMAnalysisCompletedEvent",
        timestamp="2024-01-01T00:00:01",
        source="VLM",
        original_event_id="evt_1",
        detected_type="TestType",
        description="Test",
        actions=["notify"],
        confidence=0.8,
        clip_path="",
    )
    agent_event = AgentResponseEvent(
        event_id="evt_3",
        event_type="AgentResponseEvent",
        timestamp="2024-01-01T00:00:02",
        source="Agent",
        original_event_id="evt_1",
        plan=["notify"],
        priority=2,
        estimated_time=0.4,
    )

    await system.event_bus.publish(anomaly_event)
    await system.event_bus.publish(vlm_event)
    await system.event_bus.publish(agent_event)

    vad_stats = system.vad_event_handler.get_stats()
    vlm_stats = system.vlm_event_handler.get_stats()
    agent_stats = system.agent_event_handler.get_stats()

    assert vad_stats["total_events"] == 1
    assert vlm_stats["total_events"] == 1
    assert agent_stats["total_events"] == 1
