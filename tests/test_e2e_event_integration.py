import pytest

import app.e2e_system as e2e_system
from src.utils.events import (
    AgentResponseEvent,
    AnomalyDetectedEvent,
    FrameProcessedEvent,
    StatsUpdatedEvent,
    VLMAnalysisCompletedEvent,
)


class DummyVideoSource:
    frames = []

    def __init__(self, source_type, source_path):
        self.source_type = source_type
        self.source_path = source_path
        self._frames = list(self.__class__.frames)
        self._index = 0

    def open(self):
        return True

    def read(self):
        if self._index < len(self._frames):
            frame = self._frames[self._index]
            self._index += 1
            return True, frame
        return False, None

    def close(self):
        return None

    def get_info(self):
        return {"type": "dummy", "path": self.source_path}


class DummyVADWrapper:
    score = 0.8

    def __init__(self, model_type, checkpoint_path="", gpu_id=0):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.gpu_id = gpu_id
        self._loaded = False

    def load(self):
        self._loaded = True
        return True

    def predict(self, frames):
        return float(self.__class__.score), 0.01


class DummyVLMWrapper:
    result = {
        "detected_type": "TestType",
        "description": "test description",
        "confidence": 0.9,
        "actions": ["notify"],
        "latency_ms": 12.0,
    }

    def __init__(self, model_path, n_frames, optimize, gpu_id=0):
        self.model_path = model_path
        self.n_frames = n_frames
        self.optimize = optimize
        self.gpu_id = gpu_id

    def load(self):
        return True

    def analyze(self, frames, clip_path):
        return dict(self.__class__.result)


class DummyAgentWrapper:
    result = {
        "actions": [{"action": "notify", "priority": "high"}],
        "priority": 2,
        "processing_time": 0.25,
    }

    def __init__(self, flow, gpu_id=0):
        self.flow = flow
        self.gpu_id = gpu_id

    def load(self):
        return True

    def process(self, event, vlm_result):
        return dict(self.__class__.result)


class DummyClipSaver:
    def __init__(self, clips_dir, clip_duration, fps):
        self.clips_dir = clips_dir
        self.clip_duration = clip_duration
        self.fps = fps

    def add_frame(self, frame):
        return None

    def save_clip(self, event_id):
        return "dummy_clip_path"


class DummyEventLogger:
    def __init__(self, log_dir, log_level="INFO"):
        self.log_dir = log_dir
        self.log_level = log_level
        self.events = []

    def log_event(self, event):
        self.events.append(event)

    def log_info(self, message):
        return None

    def log_warning(self, message):
        return None

    def log_error(self, message):
        return None

    def get_recent_events(self, n=10):
        return self.events[-n:]


def _build_system(monkeypatch, tmp_path, frames, vad_score=0.8):
    DummyVideoSource.frames = list(frames)
    DummyVADWrapper.score = vad_score

    monkeypatch.setattr(e2e_system, "VideoSource", DummyVideoSource)
    monkeypatch.setattr(e2e_system, "VADWrapper", DummyVADWrapper)
    monkeypatch.setattr(e2e_system, "VLMWrapper", DummyVLMWrapper)
    monkeypatch.setattr(e2e_system, "AgentWrapper", DummyAgentWrapper)
    monkeypatch.setattr(e2e_system, "ClipSaver", DummyClipSaver)
    monkeypatch.setattr(e2e_system, "EventLogger", DummyEventLogger)
    monkeypatch.setattr(e2e_system, "HAS_CV2", False)
    monkeypatch.setattr(e2e_system, "cv2", None)

    def _stub_start(self):
        self._running = True
        self._queue = None
        self._processing_task = None

    monkeypatch.setattr(e2e_system.EventBus, "start", _stub_start)

    config = e2e_system.SystemConfig(
        source_type=e2e_system.VideoSourceType.FILE,
        source_path="dummy.mp4",
        vad_model=e2e_system.VADModelType.MNAD,
        vad_threshold=0.5,
        enable_vlm=True,
        vlm_n_frames=4,
        optimize_vlm=False,
        enable_agent=True,
        agent_flow=e2e_system.AgentFlowType.SEQUENTIAL,
        save_clips=False,
        clip_duration=1.0,
        clips_dir=str(tmp_path / "clips"),
        log_dir=str(tmp_path / "logs"),
        target_fps=1000,
    )

    system = e2e_system.E2ESystem(config)
    ok, error = system.initialize()
    assert ok, error
    return system


def test_e2e_system_initializes_event_bus(monkeypatch, tmp_path):
    system = _build_system(monkeypatch, tmp_path, frames=["frame"])
    try:
        assert system.get_event_bus() is not None
        assert system.event_bus is system.get_event_bus()
        assert system.event_bus._running is True
    finally:
        system.event_bus.stop()


def test_event_handler_registration(monkeypatch, tmp_path):
    system = _build_system(monkeypatch, tmp_path, frames=["frame"])
    try:
        assert system.vad_event_handler is not None
        assert system.vlm_event_handler is not None
        assert system.agent_event_handler is not None
        assert system.event_bus.get_subscriber_count(AnomalyDetectedEvent) == 1
        assert system.event_bus.get_subscriber_count(VLMAnalysisCompletedEvent) == 1
        assert system.event_bus.get_subscriber_count(AgentResponseEvent) == 1
    finally:
        system.event_bus.stop()


def test_event_publishing_during_frame_processing(monkeypatch, tmp_path):
    system = _build_system(monkeypatch, tmp_path, frames=["frame"])
    times = iter([0.0, 0.0, 2.0, 2.5, 4.0, 4.1, 4.2])

    monkeypatch.setattr(e2e_system.time, "time", lambda: next(times))
    monkeypatch.setattr(e2e_system.time, "sleep", lambda _duration: None)

    try:
        system._process_loop()
        history = system.event_bus.get_history(limit=20)
        history_types = {type(event).__name__ for event in history}
        assert "FrameProcessedEvent" in history_types
        assert "AnomalyDetectedEvent" in history_types
        assert "VLMAnalysisCompletedEvent" in history_types
        assert "AgentResponseEvent" in history_types
        assert "StatsUpdatedEvent" in history_types
    finally:
        system.event_bus.stop()


def test_event_flow_vad_vlm_agent(monkeypatch, tmp_path):
    system = _build_system(monkeypatch, tmp_path, frames=[])
    try:
        system.event_bus.clear_history()
        system.frame_buffer.append("frame")
        system.stats.total_frames = 1
        system._handle_anomaly("frame", vad_score=0.8)
        history = system.event_bus.get_history(limit=10)
        history_types = [type(event).__name__ for event in history]
        assert history_types == [
            "AnomalyDetectedEvent",
            "VLMAnalysisCompletedEvent",
            "AgentResponseEvent",
        ]
    finally:
        system.event_bus.stop()


def test_event_handler_statistics(monkeypatch, tmp_path):
    system = _build_system(monkeypatch, tmp_path, frames=[])
    try:
        system.event_bus.clear_history()
        system.frame_buffer.append("frame")
        system.stats.total_frames = 1
        system._handle_anomaly("frame", vad_score=0.8)

        vad_stats = system.vad_event_handler.get_stats()
        vlm_stats = system.vlm_event_handler.get_stats()
        agent_stats = system.agent_event_handler.get_stats()

        assert vad_stats["total_events"] == 1
        assert vad_stats["last_event_type"] == "AnomalyDetectedEvent"
        assert vlm_stats["total_events"] == 1
        assert vlm_stats["last_event_type"] == "VLMAnalysisCompletedEvent"
        assert agent_stats["total_events"] == 1
        assert agent_stats["last_event_type"] == "AgentResponseEvent"
    finally:
        system.event_bus.stop()
