from dataclasses import dataclass
from typing import Optional

import pytest

import app.e2e_system as e2e_system
from src.agent.flows.function_calling_support import FunctionCallingSupport
from src.agent.flows.sequential import SequentialFlow


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
    event_id: str = "evt"


class DummyEventBus:
    def __init__(self, events):
        self.events = list(events)

    def get_history(self, limit=20):
        return list(self.events)[:limit]


@dataclass
class DummyE2ESystem:
    stats: Optional[DummyStats] = None
    config: Optional[DummyConfig] = None
    event_bus: Optional[DummyEventBus] = None
    is_running: bool = False


class DummyTextLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if self.responses:
            return self.responses.pop(0)
        return {"choices": [{"message": {"content": ""}}]}


class DummyLLMManager:
    def __init__(self, text_llm):
        self.text_llm = text_llm


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
    def __init__(self, model_type, checkpoint_path="", gpu_id=0):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.gpu_id = gpu_id

    def load(self):
        return True

    def predict(self, frames):
        return 0.8, 0.01


class DummyVLMWrapper:
    def __init__(self, model_path, n_frames, optimize, gpu_id=0):
        self.model_path = model_path
        self.n_frames = n_frames
        self.optimize = optimize
        self.gpu_id = gpu_id

    def load(self):
        return True

    def analyze(self, frames, clip_path):
        return {
            "detected_type": "TestType",
            "description": "test description",
            "confidence": 0.9,
            "actions": ["notify"],
            "latency_ms": 12.0,
        }


class DummyAgentWrapper:
    def __init__(self, flow, gpu_id=0, e2e_system=None):
        self.flow = flow
        self.gpu_id = gpu_id
        self.e2e_system = e2e_system

    def load(self):
        return True

    def process(self, event, vlm_result):
        return {
            "actions": [{"action": "notify", "priority": "high"}],
            "priority": 2,
            "processing_time": 0.25,
        }


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


def _build_system(monkeypatch, tmp_path):
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


def _build_dummy_e2e_system():
    return DummyE2ESystem(
        stats=DummyStats(value=7),
        config=DummyConfig(vad_threshold=0.35, enable_vlm=False),
        event_bus=DummyEventBus([DummyEvent(event_type="TestEvent")]),
        is_running=True,
    )


@pytest.mark.asyncio
async def test_function_calling_support_initialization_with_e2e_system():
    llm_manager = DummyLLMManager(text_llm=DummyTextLLM([]))
    support = FunctionCallingSupport(llm_manager, _build_dummy_e2e_system())

    assert support._ready is True
    names = {item["function"]["name"] for item in support.registry.list_functions()}
    assert "get_system_status" in names
    assert "get_recent_events" in names


def test_function_calling_support_process_query_simple():
    response = {"choices": [{"message": {"content": "Simple response"}}]}
    text_llm = DummyTextLLM([response])
    llm_manager = DummyLLMManager(text_llm=text_llm)
    support = FunctionCallingSupport(llm_manager, _build_dummy_e2e_system())

    result = support.process_query("What is the status?")

    assert result["success"] is True
    assert result["response"] == "Simple response"
    assert result["tool_calls"] == []
    assert result["tool_results"] == []
    assert text_llm.calls[0]["tools"]


def test_function_calling_support_process_query_with_function_call():
    tool_response = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_system_status", "arguments": "{}"},
                        }
                    ],
                }
            }
        ]
    }
    final_response = {"choices": [{"message": {"content": "System ok"}}]}
    text_llm = DummyTextLLM([tool_response, final_response])
    llm_manager = DummyLLMManager(text_llm=text_llm)
    support = FunctionCallingSupport(llm_manager, _build_dummy_e2e_system())

    result = support.process_query("Tell me the system status.")

    assert result["success"] is True
    assert result["response"] == "System ok"
    assert result["tool_calls"]
    assert result["tool_results"][0]["name"] == "get_system_status"
    assert result["tool_results"][0]["arguments"] == {}
    assert result["tool_results"][0]["result"]["ok"] is True
    assert "tools" not in text_llm.calls[1]


def test_sequential_flow_process_query_integration():
    text_llm = DummyTextLLM(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_system_status",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "Flow ok"}}]},
        ]
    )
    llm_manager = DummyLLMManager(text_llm=text_llm)
    dummy_system = _build_dummy_e2e_system()

    flow = SequentialFlow()
    flow._initialized = True
    flow.llm_manager = llm_manager
    flow.e2e_system = dummy_system
    flow.function_calling = FunctionCallingSupport(llm_manager, dummy_system)

    result = flow.process_query("Check status.")

    assert result["success"] is True
    assert result["response"] == "Flow ok"
    assert result["tool_results"][0]["result"]["ok"] is True


def test_e2e_system_function_registry_access(monkeypatch, tmp_path):
    system = _build_system(monkeypatch, tmp_path)
    try:
        registry = system.get_function_registry()
        assert registry is not None
        names = {item["function"]["name"] for item in registry.list_functions()}
        assert "get_system_status" in names
        assert "get_recent_events" in names
    finally:
        system.event_bus.stop()
