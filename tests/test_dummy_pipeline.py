"""
Dummy Pipeline 테스트 — 모델 없이 전체 파이프라인 E2E 검증
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dummy.vad import DummyVADModel
from src.dummy.vlm import DummyVLMAnalyzer
from src.dummy.agent import DummyAgentFlow
from src.dummy.video import DummyVideoSource
from src.pipeline.resource_pool import ResourcePool
from src.pipeline.camera_config import CameraConfig, PipelineState
from src.pipeline.camera_pipeline import CameraPipeline
from src.pipeline.multi_camera_manager import MultiCameraManager


# ── Dummy Models ──

def test_dummy_vad():
    vad = DummyVADModel()
    vad.initialize("cpu")
    assert vad.is_initialized
    assert vad.name == "dummy"
    scores = [vad.process_frame() for _ in range(200)]
    assert all(0.0 <= s <= 1.0 for s in scores)
    spikes = [s for s in scores if s > 0.5]
    assert len(spikes) > 0, "Should produce anomaly spikes"


def test_dummy_vlm():
    vlm = DummyVLMAnalyzer()
    vlm.initialize()
    assert vlm.is_initialized
    result = vlm.analyze(frames=[None, None])
    assert result.success
    assert result.detected_type in (
        "Fighting", "Arson", "Falling", "Loitering",
        "Suspicious_Object", "Road_Accident", "Normal",
    )
    assert result.n_frames == 2


def test_dummy_agent():
    agent = DummyAgentFlow(flow_type="sequential")
    agent.initialize()
    assert agent.is_initialized
    result = agent.run(situation_type="Fighting")
    assert result["success"]
    assert result["situation_type"] == "Fighting"
    assert len(result["actions"]) > 0


def test_dummy_video():
    src = DummyVideoSource(width=320, height=240, fps=10, total_frames=5)
    src.open()
    frames = []
    for _ in range(10):
        ret, frame = src.read()
        if not ret:
            break
        frames.append(frame)
    src.close()
    assert len(frames) == 5
    assert frames[0].shape == (240, 320, 3)


# ── ResourcePool dummy mode ──

def test_resource_pool_dummy():
    pool = ResourcePool(gpu_id=0, use_dummy=True)
    vad = pool.get_vad_model("mnad")
    assert vad.is_initialized
    assert vad.name == "dummy"

    vlm = pool.get_vlm_analyzer()
    assert vlm.is_initialized
    result = vlm.analyze(frames=[None])
    assert result.success

    agent = pool.get_agent_flow("sequential")
    assert agent.is_initialized
    result = agent.run()
    assert result["success"]

    pool.close()


# ── CameraPipeline with dummy ──

def test_camera_pipeline_dummy():
    pool = ResourcePool(gpu_id=0, use_dummy=True)
    config = CameraConfig(
        camera_id=99,
        source_type="dummy",
        source_path="synthetic",
        vad_threshold=0.5,
        target_fps=30,
    )

    anomalies = []

    pipeline = CameraPipeline(
        config=config,
        resource_pool=pool,
        on_anomaly=lambda e: anomalies.append(e),
    )

    pipeline.start()
    time.sleep(2)
    pipeline.stop()

    assert pipeline.status.total_frames > 0
    assert len(anomalies) > 0
    assert "vad_score" in anomalies[0]
    assert "vlm_type" in anomalies[0]
    pool.close()


# ── MultiCameraManager with dummy ──

def test_multi_camera_manager_dummy():
    anomalies = []
    mgr = MultiCameraManager(max_cameras=4, gpu_id=0, use_dummy=True)
    mgr.set_anomaly_callback(lambda e: anomalies.append(e))

    mgr.add_camera(CameraConfig(camera_id=1, source_type="dummy", source_path="synthetic", target_fps=30))
    mgr.add_camera(CameraConfig(camera_id=2, source_type="dummy", source_path="synthetic", target_fps=30))

    mgr.start_all()
    time.sleep(2)
    statuses = mgr.get_all_statuses()
    mgr.shutdown()

    assert len(statuses) == 2
    assert all(s["total_frames"] > 0 for s in statuses)
    assert len(anomalies) > 0
    cam_ids = set(e["camera_id"] for e in anomalies)
    assert cam_ids == {1, 2}
