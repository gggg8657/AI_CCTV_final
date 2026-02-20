#!/usr/bin/env python3
"""
Dummy Pipeline Demo — 모델 파일 없이 전체 파이프라인 동작 확인
==============================================================

실행:
    python3 demo_dummy_pipeline.py

모든 컴포넌트가 더미로 동작합니다:
- DummyVAD: 주기적 이상 점수 스파이크 생성
- DummyVLM: 랜덤 이상 유형 분류
- DummyAgent: 시나리오별 대응 액션 생성
- DummyVideoSource: 합성 컬러 프레임 생성

실제 모델로 교체하려면:
    ResourcePool(gpu_id=0, use_dummy=False)
    CameraConfig(source_type="rtsp", source_path="rtsp://...")
"""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.resource_pool import ResourcePool
from src.pipeline.camera_config import CameraConfig
from src.pipeline.camera_pipeline import CameraPipeline
from src.pipeline.multi_camera_manager import MultiCameraManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo")


def on_anomaly(event_data: dict) -> None:
    logger.warning(
        "ANOMALY cam=%d score=%.3f type=%s | %s",
        event_data.get("camera_id", -1),
        event_data.get("vad_score", 0),
        event_data.get("vlm_type", "N/A"),
        event_data.get("vlm_description", "")[:60],
    )


def on_frame(camera_id: int, frame, score: float) -> None:
    pass


def main() -> None:
    logger.info("=== Dummy Pipeline Demo ===")

    mgr = MultiCameraManager(max_cameras=4, gpu_id=0, use_dummy=True)
    mgr.set_anomaly_callback(on_anomaly)
    mgr.set_frame_callback(on_frame)

    cam1 = CameraConfig(
        camera_id=1,
        source_type="dummy",
        source_path="synthetic",
        location="Main Entrance",
        vad_threshold=0.5,
        target_fps=10,
    )
    cam2 = CameraConfig(
        camera_id=2,
        source_type="dummy",
        source_path="synthetic",
        location="Parking Lot",
        vad_threshold=0.5,
        target_fps=10,
    )

    mgr.add_camera(cam1)
    mgr.add_camera(cam2)
    logger.info("Cameras added: %s", mgr.camera_ids)

    mgr.start_all()
    logger.info("All cameras started (active=%d)", mgr.active_count)

    try:
        for i in range(10):
            time.sleep(1)
            statuses = mgr.get_all_statuses()
            for s in statuses:
                logger.info(
                    "  cam=%d state=%s frames=%d anomalies=%d fps=%.0f",
                    s["camera_id"], s["state"],
                    s["total_frames"], s["anomaly_count"], s["current_fps"],
                )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    mgr.shutdown()
    logger.info("=== Demo complete ===")


if __name__ == "__main__":
    main()
