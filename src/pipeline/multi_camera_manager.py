"""
MultiCameraManager — 멀티 카메라 관리자
========================================

여러 CameraPipeline 인스턴스의 수명을 관리하고,
ResourcePool을 통해 GPU 리소스를 공유.
"""

import logging
import threading
from typing import Dict, Optional, List, Callable, Any

from .camera_config import CameraConfig, CameraStatus, PipelineState
from .camera_pipeline import CameraPipeline
from .resource_pool import ResourcePool

logger = logging.getLogger(__name__)


class MultiCameraManager:
    """멀티 카메라 파이프라인 관리자"""

    def __init__(
        self,
        max_cameras: int = 16,
        gpu_id: int = 0,
        use_dummy: bool = False,
        use_dummy_vad: Optional[bool] = None,
        use_dummy_vlm: Optional[bool] = None,
        use_dummy_agent: Optional[bool] = None,
    ):
        self.max_cameras = max_cameras
        self.resource_pool = ResourcePool(
            gpu_id=gpu_id,
            use_dummy=use_dummy,
            use_dummy_vad=use_dummy_vad,
            use_dummy_vlm=use_dummy_vlm,
            use_dummy_agent=use_dummy_agent,
        )
        self._pipelines: Dict[int, CameraPipeline] = {}
        self._lock = threading.Lock()
        self._on_anomaly: Optional[Callable] = None
        self._on_frame: Optional[Callable] = None

    # ── Callbacks ──

    def set_anomaly_callback(self, callback: Callable) -> None:
        self._on_anomaly = callback

    def set_frame_callback(self, callback: Callable) -> None:
        self._on_frame = callback

    # ── Camera Lifecycle ──

    def add_camera(self, config: CameraConfig) -> int:
        with self._lock:
            if len(self._pipelines) >= self.max_cameras:
                raise ValueError(f"Maximum {self.max_cameras} cameras reached")
            if config.camera_id in self._pipelines:
                raise ValueError(f"Camera {config.camera_id} already exists")

            pipeline = CameraPipeline(
                config=config,
                resource_pool=self.resource_pool,
                on_anomaly=self._on_anomaly,
                on_frame=self._on_frame,
            )
            self._pipelines[config.camera_id] = pipeline
            logger.info("Camera %d added", config.camera_id)
            return config.camera_id

    def remove_camera(self, camera_id: int) -> bool:
        with self._lock:
            pipeline = self._pipelines.pop(camera_id, None)
        if pipeline is None:
            return False
        pipeline.stop()
        logger.info("Camera %d removed", camera_id)
        return True

    def start_camera(self, camera_id: int) -> bool:
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return False
        return pipeline.start()

    def stop_camera(self, camera_id: int) -> bool:
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return False
        pipeline.stop()
        return True

    # ── Status ──

    def get_camera_status(self, camera_id: int) -> Optional[CameraStatus]:
        pipeline = self._pipelines.get(camera_id)
        return pipeline.get_status() if pipeline else None

    def get_all_statuses(self) -> List[Dict[str, Any]]:
        return [p.get_status().to_dict() for p in self._pipelines.values()]

    @property
    def active_count(self) -> int:
        return sum(
            1 for p in self._pipelines.values()
            if p.status.state == PipelineState.RUNNING
        )

    @property
    def camera_ids(self) -> List[int]:
        return list(self._pipelines.keys())

    # ── Bulk ──

    def start_all(self) -> None:
        for cam_id in list(self._pipelines.keys()):
            self.start_camera(cam_id)

    def stop_all(self) -> None:
        for cam_id in list(self._pipelines.keys()):
            self.stop_camera(cam_id)

    def shutdown(self) -> None:
        self.stop_all()
        self.resource_pool.close()
        logger.info("MultiCameraManager shut down (%d pipelines)", len(self._pipelines))
