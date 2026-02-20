"""
CameraPipeline — 카메라별 독립 처리 파이프라인
================================================

각 카메라에 대해 스레드 1개를 할당하고,
ResourcePool에서 공유 모델을 획득하여 VAD → VLM → Agent 처리를 수행.
"""

import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional, Callable, Dict, Any

from .camera_config import CameraConfig, CameraStatus, PipelineState
from .resource_pool import ResourcePool

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class CameraPipeline:
    """카메라 1대에 대한 독립 파이프라인"""

    def __init__(
        self,
        config: CameraConfig,
        resource_pool: ResourcePool,
        on_anomaly: Optional[Callable] = None,
        on_frame: Optional[Callable] = None,
    ):
        self.config = config
        self.pool = resource_pool
        self.on_anomaly = on_anomaly
        self.on_frame = on_frame

        self.status = CameraStatus(camera_id=config.camera_id)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_buffer: deque = deque(maxlen=30)

    # ── Lifecycle ──

    def start(self) -> bool:
        if self.status.state == PipelineState.RUNNING:
            return True
        self.status.state = PipelineState.STARTING
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_safe,
            name=f"cam-{self.config.camera_id}",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self.status.state = PipelineState.IDLE

    def get_status(self) -> CameraStatus:
        return self.status

    # ── Main Loop ──

    def _run_safe(self) -> None:
        try:
            self._process_loop()
        except Exception as exc:
            logger.error("CameraPipeline %d error: %s", self.config.camera_id, exc, exc_info=True)
            self.status.state = PipelineState.ERROR
            self.status.error_message = str(exc)
        finally:
            if self.status.state != PipelineState.ERROR:
                self.status.state = PipelineState.IDLE

    def _open_video_source(self):
        """비디오 소스 열기 — dummy/file/rtsp 지원"""
        if self.config.source_type == "dummy":
            from ..dummy.video import DummyVideoSource
            src = DummyVideoSource(total_frames=self.config.__dict__.get("total_frames", 0))
            src.open()
            return src

        if not HAS_CV2:
            raise RuntimeError("OpenCV (cv2) is required for non-dummy sources")

        cap = cv2.VideoCapture(self.config.source_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {self.config.source_path}")
        return cap

    def _process_loop(self) -> None:
        source = self._open_video_source()
        is_cv2_cap = not hasattr(source, "get_info")

        self.status.state = PipelineState.RUNNING
        self.status.started_at = datetime.now()
        logger.info("CameraPipeline %d started: %s", self.config.camera_id, self.config.source_path)

        vad_model = self.pool.get_vad_model(self.config.vad_model)

        fps_counter = 0
        fps_start = time.time()

        try:
            while not self._stop_event.is_set():
                loop_start = time.time()

                ret, frame = source.read()
                if not ret:
                    if self.config.source_type in ("file", "dummy"):
                        break
                    time.sleep(0.05)
                    continue

                self.status.total_frames += 1
                fps_counter += 1
                self.status.last_frame_at = datetime.now()
                self._frame_buffer.append(frame)

                score = None
                if hasattr(vad_model, "process_frame"):
                    score = vad_model.process_frame(frame)

                if self.on_frame:
                    self.on_frame(self.config.camera_id, frame, score or 0.0)

                if score is not None and score >= self.config.vad_threshold:
                    self.status.anomaly_count += 1
                    self._handle_anomaly(frame, score)

                now = time.time()
                if now - fps_start >= 1.0:
                    self.status.current_fps = fps_counter
                    fps_counter = 0
                    fps_start = now

                elapsed = time.time() - loop_start
                target = 1.0 / self.config.target_fps
                if elapsed < target:
                    time.sleep(target - elapsed)
        finally:
            if is_cv2_cap:
                source.release()
            else:
                source.close()
            logger.info("CameraPipeline %d stopped", self.config.camera_id)

    def _handle_anomaly(self, frame: Any, vad_score: float) -> None:
        anomaly_data: Dict[str, Any] = {
            "camera_id": self.config.camera_id,
            "timestamp": datetime.now().isoformat(),
            "frame_number": self.status.total_frames,
            "vad_score": vad_score,
            "threshold": self.config.vad_threshold,
        }

        if self.config.enable_vlm:
            try:
                vlm = self.pool.get_vlm_analyzer()
                if vlm and hasattr(vlm, "is_initialized") and vlm.is_initialized:
                    frames = list(self._frame_buffer)
                    result = vlm.analyze(frames=frames)
                    anomaly_data["vlm_type"] = getattr(result, "detected_type", "Unknown")
                    anomaly_data["vlm_description"] = getattr(result, "description", "")
            except Exception as exc:
                logger.warning("VLM analysis failed for cam %d: %s", self.config.camera_id, exc)

        if self.on_anomaly:
            self.on_anomaly(anomaly_data)
