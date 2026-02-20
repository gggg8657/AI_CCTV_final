"""
ResourcePool — 모델 리소스 공유 풀
===================================

VAD/VLM/Agent 모델을 카메라 파이프라인 간 공유하여 GPU 메모리 절약.
Double-check locking 패턴으로 스레드 안전 보장.

use_dummy=True 이면 실제 모델 대신 Dummy 구현을 사용하여
모델 파일 없이도 전체 파이프라인을 실행할 수 있음.
"""

import logging
import threading
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ResourcePool:
    """스레드 안전한 모델 리소스 풀"""

    def __init__(self, gpu_id: int = 0, use_dummy: bool = False):
        self.gpu_id = gpu_id
        self.use_dummy = use_dummy
        self._vad_models: Dict[str, Any] = {}
        self._vlm_analyzer: Optional[Any] = None
        self._agent_flows: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._closed = False

    # ── VAD ──

    def get_vad_model(self, model_type: str) -> Any:
        key = model_type if not self.use_dummy else f"dummy:{model_type}"
        if key in self._vad_models:
            return self._vad_models[key]
        with self._lock:
            if key not in self._vad_models:
                self._vad_models[key] = self._create_vad_model(model_type)
            return self._vad_models[key]

    def _create_vad_model(self, model_type: str) -> Any:
        if self.use_dummy:
            from ..dummy.vad import DummyVADModel
            model = DummyVADModel()
            model.initialize("cpu")
            logger.info("ResourcePool: Dummy VAD model loaded (type=%s)", model_type)
            return model

        from ..vad import create_model as create_vad_model
        model = create_vad_model(model_type)
        device_str = f"cuda:{self.gpu_id}" if HAS_TORCH and torch.cuda.is_available() else "cpu"
        if hasattr(model, "initialize"):
            model.initialize(device_str)
        logger.info("ResourcePool: VAD model '%s' loaded on %s", model_type, device_str)
        return model

    # ── VLM ──

    def get_vlm_analyzer(self, **kwargs: Any) -> Any:
        if self._vlm_analyzer is not None:
            return self._vlm_analyzer
        with self._lock:
            if self._vlm_analyzer is None:
                self._vlm_analyzer = self._create_vlm_analyzer(**kwargs)
            return self._vlm_analyzer

    def _create_vlm_analyzer(self, **kwargs: Any) -> Any:
        if self.use_dummy:
            from ..dummy.vlm import DummyVLMAnalyzer
            analyzer = DummyVLMAnalyzer(**kwargs)
            analyzer.initialize()
            logger.info("ResourcePool: Dummy VLM analyzer loaded")
            return analyzer

        from ..vlm import VLMAnalyzer
        analyzer = VLMAnalyzer(
            gpu_id=self.gpu_id,
            optimize_speed=kwargs.get("optimize_speed", True),
            **{k: v for k, v in kwargs.items() if k != "optimize_speed"},
        )
        if analyzer.initialize():
            logger.info("ResourcePool: VLM analyzer loaded")
        else:
            logger.warning("ResourcePool: VLM analyzer initialization failed")
        return analyzer

    # ── Agent ──

    def get_agent_flow(self, flow_type: str, **kwargs: Any) -> Any:
        key = flow_type if not self.use_dummy else f"dummy:{flow_type}"
        if key in self._agent_flows:
            return self._agent_flows[key]
        with self._lock:
            if key not in self._agent_flows:
                self._agent_flows[key] = self._create_agent_flow(flow_type, **kwargs)
            return self._agent_flows[key]

    def _create_agent_flow(self, flow_type: str, **kwargs: Any) -> Any:
        if self.use_dummy:
            from ..dummy.agent import DummyAgentFlow
            flow = DummyAgentFlow(flow_type=flow_type)
            flow.initialize()
            logger.info("ResourcePool: Dummy Agent flow '%s' loaded", flow_type)
            return flow

        from ..agent import create_flow
        flow = create_flow(flow_type, gpu_id=self.gpu_id, **kwargs)
        if hasattr(flow, "initialize"):
            flow.initialize()
        logger.info("ResourcePool: Agent flow '%s' loaded", flow_type)
        return flow

    # ── GPU Memory ──

    def gpu_memory_info(self) -> Dict[str, float]:
        if not HAS_TORCH or not torch.cuda.is_available():
            return {}
        return {
            "allocated_mb": torch.cuda.memory_allocated(self.gpu_id) / 1024 ** 2,
            "reserved_mb": torch.cuda.memory_reserved(self.gpu_id) / 1024 ** 2,
            "max_allocated_mb": torch.cuda.max_memory_allocated(self.gpu_id) / 1024 ** 2,
        }

    # ── Lifecycle ──

    @property
    def loaded_models(self) -> Dict[str, int]:
        return {
            "vad": len(self._vad_models),
            "vlm": 1 if self._vlm_analyzer else 0,
            "agent": len(self._agent_flows),
        }

    def close(self) -> None:
        with self._lock:
            self._vad_models.clear()
            self._vlm_analyzer = None
            self._agent_flows.clear()
            self._closed = True
            logger.info("ResourcePool: all resources released")
