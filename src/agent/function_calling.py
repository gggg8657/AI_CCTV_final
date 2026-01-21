"""
Function calling registry for Qwen3/OpenAI-compatible tools.
"""

from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.e2e_system import E2ESystem


class FunctionRegistry:
    """Register and execute functions with OpenAI-compatible schemas."""

    def __init__(self) -> None:
        self._functions: Dict[str, Callable[..., Any]] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable[..., Any],
    ) -> None:
        if name in self._functions:
            raise ValueError(f"Function already registered: {name}")

        self._functions[name] = func
        self._schemas[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }

    def call(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        if name not in self._functions:
            raise KeyError(f"Unknown function: {name}")

        args = arguments or {}
        return self._functions[name](**args)

    def list_functions(self) -> List[Dict[str, Any]]:
        return [
            {"type": "function", "function": schema}
            for schema in self._schemas.values()
        ]

    def get_function_schema(self, name: str) -> Dict[str, Any]:
        if name not in self._schemas:
            raise KeyError(f"Unknown function schema: {name}")
        return self._schemas[name]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, set):
        return [_to_jsonable(item) for item in value]
    return str(value)


def _event_to_dict(event: Any) -> Dict[str, Any]:
    if is_dataclass(event):
        data = asdict(event)
    else:
        data = dict(getattr(event, "__dict__", {}))
    if "frame" in data:
        data["frame"] = None
    jsonable = _to_jsonable(data)
    return jsonable if isinstance(jsonable, dict) else {"value": jsonable}


def get_system_status(e2e_system: "E2ESystem") -> Dict[str, Any]:
    try:
        if e2e_system is None:
            return {"ok": False, "error": "E2ESystem instance is required"}

        stats = {}
        if getattr(e2e_system, "stats", None) is not None:
            stats = e2e_system.stats.to_dict()

        config = {}
        if getattr(e2e_system, "config", None) is not None:
            config = _to_jsonable(asdict(e2e_system.config))

        return {
            "ok": True,
            "data": {
                "is_running": bool(getattr(e2e_system, "is_running", False)),
                "stats": stats,
                "config": config,
            },
        }
    except Exception as exc:
        return {"ok": False, "error": f"Failed to get system status: {exc}"}


def get_recent_events(e2e_system: "E2ESystem", limit: int = 20) -> Dict[str, Any]:
    try:
        if e2e_system is None:
            return {"ok": False, "error": "E2ESystem instance is required"}

        if not isinstance(limit, int):
            return {"ok": False, "error": "limit must be an integer"}
        if limit <= 0:
            return {"ok": False, "error": "limit must be greater than 0"}

        event_bus = getattr(e2e_system, "event_bus", None)
        if event_bus is None:
            return {"ok": False, "error": "EventBus is not available"}

        capped_limit = min(limit, 1000)
        events = event_bus.get_history(limit=capped_limit)
        payload = [_event_to_dict(event) for event in events]

        return {
            "ok": True,
            "data": {
                "events": payload,
                "count": len(payload),
                "limit": capped_limit,
            },
        }
    except Exception as exc:
        return {"ok": False, "error": f"Failed to get recent events: {exc}"}


def get_anomaly_statistics(e2e_system: "E2ESystem") -> Dict[str, Any]:
    try:
        if e2e_system is None:
            return {"ok": False, "error": "E2ESystem instance is required"}

        event_bus = getattr(e2e_system, "event_bus", None)
        if event_bus is None:
            return {"ok": False, "error": "EventBus is not available"}

        events = event_bus.get_history(limit=1000)
        counts: Dict[str, int] = {}
        anomaly_scores: List[float] = []
        thresholds: List[float] = []
        latest_anomaly_ts: Optional[str] = None

        for event in events:
            event_type = getattr(event, "event_type", type(event).__name__)
            counts[event_type] = counts.get(event_type, 0) + 1

            if event_type == "AnomalyDetectedEvent":
                score = getattr(event, "score", None)
                threshold = getattr(event, "threshold", None)
                timestamp = getattr(event, "timestamp", None)
                if isinstance(score, (int, float)):
                    anomaly_scores.append(float(score))
                if isinstance(threshold, (int, float)):
                    thresholds.append(float(threshold))
                if isinstance(timestamp, str):
                    latest_anomaly_ts = timestamp

        anomaly_count = counts.get("AnomalyDetectedEvent", 0)
        avg_score = sum(anomaly_scores) / len(anomaly_scores) if anomaly_scores else 0.0
        avg_threshold = sum(thresholds) / len(thresholds) if thresholds else 0.0

        return {
            "ok": True,
            "data": {
                "total_events": len(events),
                "anomaly_count": anomaly_count,
                "avg_anomaly_score": round(avg_score, 4),
                "avg_anomaly_threshold": round(avg_threshold, 4),
                "latest_anomaly_timestamp": latest_anomaly_ts,
                "events_by_type": counts,
            },
        }
    except Exception as exc:
        return {"ok": False, "error": f"Failed to get anomaly statistics: {exc}"}


def update_vad_threshold(e2e_system: "E2ESystem", value: float) -> Dict[str, Any]:
    try:
        if e2e_system is None:
            return {"ok": False, "error": "E2ESystem instance is required"}
        if not isinstance(value, (int, float)):
            return {"ok": False, "error": "value must be a number"}

        config = getattr(e2e_system, "config", None)
        if config is None:
            return {"ok": False, "error": "System config is not available"}

        previous_value = getattr(config, "vad_threshold", None)
        config.vad_threshold = float(value)

        return {
            "ok": True,
            "data": {
                "previous_vad_threshold": previous_value,
                "vad_threshold": config.vad_threshold,
            },
        }
    except Exception as exc:
        return {"ok": False, "error": f"Failed to update VAD threshold: {exc}"}


def enable_vlm(e2e_system: "E2ESystem", enabled: bool) -> Dict[str, Any]:
    try:
        if e2e_system is None:
            return {"ok": False, "error": "E2ESystem instance is required"}
        if not isinstance(enabled, bool):
            return {"ok": False, "error": "enabled must be a boolean"}

        config = getattr(e2e_system, "config", None)
        if config is None:
            return {"ok": False, "error": "System config is not available"}

        previous_value = getattr(config, "enable_vlm", None)
        config.enable_vlm = enabled

        return {
            "ok": True,
            "data": {
                "previous_enable_vlm": previous_value,
                "enable_vlm": config.enable_vlm,
            },
        }
    except Exception as exc:
        return {"ok": False, "error": f"Failed to update VLM setting: {exc}"}


def register_core_functions(
    registry: FunctionRegistry,
    e2e_system: "E2ESystem",
) -> None:
    registry.register(
        name="get_system_status",
        description="Get current system status and configuration.",
        parameters={"type": "object", "properties": {}, "required": []},
        func=partial(get_system_status, e2e_system),
    )
    registry.register(
        name="get_recent_events",
        description="Get recent events from the EventBus.",
        parameters={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of events to return.",
                    "default": 20,
                    "minimum": 1,
                }
            },
            "required": [],
        },
        func=partial(get_recent_events, e2e_system),
    )
    registry.register(
        name="get_anomaly_statistics",
        description="Calculate anomaly statistics from recent events.",
        parameters={"type": "object", "properties": {}, "required": []},
        func=partial(get_anomaly_statistics, e2e_system),
    )
    registry.register(
        name="update_vad_threshold",
        description="Update the VAD anomaly threshold.",
        parameters={
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "New VAD threshold value.",
                }
            },
            "required": ["value"],
        },
        func=partial(update_vad_threshold, e2e_system),
    )
    registry.register(
        name="enable_vlm",
        description="Enable or disable the VLM pipeline.",
        parameters={
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "Whether VLM is enabled.",
                }
            },
            "required": ["enabled"],
        },
        func=partial(enable_vlm, e2e_system),
    )
