"""
Event handler classes for VAD, VLM, and Agent components.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Generic, Type, TypeVar

from .event_bus import BaseEvent, EventBus
from .events import (
    AnomalyDetectedEvent,
    VLMAnalysisCompletedEvent,
    AgentResponseEvent,
)


TEvent = TypeVar("TEvent", bound=BaseEvent)


@dataclass
class HandlerStats:
    """Common handler statistics."""
    total_events: int = 0
    error_count: int = 0
    last_event_id: str = ""
    last_event_type: str = ""
    last_timestamp: str = ""
    last_source: str = ""


@dataclass
class BaseEventHandler(Generic[TEvent]):
    """Base event handler with sync/async support and thread-safe stats."""
    EVENT_TYPE: ClassVar[Type[BaseEvent]] = BaseEvent

    event_bus: EventBus
    use_async: bool = False
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    stats: HandlerStats = field(default_factory=HandlerStats, init=False)

    def __post_init__(self) -> None:
        self.subscribe()

    def subscribe(self) -> None:
        handler = self.handle_async if self.use_async else self.handle
        self.event_bus.subscribe(self.EVENT_TYPE, handler)
        self.logger.debug(
            "Subscribed %s to %s (async=%s)",
            self.__class__.__name__,
            self.EVENT_TYPE.__name__,
            self.use_async,
        )

    def unsubscribe(self) -> None:
        handler = self.handle_async if self.use_async else self.handle
        self.event_bus.unsubscribe(self.EVENT_TYPE, handler)
        self.logger.debug(
            "Unsubscribed %s from %s (async=%s)",
            self.__class__.__name__,
            self.EVENT_TYPE.__name__,
            self.use_async,
        )

    def publish_sync(self, event: BaseEvent) -> None:
        """Thread-safe synchronous publish helper."""
        self.event_bus.publish_sync(event)

    def handle(self, event: TEvent) -> None:
        try:
            self.process_event(event)
            self._update_stats(event)
        except Exception as exc:
            self._handle_error(event, exc)

    async def handle_async(self, event: TEvent) -> None:
        try:
            await self.process_event_async(event)
            self._update_stats(event)
        except Exception as exc:
            self._handle_error(event, exc)

    def process_event(self, event: TEvent) -> None:
        """Sync processing hook. Override in subclasses."""
        self.logger.info("Event received: %s", event.event_id)

    async def process_event_async(self, event: TEvent) -> None:
        """Async processing hook. Defaults to executing sync handler in executor."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.process_event, event)

    def _update_stats(self, event: TEvent) -> None:
        with self._lock:
            self.stats.total_events += 1
            self.stats.last_event_id = event.event_id
            self.stats.last_event_type = event.event_type
            self.stats.last_timestamp = event.timestamp
            self.stats.last_source = event.source
            self._update_specific_stats(event)

    def _update_specific_stats(self, event: TEvent) -> None:
        """Subclass hook for extra stats."""

    def _handle_error(self, event: TEvent, exc: Exception) -> None:
        with self._lock:
            self.stats.error_count += 1
        self.logger.error(
            "Handler error in %s for event %s: %s",
            self.__class__.__name__,
            getattr(event, "event_id", "unknown"),
            exc,
            exc_info=True,
        )

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            base = {
                "total_events": self.stats.total_events,
                "error_count": self.stats.error_count,
                "last_event_id": self.stats.last_event_id,
                "last_event_type": self.stats.last_event_type,
                "last_timestamp": self.stats.last_timestamp,
                "last_source": self.stats.last_source,
            }
        return base


@dataclass
class VADEventHandler(BaseEventHandler[AnomalyDetectedEvent]):
    """Handles AnomalyDetectedEvent from VAD."""
    EVENT_TYPE: ClassVar[Type[BaseEvent]] = AnomalyDetectedEvent

    _score_sum: float = field(default=0.0, init=False, repr=False)
    _last_score: float = field(default=0.0, init=False, repr=False)
    _last_threshold: float = field(default=0.0, init=False, repr=False)

    def process_event(self, event: AnomalyDetectedEvent) -> None:
        self.logger.info(
            "VAD anomaly detected: frame=%s score=%.4f threshold=%.4f",
            event.frame_id,
            event.score,
            event.threshold,
        )

    def _update_specific_stats(self, event: AnomalyDetectedEvent) -> None:
        self._score_sum += event.score
        self._last_score = event.score
        self._last_threshold = event.threshold

    def get_stats(self) -> Dict[str, Any]:
        base = super().get_stats()
        with self._lock:
            avg_score = self._score_sum / self.stats.total_events if self.stats.total_events else 0.0
            base.update(
                {
                    "avg_score": avg_score,
                    "last_score": self._last_score,
                    "last_threshold": self._last_threshold,
                }
            )
        return base


@dataclass
class VLMEventHandler(BaseEventHandler[VLMAnalysisCompletedEvent]):
    """Handles VLMAnalysisCompletedEvent from VLM."""
    EVENT_TYPE: ClassVar[Type[BaseEvent]] = VLMAnalysisCompletedEvent

    _confidence_sum: float = field(default=0.0, init=False, repr=False)
    _last_detected_type: str = field(default="", init=False, repr=False)
    _last_actions_count: int = field(default=0, init=False, repr=False)

    def process_event(self, event: VLMAnalysisCompletedEvent) -> None:
        self.logger.info(
            "VLM analysis completed: event=%s type=%s confidence=%.3f actions=%d",
            event.original_event_id,
            event.detected_type,
            event.confidence,
            len(event.actions),
        )

    def _update_specific_stats(self, event: VLMAnalysisCompletedEvent) -> None:
        self._confidence_sum += event.confidence
        self._last_detected_type = event.detected_type
        self._last_actions_count = len(event.actions)

    def get_stats(self) -> Dict[str, Any]:
        base = super().get_stats()
        with self._lock:
            avg_confidence = (
                self._confidence_sum / self.stats.total_events if self.stats.total_events else 0.0
            )
            base.update(
                {
                    "avg_confidence": avg_confidence,
                    "last_detected_type": self._last_detected_type,
                    "last_actions_count": self._last_actions_count,
                }
            )
        return base


@dataclass
class AgentEventHandler(BaseEventHandler[AgentResponseEvent]):
    """Handles AgentResponseEvent from Agent."""
    EVENT_TYPE: ClassVar[Type[BaseEvent]] = AgentResponseEvent

    _estimated_time_sum: float = field(default=0.0, init=False, repr=False)
    _last_priority: int = field(default=0, init=False, repr=False)
    _last_plan_count: int = field(default=0, init=False, repr=False)

    def process_event(self, event: AgentResponseEvent) -> None:
        self.logger.info(
            "Agent response: event=%s priority=%d plan_items=%d eta=%.2fs",
            event.original_event_id,
            event.priority,
            len(event.plan),
            event.estimated_time,
        )

    def _update_specific_stats(self, event: AgentResponseEvent) -> None:
        self._estimated_time_sum += event.estimated_time
        self._last_priority = event.priority
        self._last_plan_count = len(event.plan)

    def get_stats(self) -> Dict[str, Any]:
        base = super().get_stats()
        with self._lock:
            avg_estimated_time = (
                self._estimated_time_sum / self.stats.total_events if self.stats.total_events else 0.0
            )
            base.update(
                {
                    "avg_estimated_time": avg_estimated_time,
                    "last_priority": self._last_priority,
                    "last_plan_count": self._last_plan_count,
                }
            )
        return base
