"""
EventBus 모듈 테스트
"""

import pytest
import asyncio
import threading
import time
from src.utils.event_bus import EventBus, BaseEvent
from src.utils.events import AnomalyDetectedEvent


class TestEvent(BaseEvent):
    """테스트용 이벤트"""
    pass


def test_event_bus_initialization():
    """EventBus 초기화 테스트"""
    bus = EventBus()
    assert bus is not None
    assert bus._running is False
    assert bus._queue is None


def test_event_bus_subscribe():
    """이벤트 구독 테스트"""
    bus = EventBus()
    handler_called = []
    
    def handler(event):
        handler_called.append(event)
    
    bus.subscribe(TestEvent, handler)
    assert bus.get_subscriber_count(TestEvent) == 1
    
    # 중복 구독 방지
    bus.subscribe(TestEvent, handler)
    assert bus.get_subscriber_count(TestEvent) == 1


def test_event_bus_unsubscribe():
    """이벤트 구독 해제 테스트"""
    bus = EventBus()
    handler_called = []
    
    def handler(event):
        handler_called.append(event)
    
    bus.subscribe(TestEvent, handler)
    assert bus.get_subscriber_count(TestEvent) == 1
    
    bus.unsubscribe(TestEvent, handler)
    assert bus.get_subscriber_count(TestEvent) == 0


def test_event_bus_publish_sync():
    """동기 이벤트 발행 테스트"""
    bus = EventBus()
    handler_called = []
    
    def handler(event):
        handler_called.append(event)
    
    bus.subscribe(TestEvent, handler)
    
    event = TestEvent(
        event_id="test_1",
        event_type="TestEvent",
        timestamp="2026-01-01T00:00:00",
        source="test"
    )
    
    bus.publish_sync(event)
    assert len(handler_called) == 1
    assert handler_called[0] == event


def test_event_bus_history():
    """이벤트 히스토리 테스트"""
    bus = EventBus(max_history=10)
    
    for i in range(5):
        event = TestEvent(
            event_id=f"test_{i}",
            event_type="TestEvent",
            timestamp="2026-01-01T00:00:00",
            source="test"
        )
        bus.publish_sync(event)
    
    history = bus.get_history(limit=100)
    assert len(history) == 5
    
    # 히스토리 초기화
    bus.clear_history()
    history = bus.get_history()
    assert len(history) == 0


@pytest.mark.asyncio
async def test_event_bus_async_publish():
    """비동기 이벤트 발행 테스트"""
    bus = EventBus()
    handler_called = []
    
    async def async_handler(event):
        handler_called.append(event)
        await asyncio.sleep(0.01)
    
    bus.subscribe(TestEvent, async_handler)
    bus.start()
    
    try:
        event = TestEvent(
            event_id="test_async",
            event_type="TestEvent",
            timestamp="2026-01-01T00:00:00",
            source="test"
        )
        
        await bus.publish(event)
        
        # 이벤트 처리 대기
        await asyncio.sleep(0.1)
        
        assert len(handler_called) == 1
        assert handler_called[0] == event
    finally:
        bus.stop()


def test_event_bus_start_stop():
    """EventBus 시작/중지 테스트"""
    bus = EventBus()
    assert bus._running is False
    
    bus.start()
    assert bus._running is True
    assert bus._queue is not None
    
    bus.stop()
    assert bus._running is False
    assert bus._queue is None


def test_anomaly_detected_event():
    """AnomalyDetectedEvent 생성 테스트"""
    event = AnomalyDetectedEvent(
        frame_id=1,
        score=0.8,
        threshold=0.5
    )
    
    assert event.frame_id == 1
    assert event.score == 0.8
    assert event.threshold == 0.5
    assert event.event_id != ""
    assert event.event_type == "AnomalyDetectedEvent"
    assert event.timestamp != ""
    assert event.source == "VAD"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
