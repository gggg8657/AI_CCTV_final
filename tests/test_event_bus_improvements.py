"""
EventBus 개선 사항 테스트
- 비동기 처리 개선
- BaseEvent 기본값 테스트
"""

import pytest
import asyncio
import threading
import time
from src.utils.event_bus import EventBus, BaseEvent
from src.utils.events import AnomalyDetectedEvent


def test_base_event_default_values():
    """BaseEvent 기본값 테스트"""
    # 기본값으로 생성 가능해야 함
    event = BaseEvent()
    assert event.event_id == ""
    assert event.event_type == ""
    assert event.timestamp == ""
    assert event.source == ""
    
    # 일부 필드만 설정 가능
    event = BaseEvent(event_id="test_1", event_type="TestEvent")
    assert event.event_id == "test_1"
    assert event.event_type == "TestEvent"
    assert event.timestamp == ""
    assert event.source == ""


def test_event_bus_start_without_loop():
    """이벤트 루프 없이 EventBus 시작 테스트"""
    bus = EventBus()
    
    # 이벤트 루프가 없는 상태에서 시작
    bus.start()
    assert bus._running is True
    
    # 정상적으로 중지
    bus.stop()
    assert bus._running is False


def test_event_bus_start_with_existing_loop():
    """기존 이벤트 루프가 있을 때 EventBus 시작 테스트"""
    bus = EventBus()
    
    # 새 이벤트 루프 생성
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        bus.start()
        assert bus._running is True
        assert bus._queue is not None
    finally:
        bus.stop()
        # 루프가 실행 중이 아니면 닫기
        if not loop.is_running():
            try:
                loop.close()
            except RuntimeError:
                # 이미 닫혔거나 실행 중인 경우 무시
                pass


def test_event_bus_stop_with_pending_events():
    """대기 중인 이벤트가 있을 때 EventBus 중지 테스트"""
    bus = EventBus()
    handler_called = []
    
    def handler(event):
        handler_called.append(event)
        time.sleep(0.1)  # 처리 시간 시뮬레이션
    
    bus.subscribe(BaseEvent, handler)
    bus.start()
    
    try:
        # 여러 이벤트 발행
        for i in range(5):
            event = BaseEvent(
                event_id=f"test_{i}",
                event_type="TestEvent",
                timestamp="2026-01-01T00:00:00",
                source="test"
            )
            asyncio.run(bus.publish(event))
        
        # 즉시 중지 (대기 중인 이벤트 처리 대기)
        bus.stop()
        
        # 중지 후 상태 확인
        assert bus._running is False
        assert bus._queue is None
    finally:
        if bus._running:
            bus.stop()


@pytest.mark.asyncio
async def test_event_bus_async_handler_error_handling():
    """비동기 핸들러 에러 처리 테스트"""
    bus = EventBus()
    handler_called = []
    
    async def error_handler(event):
        handler_called.append(event)
        raise ValueError("Test error")
    
    async def normal_handler(event):
        handler_called.append(event)
    
    bus.subscribe(BaseEvent, error_handler)
    bus.subscribe(BaseEvent, normal_handler)
    bus.start()
    
    try:
        event = BaseEvent(
            event_id="test_error",
            event_type="TestEvent",
            timestamp="2026-01-01T00:00:00",
            source="test"
        )
        
        # 에러가 발생해도 다른 핸들러는 실행되어야 함
        await bus.publish(event)
        await asyncio.sleep(0.1)
        
        # 두 핸들러 모두 호출되어야 함 (에러 핸들러는 실패하지만 로깅만)
        assert len(handler_called) == 2
    finally:
        bus.stop()


def test_event_bus_sync_handler_error_handling():
    """동기 핸들러 에러 처리 테스트"""
    bus = EventBus()
    handler_called = []
    
    def error_handler(event):
        handler_called.append(event)
        raise ValueError("Test error")
    
    def normal_handler(event):
        handler_called.append(event)
    
    bus.subscribe(BaseEvent, error_handler)
    bus.subscribe(BaseEvent, normal_handler)
    
    event = BaseEvent(
        event_id="test_error",
        event_type="TestEvent",
        timestamp="2026-01-01T00:00:00",
        source="test"
    )
    
    # 에러가 발생해도 다른 핸들러는 실행되어야 함
    bus.publish_sync(event)
    
    # 두 핸들러 모두 호출되어야 함
    assert len(handler_called) == 2


def test_anomaly_event_post_init():
    """AnomalyDetectedEvent __post_init__ 테스트"""
    # 필수 필드만 제공
    event = AnomalyDetectedEvent(
        frame_id=1,
        score=0.8,
        threshold=0.5
    )
    
    # __post_init__에서 자동 설정된 필드 확인
    assert event.event_id != ""
    assert event.event_type == "AnomalyDetectedEvent"
    assert event.timestamp != ""
    assert event.source == "VAD"
    
    # 모든 필드 제공 시에도 정상 작동
    event2 = AnomalyDetectedEvent(
        event_id="custom_id",
        event_type="CustomType",
        timestamp="2026-01-01T00:00:00",
        source="CustomSource",
        frame_id=2,
        score=0.9,
        threshold=0.6
    )
    
    assert event2.event_id == "custom_id"
    assert event2.event_type == "CustomType"
    assert event2.timestamp == "2026-01-01T00:00:00"
    assert event2.source == "CustomSource"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
