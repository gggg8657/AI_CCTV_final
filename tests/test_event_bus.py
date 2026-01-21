import asyncio
import threading
from dataclasses import dataclass

import pytest

from src.utils.event_bus import BaseEvent, EventBus


@dataclass
class TestEvent(BaseEvent):
    payload: str = ""


@dataclass
class AnotherTestEvent(BaseEvent):
    value: int = 0


def make_event(index: int = 0) -> TestEvent:
    return TestEvent(
        event_id=f"id-{index}",
        event_type="TestEvent",
        timestamp="now",
        source="test",
        payload=f"payload-{index}",
    )


def make_another_event(index: int = 0) -> AnotherTestEvent:
    return AnotherTestEvent(
        event_id=f"another-{index}",
        event_type="AnotherTestEvent",
        timestamp="now",
        source="test",
        value=index,
    )


def test_initialization_and_empty_history():
    bus = EventBus(max_history=5)

    assert bus.get_history() == []
    assert bus.get_history(limit=2) == []
    assert bus.get_subscriber_count(TestEvent) == 0


def test_subscribe_unsubscribe_and_clear():
    bus = EventBus()
    seen = []

    def handler(event: BaseEvent):
        seen.append(event.event_id)

    def handler_two(event: BaseEvent):
        seen.append(f"two-{event.event_id}")

    bus.subscribe(TestEvent, handler)
    bus.subscribe(TestEvent, handler)
    assert bus.get_subscriber_count(TestEvent) == 1

    bus.subscribe(AnotherTestEvent, handler_two)
    assert bus.get_subscriber_count(AnotherTestEvent) == 1

    bus.unsubscribe(TestEvent, handler)
    assert bus.get_subscriber_count(TestEvent) == 0

    bus.subscribe(TestEvent, handler)
    bus.clear_subscribers(TestEvent)
    assert bus.get_subscriber_count(TestEvent) == 0
    assert bus.get_subscriber_count(AnotherTestEvent) == 1

    bus.clear_subscribers()
    assert bus.get_subscriber_count(AnotherTestEvent) == 0


@pytest.mark.asyncio
async def test_publish_async_without_start_runs_handlers():
    bus = EventBus()
    sync_called = threading.Event()
    async_called = asyncio.Event()

    def sync_handler(event: BaseEvent):
        sync_called.set()

    async def async_handler(event: BaseEvent):
        async_called.set()

    bus.subscribe(TestEvent, sync_handler)
    bus.subscribe(TestEvent, async_handler)

    event = make_event(1)
    await bus.publish(event)

    await asyncio.wait_for(async_called.wait(), timeout=1)
    assert sync_called.is_set()
    assert bus.get_history(limit=1)[0].event_id == event.event_id


def test_publish_sync_with_sync_and_async_handlers():
    bus = EventBus()
    sync_called = threading.Event()
    async_called = threading.Event()

    def sync_handler(event: BaseEvent):
        sync_called.set()

    async def async_handler(event: BaseEvent):
        async_called.set()

    bus.subscribe(TestEvent, sync_handler)
    bus.subscribe(TestEvent, async_handler)

    bus.publish_sync(make_event(2))

    assert sync_called.is_set()
    assert async_called.is_set()


@pytest.mark.asyncio
async def test_start_stop_and_publish_via_queue():
    bus = EventBus()
    async_called = asyncio.Event()

    async def async_handler(event: BaseEvent):
        async_called.set()

    bus.subscribe(TestEvent, async_handler)
    bus.start()

    await bus.publish(make_event(3))
    await asyncio.wait_for(async_called.wait(), timeout=1)

    bus.stop()

    assert bus._queue is None
    assert bus._processing_task is None
    assert bus._running is False


def test_event_history_management_and_limits():
    bus = EventBus(max_history=3)

    for idx in range(5):
        bus.publish_sync(make_event(idx))

    history = bus.get_history()
    assert len(history) == 3
    assert [event.event_id for event in history] == ["id-2", "id-3", "id-4"]

    limited = bus.get_history(limit=2)
    assert [event.event_id for event in limited] == ["id-3", "id-4"]

    bus.clear_history()
    assert bus.get_history() == []


def test_error_handling_does_not_stop_other_handlers():
    bus = EventBus()
    good_called = threading.Event()

    def bad_handler(event: BaseEvent):
        raise RuntimeError("boom")

    def good_handler(event: BaseEvent):
        good_called.set()

    bus.subscribe(TestEvent, bad_handler)
    bus.subscribe(TestEvent, good_handler)

    bus.publish_sync(make_event(4))
    assert good_called.is_set()


def test_thread_safety_with_concurrent_publish_sync():
    bus = EventBus(max_history=300)
    counter_lock = threading.Lock()
    total_calls = 0

    def handler(event: BaseEvent):
        nonlocal total_calls
        with counter_lock:
            total_calls += 1

    bus.subscribe(TestEvent, handler)

    threads = []
    events_per_thread = 50
    thread_count = 5

    def publish_events(thread_index: int):
        base = thread_index * events_per_thread
        for idx in range(events_per_thread):
            bus.publish_sync(make_event(base + idx))

    for t_idx in range(thread_count):
        thread = threading.Thread(target=publish_events, args=(t_idx,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    assert total_calls == thread_count * events_per_thread
    assert len(bus.get_history(limit=300)) == thread_count * events_per_thread
