"""
이벤트 버스 시스템
==================

asyncio.Queue 기반 Pub/Sub 이벤트 버스
"""

import asyncio
import threading
from collections import defaultdict, deque
from typing import Dict, List, Callable, Type, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class BaseEvent:
    """이벤트 기본 클래스"""
    event_id: str
    event_type: str
    timestamp: str
    source: str


class EventBus:
    """
    중앙 집중식 이벤트 버스
    
    asyncio.Queue를 사용한 비동기 이벤트 처리
    Thread-safe 구현 (threading.Lock 사용)
    """
    
    def __init__(self, max_history: int = 1000):
        """
        EventBus 초기화
        
        Args:
            max_history: 이벤트 히스토리 최대 개수
        """
        # 구독자 관리 (이벤트 타입별 핸들러 리스트)
        self._subscribers: Dict[Type[BaseEvent], List[Callable]] = defaultdict(list)
        
        # 이벤트 히스토리
        self._event_history: deque = deque(maxlen=max_history)
        
        # Thread-safe를 위한 Lock
        self._lock = threading.Lock()
        
        # asyncio.Queue (이벤트 처리용)
        self._queue: Optional[asyncio.Queue] = None
        
        # 이벤트 처리 태스크
        self._processing_task: Optional[asyncio.Task] = None
        
        # 실행 중 플래그
        self._running = False
    
    def subscribe(self, event_type: Type[BaseEvent], handler: Callable):
        """
        이벤트 구독
        
        Args:
            event_type: 이벤트 타입 (클래스)
            handler: 핸들러 함수 (async 또는 sync)
        """
        with self._lock:
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type: Type[BaseEvent], handler: Callable):
        """
        이벤트 구독 해제
        
        Args:
            event_type: 이벤트 타입 (클래스)
            handler: 핸들러 함수
        """
        with self._lock:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
    
    async def publish(self, event: BaseEvent):
        """
        이벤트 발행 (비동기)
        
        Args:
            event: 이벤트 객체
        """
        # 이벤트 히스토리에 추가
        with self._lock:
            self._event_history.append(event)
        
        # Queue에 이벤트 추가
        if self._queue is None:
            # Queue가 초기화되지 않았으면 동기적으로 처리
            await self._process_event_sync(event)
        else:
            await self._queue.put(event)
    
    def publish_sync(self, event: BaseEvent):
        """
        이벤트 발행 (동기, 비동기 루프가 없을 때 사용)
        
        Args:
            event: 이벤트 객체
        """
        # 이벤트 히스토리에 추가
        with self._lock:
            self._event_history.append(event)
        
        # 구독자들에게 직접 전달 (동기)
        with self._lock:
            handlers = self._subscribers[type(event)].copy()
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    # 비동기 핸들러는 새 이벤트 루프에서 실행
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(handler(event))
                    loop.close()
                else:
                    # 동기 핸들러는 직접 실행
                    handler(event)
            except Exception as e:
                # 핸들러 에러는 로깅만 하고 계속 진행
                import logging
                logging.error(f"Event handler error: {e}", exc_info=True)
    
    async def _process_event_sync(self, event: BaseEvent):
        """이벤트를 동기적으로 처리 (Queue 없을 때)"""
        # 구독자 목록 가져오기
        with self._lock:
            handlers = self._subscribers[type(event)].copy()
        
        # 각 핸들러 실행
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    # 동기 핸들러는 run_in_executor로 실행
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, handler, event)
            except Exception as e:
                # 핸들러 에러는 로깅만 하고 계속 진행
                import logging
                logging.error(f"Event handler error: {e}", exc_info=True)
    
    async def _process_queue(self):
        """이벤트 큐 처리 루프"""
        while self._running:
            try:
                # Queue에서 이벤트 가져오기
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                
                # 구독자 목록 가져오기
                with self._lock:
                    handlers = self._subscribers[type(event)].copy()
                
                # 각 핸들러 실행
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            # 동기 핸들러는 run_in_executor로 실행
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, handler, event)
                    except Exception as e:
                        # 핸들러 에러는 로깅만 하고 계속 진행
                        import logging
                        logging.error(f"Event handler error: {e}", exc_info=True)
                
                # 작업 완료 표시
                self._queue.task_done()
                
            except asyncio.TimeoutError:
                # 타임아웃은 정상 (루프 유지)
                continue
            except Exception as e:
                import logging
                logging.error(f"Event processing error: {e}", exc_info=True)
    
    def start(self):
        """이벤트 버스 시작 (비동기 루프 시작)"""
        if self._running:
            return
        
        self._running = True
        self._queue = asyncio.Queue()
        
        # 이벤트 처리 태스크 시작
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        self._processing_task = loop.create_task(self._process_queue())
    
    def stop(self):
        """이벤트 버스 중지"""
        self._running = False
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 실행 중인 루프에서는 취소만
                    pass
                else:
                    try:
                        loop.run_until_complete(self._processing_task)
                    except (asyncio.CancelledError, RuntimeError):
                        # CancelledError는 정상적인 취소 동작
                        # RuntimeError는 이미 완료된 태스크일 수 있음
                        pass
            except (RuntimeError, Exception):
                # 루프가 없거나 이미 닫힌 경우
                pass
        
        self._queue = None
        self._processing_task = None
    
    def get_history(self, limit: int = 100) -> List[BaseEvent]:
        """
        이벤트 히스토리 조회
        
        Args:
            limit: 조회할 최대 개수
        
        Returns:
            최근 이벤트 리스트
        """
        with self._lock:
            return list(self._event_history)[-limit:]
    
    def get_subscriber_count(self, event_type: Type[BaseEvent]) -> int:
        """
        특정 이벤트 타입의 구독자 수 조회
        
        Args:
            event_type: 이벤트 타입
        
        Returns:
            구독자 수
        """
        with self._lock:
            return len(self._subscribers[event_type])
    
    def clear_history(self):
        """이벤트 히스토리 초기화"""
        with self._lock:
            self._event_history.clear()
    
    def clear_subscribers(self, event_type: Optional[Type[BaseEvent]] = None):
        """
        구독자 초기화
        
        Args:
            event_type: 이벤트 타입 (None이면 모든 구독자 제거)
        """
        with self._lock:
            if event_type is None:
                self._subscribers.clear()
            else:
                self._subscribers[event_type].clear()
