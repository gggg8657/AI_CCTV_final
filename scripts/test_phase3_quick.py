#!/usr/bin/env python3
"""
Phase 3 빠른 테스트 (Mock 데이터)
=================================

실제 비디오 없이 Phase 3 컴포넌트만 빠르게 테스트합니다.
"""

import sys
import time
from pathlib import Path

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.package_detection import PackageDetector, PackageTracker, TheftDetector
from src.utils.event_bus import EventBus
from src.utils.events import PackageDetectedEvent, PackageDisappearedEvent, TheftDetectedEvent


def test_package_detection_components():
    """Phase 3 컴포넌트 빠른 테스트"""
    print("=" * 60)
    print("Phase 3 컴포넌트 빠른 테스트")
    print("=" * 60)
    print()
    
    # EventBus 생성
    print("[1/4] EventBus 생성...")
    event_bus = EventBus(max_history=100)
    event_bus.start()
    print("  ✅ EventBus 시작됨")
    print()
    
    # 이벤트 수집
    events_received = []
    
    def collect_event(event):
        events_received.append(event)
        print(f"  📨 이벤트 수신: {event.event_type}")
    
    event_bus.subscribe(PackageDetectedEvent, collect_event)
    event_bus.subscribe(PackageDisappearedEvent, collect_event)
    event_bus.subscribe(TheftDetectedEvent, collect_event)
    
    # PackageTracker 생성
    print("[2/4] PackageTracker 생성...")
    tracker = PackageTracker(
        iou_threshold=0.3,
        max_age=30.0,
        missing_threshold=1.0,
        event_bus=event_bus,
        camera_id=0,
    )
    print("  ✅ PackageTracker 생성됨")
    print()
    
    # TheftDetector 생성
    print("[3/4] TheftDetector 생성...")
    theft_detector = TheftDetector(
        confirmation_time=1.0,  # 테스트용으로 짧게
        evidence_buffer_size=10,
        event_bus=event_bus,
        camera_id=0,
    )
    print("  ✅ TheftDetector 생성됨")
    print()
    
    # Mock Detection 생성
    print("[4/4] Mock 패키지 감지 테스트...")
    from src.package_detection import Detection
    
    # 첫 번째 프레임: 패키지 감지
    detection1 = Detection(
        bbox=(100, 100, 200, 200),
        confidence=0.8,
        class_id=26,  # handbag
        class_name="handbag",
        timestamp=time.time(),
    )
    
    timestamp = time.monotonic()
    tracked = tracker.track([detection1], timestamp)
    print(f"  ✅ 패키지 추적 시작: {tracked[0].package_id}")
    print(f"     상태: {tracked[0].status}")
    print()
    
    # 두 번째 프레임: 같은 위치 (업데이트)
    time.sleep(0.1)
    detection2 = Detection(
        bbox=(105, 105, 205, 205),  # 약간 이동
        confidence=0.85,
        class_id=26,
        class_name="handbag",
        timestamp=time.time(),
    )
    timestamp = time.monotonic()
    tracked = tracker.track([detection2], timestamp)
    print(f"  ✅ 패키지 업데이트: {tracked[0].package_id}")
    print(f"     감지 횟수: {len(tracked[0].detections)}")
    print()
    
    # 세 번째 프레임: 패키지 사라짐 (missing 상태로 전환)
    time.sleep(1.5)  # missing_threshold(1.0초) 초과
    timestamp = time.monotonic()
    tracked = tracker.track([], timestamp)  # 감지 없음
    print(f"  ✅ 패키지 사라짐 감지: {tracked[0].package_id}")
    print(f"     상태: {tracked[0].status}")
    print()
    
    # 도난 감지 테스트
    time.sleep(1.5)  # confirmation_time(1.0초) 초과
    timestamp = time.monotonic()
    theft_event = theft_detector.check_theft(tracked, timestamp)
    
    if theft_event:
        print(f"  🚨 도난 감지: {theft_event.package_id}")
        print(f"     시간: {theft_event.theft_time}")
    else:
        print("  ⚠️  도난 감지 안됨 (예상보다 빠름)")
    print()
    
    # 이벤트 확인
    time.sleep(0.5)  # 이벤트 처리 대기
    print(f"[결과] 수신된 이벤트: {len(events_received)}개")
    for i, event in enumerate(events_received, 1):
        print(f"  {i}. {event.event_type}: {getattr(event, 'package_id', 'N/A')}")
    print()
    
    # 정리
    event_bus.stop()
    print("✅ 테스트 완료!")
    print("=" * 60)
    
    return len(events_received) > 0


if __name__ == "__main__":
    try:
        success = test_package_detection_components()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
