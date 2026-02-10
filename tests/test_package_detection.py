"""
Package Detection 단위 테스트
- Detection 데이터클래스
- TrackedPackage 데이터클래스
- PackageDetector
- PackageTracker (IOU, matching, lifecycle)
- TheftDetector (confirmation logic)
"""

import pytest
import time
from collections import deque
from unittest.mock import MagicMock, patch
import numpy as np

from src.package_detection import (
    PackageDetector,
    PackageTracker,
    TheftDetector,
    Detection,
    TrackedPackage,
)
from src.utils.events import (
    PackageDetectedEvent,
    PackageDisappearedEvent,
    TheftDetectedEvent,
)


class TestDetectionDataclass:
    """Detection 데이터클래스 테스트"""
    
    def test_detection_creation(self):
        """Detection 생성 테스트"""
        detection = Detection(
            bbox=(10, 20, 100, 200),
            confidence=0.85,
            class_id=26,
            class_name="handbag",
            timestamp=1234567890.0,
        )
        
        assert detection.bbox == (10, 20, 100, 200)
        assert detection.confidence == 0.85
        assert detection.class_id == 26
        assert detection.class_name == "handbag"
        assert detection.timestamp == 1234567890.0
    
    def test_detection_default_values(self):
        """Detection 기본값 테스트"""
        detection = Detection(
            bbox=(0, 0, 50, 50),
            confidence=0.5,
            class_id=27,
        )
        
        assert detection.class_name == "package"
        assert detection.timestamp == 0.0
    
    def test_detection_timestamp_iso(self):
        """Detection ISO 타임스탬프 테스트"""
        detection = Detection(
            bbox=(0, 0, 50, 50),
            confidence=0.5,
            class_id=27,
            timestamp=1704067200.0,  # 2024-01-01 00:00:00 UTC
        )
        
        iso = detection.timestamp_iso
        assert "2024-01-01" in iso


class TestTrackedPackageDataclass:
    """TrackedPackage 데이터클래스 테스트"""
    
    def test_tracked_package_creation(self):
        """TrackedPackage 생성 테스트"""
        package = TrackedPackage(
            package_id="pkg_0001",
            first_seen=1704067200.0,
            last_seen=1704067300.0,
            current_position=(100, 100, 200, 200),
            status="present",
            camera_id=1,
        )
        
        assert package.package_id == "pkg_0001"
        assert package.status == "present"
        assert package.camera_id == 1
    
    def test_tracked_package_to_dict(self):
        """TrackedPackage to_dict 테스트"""
        package = TrackedPackage(
            package_id="pkg_0002",
            first_seen=1704067200.0,
            last_seen=1704067300.0,
            current_position=(50, 50, 150, 150),
            status="missing",
            camera_id=0,
        )
        
        d = package.to_dict()
        
        assert d["package_id"] == "pkg_0002"
        assert d["status"] == "missing"
        assert d["current_position"] == (50, 50, 150, 150)
        assert d["detection_count"] == 0
        assert d["camera_id"] == 0
    
    def test_tracked_package_iso_times(self):
        """TrackedPackage ISO 시간 테스트"""
        package = TrackedPackage(
            package_id="pkg_0003",
            first_seen=1704067200.0,
            last_seen=1704067300.0,
        )
        
        assert "2024-01-01" in package.first_seen_iso
        assert "2024-01-01" in package.last_seen_iso


class TestPackageDetector:
    """PackageDetector 테스트"""
    
    def test_detector_initialization(self):
        """PackageDetector 초기화 테스트"""
        detector = PackageDetector(
            model_path="yolo12n.pt",
            device="cpu",
            confidence_threshold=0.6,
        )
        
        assert detector._model_path == "yolo12n.pt"
        assert detector._device == "cpu"
        assert detector._confidence_threshold == 0.6
        assert detector._model is None
    
    def test_detector_custom_class_ids(self):
        """PackageDetector 커스텀 클래스 ID 테스트"""
        detector = PackageDetector(
            model_path="yolo12n.pt",
            device="cpu",
            confidence_threshold=0.5,
            target_class_ids=(26, 27, 28, 29),
        )
        
        assert detector._target_class_ids == (26, 27, 28, 29)
    
    def test_detector_detect_invalid_frame(self):
        """유효하지 않은 프레임 감지 테스트"""
        detector = PackageDetector(
            model_path="yolo12n.pt",
            device="cpu",
            confidence_threshold=0.5,
        )
        
        # None 프레임
        result = detector.detect(None)
        assert result == []
        
        # 문자열 프레임
        result = detector.detect("not a frame")
        assert result == []
    
    def test_detector_detect_with_mock_model(self):
        """Mock 모델을 사용한 감지 테스트"""
        detector = PackageDetector(
            model_path="yolo12n.pt",
            device="cpu",
            confidence_threshold=0.5,
        )
        
        # Mock YOLO 모델 설정
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_boxes = MagicMock()
        
        # boxes 속성 설정
        mock_boxes.cls = [MagicMock(item=lambda: 26)]  # handbag
        mock_boxes.conf = [MagicMock(item=lambda: 0.8)]
        mock_boxes.xyxy = [MagicMock(tolist=lambda: [100.0, 100.0, 200.0, 200.0])]
        mock_boxes.__len__ = lambda self: 1
        
        mock_result.boxes = mock_boxes
        mock_model.predict.return_value = [mock_result]
        
        detector._model = mock_model
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        
        assert mock_model.predict.called


class TestPackageTracker:
    """PackageTracker 테스트"""
    
    def test_tracker_initialization(self):
        """PackageTracker 초기화 테스트"""
        tracker = PackageTracker(
            iou_threshold=0.4,
            max_age=60.0,
            missing_threshold=2.0,
            history_size=100,
            camera_id=1,
        )
        
        assert tracker._iou_threshold == 0.4
        assert tracker._max_age == 60.0
        assert tracker._missing_threshold == 2.0
        assert tracker._history_size == 100
        assert tracker._camera_id == 1
        assert len(tracker._tracked) == 0
    
    def test_tracker_create_new_package(self):
        """새 패키지 생성 테스트"""
        tracker = PackageTracker()
        
        detection = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.8,
            class_id=26,
            class_name="handbag",
            timestamp=time.time(),
        )
        
        timestamp = time.monotonic()
        tracked = tracker.track([detection], timestamp)
        
        assert len(tracked) == 1
        assert tracked[0].package_id == "pkg_0001"
        assert tracked[0].status == "present"
    
    def test_tracker_match_existing_package(self):
        """기존 패키지 매칭 테스트"""
        tracker = PackageTracker(iou_threshold=0.3)
        
        # 첫 번째 프레임: 패키지 생성
        detection1 = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.8,
            class_id=26,
            timestamp=time.time(),
        )
        ts1 = time.monotonic()
        tracker.track([detection1], ts1)
        
        # 두 번째 프레임: 같은 위치 (매칭되어야 함)
        detection2 = Detection(
            bbox=(105, 105, 205, 205),  # 약간 이동
            confidence=0.85,
            class_id=26,
            timestamp=time.time(),
        )
        ts2 = time.monotonic()
        tracked = tracker.track([detection2], ts2)
        
        # 여전히 1개 패키지 (새로 생성되지 않음)
        assert len(tracked) == 1
        assert tracked[0].package_id == "pkg_0001"
    
    def test_tracker_package_missing(self):
        """패키지 사라짐 테스트"""
        tracker = PackageTracker(
            missing_threshold=0.1,  # 매우 짧게 설정
        )
        
        # 패키지 생성
        detection = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.8,
            class_id=26,
            timestamp=time.time(),
        )
        ts1 = time.monotonic()
        tracker.track([detection], ts1)
        
        # 잠시 대기
        time.sleep(0.2)
        
        # 빈 감지로 업데이트 (패키지 사라짐)
        ts2 = time.monotonic()
        tracked = tracker.track([], ts2)
        
        assert len(tracked) == 1
        assert tracked[0].status == "missing"
    
    def test_tracker_get_package(self):
        """패키지 조회 테스트"""
        tracker = PackageTracker()
        
        detection = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.8,
            class_id=26,
            timestamp=time.time(),
        )
        ts = time.monotonic()
        tracker.track([detection], ts)
        
        # 존재하는 패키지
        package = tracker.get_package("pkg_0001")
        assert package is not None
        assert package.package_id == "pkg_0001"
        
        # 존재하지 않는 패키지
        package = tracker.get_package("pkg_9999")
        assert package is None
    
    def test_tracker_iou_calculation(self):
        """IOU 계산 테스트"""
        tracker = PackageTracker()
        
        # 완전히 겹치는 박스
        iou = tracker._calculate_iou(
            (0, 0, 100, 100),
            (0, 0, 100, 100),
        )
        assert iou == 1.0
        
        # 전혀 겹치지 않는 박스
        iou = tracker._calculate_iou(
            (0, 0, 100, 100),
            (200, 200, 300, 300),
        )
        assert iou == 0.0
        
        # 50% 겹치는 박스
        iou = tracker._calculate_iou(
            (0, 0, 100, 100),
            (50, 0, 150, 100),
        )
        assert 0.3 < iou < 0.4  # 약 1/3


class TestTheftDetector:
    """TheftDetector 테스트"""
    
    def test_theft_detector_initialization(self):
        """TheftDetector 초기화 테스트"""
        detector = TheftDetector(
            confirmation_time=5.0,
            evidence_buffer_size=20,
            camera_id=2,
        )
        
        assert detector._confirmation_time == 5.0
        assert detector._camera_id == 2
        assert len(detector._evidence_frames) == 0
    
    def test_theft_detector_add_evidence(self):
        """증거 프레임 추가 테스트"""
        detector = TheftDetector(evidence_buffer_size=5)
        
        for i in range(10):
            detector.add_evidence_frame(f"/path/to/frame_{i}.jpg")
        
        # 버퍼 크기 제한 확인
        assert len(detector._evidence_frames) == 5
        
        # 가장 최근 프레임이 마지막에 있는지 확인
        assert detector._evidence_frames[-1] == "/path/to/frame_9.jpg"
    
    def test_theft_detector_no_theft_present_package(self):
        """present 상태 패키지는 도난 아님"""
        detector = TheftDetector(confirmation_time=1.0)
        
        package = TrackedPackage(
            package_id="pkg_0001",
            status="present",
            missing_since_monotonic=None,
        )
        
        ts = time.monotonic()
        result = detector.check_theft([package], ts)
        
        assert result is None
        assert package.status == "present"
    
    def test_theft_detector_no_theft_short_missing(self):
        """짧은 missing은 도난 아님"""
        detector = TheftDetector(confirmation_time=3.0)
        
        package = TrackedPackage(
            package_id="pkg_0001",
            status="missing",
            missing_since_monotonic=time.monotonic() - 1.0,  # 1초 전
        )
        
        ts = time.monotonic()
        result = detector.check_theft([package], ts)
        
        assert result is None
        assert package.status == "missing"  # 여전히 missing
    
    def test_theft_detector_theft_detected(self):
        """도난 감지 테스트"""
        detector = TheftDetector(confirmation_time=1.0)
        detector.add_evidence_frame("/path/to/evidence.jpg")
        
        package = TrackedPackage(
            package_id="pkg_0001",
            status="missing",
            missing_since_monotonic=time.monotonic() - 2.0,  # 2초 전 (confirmation_time 초과)
            camera_id=0,
        )
        
        ts = time.monotonic()
        result = detector.check_theft([package], ts)
        
        assert result is not None
        assert isinstance(result, TheftDetectedEvent)
        assert result.package_id == "pkg_0001"
        assert package.status == "stolen"
        assert len(result.evidence_frame_paths) == 1
    
    def test_theft_detector_skip_already_stolen(self):
        """이미 stolen인 패키지는 스킵"""
        detector = TheftDetector(confirmation_time=1.0)
        
        package = TrackedPackage(
            package_id="pkg_0001",
            status="stolen",
            missing_since_monotonic=time.monotonic() - 10.0,
        )
        
        ts = time.monotonic()
        result = detector.check_theft([package], ts)
        
        assert result is None  # 이미 stolen이므로 새 이벤트 없음


class TestPackageDetectionWithEventBus:
    """EventBus 연동 테스트"""
    
    def test_tracker_publishes_detected_event(self):
        """패키지 감지 이벤트 발행 테스트"""
        from src.utils.event_bus import EventBus
        
        event_bus = EventBus()
        event_bus.start()
        
        events = []
        event_bus.subscribe(PackageDetectedEvent, lambda e: events.append(e))
        
        tracker = PackageTracker(event_bus=event_bus)
        
        detection = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.8,
            class_id=26,
            timestamp=time.time(),
        )
        
        tracker.track([detection], time.monotonic())
        time.sleep(0.1)  # 이벤트 처리 대기
        
        assert len(events) == 1
        assert events[0].package_id == "pkg_0001"
        
        event_bus.stop()
    
    def test_tracker_publishes_disappeared_event(self):
        """패키지 사라짐 이벤트 발행 테스트"""
        from src.utils.event_bus import EventBus
        
        event_bus = EventBus()
        event_bus.start()
        
        events = []
        event_bus.subscribe(PackageDisappearedEvent, lambda e: events.append(e))
        
        tracker = PackageTracker(
            event_bus=event_bus,
            missing_threshold=0.05,  # 매우 짧게 설정
        )
        
        # 패키지 생성
        detection = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.8,
            class_id=26,
            timestamp=time.time(),
        )
        tracker.track([detection], time.monotonic())
        
        time.sleep(0.1)
        
        # 패키지 사라짐
        tracker.track([], time.monotonic())
        time.sleep(0.1)
        
        assert len(events) == 1
        assert events[0].package_id == "pkg_0001"
        
        event_bus.stop()


class TestE2ESystemPackageDetection:
    """E2ESystem Package Detection 통합 테스트"""
    
    def test_e2e_system_config_has_package_detection(self):
        """SystemConfig에 package_detection 설정 확인"""
        from app.e2e_system import SystemConfig
        
        config = SystemConfig()
        
        assert hasattr(config, 'enable_package_detection')
        assert hasattr(config, 'package_model')
        assert hasattr(config, 'package_confidence')
        assert hasattr(config, 'package_tracker_max_age')
        assert hasattr(config, 'theft_confirmation_time')
        
        # 기본값 확인
        assert config.enable_package_detection is True
        assert config.package_confidence == 0.5
        assert config.theft_confirmation_time == 3.0
    
    def test_e2e_system_has_package_detection_methods(self):
        """E2ESystem에 package detection 메서드 확인"""
        from app.e2e_system import E2ESystem, SystemConfig
        
        config = SystemConfig()
        system = E2ESystem(config)
        
        assert hasattr(system, 'package_detector')
        assert hasattr(system, 'package_tracker')
        assert hasattr(system, 'theft_detector')
        assert hasattr(system, 'get_package_tracker')
        assert hasattr(system, 'get_tracked_packages')
        assert hasattr(system, 'get_package_count')
    
    def test_e2e_system_get_package_count_empty(self):
        """패키지 없을 때 통계 테스트"""
        from app.e2e_system import E2ESystem, SystemConfig
        
        config = SystemConfig()
        system = E2ESystem(config)
        
        stats = system.get_package_count()
        
        assert stats["total"] == 0
        assert stats["present"] == 0
        assert stats["missing"] == 0
        assert stats["stolen"] == 0
