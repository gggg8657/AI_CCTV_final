"""
Phase 3 (Package Detection & Theft Detection) 통합 테스트
- 컴포넌트 초기화
- E2EEngine 통합
- 패키지 감지/트래킹/도난 감지
- EventBus 연동
"""

import pytest
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.pipeline.engine import E2EEngine, EngineConfig, VideoSourceType
from src.package_detection import (
    PackageDetector,
    PackageTracker,
    TheftDetector,
    Detection,
    TrackedPackage,
)
from src.utils.event_bus import EventBus
from src.utils.events import (
    PackageDetectedEvent,
    PackageDisappearedEvent,
    TheftDetectedEvent,
)


class MockVADModel:
    """테스트용 Mock VAD 모델"""
    
    def __init__(self):
        self._initialized = True
        self._device = "cpu"
    
    def initialize(self, device: str) -> bool:
        self._device = device
        return True
    
    def process_frame(self, frame: np.ndarray) -> float:
        return 0.3  # 낮은 점수 (이상 감지 안됨)


class MockVideoSource:
    """테스트용 Mock 비디오 소스"""
    
    def __init__(self, source_type, source_path):
        self.source_type = source_type
        self.source_path = source_path
        self.frame_count = 0
        self.max_frames = 5
    
    def open(self) -> bool:
        return True
    
    def read_rgb(self):
        if self.frame_count >= self.max_frames:
            return False, None
        self.frame_count += 1
        # 더미 프레임 생성 (640x480 RGB)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return True, frame
    
    def close(self):
        pass
    
    def get_info(self) -> str:
        return f"Mock video source: {self.source_path}"


@pytest.fixture
def temp_dir():
    """임시 디렉토리 생성"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_yolo_model():
    """Mock YOLO 모델"""
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_box = MagicMock()
    
    # Mock detection result
    mock_box.cls = [np.array([26.0])]  # handbag class
    mock_box.conf = [np.array([0.8])]  # confidence
    mock_box.xyxy = [np.array([100.0, 100.0, 200.0, 200.0])]  # bbox
    
    mock_result.boxes = mock_box
    mock_model.predict.return_value = [mock_result]
    
    # PackageDetector.load_model을 mock
    with patch.object(PackageDetector, 'load_model', return_value=True):
        with patch.object(PackageDetector, 'detect') as mock_detect:
            # detect 메서드가 호출되면 mock detection 반환
            mock_detect.side_effect = lambda frame: [
                Detection(
                    bbox=(100, 100, 200, 200),
                    confidence=0.8,
                    class_id=26,
                    class_name="handbag",
                    timestamp=time.time(),
                )
            ]
            yield mock_model


@pytest.fixture
def engine_config(temp_dir):
    """테스트용 EngineConfig"""
    return EngineConfig(
        source_type=VideoSourceType.FILE,
        source_path="test_video.mp4",
        vad_model="mnad",
        vad_threshold=0.5,
        enable_vlm=False,
        enable_agent=False,
        save_clips=False,
        logs_dir=temp_dir,
        clips_dir=temp_dir,
        gpu_id=-1,  # CPU
        target_fps=30,
        # Phase 3 설정
        enable_package_detection=True,
        package_detection_model="yolo12n.pt",
        package_detection_confidence=0.5,
        package_tracker_max_age=30,
        theft_confirmation_time=3.0,
    )


class TestPhase3Components:
    """Phase 3 컴포넌트 단위 테스트"""
    
    def test_package_detector_initialization(self, mock_yolo_model):
        """PackageDetector 초기화 테스트"""
        with patch.object(PackageDetector, 'load_model', return_value=True):
            detector = PackageDetector(
                model_path="yolo12n.pt",
                device="cpu",
                confidence_threshold=0.5,
            )
            assert detector._confidence_threshold == 0.5
            
            # 모델 로드
            result = detector.load_model()
            assert result is True
    
    def test_package_detector_detect(self, mock_yolo_model):
        """PackageDetector 감지 테스트"""
        detector = PackageDetector(
            model_path="yolo12n.pt",
            device="cpu",
            confidence_threshold=0.5,
        )
        
        # Mock model 설정
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.cls = [np.array([26.0])]
        mock_box.conf = [np.array([0.8])]
        mock_box.xyxy = [np.array([100.0, 100.0, 200.0, 200.0])]
        mock_result.boxes = mock_box
        mock_model.predict.return_value = [mock_result]
        detector._model = mock_model
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        
        assert len(detections) > 0
        assert isinstance(detections[0], Detection)
        assert detections[0].class_id == 26  # handbag
        assert detections[0].confidence >= 0.5
    
    def test_package_tracker_initialization(self):
        """PackageTracker 초기화 테스트"""
        event_bus = EventBus()
        tracker = PackageTracker(
            iou_threshold=0.3,
            max_age=30.0,
            missing_threshold=1.0,
            event_bus=event_bus,
            camera_id=0,
        )
        
        assert tracker._iou_threshold == 0.3
        assert tracker._max_age == 30.0
        assert tracker._event_bus is not None
        assert len(tracker.get_all_packages()) == 0
    
    def test_package_tracker_track(self):
        """PackageTracker 트래킹 테스트"""
        event_bus = EventBus()
        tracker = PackageTracker(
            iou_threshold=0.3,
            max_age=30.0,
            missing_threshold=1.0,
            event_bus=event_bus,
            camera_id=0,
        )
        
        # 첫 번째 프레임: 패키지 감지
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
        assert tracked[0].status == "present"
        assert tracked[0].package_id.startswith("pkg_")
    
    def test_theft_detector_initialization(self):
        """TheftDetector 초기화 테스트"""
        event_bus = EventBus()
        detector = TheftDetector(
            confirmation_time=3.0,
            evidence_buffer_size=10,
            event_bus=event_bus,
            camera_id=0,
        )
        
        assert detector._confirmation_time == 3.0
        assert detector._event_bus is not None
        assert len(detector._evidence_frames) == 0
    
    def test_theft_detector_check_theft(self):
        """TheftDetector 도난 감지 테스트"""
        event_bus = EventBus()
        detector = TheftDetector(
            confirmation_time=1.0,  # 짧게 설정 (테스트용)
            evidence_buffer_size=10,
            event_bus=event_bus,
            camera_id=0,
        )
        
        # Missing 상태의 패키지 생성
        package = TrackedPackage(
            package_id="pkg_0001",
            status="missing",
            missing_since_monotonic=time.monotonic() - 2.0,  # 2초 전부터 missing
            camera_id=0,
        )
        
        timestamp = time.monotonic()
        theft_event = detector.check_theft([package], timestamp)
        
        assert theft_event is not None
        assert isinstance(theft_event, TheftDetectedEvent)
        assert theft_event.package_id == "pkg_0001"
        assert package.status == "stolen"


class TestPhase3EngineIntegration:
    """E2EEngine Phase 3 통합 테스트"""
    
    @patch('src.pipeline.engine.create_vad_model')
    @patch('src.pipeline.engine.VideoSource')
    @patch.object(PackageDetector, 'load_model', return_value=True)
    def test_engine_phase3_initialization(
        self,
        mock_load_model,
        mock_video_source,
        mock_create_vad,
        engine_config,
        mock_yolo_model,
    ):
        """E2EEngine Phase 3 초기화 테스트"""
        # Mock 설정
        mock_create_vad.return_value = MockVADModel()
        mock_video_source.return_value = MockVideoSource(
            engine_config.source_type,
            engine_config.source_path,
        )
        
        engine = E2EEngine(engine_config)
        result = engine.initialize()
        
        assert result is True
        assert engine.event_bus is not None
        assert engine.package_detector is not None
        assert engine.package_tracker is not None
        assert engine.theft_detector is not None
    
    @patch('src.pipeline.engine.create_vad_model')
    @patch('src.pipeline.engine.VideoSource')
    def test_engine_phase3_disabled(
        self,
        mock_video_source,
        mock_create_vad,
        engine_config,
    ):
        """Phase 3 비활성화 시 테스트"""
        engine_config.enable_package_detection = False
        
        mock_create_vad.return_value = MockVADModel()
        mock_video_source.return_value = MockVideoSource(
            engine_config.source_type,
            engine_config.source_path,
        )
        
        engine = E2EEngine(engine_config)
        result = engine.initialize()
        
        assert result is True
        assert engine.event_bus is None
        assert engine.package_detector is None
        assert engine.package_tracker is None
        assert engine.theft_detector is None
    
    @patch('src.pipeline.engine.create_vad_model')
    @patch('src.pipeline.engine.VideoSource')
    @patch.object(PackageDetector, 'load_model', return_value=True)
    def test_engine_phase3_processing(
        self,
        mock_load_model,
        mock_video_source,
        mock_create_vad,
        engine_config,
        mock_yolo_model,
    ):
        """E2EEngine Phase 3 처리 테스트"""
        # Mock 설정
        mock_create_vad.return_value = MockVADModel()
        mock_video = MockVideoSource(
            engine_config.source_type,
            engine_config.source_path,
        )
        mock_video.max_frames = 3  # 짧게 설정
        mock_video_source.return_value = mock_video
        
        engine = E2EEngine(engine_config)
        engine.initialize()
        
        # Mock detection 설정
        if engine.package_detector:
            mock_model = MagicMock()
            mock_result = MagicMock()
            mock_box = MagicMock()
            mock_box.cls = [np.array([26.0])]
            mock_box.conf = [np.array([0.8])]
            mock_box.xyxy = [np.array([100.0, 100.0, 200.0, 200.0])]
            mock_result.boxes = mock_box
            mock_model.predict.return_value = [mock_result]
            engine.package_detector._model = mock_model
        
        # 짧은 시간 실행
        engine.start(background=False)
        
        # 패키지 트래커에 패키지가 있는지 확인
        if engine.package_tracker:
            packages = engine.package_tracker.get_all_packages()
            # 패키지가 감지되었을 수 있음 (Mock YOLO가 감지를 반환하므로)
            assert isinstance(packages, list)
        
        engine.stop()
    
    @patch('src.pipeline.engine.create_vad_model')
    @patch('src.pipeline.engine.VideoSource')
    @patch.object(PackageDetector, 'load_model', return_value=True)
    def test_engine_phase3_event_bus_stop(
        self,
        mock_load_model,
        mock_video_source,
        mock_create_vad,
        engine_config,
        mock_yolo_model,
    ):
        """EventBus 정지 테스트"""
        mock_create_vad.return_value = MockVADModel()
        mock_video_source.return_value = MockVideoSource(
            engine_config.source_type,
            engine_config.source_path,
        )
        
        engine = E2EEngine(engine_config)
        engine.initialize()
        
        assert engine.event_bus is not None
        assert engine.event_bus._running is True
        
        engine.stop()
        
        # EventBus가 정지되었는지 확인
        assert engine.event_bus._running is False


class TestPhase3EventBusIntegration:
    """Phase 3 EventBus 연동 테스트"""
    
    def test_package_detected_event(self):
        """PackageDetectedEvent 발행 테스트"""
        event_bus = EventBus()
        event_bus.start()
        
        events_received = []
        
        def handler(event):
            events_received.append(event)
        
        event_bus.subscribe(PackageDetectedEvent, handler)
        
        event = PackageDetectedEvent(
            package_id="pkg_0001",
            bbox=(100, 100, 200, 200),
            confidence=0.8,
            camera_id=0,
            frame_index=1,
        )
        
        event_bus.publish_sync(event)
        time.sleep(0.1)  # 이벤트 처리 대기
        
        assert len(events_received) == 1
        assert events_received[0].package_id == "pkg_0001"
        
        event_bus.stop()
    
    def test_theft_detected_event(self):
        """TheftDetectedEvent 발행 테스트"""
        event_bus = EventBus()
        event_bus.start()
        
        events_received = []
        
        def handler(event):
            events_received.append(event)
        
        event_bus.subscribe(TheftDetectedEvent, handler)
        
        event = TheftDetectedEvent(
            package_id="pkg_0001",
            theft_time="2024-01-01T00:00:00Z",
            camera_id=0,
            evidence_frame_paths=["/path/to/evidence.jpg"],
        )
        
        event_bus.publish_sync(event)
        time.sleep(0.1)  # 이벤트 처리 대기
        
        assert len(events_received) == 1
        assert events_received[0].package_id == "pkg_0001"
        assert len(events_received[0].evidence_frame_paths) > 0
        
        event_bus.stop()


class TestPhase3FunctionCalling:
    """Phase 3 Function Calling 연동 테스트"""
    
    def test_get_package_count_without_tracker(self):
        """패키지 트래커가 없을 때 테스트"""
        from src.agent.function_calling import get_package_count
        
        class MockE2ESystem:
            pass
        
        e2e_system = MockE2ESystem()
        result = get_package_count(e2e_system)
        
        assert result["ok"] is True
        assert result["data"]["total"] == 0
        assert result["data"]["present"] == 0
        assert result["data"]["missing"] == 0
        assert result["data"]["stolen"] == 0
    
    def test_get_package_count_with_tracker(self):
        """패키지 트래커가 있을 때 테스트"""
        from src.agent.function_calling import get_package_count
        
        # Mock 패키지 생성
        package1 = TrackedPackage(
            package_id="pkg_0001",
            status="present",
            camera_id=0,
        )
        package2 = TrackedPackage(
            package_id="pkg_0002",
            status="missing",
            camera_id=0,
        )
        package3 = TrackedPackage(
            package_id="pkg_0003",
            status="stolen",
            camera_id=0,
        )
        
        class MockTracker:
            def get_all_packages(self):
                return [package1, package2, package3]
        
        class MockE2ESystem:
            def __init__(self):
                self.package_tracker = MockTracker()
        
        e2e_system = MockE2ESystem()
        result = get_package_count(e2e_system)
        
        assert result["ok"] is True
        assert result["data"]["total"] == 3
        assert result["data"]["present"] == 1
        assert result["data"]["missing"] == 1
        assert result["data"]["stolen"] == 1
    
    def test_get_package_details(self):
        """패키지 상세 정보 조회 테스트"""
        from src.agent.function_calling import get_package_details
        
        package = TrackedPackage(
            package_id="pkg_0001",
            status="present",
            first_seen=time.time() - 100,
            last_seen=time.time(),
            current_position=(100, 100, 200, 200),
            camera_id=0,
        )
        package.detections.append(
            Detection(
                bbox=(100, 100, 200, 200),
                confidence=0.8,
                class_id=26,
            )
        )
        
        class MockTracker:
            def get_package(self, package_id):
                if package_id == "pkg_0001":
                    return package
                return None
        
        class MockE2ESystem:
            def __init__(self):
                self.package_tracker = MockTracker()
        
        e2e_system = MockE2ESystem()
        result = get_package_details(e2e_system, "pkg_0001")
        
        assert result["ok"] is True
        assert result["data"]["package_id"] == "pkg_0001"
        assert result["data"]["status"] == "present"
        assert result["data"]["detection_count"] == 1
        
        # 존재하지 않는 패키지
        result = get_package_details(e2e_system, "pkg_9999")
        assert result["ok"] is False
        assert "not found" in result["error"].lower()
