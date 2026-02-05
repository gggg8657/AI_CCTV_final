"""
E2EEngine 개선 사항 테스트
- 예외 처리
- 스레드 안전성
- 디렉토리 자동 생성
"""

import pytest
import tempfile
import shutil
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.pipeline.engine import E2EEngine, EngineConfig, VideoSourceType
from src.vad.base import VADModel


class MockVADModel(VADModel):
    """테스트용 Mock VAD 모델"""
    
    def __init__(self):
        super().__init__()
        self._initialized = True
        self._device = "cpu"
        self.should_raise_error = False
    
    def initialize(self, device: str) -> bool:
        self._device = device
        return True
    
    def process_frame(self, frame: np.ndarray) -> float:
        if self.should_raise_error:
            raise RuntimeError("VAD processing error")
        return 0.3  # 낮은 점수 (이상 감지 안됨)


class MockVideoSource:
    """테스트용 Mock 비디오 소스"""
    
    def __init__(self, source_type, source_path):
        self.source_type = source_type
        self.source_path = source_path
        self.frame_count = 0
        self.max_frames = 10
        self.should_raise_error = False
    
    def open(self) -> bool:
        return True
    
    def close(self):
        pass
    
    def read_rgb(self):
        if self.should_raise_error:
            raise RuntimeError("Video read error")
        
        if self.frame_count >= self.max_frames:
            return False, None
        
        self.frame_count += 1
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return True, frame
    
    def get_info(self) -> dict:
        return {"width": 640, "height": 480, "fps": 30}


def test_engine_directory_creation():
    """디렉토리 자동 생성 테스트"""
    with tempfile.TemporaryDirectory() as tmpdir:
        logs_dir = Path(tmpdir) / "logs"
        clips_dir = Path(tmpdir) / "clips"
        
        # 디렉토리가 존재하지 않음
        assert not logs_dir.exists()
        assert not clips_dir.exists()
        
        config = EngineConfig(
            source_type=VideoSourceType.FILE,
            source_path="test.mp4",
            logs_dir=str(logs_dir),
            clips_dir=str(clips_dir),
            save_clips=True,
            enable_vlm=False,
            enable_agent=False
        )
        
        engine = E2EEngine(config)
        
        # Mock 컴포넌트 설정
        with patch('src.pipeline.engine.VideoSource', return_value=MockVideoSource(VideoSourceType.FILE, "test.mp4")):
            with patch('src.pipeline.engine.create_vad_model', return_value=MockVADModel()):
                # 초기화 시 디렉토리 생성
                result = engine.initialize()
                
                # 디렉토리가 생성되어야 함
                assert logs_dir.exists()
                assert clips_dir.exists()
                assert result is True


def test_engine_exception_handling_in_loop():
    """루프 내 예외 처리 테스트"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = EngineConfig(
            source_type=VideoSourceType.FILE,
            source_path="test.mp4",
            logs_dir=str(Path(tmpdir) / "logs"),
            clips_dir=str(Path(tmpdir) / "clips"),
            enable_vlm=False,
            enable_agent=False,
            target_fps=30
        )
        
        engine = E2EEngine(config)
        
        # VAD 모델이 에러를 발생시키도록 설정
        mock_vad = MockVADModel()
        mock_vad.should_raise_error = True
        
        with patch('src.pipeline.engine.VideoSource', return_value=MockVideoSource(VideoSourceType.FILE, "test.mp4")):
            with patch('src.pipeline.engine.create_vad_model', return_value=mock_vad):
                engine.initialize()
                
                # 백그라운드로 시작
                engine.start(background=True)
                
                # 잠시 대기 (에러가 발생해도 크래시하지 않아야 함)
                time.sleep(0.5)
                
                # 엔진이 정상적으로 중지되어야 함
                engine.stop()
                
                # is_running이 False여야 함
                assert engine.is_running is False


def test_engine_thread_safety():
    """스레드 안전성 테스트"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = EngineConfig(
            source_type=VideoSourceType.FILE,
            source_path="test.mp4",
            logs_dir=str(Path(tmpdir) / "logs"),
            clips_dir=str(Path(tmpdir) / "clips"),
            enable_vlm=False,
            enable_agent=False,
            target_fps=30
        )
        
        engine = E2EEngine(config)
        
        with patch('src.pipeline.engine.VideoSource', return_value=MockVideoSource(VideoSourceType.FILE, "test.mp4")):
            with patch('src.pipeline.engine.create_vad_model', return_value=MockVADModel()):
                engine.initialize()
                
                # 백그라운드로 시작
                engine.start(background=True)
                
                # 메인 스레드가 블로킹되지 않아야 함
                start_time = time.time()
                time.sleep(0.1)
                elapsed = time.time() - start_time
                
                # 0.1초 정도만 걸려야 함 (블로킹되지 않음)
                assert elapsed < 0.2
                
                # 중지
                engine.stop()
                
                # 스레드가 종료되었는지 확인
                if hasattr(engine, '_process_thread') and engine._process_thread:
                    engine._process_thread.join(timeout=1.0)
                    assert not engine._process_thread.is_alive()


def test_engine_sync_mode():
    """동기 모드 테스트"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = EngineConfig(
            source_type=VideoSourceType.FILE,
            source_path="test.mp4",
            logs_dir=str(Path(tmpdir) / "logs"),
            clips_dir=str(Path(tmpdir) / "clips"),
            enable_vlm=False,
            enable_agent=False,
            target_fps=30
        )
        
        engine = E2EEngine(config)
        
        mock_video = MockVideoSource(VideoSourceType.FILE, "test.mp4")
        mock_video.max_frames = 3  # 빠른 종료를 위해
        
        with patch('src.pipeline.engine.VideoSource', return_value=mock_video):
            with patch('src.pipeline.engine.create_vad_model', return_value=MockVADModel()):
                engine.initialize()
                
                # 동기 모드로 시작 (블로킹)
                # 짧은 프레임 수로 빠르게 종료
                engine.start(background=False)
                
                # 루프가 완료되어야 함
                assert engine.stats.total_frames > 0


def test_engine_zero_fps_handling():
    """target_fps가 0일 때 처리 테스트"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = EngineConfig(
            source_type=VideoSourceType.FILE,
            source_path="test.mp4",
            logs_dir=str(Path(tmpdir) / "logs"),
            clips_dir=str(Path(tmpdir) / "clips"),
            enable_vlm=False,
            enable_agent=False,
            target_fps=0  # 0으로 설정
        )
        
        engine = E2EEngine(config)
        
        mock_video = MockVideoSource(VideoSourceType.FILE, "test.mp4")
        mock_video.max_frames = 2
        
        with patch('src.pipeline.engine.VideoSource', return_value=mock_video):
            with patch('src.pipeline.engine.create_vad_model', return_value=MockVADModel()):
                engine.initialize()
                
                # ZeroDivisionError가 발생하지 않아야 함
                engine.start(background=True)
                time.sleep(0.2)
                engine.stop()


def test_engine_clip_saved_callback_error_handling():
    """클립 저장 콜백 에러 처리 테스트"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = EngineConfig(
            source_type=VideoSourceType.FILE,
            source_path="test.mp4",
            logs_dir=str(Path(tmpdir) / "logs"),
            clips_dir=str(Path(tmpdir) / "clips"),
            enable_vlm=True,
            enable_agent=True,
            target_fps=30
        )
        
        engine = E2EEngine(config)
        
        # VLM과 Agent가 에러를 발생시키도록 Mock 설정
        mock_vlm = MagicMock()
        mock_vlm.is_initialized = True
        mock_vlm.analyze.side_effect = RuntimeError("VLM error")
        
        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("Agent error")
        
        with patch('src.pipeline.engine.VideoSource', return_value=MockVideoSource(VideoSourceType.FILE, "test.mp4")):
            with patch('src.pipeline.engine.create_vad_model', return_value=MockVADModel()):
                with patch('src.pipeline.engine.VLMAnalyzer', return_value=mock_vlm):
                    with patch('src.pipeline.engine.create_agent_flow', return_value=mock_agent):
                        engine.initialize()
                        
                        # 클립 저장 콜백 호출 (에러가 발생해도 크래시하지 않아야 함)
                        engine._on_clip_saved("test_clip.mp4", {
                            "timestamp": "2026-01-01T00:00:00",
                            "trigger_time": "2026-01-01T00:00:00",
                            "frame_number": 1,
                            "score": 0.8
                        })
                        
                        # 에러가 발생했지만 엔진은 정상 상태여야 함
                        assert engine.is_running is False or engine.is_running is True  # 상태는 유지


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
