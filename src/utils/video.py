"""
비디오 처리 유틸리티
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Generator
from enum import Enum


class VideoSourceType(Enum):
    """비디오 소스 타입"""
    FILE = "file"
    RTSP = "rtsp"
    WEBCAM = "webcam"


class VideoSource:
    """비디오 소스 관리자"""
    
    def __init__(self, source_type: VideoSourceType, source_path: str):
        self.source_type = source_type
        self.source_path = source_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 30.0
        self.width: int = 640
        self.height: int = 480
        self.total_frames: int = 0
    
    def open(self) -> bool:
        """비디오 소스 열기"""
        if self.source_type == VideoSourceType.WEBCAM:
            source = int(self.source_path) if self.source_path.isdigit() else 0
        else:
            source = self.source_path
        
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            return False
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        return True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """프레임 읽기 (BGR)"""
        if self.cap is None:
            return False, None
        return self.cap.read()
    
    def read_rgb(self) -> Tuple[bool, Optional[np.ndarray]]:
        """프레임 읽기 (RGB)"""
        ret, frame = self.read()
        if ret and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return ret, frame
    
    def close(self):
        """비디오 소스 닫기"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def release(self):
        """alias for close"""
        self.close()
    
    def get_info(self) -> dict:
        """비디오 정보"""
        return {
            "type": self.source_type.value,
            "path": self.source_path,
            "fps": self.fps,
            "resolution": f"{self.width}x{self.height}",
            "total_frames": self.total_frames
        }
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def extract_frames(
    video_path: str,
    n_frames: int = 8,
    target_size: Optional[Tuple[int, int]] = None
) -> List[np.ndarray]:
    """
    비디오에서 균등 간격으로 프레임 추출
    
    Args:
        video_path: 비디오 파일 경로
        n_frames: 추출할 프레임 수
        target_size: 리사이즈 크기 (width, height)
    
    Returns:
        프레임 리스트 (RGB)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return []
    
    # 균등 간격으로 프레임 인덱스 계산
    n = min(n_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, n, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 리사이즈
            if target_size is not None:
                frame = cv2.resize(frame, target_size)
            
            frames.append(frame)
    
    cap.release()
    return frames


def frames_to_grid(
    frames: List[np.ndarray],
    rows: int = 2,
    cols: int = 4
) -> np.ndarray:
    """
    프레임들을 그리드로 결합
    
    Args:
        frames: 프레임 리스트
        rows: 행 수
        cols: 열 수
    
    Returns:
        그리드 이미지
    """
    if not frames:
        return None
    
    n_frames = len(frames)
    h, w = frames[0].shape[:2]
    cell_h, cell_w = h // 2, w // 2
    
    grid = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)
    
    for i, frame in enumerate(frames[:rows * cols]):
        r, c = i // cols, i % cols
        resized = cv2.resize(frame, (cell_w, cell_h))
        grid[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = resized
    
    return grid


def encode_frame_base64(frame: np.ndarray, quality: int = 85) -> str:
    """프레임을 base64로 인코딩"""
    import base64
    
    # RGB -> BGR for cv2.imencode
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame_bgr = frame
    
    _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')



