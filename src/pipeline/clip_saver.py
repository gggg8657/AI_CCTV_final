"""
클립 저장기
==========

이상 감지 시 영상 클립 저장
"""

import os
import json
import threading
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Optional, Callable, Dict, List, Any

import cv2
import numpy as np


class ClipSaver:
    """
    이상 감지 클립 저장기
    
    이상 감지 시점부터 지정된 시간만큼 영상 저장 후 콜백 호출
    """
    
    def __init__(
        self,
        output_dir: str,
        record_seconds: float = 3.0,
        fps: float = 30.0,
        on_clip_saved: Optional[Callable[[str, Dict], None]] = None,
    ):
        """
        Args:
            output_dir: 클립 저장 디렉토리
            record_seconds: 이상 감지 후 녹화 시간 (초)
            fps: 프레임 레이트
            on_clip_saved: 클립 저장 완료 콜백 (clip_path, metadata)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.record_frames = int(record_seconds * fps)
        self.fps = fps
        self.record_seconds = record_seconds
        self.on_clip_saved = on_clip_saved
        
        # 녹화 상태
        self._recording = False
        self._record_frames_left = 0
        self._current_clip: List[np.ndarray] = []
        self._clip_info: Optional[Dict] = None
        
        # 스레드 안전
        self._lock = threading.Lock()
    
    def add_frame(self, frame: np.ndarray) -> Optional[str]:
        """
        프레임 추가
        
        녹화 중이면 프레임을 버퍼에 추가하고,
        녹화 완료 시 클립 저장 후 경로 반환
        
        Args:
            frame: RGB 또는 BGR 프레임
        
        Returns:
            녹화 완료 시 클립 경로, 아니면 None
        """
        with self._lock:
            if self._recording:
                self._current_clip.append(frame.copy())
                self._record_frames_left -= 1
                
                if self._record_frames_left <= 0:
                    return self._save_clip()
        
        return None
    
    def trigger_save(self, score: float, frame_number: int) -> Optional[str]:
        """
        이상 감지 시 녹화 트리거
        
        Args:
            score: 이상 점수
            frame_number: 현재 프레임 번호
        
        Returns:
            트리거 시간 (timestamp) 또는 None (이미 녹화 중)
        """
        with self._lock:
            if self._recording:
                return None  # 이미 녹화 중
            
            self._recording = True
            self._record_frames_left = self.record_frames
            self._current_clip = []
            
            # 클립 정보 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._clip_info = {
                'timestamp': timestamp,
                'frame_number': frame_number,
                'score': score,
                'trigger_time': datetime.now().isoformat(),
            }
            
            return timestamp
    
    def _save_clip(self) -> Optional[str]:
        """클립 저장"""
        if not self._current_clip:
            self._recording = False
            return None
        
        timestamp = self._clip_info['timestamp']
        clip_path = self.output_dir / f"anomaly_{timestamp}.mp4"
        
        # 비디오 저장
        h, w = self._current_clip[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(clip_path), fourcc, self.fps, (w, h))
        
        for frame in self._current_clip:
            # RGB -> BGR (cv2는 BGR 사용)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            writer.write(frame_bgr)
        
        writer.release()
        
        # 메타데이터 저장
        clip_meta = {
            **self._clip_info,
            'frames': len(self._current_clip),
            'duration': len(self._current_clip) / self.fps,
            'clip_path': str(clip_path),
        }
        
        meta_path = clip_path.with_suffix('.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(clip_meta, f, indent=2, ensure_ascii=False)
        
        print(f"[ClipSaver] 저장 완료: {clip_path.name} ({len(self._current_clip)} frames, {clip_meta['duration']:.1f}초)")
        
        # 콜백 호출
        if self.on_clip_saved:
            self.on_clip_saved(str(clip_path), clip_meta)
        
        # 상태 초기화
        result = str(clip_path)
        self._recording = False
        self._current_clip = []
        self._clip_info = None
        
        return result
    
    @property
    def is_recording(self) -> bool:
        """녹화 중 여부"""
        return self._recording
    
    @property
    def remaining_frames(self) -> int:
        """남은 녹화 프레임 수"""
        return self._record_frames_left if self._recording else 0



