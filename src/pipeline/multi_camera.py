"""
다중 카메라 통합 시스템
=====================

여러 카메라의 스트림을 동시에 처리하고, 공간 추론 및 이벤트 상관관계 분석

주요 기능:
- 여러 RTSP 스트림 동시 처리
- 공간 추론 (카메라 간 이벤트 상관관계)
- 이벤트 병합 및 우선순위 결정
"""

import threading
import queue
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from collections import deque

from ..vad import VADModel
from ..vlm import VLMAnalyzer
from ..agent import AgentFlow


@dataclass
class CameraConfig:
    """카메라 설정"""
    camera_id: str
    rtsp_url: str
    location: str  # 카메라 위치 (예: "Entrance", "Parking Lot")
    priority: int = 1  # 우선순위 (1: 높음, 2: 중간, 3: 낮음)
    enabled: bool = True


@dataclass
class MultiCameraEvent:
    """다중 카메라 이벤트"""
    event_id: str
    timestamp: datetime
    camera_id: str
    location: str
    vad_score: float
    vlm_type: str
    vlm_description: str
    agent_actions: List[Dict]
    
    # 공간 추론 정보
    related_events: List[str] = None  # 관련 이벤트 ID 목록
    spatial_context: str = ""  # 공간적 맥락 설명


class MultiCameraSystem:
    """다중 카메라 통합 시스템"""
    
    def __init__(
        self,
        cameras: List[CameraConfig],
        vad_model: VADModel,
        vlm_analyzer: Optional[VLMAnalyzer] = None,
        agent_flow: Optional[AgentFlow] = None,
    ):
        self.cameras = cameras
        self.vad_model = vad_model
        self.vlm_analyzer = vlm_analyzer
        self.agent_flow = agent_flow
        
        # 이벤트 저장소
        self.events: deque = deque(maxlen=1000)
        self.event_lock = threading.Lock()
        
        # 카메라별 처리 스레드
        self.camera_threads: Dict[str, threading.Thread] = {}
        self.stop_event = threading.Event()
        
        # 공간 추론 설정
        self.spatial_radius = 50.0  # 미터 단위 (같은 위치로 간주하는 반경)
        self.temporal_window = 30.0  # 초 단위 (같은 시간대로 간주하는 윈도우)
    
    def start(self):
        """시스템 시작"""
        self.stop_event.clear()
        
        # 각 카메라에 대해 처리 스레드 시작
        for camera in self.cameras:
            if camera.enabled:
                thread = threading.Thread(
                    target=self._process_camera,
                    args=(camera,),
                    daemon=True
                )
                thread.start()
                self.camera_threads[camera.camera_id] = thread
        
        print(f"[MultiCamera] {len(self.camera_threads)}개 카메라 처리 시작")
    
    def stop(self):
        """시스템 중지"""
        self.stop_event.set()
        for thread in self.camera_threads.values():
            thread.join(timeout=5.0)
        print("[MultiCamera] 시스템 중지됨")
    
    def _process_camera(self, camera: CameraConfig):
        """개별 카메라 처리"""
        # TODO: 실제 구현
        # 1. RTSP 스트림 연결
        # 2. 프레임 읽기
        # 3. VAD 추론
        # 4. 이상 감지 시 VLM + Agent 처리
        # 5. 이벤트 저장 및 공간 추론
        pass
    
    def analyze_spatial_correlation(self, event: MultiCameraEvent) -> List[str]:
        """공간 상관관계 분석"""
        related_events = []
        
        with self.event_lock:
            for other_event in self.events:
                # 시간적 근접성 확인
                time_diff = abs((event.timestamp - other_event.timestamp).total_seconds())
                if time_diff > self.temporal_window:
                    continue
                
                # 공간적 근접성 확인 (같은 위치 또는 인접 위치)
                if (event.location == other_event.location or
                    self._are_locations_nearby(event.location, other_event.location)):
                    related_events.append(other_event.event_id)
        
        return related_events
    
    def _are_locations_nearby(self, loc1: str, loc2: str) -> bool:
        """위치가 인접한지 확인 (간단한 구현)"""
        # TODO: 실제 위치 좌표를 사용한 거리 계산
        # 현재는 이름 기반 간단한 판단
        location_groups = {
            'entrance': ['entrance', 'lobby', 'reception'],
            'parking': ['parking', 'lot', 'garage'],
            'hallway': ['hallway', 'corridor', 'passage'],
        }
        
        for group in location_groups.values():
            if loc1.lower() in group and loc2.lower() in group:
                return True
        
        return False
    
    def get_recent_events(self, n: int = 10) -> List[MultiCameraEvent]:
        """최근 이벤트 조회"""
        with self.event_lock:
            return list(self.events)[-n:]
    
    def get_events_by_location(self, location: str) -> List[MultiCameraEvent]:
        """위치별 이벤트 조회"""
        with self.event_lock:
            return [e for e in self.events if e.location == location]


def main():
    """테스트용 메인"""
    print("다중 카메라 통합 시스템")
    print("구현 필요: 실제 카메라 처리 로직")


if __name__ == "__main__":
    main()





