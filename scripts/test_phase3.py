#!/usr/bin/env python3
"""
Phase 3 Package Detection 테스트 스크립트
========================================

실제 비디오로 Phase 3 패키지 감지 및 도난 감지 시스템을 테스트합니다.

사용법:
    python scripts/test_phase3.py --source /path/to/video.mp4
    python scripts/test_phase3.py --source 0 --source-type webcam
    python scripts/test_phase3.py --source rtsp://192.168.1.100:554/stream --source-type rtsp
"""

import os
import sys
import argparse
import time
import signal
from pathlib import Path

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.engine import E2EEngine, EngineConfig, VideoSourceType, AgentFlowType
from src.utils.events import (
    PackageDetectedEvent,
    PackageDisappearedEvent,
    TheftDetectedEvent,
)


class Phase3TestRunner:
    """Phase 3 테스트 러너"""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.engine = None
        self.running = True
        self.stats = {
            "packages_detected": 0,
            "packages_disappeared": 0,
            "thefts_detected": 0,
            "frames_processed": 0,
        }
        
        # Signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """시그널 핸들러"""
        print("\n[테스트] 종료 신호 수신...")
        self.running = False
        if self.engine:
            self.engine.stop()
    
    def _on_package_detected(self, event: PackageDetectedEvent):
        """패키지 감지 이벤트 핸들러"""
        self.stats["packages_detected"] += 1
        print(f"[패키지 감지] ID: {event.package_id}, "
              f"위치: {event.bbox}, "
              f"신뢰도: {event.confidence:.2f}, "
              f"카메라: {event.camera_id}")
    
    def _on_package_disappeared(self, event: PackageDisappearedEvent):
        """패키지 사라짐 이벤트 핸들러"""
        self.stats["packages_disappeared"] += 1
        print(f"[패키지 사라짐] ID: {event.package_id}, "
              f"최종 감지: {event.last_seen}, "
              f"카메라: {event.camera_id}")
    
    def _on_theft_detected(self, event: TheftDetectedEvent):
        """도난 감지 이벤트 핸들러"""
        self.stats["thefts_detected"] += 1
        print(f"\n{'='*60}")
        print(f"[🚨 도난 감지!] 패키지 ID: {event.package_id}")
        print(f"   시간: {event.theft_time}")
        print(f"   카메라: {event.camera_id}")
        print(f"   증거 영상: {len(event.evidence_frame_paths)}개")
        print(f"{'='*60}\n")
    
    def _on_frame_processed(self, frame, score):
        """프레임 처리 콜백"""
        self.stats["frames_processed"] += 1
        if self.stats["frames_processed"] % 30 == 0:  # 30프레임마다
            self._print_stats()
    
    def _print_stats(self):
        """통계 출력"""
        if self.engine:
            engine_stats = self.engine.get_stats()
            print(f"\n[통계] 프레임: {self.stats['frames_processed']}, "
                  f"FPS: {engine_stats.get('current_fps', 0):.1f}, "
                  f"패키지 감지: {self.stats['packages_detected']}, "
                  f"사라짐: {self.stats['packages_disappeared']}, "
                  f"도난: {self.stats['thefts_detected']}")
    
    def run(self):
        """테스트 실행"""
        print("=" * 60)
        print("Phase 3 Package Detection 테스트")
        print("=" * 60)
        print(f"비디오 소스: {self.config.source_path}")
        print(f"소스 타입: {self.config.source_type.value}")
        print(f"패키지 감지: {'활성화' if self.config.enable_package_detection else '비활성화'}")
        if self.config.enable_package_detection:
            print(f"  모델: {self.config.package_detection_model}")
            print(f"  신뢰도 임계값: {self.config.package_detection_confidence}")
            print(f"  도난 확인 시간: {self.config.theft_confirmation_time}초")
        print("=" * 60)
        print()
        
        # 엔진 생성
        self.engine = E2EEngine(self.config)
        
        # 초기화
        print("[초기화] 엔진 초기화 중...")
        if not self.engine.initialize():
            print("[오류] 엔진 초기화 실패")
            return False
        
        print("[초기화] 완료!")
        print()
        
        # EventBus 이벤트 핸들러 등록
        if self.engine.event_bus:
            self.engine.event_bus.subscribe(PackageDetectedEvent, self._on_package_detected)
            self.engine.event_bus.subscribe(PackageDisappearedEvent, self._on_package_disappeared)
            self.engine.event_bus.subscribe(TheftDetectedEvent, self._on_theft_detected)
            print("[이벤트] EventBus 핸들러 등록 완료")
        
        # 콜백 설정
        self.engine.on_frame_callback = self._on_frame_processed
        
        # 엔진 시작
        print("[시작] 엔진 실행 중...")
        print("Ctrl+C로 종료할 수 있습니다.\n")
        
        try:
            self.engine.start(background=False)
        except KeyboardInterrupt:
            print("\n[종료] 사용자 중단")
        except Exception as e:
            print(f"\n[오류] 실행 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.engine.stop()
            self._print_final_stats()
        
        return True
    
    def _print_final_stats(self):
        """최종 통계 출력"""
        print("\n" + "=" * 60)
        print("테스트 완료 - 최종 통계")
        print("=" * 60)
        print(f"처리된 프레임: {self.stats['frames_processed']}")
        print(f"패키지 감지 이벤트: {self.stats['packages_detected']}")
        print(f"패키지 사라짐 이벤트: {self.stats['packages_disappeared']}")
        print(f"도난 감지 이벤트: {self.stats['thefts_detected']}")
        
        if self.engine:
            engine_stats = self.engine.get_stats()
            print(f"\n엔진 통계:")
            print(f"  총 프레임: {engine_stats.get('total_frames', 0)}")
            print(f"  평균 FPS: {engine_stats.get('current_fps', 0):.2f}")
            print(f"  이상 감지: {engine_stats.get('anomaly_count', 0)}")
            if self.engine.package_tracker:
                packages = self.engine.package_tracker.get_all_packages()
                print(f"  현재 추적 중인 패키지: {len(packages)}")
                for pkg in packages:
                    print(f"    - {pkg.package_id}: {pkg.status} "
                          f"(감지 횟수: {len(pkg.detections)})")
        
        print("=" * 60)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Phase 3 Package Detection 테스트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 비디오 파일 테스트
  python scripts/test_phase3.py --source /path/to/video.mp4
  
  # 웹캠 테스트
  python scripts/test_phase3.py --source 0 --source-type webcam
  
  # RTSP 스트림 테스트
  python scripts/test_phase3.py --source rtsp://192.168.1.100:554/stream --source-type rtsp
  
  # CPU 모드 (GPU 없을 때)
  python scripts/test_phase3.py --source /path/to/video.mp4 --gpu -1
  
  # 신뢰도 임계값 조정
  python scripts/test_phase3.py --source /path/to/video.mp4 --confidence 0.6
        """
    )
    
    parser.add_argument("--source", "-s", type=str, required=True,
                        help="비디오 소스 (파일 경로, RTSP URL, 또는 웹캠 인덱스)")
    parser.add_argument("--source-type", type=str, 
                        choices=["file", "rtsp", "webcam"], default="file",
                        help="소스 타입 (기본값: file)")
    parser.add_argument("--gpu", "-g", type=int, default=2,
                        help="GPU 디바이스 ID (기본값: 2, CPU: -1)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="패키지 감지 신뢰도 임계값 (기본값: 0.5)")
    parser.add_argument("--model", type=str, default="yolo12n.pt",
                        help="YOLO 모델 경로 또는 이름 (기본값: yolo12n.pt)")
    parser.add_argument("--theft-time", type=float, default=3.0,
                        help="도난 확인 시간(초) (기본값: 3.0)")
    parser.add_argument("--max-age", type=int, default=30,
                        help="패키지 추적 최대 유지 시간(초) (기본값: 30)")
    parser.add_argument("--fps", type=int, default=30,
                        help="목표 FPS (기본값: 30)")
    parser.add_argument("--no-vad", action="store_true",
                        help="VAD 비활성화 (패키지 감지만 테스트)")
    parser.add_argument("--no-vlm", action="store_true",
                        help="VLM 비활성화")
    parser.add_argument("--no-agent", action="store_true",
                        help="Agent 비활성화")
    
    args = parser.parse_args()
    
    # 소스 타입 변환
    source_type_map = {
        "file": VideoSourceType.FILE,
        "rtsp": VideoSourceType.RTSP,
        "webcam": VideoSourceType.WEBCAM,
    }
    
    # 설정 생성
    config = EngineConfig(
        source_type=source_type_map[args.source_type],
        source_path=args.source,
        vad_model="mnad" if not args.no_vad else None,
        vad_threshold=0.5,
        enable_vlm=not args.no_vlm,
        vlm_n_frames=8,
        optimize_vlm=True,
        enable_agent=not args.no_agent,
        agent_flow=AgentFlowType.SEQUENTIAL,
        save_clips=True,
        clip_duration=3.0,
        clips_dir="./clips",
        logs_dir="./logs",
        gpu_id=args.gpu,
        target_fps=args.fps,
        # Phase 3 설정
        enable_package_detection=True,
        package_detection_model=args.model,
        package_detection_confidence=args.confidence,
        package_tracker_max_age=args.max_age,
        theft_confirmation_time=args.theft_time,
    )
    
    # 테스트 실행
    runner = Phase3TestRunner(config)
    success = runner.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
