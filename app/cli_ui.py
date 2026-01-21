#!/usr/bin/env python3
"""
CLI 기반 보안 모니터링 대시보드
================================
Rich 라이브러리를 사용한 터미널 UI

기능:
- 실시간 통계 표시
- 이벤트 로그 스트림
- 이상 점수 히스토그램
- 설정 표시
"""

import os
import sys
import time
import threading
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from collections import deque

# Rich 라이브러리 import
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.style import Style
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.e2e_system import (
    E2ESystem, SystemConfig, VideoSourceType, VADModelType, AgentFlowType,
    AnomalyEvent, SystemStats
)


class CLIDashboard:
    """CLI 기반 대시보드"""
    
    def __init__(self, system: E2ESystem):
        self.system = system
        self.console = Console() if HAS_RICH else None
        
        # 상태
        self.recent_scores: deque = deque(maxlen=50)
        self.recent_events: List[AnomalyEvent] = []
        self.log_messages: deque = deque(maxlen=20)
        
        # 실행 상태
        self.is_running = False
        self._stop_event = threading.Event()
        
        # 콜백 설정
        self.system.on_frame_callback = self._on_frame
        self.system.on_anomaly_callback = self._on_anomaly
        self.system.on_stats_callback = self._on_stats
    
    def _on_frame(self, frame, score: float):
        """프레임 콜백"""
        self.recent_scores.append(score)
    
    def _on_anomaly(self, event: AnomalyEvent):
        """이상 감지 콜백"""
        self.recent_events.insert(0, event)
        if len(self.recent_events) > 10:
            self.recent_events = self.recent_events[:10]
        
        self.log_messages.append(
            f"[{event.timestamp.strftime('%H:%M:%S')}] "
            f"ANOMALY: {event.vlm_type} (score={event.vad_score:.3f})"
        )
    
    def _on_stats(self, stats: SystemStats):
        """통계 콜백"""
        pass
    
    def _create_header(self) -> Panel:
        """헤더 패널"""
        title = Text()
        title.append("🔒 ", style="bold blue")
        title.append("E2E Security Monitoring System", style="bold white")
        title.append(" 🔒", style="bold blue")
        
        return Panel(
            title,
            box=box.DOUBLE,
            style="blue"
        )
    
    def _create_stats_panel(self) -> Panel:
        """통계 패널"""
        stats = self.system.get_stats()
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Runtime", f"{stats['runtime_seconds']:.0f}s")
        table.add_row("Total Frames", f"{stats['total_frames']:,}")
        table.add_row("Current FPS", f"{stats['current_fps']:.1f}")
        table.add_row("Anomalies", f"{stats['anomaly_count']}")
        table.add_row("VAD Time", f"{stats['avg_vad_time_ms']:.1f}ms")
        
        if self.system.config.enable_vlm:
            table.add_row("VLM Time", f"{stats['avg_vlm_time_ms']:.1f}ms")
        
        if self.system.config.enable_agent:
            table.add_row("Agent Time", f"{stats['avg_agent_time_ms']:.1f}ms")
        
        return Panel(table, title="📊 Statistics", border_style="green")
    
    def _create_config_panel(self) -> Panel:
        """설정 패널"""
        config = self.system.config
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="yellow")
        
        table.add_row("Source", config.source_type.value)
        table.add_row("VAD Model", config.vad_model.value)
        table.add_row("Threshold", f"{config.vad_threshold:.2f}")
        table.add_row("VLM", "✓" if config.enable_vlm else "✗")
        table.add_row("Agent", config.agent_flow.value if config.enable_agent else "✗")
        table.add_row("Clip Save", "✓" if config.save_clips else "✗")
        table.add_row("GPU", f"cuda:{config.gpu_id}")
        
        return Panel(table, title="⚙️ Configuration", border_style="yellow")
    
    def _create_score_panel(self) -> Panel:
        """점수 히스토그램 패널"""
        if not self.recent_scores:
            return Panel("No data", title="📈 Score History", border_style="blue")
        
        # ASCII 히스토그램
        max_height = 8
        width = min(len(self.recent_scores), 50)
        
        scores = list(self.recent_scores)[-width:]
        
        lines = []
        for h in range(max_height, 0, -1):
            line = ""
            threshold_line = h / max_height
            for score in scores:
                if score >= threshold_line:
                    if score >= self.system.config.vad_threshold:
                        line += "█"  # 이상
                    else:
                        line += "▓"  # 정상
                else:
                    line += " "
            lines.append(line)
        
        # 임계값 라인
        threshold_pos = int(self.system.config.vad_threshold * max_height)
        
        hist_text = "\n".join(lines)
        hist_text += "\n" + "─" * width
        hist_text += f"\nThreshold: {self.system.config.vad_threshold:.2f}"
        
        current = scores[-1] if scores else 0
        hist_text += f" | Current: {current:.3f}"
        
        style = "red" if current >= self.system.config.vad_threshold else "green"
        
        return Panel(hist_text, title="📈 Score History", border_style=style)
    
    def _create_events_panel(self) -> Panel:
        """이벤트 목록 패널"""
        if not self.recent_events:
            return Panel("No anomalies detected", title="⚠️ Recent Events", border_style="red")
        
        table = Table(show_header=True, box=box.SIMPLE, padding=(0, 1))
        table.add_column("Time", style="cyan", width=10)
        table.add_column("Type", style="red", width=15)
        table.add_column("Score", style="yellow", width=8)
        table.add_column("Actions", style="green", width=10)
        
        for event in self.recent_events[:5]:
            table.add_row(
                event.timestamp.strftime('%H:%M:%S'),
                event.vlm_type[:15],
                f"{event.vad_score:.3f}",
                str(len(event.agent_actions))
            )
        
        return Panel(table, title="⚠️ Recent Events", border_style="red")
    
    def _create_log_panel(self) -> Panel:
        """로그 패널"""
        if not self.log_messages:
            return Panel("Waiting for events...", title="📝 Log", border_style="white")
        
        log_text = "\n".join(list(self.log_messages)[-10:])
        return Panel(log_text, title="📝 Log", border_style="white")
    
    def _create_layout(self) -> Layout:
        """레이아웃 생성"""
        layout = Layout()
        
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=12)
        )
        
        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=2)
        )
        
        layout["left"].split(
            Layout(name="stats"),
            Layout(name="config")
        )
        
        layout["right"].split(
            Layout(name="score"),
            Layout(name="events")
        )
        
        return layout
    
    def _update_layout(self, layout: Layout):
        """레이아웃 업데이트"""
        layout["header"].update(self._create_header())
        layout["stats"].update(self._create_stats_panel())
        layout["config"].update(self._create_config_panel())
        layout["score"].update(self._create_score_panel())
        layout["events"].update(self._create_events_panel())
        layout["footer"].update(self._create_log_panel())
    
    def run(self):
        """대시보드 실행"""
        if not HAS_RICH:
            print("Rich library not available. Running in simple mode.")
            self._run_simple()
            return
        
        self.is_running = True
        layout = self._create_layout()
        
        # 시스템 스레드 시작
        system_thread = threading.Thread(target=self.system.start)
        system_thread.daemon = True
        system_thread.start()
        
        try:
            with Live(layout, console=self.console, refresh_per_second=4) as live:
                while self.is_running and not self._stop_event.is_set():
                    self._update_layout(layout)
                    time.sleep(0.25)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def _run_simple(self):
        """Rich 없이 간단한 출력"""
        self.is_running = True
        
        # 시스템 스레드 시작
        system_thread = threading.Thread(target=self.system.start)
        system_thread.daemon = True
        system_thread.start()
        
        try:
            while self.is_running and not self._stop_event.is_set():
                stats = self.system.get_stats()
                score = self.system.get_current_score()
                
                status = "ANOMALY!" if score >= self.system.config.vad_threshold else "Normal"
                
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Frames: {stats['total_frames']:,} | "
                      f"FPS: {stats['current_fps']:.1f} | "
                      f"Score: {score:.3f} | "
                      f"Anomalies: {stats['anomaly_count']} | "
                      f"Status: {status}", end="", flush=True)
                
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            print()
            self.stop()
    
    def stop(self):
        """대시보드 중지"""
        self._stop_event.set()
        self.is_running = False
        self.system.stop()


def main():
    """CLI 대시보드 메인"""
    parser = argparse.ArgumentParser(description="CLI Security Monitoring Dashboard")
    parser.add_argument("--source", type=str, required=True, help="Video source")
    parser.add_argument("--vad-model", type=str, default="mnad", 
                        choices=["mnad", "mulde", "memae", "stae"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--enable-vlm", action="store_true")
    parser.add_argument("--vlm-frames", type=int, default=4)
    parser.add_argument("--optimize-vlm", action="store_true")
    parser.add_argument("--enable-agent", action="store_true")
    parser.add_argument("--agent-flow", type=str, default="sequential",
                        choices=["hierarchical", "sequential", "collaborative"])
    parser.add_argument("--save-clips", action="store_true", default=True)
    parser.add_argument("--clip-duration", type=float, default=3.0)
    parser.add_argument("--clips-dir", type=str, default="clips")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--simple", action="store_true", help="Use simple output mode")
    
    args = parser.parse_args()
    
    # 소스 타입 결정
    if args.source.startswith("rtsp://"):
        source_type = VideoSourceType.RTSP
    elif args.source.isdigit():
        source_type = VideoSourceType.WEBCAM
    else:
        source_type = VideoSourceType.FILE
    
    config = SystemConfig(
        source_type=source_type,
        source_path=args.source,
        vad_model=VADModelType(args.vad_model),
        vad_threshold=args.threshold,
        enable_vlm=args.enable_vlm,
        vlm_n_frames=args.vlm_frames,
        optimize_vlm=args.optimize_vlm,
        enable_agent=args.enable_agent,
        agent_flow=AgentFlowType(args.agent_flow),
        save_clips=args.save_clips,
        clip_duration=args.clip_duration,
        clips_dir=args.clips_dir,
        log_dir=args.log_dir,
        gpu_id=args.gpu,
        target_fps=args.fps
    )
    
    system = E2ESystem(config)
    
    success, error_msg = system.initialize()
    if not success:
        print(f"Failed to initialize system: {error_msg or 'Unknown error'}")
        sys.exit(1)
    
    dashboard = CLIDashboard(system)
    
    if args.simple or not HAS_RICH:
        dashboard._run_simple()
    else:
        dashboard.run()


if __name__ == "__main__":
    main()

