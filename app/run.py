#!/usr/bin/env python3
"""
E2E Security Monitoring System - 메인 실행 스크립트
===================================================

실행 방법:
    # CLI 모드
    python app/run.py --mode cli --source /path/to/video.mp4
    
    # Web UI 모드
    python app/run.py --mode web --source /path/to/video.mp4
    
    # 설정 파일 사용
    python app/run.py --config app/config.yaml
    
    # RTSP 스트림
    python app/run.py --mode cli --source rtsp://192.168.1.100:554/stream
    
    # 웹캠
    python app/run.py --mode cli --source 0 --source-type webcam
"""

import os
import sys
import argparse
from pathlib import Path

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# YAML 파싱
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def load_config(config_path: str) -> dict:
    """설정 파일 로드"""
    if not HAS_YAML:
        print("Warning: PyYAML not installed. Using default config.")
        return {}
    
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found: {config_path}")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def merge_config_with_args(config: dict, args) -> dict:
    """설정 파일과 CLI 인자 병합 (CLI 인자가 우선)"""
    # 기본값 설정
    merged = {
        "source_type": "file",
        "source_path": "",
        "vad_model": "mnad",
        "threshold": 0.5,
        "enable_vlm": True,
        "vlm_frames": 4,
        "optimize_vlm": True,
        "enable_agent": True,
        "agent_flow": "sequential",
        "save_clips": True,
        "clip_duration": 3.0,
        "clips_dir": "clips",
        "log_dir": "logs",
        "gpu_id": 2,
        "target_fps": 30,
        "mode": "cli",
        "web_port": 8501
    }
    
    # 설정 파일에서 로드
    if config:
        if "video" in config:
            merged["source_type"] = config["video"].get("source_type", merged["source_type"])
            merged["source_path"] = config["video"].get("source_path", merged["source_path"])
            merged["target_fps"] = config["video"].get("target_fps", merged["target_fps"])
        
        if "vad" in config:
            merged["vad_model"] = config["vad"].get("model", merged["vad_model"])
            merged["threshold"] = config["vad"].get("threshold", merged["threshold"])
        
        if "vlm" in config:
            merged["enable_vlm"] = config["vlm"].get("enabled", merged["enable_vlm"])
            merged["vlm_frames"] = config["vlm"].get("n_frames", merged["vlm_frames"])
            merged["optimize_vlm"] = config["vlm"].get("optimize", merged["optimize_vlm"])
        
        if "agent" in config:
            merged["enable_agent"] = config["agent"].get("enabled", merged["enable_agent"])
            merged["agent_flow"] = config["agent"].get("flow", merged["agent_flow"])
            # LLM 모델 경로 설정
            if "llm" in config["agent"]:
                merged["agent_text_model_path"] = config["agent"]["llm"].get("text_model_path", "")
                merged["agent_vision_model_path"] = config["agent"]["llm"].get("vision_model_path", "")
                merged["agent_vision_mmproj_path"] = config["agent"]["llm"].get("vision_mmproj_path", "")
                merged["agent_n_gpu_layers"] = config["agent"]["llm"].get("n_gpu_layers", -1)
                merged["agent_n_ctx"] = config["agent"]["llm"].get("n_ctx", 32768)
                merged["agent_n_threads"] = config["agent"]["llm"].get("n_threads", 16)
                merged["agent_n_batch"] = config["agent"]["llm"].get("n_batch", 512)
        
        if "clip" in config:
            merged["save_clips"] = config["clip"].get("enabled", merged["save_clips"])
            merged["clip_duration"] = config["clip"].get("duration", merged["clip_duration"])
            merged["clips_dir"] = config["clip"].get("directory", merged["clips_dir"])
        
        if "logging" in config:
            merged["log_dir"] = config["logging"].get("directory", merged["log_dir"])
        
        if "gpu" in config:
            merged["gpu_id"] = config["gpu"].get("device_id", merged["gpu_id"])
        
        if "ui" in config:
            merged["mode"] = config["ui"].get("default_mode", merged["mode"])
            merged["web_port"] = config["ui"].get("web_port", merged["web_port"])
    
    # CLI 인자로 오버라이드
    if args.source:
        merged["source_path"] = args.source
    if args.source_type:
        merged["source_type"] = args.source_type
    if args.mode:
        merged["mode"] = args.mode
    if args.vad_model:
        merged["vad_model"] = args.vad_model
    if args.threshold is not None:
        merged["threshold"] = args.threshold
    if args.gpu is not None:
        merged["gpu_id"] = args.gpu
    if args.no_vlm:
        merged["enable_vlm"] = False
    if args.no_agent:
        merged["enable_agent"] = False
    if args.agent_flow:
        merged["agent_flow"] = args.agent_flow
    if args.port:
        merged["web_port"] = args.port
    
    return merged


def run_cli(settings: dict):
    """CLI 모드 실행"""
    from app.e2e_system import (
        E2ESystem, SystemConfig, VideoSourceType, VADModelType, AgentFlowType
    )
    from app.cli_ui import CLIDashboard
    
    # 소스 타입 변환
    source_type_map = {
        "file": VideoSourceType.FILE,
        "rtsp": VideoSourceType.RTSP,
        "webcam": VideoSourceType.WEBCAM
    }
    
    config = SystemConfig(
        source_type=source_type_map.get(settings["source_type"], VideoSourceType.FILE),
        source_path=settings["source_path"],
        vad_model=VADModelType(settings["vad_model"]),
        vad_threshold=settings["threshold"],
        enable_vlm=settings["enable_vlm"],
        vlm_n_frames=settings["vlm_frames"],
        optimize_vlm=settings["optimize_vlm"],
        enable_agent=settings["enable_agent"],
        agent_flow=AgentFlowType(settings["agent_flow"]),
        # LLM 모델 경로 설정
        agent_text_model_path=settings.get("agent_text_model_path", ""),
        agent_vision_model_path=settings.get("agent_vision_model_path", ""),
        agent_vision_mmproj_path=settings.get("agent_vision_mmproj_path", ""),
        agent_n_gpu_layers=settings.get("agent_n_gpu_layers", -1),
        agent_n_ctx=settings.get("agent_n_ctx", 32768),
        agent_n_threads=settings.get("agent_n_threads", 16),
        agent_n_batch=settings.get("agent_n_batch", 512),
        save_clips=settings["save_clips"],
        clip_duration=settings["clip_duration"],
        clips_dir=settings["clips_dir"],
        log_dir=settings["log_dir"],
        gpu_id=settings["gpu_id"],
        target_fps=settings["target_fps"]
    )
    
    system = E2ESystem(config)
    
    success, error_msg = system.initialize()
    if not success:
        print(f"Failed to initialize system: {error_msg or 'Unknown error'}")
        sys.exit(1)
    
    dashboard = CLIDashboard(system)
    dashboard.run()


def run_web(settings: dict):
    """Web UI 모드 실행"""
    import subprocess
    
    port = settings.get("web_port", 8501)
    
    # 환경 변수로 설정 전달
    env = os.environ.copy()
    env["E2E_SOURCE"] = settings.get("source_path", "")
    env["E2E_SOURCE_TYPE"] = settings.get("source_type", "file")
    env["E2E_VAD_MODEL"] = settings.get("vad_model", "mnad")
    env["E2E_THRESHOLD"] = str(settings.get("threshold", 0.5))
    env["E2E_GPU"] = str(settings.get("gpu_id", 2))
    env["CUDA_VISIBLE_DEVICES"] = str(settings.get("gpu_id", 2))
    
    web_ui_path = PROJECT_ROOT / "app" / "web_ui.py"
    
    print(f"Starting Web UI on port {port}...")
    print(f"Access at: http://localhost:{port}")
    print("Press Ctrl+C to stop\n")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(web_ui_path),
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ], env=env)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="E2E Security Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CLI mode with video file
  python app/run.py --mode cli --source /path/to/video.mp4
  
  # Web UI mode
  python app/run.py --mode web --source /path/to/video.mp4 --port 8080
  
  # RTSP stream
  python app/run.py --mode cli --source rtsp://192.168.1.100:554/stream --source-type rtsp
  
  # Webcam
  python app/run.py --mode cli --source 0 --source-type webcam
  
  # With config file
  python app/run.py --config app/config.yaml --source /path/to/video.mp4
        """
    )
    
    # 기본 인자
    parser.add_argument("--config", "-c", type=str, default="app/config.yaml",
                        help="Config file path (default: app/config.yaml)")
    parser.add_argument("--mode", "-m", type=str, choices=["cli", "web"],
                        help="UI mode: cli or web")
    parser.add_argument("--source", "-s", type=str,
                        help="Video source (file path, RTSP URL, or webcam index)")
    parser.add_argument("--source-type", type=str, choices=["file", "rtsp", "webcam"],
                        help="Source type")
    
    # VAD 인자
    parser.add_argument("--vad-model", type=str, choices=["mnad", "mulde", "memae", "stae"],
                        help="VAD model")
    parser.add_argument("--threshold", "-t", type=float,
                        help="Anomaly threshold (0.0-1.0)")
    
    # VLM/Agent 인자
    parser.add_argument("--no-vlm", action="store_true",
                        help="Disable VLM analysis")
    parser.add_argument("--no-agent", action="store_true",
                        help="Disable Agent")
    parser.add_argument("--agent-flow", type=str,
                        choices=["sequential", "hierarchical", "collaborative"],
                        help="Agent flow type")
    
    # 기타 인자
    parser.add_argument("--gpu", "-g", type=int,
                        help="GPU device ID")
    parser.add_argument("--port", "-p", type=int,
                        help="Web UI port (default: 8501)")
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    settings = merge_config_with_args(config, args)
    
    # 소스 검증
    if not settings["source_path"]:
        print("Error: Video source is required")
        print("Use --source to specify a video file, RTSP URL, or webcam index")
        parser.print_help()
        sys.exit(1)
    
    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = str(settings["gpu_id"])
    
    print("=" * 60)
    print("E2E Security Monitoring System")
    print("=" * 60)
    print(f"Mode:        {settings['mode'].upper()}")
    print(f"Source:      {settings['source_path']}")
    print(f"VAD Model:   {settings['vad_model']}")
    print(f"Threshold:   {settings['threshold']}")
    print(f"VLM:         {'Enabled' if settings['enable_vlm'] else 'Disabled'}")
    print(f"Agent:       {settings['agent_flow'] if settings['enable_agent'] else 'Disabled'}")
    print(f"GPU:         cuda:{settings['gpu_id']}")
    print("=" * 60)
    print()
    
    # 모드별 실행
    if settings["mode"] == "web":
        run_web(settings)
    else:
        run_cli(settings)


if __name__ == "__main__":
    main()

