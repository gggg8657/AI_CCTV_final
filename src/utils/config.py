"""
설정 관리 유틸리티
=================

환경 변수 및 설정 파일 관리
크로스 플랫폼 호환성 고려
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# .env 파일 로드
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """환경 변수 조회"""
    return os.getenv(key, default)


def get_model_path(key: str, default_paths: list = None) -> Optional[Path]:
    """
    모델 경로 조회 (크로스 플랫폼)
    
    Args:
        key: 환경 변수 키 (예: "VLM_MODEL_PATH")
        default_paths: 기본 경로 목록 (환경 변수가 없을 때 시도)
    
    Returns:
        Path 객체 또는 None
    """
    # 환경 변수에서 먼저 조회
    env_path = get_env(key)
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
    
    # 기본 경로 시도
    if default_paths:
        for default_path in default_paths:
            path = Path(default_path)
            if path.exists():
                return path
    
    return None


def get_vlm_model_path() -> Optional[Path]:
    """VLM 모델 경로 조회"""
    default_paths = [
        # Mac/Linux 경로
        Path.home() / "models" / "Qwen2.5-VL-7B-Instruct-q4_k_m.gguf",
        Path("/data/DJ/models/Qwen2.5-VL-7B-Instruct-q4_k_m.gguf"),
        Path("/home/dongjukim/models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"),
        # Windows 경로
        Path("C:/Users") / os.getenv("USERNAME", "user") / "models" / "Qwen2.5-VL-7B-Instruct-q4_k_m.gguf",
    ]
    return get_model_path("VLM_MODEL_PATH", default_paths)


def get_vlm_mmproj_path() -> Optional[Path]:
    """VLM mmproj 경로 조회"""
    # VLM 모델 경로에서 자동으로 mmproj 경로 생성 시도
    vlm_path = get_vlm_model_path()
    if vlm_path:
        mmproj_path = vlm_path.parent / f"{vlm_path.stem}-mmproj-f16.gguf"
        if mmproj_path.exists():
            return mmproj_path
    
    # 환경 변수에서 직접 조회
    default_paths = [
        Path.home() / "models" / "Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf",
        Path("/data/DJ/models/Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf"),
        Path("/home/dongjukim/models/Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf"),
        Path("C:/Users") / os.getenv("USERNAME", "user") / "models" / "Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf",
    ]
    return get_model_path("VLM_MMPROJ_PATH", default_paths)


def get_agent_model_path() -> Optional[Path]:
    """Agent 텍스트 모델 경로 조회"""
    default_paths = [
        Path.home() / "models" / "Qwen3-8B-Q4_K_M.gguf",
        Path("/data/DJ/models/Qwen3-8B-Q4_K_M.gguf"),
        Path("C:/Users") / os.getenv("USERNAME", "user") / "models" / "Qwen3-8B-Q4_K_M.gguf",
    ]
    return get_model_path("AGENT_TEXT_MODEL_PATH", default_paths)


def get_database_url() -> str:
    """데이터베이스 URL 조회"""
    return get_env(
        "DATABASE_URL",
        "postgresql://user:password@localhost/ai_cctv"
    )


def get_gpu_id() -> int:
    """GPU ID 조회"""
    cuda_visible = get_env("CUDA_VISIBLE_DEVICES", "0")
    try:
        return int(cuda_visible)
    except ValueError:
        return 0


def get_api_config() -> dict:
    """API 서버 설정 조회"""
    return {
        "host": get_env("API_HOST", "0.0.0.0"),
        "port": int(get_env("API_PORT", "8000")),
    }


def get_log_config() -> dict:
    """로깅 설정 조회"""
    return {
        "level": get_env("LOG_LEVEL", "INFO"),
        "dir": get_env("LOG_DIR", "logs"),
    }


def get_clips_dir() -> Path:
    """클립 저장 디렉토리 조회"""
    clips_dir = get_env("CLIPS_DIR", "clips")
    return Path(clips_dir)
