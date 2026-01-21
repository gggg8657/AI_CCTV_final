# 모델 설치 및 연결 가이드

## 개요

이 가이드는 AI_CCTV_final 프로젝트에서 사용하는 LLM 및 VLM 모델을 설치하고 연결하는 방법을 설명합니다.

## 필요한 모델

### 1. Text LLM: Qwen3-8B
- **용도**: Agent의 자연어 처리 및 Function Calling
- **형식**: GGUF (Q4_K_M 권장)
- **파일**: `Qwen3-8B-Q4_K_M.gguf`

### 2. Vision LLM: Qwen2.5-VL-7B
- **용도**: 비디오 프레임 분석 및 상황 이해
- **형식**: GGUF (q4_k_m 권장)
- **파일들**:
  - `Qwen2.5-VL-7B-Instruct-q4_k_m.gguf` (메인 모델)
  - `Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf` (MM Projection)

## 모델 다운로드

### 방법 1: Hugging Face에서 직접 다운로드

```bash
# Qwen3-8B Text Model
# https://huggingface.co/Qwen/Qwen3-8B-Instruct-GGUF 에서 다운로드
# Qwen3-8B-Instruct-Q4_K_M.gguf 파일 선택

# Qwen2.5-VL-7B Vision Model
# https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-GGUF 에서 다운로드
# Qwen2.5-VL-7B-Instruct-q4_k_m.gguf 및 mmproj-f16.gguf 파일 선택
```

### 방법 2: llama.cpp를 사용한 변환

기존 모델이 있다면 GGUF로 변환:

```bash
# llama.cpp 설치 필요
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make

# 변환 (예시)
python convert.py --outfile Qwen3-8B-Q4_K_M.gguf --outtype q4_k_m /path/to/original/model
```

## 모델 경로 설정

### 1. config.yaml 설정

`configs/config.yaml` 파일을 열고 모델 경로를 설정합니다:

```yaml
agent:
  enabled: true
  flow: sequential
  llm:
    # Text LLM 모델 경로
    text_model_path: "/path/to/Qwen3-8B-Q4_K_M.gguf"
    
    # Vision LLM 모델 경로
    vision_model_path: "/path/to/Qwen2.5-VL-7B-Instruct-q4_k_m.gguf"
    
    # Vision LLM MM Projection 경로
    vision_mmproj_path: "/path/to/Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf"
    
    # GPU 설정
    n_gpu_layers: -1  # -1 = 모든 레이어 GPU, 0 = CPU만
    n_ctx: 32768      # 컨텍스트 길이
    n_threads: 16     # CPU 스레드 수
    n_batch: 512      # 배치 크기
```

### 2. 환경 변수 설정 (선택)

`.env` 파일을 생성하여 환경 변수로도 설정 가능:

```bash
AGENT_TEXT_MODEL_PATH=/path/to/Qwen3-8B-Q4_K_M.gguf
AGENT_VISION_MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct-q4_k_m.gguf
AGENT_VISION_MMPROJ_PATH=/path/to/Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf
```

## 의존성 설치

### llama-cpp-python 설치

```bash
# 기본 설치
pip install llama-cpp-python

# GPU 지원 (CUDA)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# GPU 지원 (Metal - macOS)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

### 전체 의존성 설치

```bash
pip install -r requirements.txt
```

## 연결 확인

### 1. 모델 파일 존재 확인

```python
import os
from pathlib import Path

# config.yaml에서 읽은 경로 확인
text_model = Path("/path/to/Qwen3-8B-Q4_K_M.gguf")
vision_model = Path("/path/to/Qwen2.5-VL-7B-Instruct-q4_k_m.gguf")
mmproj = Path("/path/to/Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf")

assert text_model.exists(), f"Text model not found: {text_model}"
assert vision_model.exists(), f"Vision model not found: {vision_model}"
assert mmproj.exists(), f"MMProj not found: {mmproj}"
```

### 2. LLM 로드 테스트

```python
from src.agent.base import LLMManager

# Config 준비
config = {
    "llm": {
        "text_model_path": "/path/to/Qwen3-8B-Q4_K_M.gguf",
        "vision_model_path": "/path/to/Qwen2.5-VL-7B-Instruct-q4_k_m.gguf",
        "vision_mmproj_path": "/path/to/Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf",
        "n_gpu_layers": -1,
        "n_ctx": 32768,
        "n_threads": 16,
        "n_batch": 512,
    },
    "gpu": {
        "device_id": 0,
    }
}

# LLMManager 초기화
llm_manager = LLMManager(config=config)

# 모델 로드
print("Loading Text LLM...")
if llm_manager.load_text_llm():
    print("✓ Text LLM loaded successfully")
else:
    print("✗ Text LLM load failed")

print("Loading Vision LLM...")
if llm_manager.load_vision_llm():
    print("✓ Vision LLM loaded successfully")
else:
    print("✗ Vision LLM load failed")
```

### 3. Function Calling 테스트

```python
from src.agent.flows.function_calling_support import FunctionCallingSupport
from app.e2e_system import E2ESystem, SystemConfig

# E2ESystem 초기화 (config.yaml에서 읽음)
config = SystemConfig(...)  # config.yaml에서 로드
system = E2ESystem(config)
system.initialize()

# Function Calling 테스트
if system.agent and system.agent.flow:
    flow = system.agent.flow
    if hasattr(flow, 'process_query'):
        result = flow.process_query("시스템 상태를 알려주세요")
        print(result)
```

## 문제 해결

### 모델 파일을 찾을 수 없음

- 경로가 올바른지 확인
- 절대 경로 사용 권장
- 파일 권한 확인

### GPU 메모리 부족

- `n_gpu_layers`를 줄이기 (예: 20)
- 배치 크기 줄이기 (`n_batch`: 256)
- 더 작은 양자화 모델 사용 (Q3_K_M, Q2_K)

### llama-cpp-python 설치 실패

- Python 버전 확인 (3.9+ 권장)
- CMake 설치 확인
- GPU 드라이버 확인 (CUDA/Metal)

### Function Calling이 작동하지 않음

- Text LLM이 올바르게 로드되었는지 확인
- Qwen3 모델인지 확인 (Qwen2는 Function Calling 미지원)
- `tools` 파라미터가 올바르게 전달되는지 확인

## 권장 모델 경로 구조

```
/path/to/models/
├── Qwen3-8B-Q4_K_M.gguf
├── Qwen2.5-VL-7B-Instruct-q4_k_m.gguf
└── Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf
```

또는

```
~/models/
├── Qwen3-8B-Q4_K_M.gguf
├── Qwen2.5-VL-7B-Instruct-q4_k_m.gguf
└── Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf
```

## 다음 단계

모델이 설치되고 연결되면:

1. `python app/run.py` 실행하여 전체 시스템 테스트
2. Function Calling 기능 테스트
3. 실제 비디오로 이상 탐지 테스트
