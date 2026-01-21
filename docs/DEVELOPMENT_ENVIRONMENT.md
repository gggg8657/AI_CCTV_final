# 개발 환경 가이드

**작성일**: 2025-01-21

---

## 개발 환경 구성

### Mac (개발 환경)
- **용도**: 코드 개발, Git 커밋/푸시
- **위치**: `/Users/gimdongju/Documents/workspace/secu/AI_CCTV_final`
- **작업**: 코드 작성, 문서 작성, Git 형상관리

### Windows PC (테스트/실행 환경)
- **용도**: 테스트 및 실제 동작
- **작업**: 시스템 테스트, 성능 테스트, 실제 카메라 연결 테스트

---

## 크로스 플랫폼 호환성 고려사항

### 1. 경로 처리

#### 문제점
- Mac: `/Users/...` (Unix 경로)
- Windows: `C:\Users\...` (Windows 경로)

#### 해결책
```python
# ❌ 나쁜 예
path = "/Users/gimdongju/models/model.gguf"

# ✅ 좋은 예
from pathlib import Path
path = Path.home() / "models" / "model.gguf"
# 또는
path = Path("models") / "model.gguf"  # 상대 경로
```

**확인 필요 파일**:
- `app/e2e_system.py` - 모델 경로 하드코딩 확인
- `src/vlm/analyzer.py` - 모델 경로 확인
- 모든 설정 파일의 경로 처리

---

### 2. 환경 변수

#### Mac에서 설정
```bash
# .env 파일 또는 환경 변수
export DATABASE_URL="postgresql://user:pass@localhost/ai_cctv"
export CUDA_VISIBLE_DEVICES=0
export VLM_MODEL_PATH="/Users/gimdongju/models/Qwen2.5-VL-7B.gguf"
```

#### Windows에서 설정
```powershell
# .env 파일 또는 환경 변수
$env:DATABASE_URL="postgresql://user:pass@localhost/ai_cctv"
$env:CUDA_VISIBLE_DEVICES=0
$env:VLM_MODEL_PATH="C:\Users\gimdongju\models\Qwen2.5-VL-7B.gguf"
```

**권장**: `.env` 파일 사용 (python-dotenv)

---

### 3. GPU 설정

#### Mac (개발)
- GPU 없거나 MPS (Metal Performance Shaders)
- CPU 모드로 개발 가능

#### Windows (실행)
- NVIDIA GPU (1060 등)
- CUDA 사용

**코드에서 처리**:
```python
import torch

if torch.cuda.is_available():
    device = f"cuda:{gpu_id}"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"  # Mac M1/M2
else:
    device = "cpu"
```

---

### 4. 의존성 관리

#### requirements.txt
- Mac과 Windows 모두에서 동일하게 작동하도록
- 플랫폼별 선택적 의존성은 `requirements-*.txt`로 분리

#### 예시
```txt
# requirements.txt (공통)
torch>=2.0.0
fastapi>=0.100.0
...

# requirements-mac.txt (Mac 전용)
# MPS 관련 패키지 등

# requirements-windows.txt (Windows 전용)
# CUDA 관련 패키지 등
```

---

### 5. 데이터베이스

#### Mac (개발)
- 로컬 PostgreSQL 또는 SQLite (개발용)

#### Windows (실행)
- PostgreSQL (실제 운영)

**권장**: 환경 변수로 `DATABASE_URL` 관리

---

### 6. 파일 시스템 차이

#### 대소문자 구분
- Mac/Linux: 대소문자 구분
- Windows: 대소문자 구분 안 함 (기본)

**해결책**: 파일명은 소문자와 언더스코어 사용
```
✅ 좋은 예: event_logger.py, camera_config.py
❌ 나쁜 예: EventLogger.py, CameraConfig.py
```

#### 라인 엔딩
- Mac/Linux: `\n` (LF)
- Windows: `\r\n` (CRLF)

**해결책**: Git 설정으로 자동 변환
```bash
git config core.autocrlf input  # Mac
git config core.autocrlf true   # Windows
```

---

## 개발 워크플로우

### Mac에서 개발
```bash
# 1. 코드 작성
vim app/api/main.py

# 2. Git 커밋
git add .
git commit -m "feat: ... [CHA-XX]"

# 3. 푸시
git push origin main
```

### Windows에서 테스트
```bash
# 1. 저장소 클론/업데이트
git pull origin main

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경 변수 설정
# .env 파일 또는 환경 변수 설정

# 4. 테스트 실행
python -m pytest tests/
python app/api/main.py
```

---

## 환경별 설정 파일

### .env.example (템플릿)
```bash
# 데이터베이스
DATABASE_URL=postgresql://user:password@localhost/ai_cctv

# GPU
CUDA_VISIBLE_DEVICES=0

# 모델 경로 (환경별로 다름)
# Mac
VLM_MODEL_PATH=/Users/gimdongju/models/Qwen2.5-VL-7B.gguf
# Windows
# VLM_MODEL_PATH=C:\Users\gimdongju\models\Qwen2.5-VL-7B.gguf

# API
API_HOST=0.0.0.0
API_PORT=8000
```

### .gitignore에 추가
```
.env
.env.local
.env.mac
.env.windows
```

---

## 테스트 전략

### Mac에서 할 수 있는 테스트
- 단위 테스트 (비즈니스 로직)
- API 엔드포인트 테스트 (Mock 데이터)
- 코드 스타일 검사
- 타입 체크

### Windows에서 해야 하는 테스트
- 통합 테스트 (실제 모델 사용)
- 성능 테스트 (GPU 사용)
- 실제 카메라 연결 테스트
- E2E 테스트

---

## 주의사항

### Mac에서 개발 시
1. ✅ 경로는 `pathlib.Path` 사용
2. ✅ 하드코딩된 경로 제거
3. ✅ 환경 변수 사용
4. ✅ 상대 경로 사용

### Windows에서 테스트 시
1. ✅ 환경 변수 확인
2. ✅ 모델 경로 확인
3. ✅ GPU 드라이버 확인
4. ✅ 의존성 버전 확인

---

## 체크리스트

### 코드 작성 시
- [ ] 하드코딩된 경로 없음
- [ ] `pathlib.Path` 사용
- [ ] 환경 변수 사용
- [ ] 플랫폼별 분기 처리 (필요시)

### Windows 테스트 전
- [ ] Git 최신 버전 pull
- [ ] 의존성 설치 확인
- [ ] 환경 변수 설정 확인
- [ ] 모델 파일 경로 확인

---

**다음 단계**: 코드 개발 시 위 사항들을 고려하여 크로스 플랫폼 호환성을 유지하겠습니다.
