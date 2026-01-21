# Windows 테스트 가이드

**작성일**: 2025-01-21

---

## Windows 환경 설정

### 1. 저장소 클론

```powershell
# Git 저장소 클론
git clone https://github.com/gggg8657/AI_CCTV_final.git
cd AI_CCTV_final
```

### 2. Python 환경 설정

```powershell
# 가상 환경 생성 (권장)
python -m venv venv

# 가상 환경 활성화
.\venv\Scripts\Activate.ps1

# 의존성 설치
pip install -r requirements.txt
```

### 3. 환경 변수 설정

#### 방법 1: .env 파일 사용 (권장)

```powershell
# .env.example을 .env로 복사
Copy-Item .env.example .env

# .env 파일 편집 (메모장 또는 에디터)
notepad .env
```

`.env` 파일 내용 예시:
```bash
# 데이터베이스
DATABASE_URL=postgresql://user:password@localhost/ai_cctv

# GPU
CUDA_VISIBLE_DEVICES=0

# VLM 모델 경로 (Windows 경로)
VLM_MODEL_PATH=C:\Users\gimdongju\models\Qwen2.5-VL-7B-Instruct-q4_k_m.gguf
VLM_MMPROJ_PATH=C:\Users\gimdongju\models\Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf

# Agent 모델 경로
AGENT_TEXT_MODEL_PATH=C:\Users\gimdongju\models\Qwen3-8B-Q4_K_M.gguf

# API 서버
API_HOST=0.0.0.0
API_PORT=8000
```

#### 방법 2: 환경 변수 직접 설정

```powershell
# PowerShell에서
$env:DATABASE_URL="postgresql://user:password@localhost/ai_cctv"
$env:CUDA_VISIBLE_DEVICES="0"
$env:VLM_MODEL_PATH="C:\Users\gimdongju\models\Qwen2.5-VL-7B-Instruct-q4_k_m.gguf"
```

### 4. 데이터베이스 설정

#### PostgreSQL 설치 및 실행
```powershell
# PostgreSQL이 설치되어 있다면
# 서비스 시작 확인
Get-Service postgresql*

# 또는 수동 시작
net start postgresql-x64-XX
```

#### 데이터베이스 생성
```sql
-- psql에서 실행
CREATE DATABASE ai_cctv;
```

### 5. 모델 파일 확인

모델 파일이 다음 경로에 있는지 확인:
- VLM 모델: `C:\Users\gimdongju\models\Qwen2.5-VL-7B-Instruct-q4_k_m.gguf`
- VLM mmproj: `C:\Users\gimdongju\models\Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf`
- Agent 모델: `C:\Users\gimdongju\models\Qwen3-8B-Q4_K_M.gguf`

경로가 다르면 `.env` 파일에서 수정

---

## 테스트 실행

### 1. 데이터베이스 마이그레이션

```powershell
# Alembic 마이그레이션 실행
alembic upgrade head
```

### 2. API 서버 실행

```powershell
# FastAPI 서버 시작
python -m app.api.main

# 또는 uvicorn 직접 사용
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**접속**: http://localhost:8000/docs (Swagger UI)

### 3. E2E 시스템 테스트

```powershell
# CLI 모드로 테스트
python app/run.py --mode cli --source "C:\path\to\test_video.mp4"

# Web UI 모드
python app/run.py --mode web --source "C:\path\to\test_video.mp4"
```

### 4. 단위 테스트

```powershell
# pytest 실행
pytest tests/

# 특정 테스트만
pytest tests/test_vad.py
```

---

## 문제 해결

### 문제 1: 모델 경로를 찾을 수 없음

**증상**: `VLM model path not found`

**해결**:
1. `.env` 파일에서 `VLM_MODEL_PATH` 확인
2. 모델 파일이 실제로 존재하는지 확인
3. 경로에 한글이나 특수문자 없는지 확인

### 문제 2: CUDA 오류

**증상**: `CUDA out of memory` 또는 `CUDA not available`

**해결**:
1. NVIDIA 드라이버 설치 확인
2. CUDA 버전 확인: `nvidia-smi`
3. `CUDA_VISIBLE_DEVICES` 환경 변수 확인
4. GPU 메모리 확인 (다른 프로세스가 사용 중인지)

### 문제 3: 데이터베이스 연결 실패

**증상**: `Connection refused` 또는 `Authentication failed`

**해결**:
1. PostgreSQL 서비스 실행 확인
2. `DATABASE_URL` 환경 변수 확인
3. 사용자명/비밀번호 확인
4. 방화벽 설정 확인

### 문제 4: 의존성 설치 실패

**증상**: `pip install` 실패

**해결**:
1. Python 버전 확인 (3.10+ 필요)
2. pip 업그레이드: `python -m pip install --upgrade pip`
3. Visual C++ 빌드 도구 필요할 수 있음 (Windows)

---

## 성능 테스트

### GPU 사용량 확인

```powershell
# nvidia-smi로 GPU 사용량 모니터링
nvidia-smi -l 1
```

### 메모리 사용량 확인

```powershell
# 작업 관리자에서 확인
# 또는 PowerShell
Get-Process python | Select-Object ProcessName, @{Name="Memory(MB)";Expression={$_.WS/1MB}}
```

---

## 체크리스트

### 테스트 전 확인
- [ ] Git 최신 버전 pull 완료
- [ ] 가상 환경 활성화
- [ ] 의존성 설치 완료
- [ ] `.env` 파일 설정 완료
- [ ] 모델 파일 경로 확인
- [ ] PostgreSQL 실행 중
- [ ] 데이터베이스 마이그레이션 완료

### 테스트 실행
- [ ] API 서버 실행 성공
- [ ] Swagger UI 접속 가능
- [ ] E2E 시스템 실행 성공
- [ ] GPU 사용 확인
- [ ] 데이터베이스 저장 확인

---

## Mac과 Windows 간 동기화

### Mac에서 작업 후
```bash
# 커밋 및 푸시
git add .
git commit -m "feat: ... [CHA-XX]"
git push origin main
```

### Windows에서 업데이트
```powershell
# 최신 버전 가져오기
git pull origin main

# 의존성 업데이트 (필요시)
pip install -r requirements.txt --upgrade
```

---

**다음 단계**: Mac에서 코드 개발 후 Windows에서 테스트하는 워크플로우로 진행합니다.
