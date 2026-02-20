# UI 테스트 가이드

## 준비

```bash
conda activate agent
pip install -r requirements.txt
cd ui && npm install && cd ..
```

## 방법 1: 더미 모드 (모델 없이 전체 동작)

```bash
# 백엔드
PIPELINE_DUMMY=true uvicorn app.api.main:app --port 8000

# React UI (별도 터미널)
cd ui
npm run dev
```

- React UI: http://localhost:3000
- Swagger UI: http://localhost:8000/docs
- Health: http://localhost:8000/health

## 방법 2: Docker

```bash
docker compose up --build
```

- API: http://localhost:8000
- Swagger: http://localhost:8000/docs

## 확인 가능한 기능

**모델 없이 확인 가능 (더미 모드):**
- 회원가입 / 로그인 (JWT)
- 카메라 등록 / 시작 / 중지
- 더미 이벤트 생성 및 조회
- 통계 대시보드
- 알림 규칙 설정 및 테스트
- API 전체 엔드포인트 (Swagger)

**실제 모델 필요:**
- 실제 영상 기반 VAD 이상 탐지
- VLM 상황 분석
- Agent 대응 계획 생성

## 자동 테스트

```bash
PIPELINE_DUMMY=true python -m pytest tests/ -v
# 58 tests 통과
```
