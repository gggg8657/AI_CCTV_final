# 작업 진행 상황

**최종 업데이트**: 2026-02-20

---

## 스프린트 상태

| 스프린트 | 범위 | 상태 |
|----------|------|------|
| Sprint 1 | 프로젝트 구조, DB 스키마, FastAPI 기본, 인증, API 라우터 | 완료 |
| Sprint 2 | 멀티카메라 파이프라인, 더미 모델, API-Pipeline 통합, 알림, Docker, React UI 통합 | 완료 |
| Sprint 3 | E2E 테스트 자동화, WebSocket 실시간, GPU Docker | 계획 중 |

---

## 완료된 작업 (Sprint 1-2)

### 백엔드 — FastAPI + DB
- [x] FastAPI 앱 구조 (main.py, lifespan, routers)
- [x] SQLAlchemy ORM 모델 (User, Camera, Event, DailyStatistics, NotificationRule)
- [x] Alembic 마이그레이션
- [x] JWT 인증 (register, login, refresh, me)
- [x] Camera CRUD + start/stop + pipeline-status
- [x] Event 조회 (필터/페이징/확인)
- [x] Stats summary
- [x] Notification CRUD + 테스트 발송
- [x] WebSocket 스트리밍 기본 구조

### 파이프라인
- [x] ResourcePool (스레드-안전 모델 공유)
- [x] CameraPipeline (독립 처리 스레드)
- [x] MultiCameraManager (생명주기 관리)
- [x] AsyncEventLogger (배치 DB 저장)
- [x] API ↔ MultiCameraManager 통합 (pipeline_state.py)

### 더미 모델
- [x] DummyVADModel, DummyVLMAnalyzer, DummyAgentFlow, DummyVideoSource
- [x] 환경변수 제어 (PIPELINE_DUMMY, PIPELINE_DUMMY_VLM, PIPELINE_DUMMY_AGENT)

### 알림 시스템
- [x] NotificationChannel (base), ConsoleChannel, WebhookChannel, EmailChannel
- [x] NotificationEngine (규칙, cooldown, 중복 방지)

### Docker
- [x] Dockerfile + docker-compose.yml
- [x] .dockerignore, .env.example

### React UI
- [x] JWT API 클라이언트 (자동 토큰 갱신)
- [x] AuthContext + LoginPage
- [x] LiveCameraGrid (실제 API 연동, start/stop)
- [x] StatsDashboard (실제 API 연동)
- [x] Vite proxy 설정 (/api, /health, /ws)

### 테스트
- [x] API smoke: 18 tests
- [x] Dummy pipeline: 7 tests
- [x] Sprint 2 pipeline: 15 tests
- [x] API-Pipeline integration: 6 tests
- [x] Notifications: 12 tests
- [x] **총 58 tests PASSED**

---

## 남은 작업 후보

### E2E 테스트 자동화
- Docker compose 기반 통합 테스트
- GitHub Actions CI/CD

### Docker GPU 지원
- nvidia-docker / CUDA runtime
- GPU 스케줄링

### WebSocket 실시간 이벤트 피드
- 이벤트 발생 시 실시간 브로드캐스트
- React UI 실시간 알림 토스트

### 실제 모델 통합
- VAD 모델 가중치 연결 (MNAD 등)
- VLM 모델 배포 (Qwen2.5-VL)
- Agent LLM 서빙 (Qwen3-8B / API)
