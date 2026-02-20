# 구현 체크리스트

**프로젝트**: AI CCTV 통합 시스템
**시작일**: 2025-01-21
**최종 업데이트**: 2026-02-20

---

## Sprint 1: 기반 구축 — 완료

### US-001: 데이터베이스 스키마 설계 및 구현

- [x] SQLite(dev) / PostgreSQL(prod) 듀얼 지원
- [x] SQLAlchemy 모델 정의
  - [x] `User` 모델
  - [x] `Camera` 모델
  - [x] `Event` 모델
  - [x] `DailyStatistics` 모델
  - [x] `CameraAccess` 모델
  - [x] `NotificationRule` 모델
- [x] Alembic 설정
- [x] 초기 마이그레이션 생성
- [x] 데이터베이스 연결 유틸리티 구현
- [x] 단위 테스트 작성

### US-002: FastAPI 프로젝트 구조 생성

- [x] `app/api/` 디렉토리 구조 생성
- [x] FastAPI 앱 초기화 (`main.py` + lifespan)
- [x] 라우터 모듈 생성
  - [x] `routers/cameras.py`
  - [x] `routers/events.py`
  - [x] `routers/stats.py`
  - [x] `routers/auth.py`
  - [x] `routers/stream.py`
  - [x] `routers/notifications.py`
- [x] 의존성 주입 설정 (`dependencies.py`)
- [x] CORS 미들웨어 설정
- [x] Swagger UI 접근 확인
- [x] 헬스체크 엔드포인트 (`GET /health`)

### US-003: 기본 인증 시스템 구현

- [x] 비밀번호 해싱 유틸리티 (bcrypt)
- [x] JWT 토큰 생성/검증 함수
- [x] 사용자 등록 API (`POST /api/v1/auth/register`)
- [x] 로그인 API (`POST /api/v1/auth/login`)
- [x] 토큰 갱신 API (`POST /api/v1/auth/refresh`)
- [x] 내 정보 API (`GET /api/v1/auth/me`)
- [x] 인증 미들웨어 구현 (HTTPBearer 의존성)
- [x] 통합 테스트 작성

### US-004: 카메라 CRUD API 구현

- [x] 카메라 목록 조회 (`GET /api/v1/cameras`)
- [x] 카메라 상세 조회 (`GET /api/v1/cameras/{id}`)
- [x] 카메라 생성 (`POST /api/v1/cameras`)
- [x] 카메라 수정 (`PUT /api/v1/cameras/{id}`)
- [x] 카메라 삭제 (`DELETE /api/v1/cameras/{id}`)
- [x] 카메라 시작 (`POST /api/v1/cameras/{id}/start`)
- [x] 카메라 중지 (`POST /api/v1/cameras/{id}/stop`)
- [x] 파이프라인 상태 (`GET /api/v1/cameras/{id}/pipeline-status`)
- [x] Pydantic 스키마 정의 (`schemas.py`)

### US-005: 이벤트 API 구현

- [x] 이벤트 목록 조회 (`GET /api/v1/events`)
- [x] 이벤트 상세 조회 (`GET /api/v1/events/{id}`)
- [x] 이벤트 확인 (`POST /api/v1/events/{id}/ack`)
- [x] 필터링 (camera_id, date, vlm_type, min_score)
- [x] 페이지네이션

---

## Sprint 2: 멀티카메라 & 파이프라인 통합 — 완료

### US-006: 멀티카메라 관리자 구현

- [x] `MultiCameraManager` 클래스
- [x] 카메라 추가/제거 (`add_camera`, `remove_camera`)
- [x] 카메라 상태 관리
- [x] `ResourcePool` (스레드-안전 모델 공유)
- [x] API ↔ MultiCameraManager 통합 (`pipeline_state.py`)

### US-007: 카메라별 독립 파이프라인 구현

- [x] `CameraPipeline` 클래스
- [x] 스레드 기반 병렬 처리
- [x] 에러 핸들링 및 복구
- [x] `AsyncEventLogger` (배치 DB 저장)

### US-008: 더미 모델 시스템

- [x] `DummyVADModel`
- [x] `DummyVLMAnalyzer`
- [x] `DummyAgentFlow`
- [x] `DummyVideoSource`
- [x] 환경변수 제어 (PIPELINE_DUMMY, PIPELINE_DUMMY_VLM, PIPELINE_DUMMY_AGENT)

### US-009: WebSocket 스트리밍

- [x] WebSocket 연결 핸들러 구현
- [x] WebSocket 매니저 (`websocket/manager.py`)
- [ ] React UI WebSocket 클라이언트 통합 (Sprint 3)

### US-010: 통계 API

- [x] 통계 summary API (`GET /api/v1/stats/summary`)

---

## Sprint 3 (일부): 알림 & 프론트엔드 통합 — 완료

### US-011~014: 알림 시스템

- [x] `NotificationChannel` 인터페이스
- [x] `ConsoleChannel`
- [x] `WebhookChannel` (Slack/Discord)
- [x] `EmailChannel` (SMTP)
- [x] `NotificationEngine` (규칙, cooldown, 중복 방지)
- [x] 알림 API (`/api/v1/notifications/`)
- [x] 12 tests 통과

### US-015: React UI — API 연동

- [x] API 클라이언트 (`ui/src/lib/api.ts`)
- [x] JWT 토큰 관리 + 자동 갱신
- [x] `AuthContext` + `LoginPage`
- [x] `LiveCameraGrid` API 연동 (start/stop)
- [x] `StatsDashboard` API 연동
- [x] Mock 데이터 제거

### Docker 배포

- [x] Dockerfile
- [x] docker-compose.yml
- [x] .dockerignore
- [x] .env.example

---

## Sprint 4: 미착수

### US-017: RBAC
- [ ] 역할 기반 접근 제어 (viewer, operator, admin)

### US-018: 카메라별 접근 권한
- [ ] `CameraAccess` 활용한 권한 관리

### US-019: 성능 모니터링
- [ ] Prometheus 메트릭

### US-020: 클립 관리
- [ ] 클립 다운로드/검색/자동 삭제

### US-021: 시스템 테스트
- [ ] E2E 자동화 테스트
- [ ] 부하 테스트
- [ ] 보안 테스트

---

## 진행 상황 추적

| Sprint | 상태 | 완료율 |
|--------|------|--------|
| Sprint 1 | 완료 | 100% |
| Sprint 2 | 완료 | 100% |
| Sprint 3 (알림/UI) | 완료 | 100% |
| Sprint 4 (RBAC/최적화) | 미착수 | 0% |

**총 테스트**: 58 PASSED
