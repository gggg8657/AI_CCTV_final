# 구현 체크리스트

**프로젝트**: AI CCTV 통합 시스템  
**시작일**: 2025-01-21

---

## Sprint 1: 기반 구축 (Week 1-2)

### US-001: 데이터베이스 스키마 설계 및 구현

- [ ] PostgreSQL 데이터베이스 설정
- [ ] SQLAlchemy 모델 정의
  - [ ] `User` 모델
  - [ ] `Camera` 모델
  - [ ] `Event` 모델
  - [ ] `DailyStatistics` 모델
  - [ ] `CameraAccess` 모델
  - [ ] `NotificationRule` 모델
- [ ] Alembic 설정
- [ ] 초기 마이그레이션 생성
- [ ] 데이터베이스 연결 유틸리티 구현
- [ ] 테스트 데이터 생성 스크립트 작성
- [ ] 단위 테스트 작성

**파일**:
- `src/database/models.py`
- `src/database/db.py`
- `alembic.ini`
- `alembic/versions/001_initial.py`

---

### US-002: FastAPI 프로젝트 구조 생성

- [ ] `app/api/` 디렉토리 구조 생성
- [ ] FastAPI 앱 초기화 (`main.py`)
- [ ] 라우터 모듈 생성
  - [ ] `routers/cameras.py`
  - [ ] `routers/events.py`
  - [ ] `routers/stats.py`
  - [ ] `routers/auth.py`
  - [ ] `routers/stream.py`
- [ ] 의존성 주입 설정 (`dependencies.py`)
- [ ] CORS 미들웨어 설정
- [ ] Swagger UI 접근 확인
- [ ] 기본 헬스체크 엔드포인트 (`GET /health`)

**파일**:
- `app/api/main.py`
- `app/api/dependencies.py`
- `app/api/routers/__init__.py`
- `app/api/routers/cameras.py`
- `app/api/routers/events.py`
- `app/api/routers/stats.py`
- `app/api/routers/auth.py`
- `app/api/routers/stream.py`

---

### US-003: 기본 인증 시스템 구현

- [ ] 사용자 모델에 `password_hash` 필드 추가
- [ ] 비밀번호 해싱 유틸리티 구현 (bcrypt)
- [ ] JWT 토큰 생성 함수 구현
- [ ] JWT 토큰 검증 함수 구현
- [ ] 사용자 등록 API (`POST /api/v1/auth/register`)
- [ ] 로그인 API (`POST /api/v1/auth/login`)
- [ ] 토큰 갱신 API (`POST /api/v1/auth/refresh`)
- [ ] 인증 미들웨어 구현 (의존성)
- [ ] 단위 테스트 작성
- [ ] 통합 테스트 작성

**파일**:
- `src/auth/password.py`
- `src/auth/jwt.py`
- `app/api/routers/auth.py`
- `app/api/dependencies.py` (인증 의존성)

---

### US-004: 카메라 CRUD API 구현

- [ ] 카메라 목록 조회 API (`GET /api/v1/cameras`)
- [ ] 카메라 상세 조회 API (`GET /api/v1/cameras/{id}`)
- [ ] 카메라 생성 API (`POST /api/v1/cameras`)
- [ ] 카메라 수정 API (`PUT /api/v1/cameras/{id}`)
- [ ] 카메라 삭제 API (`DELETE /api/v1/cameras/{id}`)
- [ ] Pydantic 모델 정의 (요청/응답)
- [ ] 권한 검사 로직 구현
- [ ] 단위 테스트 작성
- [ ] API 문서 확인

**파일**:
- `app/api/routers/cameras.py`
- `app/api/models/camera.py`

---

### US-005: 이벤트 저장 API 구현

- [ ] 이벤트 목록 조회 API (`GET /api/v1/events`)
- [ ] 이벤트 상세 조회 API (`GET /api/v1/events/{id}`)
- [ ] 이벤트 확인 API (`POST /api/v1/events/{id}/ack`)
- [ ] 필터링 로직 구현 (camera_id, date, vlm_type 등)
- [ ] 페이지네이션 구현
- [ ] Pydantic 모델 정의
- [ ] 단위 테스트 작성

**파일**:
- `app/api/routers/events.py`
- `app/api/models/event.py`

---

## Sprint 2: 멀티 카메라 & API 확장 (Week 3-4)

### US-006: 멀티 카메라 관리자 구현

- [ ] `MultiCameraManager` 클래스 구현
- [ ] 카메라 레지스트리 구현
- [ ] 카메라 추가 로직 (`add_camera`)
- [ ] 카메라 제거 로직 (`remove_camera`)
- [ ] 카메라 상태 관리 시스템
- [ ] 리소스 풀 관리자 구현
- [ ] GPU 메모리 추적기 구현
- [ ] 단위 테스트 작성

**파일**:
- `src/pipeline/multi_camera.py`
- `src/pipeline/resource_pool.py`
- `src/pipeline/camera_config.py`

---

### US-007: 카메라별 독립 파이프라인 구현

- [ ] `CameraPipeline` 클래스 구현
- [ ] 카메라별 E2ESystem 인스턴스 생성
- [ ] 스레드 기반 병렬 처리
- [ ] 이벤트 큐 시스템 구현
- [ ] 에러 핸들링 및 복구 메커니즘
- [ ] 리소스 사용량 모니터링
- [ ] 통합 테스트 작성

**파일**:
- `src/pipeline/camera_pipeline.py`

---

### US-008: 통합 대시보드 API 구현

- [ ] 모든 카메라 상태 조회 API
- [ ] 실시간 통계 집계
- [ ] 이벤트 집계 및 우선순위 처리
- [ ] 성능 메트릭 수집
- [ ] 단위 테스트 작성

**파일**:
- `app/api/routers/dashboard.py`

---

### US-009: WebSocket 스트리밍 구현

- [ ] WebSocket 연결 핸들러 구현
- [ ] 프레임 스트리밍 로직
- [ ] 이벤트 브로드캐스트 로직
- [ ] 통계 업데이트 브로드캐스트
- [ ] 연결 관리 (구독/해제)
- [ ] 에러 핸들링
- [ ] 통합 테스트 작성

**파일**:
- `app/api/routers/stream.py`
- `app/api/websocket/manager.py`

---

### US-010: 통계 집계 API 구현

- [ ] 일별 통계 집계 API (`GET /api/v1/stats`)
- [ ] 트렌드 조회 API (`GET /api/v1/stats/trends`)
- [ ] 배치 작업 구현 (일별 통계 자동 집계)
- [ ] 캐싱 전략 구현
- [ ] 단위 테스트 작성

**파일**:
- `app/api/routers/stats.py`
- `src/database/statistics.py`

---

## Sprint 3: 알림 시스템 & 프론트엔드 통합 (Week 5-6)

### US-011: 알림 채널 추상화

- [ ] `NotificationChannel` 인터페이스 정의
- [ ] 채널 팩토리 구현
- [ ] 채널 등록 시스템
- [ ] 단위 테스트 작성

**파일**:
- `src/notifications/base.py`
- `src/notifications/factory.py`

---

### US-012: 이메일 알림 구현

- [ ] `EmailChannel` 클래스 구현
- [ ] SMTP 설정 관리
- [ ] 이메일 템플릿 시스템
- [ ] HTML 이메일 지원
- [ ] 단위 테스트 작성

**파일**:
- `src/notifications/email.py`
- `src/notifications/templates/email.html`

---

### US-013: 웹훅 알림 구현

- [ ] `WebhookChannel` 클래스 구현
- [ ] HTTP POST 요청 로직
- [ ] 재시도 메커니즘
- [ ] 타임아웃 처리
- [ ] 단위 테스트 작성

**파일**:
- `src/notifications/webhook.py`

---

### US-014: 알림 규칙 엔진 구현

- [ ] `NotificationEngine` 클래스 구현
- [ ] 규칙 매칭 로직
- [ ] 메시지 생성 로직
- [ ] 채널 디스패처 구현
- [ ] E2E 시스템과 통합
- [ ] 단위 테스트 작성
- [ ] 통합 테스트 작성

**파일**:
- `src/notifications/engine.py`
- `src/notifications/rules.py`

---

### US-015: React UI - API 연동

- [ ] API 클라이언트 라이브러리 생성 (`src/lib/api.ts`)
- [ ] Axios 설정
- [ ] 인증 토큰 관리
- [ ] 카메라 목록 조회 연동
- [ ] 이벤트 목록 조회 연동
- [ ] 통계 데이터 조회 연동
- [ ] 에러 핸들링 구현
- [ ] 로딩 상태 관리
- [ ] Mock 데이터 제거

**파일**:
- `ui/src/lib/api.ts`
- `ui/src/lib/auth.ts`
- `ui/src/components/LiveCameraGrid.tsx` (수정)
- `ui/src/components/AIAnalysisPanel.tsx` (수정)
- `ui/src/components/StatsDashboard.tsx` (수정)

---

### US-016: 실시간 대시보드 통합

- [ ] WebSocket 클라이언트 구현
- [ ] 실시간 프레임 업데이트
- [ ] 실시간 이벤트 알림
- [ ] 실시간 통계 업데이트
- [ ] 연결 관리 (재연결 로직)
- [ ] 통합 테스트 작성

**파일**:
- `ui/src/lib/websocket.ts`
- `ui/src/components/LiveCameraGrid.tsx` (WebSocket 통합)

---

## Sprint 4: 권한 관리 & 최적화 (Week 7-8)

### US-017: 역할 기반 접근 제어 (RBAC) 구현

- [ ] 역할 정의 (viewer, operator, admin)
- [ ] 권한 체크 미들웨어 구현
- [ ] API 엔드포인트별 권한 설정
- [ ] 단위 테스트 작성

**파일**:
- `app/api/dependencies.py` (권한 체크)
- `src/auth/rbac.py`

---

### US-018: 카메라별 접근 권한 관리

- [ ] `CameraAccess` 모델 활용
- [ ] 카메라 접근 권한 체크 로직
- [ ] 사용자별 접근 가능 카메라 필터링
- [ ] 권한 부여/해제 API
- [ ] 단위 테스트 작성

**파일**:
- `app/api/routers/camera_access.py`
- `app/api/dependencies.py` (카메라 접근 체크)

---

### US-019: 성능 모니터링 도구 추가

- [ ] Prometheus 메트릭 수집
- [ ] 성능 메트릭 정의
  - [ ] API 응답 시간
  - [ ] 처리량 (throughput)
  - [ ] 에러율
  - [ ] 리소스 사용량
- [ ] 메트릭 엔드포인트 (`/metrics`)
- [ ] Grafana 대시보드 설정 (선택적)

**파일**:
- `app/api/monitoring.py`
- `prometheus.yml` (설정 파일)

---

### US-020: 클립 관리 시스템 구현

- [ ] 클립 목록 조회 API
- [ ] 클립 다운로드 API
- [ ] 클립 검색 및 필터링
- [ ] 썸네일 생성
- [ ] 자동 삭제 정책 구현
- [ ] 단위 테스트 작성

**파일**:
- `app/api/routers/clips.py`
- `src/utils/clip_manager.py`

---

### US-021: 시스템 테스트 및 버그 수정

- [ ] 전체 시스템 통합 테스트
- [ ] 성능 테스트
- [ ] 부하 테스트
- [ ] 보안 테스트
- [ ] 버그 수정
- [ ] 문서 업데이트

---

## 공통 작업

### 문서화
- [ ] API 문서 자동 생성 확인
- [ ] README 업데이트
- [ ] 설치 가이드 작성
- [ ] 배포 가이드 작성

### 테스트
- [ ] 단위 테스트 커버리지 > 70%
- [ ] 통합 테스트 작성
- [ ] E2E 테스트 작성

### 코드 품질
- [ ] 코드 리뷰
- [ ] 린터 통과 (flake8, black)
- [ ] 타입 힌트 추가
- [ ] 주석 작성

### 배포 준비
- [ ] Docker 이미지 빌드
- [ ] Docker Compose 설정
- [ ] 환경 변수 문서화
- [ ] 배포 스크립트 작성

---

## 진행 상황 추적

### Sprint 1
- 시작일: 2025-01-21
- 종료일: 2025-02-04
- 완료율: 0%

### Sprint 2
- 시작일: 2025-02-04
- 종료일: 2025-02-18
- 완료율: 0%

### Sprint 3
- 시작일: 2025-02-18
- 종료일: 2025-03-04
- 완료율: 0%

### Sprint 4
- 시작일: 2025-03-04
- 종료일: 2025-03-18
- 완료율: 0%

---

**최종 업데이트**: 2025-01-21
