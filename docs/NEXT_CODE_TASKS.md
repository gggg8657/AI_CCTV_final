# 다음 코드 개발 작업 목록

**작성일**: 2025-01-21  
**우선순위**: 높은 순서대로

---

## 🔴 우선순위 높음 (P0) - 즉시 시작

### 1. Alembic 마이그레이션 설정 (30분)

**목적**: 데이터베이스 스키마를 코드로 관리

**작업 내용**:
- [ ] `alembic.ini` 파일 생성 및 설정
- [ ] `alembic/env.py` 설정 (SQLAlchemy 모델 연결)
- [ ] 초기 마이그레이션 생성 (`alembic revision --autogenerate -m "initial"`)
- [ ] 마이그레이션 테스트 (`alembic upgrade head`)

**파일**:
- `alembic.ini`
- `alembic/env.py`
- `alembic/versions/001_initial.py`

**예상 시간**: 30분

---

### 2. EventLogger 확장 - 비동기 배치 저장 (2시간)

**목적**: 이벤트를 비동기로 데이터베이스에 저장

**작업 내용**:
- [ ] `AsyncEventLogger` 클래스 구현
  - [ ] 메모리 버퍼 (10개 또는 1초)
  - [ ] 백그라운드 스레드로 DB 저장
  - [ ] 기존 `EventLogger`와 호환성 유지
- [ ] `app/e2e_system.py`에서 `AsyncEventLogger` 사용하도록 수정
- [ ] 테스트 작성

**파일**:
- `src/database/event_logger.py` (새로 생성)
- `app/e2e_system.py` (수정)

**예상 시간**: 2시간

**핵심 로직**:
```python
class AsyncEventLogger(EventLogger):
    def __init__(self, log_dir, db_session):
        super().__init__(log_dir)
        self.db = db_session
        self.buffer = []
        self.buffer_lock = threading.Lock()
        # 백그라운드 스레드 시작
    
    def log_event(self, event):
        super().log_event(event)  # 기존 JSON 로그
        with self.buffer_lock:
            self.buffer.append(event)
    
    def _save_loop(self):
        # 1초마다 또는 10개 도달 시 DB 저장
```

---

### 3. ResourcePool 구현 (2시간)

**목적**: VAD/VLM/Agent 모델을 카메라 간 공유하여 메모리 절약

**작업 내용**:
- [ ] `ResourcePool` 클래스 구현
  - [ ] VAD 모델 공유 (타입별로 1개씩)
  - [ ] VLM 분석기 공유 (1개)
  - [ ] Agent Flow 공유 (타입별로 1개씩)
  - [ ] 스레드 안전한 락 사용 (`threading.RLock`)
- [ ] GPU 메모리 추적 기능
- [ ] 테스트 작성

**파일**:
- `src/pipeline/resource_pool.py` (새로 생성)

**예상 시간**: 2시간

**핵심 로직**:
```python
class ResourcePool:
    def __init__(self, gpu_id=0):
        self.vad_models = {}
        self.vlm_analyzer = None
        self.agent_flows = {}
        self.lock = threading.RLock()
    
    def get_vad_model(self, model_type):
        if model_type not in self.vad_models:
            with self.lock:
                if model_type not in self.vad_models:
                    self.vad_models[model_type] = create_vad_model(model_type)
        return self.vad_models[model_type]
```

---

### 4. MultiCameraManager 구현 (3시간)

**목적**: 여러 카메라를 동시에 관리하고 처리

**작업 내용**:
- [ ] `MultiCameraManager` 클래스 구현
  - [ ] E2ESystem 인스턴스 관리
  - [ ] ResourcePool 통합
  - [ ] 카메라 추가/삭제/수정
  - [ ] 카메라 시작/중지
  - [ ] 상태 관리
- [ ] `CameraPipeline` 클래스 구현
  - [ ] 카메라별 독립 파이프라인
  - [ ] 스레드 기반 실행
  - [ ] 에러 핸들링
- [ ] `CameraConfig`, `CameraStatus` 데이터 클래스
- [ ] 테스트 작성

**파일**:
- `src/pipeline/multi_camera.py` (기존 파일 확장)
- `src/pipeline/camera_pipeline.py` (새로 생성)
- `src/pipeline/camera_config.py` (새로 생성)

**예상 시간**: 3시간

---

## 🟡 우선순위 중간 (P1) - 다음 단계

### 5. 인증 시스템 구현 (2시간)

**목적**: 사용자 인증 및 JWT 토큰 관리

**작업 내용**:
- [ ] 비밀번호 해싱 유틸리티 (bcrypt)
- [ ] JWT 토큰 생성/검증 함수
- [ ] 사용자 등록 API 구현 (`POST /api/v1/auth/register`)
- [ ] 로그인 API 구현 (`POST /api/v1/auth/login`)
- [ ] 토큰 갱신 API 구현 (`POST /api/v1/auth/refresh`)
- [ ] 인증 미들웨어 구현 (의존성)

**파일**:
- `src/auth/password.py` (새로 생성)
- `src/auth/jwt.py` (새로 생성)
- `app/api/routers/auth.py` (구현)
- `app/api/dependencies.py` (새로 생성)

**예상 시간**: 2시간

---

### 6. 카메라 관리 API 구현 (3시간)

**목적**: 카메라 CRUD 및 제어 API

**작업 내용**:
- [ ] 카메라 목록 조회 (`GET /api/v1/cameras`)
- [ ] 카메라 상세 조회 (`GET /api/v1/cameras/{id}`)
- [ ] 카메라 생성 (`POST /api/v1/cameras`)
- [ ] 카메라 수정 (`PUT /api/v1/cameras/{id}`)
- [ ] 카메라 삭제 (`DELETE /api/v1/cameras/{id}`)
- [ ] 카메라 시작 (`POST /api/v1/cameras/{id}/start`)
- [ ] 카메라 중지 (`POST /api/v1/cameras/{id}/stop`)
- [ ] MultiCameraManager와 통합
- [ ] Pydantic 모델 정의

**파일**:
- `app/api/routers/cameras.py` (구현)
- `app/api/models/camera.py` (새로 생성)
- `app/api/dependencies.py` (MultiCameraManager 의존성)

**예상 시간**: 3시간

---

### 7. 이벤트 API 구현 (2시간)

**목적**: 이벤트 조회 및 확인 처리

**작업 내용**:
- [ ] 이벤트 목록 조회 (`GET /api/v1/events`)
  - [ ] 필터링 (camera_id, date, vlm_type, min_score)
  - [ ] 페이지네이션
- [ ] 이벤트 상세 조회 (`GET /api/v1/events/{id}`)
- [ ] 이벤트 확인 (`POST /api/v1/events/{id}/ack`)
- [ ] Pydantic 모델 정의

**파일**:
- `app/api/routers/events.py` (구현)
- `app/api/models/event.py` (새로 생성)

**예상 시간**: 2시간

---

### 8. 통계 API 구현 (2시간)

**목적**: 통계 조회 및 트렌드 분석

**작업 내용**:
- [ ] 통계 조회 API (`GET /api/v1/stats`)
  - [ ] 일별/주별/월별 통계
  - [ ] 카메라별 통계
- [ ] 통계 트렌드 API (`GET /api/v1/stats/trends`)
- [ ] 배치 작업 구현 (일별 통계 자동 집계)
- [ ] 캐싱 전략

**파일**:
- `app/api/routers/stats.py` (구현)
- `src/database/statistics.py` (새로 생성)

**예상 시간**: 2시간

---

## 🟢 우선순위 낮음 (P2) - 나중에

### 9. WebSocket 스트리밍 구현 (3시간)

**목적**: 실시간 프레임 및 이벤트 스트리밍

**작업 내용**:
- [ ] WebSocket 연결 관리
- [ ] 프레임 스트리밍 (5 FPS)
- [ ] 이벤트 브로드캐스트
- [ ] 통계 업데이트 브로드캐스트
- [ ] E2ESystem과 통합

**파일**:
- `app/api/routers/stream.py` (구현)
- `app/api/websocket/manager.py` (새로 생성)

**예상 시간**: 3시간

---

### 10. 알림 시스템 구현 (4시간)

**목적**: 이상 상황 발생 시 알림 발송

**작업 내용**:
- [ ] 알림 채널 추상화
- [ ] 이메일 알림 구현
- [ ] 웹훅 알림 구현
- [ ] 알림 규칙 엔진
- [ ] 우선순위 기반 발송
- [ ] 15초 중복 방지
- [ ] E2ESystem과 통합

**파일**:
- `src/notifications/base.py` (새로 생성)
- `src/notifications/email.py` (새로 생성)
- `src/notifications/webhook.py` (새로 생성)
- `src/notifications/engine.py` (새로 생성)

**예상 시간**: 4시간

---

## 📊 작업 우선순위 요약

### 즉시 시작 (오늘/내일)
1. ✅ **Alembic 마이그레이션 설정** (30분)
2. ✅ **EventLogger 확장** (2시간)
3. ✅ **ResourcePool 구현** (2시간)
4. ✅ **MultiCameraManager 구현** (3시간)

**총 예상 시간**: 7.5시간

### 다음 단계 (Sprint 1 후반)
5. 인증 시스템 구현 (2시간)
6. 카메라 관리 API 구현 (3시간)
7. 이벤트 API 구현 (2시간)
8. 통계 API 구현 (2시간)

**총 예상 시간**: 9시간

### Sprint 2-3
9. WebSocket 스트리밍 (3시간)
10. 알림 시스템 (4시간)

---

## 🎯 추천 시작 순서

### 오늘 할 수 있는 작업 (3-4시간)

1. **Alembic 마이그레이션 설정** (30분) - 빠르게 완료 가능
2. **EventLogger 확장** (2시간) - 핵심 기능
3. **ResourcePool 구현** (2시간) - 멀티 카메라 전제 조건

### 내일 할 작업 (3시간)

4. **MultiCameraManager 구현** (3시간) - 멀티 카메라 핵심

---

## 📝 각 작업별 상세 체크리스트

각 작업을 시작할 때 해당 체크리스트를 참고하세요:

- [ ] 필요한 파일 생성
- [ ] 기본 클래스/함수 구조 작성
- [ ] 핵심 로직 구현
- [ ] 에러 핸들링 추가
- [ ] 로깅 추가
- [ ] 단위 테스트 작성 (선택적)
- [ ] 통합 테스트 (선택적)
- [ ] 문서 업데이트

---

**다음 액션**: 위 작업 중 어떤 것부터 시작할지 알려주시면 바로 구현하겠습니다!
