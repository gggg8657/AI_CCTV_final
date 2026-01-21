# 구현 계획 (결정 사항 반영)

**작성일**: 2025-01-21  
**하드웨어**: GPU 1060 (6GB VRAM)

---

## Phase 1: 기반 구축 (Sprint 1)

### 1. 프로젝트 구조 생성 (30분) ✅ 즉시 시작 가능

```bash
mkdir -p app/api/{routers,models}
mkdir -p src/database
mkdir -p src/notifications
mkdir -p tests/{unit,integration}
```

**파일 생성**:
- `app/api/__init__.py`
- `app/api/main.py` (FastAPI 앱)
- `src/database/__init__.py`
- `src/database/db.py` (DB 연결)
- `src/database/models.py` (SQLAlchemy 모델)

---

### 2. FastAPI 기본 구조 (1시간)

**구현 내용**:
- FastAPI 앱 초기화
- CORS 설정
- 기본 라우터 구조
- Swagger UI 확인
- 헬스체크 엔드포인트

**파일**:
- `app/api/main.py`
- `app/api/routers/__init__.py`

---

### 3. 데이터베이스 스키마 설계 및 구현 (2시간)

**구현 내용**:
- SQLAlchemy 모델 정의
  - `User`, `Camera`, `Event`, `DailyStatistics`, `CameraAccess`, `NotificationRule`
- Alembic 설정
- 초기 마이그레이션 생성
- DB 연결 유틸리티

**파일**:
- `src/database/models.py`
- `src/database/db.py`
- `alembic.ini`
- `alembic/versions/001_initial.py`

---

### 4. EventLogger 확장 - 비동기 배치 저장 (2시간)

**구현 내용**:
- `AsyncEventLogger` 클래스 생성
- 메모리 버퍼 (10개 또는 1초)
- 백그라운드 스레드로 DB 저장
- 기존 `EventLogger`와 호환성 유지

**파일**:
- `app/e2e_system.py` (수정)
- `src/database/event_logger.py` (새로 생성)

**구현 예시**:
```python
class AsyncEventLogger(EventLogger):
    """비동기 배치 저장 이벤트 로거"""
    
    def __init__(self, log_dir: str, db_session, log_level: str = "INFO"):
        super().__init__(log_dir, log_level)
        self.db = db_session
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.last_save_time = time.time()
        self.save_thread = threading.Thread(target=self._save_loop, daemon=True)
        self.save_thread.start()
    
    def log_event(self, event: AnomalyEvent):
        """이벤트 버퍼에 추가"""
        super().log_event(event)  # 기존 JSON 로그도 유지
        
        with self.buffer_lock:
            self.buffer.append(event)
    
    def _save_loop(self):
        """백그라운드에서 주기적으로 DB 저장"""
        while True:
            time.sleep(1)  # 1초마다 체크
            
            with self.buffer_lock:
                now = time.time()
                should_save = (
                    len(self.buffer) >= 10 or 
                    (now - self.last_save_time) >= 1.0
                )
                
                if should_save:
                    events_to_save = self.buffer[:10]
                    self.buffer = self.buffer[10:]
                    self.last_save_time = now
                    
                    # DB 저장
                    self._save_to_db(events_to_save)
    
    def _save_to_db(self, events: List[AnomalyEvent]):
        """이벤트를 DB에 저장"""
        try:
            for event in events:
                db_event = Event(
                    camera_id=event.metadata.get("camera_id"),
                    timestamp=event.timestamp,
                    frame_number=event.frame_number,
                    vad_score=event.vad_score,
                    threshold=event.threshold,
                    vlm_type=event.vlm_type,
                    vlm_description=event.vlm_description,
                    vlm_confidence=event.vlm_confidence,
                    agent_actions=json.dumps(event.agent_actions),
                    agent_response_time=event.agent_response_time,
                    clip_path=event.clip_path
                )
                self.db.add(db_event)
            
            self.db.commit()
        except Exception as e:
            self.logger.error(f"Failed to save events to DB: {e}")
            self.db.rollback()
```

---

### 5. ResourcePool 구현 (2시간)

**구현 내용**:
- VAD 모델 공유 (타입별로 1개씩)
- VLM 분석기 공유 (1개)
- Agent Flow 공유 (타입별로 1개씩)
- 스레드 안전한 락 사용
- GPU 메모리 추적

**파일**:
- `src/pipeline/resource_pool.py` (새로 생성)

**구현 예시**:
```python
class ResourcePool:
    """리소스 풀 - 모델 공유 관리"""
    
    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.vad_models: Dict[str, VADModel] = {}
        self.vlm_analyzer: Optional[VLMAnalyzer] = None
        self.agent_flows: Dict[str, AgentFlow] = {}
        self.lock = threading.RLock()  # 재진입 가능한 락
    
    def get_vad_model(self, model_type: str) -> VADModel:
        """VAD 모델 가져오기 (없으면 생성)"""
        if model_type not in self.vad_models:
            with self.lock:
                # Double-check locking
                if model_type not in self.vad_models:
                    from src.vad import create_model
                    self.vad_models[model_type] = create_model(
                        model_type,
                        gpu_id=self.gpu_id
                    )
        return self.vad_models[model_type]
    
    def get_vlm_analyzer(self) -> VLMAnalyzer:
        """VLM 분석기 가져오기 (싱글톤)"""
        if self.vlm_analyzer is None:
            with self.lock:
                if self.vlm_analyzer is None:
                    from src.vlm import VLMAnalyzer
                    self.vlm_analyzer = VLMAnalyzer(
                        gpu_id=self.gpu_id,
                        optimize_speed=True
                    )
                    self.vlm_analyzer.initialize()
        return self.vlm_analyzer
    
    def get_agent_flow(self, flow_type: str) -> AgentFlow:
        """Agent Flow 가져오기 (없으면 생성)"""
        if flow_type not in self.agent_flows:
            with self.lock:
                if flow_type not in self.agent_flows:
                    from src.agent import create_flow
                    self.agent_flows[flow_type] = create_flow(
                        flow_type,
                        gpu_id=self.gpu_id
                    )
                    self.agent_flows[flow_type].initialize()
        return self.agent_flows[flow_type]
    
    def check_gpu_memory(self) -> Dict[str, float]:
        """GPU 메모리 사용량 확인"""
        if not HAS_TORCH or not torch.cuda.is_available():
            return {}
        
        return {
            "allocated_mb": torch.cuda.memory_allocated(self.gpu_id) / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved(self.gpu_id) / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated(self.gpu_id) / 1024**2
        }
```

---

### 6. MultiCameraManager 구현 (3시간)

**구현 내용**:
- E2ESystem 인스턴스 관리
- ResourcePool 통합
- 카메라 추가/삭제/수정
- 상태 관리
- 에러 핸들링

**파일**:
- `src/pipeline/multi_camera.py` (기존 파일 확장)
- `src/pipeline/camera_pipeline.py` (새로 생성)

**구현 예시**:
```python
class MultiCameraManager:
    """멀티 카메라 관리자"""
    
    def __init__(self, max_cameras: int = 16, gpu_id: int = 0):
        self.max_cameras = max_cameras
        self.gpu_id = gpu_id
        self.cameras: Dict[int, CameraPipeline] = {}
        self.resource_pool = ResourcePool(gpu_id=gpu_id)
        self.lock = threading.Lock()
        self.db_session = None  # 나중에 주입
    
    def add_camera(self, camera_config: CameraConfig) -> int:
        """카메라 추가"""
        with self.lock:
            if len(self.cameras) >= self.max_cameras:
                raise ValueError(f"Maximum {self.max_cameras} cameras allowed")
            
            camera_id = camera_config.id
            
            # E2ESystem 생성 (ResourcePool 사용)
            e2e_config = SystemConfig(
                source_type=camera_config.source_type,
                source_path=camera_config.source_path,
                vad_model=camera_config.vad_model,
                vad_threshold=camera_config.vad_threshold,
                enable_vlm=camera_config.enable_vlm,
                enable_agent=camera_config.enable_agent,
                agent_flow=camera_config.agent_flow,
                gpu_id=self.gpu_id
            )
            
            # EventLogger에 DB 세션 주입
            event_logger = AsyncEventLogger(
                log_dir=f"logs/camera_{camera_id}",
                db_session=self.db_session
            )
            
            # CameraPipeline 생성
            pipeline = CameraPipeline(
                camera_id=camera_id,
                config=camera_config,
                e2e_config=e2e_config,
                resource_pool=self.resource_pool,
                event_logger=event_logger
            )
            
            self.cameras[camera_id] = pipeline
            return camera_id
    
    def start_camera(self, camera_id: int) -> bool:
        """카메라 시작"""
        with self.lock:
            if camera_id not in self.cameras:
                return False
            return self.cameras[camera_id].start()
    
    def stop_camera(self, camera_id: int) -> bool:
        """카메라 중지"""
        with self.lock:
            if camera_id not in self.cameras:
                return False
            return self.cameras[camera_id].stop()
    
    def get_camera_status(self, camera_id: int) -> Optional[CameraStatus]:
        """카메라 상태 조회"""
        with self.lock:
            if camera_id not in self.cameras:
                return None
            return self.cameras[camera_id].get_status()
```

---

## Phase 2: API 구현 (Sprint 1-2)

### 7. 인증 API (2시간)
- 사용자 등록/로그인
- JWT 토큰 발급/검증
- 비밀번호 해싱

### 8. 카메라 관리 API (3시간)
- CRUD 작업
- 시작/중지
- 상태 조회

### 9. 이벤트 API (2시간)
- 목록 조회 (필터링, 페이지네이션)
- 상세 조회
- 확인 처리

---

## Phase 3: 알림 시스템 (Sprint 3)

### 10. 알림 엔진 구현 (3시간)
- 채널 추상화
- 이메일/웹훅 구현
- 우선순위 기반 발송
- 15초 중복 방지

---

## 📊 작업 시간 추정

| 작업 | 시간 | 우선순위 |
|------|------|----------|
| 프로젝트 구조 생성 | 30분 | 높음 |
| FastAPI 기본 구조 | 1시간 | 높음 |
| 데이터베이스 스키마 | 2시간 | 높음 |
| EventLogger 확장 | 2시간 | 높음 |
| ResourcePool 구현 | 2시간 | 높음 |
| MultiCameraManager | 3시간 | 높음 |
| **총계** | **10.5시간** | |

---

## 🚀 오늘 시작할 작업

1. **프로젝트 구조 생성** (30분) ✅
2. **FastAPI 기본 구조** (1시간) ✅
3. **데이터베이스 스키마 설계** (2시간) ✅

**총 예상 시간**: 3.5시간

---

**다음 단계**: 위 작업들을 순서대로 진행하겠습니다!
