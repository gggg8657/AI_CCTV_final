# Vision-Agents 통합 데이터 플로우 상세

**버전**: 1.0  
**작성일**: 2026-01-21

---

## 목차

1. [전체 데이터 플로우 개요](#1-전체-데이터-플로우-개요)
2. [이벤트 기반 데이터 플로우](#2-이벤트-기반-데이터-플로우)
3. [Function Calling 데이터 플로우](#3-function-calling-데이터-플로우)
4. [패키지 감지 데이터 플로우](#4-패키지-감지-데이터-플로우)
5. [데이터 저장 및 조회](#5-데이터-저장-및-조회)

---

## 1. 전체 데이터 플로우 개요

### 1.1 통합 전 (현재)

```
Video Frame
    │
    ├─▶ VAD Model
    │   └─▶ Score
    │       └─▶ Callback 함수 호출
    │           └─▶ 직접 처리
    │
    └─▶ VLM/Agent
        └─▶ Callback 함수 호출
            └─▶ 직접 처리
```

**특징:**
- 폴링 기반
- 직접적인 함수 호출
- 강한 결합

### 1.2 통합 후 (개선)

```
Video Frame
    │
    ├─▶ VAD Model
    │   └─▶ Score
    │       └─▶ AnomalyDetectedEvent 발행
    │           └─▶ Event Bus
    │               └─▶ 구독자들에게 비동기 전달
    │
    ├─▶ Package Detector
    │   └─▶ PackageDetectedEvent 발행
    │       └─▶ Event Bus
    │
    └─▶ Event Bus
        ├─▶ Event Logger (DB 저장)
        ├─▶ UI 업데이트
        ├─▶ 알림 시스템
        └─▶ 커스텀 핸들러
```

**특징:**
- 이벤트 기반
- 느슨한 결합
- 확장 가능

---

## 2. 이벤트 기반 데이터 플로우

### 2.1 이벤트 타입 정의

```python
# 이벤트 기본 클래스
@dataclass
class Event:
    event_id: str
    event_type: str
    timestamp: datetime
    source: str

# 구체적인 이벤트 타입들
@dataclass
class AnomalyDetectedEvent(Event):
    frame_id: int
    score: float
    threshold: float
    frame: np.ndarray

@dataclass
class VLMAnalysisCompletedEvent(Event):
    event_id: str  # 원본 AnomalyDetectedEvent ID
    detected_type: str
    description: str
    actions: List[str]
    confidence: float

@dataclass
class AgentResponseEvent(Event):
    event_id: str  # 원본 AnomalyDetectedEvent ID
    plan: List[str]
    priority: int
    estimated_time: float

@dataclass
class PackageDetectedEvent(Event):
    package_id: str
    is_new: bool
    confidence: float
    bbox: Tuple[int, int, int, int]
    frame_id: int

@dataclass
class PackageDisappearedEvent(Event):
    package_id: str
    picker_face_id: Optional[str]
    timestamp: datetime

@dataclass
class TheftDetectedEvent(Event):
    package_id: str
    suspect_face_id: Optional[str]
    timestamp: datetime

@dataclass
class FrameProcessedEvent(Event):
    frame_id: int
    processing_time: float
    vad_score: float

@dataclass
class StatsUpdatedEvent(Event):
    total_frames: int
    anomaly_count: int
    avg_processing_time: float
    current_fps: float
```

### 2.2 이벤트 발행 플로우

```
컴포넌트 (VAD/VLM/Agent/Package Detector)
    │
    ├─▶ 이벤트 객체 생성
    │   └─▶ Event(...)
    │
    ├─▶ EventBus.publish(event)
    │   │
    │   ├─▶ 이벤트 타입 확인
    │   ├─▶ 구독자 목록 조회
    │   ├─▶ 이벤트 히스토리에 추가
    │   └─▶ 비동기 전달
    │       │
    │       ├─▶ 구독자 1: EventLogger
    │       │   └─▶ 데이터베이스 저장
    │       │
    │       ├─▶ 구독자 2: UI 업데이트
    │       │   └─▶ 실시간 대시보드 갱신
    │       │
    │       ├─▶ 구독자 3: 알림 시스템
    │       │   └─▶ 이메일/웹훅 전송
    │       │
    │       └─▶ 구독자 N: 커스텀 핸들러
    │           └─▶ 사용자 정의 로직
```

### 2.3 이벤트 구독 플로우

```python
# 구독자 등록
@event_bus.subscribe(AnomalyDetectedEvent)
async def on_anomaly_detected(event: AnomalyDetectedEvent):
    """이상 감지 이벤트 처리"""
    # 1. 로깅
    logger.info(f"Anomaly detected: score={event.score}")
    
    # 2. 클립 저장 시작
    clip_saver.trigger_save(event.frame, event.score)
    
    # 3. UI 업데이트
    ui.update_status("anomaly_detected", event.score)
    
    # 4. 알림 (선택)
    if event.score > 0.8:
        notification.send("High severity anomaly detected!")

# 이벤트 체인
AnomalyDetectedEvent
    │
    ├─▶ Clip Saver 시작
    │   └─▶ 클립 저장 완료
    │       └─▶ VLMAnalysisCompletedEvent 발행
    │           │
    │           └─▶ AgentResponseEvent 발행
    │               │
    │               └─▶ 최종 이벤트 로깅
```

---

## 3. Function Calling 데이터 플로우

### 3.1 자연어 질의 플로우

```
사용자 입력: "시스템 상태 알려줘"
    │
    ├─▶ Agent System (Qwen3-8B)
    │   ├─▶ Function Calling 분석
    │   │   ├─▶ 필요한 함수: get_system_status()
    │   │   └─▶ 파라미터: 없음
    │   │
    │   ├─▶ Function Registry 호출
    │   │   └─▶ get_system_status() 실행
    │   │       │
    │   │       ├─▶ 시스템 상태 수집
    │   │       │   ├─▶ E2ESystem.get_stats()
    │   │       │   ├─▶ VAD 상태
    │   │       │   ├─▶ VLM 상태
    │   │       │   └─▶ Agent 상태
    │   │       │
    │   │       └─▶ 결과 반환
    │   │           {
    │   │               "status": "running",
    │   │               "total_frames": 12345,
    │   │               "anomaly_count": 12,
    │   │               ...
    │   │           }
    │   │
    │   └─▶ 결과를 자연어로 변환
    │       └─▶ "시스템은 정상 작동 중이며, 현재까지 12건의 이상 상황이 탐지되었습니다."
    │
    └─▶ 사용자에게 응답
        ├─▶ 텍스트 출력 (CLI/Web UI)
        └─▶ 음성 출력 (TTS, 선택)
```

### 3.2 시스템 제어 플로우

```
사용자 입력: "VAD 임계값을 0.7로 올려줘"
    │
    ├─▶ Agent System
    │   ├─▶ Function Calling 분석
    │   │   ├─▶ 필요한 함수: update_vad_threshold()
    │   │   └─▶ 파라미터: value=0.7
    │   │
    │   ├─▶ Function Registry 호출
    │   │   └─▶ update_vad_threshold(0.7) 실행
    │   │       │
    │   │       ├─▶ 유효성 검증
    │   │       │   └─▶ 0.0 <= 0.7 <= 1.0 ✓
    │   │       │
    │   │       ├─▶ E2ESystem.config.vad_threshold = 0.7
    │   │       │
    │   │       └─▶ 결과 반환
    │   │           {"success": true, "new_threshold": 0.7}
    │   │
    │   └─▶ 결과를 자연어로 변환
    │       └─▶ "VAD 임계값이 0.7로 변경되었습니다."
    │
    └─▶ 사용자에게 응답
        └─▶ "임계값 변경 완료"
```

### 3.3 복합 질의 플로우

```
사용자 입력: "최근 이상 탐지 이벤트 10개와 패키지 통계를 알려줘"
    │
    ├─▶ Agent System
    │   ├─▶ Function Calling 분석
    │   │   ├─▶ 함수 1: get_recent_events(limit=10)
    │   │   └─▶ 함수 2: get_package_count()
    │   │
    │   ├─▶ 순차 실행
    │   │   ├─▶ get_recent_events(10)
    │   │   │   └─▶ 데이터베이스 조회
    │   │   │       └─▶ 최근 10개 이벤트 반환
    │   │   │
    │   │   └─▶ get_package_count()
    │   │       └─▶ 패키지 통계 계산
    │   │           └─▶ 통계 반환
    │   │
    │   └─▶ 결과 통합 및 자연어 변환
    │       └─▶ "최근 10건의 이상 탐지 이벤트는 다음과 같습니다: ... 
    │           현재 보이는 패키지는 2개이며, 총 15개 중 3개가 도난되었습니다."
    │
    └─▶ 사용자에게 응답
```

---

## 4. 패키지 감지 데이터 플로우

### 4.1 패키지 감지 플로우

```
Frame (비디오 입력)
    │
    ├─▶ Package Detector (YOLO v12 nano)
    │   ├─▶ 객체 감지
    │   │   └─▶ 결과: [
    │   │       {
    │   │           "class": "package",
    │   │           "confidence": 0.85,
    │   │           "bbox": [100, 150, 200, 250]
    │   │       }
    │   │   ]
    │   │
    │   ├─▶ 패키지 필터링
    │   │   └─▶ confidence >= 0.7
    │   │
    │   ├─▶ 패키지 추적
    │   │   ├─▶ 기존 패키지와 매칭 (IoU 기반)
    │   │   └─▶ 새 패키지면 ID 할당
    │   │
    │   └─▶ PackageDetectedEvent 발행
    │       │
    │       ├─▶ 패키지 데이터베이스 저장
    │       │   └─▶ INSERT INTO packages (...)
    │       │
    │       ├─▶ UI 업데이트
    │       │   └─▶ 패키지 표시
    │       │
    │       └─▶ 로깅
    │
    └─▶ 다음 프레임 처리
```

### 4.2 패키지 추적 플로우

```
패키지 감지 (Frame N)
    │
    ├─▶ 패키지 ID: "pkg_001"
    ├─▶ 위치: bbox = [100, 150, 200, 250]
    ├─▶ 데이터베이스 저장
    │   └─▶ first_seen = NOW()
    │       last_seen = NOW()
    │       detection_count = 1
    │
    └─▶ 다음 프레임 (Frame N+1)
        │
        ├─▶ 패키지 감지
        │   └─▶ bbox = [105, 155, 205, 255]
        │
        ├─▶ IoU 계산
        │   └─▶ IoU > 0.5 → 같은 패키지
        │
        ├─▶ 데이터베이스 업데이트
        │   └─▶ last_seen = NOW()
        │       detection_count += 1
        │
        └─▶ 계속 추적...
```

### 4.3 도난 감지 플로우

```
패키지 추적 중...
    │
    └─▶ Frame N: 패키지 감지됨 ✓
        └─▶ last_seen = Frame N
        
    └─▶ Frame N+1: 패키지 없음 ❌
        └─▶ PackageDisappearedEvent 발행
            │
            ├─▶ delayed_theft_check() 시작
            │   └─▶ 3초 타이머 시작
            │
            └─▶ _pending_theft_tasks에 추가
                └─▶ {package_id: task}
            
    └─▶ 3초 대기 중...
        │
        ├─▶ 시나리오 1: 패키지 재등장 (Frame N+2, 1초 후)
        │   └─▶ PackageDetectedEvent 발행
        │       └─▶ delayed_theft_check() 취소 ✓
        │           └─▶ 거짓 경보 무시
        │
        └─▶ 시나리오 2: 패키지 여전히 없음 (3초 후)
            └─▶ 타이머 만료
                └─▶ TheftDetectedEvent 발행
                    │
                    ├─▶ 데이터베이스 업데이트
                    │   └─▶ status = "stolen"
                    │       picked_up_by = suspect_id
                    │
                    ├─▶ 알림 시스템 (선택)
                    │   └─▶ 이메일/웹훅 전송
                    │
                    └─▶ UI 업데이트
                        └─▶ 경고 표시
```

---

## 5. 데이터 저장 및 조회

### 5.1 데이터베이스 스키마

```sql
-- 이벤트 테이블
CREATE TABLE events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    source TEXT,
    data TEXT,  -- JSON
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_events_type ON events(event_type);
CREATE INDEX idx_events_timestamp ON events(timestamp);

-- 패키지 테이블
CREATE TABLE packages (
    package_id TEXT PRIMARY KEY,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    detection_count INTEGER DEFAULT 0,
    confidence REAL,
    picked_up_by TEXT,
    status TEXT,  -- "present" | "stolen" | "removed"
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_w INTEGER,
    bbox_h INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_packages_status ON packages(status);
CREATE INDEX idx_packages_last_seen ON packages(last_seen);

-- 통계 테이블 (선택)
CREATE TABLE statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_statistics_metric ON statistics(metric_name);
```

### 5.2 데이터 저장 플로우

```
이벤트 발생
    │
    ├─▶ EventBus.publish(event)
    │   └─▶ 구독자: EventLogger
    │       │
    │       ├─▶ 이벤트 타입별 처리
    │       │   ├─▶ AnomalyDetectedEvent
    │       │   │   └─▶ INSERT INTO events (...)
    │       │   │
    │       │   ├─▶ PackageDetectedEvent
    │       │   │   ├─▶ INSERT INTO events (...)
    │       │   │   └─▶ INSERT/UPDATE packages (...)
    │       │   │
    │       │   └─▶ TheftDetectedEvent
    │       │       ├─▶ INSERT INTO events (...)
    │       │       └─▶ UPDATE packages SET status='stolen' ...
    │       │
    │       └─▶ 트랜잭션 커밋
    │
    └─▶ 저장 완료
```

### 5.3 데이터 조회 플로우

```
Function Calling: get_recent_events(limit=20)
    │
    ├─▶ Database.query()
    │   └─▶ SELECT * FROM events 
    │       WHERE event_type IN ('AnomalyDetectedEvent', ...)
    │       ORDER BY timestamp DESC 
    │       LIMIT 20
    │
    ├─▶ 결과 변환
    │   └─▶ List[Dict] 형태로 변환
    │
    └─▶ 반환
        └─▶ [
            {
                "event_id": "evt_001",
                "event_type": "AnomalyDetectedEvent",
                "timestamp": "2026-01-21T12:34:56",
                "data": {...}
            },
            ...
        ]
```

---

## 6. 성능 최적화 고려사항

### 6.1 이벤트 처리 최적화

- **비동기 처리**: 이벤트 구독자는 비동기로 실행
- **이벤트 버퍼링**: 높은 빈도 이벤트는 버퍼링 후 배치 처리
- **우선순위 큐**: 중요 이벤트 우선 처리

### 6.2 데이터베이스 최적화

- **인덱스**: 자주 조회하는 컬럼에 인덱스 생성
- **파티셔닝**: 날짜별 파티셔닝 (선택)
- **캐싱**: 자주 조회하는 데이터는 메모리 캐시

### 6.3 패키지 감지 최적화

- **프레임 스킵**: 모든 프레임이 아닌 N프레임마다 감지
- **ROI (Region of Interest)**: 관심 영역만 처리
- **모델 최적화**: TensorRT 또는 ONNX 변환

---

*이 문서는 프로젝트 진행에 따라 지속적으로 업데이트됩니다.*
