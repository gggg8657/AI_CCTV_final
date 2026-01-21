# Vision-Agents 통합 기능 명세서

**버전**: 1.0  
**작성일**: 2026-01-21

---

## 목차

1. [이벤트 시스템 기능 명세](#1-이벤트-시스템-기능-명세)
2. [Function Calling 기능 명세](#2-function-calling-기능-명세)
3. [패키지 감지 기능 명세](#3-패키지-감지-기능-명세)
4. [음성 인터랙션 기능 명세](#4-음성-인터랙션-기능-명세)
5. [데이터베이스 기능 명세](#5-데이터베이스-기능-명세)

---

## 1. 이벤트 시스템 기능 명세

### 1.1 EventBus 클래스

**파일**: `src/utils/event_bus.py`

**클래스 정의**:
```python
class EventBus:
    """중앙 집중식 이벤트 버스"""
    
    def __init__(self):
        self._subscribers: Dict[Type[Event], List[Callable]] = {}
        self._event_history: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def subscribe(self, event_type: Type[Event], handler: Callable):
        """이벤트 구독"""
        
    def unsubscribe(self, event_type: Type[Event], handler: Callable):
        """이벤트 구독 해제"""
        
    def publish(self, event: Event):
        """이벤트 발행"""
        
    def get_history(self, limit: int = 100) -> List[Event]:
        """이벤트 히스토리 조회"""
```

**주요 메서드**:

#### subscribe()
- **목적**: 이벤트 타입별 핸들러 등록
- **파라미터**: 
  - `event_type`: 이벤트 타입 (클래스)
  - `handler`: 핸들러 함수 (async 또는 sync)
- **반환값**: 없음
- **예외**: 없음

#### publish()
- **목적**: 이벤트를 구독자들에게 전달
- **파라미터**: 
  - `event`: 이벤트 객체
- **반환값**: 없음
- **예외**: 핸들러 실행 중 예외는 로깅만 하고 계속 진행

**사용 예시**:
```python
# 구독자 등록
@event_bus.subscribe(AnomalyDetectedEvent)
async def on_anomaly(event: AnomalyDetectedEvent):
    logger.info(f"Anomaly: {event.score}")

# 이벤트 발행
event = AnomalyDetectedEvent(
    event_id="evt_001",
    event_type="AnomalyDetectedEvent",
    timestamp=datetime.now(),
    source="VAD",
    frame_id=12345,
    score=0.75,
    threshold=0.5,
    frame=frame
)
event_bus.publish(event)
```

### 1.2 이벤트 타입 상세

#### AnomalyDetectedEvent

**발행 시점**: VAD 모델이 이상 점수가 임계값을 초과할 때

**속성**:
```python
@dataclass
class AnomalyDetectedEvent(Event):
    frame_id: int              # 프레임 번호
    score: float               # 이상 점수 (0.0 ~ 1.0)
    threshold: float           # 임계값
    frame: np.ndarray          # 프레임 이미지 (선택)
```

**구독자 예시**:
- EventLogger: 데이터베이스 저장
- ClipSaver: 클립 저장 시작
- UI: 실시간 업데이트

#### VLMAnalysisCompletedEvent

**발행 시점**: VLM 분석이 완료될 때

**속성**:
```python
@dataclass
class VLMAnalysisCompletedEvent(Event):
    event_id: str              # 원본 AnomalyDetectedEvent ID
    detected_type: str         # "Fighting", "Arson", etc.
    description: str           # 상황 설명
    actions: List[str]         # 권장 대응
    confidence: float          # 신뢰도 (0.0 ~ 1.0)
    clip_path: str             # 분석한 클립 경로
```

#### AgentResponseEvent

**발행 시점**: Agent 대응 계획이 수립될 때

**속성**:
```python
@dataclass
class AgentResponseEvent(Event):
    event_id: str              # 원본 AnomalyDetectedEvent ID
    plan: List[str]            # 대응 계획
    priority: int              # 우선순위 (1=높음, 5=낮음)
    estimated_time: float      # 예상 소요 시간 (초)
```

#### PackageDetectedEvent

**발행 시점**: 패키지가 감지될 때

**속성**:
```python
@dataclass
class PackageDetectedEvent(Event):
    package_id: str           # 패키지 고유 ID
    is_new: bool              # 새로 감지된 패키지 여부
    confidence: float         # 신뢰도 (0.0 ~ 1.0)
    bbox: Tuple[int, int, int, int]  # [x, y, w, h]
    frame_id: int             # 프레임 번호
```

#### PackageDisappearedEvent

**발행 시점**: 패키지가 프레임에서 사라질 때

**속성**:
```python
@dataclass
class PackageDisappearedEvent(Event):
    package_id: str           # 사라진 패키지 ID
    picker_face_id: Optional[str]  # 집어간 사람 ID (없으면 None)
    timestamp: datetime       # 사라진 시간
```

#### TheftDetectedEvent

**발행 시점**: 도난이 확정될 때 (3초 확인 후)

**속성**:
```python
@dataclass
class TheftDetectedEvent(Event):
    package_id: str           # 도난된 패키지 ID
    suspect_face_id: Optional[str]  # 용의자 ID
    timestamp: datetime       # 도난 시간
```

---

## 2. Function Calling 기능 명세

### 2.1 FunctionRegistry 클래스

**파일**: `src/agent/function_calling.py`

**클래스 정의**:
```python
class FunctionRegistry:
    """Function Calling 레지스트리"""
    
    def __init__(self):
        self._functions: Dict[str, Callable] = {}
        self._descriptions: Dict[str, str] = {}
    
    def register(self, func: Callable, description: str):
        """함수 등록"""
        
    def call(self, function_name: str, arguments: Dict) -> Any:
        """함수 실행"""
        
    def list_functions(self) -> List[Dict]:
        """등록된 함수 목록"""
```

### 2.2 핵심 함수 명세

#### get_system_status()

**함수명**: `get_system_status`

**설명**: 현재 시스템 상태 조회

**파라미터**: 없음

**반환값**:
```python
{
    "status": "running" | "stopped" | "error",
    "vad_model": "mnad",
    "vad_threshold": 0.5,
    "vlm_enabled": true,
    "agent_enabled": true,
    "package_detector_enabled": true,
    "total_frames": 12345,
    "anomaly_count": 12,
    "package_count": 2,
    "current_fps": 30.5,
    "uptime_seconds": 3600
}
```

**에러 처리**:
- 시스템 오류 시: `{"status": "error", "error": "..."}`

#### get_recent_events()

**함수명**: `get_recent_events`

**설명**: 최근 이상 탐지 이벤트 조회

**파라미터**:
- `limit: int = 20` (기본값: 20)

**반환값**:
```python
{
    "events": [
        {
            "event_id": "evt_001",
            "timestamp": "2026-01-21T12:34:56",
            "score": 0.75,
            "detected_type": "Fighting",
            "description": "..."
        },
        ...
    ],
    "total": 20
}
```

#### get_anomaly_statistics()

**함수명**: `get_anomaly_statistics`

**설명**: 이상 탐지 통계 조회

**파라미터**: 없음

**반환값**:
```python
{
    "total_anomalies": 12,
    "anomalies_by_type": {
        "Fighting": 5,
        "Arson": 3,
        "Vandalism": 4
    },
    "avg_score": 0.72,
    "max_score": 0.95,
    "min_score": 0.51,
    "time_range": {
        "start": "2026-01-21T00:00:00",
        "end": "2026-01-21T23:59:59"
    }
}
```

#### get_package_count()

**함수명**: `get_package_count`

**설명**: 패키지 통계 조회

**파라미터**: 없음

**반환값**:
```python
{
    "currently_visible": 2,
    "total_seen": 15,
    "stolen_count": 3,
    "removed_count": 10
}
```

#### get_package_details()

**함수명**: `get_package_details`

**설명**: 모든 패키지의 상세 정보 조회

**파라미터**: 없음

**반환값**:
```python
{
    "packages": [
        {
            "package_id": "pkg_001",
            "first_seen": "2026-01-21T12:00:00",
            "last_seen": "2026-01-21T12:05:00",
            "detection_count": 3,
            "status": "stolen",
            "picked_up_by": "suspect_123"
        },
        ...
    ],
    "total": 15
}
```

#### get_activity_log()

**함수명**: `get_activity_log`

**설명**: 최근 활동 로그 조회

**파라미터**:
- `limit: int = 20` (기본값: 20)

**반환값**:
```python
{
    "activity_log": [
        {
            "timestamp": "2026-01-21T12:34:56",
            "type": "anomaly_detected",
            "description": "Fighting detected"
        },
        {
            "timestamp": "2026-01-21T12:35:00",
            "type": "package_detected",
            "description": "Package pkg_001 detected"
        },
        ...
    ],
    "total": 20
}
```

#### update_vad_threshold()

**함수명**: `update_vad_threshold`

**설명**: VAD 이상 탐지 임계값 변경

**파라미터**:
- `value: float` (0.0 ~ 1.0)

**반환값**:
```python
{
    "success": true,
    "old_threshold": 0.5,
    "new_threshold": 0.7
}
```

**에러 처리**:
- 범위 초과: `{"success": false, "error": "Threshold must be between 0.0 and 1.0"}`

#### enable_vlm()

**함수명**: `enable_vlm`

**설명**: VLM 분석 활성화/비활성화

**파라미터**:
- `enabled: bool`

**반환값**:
```python
{
    "success": true,
    "vlm_enabled": true
}
```

#### set_agent_flow()

**함수명**: `set_agent_flow`

**설명**: Agent Flow 타입 변경

**파라미터**:
- `flow_type: str` ("sequential" | "hierarchical" | "collaborative")

**반환값**:
```python
{
    "success": true,
    "old_flow": "sequential",
    "new_flow": "hierarchical"
}
```

**에러 처리**:
- 잘못된 타입: `{"success": false, "error": "Invalid flow type"}`

---

## 3. 패키지 감지 기능 명세

### 3.1 PackageDetector 클래스

**파일**: `src/pipeline/package_detector.py`

**클래스 정의**:
```python
class PackageDetector:
    """YOLO v12 nano 기반 패키지 감지기"""
    
    def __init__(
        self,
        model_path: str = "yolo12n.pt",
        confidence_threshold: float = 0.7,
        device: str = "cuda:2"
    ):
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.tracker = PackageTracker()
        self.event_bus = None
    
    def initialize(self) -> bool:
        """초기화"""
        
    def detect(self, frame: np.ndarray) -> List[Package]:
        """패키지 감지"""
        
    def process_frame(self, frame: np.ndarray, frame_id: int):
        """프레임 처리 (감지 + 추적 + 이벤트 발행)"""
```

### 3.2 PackageTracker 클래스

**파일**: `src/pipeline/package_tracker.py`

**클래스 정의**:
```python
class PackageTracker:
    """패키지 추적 시스템"""
    
    def __init__(self, max_age: int = 30):
        self.tracks: Dict[str, Track] = {}
        self.max_age = max_age  # 프레임 단위
    
    def update(self, detections: List[Package]) -> Tuple[List[str], List[str]]:
        """추적 업데이트
        
        Returns:
            (new_package_ids, disappeared_package_ids)
        """
        
    def get_track(self, package_id: str) -> Optional[Track]:
        """패키지 트랙 조회"""
```

### 3.3 도난 감지 로직

**파일**: `src/pipeline/theft_detector.py`

**클래스 정의**:
```python
class TheftDetector:
    """도난 감지 시스템"""
    
    def __init__(self, confirmation_delay: float = 3.0):
        self.confirmation_delay = confirmation_delay
        self.pending_checks: Dict[str, asyncio.Task] = {}
    
    async def check_theft(
        self,
        package_id: str,
        disappeared_at: datetime
    ) -> bool:
        """도난 확인 (3초 대기 후)"""
        
    def cancel_check(self, package_id: str):
        """도난 확인 취소 (패키지 재등장 시)"""
```

**로직 흐름**:
```python
# 패키지 사라짐 감지
package_disappeared_event = PackageDisappearedEvent(...)

# 3초 대기 작업 시작
task = asyncio.create_task(
    theft_detector.check_theft(
        package_id,
        disappeared_at
    )
)
pending_checks[package_id] = task

# 3초 후
if package_still_missing:
    TheftDetectedEvent 발행
else:
    # 패키지 재등장 시
    task.cancel()
    pending_checks.pop(package_id)
```

---

## 4. 음성 인터랙션 기능 명세 (선택)

### 4.1 VoiceInteraction 클래스

**파일**: `src/utils/voice_interaction.py`

**클래스 정의**:
```python
class VoiceInteraction:
    """음성 인터랙션 시스템"""
    
    def __init__(
        self,
        stt_provider: str = "deepgram",
        tts_provider: str = "elevenlabs",
        function_registry: FunctionRegistry = None
    ):
        self.stt = None
        self.tts = None
        self.function_registry = function_registry
        self.agent = None
    
    def initialize(self) -> bool:
        """초기화"""
        
    async def process_audio(self, audio_data: bytes):
        """음성 입력 처리"""
        
    async def speak(self, text: str):
        """텍스트를 음성으로 변환"""
```

### 4.2 STT 통합

**옵션 1: Deepgram (권장)**
- 빠른 처리 (< 100ms)
- 높은 정확도
- 한국어 지원

**옵션 2: Fast-Whisper (로컬)**
- 오프라인 처리
- 무료
- 느린 처리 (200-500ms)

### 4.3 TTS 통합

**옵션 1: ElevenLabs (권장)**
- 자연스러운 음성
- 빠른 처리 (< 200ms)
- 한국어 지원

**옵션 2: Kokoro (로컬)**
- 오프라인 처리
- 무료
- 음질: ElevenLabs보다 낮음

### 4.4 음성 인터랙션 플로우

```
음성 입력 (마이크)
    │
    ├─▶ STT (Deepgram)
    │   └─▶ 텍스트 변환
    │       예: "시스템 상태 알려줘"
    │
    ├─▶ Agent System
    │   ├─▶ Function Calling 분석
    │   │   └─▶ get_system_status()
    │   │
    │   ├─▶ 함수 실행
    │   │   └─▶ 결과 반환
    │   │
    │   └─▶ 자연어 응답 생성
    │       └─▶ "시스템은 정상 작동 중입니다..."
    │
    └─▶ TTS (ElevenLabs)
        └─▶ 음성 합성
            └─▶ 스피커 출력
```

---

## 5. 데이터베이스 기능 명세

### 5.1 DatabaseManager 클래스

**파일**: `src/database/db.py`

**클래스 정의**:
```python
class DatabaseManager:
    """SQLite 데이터베이스 관리자"""
    
    def __init__(self, db_path: str = "data/system.db"):
        self.db_path = db_path
        self.conn = None
    
    def initialize(self) -> bool:
        """데이터베이스 초기화 (스키마 생성)"""
        
    def save_event(self, event: Event) -> bool:
        """이벤트 저장"""
        
    def save_package(self, package: Package) -> bool:
        """패키지 저장/업데이트"""
        
    def get_recent_events(
        self,
        limit: int = 20,
        event_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """최근 이벤트 조회"""
        
    def get_package_count(self) -> Dict:
        """패키지 통계 조회"""
        
    def get_package_details(self) -> List[Dict]:
        """패키지 상세 정보 조회"""
```

### 5.2 데이터베이스 스키마 상세

#### events 테이블

```sql
CREATE TABLE events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    source TEXT,
    data TEXT,  -- JSON 문자열
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스
CREATE INDEX idx_events_type ON events(event_type);
CREATE INDEX idx_events_timestamp ON events(timestamp DESC);
CREATE INDEX idx_events_type_timestamp ON events(event_type, timestamp DESC);
```

**data 컬럼 JSON 구조**:
```json
{
    "frame_id": 12345,
    "score": 0.75,
    "threshold": 0.5,
    "detected_type": "Fighting",
    "description": "...",
    "actions": ["보안요원 출동", "경찰 신고"]
}
```

#### packages 테이블

```sql
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
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스
CREATE INDEX idx_packages_status ON packages(status);
CREATE INDEX idx_packages_last_seen ON packages(last_seen DESC);
CREATE INDEX idx_packages_status_last_seen ON packages(status, last_seen DESC);
```

### 5.3 데이터 조회 쿼리

#### 최근 이벤트 조회

```sql
SELECT 
    event_id,
    event_type,
    timestamp,
    source,
    data
FROM events
WHERE event_type IN ('AnomalyDetectedEvent', 'TheftDetectedEvent')
ORDER BY timestamp DESC
LIMIT ?;
```

#### 패키지 통계 조회

```sql
SELECT 
    COUNT(*) FILTER (WHERE status = 'present') as currently_visible,
    COUNT(*) as total_seen,
    COUNT(*) FILTER (WHERE status = 'stolen') as stolen_count,
    COUNT(*) FILTER (WHERE status = 'removed') as removed_count
FROM packages;
```

#### 패키지 상세 정보 조회

```sql
SELECT 
    package_id,
    first_seen,
    last_seen,
    detection_count,
    confidence,
    picked_up_by,
    status
FROM packages
ORDER BY last_seen DESC;
```

---

## 6. 통합 인터페이스 명세

### 6.1 E2ESystem 확장

**기존 메서드 유지**:
- `initialize()`
- `start()`
- `stop()`
- `get_stats()`

**신규 메서드 추가**:
```python
class E2ESystem:
    # ... 기존 메서드 ...
    
    # 신규 메서드
    def get_event_bus(self) -> EventBus:
        """이벤트 버스 반환"""
        
    def get_function_registry(self) -> FunctionRegistry:
        """Function Registry 반환"""
        
    def get_package_detector(self) -> Optional[PackageDetector]:
        """패키지 감지기 반환"""
        
    def enable_voice_interaction(self, enabled: bool):
        """음성 인터랙션 활성화/비활성화"""
```

### 6.2 Agent System 확장

**기존 Agent Flow 유지**:
- Sequential Flow
- Hierarchical Flow
- Collaborative Flow

**신규 기능 추가**:
- Function Calling 통합
- 자연어 질의응답
- 음성 인터랙션 (선택)

---

*이 문서는 프로젝트 진행에 따라 지속적으로 업데이트됩니다.*
