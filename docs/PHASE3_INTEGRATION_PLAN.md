# Phase 3 통합 계획서

**작성일**: 2026-01-22  
**목적**: Phase 3 (Package Detection & Theft Detection)를 기존 시스템에 통합

---

## 1. 통합 포인트

### 1.1 E2EEngine 통합

**위치**: `src/pipeline/engine.py`

**통합 방법**:
```python
class E2EEngine:
    def __init__(self, config: EngineConfig):
        # 기존 컴포넌트
        self.vad_model: Optional[VADModel] = None
        self.vlm_analyzer: Optional[VLMAnalyzer] = None
        self.agent_flow = None
        
        # Phase 3 추가
        self.package_detector: Optional[BaseDetector] = None
        self.package_tracker: Optional[BaseTracker] = None
        self.theft_detector: Optional[BaseTheftDetector] = None
        self.event_bus: Optional[EventBus] = None  # 기존 EventBus 사용
```

**초기화 순서**:
1. EventBus 초기화 (기존)
2. VAD 모델 초기화 (기존)
3. Package Detection 초기화 (Phase 3)
4. VLM 분석기 초기화 (기존)
5. Agent Flow 초기화 (기존)

**프레임 처리 순서**:
```
프레임 읽기
  ↓
VAD 처리 (기존)
  ↓
Package Detection 처리 (Phase 3) ← 병렬 또는 순차
  ↓
이상 감지 시 클립 저장 (기존)
  ↓
VLM 분석 (기존)
  ↓
Agent 대응 (기존)
```

### 1.2 EventBus 통합

**위치**: `src/utils/event_bus.py` (기존)

**이벤트 타입** (이미 정의됨):
- `PackageDetectedEvent` ✅
- `PackageDisappearedEvent` ✅
- `TheftDetectedEvent` ✅

**이벤트 발행 위치**:
- `PackageDetector.detect()` → `PackageDetectedEvent`
- `PackageTracker.track()` → `PackageDisappearedEvent`
- `TheftDetector.check_theft()` → `TheftDetectedEvent`

### 1.3 Function Calling 통합

**위치**: `src/agent/function_calling.py` (기존)

**추가할 함수**:
```python
def get_package_count(e2e_system: "E2ESystem") -> Dict[str, Any]:
    """현재 추적 중인 패키지 수 조회"""
    if not hasattr(e2e_system, 'package_tracker'):
        return {"count": 0, "error": "Package tracker not initialized"}
    
    packages = e2e_system.package_tracker.get_all_packages()
    return {"count": len(packages), "packages": [p.package_id for p in packages]}

def get_package_details(e2e_system: "E2ESystem", package_id: str) -> Dict[str, Any]:
    """특정 패키지 상세 정보 조회"""
    if not hasattr(e2e_system, 'package_tracker'):
        return {"error": "Package tracker not initialized"}
    
    package = e2e_system.package_tracker.get_package(package_id)
    if not package:
        return {"error": f"Package {package_id} not found"}
    
    return {
        "package_id": package.package_id,
        "status": package.status,
        "first_detected": package.first_detected,
        "last_detected": package.last_detected,
        "detection_count": len(package.detections)
    }

def get_activity_log(e2e_system: "E2ESystem", limit: int = 10) -> Dict[str, Any]:
    """최근 패키지 활동 로그 조회"""
    if not hasattr(e2e_system, 'event_bus'):
        return {"error": "Event bus not initialized"}
    
    # Package 관련 이벤트만 필터링
    history = e2e_system.event_bus.get_history(limit=limit * 2)
    package_events = [
        e for e in history 
        if e.event_type in ["PackageDetectedEvent", "PackageDisappearedEvent", "TheftDetectedEvent"]
    ][:limit]
    
    return {
        "count": len(package_events),
        "events": [_event_to_dict(e) for e in package_events]
    }
```

**등록 위치**:
```python
# src/agent/function_calling.py의 register_functions() 함수에 추가
registry.register(
    name="get_package_count",
    description="현재 추적 중인 패키지 수를 조회합니다",
    parameters={"type": "object", "properties": {}},
    func=partial(get_package_count, e2e_system=system)
)
```

---

## 2. 디렉토리 구조

```
src/
├── package_detection/          # Phase 3 모듈 (Codex가 생성)
│   ├── __init__.py
│   ├── base.py                # BaseDetector, BaseTracker, BaseTheftDetector
│   ├── detector.py             # PackageDetector (YOLO v12 nano)
│   ├── tracker.py              # PackageTracker (IOU 기반)
│   └── theft_detector.py       # TheftDetector (3초 확인 로직)
├── pipeline/
│   └── engine.py               # E2EEngine (통합 지점)
├── utils/
│   ├── event_bus.py            # EventBus (기존)
│   └── events.py                # 이벤트 타입 (Package 이벤트 이미 정의됨)
└── agent/
    └── function_calling.py      # Function Calling (통합 지점)
```

---

## 3. 통합 순서

### Step 1: Phase 3 모듈 구현 (Codex)
- `src/package_detection/` 디렉토리 생성
- Base 클래스 구현
- Detector, Tracker, TheftDetector 구현

### Step 2: E2EEngine 통합
- `EngineConfig`에 Package Detection 설정 추가
- `E2EEngine.initialize()`에 Package Detection 초기화 추가
- `E2EEngine._process_loop()`에 Package Detection 처리 추가

### Step 3: Function Calling 통합
- `get_package_count()`, `get_package_details()`, `get_activity_log()` 구현
- `FunctionRegistry`에 등록

### Step 4: 테스트
- 단위 테스트
- 통합 테스트
- E2E 테스트

---

## 4. 설정 추가

**`EngineConfig`에 추가할 필드**:
```python
@dataclass
class EngineConfig:
    # 기존 필드...
    
    # Phase 3: Package Detection 설정
    enable_package_detection: bool = False
    package_detection_model: str = "yolo12n.pt"
    package_detection_confidence: float = 0.7
    package_tracker_max_age: int = 30
    theft_confirmation_time: float = 3.0  # 초
```

---

## 5. 의존성

**추가할 패키지** (requirements.txt):
```
ultralytics>=8.0.0  # YOLO v12
```

---

## 6. 통합 체크리스트

- [ ] Phase 3 모듈 구현 완료 (Codex)
- [ ] E2EEngine에 Package Detection 통합
- [ ] EventBus 이벤트 발행 확인
- [ ] Function Calling 함수 등록
- [ ] 단위 테스트 작성
- [ ] 통합 테스트 작성
- [ ] 성능 테스트 (FPS, 메모리)
- [ ] 문서 업데이트
