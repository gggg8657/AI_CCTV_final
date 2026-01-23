# Phase 3: 시스템 설계 문서 (System Design Document)

**버전**: 1.0  
**작성일**: 2026-01-22  
**Linear Issue**: CHA-55

---

## 목차

1. [시스템 아키텍처](#1-시스템-아키텍처)
2. [컴포넌트 설계](#2-컴포넌트-설계)
3. [데이터 흐름](#3-데이터-흐름)
4. [클래스 다이어그램](#4-클래스-다이어그램)
5. [시퀀스 다이어그램](#5-시퀀스-다이어그램)
6. [상태 다이어그램](#6-상태-다이어그램)

---

## 1. 시스템 아키텍처

### 1.1 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                      E2ESystem                              │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  VAD Engine  │  │ Package Det. │  │  VLM Analyzer│     │
│  │              │  │   System     │  │              │     │
│  │              │  │              │  │              │     │
│  │  - MNAD      │  │  - Detector  │  │  - Qwen2.5-VL │     │
│  │  - STAE      │  │  - Tracker   │  │              │     │
│  │  - etc.      │  │  - Theft Det.│  │              │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            │                                 │
│                    ┌───────▼────────┐                        │
│                    │   EventBus      │                        │
│                    │                 │                        │
│                    │  - Anomaly     │                        │
│                    │  - Package     │                        │
│                    │  - Theft       │                        │
│                    └───────┬────────┘                        │
│                            │                                 │
│                    ┌───────▼────────┐                        │
│                    │  Agent System   │                        │
│                    │                 │                        │
│                    │  - Function    │                        │
│                    │    Calling     │                        │
│                    │  - Flows      │                        │
│                    └────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Package Detection System 내부 구조

```
Package Detection System
├── PackageDetector
│   ├── YOLO Model (v12 nano)
│   ├── Preprocessing
│   └── Postprocessing
├── PackageTracker
│   ├── IOU-based Matching
│   ├── ID Management
│   └── State Management
└── TheftDetector
    ├── Confirmation Logic (3초)
    ├── False Positive Filter
    └── Event Generator
```

---

## 2. 컴포넌트 설계

### 2.1 PackageDetector

**책임**: YOLO 모델을 사용하여 패키지 객체 감지

**설계 원칙**:
- **SRP**: 객체 감지만 담당
- **OCP**: 다른 감지 모델로 확장 가능 (인터페이스 기반)
- **DIP**: 구체적인 YOLO 구현이 아닌 추상화에 의존

**클래스 구조**:
```python
class PackageDetector:
    """패키지 객체 감지기"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = None
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = 0.5
        self.class_ids = [COCO_PACKAGE_CLASS_ID]  # COCO 클래스 ID
    
    def load_model(self) -> bool:
        """YOLO 모델 로드"""
        pass
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """프레임에서 패키지 감지"""
        pass
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """프레임 전처리"""
        pass
    
    def _postprocess(self, results) -> List[Detection]:
        """YOLO 결과 후처리"""
        pass
```

**의존성**:
- `ultralytics` (YOLO)
- `torch`
- `numpy`
- `opencv-python`

---

### 2.2 PackageTracker

**책임**: 감지된 패키지를 시간에 따라 추적

**설계 원칙**:
- **SRP**: 패키지 추적만 담당
- **OCP**: 다른 추적 알고리즘으로 확장 가능
- **ISP**: 추적 관련 메서드만 노출

**클래스 구조**:
```python
class PackageTracker:
    """패키지 추적기"""
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.tracked_packages: Dict[str, TrackedPackage] = {}
        self.iou_threshold = iou_threshold
        self.max_age = max_age  # 초
        self.next_id = 1
    
    def track(self, detections: List[Detection]) -> List[TrackedPackage]:
        """감지 결과를 추적"""
        # 1. 기존 패키지와 IOU 매칭
        # 2. 새 패키지 ID 할당
        # 3. 패키지 상태 업데이트
        # 4. 만료된 패키지 제거
        pass
    
    def _match_detections(
        self, 
        detections: List[Detection],
        tracked: List[TrackedPackage]
    ) -> Dict[int, str]:
        """IOU 기반 매칭"""
        pass
    
    def _calculate_iou(
        self, 
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """IOU 계산"""
        pass
    
    def _update_package(
        self, 
        package_id: str, 
        detection: Detection
    ):
        """패키지 정보 업데이트"""
        pass
    
    def _remove_expired(self, current_time: str):
        """만료된 패키지 제거"""
        pass
```

---

### 2.3 TheftDetector

**책임**: 패키지 사라짐을 감지하고 도난 판단

**설계 원칙**:
- **SRP**: 도난 감지만 담당
- **OCP**: 다른 감지 로직으로 확장 가능

**클래스 구조**:
```python
class TheftDetector:
    """도난 감지기"""
    
    def __init__(self, confirmation_time: float = 3.0):
        self.confirmation_time = confirmation_time
        self.missing_packages: Dict[str, str] = {}  # package_id -> missing_since
    
    def check_theft(
        self,
        tracked_packages: List[TrackedPackage],
        current_time: str
    ) -> Optional[TheftDetectedEvent]:
        """도난 감지 확인"""
        # 1. "present" -> "missing" 전환 감지
        # 2. 3초 확인 로직
        # 3. 도난 이벤트 생성
        pass
    
    def _update_missing_state(
        self,
        tracked_packages: List[TrackedPackage],
        current_time: str
    ):
        """missing 상태 업데이트"""
        pass
    
    def _confirm_theft(
        self,
        package_id: str,
        current_time: str
    ) -> bool:
        """3초 확인 로직"""
        pass
```

---

## 3. 데이터 흐름

### 3.1 패키지 감지 플로우

```
Frame Input
    ↓
PackageDetector.detect()
    ↓
YOLO Model Inference
    ↓
Postprocessing
    ↓
List[Detection]
    ↓
PackageTracker.track()
    ↓
IOU Matching
    ↓
List[TrackedPackage]
    ↓
TheftDetector.check_theft()
    ↓
Optional[TheftDetectedEvent]
    ↓
EventBus.emit()
```

### 3.2 이벤트 플로우

```
PackageDetectedEvent
    ↓
EventBus
    ↓
PackageEventHandler
    ↓
Statistics Update
    ↓
Function Calling (get_package_count)
```

---

## 4. 클래스 다이어그램

```
┌─────────────────────┐
│   PackageDetector   │
├─────────────────────┤
│ - model: YOLO       │
│ - device: str       │
│ + detect()          │
│ + load_model()      │
└──────────┬──────────┘
           │ uses
           ▼
┌─────────────────────┐
│     Detection       │
├─────────────────────┤
│ + bbox              │
│ + confidence        │
│ + class_id          │
│ + timestamp         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   PackageTracker    │
├─────────────────────┤
│ - tracked_packages  │
│ + track()           │
│ + get_package()     │
└──────────┬──────────┘
           │ creates
           ▼
┌─────────────────────┐
│  TrackedPackage     │
├─────────────────────┤
│ + package_id        │
│ + detections        │
│ + status            │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   TheftDetector     │
├─────────────────────┤
│ + check_theft()     │
└──────────┬──────────┘
           │ creates
           ▼
┌─────────────────────┐
│ TheftDetectedEvent  │
└─────────────────────┘
```

---

## 5. 시퀀스 다이어그램

### 5.1 패키지 감지 및 도난 감지 시퀀스

```
E2ESystem -> PackageDetector: detect(frame)
PackageDetector -> YOLO Model: inference
YOLO Model -> PackageDetector: results
PackageDetector -> E2ESystem: List[Detection]

E2ESystem -> PackageTracker: track(detections)
PackageTracker -> PackageTracker: IOU matching
PackageTracker -> E2ESystem: List[TrackedPackage]

E2ESystem -> TheftDetector: check_theft(tracked_packages)
TheftDetector -> TheftDetector: 3초 확인 로직
alt 도난 확인됨
    TheftDetector -> E2ESystem: TheftDetectedEvent
    E2ESystem -> EventBus: emit(event)
end
```

---

## 6. 상태 다이어그램

### 6.1 패키지 상태 전이

```
[초기] 
  ↓ (패키지 감지)
[present]
  ↓ (1초 이상 감지 안됨)
[missing]
  ↓ (3초 확인)
[stolen]
  ↓ (이벤트 발행)
[완료]
```

---

## 7. 인터페이스 설계

### 7.1 추상화 계층

```python
from abc import ABC, abstractmethod

class BaseDetector(ABC):
    """감지기 추상 클래스"""
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        pass

class BaseTracker(ABC):
    """추적기 추상 클래스"""
    
    @abstractmethod
    def track(self, detections: List[Detection]) -> List[TrackedPackage]:
        pass

class BaseTheftDetector(ABC):
    """도난 감지기 추상 클래스"""
    
    @abstractmethod
    def check_theft(
        self,
        tracked_packages: List[TrackedPackage],
        current_time: str
    ) -> Optional[TheftDetectedEvent]:
        pass
```

**설계 원칙 적용**:
- **DIP**: 구체적인 구현이 아닌 추상화에 의존
- **OCP**: 새로운 감지/추적 알고리즘 추가 시 기존 코드 수정 불필요
- **LSP**: 모든 구현체가 추상 클래스를 대체 가능

---

## 8. 통합 포인트

### 8.1 E2ESystem 통합

```python
class E2ESystem:
    def __init__(self, config):
        # ... 기존 코드 ...
        self.package_detector: Optional[PackageDetector] = None
        self.package_tracker: Optional[PackageTracker] = None
        self.theft_detector: Optional[TheftDetector] = None
    
    def initialize(self):
        # ... 기존 초기화 ...
        if config.get('package_detection', {}).get('enabled', False):
            self._initialize_package_detection()
    
    def _initialize_package_detection(self):
        """패키지 감지 시스템 초기화"""
        pass
    
    def _process_frame_with_package_detection(self, frame):
        """프레임 처리 (패키지 감지 포함)"""
        # 1. VAD 처리
        # 2. 패키지 감지
        # 3. 패키지 추적
        # 4. 도난 감지
        # 5. 이벤트 발행
        pass
```

### 8.2 EventBus 통합

```python
# 이벤트 타입 추가
@dataclass
class PackageDetectedEvent(BaseEvent):
    package_id: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    camera_id: int
    frame_index: int

# 이벤트 핸들러 추가
class PackageEventHandler(BaseEventHandler):
    def handle_package_detected(self, event: PackageDetectedEvent):
        pass
```

### 8.3 Function Calling 통합

```python
# function_calling.py에 함수 추가
def get_package_count(e2e_system) -> Dict:
    """패키지 통계 조회"""
    pass

def get_package_details(e2e_system, package_id: str) -> Dict:
    """패키지 상세 정보 조회"""
    pass

def get_activity_log(e2e_system, limit: int = 10) -> Dict:
    """활동 로그 조회"""
    pass
```

---

## 9. 성능 고려사항

### 9.1 최적화 전략

1. **모델 최적화**
   - YOLO v12 nano 사용 (경량화)
   - 모델 양자화 (INT8)
   - 배치 처리 (가능한 경우)

2. **추적 최적화**
   - IOU 계산 최적화 (벡터화)
   - 만료된 패키지 주기적 정리
   - 메모리 효율적인 데이터 구조

3. **통합 최적화**
   - 비동기 처리 (가능한 경우)
   - 프레임 스킵 (옵션)
   - 멀티스레딩 활용

---

## 10. 보안 고려사항

1. **입력 검증**: 모든 입력 프레임 검증
2. **에러 처리**: 민감한 정보 노출 방지
3. **로깅**: 개인정보 포함 로깅 방지

---

## 11. 확장성 고려사항

1. **다른 객체 감지**: 패키지 외 다른 객체도 감지 가능하도록 확장
2. **다른 추적 알고리즘**: DeepSORT 등 고급 추적 알고리즘 지원
3. **멀티 카메라**: 여러 카메라에서 패키지 추적

