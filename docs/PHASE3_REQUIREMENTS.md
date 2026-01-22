# Phase 3: 요구사항 명세서 (Requirements Specification)

**버전**: 1.0  
**작성일**: 2026-01-22  
**Linear Issue**: CHA-55

---

## 1. 기능 요구사항 (Functional Requirements)

### FR-1: 패키지 객체 감지

**ID**: FR-1  
**우선순위**: High  
**설명**: YOLO v12 nano 모델을 사용하여 비디오 프레임에서 패키지 객체를 감지합니다.

**입력**:
- 비디오 프레임 (numpy.ndarray, shape: (H, W, 3), dtype: uint8)
- 모델 경로 (str, optional)

**출력**:
- 감지 결과 리스트 (List[Detection])
  - Detection 객체:
    - bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    - confidence: float  # 0.0 ~ 1.0
    - class_id: int  # COCO 클래스 ID
    - class_name: str  # "package" 또는 COCO 클래스명
    - timestamp: str  # ISO 8601 형식

**전제조건**:
- YOLO v12 nano 모델이 설치되어 있어야 함
- 입력 프레임이 유효한 이미지 형식이어야 함

**사후조건**:
- 감지 결과가 반환됨 (빈 리스트 가능)
- 모델이 정상적으로 로드됨

**예외 처리**:
- 모델 로드 실패 시 에러 로깅 및 False 반환
- 프레임이 None이거나 잘못된 형식이면 빈 리스트 반환

---

### FR-2: 패키지 추적

**ID**: FR-2  
**우선순위**: High  
**설명**: 감지된 패키지를 시간에 따라 추적하고 고유 ID를 부여합니다.

**입력**:
- 현재 프레임의 감지 결과 (List[Detection])
- 이전 프레임의 추적 결과 (List[TrackedPackage])

**출력**:
- 추적된 패키지 리스트 (List[TrackedPackage])
  - TrackedPackage 객체:
    - package_id: str  # 고유 ID (예: "pkg_001")
    - detections: List[Detection]  # 이력
    - first_seen: str  # ISO 8601
    - last_seen: str  # ISO 8601
    - current_position: Tuple[int, int, int, int]
    - status: str  # "present", "missing", "stolen"

**전제조건**:
- 감지 결과가 제공되어야 함

**사후조건**:
- 모든 감지된 패키지에 ID가 부여됨
- 이전에 추적 중이던 패키지는 동일한 ID 유지 (IOU 기반 매칭)

**추적 알고리즘**:
- IOU (Intersection over Union) 기반 매칭
- IOU threshold: 0.3
- 최대 추적 시간: 30초 (비활성 패키지는 만료)

---

### FR-3: 도난 감지

**ID**: FR-3  
**우선순위**: High  
**설명**: 패키지가 사라진 경우 도난으로 판단합니다. 3초 확인 로직을 사용하여 거짓 경보를 방지합니다.

**입력**:
- 추적된 패키지 리스트 (List[TrackedPackage])
- 현재 시간 (str, ISO 8601)

**출력**:
- 도난 이벤트 (Optional[TheftDetectedEvent])
  - package_id: str
  - theft_time: str
  - camera_id: int
  - evidence_frames: List[str]  # base64 encoded frames

**도난 판단 로직**:
1. 패키지가 "present" 상태에서 감지되지 않음
2. 3초 동안 연속으로 감지되지 않음
3. 패키지가 프레임 밖으로 나가지 않음 (옵션)

**전제조건**:
- 추적 시스템이 활성화되어 있어야 함
- 패키지가 최소 1초 이상 추적되었어야 함

**사후조건**:
- 도난이 감지되면 TheftDetectedEvent 생성
- 패키지 상태가 "stolen"으로 변경됨

---

### FR-4: 이벤트 통합

**ID**: FR-4  
**우선순위**: Medium  
**설명**: 패키지 관련 이벤트를 EventBus를 통해 발행합니다.

**이벤트 타입**:
1. PackageDetectedEvent
   - 패키지가 처음 감지되었을 때
   - 발행 주기: 패키지 ID 생성 시 1회

2. PackageDisappearedEvent
   - 패키지가 감지되지 않기 시작했을 때
   - 발행 주기: 패키지가 1초 이상 감지되지 않을 때

3. TheftDetectedEvent
   - 도난이 확인되었을 때
   - 발행 주기: 3초 확인 로직 통과 시 1회

**전제조건**:
- EventBus가 초기화되어 있어야 함
- 이벤트 핸들러가 등록되어 있어야 함

**사후조건**:
- 이벤트가 EventBus를 통해 발행됨
- 이벤트 히스토리에 기록됨

---

### FR-5: Function Calling 통합

**ID**: FR-5  
**우선순위**: Medium  
**설명**: Agent가 패키지 정보를 조회할 수 있는 함수를 제공합니다.

**함수 목록**:

#### get_package_count()
- **설명**: 현재 추적 중인 패키지 개수 반환
- **파라미터**: 없음
- **반환값**: 
  ```python
  {
      "ok": bool,
      "data": {
          "total": int,
          "present": int,
          "missing": int,
          "stolen": int
      }
  }
  ```

#### get_package_details(package_id: str)
- **설명**: 특정 패키지의 상세 정보 반환
- **파라미터**: package_id (str)
- **반환값**:
  ```python
  {
      "ok": bool,
      "data": {
          "package_id": str,
          "first_seen": str,
          "last_seen": str,
          "status": str,
          "current_position": Tuple[int, int, int, int],
          "detection_count": int
      }
  }
  ```

#### get_activity_log(limit: int = 10)
- **설명**: 패키지 활동 로그 반환
- **파라미터**: limit (int, 기본값: 10)
- **반환값**:
  ```python
  {
      "ok": bool,
      "data": {
          "activities": List[Dict],
          "total": int
      }
  }
  ```

---

## 2. 비기능 요구사항 (Non-Functional Requirements)

### NFR-1: 성능

- **패키지 감지 속도**: 최소 30 FPS (YOLO v12 nano)
- **전체 파이프라인 레이턴시**: VAD + Package Detection 통합 시 < 200ms
- **메모리 사용량**: 기존 시스템 대비 < 2GB 추가

### NFR-2: 정확도

- **패키지 감지 정확도**: COCO 데이터셋 기준 mAP@0.5 > 0.5
- **추적 정확도**: 패키지 ID 유지율 > 80%
- **도난 감지 거짓 경보율**: < 5%

### NFR-3: 확장성

- **동시 추적 패키지 수**: 최소 10개
- **멀티 카메라 지원**: 향후 확장 가능한 구조

### NFR-4: 유지보수성

- **코드 커버리지**: > 80%
- **문서화**: 모든 public 메서드에 docstring
- **타입 힌트**: 모든 함수에 타입 힌트

### NFR-5: 안정성

- **에러 처리**: 모든 예외 상황에 대한 적절한 처리
- **로깅**: 중요한 이벤트는 로깅
- **복구**: 모델 로드 실패 시 graceful degradation

---

## 3. 제약사항 (Constraints)

### C-1: 기술적 제약
- YOLO v12 nano 모델 사용 (경량화)
- Python 3.9+ 호환
- PyTorch 기반

### C-2: 리소스 제약
- GPU 메모리: 최대 4GB (기존 시스템과 공유)
- CPU: 멀티스레딩 지원

### C-3: 호환성 제약
- 기존 EventBus 시스템과 호환
- 기존 Function Calling 시스템과 호환
- 기존 E2ESystem과 통합 가능

---

## 4. 인터페이스 명세

### 4.1 PackageDetector 인터페이스

```python
class PackageDetector:
    """패키지 객체 감지기"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Args:
            model_path: YOLO 모델 경로
            device: "cuda" 또는 "cpu"
        """
    
    def load_model(self) -> bool:
        """모델 로드"""
        pass
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        프레임에서 패키지 감지
        
        Args:
            frame: 입력 프레임 (H, W, 3)
        
        Returns:
            감지 결과 리스트
        """
        pass
```

### 4.2 PackageTracker 인터페이스

```python
class PackageTracker:
    """패키지 추적기"""
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        """
        Args:
            iou_threshold: IOU 매칭 임계값
            max_age: 패키지 최대 추적 시간 (초)
        """
    
    def track(self, detections: List[Detection]) -> List[TrackedPackage]:
        """
        감지 결과를 추적
        
        Args:
            detections: 현재 프레임의 감지 결과
        
        Returns:
            추적된 패키지 리스트
        """
        pass
    
    def get_package(self, package_id: str) -> Optional[TrackedPackage]:
        """특정 패키지 정보 조회"""
        pass
    
    def get_all_packages(self) -> List[TrackedPackage]:
        """모든 추적 중인 패키지 조회"""
        pass
```

### 4.3 TheftDetector 인터페이스

```python
class TheftDetector:
    """도난 감지기"""
    
    def __init__(self, confirmation_time: float = 3.0):
        """
        Args:
            confirmation_time: 확인 시간 (초)
        """
    
    def check_theft(
        self, 
        tracked_packages: List[TrackedPackage],
        current_time: str
    ) -> Optional[TheftDetectedEvent]:
        """
        도난 감지 확인
        
        Args:
            tracked_packages: 추적 중인 패키지 리스트
            current_time: 현재 시간 (ISO 8601)
        
        Returns:
            도난 이벤트 (없으면 None)
        """
        pass
```

---

## 5. 데이터 모델 명세

### 5.1 Detection

```python
@dataclass
class Detection:
    """객체 감지 결과"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float  # 0.0 ~ 1.0
    class_id: int  # COCO 클래스 ID
    class_name: str  # 클래스명
    timestamp: str  # ISO 8601
```

### 5.2 TrackedPackage

```python
@dataclass
class TrackedPackage:
    """추적 중인 패키지"""
    package_id: str  # 고유 ID
    detections: List[Detection]  # 감지 이력
    first_seen: str  # ISO 8601
    last_seen: str  # ISO 8601
    current_position: Tuple[int, int, int, int]  # 최신 위치
    status: str  # "present", "missing", "stolen"
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        pass
```

---

## 6. 에러 처리 명세

### 6.1 에러 타입

- `ModelLoadError`: 모델 로드 실패
- `DetectionError`: 감지 과정에서 에러
- `TrackingError`: 추적 과정에서 에러
- `TheftDetectionError`: 도난 감지 과정에서 에러

### 6.2 에러 처리 전략

- **모델 로드 실패**: 로깅 후 False 반환, 시스템은 계속 동작 (패키지 감지 비활성화)
- **감지 에러**: 로깅 후 빈 리스트 반환
- **추적 에러**: 로깅 후 이전 상태 유지
- **도난 감지 에러**: 로깅 후 None 반환 (이벤트 미발행)

---

## 7. 테스트 요구사항

### 7.1 단위 테스트
- 각 클래스의 모든 public 메서드 테스트
- 에러 케이스 테스트
- 엣지 케이스 테스트

### 7.2 통합 테스트
- E2ESystem 통합 테스트
- EventBus 통합 테스트
- Function Calling 통합 테스트

### 7.3 성능 테스트
- FPS 측정
- 메모리 사용량 측정
- 레이턴시 측정

---

## 8. 승인 기준

다음 조건을 모두 만족해야 Phase 3가 완료된 것으로 간주:

1. ✅ 모든 기능 요구사항 구현 완료
2. ✅ 모든 비기능 요구사항 충족
3. ✅ 코드 커버리지 > 80%
4. ✅ 모든 테스트 통과
5. ✅ 문서화 완료
6. ✅ 코드 리뷰 완료 (Codex 검수 포함)
