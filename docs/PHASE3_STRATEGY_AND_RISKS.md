# Phase 3: 전략 수립 및 리스크 관리

**작성일**: 2026-01-22  
**기반**: Codex 실제 검토 응답  
**Linear Issue**: CHA-55

---

## 목차

1. [전략 수립](#1-전략-수립)
2. [리스크 분석 및 대응](#2-리스크-분석-및-대응)
3. [즉시 조치 사항](#3-즉시-조치-사항)
4. [수정된 계획](#4-수정된-계획)

---

## 1. 전략 수립

### 1.1 COCO 클래스 매핑 전략 (P0)

**문제**: COCO 데이터셋에 "package" 클래스가 명확히 존재하지 않음

**전략**:

#### 옵션 1: COCO 기존 클래스 활용 (권장)
```python
# COCO 클래스 매핑
COCO_PACKAGE_LIKE_CLASSES = {
    26: "handbag",    # 핸드백
    27: "backpack",  # 백팩
    28: "suitcase",  # 여행가방
    24: "backpack",  # 중복 확인 필요
}

# PackageDetector에서 사용
class PackageDetector:
    def __init__(self, ...):
        # 패키지로 간주할 COCO 클래스 ID
        self.target_class_ids = [26, 27, 28]
        self.class_names = ["handbag", "backpack", "suitcase"]
```

**장점**:
- 커스텀 학습 불필요
- 즉시 사용 가능
- COCO 모델 그대로 활용

**단점**:
- 정확도가 실제 "package"보다 낮을 수 있음
- "package"와 "suitcase"는 다를 수 있음

#### 옵션 2: 커스텀 데이터셋 준비 (향후)
- 패키지 이미지 수집
- YOLO 커스텀 학습
- 정확도 향상 가능

**결정**: **옵션 1로 시작**, 필요 시 옵션 2로 전환

**구현 계획**:
1. COCO 클래스 목록 확인 및 매핑 테이블 작성
2. PackageDetector에서 target_class_ids 설정
3. 요구사항 문서 수정: "package" → "package-like objects (suitcase/handbag/backpack)"

---

### 1.2 성능 목표 조정 전략 (P0)

**문제**: 30 FPS 목표가 현실적이지 않음

**전략**:

#### 단계별 성능 목표
1. **Phase 3 기본 목표**: 20 FPS 이상
2. **Phase 3 최적화 목표**: 25 FPS 이상
3. **향후 최적화 목표**: 30 FPS

#### 성능 최적화 전략

**1. 프레임 스킵 옵션**
```yaml
# config.yaml
package_detection:
  enabled: true
  frame_skip: 2  # 2프레임마다 1회 감지
  target_fps: 20
```

**2. 비동기 처리 (Sprint 6)**
```python
# 비동기 파이프라인
async def process_frame_async(frame):
    # YOLO 추론을 별도 스레드에서
    detections = await asyncio.to_thread(detector.detect, frame)
    tracked = await asyncio.to_thread(tracker.track, detections)
    theft = await asyncio.to_thread(theft_detector.check_theft, tracked)
```

**3. GPU 메모리 최적화**
- 모델 양자화 (INT8)
- 배치 크기 조정
- 프레임 해상도 조정 옵션

**결정**:
- **Phase 3 목표**: 20 FPS 이상
- **프레임 스킵 옵션**: config.yaml에 추가
- **비동기 처리**: Sprint 6에서 검토

---

### 1.3 이벤트 메모리 최적화 전략 (P0)

**문제**: `evidence_frames`를 base64로 직접 저장하면 메모리 부담 큼

**전략**:

#### 옵션 1: 파일 경로 저장 (권장)
```python
@dataclass
class TheftDetectedEvent(BaseEvent):
    package_id: str
    theft_time: str
    camera_id: int
    evidence_frame_paths: List[str]  # 파일 경로만 저장
    # 필요 시에만 프레임 로드
```

**장점**:
- 메모리 사용량 최소화
- 이벤트 크기 작음
- 파일 시스템 활용

**단점**:
- 파일 관리 필요
- 파일 삭제 정책 필요

#### 옵션 2: DB 참조 ID 저장
```python
@dataclass
class TheftDetectedEvent(BaseEvent):
    evidence_frame_ids: List[str]  # DB 참조 ID
```

**결정**: **옵션 1 (파일 경로 저장)**

**구현 계획**:
1. TheftDetectedEvent 수정
2. 프레임 저장 로직 추가 (clips 디렉토리 활용)
3. 파일 경로만 이벤트에 포함
4. 파일 삭제 정책 수립 (예: 7일 후 자동 삭제)

---

### 1.4 타임스탬프 처리 개선 전략 (P1)

**문제**: ISO string으로만 관리하면 시간 계산 비효율적

**전략**:

```python
import time
from datetime import datetime

class TimeManager:
    """시간 관리 유틸리티"""
    
    @staticmethod
    def get_monotonic_time() -> float:
        """3초 확인 로직용 (시스템 시간 변경에 영향받지 않음)"""
        return time.monotonic()
    
    @staticmethod
    def get_unix_time() -> float:
        """일반 시간 처리용"""
        return time.time()
    
    @staticmethod
    def to_iso_string(timestamp: float) -> str:
        """이벤트/API용 ISO string 변환"""
        return datetime.fromtimestamp(timestamp).isoformat()
    
    @staticmethod
    def from_iso_string(iso_string: str) -> float:
        """ISO string을 float로 변환"""
        return datetime.fromisoformat(iso_string).timestamp()
```

**데이터 모델 수정**:
```python
@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    timestamp: float  # Unix time (내부 처리)
    timestamp_iso: str  # ISO string (이벤트/API용)
    
    @classmethod
    def create(cls, bbox, confidence, class_id, class_name):
        now = time.time()
        return cls(
            bbox=bbox,
            confidence=confidence,
            class_id=class_id,
            class_name=class_name,
            timestamp=now,
            timestamp_iso=TimeManager.to_iso_string(now)
        )
```

**결정**:
- **내부 처리**: float (Unix time) 사용
- **이벤트/API**: ISO string으로 변환
- **3초 확인 로직**: `time.monotonic()` 사용

---

### 1.5 이벤트 폭주 처리 전략 (P0)

**문제**: 이벤트 폭주 및 중복 제거 전략 필요

**전략**:

#### 1. 이벤트 디바운싱
```python
class EventDebouncer:
    """이벤트 중복 제거"""
    
    def __init__(self, debounce_time: float = 1.0):
        self.debounce_time = debounce_time
        self.last_events: Dict[str, float] = {}  # event_key -> last_time
    
    def should_emit(self, event_key: str) -> bool:
        """이벤트 발행 여부 결정"""
        now = time.monotonic()
        if event_key in self.last_events:
            if now - self.last_events[event_key] < self.debounce_time:
                return False
        self.last_events[event_key] = now
        return True
```

#### 2. 이벤트 큐 및 배치 처리
```python
class EventQueue:
    """이벤트 큐 관리"""
    
    def __init__(self, max_size: int = 100):
        self.queue: List[BaseEvent] = []
        self.max_size = max_size
    
    def add(self, event: BaseEvent):
        """이벤트 추가 (크기 제한)"""
        if len(self.queue) >= self.max_size:
            # 오래된 이벤트 제거
            self.queue.pop(0)
        self.queue.append(event)
```

**결정**:
- **이벤트 디바운싱**: PackageDetectedEvent에 적용 (같은 패키지 ID 중복 방지)
- **이벤트 큐 크기 제한**: EventBus에 적용
- **배치 처리**: 필요 시 검토

---

### 1.6 Factory 패턴 및 DI 전략 (P1)

**문제**: E2ESystem에서 구체 구현체 직접 생성 (DIP 위반)

**전략**:

#### Sprint 1: 간단한 시작
```python
# Sprint 1: 직접 의존 (빠른 프로토타입)
class E2ESystem:
    def _initialize_package_detection(self):
        self.package_detector = PackageDetector(
            model_path=self.config['package_detection']['model_path']
        )
```

#### Sprint 6: Factory 패턴으로 리팩토링
```python
class DetectorFactory:
    @staticmethod
    def create_detector(config: dict) -> BaseDetector:
        detector_type = config.get('type', 'yolo')
        if detector_type == 'yolo':
            return YOLODetector(config)
        elif detector_type == 'custom':
            return CustomDetector(config)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

class E2ESystem:
    def _initialize_package_detection(self):
        self.package_detector = DetectorFactory.create_detector(
            self.config['package_detection']
        )
```

**결정**:
- **Sprint 1**: 직접 의존으로 시작
- **Sprint 6**: Factory 패턴으로 리팩토링

---

## 2. 리스크 분석 및 대응

### 2.1 기술적 리스크

#### 리스크 1: COCO 클래스 매핑 정확도 (P0)

**리스크 설명**:
- COCO의 "suitcase/handbag/backpack"이 실제 "package"와 다를 수 있음
- 감지 정확도 요구사항(mAP@0.5 > 0.5) 달성 어려울 수 있음

**영향도**: High  
**확률**: Medium

**대응 방안**:
1. **즉시**: COCO 클래스 매핑 테이블 작성 및 테스트
2. **단기**: 실제 데이터로 정확도 측정
3. **중기**: 정확도가 낮으면 커스텀 데이터셋 준비 검토

**완화 전략**:
- 여러 클래스 조합 사용 (suitcase + handbag + backpack)
- Confidence threshold 조정
- 후처리 필터링 추가

---

#### 리스크 2: 성능 목표 미달 (P0)

**리스크 설명**:
- 30 FPS 목표 달성 어려움
- VAD + Package Detection + VLM 통합 시 병목 발생

**영향도**: High  
**확률**: High

**대응 방안**:
1. **즉시**: 성능 목표를 20 FPS로 조정
2. **단기**: 프레임 스킵 옵션 추가
3. **중기**: 비동기 처리 구현 (Sprint 6)
4. **장기**: GPU 최적화, 모델 양자화

**완화 전략**:
- 프레임 스킵 옵션 (config.yaml)
- 비동기 파이프라인
- 성능 모니터링 도구 추가

---

#### 리스크 3: 메모리 사용량 초과 (P0)

**리스크 설명**:
- `evidence_frames`를 base64로 저장하면 메모리 부담
- 목표 2GB 초과 가능

**영향도**: High  
**확률**: Medium

**대응 방안**:
1. **즉시**: TheftDetectedEvent 수정 (파일 경로 저장)
2. **단기**: 프레임 저장 정책 수립
3. **중기**: 파일 자동 삭제 정책 구현

**완화 전략**:
- 파일 경로만 저장
- 프레임 저장소 크기 제한
- 자동 정리 스크립트

---

#### 리스크 4: YOLO v12 nano 공개 여부 미확실 (P2)

**리스크 설명**:
- YOLO v12 nano가 Ultralytics에 포함되어 있는지 불확실
- YOLO v11 또는 v8로 대체 필요할 수 있음

**영향도**: Medium  
**확률**: Low

**대응 방안**:
1. **즉시**: YOLO v11 nano 또는 v8 nano로 대체 가능하도록 설계
2. **단기**: 실제 모델 확인 및 테스트
3. **중기**: 필요 시 모델 변경

**완화 전략**:
- 모델 경로를 config.yaml로 외부화
- 여러 모델 버전 지원 가능하도록 설계

---

#### 리스크 5: IOU 추적 정확도 (P1)

**리스크 설명**:
- 패키지가 겹치거나 가려질 때 추적 실패
- ID 유지율 80% 목표 달성 어려울 수 있음

**영향도**: Medium  
**확률**: Medium

**대응 방안**:
1. **단기**: IOU threshold 튜닝
2. **중기**: DeepSORT 알고리즘으로 업그레이드 검토
3. **장기**: 커스텀 추적 알고리즘 개발

**완화 전략**:
- IOU threshold 조정 가능하도록 (config.yaml)
- 추적 실패 시 로깅 강화
- 향후 DeepSORT 통합 계획

---

### 2.2 일정 리스크

#### 리스크 6: 일정 지연 (P1)

**리스크 설명**:
- 14일 일정이 타이트함
- 예상치 못한 이슈 발생 시 지연 가능

**영향도**: Medium  
**확률**: Medium

**대응 방안**:
1. **즉시**: 버퍼 포함 일정 수립 (2.5주)
2. **단기**: 우선순위 조정 (핵심 기능 우선)
3. **중기**: 기능 축소 옵션 준비

**완화 전략**:
- 핵심 기능 우선 구현
- 부가 기능은 선택적
- 일정 모니터링 및 조정

---

### 2.3 통합 리스크

#### 리스크 7: 기존 시스템과의 통합 복잡도 (P1)

**리스크 설명**:
- E2ESystem에 기존 VAD/Agent 시스템과 통합 시 성능/동기화 이슈
- 이벤트 플로우 복잡도 증가

**영향도**: Medium  
**확률**: Medium

**대응 방안**:
1. **단기**: 점진적 통합 (단계별 테스트)
2. **중기**: 이벤트 핸들러 모듈화
3. **장기**: 성능 프로파일링 및 최적화

**완화 전략**:
- 단계별 통합 테스트
- 이벤트 순서 보장 (타임스탬프 기반)
- 동시성 처리 (Thread-safe)

---

## 3. 즉시 조치 사항

### 3.1 Sprint 1 시작 전 (필수)

#### 1. COCO 클래스 매핑 전략 수립
- [ ] COCO 클래스 목록 확인
- [ ] 매핑 테이블 작성
- [ ] PackageDetector에 target_class_ids 설정
- [ ] 요구사항 문서 수정

#### 2. TheftDetectedEvent 수정
- [ ] `evidence_frames` → `evidence_frame_paths` 변경
- [ ] 프레임 저장 로직 설계
- [ ] 파일 삭제 정책 수립

#### 3. 성능 목표 조정
- [ ] Phase 3 목표: 20 FPS로 변경
- [ ] 프레임 스킵 옵션 추가 (config.yaml)
- [ ] 성능 모니터링 계획 수립

#### 4. 타임스탬프 처리 개선
- [ ] TimeManager 클래스 구현
- [ ] Detection 데이터 모델 수정
- [ ] 3초 확인 로직에 monotonic time 사용

---

### 3.2 Sprint 1-5 중 (권장)

#### 1. 이벤트 디바운싱 구현
- [ ] EventDebouncer 클래스 구현
- [ ] PackageDetectedEvent에 적용

#### 2. 에러 처리 강화
- [ ] YOLO 모델 로드 실패 시 graceful degradation
- [ ] 추적 실패 시 이전 상태 유지

#### 3. 테스트 전략 강화
- [ ] TDD 접근
- [ ] Mock 객체 사용
- [ ] 각 Sprint마다 테스트 작성

---

### 3.3 Sprint 6 (리팩토링)

#### 1. Factory 패턴 도입
- [ ] DetectorFactory 구현
- [ ] E2ESystem 리팩토링

#### 2. 성능 최적화
- [ ] 비동기 처리 검토
- [ ] GPU 메모리 최적화
- [ ] 프레임 스킵 최적화

---

## 4. 수정된 계획

### 4.1 수정된 일정

```
Week 1:
- Day 1-4: Sprint 1 (YOLO 통합) ← 1일 추가
  - COCO 클래스 매핑 전략 수립
  - TheftDetectedEvent 수정
  - 성능 목표 조정
- Day 5-8: Sprint 2 (패키지 추적) ← 1일 추가

Week 2:
- Day 9-10: Sprint 3 (도난 감지) ✅
- Day 11-12: Sprint 4 (이벤트 통합) ✅
- Day 13-14: Sprint 5 (Function Calling) ✅

Week 3 (버퍼):
- Day 15-17: Sprint 6 (통합 및 최적화) ← 1일 추가
```

**총 기간**: 2.5주 (17일)

---

### 4.2 수정된 성공 기준

**기능적 기준**:
- ✅ 패키지 감지 정확도 > 50% (mAP@0.5) - COCO 클래스 매핑 기준
- ✅ 패키지 추적 정확도 > 80%
- ✅ 도난 감지 거짓 경보율 < 5%

**비기능적 기준**:
- ✅ 패키지 감지 FPS > 20 (Phase 3 목표) ← 조정
- ✅ 최적화 후 FPS > 30 (향후 목표)
- ✅ 전체 파이프라인 레이턴시 < 250ms (조정)
- ✅ 메모리 사용량 < 2GB (추가)
- ✅ 코드 커버리지 > 80%

---

### 4.3 수정된 데이터 모델

#### Detection
```python
@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    timestamp: float  # Unix time (내부 처리)
    timestamp_iso: str  # ISO string (이벤트/API용)
```

#### TheftDetectedEvent
```python
@dataclass
class TheftDetectedEvent(BaseEvent):
    package_id: str
    theft_time: str
    camera_id: int
    evidence_frame_paths: List[str]  # base64 대신 파일 경로
```

---

### 4.4 수정된 Config 구조

```yaml
package_detection:
  enabled: true
  model:
    type: yolo  # yolo, custom
    path: yolo11n.pt  # 또는 yolo12n.pt
    target_classes: [26, 27, 28]  # COCO 클래스 ID
    class_names: [handbag, backpack, suitcase]
  performance:
    frame_skip: 2  # 프레임 스킵 옵션
    target_fps: 20  # 목표 FPS
  tracking:
    iou_threshold: 0.3
    max_age: 30  # 초
  theft_detection:
    confirmation_time: 3.0  # 초
  storage:
    evidence_frame_dir: clips/theft_evidence
    max_storage_days: 7  # 자동 삭제 기간
```

---

## 5. 실행 계획

### 5.1 즉시 실행 (오늘)

1. ✅ COCO 클래스 매핑 전략 수립
2. ✅ Phase 3 계획 문서 업데이트
3. ✅ Linear 이슈 업데이트
4. ✅ Config 구조 설계

### 5.2 Sprint 1 시작 전

1. ⏳ COCO 클래스 목록 확인
2. ⏳ TheftDetectedEvent 수정
3. ⏳ TimeManager 구현
4. ⏳ Config.yaml 업데이트

### 5.3 Sprint 진행 중

1. ⏳ 각 Sprint마다 테스트 작성
2. ⏳ 성능 모니터링
3. ⏳ 리스크 모니터링

---

## 6. 리스크 대응 매트릭스

| 리스크 | 우선순위 | 영향도 | 확률 | 대응 방안 | 담당 Sprint |
|--------|---------|--------|------|-----------|-------------|
| COCO 클래스 매핑 | P0 | High | Medium | 클래스 매핑 전략 | Sprint 1 |
| 성능 목표 미달 | P0 | High | High | 목표 조정, 프레임 스킵 | Sprint 1, 6 |
| 메모리 사용량 초과 | P0 | High | Medium | 이벤트 구조 개선 | Sprint 1 |
| 이벤트 폭주 | P0 | High | Medium | 디바운싱 구현 | Sprint 4 |
| IOU 추적 정확도 | P1 | Medium | Medium | 튜닝, DeepSORT 검토 | Sprint 2, 향후 |
| 일정 지연 | P1 | Medium | Medium | 버퍼 포함, 우선순위 조정 | 전체 |
| 통합 복잡도 | P1 | Medium | Medium | 점진적 통합, 테스트 | Sprint 6 |
| YOLO v12 공개 여부 | P2 | Medium | Low | v11/v8 대체 가능 | Sprint 1 |

---

## 7. 다음 단계

1. ✅ 전략 수립 완료
2. ⏳ COCO 클래스 매핑 전략 상세화
3. ⏳ Phase 3 계획 문서 업데이트
4. ⏳ Linear 이슈 업데이트
5. ⏳ Sprint 1 시작 준비

---

**작성 완료일**: 2026-01-22  
**승인 필요**: 사용자 검토 및 승인
