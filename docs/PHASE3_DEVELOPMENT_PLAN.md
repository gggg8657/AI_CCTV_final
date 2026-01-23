# Phase 3: Package Detection & Theft Detection 개발 기획서

**버전**: 1.0  
**작성일**: 2026-01-22  
**작성자**: AI Assistant  
**Linear Issue**: CHA-55 (생성 예정)

---

## 목차

1. [요구사항 분석](#1-요구사항-분석)
2. [시스템 설계](#2-시스템-설계)
3. [구현 계획](#3-구현-계획)
4. [테스트 계획](#4-테스트-계획)
5. [리스크 분석 및 대응](#5-리스크-분석-및-대응)
6. [일정 계획](#6-일정-계획)

---

## 1. 요구사항 분석

### 1.1 기능 요구사항

#### FR-1: 패키지 감지
- **설명**: YOLO v12 nano를 사용하여 비디오 프레임에서 패키지 객체를 감지
- **입력**: 비디오 프레임 (numpy array)
- **출력**: 감지된 패키지 리스트 (bounding box, confidence, class)
- **우선순위**: High

#### FR-2: 패키지 추적
- **설명**: 감지된 패키지를 시간에 따라 추적하고 ID 관리
- **입력**: 감지된 패키지 리스트
- **출력**: 추적된 패키지 정보 (ID, 위치, 시간)
- **우선순위**: High

#### FR-3: 도난 감지
- **설명**: 패키지가 사라진 경우 도난으로 판단 (3초 확인 로직)
- **입력**: 패키지 추적 정보
- **출력**: 도난 이벤트
- **우선순위**: High

#### FR-4: 이벤트 통합
- **설명**: 패키지 관련 이벤트를 EventBus를 통해 발행
- **입력**: 패키지 감지/추적/도난 정보
- **출력**: 이벤트 발행
- **우선순위**: Medium

#### FR-5: Function Calling 통합
- **설명**: Agent가 패키지 정보를 조회할 수 있는 함수 제공
- **입력**: 함수 호출 요청
- **출력**: 패키지 통계/상세 정보
- **우선순위**: Medium

### 1.2 비기능 요구사항

#### NFR-1: 성능
- 패키지 감지: 최소 30 FPS (YOLO v12 nano)
- 전체 파이프라인: VAD + Package Detection + VLM 통합 시 지연 최소화

#### NFR-2: 정확도
- 패키지 감지 정확도: COCO 데이터셋 기준 mAP@0.5 > 0.5
- 도난 감지 거짓 경보율: < 5%

#### NFR-3: 확장성
- 여러 패키지 동시 추적 지원
- 멀티 카메라 환경 지원 (향후)

#### NFR-4: 유지보수성
- 모듈화된 설계
- 명확한 인터페이스
- 테스트 가능한 구조

---

## 2. 시스템 설계

### 2.1 아키텍처 개요

```
┌─────────────────────────────────────────────────────────┐
│                    E2ESystem                            │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  VAD Engine  │  │ Package Det. │  │  VLM Analyzer│ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                  │                  │          │
│         └──────────────────┼──────────────────┘          │
│                            │                             │
│                    ┌───────▼────────┐                     │
│                    │   EventBus     │                     │
│                    └───────┬────────┘                     │
│                            │                             │
│                    ┌───────▼────────┐                     │
│                    │  Agent System  │                     │
│                    └────────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

### 2.2 컴포넌트 설계

#### 2.2.1 PackageDetector
- **책임**: YOLO 모델을 사용하여 패키지 객체 감지
- **인터페이스**:
  ```python
  class PackageDetector:
      def detect(self, frame: np.ndarray) -> List[Detection]
      def load_model(self, model_path: str) -> bool
  ```

#### 2.2.2 PackageTracker
- **책임**: 감지된 패키지를 시간에 따라 추적
- **인터페이스**:
  ```python
  class PackageTracker:
      def track(self, detections: List[Detection]) -> List[TrackedPackage]
      def get_package(self, package_id: str) -> Optional[TrackedPackage]
      def get_all_packages(self) -> List[TrackedPackage]
  ```

#### 2.2.3 TheftDetector
- **책임**: 패키지 사라짐을 감지하고 도난 판단
- **인터페이스**:
  ```python
  class TheftDetector:
      def check_theft(self, tracked_packages: List[TrackedPackage]) -> Optional[TheftEvent]
      def update_state(self, tracked_packages: List[TrackedPackage])
  ```

### 2.3 데이터 모델

#### Detection
```python
@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    timestamp: str
```

#### TrackedPackage
```python
@dataclass
class TrackedPackage:
    package_id: str
    detections: List[Detection]
    first_seen: str
    last_seen: str
    current_position: Tuple[int, int, int, int]
    status: str  # "present", "missing", "stolen"
```

### 2.4 이벤트 설계

#### PackageDetectedEvent
```python
@dataclass
class PackageDetectedEvent(BaseEvent):
    package_id: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    camera_id: int
    frame_index: int
```

#### PackageDisappearedEvent
```python
@dataclass
class PackageDisappearedEvent(BaseEvent):
    package_id: str
    last_seen: str
    camera_id: int
```

#### TheftDetectedEvent
```python
@dataclass
class TheftDetectedEvent(BaseEvent):
    package_id: str
    theft_time: str
    camera_id: int
    evidence_frames: List[str]  # base64 encoded frames
```

---

## 3. 구현 계획

### 3.1 단계별 구현 (Sprint 기반)

#### Sprint 1: YOLO 통합 및 기본 감지 (3일)
- [ ] YOLO v12 nano 모델 다운로드 및 설정
- [ ] PackageDetector 클래스 구현
- [ ] 기본 감지 기능 테스트
- [ ] E2ESystem에 통합

#### Sprint 2: 패키지 추적 시스템 (3일)
- [ ] PackageTracker 클래스 구현
- [ ] 추적 알고리즘 구현 (IOU 기반 또는 DeepSORT)
- [ ] 패키지 ID 관리 시스템
- [ ] 추적 기능 테스트

#### Sprint 3: 도난 감지 로직 (2일)
- [ ] TheftDetector 클래스 구현
- [ ] 3초 확인 로직 구현
- [ ] 거짓 경보 필터링
- [ ] 도난 감지 테스트

#### Sprint 4: 이벤트 통합 (2일)
- [ ] 이벤트 타입 정의 및 구현
- [ ] EventBus에 이벤트 핸들러 등록
- [ ] 이벤트 발행 로직 통합
- [ ] 이벤트 플로우 테스트

#### Sprint 5: Function Calling 통합 (2일)
- [ ] Function Calling 함수 구현
  - `get_package_count()`
  - `get_package_details()`
  - `get_activity_log()`
- [ ] FunctionRegistry에 등록
- [ ] Function Calling 테스트

#### Sprint 6: 통합 및 최적화 (2일)
- [ ] 전체 파이프라인 통합 테스트
- [ ] 성능 최적화
- [ ] 문서화
- [ ] 코드 리뷰 및 개선

### 3.2 파일 구조

```
src/
├── detection/
│   ├── __init__.py
│   ├── package_detector.py      # YOLO 기반 패키지 감지
│   ├── package_tracker.py        # 패키지 추적
│   └── theft_detector.py         # 도난 감지
├── agent/
│   └── function_calling.py       # Function Calling 함수 추가
└── utils/
    └── events.py                 # 이벤트 타입 추가
```

---

## 4. 테스트 계획

### 4.1 단위 테스트

#### PackageDetector 테스트
- [ ] 모델 로드 테스트
- [ ] 감지 기능 테스트 (더미 프레임)
- [ ] 에러 처리 테스트

#### PackageTracker 테스트
- [ ] 추적 시작 테스트
- [ ] ID 관리 테스트
- [ ] 패키지 업데이트 테스트
- [ ] 패키지 만료 테스트

#### TheftDetector 테스트
- [ ] 3초 확인 로직 테스트
- [ ] 거짓 경보 필터링 테스트
- [ ] 도난 이벤트 생성 테스트

### 4.2 통합 테스트

- [ ] E2ESystem과 통합 테스트
- [ ] EventBus 이벤트 플로우 테스트
- [ ] Function Calling 통합 테스트
- [ ] 전체 파이프라인 테스트 (VAD + Package Detection + VLM)

### 4.3 성능 테스트

- [ ] 패키지 감지 FPS 측정
- [ ] 메모리 사용량 측정
- [ ] 전체 파이프라인 레이턴시 측정

---

## 5. 리스크 분석 및 대응

### 5.1 기술적 리스크

| 리스크 | 영향도 | 확률 | 대응 방안 |
|--------|--------|------|-----------|
| YOLO 모델 성능 부족 | High | Medium | 더 큰 모델로 업그레이드 또는 커스텀 학습 |
| 추적 알고리즘 정확도 낮음 | Medium | Medium | DeepSORT 등 고급 추적 알고리즘 적용 |
| 메모리 사용량 과다 | Medium | Low | 모델 양자화, 배치 크기 조정 |
| GPU 메모리 부족 | High | Medium | CPU 모드 지원, 모델 경량화 |

### 5.2 일정 리스크

| 리스크 | 영향도 | 확률 | 대응 방안 |
|--------|--------|------|-----------|
| 구현 지연 | Medium | Medium | 우선순위 조정, 기능 축소 |
| 테스트 시간 부족 | Medium | Low | 자동화 테스트 강화 |

### 5.3 통합 리스크

| 리스크 | 영향도 | 확률 | 대응 방안 |
|--------|--------|------|-----------|
| 기존 시스템과 충돌 | High | Low | 점진적 통합, 철저한 테스트 |
| 이벤트 플로우 복잡도 증가 | Medium | Medium | 이벤트 핸들러 모듈화 |

---

## 6. 일정 계획

### 6.1 전체 일정 (2주)

```
Week 1:
- Day 1-3: Sprint 1 (YOLO 통합)
- Day 4-6: Sprint 2 (패키지 추적)

Week 2:
- Day 7-8: Sprint 3 (도난 감지)
- Day 9-10: Sprint 4 (이벤트 통합)
- Day 11-12: Sprint 5 (Function Calling)
- Day 13-14: Sprint 6 (통합 및 최적화)
```

### 6.2 마일스톤

- **M1 (Day 3)**: YOLO 통합 완료, 기본 감지 동작 확인
- **M2 (Day 6)**: 패키지 추적 시스템 완료
- **M3 (Day 8)**: 도난 감지 로직 완료
- **M4 (Day 10)**: 이벤트 통합 완료
- **M5 (Day 12)**: Function Calling 통합 완료
- **M6 (Day 14)**: 전체 통합 및 문서화 완료

---

## 7. 의존성

### 7.1 외부 라이브러리

- `ultralytics` (YOLO v12)
- `torch` (이미 설치됨)
- `numpy` (이미 설치됨)
- `opencv-python` (이미 설치됨)

### 7.2 내부 의존성

- EventBus 시스템 (Phase 1 완료)
- Function Calling 시스템 (Phase 2 완료)
- E2ESystem (기존)

---

## 8. 성공 기준

### 8.1 기능적 성공 기준

- ✅ 패키지 감지 정확도 > 50% (mAP@0.5)
- ✅ 패키지 추적 정확도 > 80%
- ✅ 도난 감지 거짓 경보율 < 5%
- ✅ 모든 단위 테스트 통과
- ✅ 모든 통합 테스트 통과

### 8.2 비기능적 성공 기준

- ✅ 패키지 감지 FPS > 30
- ✅ 전체 파이프라인 레이턴시 < 200ms
- ✅ 메모리 사용량 < 2GB (추가)
- ✅ 코드 커버리지 > 80%

---

## 9. 다음 단계

1. Linear 이슈 생성 및 할당
2. 브랜치 생성: `feature/cha-55-package-detection`
3. Sprint 1 시작: YOLO 통합
