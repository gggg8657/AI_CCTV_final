# Codex 검토 피드백: Phase 3 개발 기획서

**검토일**: 2026-01-22  
**검토자**: Codex (코드 개발자 관점)  
**검토 대상**: Phase 3 Package Detection & Theft Detection 시스템

---

## 1. 아키텍처 설계 검토

### ✅ 긍정적인 점

1. **컴포넌트 분리 적절성**
   - PackageDetector, PackageTracker, TheftDetector의 책임 분리가 명확함
   - 각 컴포넌트가 단일 책임을 가짐 (SRP 준수)

2. **추상화 계층 설계**
   - BaseDetector, BaseTracker, BaseTheftDetector 추상 클래스 도입
   - 확장 가능한 구조 (OCP 준수)

### ⚠️ 개선 필요 사항

1. **의존성 관리**
   ```python
   # 현재 설계
   PackageDetector -> YOLO (직접 의존)
   
   # 개선 제안: DIP 적용
   PackageDetector -> BaseDetector -> YOLODetector
   ```
   - **문제**: PackageDetector가 YOLO에 직접 의존
   - **개선**: DetectorFactory 패턴 도입으로 다른 모델로 교체 가능하도록

2. **TheftDetector의 책임 범위**
   - **현재**: TheftDetector가 상태 관리와 이벤트 생성 모두 담당
   - **개선 제안**: TheftDetector는 도난 판단만, 상태 관리는 별도 클래스로 분리
   ```python
   class TheftDetector:
       """도난 판단만 담당"""
       def check_theft(...) -> bool
   
   class TheftStateManager:
       """도난 상태 관리"""
       def update_state(...)
       def create_event(...) -> TheftDetectedEvent
   ```

---

## 2. SOLID 원칙 준수 검토

### ✅ 잘 준수된 부분

1. **SRP (Single Responsibility Principle)**
   - 각 클래스가 명확한 단일 책임을 가짐
   - PackageDetector: 감지만
   - PackageTracker: 추적만
   - TheftDetector: 도난 판단만

2. **OCP (Open-Closed Principle)**
   - 추상화 계층을 통한 확장 가능성
   - 새로운 감지/추적 알고리즘 추가 시 기존 코드 수정 불필요

### ⚠️ 개선 필요 사항

1. **ISP (Interface Segregation Principle)**
   ```python
   # 현재 설계
   class BaseDetector(ABC):
       @abstractmethod
       def detect(self, frame) -> List[Detection]
       @abstractmethod
       def load_model(self) -> bool
   
   # 문제: 모든 Detector가 load_model을 구현해야 함
   # 개선: 인터페이스 분리
   class IDetector(ABC):
       @abstractmethod
       def detect(self, frame) -> List[Detection]
   
   class IModelLoader(ABC):
       @abstractmethod
       def load_model(self) -> bool
   ```

2. **DIP (Dependency Inversion Principle)**
   - PackageDetector가 구체적인 YOLO 구현에 의존
   - 개선: Factory 패턴 도입
   ```python
   class DetectorFactory:
       @staticmethod
       def create(model_type: str) -> BaseDetector:
           if model_type == "yolo":
               return YOLODetector()
           elif model_type == "custom":
               return CustomDetector()
   ```

---

## 3. 구현 복잡도 평가

### 일정 현실성

**현재 계획**: 2주 (14일, 6개 Sprint)

**평가**:
- ⚠️ **약간 낙관적**: 실제 개발 경험상 2.5-3주가 더 현실적
- **이유**:
  1. YOLO 통합 시 예상치 못한 이슈 발생 가능 (3일 → 4-5일)
  2. IOU 추적 알고리즘 튜닝 시간 필요 (3일 → 4일)
  3. 통합 테스트 및 버그 수정 시간 부족 (2일 → 3-4일)

**권장 일정**:
- Sprint 1: YOLO 통합 (4일) ← 1일 추가
- Sprint 2: 패키지 추적 (4일) ← 1일 추가
- Sprint 3: 도난 감지 (2일) ✅
- Sprint 4: 이벤트 통합 (2일) ✅
- Sprint 5: Function Calling (2일) ✅
- Sprint 6: 통합 및 최적화 (3일) ← 1일 추가
- **총 17일 (약 2.5주)**

### 기술적 난이도

**중간-높음**:
1. YOLO 통합: 중간 (ultralytics 라이브러리 사용으로 난이도 감소)
2. IOU 추적: 중간 (알고리즘 자체는 단순하나 튜닝 필요)
3. 도난 감지 로직: 낮음-중간 (비즈니스 로직)
4. 통합: 높음 (기존 시스템과의 통합 복잡도)

---

## 4. 성능 고려사항

### ⚠️ 잠재적 문제점

1. **YOLO 모델 통합 시 성능 영향**
   - **현재 목표**: 30 FPS
   - **실제 예상**: YOLO v12 nano는 30 FPS 가능하나, VAD + Package Detection + VLM 통합 시 **20-25 FPS** 예상
   - **개선 제안**:
     - 프레임 스킵 옵션 추가 (예: 2프레임마다 1회 감지)
     - 비동기 처리 (YOLO 추론을 별도 스레드에서)

2. **메모리 사용량**
   - **목표**: < 2GB 추가
   - **예상**: YOLO 모델만 500MB-1GB, 추적 데이터 구조 100-200MB
   - **총 예상**: 1-1.5GB 추가 (목표 달성 가능)

3. **실시간 처리 가능성**
   - **현재 설계**: 동기 처리
   - **개선 제안**: 비동기 파이프라인
   ```python
   # 개선안
   async def process_frame_async(frame):
       detections = await detector.detect_async(frame)
       tracked = await tracker.track_async(detections)
       theft = await theft_detector.check_async(tracked)
   ```

---

## 5. 개선 제안

### 우선순위 높음 (P0)

1. **Factory 패턴 도입**
   ```python
   class DetectorFactory:
       @staticmethod
       def create(config: dict) -> BaseDetector:
           detector_type = config.get('type', 'yolo')
           if detector_type == 'yolo':
               return YOLODetector(config)
           # 확장 가능
   ```

2. **에러 처리 강화**
   - YOLO 모델 로드 실패 시 graceful degradation
   - 추적 실패 시 이전 상태 유지

3. **테스트 전략 강화**
   - Mock 객체를 사용한 단위 테스트
   - 통합 테스트용 더미 데이터 생성기

### 우선순위 중간 (P1)

1. **설정 외부화**
   - IOU threshold, max_age 등을 config.yaml로 이동
   - 런타임 조정 가능하도록

2. **로깅 개선**
   - 구조화된 로깅 (JSON 형식)
   - 성능 메트릭 로깅

3. **메모리 최적화**
   - TrackedPackage의 detections 리스트 크기 제한
   - 오래된 detection 자동 정리

### 우선순위 낮음 (P2)

1. **비동기 처리**
   - 성능 향상을 위한 비동기 파이프라인
   - 프레임 큐 관리

2. **캐싱 전략**
   - YOLO 추론 결과 캐싱 (동일 프레임 재사용)

---

## 6. 잠재적 문제점

### 기술적 이슈

1. **YOLO 모델 버전 호환성**
   - **문제**: ultralytics 라이브러리 버전 업데이트 시 API 변경 가능
   - **대응**: requirements.txt에 버전 고정, 테스트 강화

2. **IOU 추적 정확도**
   - **문제**: 패키지가 겹치거나 가려질 때 추적 실패 가능
   - **대응**: DeepSORT 알고리즘으로 업그레이드 고려 (향후)

3. **타임스탬프 동기화**
   - **문제**: 여러 컴포넌트 간 시간 동기화
   - **대응**: 중앙화된 시간 관리자 도입

### 통합 복잡도

1. **EventBus 통합**
   - **문제**: 이벤트 순서 보장 필요
   - **대응**: 이벤트 타임스탬프 기반 정렬

2. **Function Calling 통합**
   - **문제**: PackageTracker 상태 접근 시 동시성 이슈
   - **대응**: Thread-safe 구현 또는 Lock 사용

### 유지보수성

1. **코드 중복**
   - **문제**: Detection, TrackedPackage 변환 로직 중복 가능
   - **대응**: Mapper 클래스 도입

2. **테스트 커버리지**
   - **목표**: > 80%
   - **권장**: 각 Sprint마다 테스트 작성 (TDD 접근)

---

## 7. 구체적인 코드 개선 예시

### 예시 1: Factory 패턴 적용

```python
# 개선 전
class PackageDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)  # 직접 의존

# 개선 후
class DetectorFactory:
    @staticmethod
    def create(config: dict) -> BaseDetector:
        detector_type = config.get('type', 'yolo')
        if detector_type == 'yolo':
            return YOLODetector(config)
        elif detector_type == 'custom':
            return CustomDetector(config)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

class PackageDetector(BaseDetector):
    def __init__(self, config: dict):
        self.detector = DetectorFactory.create(config)
```

### 예시 2: ISP 적용

```python
# 개선 전
class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame) -> List[Detection]
    @abstractmethod
    def load_model(self) -> bool  # 모든 Detector가 필요하지 않음

# 개선 후
class IDetector(ABC):
    """감지 기능만"""
    @abstractmethod
    def detect(self, frame) -> List[Detection]

class IModelLoader(ABC):
    """모델 로드 기능만"""
    @abstractmethod
    def load_model(self) -> bool

class PackageDetector(IDetector, IModelLoader):
    """두 인터페이스 모두 구현"""
    pass
```

---

## 8. 최종 권장사항

### 즉시 적용 (Sprint 1 전)

1. ✅ Factory 패턴 도입
2. ✅ ISP 적용 (인터페이스 분리)
3. ✅ 에러 처리 강화

### Sprint 중 적용

1. ✅ 설정 외부화
2. ✅ 로깅 개선
3. ✅ 테스트 전략 강화

### 향후 개선

1. 비동기 처리
2. DeepSORT 알고리즘 업그레이드
3. 성능 최적화

---

## 9. 결론

**전체 평가**: ⭐⭐⭐⭐ (4/5)

**장점**:
- 명확한 컴포넌트 분리
- SOLID 원칙 대부분 준수
- 확장 가능한 설계

**개선 필요**:
- DIP 완전 적용 (Factory 패턴)
- ISP 적용 (인터페이스 분리)
- 일정 여유 추가 (2주 → 2.5주)

**권장 조치**:
1. Sprint 1 시작 전 Factory 패턴 및 ISP 적용
2. 일정을 2.5주로 조정
3. 각 Sprint마다 테스트 작성 (TDD)

---

**검토 완료일**: 2026-01-22  
**다음 단계**: 피드백 반영 및 설계 개선
