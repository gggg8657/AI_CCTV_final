# Codex 실제 검토 응답 및 논의

**검토일**: 2026-01-22  
**검토자**: Codex CLI (실제 응답)  
**검토 대상**: Phase 3 Package Detection & Theft Detection 시스템

---

## Codex 실제 검토 응답 요약

### 주요 지적사항 (우선순위별)

#### P0 (필수 해결)
1. **COCO 데이터셋에 "package" 클래스 없음** ⚠️
   - 커스텀 클래스 매핑 또는 커스텀 데이터셋 필요
   - mAP 측정 기준 재검토 필요

2. **성능 목표 30 FPS 현실성 부족**
   - VAD + Package Detection + VLM 통합 시 30 FPS 달성 어려움
   - 프레임 스킵, 비동기 처리, 배칭 필요

3. **이벤트 메모리 폭증 위험**
   - `evidence_frames`를 base64로 직접 저장하면 메모리 부담 큼
   - 파일 경로/ID 참조로 변경 필요

4. **이벤트 폭주 처리 및 중복 제거 전략 필요**

#### P1 (성능 최적화)
1. **타임스탬프 처리 개선**
   - ISO string 대신 float (Unix time) 사용 권장
   - monotonic timestamp 사용 권장 (3초 확인 로직)

2. **IOU 추적 한계**
   - 미싱/가려짐 이슈
   - 향후 DeepSORT 고려

3. **일정 버퍼 필요**
   - 14일은 타이트함
   - 버퍼 포함 스케줄 제안

#### P2 (장기 검토)
1. **YOLO v12 nano 공개 여부 미확실**
   - Ultralytics에 포함 여부 확인 필요

---

## 논의 내용

### 1. COCO 데이터셋 "package" 클래스 문제 (P0)

**Codex 지적**:
- COCO 데이터셋에는 "package" 클래스가 명확히 존재하지 않음
- 커스텀 클래스 매핑 또는 커스텀 데이터셋 필요

**Cursor 분석**:
- ✅ **중요한 지적**: 이는 Phase 3의 핵심 가정을 뒤흔드는 문제
- COCO 클래스 확인 필요:
  - "suitcase" (클래스 ID: 28)
  - "handbag" (클래스 ID: 26)
  - "backpack" (클래스 ID: 27)
  - 또는 "box" 관련 클래스

**결정**:
1. **즉시 확인**: COCO 클래스 목록 확인 및 매핑 전략 수립
2. **대안 1**: COCO의 "suitcase", "handbag", "backpack" 등을 패키지로 간주
3. **대안 2**: 커스텀 데이터셋 준비 (향후)
4. **요구사항 수정**: "package" 클래스 → "suitcase/handbag/backpack" 클래스로 변경

---

### 2. 성능 목표 30 FPS 현실성 (P0)

**Codex 지적**:
- VAD + Package Detection + VLM 통합 시 30 FPS 달성 어려움
- 프레임 스킵, 비동기 처리, 배칭 필요

**Cursor 의견**:
- ✅ **동의**: 성능 목표 조정 필요
- 💡 **제안**: 
  - Phase 3 목표: 20 FPS 이상
  - 최적화 후 목표: 30 FPS
  - 프레임 스킵 옵션 추가 (예: 2프레임마다 1회 감지)

**결정**:
- **Phase 3 목표**: 20 FPS 이상 (현실적 목표)
- **최적화 목표**: 30 FPS (향후 개선)
- **구현**: 프레임 스킵 옵션 추가 (config.yaml에 `frame_skip: 2` 옵션)

---

### 3. 이벤트 메모리 폭증 위험 (P0)

**Codex 지적**:
- `evidence_frames`를 base64로 직접 저장하면 메모리 부담 큼
- 파일 경로/ID 참조로 변경 필요

**Cursor 의견**:
- ✅ **완전 동의**: 중요한 지적
- 💡 **제안**: 
  - `evidence_frames` → `evidence_frame_paths: List[str]`
  - 또는 `evidence_frame_ids: List[str]` (데이터베이스 참조)

**결정**:
- **TheftDetectedEvent 수정**:
  ```python
  @dataclass
  class TheftDetectedEvent(BaseEvent):
      package_id: str
      theft_time: str
      camera_id: int
      evidence_frame_paths: List[str]  # base64 대신 파일 경로
      # 또는
      evidence_frame_ids: List[str]  # DB 참조 ID
  ```

---

### 4. Factory 패턴 및 DI 도입 (P1)

**Codex 지적**:
- E2ESystem에서 구체 구현체를 직접 생성하면 DIP 위반
- Config 기반 Factory 패턴 도입 필요

**Cursor 의견**:
- ✅ **동의**: 확장성을 위해 필요
- ⚠️ **우려**: 초기 구현 시 복잡도 증가
- 💡 **제안**: Sprint 1에서는 간단하게, Sprint 6에서 리팩토링

**결정**:
- **Sprint 1**: YOLO 직접 의존으로 시작 (빠른 프로토타입)
- **Sprint 6**: Factory 패턴으로 리팩토링 (확장성 확보)

---

### 5. 타임스탬프 처리 개선 (P1)

**Codex 지적**:
- ISO string 대신 float (Unix time) 사용 권장
- monotonic timestamp 사용 권장 (3초 확인 로직)

**Cursor 의견**:
- ✅ **동의**: 성능 및 정확도 향상
- 💡 **제안**: 
  - 내부 로직: `time.time()` 기반 float 사용
  - 이벤트/API: ISO string으로 변환 (호환성)

**결정**:
- **내부 처리**: float (Unix time) 사용
- **이벤트/API**: ISO string으로 변환
- **3초 확인 로직**: `time.monotonic()` 사용

---

### 6. 일정 버퍼 필요 (P1)

**Codex 지적**:
- 14일은 타이트함
- 버퍼 포함 스케줄 제안

**Cursor 의견**:
- ✅ **동의**: 버퍼 필요
- 💡 **제안**: 2주 목표, 2.5주 실제 (버퍼 포함)

**결정**:
- **기본 일정**: 2주 유지 (목표)
- **버퍼 포함**: 2.5주 (실제 일정)
- **우선순위**: 핵심 기능 우선

---

## 즉시 수정 필요 사항 (P0)

### 1. COCO 클래스 매핑 전략 수립

**문제**: COCO에 "package" 클래스 없음

**해결 방안**:
```python
# COCO 클래스 매핑
COCO_PACKAGE_CLASSES = {
    26: "handbag",    # 핸드백
    27: "backpack",  # 백팩
    28: "suitcase",  # 여행가방
    # 또는 추가 클래스
}

# PackageDetector에서 사용
class PackageDetector:
    def __init__(self, ...):
        self.target_class_ids = [26, 27, 28]  # 패키지로 간주할 클래스
```

**요구사항 문서 수정**:
- "package" 클래스 → "suitcase/handbag/backpack" 클래스로 변경
- 또는 "package-like objects"로 표현

---

### 2. TheftDetectedEvent 수정

**현재 설계**:
```python
@dataclass
class TheftDetectedEvent(BaseEvent):
    evidence_frames: List[str]  # base64 encoded frames
```

**수정안**:
```python
@dataclass
class TheftDetectedEvent(BaseEvent):
    evidence_frame_paths: List[str]  # 파일 경로
    # 또는
    evidence_frame_ids: List[str]  # DB 참조 ID
```

---

### 3. 성능 목표 조정

**현재 목표**: 30 FPS

**수정안**:
- **Phase 3 목표**: 20 FPS 이상
- **최적화 목표**: 30 FPS (향후)
- **프레임 스킵 옵션**: config.yaml에 추가

---

## 개선된 설계안

### 1. Factory 패턴 (Sprint 6)

```python
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
```

### 2. 타임스탬프 처리 개선

```python
import time
from datetime import datetime

class TimeManager:
    @staticmethod
    def get_monotonic_time() -> float:
        """3초 확인 로직용"""
        return time.monotonic()
    
    @staticmethod
    def get_unix_time() -> float:
        """일반 시간 처리용"""
        return time.time()
    
    @staticmethod
    def to_iso_string(timestamp: float) -> str:
        """이벤트/API용 ISO string 변환"""
        return datetime.fromtimestamp(timestamp).isoformat()
```

### 3. 이벤트 메모리 최적화

```python
@dataclass
class TheftDetectedEvent(BaseEvent):
    package_id: str
    theft_time: str
    camera_id: int
    evidence_frame_paths: List[str]  # 파일 경로만 저장
    # 필요 시에만 프레임 로드
```

---

## 최종 합의사항

### 즉시 적용 (Sprint 1 전)

1. ✅ **COCO 클래스 매핑 전략 수립**
   - "package" → "suitcase/handbag/backpack" 매핑
   - 요구사항 문서 수정

2. ✅ **TheftDetectedEvent 수정**
   - `evidence_frames` → `evidence_frame_paths`

3. ✅ **성능 목표 조정**
   - 20 FPS 목표 (Phase 3)
   - 30 FPS 최적화 목표

4. ✅ **타임스탬프 처리 개선**
   - 내부: float (Unix time)
   - 이벤트/API: ISO string

### Sprint 중 적용

1. ✅ **프레임 스킵 옵션 추가**
2. ✅ **에러 처리 강화**
3. ✅ **테스트 전략 강화**

### Sprint 6에서 리팩토링

1. ✅ **Factory 패턴 도입**
2. ✅ **성능 최적화**
3. ✅ **비동기 처리 고려**

---

## 다음 단계

1. ✅ Codex 실제 검토 응답 분석 완료
2. ⏳ Phase 3 계획 문서 업데이트 (COCO 클래스, 성능 목표, 이벤트 구조)
3. ⏳ Linear 이슈 업데이트
4. ⏳ Sprint 1 시작 준비

---

**논의 완료일**: 2026-01-22  
**Codex 실제 응답**: ✅ 수신 완료  
**합의 도출**: ✅ 완료
