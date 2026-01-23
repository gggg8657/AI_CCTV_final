# Phase 3: 구현 계획서 (Implementation Plan)

**버전**: 1.0  
**작성일**: 2026-01-22  
**Linear Issue**: CHA-55

---

## 목차

1. [구현 단계](#1-구현-단계)
2. [파일 구조](#2-파일-구조)
3. [코딩 표준](#3-코딩-표준)
4. [의존성 관리](#4-의존성-관리)
5. [코드 리뷰 체크리스트](#5-코드-리뷰-체크리스트)

---

## 1. 구현 단계

### Sprint 1: YOLO 통합 및 기본 감지 (3일)

#### Day 1: 환경 설정 및 모델 준비
- [ ] YOLO v12 nano 모델 다운로드
- [ ] `ultralytics` 패키지 설치
- [ ] 모델 로드 테스트
- [ ] 기본 감지 테스트 (더미 프레임)

#### Day 2: PackageDetector 구현
- [ ] `src/detection/package_detector.py` 생성
- [ ] `PackageDetector` 클래스 구현
  - `__init__()`: 초기화
  - `load_model()`: 모델 로드
  - `detect()`: 감지 메인 로직
  - `_preprocess()`: 전처리
  - `_postprocess()`: 후처리
- [ ] `Detection` dataclass 정의
- [ ] 단위 테스트 작성

#### Day 3: E2ESystem 통합
- [ ] `E2ESystem`에 `PackageDetector` 통합
- [ ] Config 설정 추가 (`config.yaml`)
- [ ] 초기화 로직 추가
- [ ] 통합 테스트

**산출물**:
- `src/detection/package_detector.py`
- `tests/test_package_detector.py`
- Config 업데이트

---

### Sprint 2: 패키지 추적 시스템 (3일)

#### Day 4: PackageTracker 기본 구조
- [ ] `src/detection/package_tracker.py` 생성
- [ ] `TrackedPackage` dataclass 정의
- [ ] 기본 추적 구조 구현
- [ ] ID 관리 시스템 구현

#### Day 5: IOU 기반 매칭
- [ ] IOU 계산 함수 구현
- [ ] 매칭 알고리즘 구현
- [ ] 패키지 업데이트 로직
- [ ] 만료 패키지 제거 로직

#### Day 6: 테스트 및 통합
- [ ] 단위 테스트 작성
- [ ] E2ESystem 통합
- [ ] 통합 테스트

**산출물**:
- `src/detection/package_tracker.py`
- `tests/test_package_tracker.py`

---

### Sprint 3: 도난 감지 로직 (2일)

#### Day 7: TheftDetector 구현
- [ ] `src/detection/theft_detector.py` 생성
- [ ] 3초 확인 로직 구현
- [ ] 거짓 경보 필터링
- [ ] 도난 이벤트 생성

#### Day 8: 테스트 및 통합
- [ ] 단위 테스트 작성
- [ ] E2ESystem 통합
- [ ] 통합 테스트

**산출물**:
- `src/detection/theft_detector.py`
- `tests/test_theft_detector.py`

---

### Sprint 4: 이벤트 통합 (2일)

#### Day 9: 이벤트 타입 정의
- [ ] `src/utils/events.py`에 이벤트 타입 추가
  - `PackageDetectedEvent`
  - `PackageDisappearedEvent`
  - `TheftDetectedEvent`
- [ ] 이벤트 핸들러 구현
  - `PackageEventHandler`

#### Day 10: EventBus 통합
- [ ] 이벤트 발행 로직 추가
- [ ] E2ESystem에 핸들러 등록
- [ ] 이벤트 플로우 테스트

**산출물**:
- 이벤트 타입 추가
- `src/utils/event_handlers.py` 업데이트
- `tests/test_package_events.py`

---

### Sprint 5: Function Calling 통합 (2일)

#### Day 11: Function 구현
- [ ] `src/agent/function_calling.py`에 함수 추가
  - `get_package_count()`
  - `get_package_details()`
  - `get_activity_log()`
- [ ] `register_core_functions()` 업데이트

#### Day 12: 테스트 및 문서화
- [ ] Function Calling 테스트
- [ ] API 문서 업데이트
- [ ] 사용 예제 작성

**산출물**:
- Function Calling 함수 추가
- `tests/test_package_functions.py`
- 문서 업데이트

---

### Sprint 6: 통합 및 최적화 (2일)

#### Day 13: 전체 통합
- [ ] 전체 파이프라인 통합 테스트
- [ ] 성능 측정
- [ ] 버그 수정

#### Day 14: 최적화 및 문서화
- [ ] 성능 최적화
- [ ] 코드 리뷰 (Codex)
- [ ] 최종 문서화
- [ ] 릴리스 준비

**산출물**:
- 통합 테스트 결과
- 성능 리포트
- 최종 문서

---

## 2. 파일 구조

```
src/
├── detection/                    # 새 디렉토리
│   ├── __init__.py
│   ├── package_detector.py       # YOLO 기반 감지
│   ├── package_tracker.py        # 패키지 추적
│   └── theft_detector.py         # 도난 감지
│
├── agent/
│   └── function_calling.py       # 함수 추가
│
├── utils/
│   └── events.py                 # 이벤트 타입 추가
│
└── ...

tests/
├── test_package_detector.py
├── test_package_tracker.py
├── test_theft_detector.py
├── test_package_events.py
├── test_package_functions.py
└── test_package_integration.py

configs/
└── config.yaml                   # 패키지 감지 설정 추가

docs/
├── PHASE3_DEVELOPMENT_PLAN.md
├── PHASE3_REQUIREMENTS.md
├── PHASE3_DESIGN_DOCUMENT.md
└── PHASE3_IMPLEMENTATION_PLAN.md (이 문서)
```

---

## 3. 코딩 표준

### 3.1 Python 스타일 가이드
- PEP 8 준수
- 타입 힌트 필수
- Docstring 필수 (Google 스타일)

### 3.2 OOP 원칙 준수
- **SRP**: 각 클래스는 단일 책임
- **OCP**: 확장에는 열려있고 수정에는 닫혀있음
- **LSP**: 자식 클래스는 부모 클래스를 대체 가능
- **ISP**: 인터페이스 분리
- **DIP**: 추상화에 의존

### 3.3 네이밍 규칙
- 클래스: PascalCase (`PackageDetector`)
- 함수/변수: snake_case (`detect_packages`)
- 상수: UPPER_SNAKE_CASE (`IOU_THRESHOLD`)

### 3.4 에러 처리
- 모든 예외 상황 처리
- 명확한 에러 메시지
- 로깅 필수

---

## 4. 의존성 관리

### 4.1 새 의존성
```txt
# requirements.txt에 추가
ultralytics>=8.0.0  # YOLO v12
```

### 4.2 의존성 설치
```bash
pip install ultralytics
```

### 4.3 모델 다운로드
- YOLO v12 nano: 자동 다운로드 (ultralytics)
- 또는 수동 다운로드 후 경로 지정

---

## 5. 코드 리뷰 체크리스트

### 5.1 기능적 검토
- [ ] 요구사항 충족 여부
- [ ] 에러 처리 적절성
- [ ] 엣지 케이스 처리
- [ ] 성능 요구사항 충족

### 5.2 코드 품질
- [ ] SOLID 원칙 준수
- [ ] 타입 힌트 완전성
- [ ] Docstring 완전성
- [ ] 코드 가독성
- [ ] 중복 코드 제거

### 5.3 테스트
- [ ] 단위 테스트 커버리지 > 80%
- [ ] 통합 테스트 작성
- [ ] 성능 테스트 작성

### 5.4 문서화
- [ ] API 문서 완성
- [ ] 사용 예제 작성
- [ ] 아키텍처 문서 업데이트

---

## 6. 리스크 관리

### 6.1 기술적 리스크
- **YOLO 모델 성능**: 대안 모델 준비
- **추적 정확도**: 알고리즘 개선 계획
- **성능 이슈**: 최적화 계획

### 6.2 일정 리스크
- **지연 가능성**: 우선순위 조정 계획
- **테스트 시간**: 자동화 강화

---

## 7. 성공 기준

### 7.1 기능적 기준
- ✅ 모든 기능 요구사항 구현
- ✅ 모든 테스트 통과
- ✅ 성능 요구사항 충족

### 7.2 품질 기준
- ✅ 코드 커버리지 > 80%
- ✅ Codex 코드 리뷰 통과
- ✅ SOLID 원칙 준수 확인

