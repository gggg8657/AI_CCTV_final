# Vision-Agents 통합 요구사항 정의서

**버전**: 1.0  
**작성일**: 2026-01-21  
**작성자**: AI Assistant

---

## 목차

1. [개요](#1-개요)
2. [기능 요구사항](#2-기능-요구사항)
3. [비기능 요구사항](#3-비기능-요구사항)
4. [시스템 요구사항](#4-시스템-요구사항)
5. [인터페이스 요구사항](#5-인터페이스-요구사항)
6. [데이터 요구사항](#6-데이터-요구사항)
7. [보안 요구사항](#7-보안-요구사항)
8. [검증 및 테스트 요구사항](#8-검증-및-테스트-요구사항)

---

## 1. 개요

### 1.1 목적

Vision-Agents의 Security Camera Example을 참고하여 현재 AI_CCTV_final 프로젝트에 다음 기능을 통합:

1. 이벤트 기반 아키텍처
2. Function Calling 시스템
3. 얼굴 인식 및 추적
4. 패키지 감지 및 도난 감지
5. 음성 인터랙션 (선택)
6. 실시간 WebRTC 처리 (선택)

### 1.2 범위

**포함:**
- 이벤트 시스템 도입
- Function Calling 구현
- 얼굴 인식 기능
- 패키지 감지 기능
- 기존 시스템과의 통합

**제외:**
- Vision-Agents Agent 클래스 직접 사용 (패턴만 참고)
- GetStream Edge 필수 사용 (선택 사항)
- 완전한 아키텍처 재설계

### 1.3 제약사항

- 기존 VAD/VLM/Agent 시스템 유지
- 기존 UI (CLI/Streamlit/React) 호환성 유지
- Python 3.10+ 호환
- GPU 메모리 제한 고려

---

## 2. 기능 요구사항

### 2.1 이벤트 시스템 (FR-001 ~ FR-010)

#### FR-001: 이벤트 버스 구현
- **우선순위**: P0
- **설명**: 중앙 집중식 이벤트 버스 시스템 구현
- **입력**: 이벤트 객체
- **출력**: 구독자에게 이벤트 전달
- **검증**: 이벤트 발행 및 구독 테스트

#### FR-002: VAD 이벤트
- **우선순위**: P0
- **설명**: VAD 이상 탐지 시 이벤트 발행
- **이벤트 타입**: `AnomalyDetectedEvent`
- **속성**: 
  - `frame_id: int`
  - `score: float`
  - `timestamp: datetime`
  - `frame: np.ndarray`

#### FR-003: VLM 이벤트
- **우선순위**: P0
- **설명**: VLM 분석 완료 시 이벤트 발행
- **이벤트 타입**: `VLMAnalysisCompletedEvent`
- **속성**:
  - `detected_type: str`
  - `description: str`
  - `actions: List[str]`
  - `confidence: float`

#### FR-004: Agent 이벤트
- **우선순위**: P0
- **설명**: Agent 대응 계획 수립 시 이벤트 발행
- **이벤트 타입**: `AgentResponseEvent`
- **속성**:
  - `plan: List[str]`
  - `priority: int`
  - `estimated_time: float`

#### FR-005: 프레임 처리 이벤트
- **우선순위**: P1
- **설명**: 각 프레임 처리 완료 시 이벤트 발행
- **이벤트 타입**: `FrameProcessedEvent`
- **속성**:
  - `frame_id: int`
  - `processing_time: float`
  - `vad_score: float`

#### FR-006: 통계 업데이트 이벤트
- **우선순위**: P1
- **설명**: 시스템 통계 업데이트 시 이벤트 발행
- **이벤트 타입**: `StatsUpdatedEvent`
- **속성**:
  - `total_frames: int`
  - `anomaly_count: int`
  - `avg_processing_time: float`

#### FR-007: 이벤트 구독 메커니즘
- **우선순위**: P0
- **설명**: 컴포넌트가 이벤트를 구독할 수 있는 메커니즘
- **인터페이스**: `@event_bus.subscribe` 데코레이터
- **예시**:
  ```python
  @event_bus.subscribe
  async def on_anomaly(event: AnomalyDetectedEvent):
      # 처리 로직
  ```

#### FR-008: 이벤트 필터링
- **우선순위**: P2
- **설명**: 이벤트 타입별 필터링 기능
- **기능**: 특정 이벤트 타입만 구독

#### FR-009: 이벤트 로깅
- **우선순위**: P1
- **설명**: 모든 이벤트를 로그 파일에 기록
- **형식**: JSON
- **위치**: `logs/events_YYYYMMDD.json`

#### FR-010: 이벤트 히스토리
- **우선순위**: P2
- **설명**: 최근 N개 이벤트를 메모리에 보관
- **기본값**: 최근 1000개 이벤트

### 2.2 Function Calling 시스템 (FR-011 ~ FR-025)

#### FR-011: 함수 등록 시스템
- **우선순위**: P0
- **설명**: Python 함수를 LLM이 호출할 수 있도록 등록
- **인터페이스**: `@llm.register_function()` 데코레이터
- **예시**:
  ```python
  @llm.register_function(
      description="Get system status"
  )
  async def get_system_status() -> Dict:
      return {"status": "running", ...}
  ```

#### FR-012: 시스템 상태 조회 함수
- **우선순위**: P0
- **함수명**: `get_system_status()`
- **설명**: 현재 시스템 상태 조회
- **반환값**:
  ```python
  {
      "status": "running" | "stopped" | "error",
      "vad_model": "mnad",
      "vad_threshold": 0.5,
      "vlm_enabled": true,
      "agent_enabled": true,
      "total_frames": 12345,
      "anomaly_count": 12
  }
  ```

#### FR-013: 최근 이벤트 조회 함수
- **우선순위**: P0
- **함수명**: `get_recent_events(limit: int = 20)`
- **설명**: 최근 이상 탐지 이벤트 조회
- **파라미터**: `limit` (기본값: 20)
- **반환값**: 이벤트 리스트

#### FR-014: 이상 탐지 통계 함수
- **우선순위**: P0
- **함수명**: `get_anomaly_statistics()`
- **설명**: 이상 탐지 통계 조회
- **반환값**:
  ```python
  {
      "total_anomalies": 12,
      "anomalies_by_type": {...},
      "avg_score": 0.75,
      "time_range": "2026-01-21 00:00:00 ~ 2026-01-21 23:59:59"
  }
  ```

#### FR-015: VAD 임계값 변경 함수
- **우선순위**: P1
- **함수명**: `update_vad_threshold(value: float)`
- **설명**: VAD 이상 탐지 임계값 변경
- **파라미터**: `value` (0.0 ~ 1.0)
- **반환값**: 성공 여부

#### FR-016: VLM 활성화/비활성화 함수
- **우선순위**: P1
- **함수명**: `enable_vlm(enabled: bool)`
- **설명**: VLM 분석 활성화/비활성화
- **파라미터**: `enabled` (bool)
- **반환값**: 성공 여부

#### FR-017: Agent Flow 변경 함수
- **우선순위**: P1
- **함수명**: `set_agent_flow(flow_type: str)`
- **설명**: Agent Flow 타입 변경
- **파라미터**: `flow_type` ("sequential" | "hierarchical" | "collaborative")
- **반환값**: 성공 여부

#### FR-018: 자연어 질의 처리
- **우선순위**: P0
- **설명**: 사용자 자연어 질의를 Function Calling으로 변환
- **예시**:
  - "시스템 상태 알려줘" → `get_system_status()`
  - "최근 이상 탐지 이벤트 10개 보여줘" → `get_recent_events(limit=10)`
  - "VAD 임계값을 0.7로 올려줘" → `update_vad_threshold(0.7)`

#### FR-019: 함수 실행 결과 LLM 응답
- **우선순위**: P0
- **설명**: 함수 실행 결과를 자연어로 변환하여 사용자에게 전달
- **예시**: 
  - 함수 결과: `{"status": "running", "anomaly_count": 12}`
  - LLM 응답: "시스템은 정상 작동 중이며, 현재까지 12건의 이상 상황이 탐지되었습니다."

#### FR-020: 함수 실행 에러 처리
- **우선순위**: P0
- **설명**: 함수 실행 중 에러 발생 시 적절한 에러 메시지 반환
- **에러 타입**: 
  - 파라미터 오류
  - 실행 오류
  - 권한 오류

### 2.3 얼굴 인식 기능 (FR-021 ~ FR-030)

#### FR-021: 얼굴 감지
- **우선순위**: P1
- **설명**: 비디오 프레임에서 얼굴 감지
- **모델**: InsightFace 또는 FaceNet
- **정확도**: > 95%
- **처리 속도**: < 50ms per frame

#### FR-022: 얼굴 임베딩 생성
- **우선순위**: P1
- **설명**: 감지된 얼굴에서 임베딩 벡터 생성
- **차원**: 512 또는 128
- **용도**: 얼굴 비교 및 인식

#### FR-023: 얼굴 데이터베이스
- **우선순위**: P1
- **설명**: 등록된 얼굴 임베딩 저장
- **저장소**: SQLite 또는 PostgreSQL
- **스키마**:
  ```sql
  CREATE TABLE faces (
      face_id TEXT PRIMARY KEY,
      name TEXT,
      embedding BLOB,
      registered_at TIMESTAMP,
      last_seen TIMESTAMP
  )
  ```

#### FR-024: 얼굴 인식
- **우선순위**: P1
- **설명**: 감지된 얼굴을 데이터베이스와 비교하여 인식
- **임계값**: 코사인 유사도 > 0.6
- **처리 시간**: < 100ms

#### FR-025: 얼굴 추적 (30분 윈도우)
- **우선순위**: P1
- **설명**: 30분 시간 윈도우 내 얼굴 추적
- **기능**:
  - 같은 사람 재감지 시 카운트 증가
  - 30분 경과 시 새 방문자로 처리
  - 방문자 통계 관리

#### FR-026: PersonDetectedEvent
- **우선순위**: P1
- **설명**: 새로운 사람 감지 또는 재방문 시 이벤트 발행
- **속성**:
  - `face_id: str`
  - `is_new: bool`
  - `name: Optional[str]`
  - `detection_count: int`
  - `timestamp: datetime`

#### FR-027: PersonDisappearedEvent
- **우선순위**: P1
- **설명**: 사람이 프레임에서 사라질 때 이벤트 발행
- **속성**:
  - `face_id: str`
  - `name: Optional[str]`
  - `last_seen: datetime`

#### FR-028: 얼굴 등록 함수
- **우선순위**: P1
- **함수명**: `remember_my_face(name: str)`
- **설명**: 현재 프레임의 얼굴을 이름과 함께 등록
- **파라미터**: `name` (str)
- **반환값**: 성공 여부 및 face_id

#### FR-029: 방문자 수 조회 함수
- **우선순위**: P1
- **함수명**: `get_visitor_count()`
- **설명**: 지난 30분간 고유 방문자 수 조회
- **반환값**:
  ```python
  {
      "unique_visitors": 5,
      "total_detections": 12,
      "time_window": "30 minutes"
  }
  ```

#### FR-030: 방문자 상세 정보 함수
- **우선순위**: P1
- **함수명**: `get_visitor_details()`
- **설명**: 모든 방문자의 상세 정보 조회
- **반환값**: 방문자 리스트

### 2.4 패키지 감지 기능 (FR-031 ~ FR-040)

#### FR-031: 패키지 감지
- **우선순위**: P1
- **설명**: YOLO 기반 패키지 객체 감지
- **모델**: YOLOv11 또는 YOLOv8
- **커스텀 모델**: 필요 시 학습 가능
- **정확도**: > 90%
- **처리 속도**: < 100ms per frame

#### FR-032: 패키지 추적
- **우선순위**: P1
- **설명**: 감지된 패키지의 위치 및 상태 추적
- **기능**:
  - 패키지 ID 할당
  - 위치 추적 (bbox)
  - 시간 추적

#### FR-033: 패키지 데이터베이스
- **우선순위**: P1
- **설명**: 패키지 감지 이력 저장
- **스키마**:
  ```sql
  CREATE TABLE packages (
      package_id TEXT PRIMARY KEY,
      first_seen TIMESTAMP,
      last_seen TIMESTAMP,
      detection_count INT,
      confidence FLOAT,
      picked_up_by TEXT,
      status TEXT  -- "present" | "stolen" | "removed"
  )
  ```

#### FR-034: 도난 감지 로직
- **우선순위**: P1
- **설명**: 패키지가 사라질 때 도난 여부 판단
- **로직**:
  1. 패키지 사라짐 감지
  2. 3초 대기
  3. 재등장 여부 확인
  4. 재등장 없으면 도난 판정

#### FR-035: PackageDetectedEvent
- **우선순위**: P1
- **설명**: 패키지 감지 시 이벤트 발행
- **속성**:
  - `package_id: str`
  - `is_new: bool`
  - `confidence: float`
  - `bbox: Tuple[int, int, int, int]`
  - `timestamp: datetime`

#### FR-036: PackageDisappearedEvent
- **우선순위**: P1
- **설명**: 패키지 사라짐 시 이벤트 발행
- **속성**:
  - `package_id: str`
  - `picker_face_id: Optional[str]`
  - `timestamp: datetime`

#### FR-037: TheftDetectedEvent
- **우선순위**: P1
- **설명**: 도난 판정 시 이벤트 발행
- **속성**:
  - `package_id: str`
  - `suspect_face_id: Optional[str]`
  - `timestamp: datetime`

#### FR-038: 패키지 통계 함수
- **우선순위**: P1
- **함수명**: `get_package_count()`
- **설명**: 패키지 통계 조회
- **반환값**:
  ```python
  {
      "currently_visible": 2,
      "total_seen": 15,
      "stolen_count": 3
  }
  ```

#### FR-039: 패키지 상세 정보 함수
- **우선순위**: P1
- **함수명**: `get_package_details()`
- **설명**: 모든 패키지의 상세 정보 조회
- **반환값**: 패키지 리스트

#### FR-040: 활동 로그 함수
- **우선순위**: P1
- **함수명**: `get_activity_log(limit: int = 20)`
- **설명**: 최근 활동 로그 조회
- **반환값**: 활동 이벤트 리스트

### 2.5 음성 인터랙션 (선택, FR-041 ~ FR-045)

#### FR-041: STT 통합
- **우선순위**: P2
- **설명**: 음성을 텍스트로 변환
- **옵션**: Deepgram 또는 Fast-Whisper
- **지연시간**: < 200ms

#### FR-042: TTS 통합
- **우선순위**: P2
- **설명**: 텍스트를 음성으로 변환
- **옵션**: ElevenLabs 또는 Kokoro
- **지연시간**: < 300ms

#### FR-043: 음성 질의 처리
- **우선순위**: P2
- **설명**: 음성 입력을 Function Calling으로 변환
- **흐름**: 음성 → STT → Function Calling → TTS → 음성

#### FR-044: 음성 알림
- **우선순위**: P2
- **설명**: 이상 상황 감지 시 음성 알림
- **예시**: "경고: 이상 상황이 감지되었습니다."

#### FR-045: 음성 설정
- **우선순위**: P2
- **설명**: 음성 인터랙션 활성화/비활성화
- **함수**: `enable_voice_interaction(enabled: bool)`

---

## 3. 비기능 요구사항

### 3.1 성능 요구사항

| 요구사항 | 목표값 | 측정 방법 |
|---------|--------|----------|
| 이벤트 처리 지연시간 | < 10ms | 이벤트 발행부터 구독자 처리까지 |
| Function Calling 응답 시간 | < 2초 | 질의부터 응답까지 |
| 얼굴 인식 처리 시간 | < 100ms | 프레임 입력부터 인식 결과까지 |
| 패키지 감지 처리 시간 | < 100ms | 프레임 입력부터 감지 결과까지 |
| 전체 시스템 FPS | > 30 FPS | VAD만 활성화 시 |

### 3.2 확장성 요구사항

- 최대 16개 카메라 동시 처리 (멀티 카메라 확장 시)
- 이벤트 구독자 수 제한 없음
- Function Calling 함수 수 제한 없음

### 3.3 안정성 요구사항

- 한 컴포넌트의 오류가 전체 시스템에 영향 없음
- 이벤트 처리 실패 시 재시도 메커니즘
- 함수 실행 실패 시 적절한 에러 처리

### 3.4 호환성 요구사항

- 기존 VAD/VLM/Agent 시스템과 호환
- 기존 UI (CLI/Streamlit/React)와 호환
- Python 3.10+ 호환

---

## 4. 시스템 요구사항

### 4.1 하드웨어 요구사항

| 구성 | 최소 | 권장 |
|------|------|------|
| GPU | RTX 3080 (10GB) | RTX 4090 (24GB) |
| CPU | 8코어 | 16코어+ |
| RAM | 16GB | 32GB |
| 저장장치 | SSD 256GB | NVMe 1TB |

### 4.2 소프트웨어 요구사항

- Python 3.10+
- CUDA 11.8+ (GPU 사용 시)
- PyTorch 2.0+
- OpenCV 4.8+

### 4.3 외부 서비스 요구사항 (선택)

- **Deepgram API** (STT 사용 시)
- **ElevenLabs API** (TTS 사용 시)
- **GetStream 계정** (WebRTC 사용 시)

---

## 5. 인터페이스 요구사항

### 5.1 이벤트 버스 인터페이스

```python
class EventBus:
    def subscribe(self, event_type: Type[Event], handler: Callable)
    def publish(self, event: Event)
    def unsubscribe(self, event_type: Type[Event], handler: Callable)
```

### 5.2 Function Calling 인터페이스

```python
class FunctionRegistry:
    def register(self, func: Callable, description: str)
    def call(self, function_name: str, arguments: Dict) -> Any
    def list_functions(self) -> List[Dict]
```

### 5.3 얼굴 인식 인터페이스

```python
class FaceRecognition:
    def detect_faces(self, frame: np.ndarray) -> List[Face]
    def recognize_face(self, face: Face) -> Optional[str]
    def register_face(self, face: Face, name: str) -> str
    def get_visitor_count(self) -> int
```

### 5.4 패키지 감지 인터페이스

```python
class PackageDetector:
    def detect_packages(self, frame: np.ndarray) -> List[Package]
    def track_package(self, package_id: str) -> PackageStatus
    def check_theft(self, package_id: str) -> bool
```

---

## 6. 데이터 요구사항

### 6.1 얼굴 데이터베이스 스키마

```sql
CREATE TABLE faces (
    face_id TEXT PRIMARY KEY,
    name TEXT,
    embedding BLOB,
    registered_at TIMESTAMP,
    last_seen TIMESTAMP,
    detection_count INT DEFAULT 0
);

CREATE INDEX idx_faces_name ON faces(name);
CREATE INDEX idx_faces_last_seen ON faces(last_seen);
```

### 6.2 패키지 데이터베이스 스키마

```sql
CREATE TABLE packages (
    package_id TEXT PRIMARY KEY,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    detection_count INT DEFAULT 0,
    confidence FLOAT,
    picked_up_by TEXT,
    status TEXT,
    bbox_x INT,
    bbox_y INT,
    bbox_w INT,
    bbox_h INT
);

CREATE INDEX idx_packages_status ON packages(status);
CREATE INDEX idx_packages_last_seen ON packages(last_seen);
```

### 6.3 이벤트 로그 형식

```json
{
    "event_id": "evt_1234567890",
    "event_type": "AnomalyDetectedEvent",
    "timestamp": "2026-01-21T12:34:56.789Z",
    "data": {
        "frame_id": 12345,
        "score": 0.75,
        "vad_model": "mnad"
    }
}
```

---

## 7. 보안 요구사항

### 7.1 데이터 보안

- 얼굴 임베딩 암호화 저장 (선택)
- API 키 환경 변수 관리
- 로그 파일 접근 권한 제한

### 7.2 접근 제어

- Function Calling 함수별 권한 관리 (선택)
- 얼굴 등록 권한 제한
- 시스템 설정 변경 권한 제한

---

## 8. 검증 및 테스트 요구사항

### 8.1 단위 테스트

- 이벤트 버스 테스트
- Function Calling 테스트
- 얼굴 인식 테스트
- 패키지 감지 테스트

### 8.2 통합 테스트

- 이벤트 시스템 통합 테스트
- Function Calling + Agent 통합 테스트
- 얼굴 인식 + 이벤트 통합 테스트
- 패키지 감지 + 도난 감지 통합 테스트

### 8.3 성능 테스트

- 이벤트 처리 지연시간 측정
- Function Calling 응답 시간 측정
- 전체 시스템 처리량 측정

### 8.4 사용자 수용 테스트

- 자연어 질의 응답 정확도 측정
- 음성 인터랙션 응답 시간 측정 (선택)

---

## 9. 질문 및 검토 사항

### 9.1 기술적 질문

1. **얼굴 인식 모델 선택**
   - InsightFace vs FaceNet vs 기타?
   - 정확도 vs 속도 트레이드오프?

2. **패키지 감지 모델**
   - YOLOv11 vs YOLOv8?
   - 커스텀 모델 학습 필요 여부?

3. **음성 인터랙션 필요성**
   - 실제 사용 사례가 있는가?
   - 비용 대비 효과는?

4. **WebRTC 실시간 처리**
   - GetStream 계정 및 비용 부담?
   - 현재 RTSP 처리로 충분한가?

### 9.2 우선순위 질문

1. **Phase 1 (이벤트 시스템)부터 시작할지?**
   - ✅ 권장: 기반 구조 개선

2. **얼굴 인식과 패키지 감지 중 어느 것을 먼저?**
   - 얼굴 인식 권장: 더 범용적

3. **음성 인터랙션은 필수인가?**
   - 선택 사항으로 분류

### 9.3 검토 필요 사항

1. **기존 Agent 시스템과의 통합 방식**
   - Function Calling을 Agent Flow에 통합하는 방법

2. **데이터베이스 선택**
   - SQLite (간단) vs PostgreSQL (확장성)

3. **이벤트 저장 방식**
   - 메모리만 vs 파일 저장 vs 데이터베이스

---

## 10. 승인 및 다음 단계

### 10.1 승인 필요 사항

- [ ] 기능 요구사항 검토 및 승인
- [ ] 우선순위 확인
- [ ] 기술 선택 (얼굴 인식 모델, 패키지 감지 모델)
- [ ] 일정 확인

### 10.2 다음 단계

1. **기술 검토 회의** (필요 시)
2. **Phase 1 시작**: 이벤트 시스템 도입
3. **단계별 검증 및 피드백**

---

*이 요구사항 정의서는 프로젝트 진행에 따라 지속적으로 업데이트됩니다.*
