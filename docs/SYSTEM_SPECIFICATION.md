# AI CCTV 통합 시스템 명세서

**버전**: 2.0  
**작성일**: 2025-01-20  
**최종 업데이트**: 2025-01-20

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [현재 구현 상태](#2-현재-구현-상태)
3. [추가 기능 명세](#3-추가-기능-명세)
4. [기술 스택](#4-기술-스택)
5. [아키텍처](#5-아키텍처)
6. [API 명세](#6-api-명세)
7. [데이터베이스 스키마](#7-데이터베이스-스키마)
8. [배포 및 운영](#8-배포-및-운영)

---

## 1. 시스템 개요

### 1.1 목적

VAD (Video Anomaly Detection), VLM (Vision Language Model), Agentic AI, 그리고 Web UI를 통합한 실시간 CCTV 보안 모니터링 시스템입니다. 이상 상황을 자동으로 탐지하고, AI 기반 분석을 통해 상황을 이해하며, 자동 대응 계획을 수립하는 지능형 보안 시스템입니다.

### 1.2 핵심 가치

- **실시간 처리**: 최대 265 FPS 이상 탐지 (MNAD 모델 기준)
- **지능형 분석**: VLM을 통한 상황 이해 및 분류
- **자동 대응**: Agent 시스템을 통한 대응 계획 수립
- **통합 UI**: Web UI와 CLI를 통한 실시간 모니터링

---

## 2. 현재 구현 상태

### 2.1 구현 완료된 기능

#### 2.1.1 VAD (Video Anomaly Detection)

| 모델 | 상태 | 성능 | 용도 |
|------|------|------|------|
| MNAD | ✅ 완료 | AUC: 82.4%, FPS: 265 | 실시간 처리 (권장) |
| MULDE | ✅ 완료 | AUC: 89.66%, FPS: 45 | 정확도 우선 |
| MemAE | ✅ 완료 | AUC: 78.5%, FPS: 180 | 경량화 |
| STAE | ✅ 완료 | AUC: 75.2%, FPS: 320 | 초경량 |
| STEAD | ✅ 완료 | AUC: 69.47%, FPS: 118 | 균형형 |

**구현 위치**: `src/vad/`

**주요 기능**:
- 실시간 프레임 단위 이상 점수 계산
- 임계값 기반 이상 감지
- 프레임 버퍼링 (시퀀스 분석용)

#### 2.1.2 VLM (Vision Language Model)

**모델**: Qwen2.5-VL-7B-Instruct

**구현 위치**: `src/vlm/`

**주요 기능**:
- 단일 프레임 분석
- 멀티프레임 그리드 분석 (최대 8프레임)
- 이상 상황 분류 (Fighting, Arson, Explosion, Road_Accident, Suspicious_Object, Falling 등)
- 상황 설명 생성
- 대응 권장사항 제시

**성능**:
- 분석 시간: ~5초 (8프레임 기준)
- 정확도: 약 85% (실험 데이터 기준)

#### 2.1.3 Agentic AI

**모델**: Qwen3-8B (llama.cpp)

**구현 위치**: `src/agent/`

**지원 Flow**:
1. **Sequential Flow** (5.2초)
   - VideoAnalysis → Planner → Actor
   - 가장 빠른 처리 속도

2. **Hierarchical Flow** (8.5초)
   - VideoAnalysis → Supervisor → Planner → Supervisor → Actor
   - 최고 품질의 대응 계획

3. **Collaborative Flow** (6.8초)
   - VideoAnalysis → [Planner1 + Planner2] → Aggregator → Actor
   - 균형잡힌 처리 속도와 품질

**주요 기능**:
- 이상 상황 분석
- 대응 계획 수립
- 액션 우선순위 결정
- 실행 가능한 액션 생성

#### 2.1.4 E2E 시스템 엔진

**구현 위치**: `app/e2e_system.py`

**주요 기능**:
- 비디오 소스 관리 (파일, RTSP, 웹캠)
- VAD → VLM → Agent 파이프라인 통합
- 이벤트 로깅 및 클립 저장
- 실시간 통계 수집
- 콜백 기반 UI 연동

**데이터 흐름**:
```
Video Source → Frame Buffer → VAD Model → Anomaly Detection
                                                    ↓
                                            Clip Saver (3초 저장)
                                                    ↓
                                            VLM Analyzer
                                                    ↓
                                            Agent System
                                                    ↓
                                            Event Logger
```

#### 2.1.5 UI 시스템

**Web UI** (Streamlit):
- 구현 위치: `app/web_ui.py`
- 실시간 영상 표시
- 이상 점수 차트
- 이벤트 목록
- 설정 패널

**CLI UI** (Rich):
- 구현 위치: `app/cli_ui.py`
- 터미널 대시보드
- ASCII 히스토그램
- 로그 스트림
- 통계 표시

**React Web UI**:
- 구현 위치: `ui/`
- 원본: SHIBAL 저장소 (https://github.com/gggg8657/SHIBAL)
- 버전: AI CCTV System UI Flow_v2
- 통합일: 2026-01-20
- 주요 기능:
  - 실시간 모니터링 그리드 (LiveCameraGrid)
  - AI 분석 패널 (AIAnalysisPanel)
  - AI 어시스턴트 채팅 (AIAgentPanel)
  - 통계 대시보드 (StatsDashboard)
  - 설정 패널 (SettingsPanel)
- 기술 스택: React 18.3, Vite 6.3, Radix UI, Tailwind CSS, Recharts

---

## 3. 추가 기능 명세

### 3.1 우선순위 높음 (P0)

#### 3.1.1 멀티 카메라 지원

**목적**: 여러 CCTV 카메라를 동시에 모니터링

**요구사항**:
- 동시에 최대 16개 카메라 지원
- 각 카메라별 독립적인 VAD/VLM/Agent 파이프라인
- 통합 대시보드에서 모든 카메라 상태 표시
- 카메라별 설정 (모델, 임계값 등) 개별 관리

**구현 계획**:
- `src/pipeline/multi_camera.py` 확장
- 카메라별 스레드/프로세스 관리
- 리소스 풀링 (GPU 메모리 관리)
- 이벤트 집계 및 우선순위 처리

**예상 작업량**: 2주

#### 3.1.2 REST API 서버

**목적**: 외부 시스템과의 통합 및 모바일 앱 지원

**요구사항**:
- FastAPI 기반 RESTful API
- WebSocket을 통한 실시간 스트리밍
- 인증 및 권한 관리 (JWT)
- API 문서 자동 생성 (Swagger/OpenAPI)

**주요 엔드포인트**:
```
GET  /api/v1/cameras              # 카메라 목록
GET  /api/v1/cameras/{id}/status  # 카메라 상태
GET  /api/v1/events               # 이벤트 목록
POST /api/v1/events/{id}/ack      # 이벤트 확인
GET  /api/v1/stats                # 시스템 통계
WS   /ws/stream/{camera_id}       # 실시간 스트림
```

**구현 계획**:
- `app/api/` 디렉토리 생성
- FastAPI 앱 구조 설계
- WebSocket 스트리밍 구현
- 인증 미들웨어 추가

**예상 작업량**: 1주

#### 3.1.3 데이터베이스 통합

**목적**: 이벤트 영구 저장 및 분석

**요구사항**:
- PostgreSQL 또는 SQLite 사용
- 이벤트 메타데이터 저장
- 클립 경로 및 VLM/Agent 결과 저장
- 시간 기반 쿼리 및 통계 집계

**스키마 설계**:
```sql
-- 카메라 테이블
CREATE TABLE cameras (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    source_type VARCHAR(50),
    source_path TEXT,
    location VARCHAR(255),
    status VARCHAR(50),
    created_at TIMESTAMP
);

-- 이벤트 테이블
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    camera_id INTEGER REFERENCES cameras(id),
    timestamp TIMESTAMP,
    frame_number INTEGER,
    vad_score FLOAT,
    vlm_type VARCHAR(100),
    vlm_description TEXT,
    vlm_confidence FLOAT,
    agent_actions JSONB,
    clip_path TEXT,
    acknowledged BOOLEAN DEFAULT FALSE
);

-- 통계 테이블
CREATE TABLE statistics (
    id SERIAL PRIMARY KEY,
    camera_id INTEGER REFERENCES cameras(id),
    date DATE,
    total_frames INTEGER,
    anomaly_count INTEGER,
    avg_vad_time FLOAT,
    avg_vlm_time FLOAT,
    avg_agent_time FLOAT
);
```

**구현 계획**:
- SQLAlchemy ORM 사용
- 마이그레이션 시스템 (Alembic)
- 이벤트 저장 자동화
- 통계 집계 배치 작업

**예상 작업량**: 1주

#### 3.1.4 알림 시스템

**목적**: 이상 상황 발생 시 즉시 알림

**요구사항**:
- 이메일 알림
- SMS 알림 (선택적)
- 웹훅 알림
- Slack/Discord 연동
- 알림 규칙 설정 (중요도별)

**구현 계획**:
- `src/notifications/` 디렉토리 생성
- 알림 채널 추상화
- 템플릿 기반 메시지 생성
- 알림 규칙 엔진

**예상 작업량**: 1주

### 3.2 우선순위 중간 (P1)

#### 3.2.1 사용자 인증 및 권한 관리

**목적**: 다중 사용자 지원 및 권한 제어

**요구사항**:
- 사용자 등록/로그인
- 역할 기반 접근 제어 (RBAC)
- 카메라별 접근 권한
- 감사 로그

**구현 계획**:
- JWT 기반 인증
- 사용자/역할/권한 테이블
- 미들웨어를 통한 권한 검사

**예상 작업량**: 1주

#### 3.2.2 실시간 대시보드 개선

**목적**: 더 풍부한 시각화 및 분석 기능

**요구사항**:
- 이상 점수 히트맵
- 시간대별 이상 발생 통계
- 카메라별 성능 비교
- 이벤트 타임라인 뷰

**구현 계획**:
- React 차트 라이브러리 활용
- WebSocket을 통한 실시간 업데이트
- 대시보드 위젯 시스템

**예상 작업량**: 1주

#### 3.2.3 모델 성능 모니터링

**목적**: 모델 성능 추적 및 최적화

**요구사항**:
- 추론 시간 추적
- 메모리 사용량 모니터링
- 정확도 메트릭 수집
- 성능 알림 (지연 발생 시)

**구현 계획**:
- Prometheus 메트릭 수집
- Grafana 대시보드
- 성능 로깅 시스템

**예상 작업량**: 1주

#### 3.2.4 클립 관리 시스템

**목적**: 저장된 클립 효율적 관리

**요구사항**:
- 클립 검색 및 필터링
- 클립 미리보기
- 클립 다운로드
- 자동 삭제 정책 (오래된 클립)

**구현 계획**:
- 클립 메타데이터 인덱싱
- 썸네일 생성
- 스토리지 관리 시스템

**예상 작업량**: 3일

### 3.3 우선순위 낮음 (P2)

#### 3.3.1 모델 파인튜닝 도구

**목적**: 도메인 특화 모델 학습

**요구사항**:
- 학습 데이터 수집 인터페이스
- 파인튜닝 파이프라인
- 모델 버전 관리
- A/B 테스트 지원

**예상 작업량**: 2주

#### 3.3.2 모바일 앱

**목적**: 모바일에서 모니터링 및 알림 수신

**요구사항**:
- iOS/Android 앱
- 푸시 알림
- 실시간 스트림 뷰어
- 이벤트 확인 기능

**예상 작업량**: 3주

#### 3.3.3 분산 처리

**목적**: 대규모 카메라 네트워크 지원

**요구사항**:
- 여러 서버에 카메라 분산 배치
- 중앙 집중식 관리
- 로드 밸런싱

**예상 작업량**: 4주

---

## 4. 기술 스택

### 4.1 백엔드

| 기술 | 버전 | 용도 |
|------|------|------|
| Python | 3.10+ | 메인 언어 |
| PyTorch | 2.0+ | 딥러닝 프레임워크 |
| FastAPI | 0.100+ | REST API (추가 예정) |
| SQLAlchemy | 2.0+ | ORM (추가 예정) |
| OpenCV | 4.8+ | 비디오 처리 |
| llama.cpp | 최신 | LLM 추론 |

### 4.2 프론트엔드

| 기술 | 버전 | 용도 |
|------|------|------|
| React | 18.3+ | UI 프레임워크 |
| Vite | 6.3+ | 빌드 도구 |
| TypeScript | 5.0+ | 타입 안정성 |
| Tailwind CSS | 3.0+ | 스타일링 |
| Recharts | 2.15+ | 차트 |
| Streamlit | 1.28+ | 프로토타입 UI |

### 4.3 인프라

| 기술 | 용도 |
|------|------|
| Docker | 컨테이너화 |
| PostgreSQL | 데이터베이스 (추가 예정) |
| Redis | 캐싱 (선택적) |
| Nginx | 리버스 프록시 (선택적) |

---

## 5. 아키텍처

### 5.1 전체 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  React Web UI│  │  Streamlit UI│  │  CLI UI      │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            │                                  │
└────────────────────────────┼──────────────────────────────────┘
                             │
┌────────────────────────────┼──────────────────────────────────┐
│                    API LAYER (추가 예정)                      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  FastAPI Server                                        │  │
│  │  - REST API                                             │  │
│  │  - WebSocket                                            │  │
│  │  - Authentication                                        │  │
│  └────────────────────┬───────────────────────────────────┘  │
└────────────────────────┼──────────────────────────────────────┘
                          │
┌─────────────────────────┼──────────────────────────────────────┐
│                    APPLICATION LAYER                            │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  E2E System Engine (app/e2e_system.py)                   │ │
│  │  - Video Source Management                                │ │
│  │  - Pipeline Orchestration                                 │ │
│  │  - Event Management                                       │ │
│  └────────────────────┬─────────────────────────────────────┘ │
│                        │                                         │
│  ┌────────────────────┼─────────────────────────────────────┐ │
│  │  Multi-Camera Manager (추가 예정)                        │ │
│  └────────────────────┼─────────────────────────────────────┘ │
└────────────────────────┼────────────────────────────────────────┘
                          │
┌─────────────────────────┼────────────────────────────────────────┐
│                    PROCESSING LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │  VAD Models  │  │  VLM Analyzer│  │ Agent System│            │
│  │  (src/vad/)  │  │  (src/vlm/)  │  │(src/agent/) │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
└──────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────┼────────────────────────────────────────┐
│                    DATA LAYER                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  PostgreSQL  │  │  File Storage│  │  Redis Cache │          │
│  │  (추가 예정)  │  │  (Clips)     │  │  (선택적)    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 데이터 흐름

```
[Video Source]
    │
    ▼
[Frame Buffer] (16 frames)
    │
    ▼
[VAD Model] → Anomaly Score
    │
    ├─ Score < Threshold → Continue
    │
    └─ Score >= Threshold → Anomaly Detected!
                            │
                            ├─ [Clip Saver] → Save 3s clip
                            │
                            ├─ [VLM Analyzer] → Situation Analysis
                            │   │
                            │   └─ Type, Description, Confidence
                            │
                            ├─ [Agent System] → Response Plan
                            │   │
                            │   └─ Actions, Priority
                            │
                            ├─ [Event Logger] → Save to DB
                            │
                            └─ [Notification] → Alert Users
```

---

## 6. API 명세

### 6.1 REST API (추가 예정)

#### 6.1.1 카메라 관리

```http
GET /api/v1/cameras
Response: {
  "cameras": [
    {
      "id": 1,
      "name": "Camera 1",
      "source_type": "rtsp",
      "source_path": "rtsp://...",
      "status": "active",
      "location": "Building A - Floor 1"
    }
  ]
}

GET /api/v1/cameras/{id}
POST /api/v1/cameras
PUT /api/v1/cameras/{id}
DELETE /api/v1/cameras/{id}
```

#### 6.1.2 이벤트 조회

```http
GET /api/v1/events?camera_id=1&start_date=2025-01-01&limit=100
Response: {
  "events": [
    {
      "id": 123,
      "camera_id": 1,
      "timestamp": "2025-01-20T10:30:00Z",
      "vad_score": 0.85,
      "vlm_type": "Fighting",
      "vlm_description": "Two people engaged in physical altercation",
      "agent_actions": [
        {"action": "alert_security", "priority": "high"}
      ],
      "clip_path": "/clips/clip_123.mp4"
    }
  ],
  "total": 150,
  "page": 1
}

POST /api/v1/events/{id}/ack
```

#### 6.1.3 통계 조회

```http
GET /api/v1/stats?camera_id=1&date=2025-01-20
Response: {
  "camera_id": 1,
  "date": "2025-01-20",
  "total_frames": 864000,
  "anomaly_count": 12,
  "avg_vad_time_ms": 3.77,
  "avg_vlm_time_ms": 5000,
  "avg_agent_time_ms": 200
}
```

### 6.2 WebSocket API

```javascript
// 연결
const ws = new WebSocket('ws://localhost:8000/ws/stream/1');

// 프레임 수신
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // { type: 'frame', frame: base64_image, score: 0.5 }
  // { type: 'event', event: {...} }
  // { type: 'stats', stats: {...} }
};
```

---

## 7. 데이터베이스 스키마

### 7.1 테이블 구조 (추가 예정)

```sql
-- 카메라
CREATE TABLE cameras (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    source_type VARCHAR(50) NOT NULL,
    source_path TEXT NOT NULL,
    location VARCHAR(255),
    vad_model VARCHAR(50),
    vad_threshold FLOAT DEFAULT 0.5,
    status VARCHAR(50) DEFAULT 'inactive',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 이벤트
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    camera_id INTEGER REFERENCES cameras(id),
    timestamp TIMESTAMP NOT NULL,
    frame_number INTEGER,
    vad_score FLOAT NOT NULL,
    threshold FLOAT,
    vlm_type VARCHAR(100),
    vlm_description TEXT,
    vlm_confidence FLOAT,
    agent_actions JSONB,
    agent_response_time FLOAT,
    clip_path TEXT,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by INTEGER,
    acknowledged_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 통계 (일별 집계)
CREATE TABLE daily_statistics (
    id SERIAL PRIMARY KEY,
    camera_id INTEGER REFERENCES cameras(id),
    date DATE NOT NULL,
    total_frames INTEGER DEFAULT 0,
    anomaly_count INTEGER DEFAULT 0,
    avg_vad_time FLOAT,
    avg_vlm_time FLOAT,
    avg_agent_time FLOAT,
    max_vad_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(camera_id, date)
);

-- 사용자 (추가 예정)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'viewer',
    created_at TIMESTAMP DEFAULT NOW()
);

-- 알림 규칙 (추가 예정)
CREATE TABLE notification_rules (
    id SERIAL PRIMARY KEY,
    camera_id INTEGER REFERENCES cameras(id),
    vlm_type VARCHAR(100),
    min_score FLOAT,
    channels JSONB,  -- ['email', 'sms', 'webhook']
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 8. 배포 및 운영

### 8.1 Docker 구성 (추가 예정)

```dockerfile
# Dockerfile 예시
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app/run.py", "--mode", "api"]
```

### 8.2 환경 변수

```bash
# 데이터베이스
DATABASE_URL=postgresql://user:pass@localhost/dbname

# 인증
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256

# 모델 경로
VLM_MODEL_PATH=/models/Qwen2.5-VL-7B-Instruct.gguf
AGENT_MODEL_PATH=/models/Qwen3-8B.gguf

# GPU
CUDA_VISIBLE_DEVICES=0

# 알림
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-password
```

### 8.3 모니터링 (추가 예정)

- **로그**: 구조화된 JSON 로그
- **메트릭**: Prometheus + Grafana
- **알림**: 이상 상황 알림 + 시스템 상태 알림

---

## 9. 개발 로드맵

### Phase 1 (1-2주): 핵심 기능 강화
- [ ] 멀티 카메라 지원
- [ ] REST API 서버
- [ ] 데이터베이스 통합

### Phase 2 (2-3주): 운영 기능
- [ ] 알림 시스템
- [ ] 사용자 인증
- [ ] 대시보드 개선

### Phase 3 (3-4주): 고급 기능
- [ ] 모델 성능 모니터링
- [ ] 클립 관리 시스템
- [ ] 모바일 앱 (선택적)

---

## 10. 참고 자료

- [시스템 아키텍처 문서](./SYSTEM_ARCHITECTURE.md)
- [README](../README.md)
- [VAD 모델 가이드](../src/vad/README.md)
- [VLM 사용 가이드](../src/vlm/README.md)
- [Agent 시스템 가이드](../src/agent/README.md)

---

**문서 버전 관리**: 이 문서는 시스템 개발과 함께 지속적으로 업데이트됩니다.
