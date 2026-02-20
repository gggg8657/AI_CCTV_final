# 프로젝트 구조

## 디렉토리 구조

```
AI_CCTV_final/
│
├── app/                           # 애플리케이션 레이어
│   ├── api/                       # FastAPI REST API
│   │   ├── main.py               # 앱 엔트리 + lifespan (DB init, MCM startup)
│   │   ├── dependencies.py       # JWT 인증 의존성 (get_current_user)
│   │   ├── schemas.py            # Pydantic 요청/응답 스키마
│   │   ├── pipeline_state.py     # 글로벌 MultiCameraManager 싱글턴
│   │   ├── routers/
│   │   │   ├── auth.py           # 회원가입, 로그인, 토큰 갱신, 내 정보
│   │   │   ├── cameras.py        # 카메라 CRUD + start/stop + pipeline-status
│   │   │   ├── events.py         # 이벤트 조회/필터링/확인
│   │   │   ├── stats.py          # 통계 summary
│   │   │   ├── notifications.py  # 알림 규칙 CRUD + 테스트 발송
│   │   │   └── stream.py         # WebSocket 스트리밍
│   │   └── websocket/
│   │       └── manager.py        # WebSocket 연결 매니저
│   │
│   ├── e2e_system.py             # 레거시 E2E 엔진 (단일 카메라)
│   ├── cli_ui.py                 # CLI 대시보드 (Rich)
│   ├── web_ui.py                 # Streamlit Web UI (레거시)
│   ├── run.py                    # 레거시 실행 스크립트
│   └── ui_components/            # Streamlit 더미 모듈
│
├── src/                           # 핵심 모듈
│   ├── auth/                      # 인증
│   │   ├── password.py           # bcrypt 해싱
│   │   └── jwt.py                # JWT 생성/검증
│   │
│   ├── database/                  # 데이터베이스
│   │   ├── db.py                 # SQLAlchemy 엔진/세션
│   │   ├── models.py            # ORM 모델 (User, Camera, Event 등)
│   │   └── event_logger.py      # AsyncEventLogger (배치 DB 저장)
│   │
│   ├── pipeline/                  # 멀티카메라 파이프라인
│   │   ├── resource_pool.py      # 스레드-안전 모델 공유 풀
│   │   ├── camera_pipeline.py    # 카메라별 독립 처리 스레드
│   │   ├── camera_config.py      # CameraConfig/CameraStatus dataclass
│   │   ├── multi_camera_manager.py  # 생명주기 관리
│   │   ├── engine.py             # 파이프라인 엔진
│   │   └── clip_saver.py         # 이상 구간 클립 저장
│   │
│   ├── dummy/                     # 더미 모델 (모델 파일 없이 동작)
│   │   ├── vad.py                # DummyVADModel
│   │   ├── vlm.py                # DummyVLMAnalyzer
│   │   ├── agent.py              # DummyAgentFlow
│   │   └── video.py              # DummyVideoSource
│   │
│   ├── notifications/             # 알림 시스템
│   │   ├── base.py               # NotificationChannel 인터페이스
│   │   ├── console.py            # ConsoleChannel
│   │   ├── webhook.py            # WebhookChannel (Slack/Discord)
│   │   ├── email.py              # EmailChannel (SMTP)
│   │   └── engine.py             # NotificationEngine (규칙, cooldown)
│   │
│   ├── vad/                       # VAD 모델
│   │   ├── base.py               # VADModel 인터페이스
│   │   ├── mnad.py               # MNAD
│   │   ├── stae.py               # STAE
│   │   ├── stead.py              # STEAD
│   │   ├── memae.py              # MemAE
│   │   ├── attribute_based.py    # Attribute-based VAD
│   │   ├── attribute_based_aivad.py  # AiVAD
│   │   ├── adaptive_threshold.py # 적응형 임계값
│   │   ├── alpha_pose.py         # AlphaPose
│   │   └── flow_net2.py          # FlowNet2 옵티컬 플로우
│   │
│   ├── vlm/                       # VLM 분석기
│   │   ├── analyzer.py           # VLMAnalyzer (Qwen2.5-VL)
│   │   ├── prompts.py            # 프롬프트 템플릿
│   │   ├── analyzer_lightweight.py
│   │   ├── analyzer_quantized.py
│   │   └── adaptive_analyzer.py
│   │
│   ├── agent/                     # Agent 시스템
│   │   ├── base.py               # Agent 기본 클래스 + LLMManager
│   │   ├── actions.py            # 액션 정의
│   │   ├── function_calling.py   # Function Calling
│   │   ├── llm_wrapper.py        # LLM 래퍼 (API/Local)
│   │   ├── api_llm_manager.py    # API LLM 매니저
│   │   └── flows/
│   │       ├── sequential.py     # Sequential Flow
│   │       ├── hierarchical.py   # Hierarchical Flow
│   │       ├── collaborative.py  # Collaborative Flow
│   │       └── function_calling_support.py
│   │
│   ├── package_detection/         # Phase 3: 소포 탐지
│   │   ├── base.py
│   │   ├── detector.py
│   │   ├── tracker.py
│   │   └── theft_detector.py
│   │
│   └── utils/                     # 공통 유틸리티
│       ├── video.py              # 비디오 I/O
│       ├── logging.py            # 로깅
│       ├── config.py             # config 로더
│       ├── event_bus.py          # 이벤트 버스
│       ├── events.py             # 이벤트 정의
│       └── event_handlers.py     # 이벤트 핸들러
│
├── ui/                            # React Web UI
│   ├── src/
│   │   ├── lib/api.ts            # JWT API 클라이언트 (자동 토큰 갱신)
│   │   ├── context/
│   │   │   └── AuthContext.tsx    # 인증 컨텍스트
│   │   ├── components/
│   │   │   ├── LoginPage.tsx     # 로그인/회원가입
│   │   │   ├── LiveCameraGrid.tsx  # 카메라 모니터링 (실시간 API)
│   │   │   ├── StatsDashboard.tsx  # 통계 대시보드 (실시간 API)
│   │   │   ├── AIAnalysisPanel.tsx # VLM 분석 패널
│   │   │   ├── AIAgentPanel.tsx    # Agent 패널
│   │   │   ├── AIAgentChat.tsx     # Agent 채팅
│   │   │   ├── SettingsPanel.tsx   # 설정 패널
│   │   │   └── ui/               # Radix UI 공통 컴포넌트
│   │   ├── App.tsx               # 메인 앱 (인증 + 라우팅)
│   │   └── main.tsx              # 엔트리 (AuthProvider)
│   ├── vite.config.ts            # Vite + API proxy (:8000)
│   ├── package.json
│   └── tsconfig.json
│
├── alembic/                       # DB 마이그레이션
│   ├── env.py
│   └── versions/
│
├── tests/                         # pytest (58 tests)
│   ├── test_api_smoke.py         # API 엔드포인트 smoke
│   ├── test_dummy_pipeline.py    # 더미 파이프라인
│   ├── test_sprint2_pipeline.py  # Sprint 2 통합
│   ├── test_api_pipeline_integration.py  # API-Pipeline 통합
│   ├── test_notifications.py     # 알림 시스템
│   └── ...
│
├── configs/config.yaml            # 기본 설정
├── Dockerfile                     # 컨테이너 이미지
├── docker-compose.yml             # 컨테이너 오케스트레이션
├── requirements.txt               # Python 의존성
├── requirements-docker.txt        # Docker 경량 의존성
├── .env.example                   # 환경변수 템플릿
├── alembic.ini                    # Alembic 설정
├── ENVIRONMENT.md                 # Conda 환경 가이드
├── README.md                      # 프로젝트 README
└── PROJECT_STRUCTURE.md           # 이 파일
```

## 데이터 흐름

```
HTTP Request (React UI)
    ↓
FastAPI Router (app/api/routers/)
    ↓ POST /cameras/{id}/start
MultiCameraManager
    ↓
CameraPipeline (독립 스레드)
    ↓
ResourcePool → VAD/VLM/Agent (or Dummy)
    ↓
┌──────────────────────────────┐
│  VAD → anomaly score         │
│  ↓ [threshold 초과]          │
│  VLM → 상황 분류/설명        │
│  Agent → 대응 계획           │
└──────────────────────────────┘
    ↓
AsyncEventLogger → DB (SQLAlchemy)
    ↓
WebSocket broadcast → React UI
    ↓
NotificationEngine → 웹훅/이메일/콘솔
```

## 의존성 관리

- **Python**: `requirements.txt` (개발), `requirements-docker.txt` (배포)
- **Node.js**: `ui/package.json`
- **DB**: `alembic/` (마이그레이션)
- **환경 변수**: `.env.example`
