# 프로젝트 구조

## 디렉토리 구조

```
AI_CCTV_final/
│
├── app/                          # 애플리케이션 레이어
│   ├── __init__.py
│   ├── e2e_system.py            # E2E 시스템 엔진 (핵심)
│   ├── cli_ui.py                # CLI 대시보드 (Rich)
│   ├── web_ui.py                # Streamlit Web UI
│   └── run.py                   # 메인 실행 스크립트
│
├── src/                          # 핵심 모듈
│   ├── __init__.py
│   │
│   ├── vad/                     # VAD 모델
│   │   ├── __init__.py
│   │   ├── base.py              # VADModel 인터페이스
│   │   ├── mnad.py              # MNAD 모델
│   │   ├── mulde.py             # MULDE 모델
│   │   ├── memae.py             # MemAE 모델
│   │   ├── stae.py              # STAE 모델
│   │   └── stead.py             # STEAD 모델
│   │
│   ├── vlm/                     # VLM 분석기
│   │   ├── __init__.py
│   │   ├── analyzer.py          # VLMAnalyzer
│   │   ├── prompts.py           # 프롬프트 정의
│   │   ├── analyzer_lightweight.py
│   │   └── analyzer_quantized.py
│   │
│   ├── agent/                   # Agent 시스템
│   │   ├── __init__.py
│   │   ├── base.py              # Agent 기본 클래스
│   │   ├── actions.py           # 액션 정의
│   │   └── flows/               # Agent Flow
│   │       ├── __init__.py
│   │       ├── sequential.py    # Sequential Flow
│   │       ├── hierarchical.py  # Hierarchical Flow
│   │       └── collaborative.py  # Collaborative Flow
│   │
│   ├── pipeline/                # 파이프라인 유틸리티
│   │   ├── __init__.py
│   │   ├── engine.py            # 파이프라인 엔진
│   │   ├── clip_saver.py        # 클립 저장
│   │   └── multi_camera.py      # 멀티 카메라 (추가 예정)
│   │
│   └── utils/                   # 공통 유틸리티
│       ├── __init__.py
│       ├── video.py             # 비디오 처리
│       └── logging.py            # 로깅 유틸리티
│
├── ui/                           # React Web UI
│   ├── src/
│   │   ├── App.tsx              # 메인 앱
│   │   ├── components/          # UI 컴포넌트
│   │   │   ├── LiveCameraGrid.tsx
│   │   │   ├── AIAnalysisPanel.tsx
│   │   │   ├── AIAgentPanel.tsx
│   │   │   ├── StatsDashboard.tsx
│   │   │   └── SettingsPanel.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
│
├── configs/                      # 설정 파일
│   └── config.yaml               # 기본 설정
│
├── docs/                         # 문서
│   ├── SYSTEM_ARCHITECTURE.md    # 시스템 아키텍처
│   └── SYSTEM_SPECIFICATION.md   # 시스템 명세서
│
├── requirements.txt              # Python 의존성
├── .gitignore                    # Git 무시 파일
├── README.md                     # 프로젝트 README
└── PROJECT_STRUCTURE.md          # 이 파일
```

## 주요 파일 설명

### E2E 시스템 엔진 (`app/e2e_system.py`)

시스템의 핵심 엔진으로, 다음 컴포넌트들을 통합합니다:

- `VideoSource`: 비디오 입력 관리
- `VADWrapper`: VAD 모델 래퍼
- `VLMWrapper`: VLM 분석기 래퍼
- `AgentWrapper`: Agent 시스템 래퍼
- `EventLogger`: 이벤트 로깅
- `ClipSaver`: 클립 저장
- `E2ESystem`: 메인 시스템 클래스

### VAD 모델 (`src/vad/`)

다양한 VAD 모델 구현:

- `base.py`: `VADModel` 인터페이스 정의
- 각 모델 파일: 모델별 구현
- `__init__.py`: `create_model()` 팩토리 함수

### VLM 분석기 (`src/vlm/`)

Vision Language Model 기반 분석:

- `analyzer.py`: `VLMAnalyzer` 클래스
- `prompts.py`: 프롬프트 템플릿
- Qwen2.5-VL-7B 모델 사용

### Agent 시스템 (`src/agent/`)

LLM 기반 자동 대응 시스템:

- `base.py`: Agent 기본 클래스들
- `flows/`: 다양한 Flow 구현
- Qwen3-8B 모델 사용 (llama.cpp)

### UI 시스템

1. **CLI UI** (`app/cli_ui.py`): Rich 기반 터미널 대시보드
2. **Streamlit UI** (`app/web_ui.py`): 웹 대시보드
3. **React UI** (`ui/`): 고급 웹 인터페이스

## 데이터 흐름

```
Video Source
    ↓
E2ESystem (app/e2e_system.py)
    ↓
    ├─→ VAD Model (src/vad/)
    │       ↓
    │   Anomaly Score
    │       ↓
    │   [If Score >= Threshold]
    │       ↓
    │   Clip Saver
    │       ↓
    │   VLM Analyzer (src/vlm/)
    │       ↓
    │   Situation Analysis
    │       ↓
    │   Agent System (src/agent/)
    │       ↓
    │   Response Plan
    │       ↓
    │   Event Logger
    │       ↓
    │   UI Update
```

## 확장 계획

### 추가 예정 디렉토리

```
app/
└── api/                         # REST API (추가 예정)
    ├── __init__.py
    ├── main.py                  # FastAPI 앱
    ├── routes/                   # API 라우트
    │   ├── cameras.py
    │   ├── events.py
    │   └── stats.py
    └── auth.py                  # 인증

src/
└── notifications/               # 알림 시스템 (추가 예정)
    ├── __init__.py
    ├── email.py
    ├── sms.py
    └── webhook.py

src/
└── database/                    # 데이터베이스 (추가 예정)
    ├── __init__.py
    ├── models.py                # SQLAlchemy 모델
    └── migrations/              # Alembic 마이그레이션
```

## 의존성 관리

- **Python**: `requirements.txt`
- **Node.js**: `ui/package.json`

## 설정 관리

- **기본 설정**: `configs/config.yaml`
- **환경 변수**: `.env` (추가 예정)
