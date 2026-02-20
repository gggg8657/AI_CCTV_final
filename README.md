# AI CCTV 통합 보안 모니터링 시스템

VAD + VLM + Agentic AI + 실시간 파이프라인 + REST API + React UI를 통합한 CCTV 보안 시스템

## 빠른 시작

### 1. 환경 설정

```bash
git clone https://github.com/gggg8657/AI_CCTV_final.git
cd AI_CCTV_final

# Conda 환경 (agent)
conda activate agent
pip install -r requirements.txt
```

### 2. 서버 실행

```bash
# 더미 모드 (모델 없이 전체 파이프라인 동작)
PIPELINE_DUMMY=true uvicorn app.api.main:app --host 0.0.0.0 --port 8000

# 서버 모드 (실제 VAD + 더미 VLM/Agent)
PIPELINE_DUMMY_VLM=true PIPELINE_DUMMY_AGENT=true uvicorn app.api.main:app --port 8000

# 전체 실제 모델
uvicorn app.api.main:app --port 8000
```

### 3. React UI 실행

```bash
cd ui
npm install
npm run dev    # http://localhost:3000 → Vite proxy → :8000
```

### 4. Docker 실행

```bash
docker compose up --build
```

## 프로젝트 구조

```
AI_CCTV_final/
├── app/api/                   # FastAPI REST API
│   ├── main.py               # 앱 엔트리포인트 + lifespan
│   ├── dependencies.py       # JWT 인증 미들웨어
│   ├── schemas.py            # Pydantic 요청/응답 모델
│   ├── pipeline_state.py     # 글로벌 파이프라인 싱글턴
│   ├── routers/              # API 라우터
│   │   ├── auth.py           # 인증 (register/login/refresh/me)
│   │   ├── cameras.py        # 카메라 CRUD + start/stop + pipeline-status
│   │   ├── events.py         # 이벤트 조회/필터/확인
│   │   ├── stats.py          # 통계 summary
│   │   ├── notifications.py  # 알림 규칙 CRUD + 테스트
│   │   └── stream.py         # WebSocket streaming
│   └── websocket/manager.py  # WS 연결 관리
│
├── src/                       # 핵심 모듈
│   ├── auth/                  # JWT + bcrypt 인증
│   ├── database/              # SQLAlchemy 모델 + AsyncEventLogger
│   ├── pipeline/              # 멀티카메라 파이프라인
│   │   ├── resource_pool.py   # 스레드-안전 모델 공유 풀
│   │   ├── camera_pipeline.py # 카메라별 독립 처리 스레드
│   │   ├── camera_config.py   # 설정/상태 dataclass
│   │   └── multi_camera_manager.py  # 생명주기 관리
│   ├── dummy/                 # 더미 모델 (VAD/VLM/Agent/Video)
│   ├── notifications/         # 알림 엔진 (웹훅/이메일/콘솔)
│   ├── vad/                   # VAD 모델 (MNAD, MULDE, MemAE 등)
│   ├── vlm/                   # VLM 분석기 (Qwen2.5-VL)
│   └── agent/                 # Agent 시스템 (Sequential/Hierarchical/Collaborative)
│
├── ui/                        # React + Vite + Tailwind + Radix UI
│   ├── src/lib/api.ts         # JWT API 클라이언트
│   ├── src/context/           # 인증 상태 관리
│   └── src/components/        # 대시보드 컴포넌트
│
├── alembic/                   # DB 마이그레이션
├── tests/                     # pytest 테스트 (58개)
├── Dockerfile                 # 컨테이너 배포
├── docker-compose.yml
└── .env.example               # 환경변수 템플릿
```

## 주요 기능

### 구현 완료

- **VAD 모델**: MNAD, MULDE, MemAE, STAE, STEAD + 더미
- **VLM 분석**: Qwen2.5-VL 기반 상황 분석 + 더미
- **Agent 시스템**: Sequential/Hierarchical/Collaborative Flow + 더미
- **멀티카메라 파이프라인**: ResourcePool, CameraPipeline, MultiCameraManager
- **REST API**: JWT 인증, Camera/Event/Stats/Notification CRUD
- **WebSocket**: 실시간 스트리밍 (카메라/이벤트/통계)
- **알림 시스템**: 웹훅(Slack/Discord), 이메일(SMTP), 콘솔 + 15초 중복 방지
- **데이터베이스**: SQLite(dev) / PostgreSQL(prod) + Alembic 마이그레이션
- **React UI**: 로그인, 카메라 제어, 통계 대시보드 (실제 API 연동)
- **Docker**: Dockerfile + docker-compose.yml

### 파이프라인 흐름

```
Video Source → CameraPipeline (thread)
                ├── VAD Model → anomaly score
                │   └── [score >= threshold]
                │       ├── VLM Analyzer → 상황 분류/설명
                │       └── Agent Flow → 대응 계획
                ├── AsyncEventLogger → DB 저장
                ├── WebSocket → 실시간 브로드캐스트
                └── NotificationEngine → 웹훅/이메일 알림
```

## API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/v1/auth/register` | 회원가입 |
| POST | `/api/v1/auth/login` | 로그인 (JWT) |
| GET | `/api/v1/cameras/` | 카메라 목록 |
| POST | `/api/v1/cameras/{id}/start` | 파이프라인 시작 |
| GET | `/api/v1/cameras/{id}/pipeline-status` | 실시간 상태 |
| GET | `/api/v1/events/` | 이벤트 목록 (필터/페이징) |
| POST | `/api/v1/events/{id}/ack` | 이벤트 확인 |
| GET | `/api/v1/notifications/rules` | 알림 규칙 |
| WS | `/ws/events` | 실시간 이벤트 |

전체 API 문서: http://localhost:8000/docs (Swagger UI)

## 테스트

```bash
PIPELINE_DUMMY=true python -m pytest tests/ -v
```

- API smoke: 18 | Dummy pipeline: 7 | Sprint 2: 15
- API-Pipeline integration: 6 | Notifications: 12
- **총 58 tests PASSED**

## 환경변수

`.env.example` 참조. 주요 설정:

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `DATABASE_URL` | `sqlite:///./data/ai_cctv.db` | DB 연결 |
| `JWT_SECRET_KEY` | dev 값 | JWT 서명 키 |
| `PIPELINE_DUMMY` | `false` | 전체 더미 모드 |
| `PIPELINE_DUMMY_VLM` | - | VLM만 더미 |
| `PIPELINE_DUMMY_AGENT` | - | Agent만 더미 |
| `NOTIFY_WEBHOOK_URL` | - | Slack/Discord 웹훅 |

## 성능

| 모델 | AUC (%) | FPS | 메모리 |
|------|---------|-----|--------|
| **MNAD** | **82.4** | **265** | 1GB |
| MULDE | 89.66 | 45 | 2GB |
| MemAE | 78.5 | 180 | 512MB |
| STAE | 75.2 | 320 | 256MB |

## 라이선스

MIT License
