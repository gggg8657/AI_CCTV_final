# AI CCTV 통합 시스템

VAD + VLM + Agentic AI + Web UI를 통합한 실시간 CCTV 보안 모니터링 시스템

## 🚀 빠른 시작

### 설치

```bash
# 저장소 클론
git clone <repository-url>
cd AI_CCTV_final

# Python 의존성 설치
pip install -r requirements.txt

# UI 의존성 설치 (선택적)
cd ui
npm install
cd ..
```

### 실행

```bash
# CLI 모드
python app/run.py --mode cli --source /path/to/video.mp4

# Web UI (Streamlit)
python app/run.py --mode web --source rtsp://camera_ip/stream

# React Web UI (개발 모드)
cd ui
npm run dev
```

## 📁 프로젝트 구조

```
AI_CCTV_final/
├── app/                    # 애플리케이션 레이어
│   ├── e2e_system.py      # E2E 시스템 엔진
│   ├── cli_ui.py          # CLI 대시보드
│   ├── web_ui.py          # Streamlit Web UI
│   └── run.py             # 메인 실행 스크립트
│
├── src/                    # 핵심 모듈
│   ├── vad/               # VAD 모델 (MNAD, MULDE, MemAE, STAE, STEAD)
│   ├── vlm/               # VLM 분석기 (Qwen2.5-VL)
│   ├── agent/             # Agent 시스템 (Qwen3-8B)
│   ├── pipeline/          # 파이프라인 유틸리티
│   └── utils/             # 공통 유틸리티
│
├── ui/                     # React Web UI
│   ├── src/
│   │   ├── components/    # UI 컴포넌트
│   │   └── App.tsx        # 메인 앱
│   └── package.json
│
├── configs/                # 설정 파일
│   └── config.yaml         # 기본 설정
│
└── docs/                   # 문서
    ├── SYSTEM_ARCHITECTURE.md
    └── SYSTEM_SPECIFICATION.md
```

## 🎯 주요 기능

### ✅ 구현 완료

- **VAD 모델**: MNAD, MULDE, MemAE, STAE, STEAD
- **VLM 분석**: Qwen2.5-VL-7B 기반 상황 분석
- **Agent 시스템**: Sequential/Hierarchical/Collaborative Flow
- **E2E 파이프라인**: VAD → VLM → Agent 통합
- **UI**: CLI, Streamlit Web UI, React Web UI (SHIBAL 저장소 기반)

### 🚧 개발 예정

- 멀티 카메라 지원
- REST API 서버
- 데이터베이스 통합
- 알림 시스템
- 사용자 인증 및 권한 관리

자세한 내용은 [시스템 명세서](docs/SYSTEM_SPECIFICATION.md)를 참조하세요.

## 📊 성능

| 모델 | AUC (%) | FPS | 메모리 |
|------|---------|-----|--------|
| **MNAD** | **82.4** | **265** | 1GB |
| MULDE | 89.66 | 45 | 2GB |
| MemAE | 78.5 | 180 | 512MB |
| STAE | 75.2 | 320 | 256MB |

## 📖 문서

### 프로젝트 계획 및 명세
- [애자일 프로젝트 계획서](docs/AGILE_PROJECT_PLAN.md) - 스프린트 계획 및 백로그
- [상세 기능 명세서](docs/DETAILED_SPECIFICATION.md) - 구현 상세 설계
- [REST API 명세서](docs/API_SPECIFICATION.md) - API 엔드포인트 상세
- [구현 체크리스트](docs/IMPLEMENTATION_CHECKLIST.md) - 작업 추적

### 시스템 아키텍처
- [시스템 아키텍처](docs/SYSTEM_ARCHITECTURE.md) - 전체 구조 및 컴포넌트
- [시스템 명세서](docs/SYSTEM_SPECIFICATION.md) - 현재 상태 및 추가 기능

### 기타
- [프로젝트 구조](PROJECT_STRUCTURE.md) - 디렉토리 구조 설명
- [문서 인덱스](docs/README.md) - 모든 문서 목록

## 🔧 설정

`configs/config.yaml` 파일을 수정하여 시스템을 설정할 수 있습니다:

```yaml
video:
  source_type: file  # file, rtsp, webcam
  source_path: "/path/to/video.mp4"

vad:
  model: mnad
  threshold: 0.5

vlm:
  enabled: true
  n_frames: 4

agent:
  enabled: true
  flow: sequential  # sequential, hierarchical, collaborative
```

## 📝 라이선스

MIT License

## 👥 기여

이슈 및 PR 환영합니다!
