# AI CCTV React UI

React + Vite + Tailwind CSS + Radix UI 기반의 보안 모니터링 대시보드

## 기술 스택

- **React 18** — UI 프레임워크
- **Vite** — 빌드/개발 서버
- **Tailwind CSS v4** — 유틸리티 CSS
- **Radix UI** — 접근성 우수한 컴포넌트
- **Recharts** — 차트
- **Lucide React** — 아이콘

## 설치 및 실행

```bash
npm install
npm run dev    # http://localhost:3000
```

Vite proxy가 `/api`, `/health`, `/ws`를 `localhost:8000`(FastAPI)으로 포워딩합니다.
백엔드가 함께 실행 중이어야 정상 동작합니다.

```bash
# 백엔드 (별도 터미널)
cd ..
PIPELINE_DUMMY=true uvicorn app.api.main:app --port 8000
```

## 빌드

```bash
npm run build
```

결과물은 `dist/` 디렉토리에 생성됩니다.

## 주요 기능

| 컴포넌트 | 설명 | 백엔드 연동 |
|----------|------|-------------|
| `LoginPage` | 로그인 / 회원가입 | JWT (register/login/refresh) |
| `LiveCameraGrid` | 카메라 모니터링, start/stop | `GET/POST /api/v1/cameras/` |
| `StatsDashboard` | 통계 (카메라 상태, 이벤트 분포) | `GET /api/v1/cameras/`, `GET /api/v1/events/` |
| `AIAnalysisPanel` | VLM 분석 결과 | 로컬 표시 |
| `AIAgentPanel` | Agent 대응 계획 | 로컬 표시 |
| `SettingsPanel` | 시스템 설정 | 로컬 표시 |

## 아키텍처

```
main.tsx
  └── AuthProvider (context/AuthContext.tsx)
        └── App.tsx
              ├── [미인증] → LoginPage
              └── [인증됨] → 탭 네비게이션
                    ├── LiveCameraGrid
                    ├── AIAnalysisPanel
                    ├── AIAgentPanel
                    ├── StatsDashboard
                    └── SettingsPanel
```

## API 클라이언트

`src/lib/api.ts`에서 JWT 토큰 관리 및 자동 갱신을 처리합니다:
- `auth.login/register/me`
- `cameras.list/create/start/stop`
- `events.list/ack`
- `notifications.rules/createRule/test`
- `health()`
