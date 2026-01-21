# SHIBAL UI 통합

이 디렉토리는 SHIBAL 저장소의 AI CCTV System UI Flow_v2를 통합한 React 기반 웹 UI입니다.

## 기술 스택

- **React 18.3.1** - UI 프레임워크
- **Vite 6.3.5** - 빌드 도구
- **Radix UI** - 접근성 우수한 컴포넌트 라이브러리
- **Tailwind CSS** - 유틸리티 기반 CSS
- **Recharts** - 차트 라이브러리
- **Lucide React** - 아이콘 라이브러리

## 설치

```bash
npm install
```

## 개발 서버 실행

```bash
npm run dev
```

서버가 `http://localhost:3000`에서 실행됩니다.

## 빌드

```bash
npm run build
```

빌드 결과물은 `build/` 디렉토리에 생성됩니다.

## 주요 기능

- **실시간 모니터링**: LiveCameraGrid 컴포넌트
- **AI 분석**: AIAnalysisPanel 컴포넌트
- **AI 어시스턴트**: AIAgentPanel 컴포넌트
- **통계 대시보드**: StatsDashboard 컴포넌트
- **설정**: SettingsPanel 컴포넌트

## 백엔드 연동

현재는 Mock 데이터를 사용하고 있습니다. 실제 백엔드 API와 연동하려면:

1. `src/lib/api.ts` 파일 생성
2. API 엔드포인트 정의
3. 컴포넌트에서 Mock 데이터를 API 호출로 교체

## 참고 자료

- 통합 가이드: `../docs/shibal_integration_guide.md`
- 필수 파일 목록: `../docs/shibal_essential_files.md`
- 저장소 분석: `../docs/shibal_repository_analysis.md`

---

**원본 저장소**: https://github.com/gggg8657/SHIBAL  
**통합일**: 2026-01-20
