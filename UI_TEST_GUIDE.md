# UI 테스트 가이드 (모델 없이)

## 준비 사항

1. **의존성 설치**
```bash
pip install streamlit fastapi uvicorn
```

2. **Config 설정**
- Agent 비활성화: `config['agent']['enabled'] = False`
- 이미 `configs/config.yaml.backup`으로 백업됨

## 실행 방법

### 방법 1: 자동 스크립트 사용
```bash
./run_ui.sh
```

### 방법 2: 수동 실행

#### Streamlit Web UI
```bash
streamlit run app/web_ui.py
```
- URL: http://localhost:8501
- 특징: 간단한 웹 인터페이스

#### FastAPI Backend
```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```
- URL: http://localhost:8000
- API Docs: http://localhost:8000/docs

#### React Frontend
```bash
cd ui
npm install  # 처음 한 번만
npm run dev
```
- URL: http://localhost:5173
- FastAPI와 함께 실행 필요

#### CLI UI
```bash
python app/cli_ui.py
```

## 확인 가능한 기능

✅ **모델 없이 확인 가능:**
- UI 구조 및 레이아웃
- API 엔드포인트
- 이벤트 버스 동작
- Function Registry
- 기본 통계 표시

⚠️ **모델 필요:**
- Agent 기능
- VAD/VLM 분석
- 실제 비디오 처리

## Config 원복

```bash
cp configs/config.yaml.backup configs/config.yaml
```
