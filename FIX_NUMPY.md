# NumPy 버전 호환성 문제 해결

## 문제
PyTorch가 NumPy 1.x로 컴파일되어 있어서 NumPy 2.0과 호환되지 않습니다.

## 해결 방법

### 방법 1: NumPy 다운그레이드 (권장)
```bash
pip install "numpy<2"
```

### 방법 2: requirements.txt 사용
```bash
pip install -r requirements.txt
```
(requirements.txt에 `numpy>=1.24.0,<2.0.0` 제약 추가됨)

## 확인
```bash
python3 -c "import numpy; print(numpy.__version__)"
# 예상 출력: 1.26.x 또는 1.25.x
```

## Streamlit 실행
올바른 명령어:
```bash
streamlit run app/web_ui.py
```

⚠️ `python app/web_ui.py`로 직접 실행하면 Streamlit 경고가 발생합니다.
