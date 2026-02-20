# Environment — AI_CCTV_final

- **ENV_NAME**: agent
- **PYTHON**: 3.9.23
- **CUDA**: cpu (MPS on macOS, CUDA on Linux)

## CREATE

```bash
conda activate agent
pip install -r requirements.txt
```

## ACTIVATE

```bash
conda activate agent
```

## VERIFY

```bash
python -V
# Python 3.9.23

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.x.x False (macOS) / True (GPU 서버)

python -c "import sqlalchemy; print(sqlalchemy.__version__)"
# 2.0.x

python -c "from fastapi import FastAPI; print('FastAPI OK')"
```

## RUN

```bash
# 더미 모드 (모델 없이 전체 파이프라인)
PIPELINE_DUMMY=true uvicorn app.api.main:app --host 0.0.0.0 --port 8000

# React UI (별도 터미널)
cd ui && npm run dev

# 더미 파이프라인 데모
python demo_dummy_pipeline.py

# Docker
docker compose up --build
```

## TEST / SMOKE

```bash
PIPELINE_DUMMY=true python -m pytest tests/ -v
# 58 tests passed

python -m compileall src/ app/ -q
```

## NOTES

- macOS에서는 MPS 백엔드 사용 (Apple Silicon)
- GPU 서버에서는 CUDA 사용
- `PIPELINE_DUMMY=true`로 모델 파일 없이 전체 E2E 동작 가능
- NumPy < 2.0 필요 (PyTorch 호환)
- SQLite(dev) / PostgreSQL(prod) 듀얼 지원
