FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -m compileall src/ app/ -q

EXPOSE 8000

ENV PIPELINE_DUMMY=true \
    DATABASE_URL=sqlite:///./data/ai_cctv.db \
    JWT_SECRET_KEY=change-me-in-production \
    MAX_CAMERAS=16 \
    GPU_ID=0

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
