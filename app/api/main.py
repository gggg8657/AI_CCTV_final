"""
FastAPI 메인 애플리케이션
========================

REST API 서버 진입점
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import sys
from pathlib import Path

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.api.routers import auth, cameras, events, stats, stream


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # 시작 시
    print("Starting API server...")
    yield
    # 종료 시
    print("Shutting down API server...")


# FastAPI 앱 생성
app = FastAPI(
    title="AI CCTV 통합 시스템 API",
    description="VAD + VLM + Agentic AI 통합 보안 모니터링 시스템",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(auth.router, prefix="/api/v1/auth", tags=["인증"])
app.include_router(cameras.router, prefix="/api/v1/cameras", tags=["카메라"])
app.include_router(events.router, prefix="/api/v1/events", tags=["이벤트"])
app.include_router(stats.router, prefix="/api/v1/stats", tags=["통계"])
app.include_router(stream.router, prefix="/ws", tags=["스트리밍"])


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "AI CCTV 통합 시스템 API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    return {
        "status": "healthy",
        "service": "ai-cctv-api"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
