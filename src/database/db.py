"""
데이터베이스 연결 관리
====================

SQLAlchemy 세션 및 연결 풀 관리
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import os
from typing import Generator

# 데이터베이스 URL (환경 변수에서 가져오기)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost/ai_cctv"
)

# 엔진 생성
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # 연결 유효성 검사
    echo=False  # SQL 로깅 (개발 시 True)
)

# 세션 팩토리
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db() -> Generator[Session, None, None]:
    """
    데이터베이스 세션 의존성
    
    FastAPI에서 사용:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """데이터베이스 초기화 (테이블 생성)"""
    from src.database.models import Base
    Base.metadata.create_all(bind=engine)


def drop_db():
    """데이터베이스 삭제 (테이블 삭제)"""
    from src.database.models import Base
    Base.metadata.drop_all(bind=engine)
