"""
인증 API 라우터
==============

사용자 등록, 로그인, 토큰 갱신
"""

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter()
security = HTTPBearer()


# Pydantic 모델
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 3600


class RefreshTokenRequest(BaseModel):
    refresh_token: str


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister):
    """사용자 등록"""
    # TODO: 구현 예정
    return {
        "message": "User registration endpoint - 구현 예정",
        "user_data": user_data
    }


@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    """로그인"""
    # TODO: 구현 예정
    return {
        "message": "Login endpoint - 구현 예정",
        "credentials": credentials
    }


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """토큰 갱신"""
    # TODO: 구현 예정
    return {
        "message": "Token refresh endpoint - 구현 예정"
    }
