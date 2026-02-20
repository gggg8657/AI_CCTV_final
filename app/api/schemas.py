"""
Pydantic 스키마 (요청/응답 모델)
"""

from pydantic import BaseModel, EmailStr, ConfigDict
from typing import Optional, Any
from datetime import datetime, date


# ── Auth ──

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


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime


# ── Camera ──

class CameraCreate(BaseModel):
    name: str
    source_type: str = "rtsp"
    source_path: str
    location: Optional[str] = None
    vad_model: str = "mnad"
    vad_threshold: float = 0.5
    enable_vlm: bool = True
    enable_agent: bool = True
    agent_flow: str = "sequential"
    gpu_id: int = 0


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    source_path: Optional[str] = None
    location: Optional[str] = None
    vad_model: Optional[str] = None
    vad_threshold: Optional[float] = None
    enable_vlm: Optional[bool] = None
    enable_agent: Optional[bool] = None
    agent_flow: Optional[str] = None
    gpu_id: Optional[int] = None


class CameraOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    source_type: str
    source_path: str
    location: Optional[str]
    vad_model: str
    vad_threshold: float
    enable_vlm: bool
    enable_agent: bool
    agent_flow: str
    status: str
    gpu_id: int
    created_at: datetime


# ── Event ──

class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    camera_id: int
    timestamp: datetime
    frame_number: Optional[int]
    vad_score: float
    threshold: Optional[float]
    vlm_type: Optional[str]
    vlm_description: Optional[str]
    vlm_confidence: Optional[float]
    agent_actions: Optional[Any]
    agent_response_time: Optional[float]
    clip_path: Optional[str]
    acknowledged: bool
    acknowledged_by: Optional[int]
    acknowledged_at: Optional[datetime]
    note: Optional[str]
    created_at: datetime


class EventAck(BaseModel):
    note: Optional[str] = None


# ── Stats ──

class DailyStatsOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    camera_id: int
    date: date
    total_frames: int
    anomaly_count: int
    avg_vad_time: Optional[float]
    avg_vlm_time: Optional[float]
    avg_agent_time: Optional[float]
    max_vad_score: Optional[float]
    vlm_type_counts: Optional[Any]


# ── Generic ──

class PaginatedResponse(BaseModel):
    items: list
    total: int
    limit: int
    offset: int
