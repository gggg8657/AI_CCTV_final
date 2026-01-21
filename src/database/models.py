"""
SQLAlchemy 데이터베이스 모델
===========================

데이터베이스 테이블 정의
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, Text, ForeignKey, JSON, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class User(Base):
    """사용자 테이블"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="viewer")  # viewer, operator, admin
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    camera_accesses = relationship("CameraAccess", back_populates="user", cascade="all, delete-orphan")
    acknowledged_events = relationship("Event", foreign_keys="Event.acknowledged_by")


class Camera(Base):
    """카메라 테이블"""
    __tablename__ = "cameras"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    source_type = Column(String(50), nullable=False)  # file, rtsp, webcam
    source_path = Column(Text, nullable=False)
    location = Column(String(255))
    vad_model = Column(String(50), default="mnad")
    vad_threshold = Column(Float, default=0.5)
    enable_vlm = Column(Boolean, default=True)
    enable_agent = Column(Boolean, default=True)
    agent_flow = Column(String(50), default="sequential")
    status = Column(String(50), default="inactive", index=True)  # active, inactive, error
    gpu_id = Column(Integer, default=0)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    events = relationship("Event", back_populates="camera", cascade="all, delete-orphan")
    daily_statistics = relationship("DailyStatistics", back_populates="camera", cascade="all, delete-orphan")
    accesses = relationship("CameraAccess", back_populates="camera", cascade="all, delete-orphan")
    notification_rules = relationship("NotificationRule", back_populates="camera", cascade="all, delete-orphan")


class Event(Base):
    """이벤트 테이블"""
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    frame_number = Column(Integer)
    vad_score = Column(Float, nullable=False)
    threshold = Column(Float)
    vlm_type = Column(String(100), index=True)
    vlm_description = Column(Text)
    vlm_confidence = Column(Float)
    agent_actions = Column(JSON)  # JSONB in PostgreSQL
    agent_response_time = Column(Float)
    clip_path = Column(Text)
    acknowledged = Column(Boolean, default=False, index=True)
    acknowledged_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    note = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    camera = relationship("Camera", back_populates="events")


class DailyStatistics(Base):
    """일별 통계 테이블"""
    __tablename__ = "daily_statistics"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    total_frames = Column(Integer, default=0)
    anomaly_count = Column(Integer, default=0)
    avg_vad_time = Column(Float)
    avg_vlm_time = Column(Float)
    avg_agent_time = Column(Float)
    max_vad_score = Column(Float)
    min_vad_score = Column(Float)
    vlm_type_counts = Column(JSON)  # {"Fighting": 5, "Arson": 2, ...}
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    camera = relationship("Camera", back_populates="daily_statistics")
    
    # Unique constraint
    __table_args__ = (
        UniqueConstraint('camera_id', 'date', name='uq_camera_date'),
    )


class CameraAccess(Base):
    """카메라 접근 권한 테이블"""
    __tablename__ = "camera_access"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    permission = Column(String(50), default="view")  # view, control, admin
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    user = relationship("User", back_populates="camera_accesses")
    camera = relationship("Camera", back_populates="accesses")
    
    # Unique constraint
    __table_args__ = (
        UniqueConstraint('user_id', 'camera_id', name='uq_user_camera'),
    )


class NotificationRule(Base):
    """알림 규칙 테이블"""
    __tablename__ = "notification_rules"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True, index=True)  # None이면 모든 카메라
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)  # None이면 모든 사용자
    vlm_type = Column(String(100), nullable=True, index=True)  # None이면 모든 유형
    min_score = Column(Float, nullable=True)
    channels = Column(JSON, nullable=False)  # ["email", "sms", "webhook"]
    webhook_url = Column(Text, nullable=True)
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    camera = relationship("Camera", back_populates="notification_rules")
