# 상세 기능 명세서

**버전**: 1.0  
**작성일**: 2025-01-21

---

## 목차

1. [멀티 카메라 시스템](#1-멀티-카메라-시스템)
2. [REST API 서버](#2-rest-api-서버)
3. [데이터베이스 설계](#3-데이터베이스-설계)
4. [알림 시스템](#4-알림-시스템)
5. [인증 및 권한 관리](#5-인증-및-권한-관리)
6. [프론트엔드 통합](#6-프론트엔드-통합)

---

## 1. 멀티 카메라 시스템

### 1.1 요구사항

#### 기능 요구사항
- 최대 16개 카메라 동시 모니터링
- 카메라별 독립적인 VAD/VLM/Agent 파이프라인
- 카메라 추가/삭제/수정 (동적 관리)
- 카메라별 설정 관리 (모델, 임계값 등)
- 통합 대시보드에서 모든 카메라 상태 표시

#### 비기능 요구사항
- GPU 메모리 효율적 사용 (모델 공유)
- 한 카메라의 오류가 다른 카메라에 영향 없음
- 카메라별 처리 지연 시간 < 100ms
- 시스템 리소스 사용량 모니터링

### 1.2 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│              MultiCameraManager                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Camera Registry                                   │  │
│  │  - Camera ID → CameraConfig                        │  │
│  │  - Camera ID → CameraPipeline                      │  │
│  │  - Camera ID → CameraStatus                        │  │
│  └───────────────────────────────────────────────────┘  │
│                                                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Resource Pool Manager                             │  │
│  │  - GPU Memory Pool                                 │  │
│  │  - Model Instance Pool (VAD/VLM/Agent)            │  │
│  │  - Thread Pool                                     │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
    ┌───▼───┐   ┌───▼───┐   ┌───▼───┐
    │Camera1│   │Camera2│   │CameraN│
    │Pipeline│  │Pipeline│  │Pipeline│
    └───────┘   └───────┘   └───────┘
```

### 1.3 클래스 설계

#### MultiCameraManager

```python
class MultiCameraManager:
    """멀티 카메라 관리자"""
    
    def __init__(self, max_cameras: int = 16):
        self.max_cameras = max_cameras
        self.cameras: Dict[int, CameraPipeline] = {}
        self.resource_pool = ResourcePool()
        self.lock = threading.Lock()
    
    def add_camera(self, config: CameraConfig) -> int:
        """카메라 추가"""
        # 1. 카메라 ID 할당
        # 2. 리소스 할당 (GPU, 모델)
        # 3. 파이프라인 생성 및 시작
        # 4. 레지스트리에 등록
    
    def remove_camera(self, camera_id: int) -> bool:
        """카메라 제거"""
        # 1. 파이프라인 중지
        # 2. 리소스 해제
        # 3. 레지스트리에서 제거
    
    def get_camera_status(self, camera_id: int) -> CameraStatus:
        """카메라 상태 조회"""
    
    def get_all_status(self) -> Dict[int, CameraStatus]:
        """모든 카메라 상태 조회"""
```

#### CameraPipeline

```python
class CameraPipeline:
    """카메라별 파이프라인"""
    
    def __init__(self, camera_id: int, config: CameraConfig):
        self.camera_id = camera_id
        self.config = config
        self.e2e_system = None
        self.status = CameraStatus.INACTIVE
        self.event_queue = queue.Queue()
        self.thread = None
    
    def start(self):
        """파이프라인 시작"""
        # 1. E2ESystem 생성
        # 2. 콜백 설정 (이벤트 큐에 추가)
        # 3. 스레드에서 실행
    
    def stop(self):
        """파이프라인 중지"""
        # 1. E2ESystem 중지
        # 2. 스레드 종료
        # 3. 리소스 정리
    
    def _process_loop(self):
        """처리 루프"""
        # 1. E2ESystem.start() 호출
        # 2. 이벤트 수신 및 큐에 추가
```

#### ResourcePool

```python
class ResourcePool:
    """리소스 풀 관리자"""
    
    def __init__(self):
        self.vad_models: Dict[str, VADModel] = {}
        self.vlm_analyzer: Optional[VLMAnalyzer] = None
        self.agent_flows: Dict[str, AgentFlow] = {}
        self.gpu_memory_tracker = GPUMemoryTracker()
    
    def get_vad_model(self, model_type: str) -> VADModel:
        """VAD 모델 가져오기 (공유)"""
        if model_type not in self.vad_models:
            self.vad_models[model_type] = create_vad_model(model_type)
        return self.vad_models[model_type]
    
    def get_vlm_analyzer(self) -> VLMAnalyzer:
        """VLM 분석기 가져오기 (싱글톤)"""
        if self.vlm_analyzer is None:
            self.vlm_analyzer = VLMAnalyzer(...)
        return self.vlm_analyzer
    
    def check_gpu_memory(self) -> bool:
        """GPU 메모리 여유 확인"""
        return self.gpu_memory_tracker.has_available()
```

### 1.4 데이터 모델

#### CameraConfig

```python
@dataclass
class CameraConfig:
    """카메라 설정"""
    id: int
    name: str
    source_type: VideoSourceType  # FILE, RTSP, WEBCAM
    source_path: str
    location: str
    vad_model: VADModelType
    vad_threshold: float = 0.5
    enable_vlm: bool = True
    enable_agent: bool = True
    agent_flow: AgentFlowType = AgentFlowType.SEQUENTIAL
    gpu_id: int = 0
```

#### CameraStatus

```python
@dataclass
class CameraStatus:
    """카메라 상태"""
    camera_id: int
    status: str  # ACTIVE, INACTIVE, ERROR
    total_frames: int
    anomaly_count: int
    current_fps: float
    last_event_time: Optional[datetime]
    error_message: Optional[str]
    resource_usage: Dict[str, float]  # GPU, CPU, Memory
```

### 1.5 구현 파일 구조

```
src/pipeline/
├── __init__.py
├── multi_camera.py          # MultiCameraManager
├── camera_pipeline.py       # CameraPipeline
├── resource_pool.py         # ResourcePool
└── camera_config.py        # CameraConfig, CameraStatus
```

---

## 2. REST API 서버

### 2.1 API 구조

```
app/api/
├── __init__.py
├── main.py                 # FastAPI 앱
├── dependencies.py         # 의존성 (인증 등)
├── routers/
│   ├── __init__.py
│   ├── cameras.py          # 카메라 관리
│   ├── events.py           # 이벤트 조회
│   ├── stats.py            # 통계
│   ├── auth.py             # 인증
│   └── stream.py           # WebSocket 스트리밍
└── models/
    ├── __init__.py
    ├── camera.py           # Pydantic 모델
    ├── event.py
    └── user.py
```

### 2.2 API 엔드포인트 상세

#### 2.2.1 인증 API

```python
# POST /api/v1/auth/register
{
    "username": "admin",
    "email": "admin@example.com",
    "password": "secure_password"
}
Response: {
    "user_id": 1,
    "username": "admin",
    "email": "admin@example.com",
    "role": "viewer"
}

# POST /api/v1/auth/login
{
    "username": "admin",
    "password": "secure_password"
}
Response: {
    "access_token": "eyJ...",
    "refresh_token": "eyJ...",
    "token_type": "bearer",
    "expires_in": 3600
}

# POST /api/v1/auth/refresh
{
    "refresh_token": "eyJ..."
}
Response: {
    "access_token": "eyJ...",
    "token_type": "bearer",
    "expires_in": 3600
}
```

#### 2.2.2 카메라 관리 API

```python
# GET /api/v1/cameras
Response: {
    "cameras": [
        {
            "id": 1,
            "name": "Building A - Entrance",
            "source_type": "rtsp",
            "source_path": "rtsp://192.168.1.100/stream",
            "location": "Building A, Floor 1",
            "status": "active",
            "vad_model": "mnad",
            "vad_threshold": 0.5,
            "created_at": "2025-01-20T10:00:00Z"
        }
    ],
    "total": 1
}

# POST /api/v1/cameras
Request: {
    "name": "Building B - Parking",
    "source_type": "rtsp",
    "source_path": "rtsp://192.168.1.101/stream",
    "location": "Building B, Parking Lot",
    "vad_model": "mnad",
    "vad_threshold": 0.5
}
Response: {
    "id": 2,
    "name": "Building B - Parking",
    ...
}

# GET /api/v1/cameras/{id}
Response: {
    "id": 1,
    "name": "Building A - Entrance",
    "status": "active",
    "stats": {
        "total_frames": 864000,
        "anomaly_count": 12,
        "current_fps": 30.5
    }
}

# PUT /api/v1/cameras/{id}
Request: {
    "name": "Updated Name",
    "vad_threshold": 0.6
}

# DELETE /api/v1/cameras/{id}
Response: {
    "message": "Camera deleted successfully"
}

# POST /api/v1/cameras/{id}/start
Response: {
    "message": "Camera started",
    "camera_id": 1
}

# POST /api/v1/cameras/{id}/stop
Response: {
    "message": "Camera stopped",
    "camera_id": 1
}
```

#### 2.2.3 이벤트 API

```python
# GET /api/v1/events
Query Parameters:
    - camera_id: int (optional)
    - start_date: ISO datetime (optional)
    - end_date: ISO datetime (optional)
    - vlm_type: str (optional)
    - min_score: float (optional)
    - limit: int (default: 100)
    - offset: int (default: 0)

Response: {
    "events": [
        {
            "id": 123,
            "camera_id": 1,
            "camera_name": "Building A - Entrance",
            "timestamp": "2025-01-20T10:30:00Z",
            "frame_number": 108900,
            "vad_score": 0.85,
            "threshold": 0.5,
            "vlm_type": "Fighting",
            "vlm_description": "Two people engaged in physical altercation",
            "vlm_confidence": 0.92,
            "agent_actions": [
                {
                    "action": "alert_security",
                    "priority": "high",
                    "description": "Notify security team"
                }
            ],
            "clip_path": "/clips/camera_1_event_123.mp4",
            "acknowledged": false
        }
    ],
    "total": 150,
    "limit": 100,
    "offset": 0
}

# GET /api/v1/events/{id}
Response: {
    "id": 123,
    ...
}

# POST /api/v1/events/{id}/ack
Request: {
    "acknowledged": true,
    "note": "Handled by security team"
}
Response: {
    "message": "Event acknowledged",
    "event_id": 123
}
```

#### 2.2.4 통계 API

```python
# GET /api/v1/stats
Query Parameters:
    - camera_id: int (optional)
    - date: YYYY-MM-DD (optional, default: today)
    - period: str (day/week/month, default: day)

Response: {
    "period": "day",
    "date": "2025-01-20",
    "cameras": [
        {
            "camera_id": 1,
            "camera_name": "Building A - Entrance",
            "total_frames": 864000,
            "anomaly_count": 12,
            "anomaly_rate": 0.0014,
            "avg_vad_time_ms": 3.77,
            "avg_vlm_time_ms": 5000,
            "avg_agent_time_ms": 200,
            "max_vad_score": 0.95,
            "vlm_types": {
                "Fighting": 5,
                "Suspicious_Object": 3,
                "Falling": 2,
                "Normal": 2
            }
        }
    ],
    "summary": {
        "total_cameras": 1,
        "total_frames": 864000,
        "total_anomalies": 12
    }
}

# GET /api/v1/stats/trends
Query Parameters:
    - camera_id: int (optional)
    - days: int (default: 7)

Response: {
    "trends": [
        {
            "date": "2025-01-14",
            "anomaly_count": 8,
            "avg_score": 0.65
        },
        ...
    ]
}
```

### 2.3 WebSocket API

```python
# WS /ws/stream/{camera_id}
# 클라이언트 → 서버
{
    "type": "subscribe",
    "camera_id": 1
}

# 서버 → 클라이언트
{
    "type": "frame",
    "camera_id": 1,
    "frame": "base64_encoded_image",
    "score": 0.45,
    "timestamp": "2025-01-20T10:30:00Z"
}

{
    "type": "event",
    "camera_id": 1,
    "event": {
        "id": 123,
        "vad_score": 0.85,
        "vlm_type": "Fighting",
        ...
    }
}

{
    "type": "stats",
    "camera_id": 1,
    "stats": {
        "current_fps": 30.5,
        "anomaly_count": 12,
        ...
    }
}
```

### 2.4 인증 미들웨어

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """현재 사용자 조회"""
    token = credentials.credentials
    payload = verify_jwt_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    user = get_user_by_id(payload["user_id"])
    return user

async def require_role(required_role: str):
    """역할 기반 접근 제어"""
    def check_role(user: User = Depends(get_current_user)):
        if user.role != required_role and user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return user
    return check_role
```

---

## 3. 데이터베이스 설계

### 3.1 ERD

```
┌─────────────┐
│   users     │
├─────────────┤
│ id (PK)     │
│ username    │
│ email       │
│ password    │
│ role        │
│ created_at  │
└──────┬──────┘
       │
       │ 1:N
       │
┌──────▼──────────┐
│ camera_access   │
├─────────────────┤
│ user_id (FK)    │
│ camera_id (FK)  │
│ permission      │
└──────┬──────────┘
       │
       │ N:1
       │
┌──────▼──────┐      ┌──────────┐
│   cameras   │◄─────┤  events  │
├─────────────┤ 1:N  ├──────────┤
│ id (PK)     │      │ id (PK)  │
│ name        │      │ camera_id│
│ source_type │      │ timestamp│
│ source_path │      │ vad_score│
│ location    │      │ vlm_type │
│ status      │      │ ...      │
│ ...         │      └──────────┘
└─────────────┘
       │
       │ 1:N
       │
┌──────▼──────────────┐
│ daily_statistics    │
├─────────────────────┤
│ id (PK)             │
│ camera_id (FK)      │
│ date                │
│ total_frames        │
│ anomaly_count       │
│ ...                 │
└─────────────────────┘
```

### 3.2 테이블 상세

#### users

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'viewer',  -- viewer, operator, admin
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
```

#### cameras

```sql
CREATE TABLE cameras (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    source_type VARCHAR(50) NOT NULL,  -- file, rtsp, webcam
    source_path TEXT NOT NULL,
    location VARCHAR(255),
    vad_model VARCHAR(50) DEFAULT 'mnad',
    vad_threshold FLOAT DEFAULT 0.5,
    enable_vlm BOOLEAN DEFAULT TRUE,
    enable_agent BOOLEAN DEFAULT TRUE,
    agent_flow VARCHAR(50) DEFAULT 'sequential',
    status VARCHAR(50) DEFAULT 'inactive',  -- active, inactive, error
    gpu_id INTEGER DEFAULT 0,
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_cameras_status ON cameras(status);
CREATE INDEX idx_cameras_location ON cameras(location);
```

#### events

```sql
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    camera_id INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    timestamp TIMESTAMP NOT NULL,
    frame_number INTEGER,
    vad_score FLOAT NOT NULL,
    threshold FLOAT,
    vlm_type VARCHAR(100),
    vlm_description TEXT,
    vlm_confidence FLOAT,
    agent_actions JSONB,
    agent_response_time FLOAT,
    clip_path TEXT,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by INTEGER REFERENCES users(id),
    acknowledged_at TIMESTAMP,
    note TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_events_camera_id ON events(camera_id);
CREATE INDEX idx_events_timestamp ON events(timestamp);
CREATE INDEX idx_events_vlm_type ON events(vlm_type);
CREATE INDEX idx_events_acknowledged ON events(acknowledged);
```

#### daily_statistics

```sql
CREATE TABLE daily_statistics (
    id SERIAL PRIMARY KEY,
    camera_id INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    total_frames INTEGER DEFAULT 0,
    anomaly_count INTEGER DEFAULT 0,
    avg_vad_time FLOAT,
    avg_vlm_time FLOAT,
    avg_agent_time FLOAT,
    max_vad_score FLOAT,
    min_vad_score FLOAT,
    vlm_type_counts JSONB,  -- {"Fighting": 5, "Arson": 2, ...}
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(camera_id, date)
);

CREATE INDEX idx_daily_stats_camera_date ON daily_statistics(camera_id, date);
```

#### camera_access

```sql
CREATE TABLE camera_access (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    camera_id INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    permission VARCHAR(50) DEFAULT 'view',  -- view, control, admin
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, camera_id)
);

CREATE INDEX idx_camera_access_user ON camera_access(user_id);
CREATE INDEX idx_camera_access_camera ON camera_access(camera_id);
```

#### notification_rules

```sql
CREATE TABLE notification_rules (
    id SERIAL PRIMARY KEY,
    camera_id INTEGER REFERENCES cameras(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    vlm_type VARCHAR(100),
    min_score FLOAT,
    channels JSONB,  -- ["email", "sms", "webhook"]
    webhook_url TEXT,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_notification_rules_camera ON notification_rules(camera_id);
CREATE INDEX idx_notification_rules_user ON notification_rules(user_id);
```

### 3.3 SQLAlchemy 모델

```python
# src/database/models.py

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="viewer")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    camera_accesses = relationship("CameraAccess", back_populates="user")
    acknowledged_events = relationship("Event", foreign_keys="Event.acknowledged_by")

class Camera(Base):
    __tablename__ = "cameras"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    source_type = Column(String(50), nullable=False)
    source_path = Column(Text, nullable=False)
    location = Column(String(255))
    vad_model = Column(String(50), default="mnad")
    vad_threshold = Column(Float, default=0.5)
    enable_vlm = Column(Boolean, default=True)
    enable_agent = Column(Boolean, default=True)
    agent_flow = Column(String(50), default="sequential")
    status = Column(String(50), default="inactive")
    gpu_id = Column(Integer, default=0)
    created_by = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    events = relationship("Event", back_populates="camera")
    daily_statistics = relationship("DailyStatistics", back_populates="camera")
    accesses = relationship("CameraAccess", back_populates="camera")

class Event(Base):
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    frame_number = Column(Integer)
    vad_score = Column(Float, nullable=False)
    threshold = Column(Float)
    vlm_type = Column(String(100))
    vlm_description = Column(Text)
    vlm_confidence = Column(Float)
    agent_actions = Column(JSON)
    agent_response_time = Column(Float)
    clip_path = Column(Text)
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(Integer, ForeignKey("users.id"))
    acknowledged_at = Column(DateTime)
    note = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    camera = relationship("Camera", back_populates="events")
```

---

## 4. 알림 시스템

### 4.1 아키텍처

```
Event Detected
    │
    ▼
NotificationEngine
    │
    ├─→ Rule Matcher
    │       │
    │       └─→ Find Matching Rules
    │
    └─→ Channel Dispatcher
            │
            ├─→ EmailChannel
            ├─→ WebhookChannel
            └─→ SMSChannel (선택적)
```

### 4.2 구현

```python
# src/notifications/__init__.py

from abc import ABC, abstractmethod
from typing import List, Dict

class NotificationChannel(ABC):
    """알림 채널 인터페이스"""
    
    @abstractmethod
    def send(self, message: Dict) -> bool:
        """알림 발송"""
        pass

class EmailChannel(NotificationChannel):
    """이메일 알림 채널"""
    
    def __init__(self, smtp_config: Dict):
        self.smtp_host = smtp_config["host"]
        self.smtp_port = smtp_config["port"]
        self.smtp_user = smtp_config["user"]
        self.smtp_password = smtp_config["password"]
    
    def send(self, message: Dict) -> bool:
        # SMTP를 통한 이메일 발송
        pass

class WebhookChannel(NotificationChannel):
    """웹훅 알림 채널"""
    
    def send(self, message: Dict) -> bool:
        # HTTP POST 요청
        pass

class NotificationEngine:
    """알림 엔진"""
    
    def __init__(self, db_session):
        self.db = db_session
        self.channels = {
            "email": EmailChannel(...),
            "webhook": WebhookChannel(...)
        }
    
    def notify(self, event: Event):
        """이벤트 알림 처리"""
        # 1. 알림 규칙 조회
        rules = self._get_matching_rules(event)
        
        # 2. 각 규칙에 대해 알림 발송
        for rule in rules:
            channels = rule.channels
            message = self._build_message(event, rule)
            
            for channel_name in channels:
                if channel_name in self.channels:
                    self.channels[channel_name].send(message)
    
    def _get_matching_rules(self, event: Event) -> List[NotificationRule]:
        """매칭되는 알림 규칙 조회"""
        # 데이터베이스에서 규칙 조회
        # - camera_id 매칭
        # - vlm_type 매칭
        # - min_score 체크
        pass
    
    def _build_message(self, event: Event, rule: NotificationRule) -> Dict:
        """알림 메시지 생성"""
        return {
            "subject": f"Anomaly Detected: {event.vlm_type}",
            "body": f"Camera: {event.camera.name}\n"
                   f"Type: {event.vlm_type}\n"
                   f"Score: {event.vad_score}\n"
                   f"Time: {event.timestamp}",
            "event_id": event.id,
            "clip_url": f"http://api.example.com/clips/{event.clip_path}"
        }
```

### 4.3 알림 규칙 예시

```python
# 이메일 알림 규칙
{
    "camera_id": 1,
    "vlm_type": "Fighting",
    "min_score": 0.7,
    "channels": ["email"],
    "enabled": True
}

# 웹훅 알림 규칙
{
    "camera_id": None,  # 모든 카메라
    "vlm_type": "Arson",
    "min_score": 0.8,
    "channels": ["webhook"],
    "webhook_url": "https://slack.com/api/webhook/...",
    "enabled": True
}
```

---

## 5. 인증 및 권한 관리

### 5.1 인증 흐름

```
1. 사용자 등록/로그인
   ↓
2. JWT 토큰 발급
   ↓
3. API 요청 시 토큰 포함
   ↓
4. 미들웨어에서 토큰 검증
   ↓
5. 사용자 정보 추출
   ↓
6. 권한 확인
   ↓
7. 요청 처리
```

### 5.2 역할 정의

| 역할 | 권한 |
|------|------|
| **viewer** | 카메라 조회, 이벤트 조회, 통계 조회 |
| **operator** | viewer 권한 + 카메라 제어 (start/stop), 이벤트 확인 |
| **admin** | 모든 권한 + 카메라 추가/삭제, 사용자 관리, 설정 변경 |

### 5.3 카메라별 접근 제어

```python
# 사용자 A: camera 1, 2 접근 가능
# 사용자 B: camera 2, 3 접근 가능

# GET /api/v1/cameras
# → 사용자 A는 camera 1, 2만 조회 가능

# GET /api/v1/events?camera_id=3
# → 사용자 A는 403 Forbidden
```

### 5.4 구현

```python
# app/api/dependencies.py

async def get_accessible_cameras(
    user: User = Depends(get_current_user)
) -> List[int]:
    """사용자가 접근 가능한 카메라 ID 목록"""
    if user.role == "admin":
        # 관리자는 모든 카메라 접근 가능
        cameras = db.query(Camera).all()
        return [c.id for c in cameras]
    else:
        # 일반 사용자는 권한이 있는 카메라만
        accesses = db.query(CameraAccess).filter(
            CameraAccess.user_id == user.id
        ).all()
        return [a.camera_id for a in accesses]

async def check_camera_access(
    camera_id: int,
    user: User = Depends(get_current_user)
) -> Camera:
    """카메라 접근 권한 확인"""
    if user.role == "admin":
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
    else:
        access = db.query(CameraAccess).filter(
            CameraAccess.user_id == user.id,
            CameraAccess.camera_id == camera_id
        ).first()
        if not access:
            raise HTTPException(403, "Access denied")
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
    
    if not camera:
        raise HTTPException(404, "Camera not found")
    return camera
```

---

## 6. 프론트엔드 통합

### 6.1 API 클라이언트

```typescript
// ui/src/lib/api.ts

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 인증 토큰 추가
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// API 함수들
export const cameraAPI = {
  list: () => apiClient.get('/cameras'),
  get: (id: number) => apiClient.get(`/cameras/${id}`),
  create: (data: CameraCreateRequest) => apiClient.post('/cameras', data),
  update: (id: number, data: CameraUpdateRequest) => 
    apiClient.put(`/cameras/${id}`, data),
  delete: (id: number) => apiClient.delete(`/cameras/${id}`),
  start: (id: number) => apiClient.post(`/cameras/${id}/start`),
  stop: (id: number) => apiClient.post(`/cameras/${id}/stop`),
};

export const eventAPI = {
  list: (params?: EventListParams) => 
    apiClient.get('/events', { params }),
  get: (id: number) => apiClient.get(`/events/${id}`),
  acknowledge: (id: number, data: AckRequest) =>
    apiClient.post(`/events/${id}/ack`, data),
};

export const statsAPI = {
  get: (params?: StatsParams) => apiClient.get('/stats', { params }),
  trends: (params?: TrendsParams) => apiClient.get('/stats/trends', { params }),
};
```

### 6.2 WebSocket 클라이언트

```typescript
// ui/src/lib/websocket.ts

export class StreamWebSocket {
  private ws: WebSocket | null = null;
  private cameraId: number;
  private callbacks: Map<string, Function[]> = new Map();

  constructor(cameraId: number) {
    this.cameraId = cameraId;
  }

  connect() {
    const wsUrl = `ws://localhost:8000/ws/stream/${this.cameraId}`;
    this.ws = new WebSocket(wsUrl);

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.handleMessage(data);
    };

    this.ws.onopen = () => {
      this.ws?.send(JSON.stringify({
        type: 'subscribe',
        camera_id: this.cameraId,
      }));
    };
  }

  on(event: string, callback: Function) {
    if (!this.callbacks.has(event)) {
      this.callbacks.set(event, []);
    }
    this.callbacks.get(event)?.push(callback);
  }

  private handleMessage(data: any) {
    const callbacks = this.callbacks.get(data.type) || [];
    callbacks.forEach(cb => cb(data));
  }

  disconnect() {
    this.ws?.close();
  }
}
```

### 6.3 React 컴포넌트 통합

```typescript
// ui/src/components/LiveCameraGrid.tsx

import { useEffect, useState } from 'react';
import { cameraAPI } from '../lib/api';
import { StreamWebSocket } from '../lib/websocket';

export function LiveCameraGrid() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [streams, setStreams] = useState<Map<number, StreamWebSocket>>(new Map());

  useEffect(() => {
    // 카메라 목록 조회
    cameraAPI.list().then(res => {
      setCameras(res.data.cameras);
      
      // 각 카메라에 대해 WebSocket 연결
      res.data.cameras.forEach((camera: Camera) => {
        if (camera.status === 'active') {
          const ws = new StreamWebSocket(camera.id);
          ws.on('frame', (data) => {
            // 프레임 업데이트
            updateFrame(camera.id, data.frame, data.score);
          });
          ws.on('event', (data) => {
            // 이벤트 알림
            showEventNotification(data.event);
          });
          ws.connect();
          setStreams(prev => new Map(prev).set(camera.id, ws));
        }
      });
    });

    return () => {
      // 정리
      streams.forEach(ws => ws.disconnect());
    };
  }, []);

  return (
    <div className="grid grid-cols-4 gap-4">
      {cameras.map(camera => (
        <CameraCard key={camera.id} camera={camera} />
      ))}
    </div>
  );
}
```

---

## 7. 테스트 계획

### 7.1 단위 테스트

- VAD 모델 추론 테스트
- VLM 분석기 테스트
- Agent Flow 테스트
- API 엔드포인트 테스트
- 데이터베이스 모델 테스트

### 7.2 통합 테스트

- E2E 파이프라인 테스트
- 멀티 카메라 통합 테스트
- API-데이터베이스 통합 테스트
- 알림 시스템 통합 테스트

### 7.3 E2E 테스트

- 사용자 시나리오 테스트
- 성능 테스트
- 부하 테스트

---

**문서 버전**: 1.0  
**최종 업데이트**: 2025-01-21
