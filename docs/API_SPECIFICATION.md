# REST API 명세서

**버전**: 1.0  
**Base URL**: `http://localhost:8000/api/v1`  
**인증**: Bearer Token (JWT)

---

## 목차

1. [인증 API](#1-인증-api)
2. [카메라 관리 API](#2-카메라-관리-api)
3. [이벤트 API](#3-이벤트-api)
4. [통계 API](#4-통계-api)
5. [WebSocket API](#5-websocket-api)
6. [에러 응답](#6-에러-응답)

---

## 1. 인증 API

### 1.1 사용자 등록

```http
POST /auth/register
Content-Type: application/json

{
  "username": "admin",
  "email": "admin@example.com",
  "password": "secure_password123"
}
```

**Response 201 Created**:
```json
{
  "user_id": 1,
  "username": "admin",
  "email": "admin@example.com",
  "role": "viewer",
  "created_at": "2025-01-21T10:00:00Z"
}
```

**Response 400 Bad Request** (중복 사용자명):
```json
{
  "detail": "Username already exists"
}
```

---

### 1.2 로그인

```http
POST /auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "secure_password123"
}
```

**Response 200 OK**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

**Response 401 Unauthorized**:
```json
{
  "detail": "Invalid username or password"
}
```

---

### 1.3 토큰 갱신

```http
POST /auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response 200 OK**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

---

## 2. 카메라 관리 API

### 2.1 카메라 목록 조회

```http
GET /cameras
Authorization: Bearer {token}
```

**Query Parameters**:
- `status` (optional): `active`, `inactive`, `error`
- `location` (optional): 위치 필터
- `limit` (optional, default: 100): 페이지 크기
- `offset` (optional, default: 0): 오프셋

**Response 200 OK**:
```json
{
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
      "enable_vlm": true,
      "enable_agent": true,
      "agent_flow": "sequential",
      "gpu_id": 0,
      "created_at": "2025-01-20T10:00:00Z",
      "updated_at": "2025-01-20T10:00:00Z"
    }
  ],
  "total": 1,
  "limit": 100,
  "offset": 0
}
```

**권한**: 모든 인증된 사용자 (자신이 접근 가능한 카메라만)

---

### 2.2 카메라 상세 조회

```http
GET /cameras/{id}
Authorization: Bearer {token}
```

**Response 200 OK**:
```json
{
  "id": 1,
  "name": "Building A - Entrance",
  "source_type": "rtsp",
  "source_path": "rtsp://192.168.1.100/stream",
  "location": "Building A, Floor 1",
  "status": "active",
  "vad_model": "mnad",
  "vad_threshold": 0.5,
  "enable_vlm": true,
  "enable_agent": true,
  "agent_flow": "sequential",
  "gpu_id": 0,
  "stats": {
    "total_frames": 864000,
    "anomaly_count": 12,
    "current_fps": 30.5,
    "last_event_time": "2025-01-21T09:30:00Z"
  },
  "created_at": "2025-01-20T10:00:00Z",
  "updated_at": "2025-01-20T10:00:00Z"
}
```

**Response 404 Not Found**:
```json
{
  "detail": "Camera not found"
}
```

**Response 403 Forbidden** (접근 권한 없음):
```json
{
  "detail": "Access denied"
}
```

---

### 2.3 카메라 생성

```http
POST /cameras
Authorization: Bearer {token}
Content-Type: application/json

{
  "name": "Building B - Parking",
  "source_type": "rtsp",
  "source_path": "rtsp://192.168.1.101/stream",
  "location": "Building B, Parking Lot",
  "vad_model": "mnad",
  "vad_threshold": 0.5,
  "enable_vlm": true,
  "enable_agent": true,
  "agent_flow": "sequential"
}
```

**Response 201 Created**:
```json
{
  "id": 2,
  "name": "Building B - Parking",
  "source_type": "rtsp",
  "source_path": "rtsp://192.168.1.101/stream",
  "location": "Building B, Parking Lot",
  "status": "inactive",
  "vad_model": "mnad",
  "vad_threshold": 0.5,
  "enable_vlm": true,
  "enable_agent": true,
  "agent_flow": "sequential",
  "created_at": "2025-01-21T10:00:00Z"
}
```

**권한**: `admin` 역할만

---

### 2.4 카메라 수정

```http
PUT /cameras/{id}
Authorization: Bearer {token}
Content-Type: application/json

{
  "name": "Updated Name",
  "vad_threshold": 0.6,
  "location": "Updated Location"
}
```

**Response 200 OK**:
```json
{
  "id": 1,
  "name": "Updated Name",
  "vad_threshold": 0.6,
  "location": "Updated Location",
  "updated_at": "2025-01-21T10:05:00Z"
}
```

**권한**: `admin` 또는 해당 카메라에 `control` 권한

---

### 2.5 카메라 삭제

```http
DELETE /cameras/{id}
Authorization: Bearer {token}
```

**Response 200 OK**:
```json
{
  "message": "Camera deleted successfully",
  "camera_id": 1
}
```

**권한**: `admin` 역할만

---

### 2.6 카메라 시작

```http
POST /cameras/{id}/start
Authorization: Bearer {token}
```

**Response 200 OK**:
```json
{
  "message": "Camera started",
  "camera_id": 1,
  "status": "active"
}
```

**Response 400 Bad Request** (이미 실행 중):
```json
{
  "detail": "Camera is already active"
}
```

**권한**: `admin` 또는 해당 카메라에 `control` 권한

---

### 2.7 카메라 중지

```http
POST /cameras/{id}/stop
Authorization: Bearer {token}
```

**Response 200 OK**:
```json
{
  "message": "Camera stopped",
  "camera_id": 1,
  "status": "inactive"
}
```

**권한**: `admin` 또는 해당 카메라에 `control` 권한

---

## 3. 이벤트 API

### 3.1 이벤트 목록 조회

```http
GET /events
Authorization: Bearer {token}
```

**Query Parameters**:
- `camera_id` (optional): 카메라 ID 필터
- `start_date` (optional): 시작 날짜 (ISO 8601)
- `end_date` (optional): 종료 날짜 (ISO 8601)
- `vlm_type` (optional): 이상 유형 필터
- `min_score` (optional): 최소 VAD 점수
- `acknowledged` (optional): 확인 여부 (`true`/`false`)
- `limit` (optional, default: 100): 페이지 크기
- `offset` (optional, default: 0): 오프셋

**Response 200 OK**:
```json
{
  "events": [
    {
      "id": 123,
      "camera_id": 1,
      "camera_name": "Building A - Entrance",
      "timestamp": "2025-01-21T09:30:00Z",
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
          "description": "Notify security team immediately"
        },
        {
          "action": "dispatch_guard",
          "priority": "medium",
          "description": "Send security guard to location"
        }
      ],
      "agent_response_time": 0.18,
      "clip_path": "/clips/camera_1_event_123.mp4",
      "acknowledged": false,
      "acknowledged_by": null,
      "acknowledged_at": null,
      "note": null,
      "created_at": "2025-01-21T09:30:00Z"
    }
  ],
  "total": 150,
  "limit": 100,
  "offset": 0
}
```

---

### 3.2 이벤트 상세 조회

```http
GET /events/{id}
Authorization: Bearer {token}
```

**Response 200 OK**:
```json
{
  "id": 123,
  "camera_id": 1,
  "camera_name": "Building A - Entrance",
  "timestamp": "2025-01-21T09:30:00Z",
  "frame_number": 108900,
  "vad_score": 0.85,
  "threshold": 0.5,
  "vlm_type": "Fighting",
  "vlm_description": "Two people engaged in physical altercation",
  "vlm_confidence": 0.92,
  "agent_actions": [...],
  "agent_response_time": 0.18,
  "clip_path": "/clips/camera_1_event_123.mp4",
  "acknowledged": false,
  "acknowledged_by": null,
  "acknowledged_at": null,
  "note": null,
  "created_at": "2025-01-21T09:30:00Z"
}
```

---

### 3.3 이벤트 확인

```http
POST /events/{id}/ack
Authorization: Bearer {token}
Content-Type: application/json

{
  "acknowledged": true,
  "note": "Handled by security team. Situation resolved."
}
```

**Response 200 OK**:
```json
{
  "message": "Event acknowledged",
  "event_id": 123,
  "acknowledged": true,
  "acknowledged_by": 1,
  "acknowledged_at": "2025-01-21T10:00:00Z"
}
```

---

## 4. 통계 API

### 4.1 통계 조회

```http
GET /stats
Authorization: Bearer {token}
```

**Query Parameters**:
- `camera_id` (optional): 카메라 ID 필터
- `date` (optional): 날짜 (YYYY-MM-DD, default: today)
- `period` (optional): 기간 (`day`, `week`, `month`, default: `day`)

**Response 200 OK**:
```json
{
  "period": "day",
  "date": "2025-01-21",
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
      "min_vad_score": 0.52,
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
    "total_anomalies": 12,
    "avg_anomaly_rate": 0.0014
  }
}
```

---

### 4.2 통계 트렌드

```http
GET /stats/trends
Authorization: Bearer {token}
```

**Query Parameters**:
- `camera_id` (optional): 카메라 ID 필터
- `days` (optional, default: 7): 조회 기간 (일)

**Response 200 OK**:
```json
{
  "trends": [
    {
      "date": "2025-01-14",
      "anomaly_count": 8,
      "avg_score": 0.65,
      "max_score": 0.92
    },
    {
      "date": "2025-01-15",
      "anomaly_count": 10,
      "avg_score": 0.68,
      "max_score": 0.95
    }
  ]
}
```

---

## 5. WebSocket API

### 5.1 연결

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream/1');
```

### 5.2 클라이언트 → 서버 메시지

**구독 요청**:
```json
{
  "type": "subscribe",
  "camera_id": 1
}
```

**구독 해제**:
```json
{
  "type": "unsubscribe",
  "camera_id": 1
}
```

### 5.3 서버 → 클라이언트 메시지

**프레임 데이터**:
```json
{
  "type": "frame",
  "camera_id": 1,
  "frame": "base64_encoded_image",
  "score": 0.45,
  "timestamp": "2025-01-21T10:00:00Z"
}
```

**이벤트 알림**:
```json
{
  "type": "event",
  "camera_id": 1,
  "event": {
    "id": 123,
    "vad_score": 0.85,
    "vlm_type": "Fighting",
    "vlm_description": "...",
    "agent_actions": [...]
  }
}
```

**통계 업데이트**:
```json
{
  "type": "stats",
  "camera_id": 1,
  "stats": {
    "current_fps": 30.5,
    "anomaly_count": 12,
    "total_frames": 864000
  }
}
```

**에러**:
```json
{
  "type": "error",
  "camera_id": 1,
  "message": "Camera connection lost"
}
```

---

## 6. 에러 응답

### 6.1 표준 에러 형식

```json
{
  "detail": "Error message",
  "code": "ERROR_CODE",
  "timestamp": "2025-01-21T10:00:00Z"
}
```

### 6.2 HTTP 상태 코드

| 코드 | 의미 | 설명 |
|------|------|------|
| 200 | OK | 요청 성공 |
| 201 | Created | 리소스 생성 성공 |
| 400 | Bad Request | 잘못된 요청 |
| 401 | Unauthorized | 인증 실패 |
| 403 | Forbidden | 권한 없음 |
| 404 | Not Found | 리소스 없음 |
| 422 | Unprocessable Entity | 검증 실패 |
| 500 | Internal Server Error | 서버 오류 |

### 6.3 에러 예시

**401 Unauthorized**:
```json
{
  "detail": "Invalid token",
  "code": "AUTH_INVALID_TOKEN"
}
```

**403 Forbidden**:
```json
{
  "detail": "Access denied",
  "code": "PERMISSION_DENIED"
}
```

**404 Not Found**:
```json
{
  "detail": "Camera not found",
  "code": "RESOURCE_NOT_FOUND"
}
```

**422 Unprocessable Entity** (검증 오류):
```json
{
  "detail": [
    {
      "loc": ["body", "vad_threshold"],
      "msg": "value must be between 0.0 and 1.0",
      "type": "value_error"
    }
  ]
}
```

---

## 7. 인증 헤더

모든 API 요청 (인증 제외)에는 다음 헤더가 필요합니다:

```http
Authorization: Bearer {access_token}
```

예시:
```http
GET /api/v1/cameras
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

## 8. Rate Limiting

- **일반 API**: 100 requests/minute per user
- **인증 API**: 10 requests/minute per IP
- **WebSocket**: 연결당 1개

---

## 9. API 버전 관리

현재 버전: `v1`

URL에 버전 포함: `/api/v1/...`

향후 버전 변경 시: `/api/v2/...` (하위 호환성 유지)

---

**문서 버전**: 1.0  
**최종 업데이트**: 2025-01-21
