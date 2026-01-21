# 설계 결정 사항

**작성일**: 2025-01-21  
**하드웨어**: GPU 1060 (6GB VRAM), 온프레미스 시스템

---

## 1. 이벤트 저장 방식

### 1.1 배치 저장 주기
**결정**: **비동기 배치 저장**
- **버퍼 크기**: 10개 이벤트 또는 1초마다 (둘 중 먼저 도달하는 것)
- **저장 방식**: 비동기 (백그라운드 스레드)
- **이유**: 
  - GPU 1060은 메모리가 제한적 (6GB)
  - 동기 저장 시 VAD/VLM 처리 지연 발생 가능
  - 비동기로 처리하면 메인 파이프라인 블로킹 없음

### 1.2 구현 방식
```python
# 이벤트 발생 시
EventLogger.log_event(event)
  → 메모리 버퍼에 추가 (동기, 빠름)
  → 버퍼가 10개 도달 또는 1초 경과 시
  → 백그라운드 스레드에서 DB 저장 (비동기)
```

---

## 2. 멀티 카메라 구조

### 2.1 구조
**결정**: MultiCameraManager가 E2ESystem 인스턴스들을 관리

```
MultiCameraManager
  ├─ CameraPipeline (카메라 1)
  │   └─ E2ESystem 인스턴스
  ├─ CameraPipeline (카메라 2)
  │   └─ E2ESystem 인스턴스
  └─ ...
```

### 2.2 리소스 풀 (Resource Pool) 설명

**리소스 풀**이란: 여러 카메라가 공유할 수 있는 리소스(모델)를 중앙에서 관리하는 패턴

#### 문제 상황
- 카메라 16개가 각각 VAD 모델을 로드하면?
  - MNAD 모델: 약 1GB VRAM
  - 16개 × 1GB = 16GB 필요 (GPU 1060은 6GB만 있음) ❌

#### 리소스 풀 해결책
- VAD 모델 1개만 로드하고, 모든 카메라가 공유
- VLM 분석기 1개만 로드하고, 모든 카메라가 공유
- Agent Flow도 공유 가능

```
ResourcePool
  ├─ VAD Models (공유)
  │   ├─ MNAD: 1개 인스턴스 (모든 카메라가 사용)
  │   └─ MULDE: 1개 인스턴스 (필요시)
  ├─ VLM Analyzer: 1개 인스턴스 (공유)
  └─ Agent Flows: 1개 인스턴스 (공유)
```

#### 장점
- **메모리 절약**: 16GB → 1GB (16배 절약)
- **로딩 시간 단축**: 모델을 한 번만 로드

#### 단점 및 해결
- **동시성 문제**: 여러 카메라가 동시에 모델 사용 시 충돌
  - **해결**: 스레드 안전한 락(lock) 사용
  - VAD는 빠르므로(3.77ms) 대기 시간 미미

#### 구현 예시
```python
class ResourcePool:
    def __init__(self):
        self.vad_models = {}  # 모델 타입별로 1개씩만 저장
        self.vlm_analyzer = None  # 1개만
        self.lock = threading.Lock()  # 동시 접근 방지
    
    def get_vad_model(self, model_type: str):
        """모델 가져오기 (없으면 생성, 있으면 공유)"""
        if model_type not in self.vad_models:
            with self.lock:
                if model_type not in self.vad_models:
                    self.vad_models[model_type] = create_vad_model(model_type)
        return self.vad_models[model_type]
```

**결정**: ✅ 리소스 풀 사용 (메모리 절약 필수)

---

## 3. API 서버 구조

### 3.1 FastAPI Dependency
**결정**: ✅ FastAPI Dependency로 E2ESystem 관리

### 3.2 프로세스 분리 전략
**결정**: 레이턴시가 생기는 부분만 분리

#### 레이턴시 분석
| 컴포넌트 | 레이턴시 | 블로킹 여부 |
|---------|---------|------------|
| VAD 추론 | 3.77ms | ✅ 블로킹 (하지만 빠름) |
| VLM 분석 | ~5초 | ❌ 블로킹 (느림!) |
| Agent 처리 | ~200ms | ❌ 블로킹 (중간) |
| DB 저장 | ~10ms | ❌ 블로킹 (비동기로 처리) |

#### 분리 전략
```
메인 프로세스 (FastAPI + E2ESystem)
  ├─ VAD 추론: 동기 (빠르므로)
  ├─ VLM 분석: 비동기 큐 → 별도 워커 프로세스
  ├─ Agent 처리: 비동기 큐 → 별도 워커 프로세스
  └─ DB 저장: 비동기 큐 → 백그라운드 스레드
```

**구현 방식**:
- VLM/Agent는 **메시지 큐**(Redis/RabbitMQ) 또는 **멀티프로세싱 큐** 사용
- 메인 프로세스는 VAD만 처리하고 즉시 응답
- VLM/Agent는 별도 워커에서 처리 후 결과를 DB에 저장

**결정**: 
- **Phase 1**: 일단 단일 프로세스로 구현 (간단)
- **Phase 2**: VLM/Agent만 별도 프로세스로 분리 (성능 최적화)

---

## 4. WebSocket 설정

### 4.1 기본 FPS
**결정**: ✅ 5 FPS (초당 5프레임)

### 4.2 구독 기반 설명

**구독 기반**이란: 클라이언트가 원하는 카메라만 선택해서 실시간 데이터를 받는 방식

#### 브로드캐스트 vs 구독 기반

**브로드캐스트 (Broadcast)**:
```
모든 클라이언트에게 모든 카메라 데이터 전송
- 클라이언트 A: 카메라 1, 2, 3, 4, ... 모두 받음
- 클라이언트 B: 카메라 1, 2, 3, 4, ... 모두 받음
- 문제: 클라이언트가 관심 없는 카메라 데이터도 받음 (낭비)
```

**구독 기반 (Subscribe)**:
```
클라이언트가 원하는 카메라만 선택해서 받음
- 클라이언트 A: 카메라 1, 2만 구독 → 카메라 1, 2 데이터만 받음
- 클라이언트 B: 카메라 3, 4만 구독 → 카메라 3, 4 데이터만 받음
- 장점: 네트워크 대역폭 절약, 서버 부하 감소
```

#### 구현 예시
```javascript
// 클라이언트가 구독 요청
ws.send({
  type: "subscribe",
  camera_ids: [1, 2]  // 카메라 1, 2만 구독
});

// 서버는 구독한 카메라 데이터만 전송
// 카메라 1, 2의 프레임만 클라이언트에게 전송
```

**결정**: 일단 비워두고, 나중에 필요하면 추가

---

## 5. 알림 시스템

### 5.1 우선순위 기반 발송
**결정**: ✅ 우선순위 기반
- **high/critical**: 즉시 발송
- **medium/low**: 배치 발송 (1분마다)

### 5.2 중복 방지 시간 윈도우
**결정**: ✅ **15초**

#### 동작 방식
```
이벤트 발생 → 알림 규칙 매칭 → 15초 내 같은 카메라+유형 확인
  ├─ 15초 내 동일 알림 있음 → 발송 안 함 (중복 방지)
  └─ 15초 내 동일 알림 없음 → 발송
```

---

## 6. GPU 1060 (6GB) 최적화 전략

### 6.1 메모리 관리
- **VAD 모델**: 1개만 로드 (공유) → ~1GB
- **VLM 모델**: 1개만 로드 → ~4GB
- **Agent 모델**: 1개만 로드 → ~1GB
- **총 사용량**: ~6GB (한계에 근접)

### 6.2 최적화
- **모델 양자화**: VLM 모델을 Q4_K_M으로 사용 (이미 적용됨)
- **동적 로딩**: 사용하지 않는 모델은 언로드
- **배치 크기 최소화**: VLM 분석 시 프레임 수 최소화

### 6.3 동시 처리 카메라 수
- **이론적 최대**: 16개 (리소스 풀 사용 시)
- **실제 권장**: 8-12개 (안정성 고려)

---

## 7. 배치 저장 최적화 (GPU 1060 기준)

### 7.1 결정 사항
- **버퍼 크기**: 10개 또는 1초 (둘 중 먼저)
- **저장 방식**: 비동기 (백그라운드 스레드)
- **이유**: 
  - GPU 1060은 메모리가 제한적
  - DB 저장이 VAD/VLM 처리에 영향을 주지 않도록

### 7.2 구현
```python
class AsyncEventLogger:
    def __init__(self):
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.last_save_time = time.time()
        self.save_thread = threading.Thread(target=self._save_loop, daemon=True)
        self.save_thread.start()
    
    def log_event(self, event):
        """이벤트 버퍼에 추가 (비동기)"""
        with self.buffer_lock:
            self.buffer.append(event)
    
    def _save_loop(self):
        """백그라운드에서 주기적으로 저장"""
        while True:
            time.sleep(1)  # 1초마다 체크
            now = time.time()
            
            with self.buffer_lock:
                if len(self.buffer) >= 10 or (now - self.last_save_time) >= 1.0:
                    events_to_save = self.buffer[:10]
                    self.buffer = self.buffer[10:]
                    self.last_save_time = now
                    
                    # DB 저장 (비동기)
                    self._save_to_db(events_to_save)
```

---

## 📋 최종 결정 사항 요약

| 항목 | 결정 사항 |
|------|----------|
| 이벤트 저장 | 비동기 배치 (10개 또는 1초) |
| 멀티 카메라 | MultiCameraManager + E2ESystem 인스턴스 |
| 리소스 풀 | ✅ 사용 (모델 공유) |
| API 서버 | FastAPI Dependency |
| 프로세스 분리 | Phase 1: 단일 프로세스, Phase 2: VLM/Agent 분리 |
| WebSocket FPS | 5 FPS |
| 구독 기반 | 일단 비워둠 (나중에 추가) |
| 알림 우선순위 | high/critical: 즉시, medium/low: 배치 |
| 알림 중복 방지 | 15초 윈도우 |
| 최대 카메라 수 | 이론적 16개, 권장 8-12개 |

---

**다음 단계**: 위 결정 사항을 바탕으로 구현 시작!
