# E2E 보안 모니터링 시스템 아키텍처

**버전**: 1.0  
**최종 업데이트**: 2025-12-12

---

## 1. 시스템 개요

### 1.1 목적
Video Anomaly Detection (VAD)과 Agentic AI를 통합하여 실시간으로 CCTV 영상에서 이상 상황을 탐지하고 자동으로 대응하는 지능형 보안 모니터링 시스템

### 1.2 주요 기능
- **실시간 이상 탐지**: VAD 모델을 통한 프레임 단위 이상 점수 계산
- **상황 분석**: VLM을 통한 이상 상황 유형 분류 및 설명
- **자동 대응**: Agent 시스템을 통한 대응 계획 수립 및 실행
- **시각화**: 실시간 대시보드 (Web UI / CLI)
- **로깅**: 이벤트 기록 및 클립 저장

---

## 2. 시스템 아키텍처

### 2.1 전체 구조

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE LAYER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────┐ │
│  │   Web UI (Streamlit)│    │   CLI (Rich)        │    │   REST API      │ │
│  │   - 실시간 영상      │    │   - 터미널 대시보드  │    │   (확장 예정)    │ │
│  │   - 점수 차트        │    │   - ASCII 히스토그램 │    │                 │ │
│  │   - 이벤트 목록      │    │   - 로그 스트림      │    │                 │ │
│  │   - 설정 패널        │    │   - 통계 표시        │    │                 │ │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           E2E SYSTEM ENGINE                                  │
│                        (app/e2e_system.py)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         VIDEO INPUT LAYER                                ││
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 ││
│  │  │ Video File  │    │ RTSP Stream │    │   Webcam    │                 ││
│  │  │  (.mp4/.avi)│    │             │    │  (Device 0) │                 ││
│  │  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                 ││
│  │         └──────────────────┼──────────────────┘                        ││
│  │                            ▼                                            ││
│  │                   ┌─────────────────┐                                  ││
│  │                   │  VideoSource    │                                  ││
│  │                   │  - Open/Read    │                                  ││
│  │                   │  - FPS Control  │                                  ││
│  │                   │  - Resize       │                                  ││
│  │                   └────────┬────────┘                                  ││
│  └────────────────────────────┼────────────────────────────────────────────┘│
│                               ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      PROCESSING PIPELINE                                 ││
│  │                                                                          ││
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 ││
│  │  │ Frame Buffer│───▶│  VAD Model  │───▶│Score Check  │                 ││
│  │  │ (16 frames) │    │  (MNAD)     │    │(threshold)  │                 ││
│  │  └─────────────┘    │  - predict()│    └──────┬──────┘                 ││
│  │                     │  - 3.77ms   │           │                        ││
│  │                     └─────────────┘           │                        ││
│  │                                               │                        ││
│  │                     score >= threshold ───────┴───────┐                ││
│  │                              │                        │                ││
│  │                              ▼                        ▼                ││
│  │                     ┌─────────────┐          ┌─────────────┐          ││
│  │                     │ Clip Saver  │          │ Continue    │          ││
│  │                     │ (3초 저장)   │          │ Processing  │          ││
│  │                     └──────┬──────┘          └─────────────┘          ││
│  │                            │                                           ││
│  │                            ▼                                           ││
│  │                     ┌─────────────┐                                    ││
│  │                     │VLM Analyzer │                                    ││
│  │                     │(Qwen2.5-VL) │                                    ││
│  │                     │- 분류       │                                    ││
│  │                     │- 설명       │                                    ││
│  │                     │- ~5초       │                                    ││
│  │                     └──────┬──────┘                                    ││
│  │                            │                                           ││
│  │                            ▼                                           ││
│  │                     ┌─────────────┐                                    ││
│  │                     │Agent System │                                    ││
│  │                     │- Sequential │                                    ││
│  │                     │- Hierarchical│                                   ││
│  │                     │- Collaborative│                                  ││
│  │                     └──────┬──────┘                                    ││
│  │                            │                                           ││
│  │                            ▼                                           ││
│  │                     ┌─────────────┐                                    ││
│  │                     │  Response   │                                    ││
│  │                     │  Actions    │                                    ││
│  │                     └─────────────┘                                    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         OUTPUT LAYER                                     ││
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 ││
│  │  │ Event Logger│    │ Clip Storage│    │  Statistics │                 ││
│  │  │ (JSON)      │    │ (MP4)       │    │  (Metrics)  │                 ││
│  │  └─────────────┘    └─────────────┘    └─────────────┘                 ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 컴포넌트 상세

#### 2.2.1 Video Input Layer

| 컴포넌트 | 설명 | 지원 형식 |
|---------|------|----------|
| VideoSource | 비디오 입력 관리 | MP4, AVI, RTSP, Webcam |
| Frame Buffer | 시퀀스 분석용 프레임 버퍼 | 최근 16프레임 |
| Resize | 해상도 정규화 | 640x480 기본 |

#### 2.2.2 VAD Model Layer

| 모델 | AUC (%) | FPS | 메모리 | 용도 |
|------|---------|-----|--------|------|
| **MNAD** | 82.4 | 265 | 1GB | **실시간 (권장)** |
| MULDE | 89.66 | 45 | 2GB | 정확도 우선 |
| MemAE | 78.5 | 180 | 512MB | 경량화 |
| STAE | 75.2 | 320 | 256MB | 초경량 |

#### 2.2.3 VLM Analyzer

```
Input: 프레임 (8프레임 그리드)
       ↓
┌─────────────────────────────────────┐
│         Qwen2.5-VL-7B               │
│  - Vision: 이미지 인코딩            │
│  - Language: 상황 분석              │
└─────────────────────────────────────┘
       ↓
Output:
  - detected_type: Fighting, Arson, etc.
  - description: 상황 설명
  - actions: 권장 대응
```

#### 2.2.4 Agent System

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AGENT FLOW TYPES                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [Sequential Flow] - 가장 빠름 (5.2초)                              │
│  VideoAnalysis → Planner → Actor                                    │
│                                                                      │
│  [Hierarchical Flow] - 최고 품질 (8.5초)                            │
│  VideoAnalysis → Supervisor → Planner → Supervisor → Actor          │
│                                                                      │
│  [Collaborative Flow] - 균형 (6.8초)                                │
│  VideoAnalysis → [Planner1 + Planner2] → Aggregator → Actor        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 데이터 흐름

### 3.1 정상 상황

```
Frame → VAD (score: 0.3) → Continue Processing
                           └→ UI 업데이트 (녹색 표시)
```

### 3.2 이상 상황

```
Frame → VAD (score: 0.7) → Anomaly Detected!
                           │
                           ├→ Clip Saver (3초 저장)
                           │
                           ├→ VLM Analyzer
                           │   └→ "Fighting" 감지
                           │
                           ├→ Agent System
                           │   └→ [보안요원 출동, 경찰 신고]
                           │
                           ├→ Event Logger
                           │   └→ events_YYYYMMDD.json
                           │
                           └→ UI 업데이트 (빨간색 경고)
```

---

## 4. 파일 구조

```
app/
├── __init__.py          # 패키지 초기화
├── e2e_system.py        # 핵심 E2E 엔진 (900줄)
│   ├── SystemConfig     # 설정 데이터클래스
│   ├── VideoSource      # 비디오 입력
│   ├── VADWrapper       # VAD 모델 래퍼
│   ├── VLMWrapper       # VLM 래퍼
│   ├── AgentWrapper     # Agent 래퍼
│   ├── EventLogger      # 이벤트 로깅
│   ├── ClipSaver        # 클립 저장
│   └── E2ESystem        # 메인 시스템 클래스
├── cli_ui.py            # CLI 대시보드 (350줄)
│   └── CLIDashboard     # Rich 기반 UI
├── web_ui.py            # Web UI (400줄)
│   └── Streamlit App    # 웹 대시보드
├── run.py               # 메인 실행 스크립트 (250줄)
└── config.yaml          # 설정 파일
```

---

## 5. API 명세

### 5.1 E2ESystem

```python
class E2ESystem:
    def __init__(self, config: SystemConfig)
    def initialize(self) -> bool
    def start(self)
    def stop(self)
    def get_stats(self) -> Dict
    def get_recent_events(n: int) -> List[AnomalyEvent]
    def get_current_frame() -> np.ndarray
    def get_current_score() -> float
    
    # 콜백
    on_frame_callback: Callable[[frame, score], None]
    on_anomaly_callback: Callable[[AnomalyEvent], None]
    on_stats_callback: Callable[[SystemStats], None]
```

### 5.2 SystemConfig

```python
@dataclass
class SystemConfig:
    # 비디오 소스
    source_type: VideoSourceType  # FILE, RTSP, WEBCAM
    source_path: str
    
    # VAD 설정
    vad_model: VADModelType  # MNAD, MULDE, MEMAE, STAE
    vad_threshold: float = 0.5
    
    # VLM 설정
    enable_vlm: bool = True
    vlm_n_frames: int = 4
    optimize_vlm: bool = True
    
    # Agent 설정
    enable_agent: bool = True
    agent_flow: AgentFlowType  # SEQUENTIAL, HIERARCHICAL, COLLABORATIVE
    
    # 저장 설정
    save_clips: bool = True
    clip_duration: float = 3.0
    
    # GPU 설정
    gpu_id: int = 2
```

---

## 6. 성능 지표

### 6.1 레이턴시 분석

| 컴포넌트 | 시간 | 비율 |
|---------|------|------|
| VAD 추론 | 3.77ms | 0.07% |
| VLM 분석 | ~5,000ms | 95.7% |
| Agent 처리 | ~200ms | 3.8% |
| 기타 | ~20ms | 0.4% |
| **총 레이턴시** | **~5,223ms** | 100% |

### 6.2 리소스 사용

| 구성 | GPU 메모리 | CPU |
|------|-----------|-----|
| VAD Only | 1GB | 10% |
| VAD + VLM | 9GB | 30% |
| VAD + VLM + Agent | 12GB | 40% |

### 6.3 처리량

| 모드 | 처리 FPS | 실시간 여부 |
|------|---------|-----------|
| VAD Only | 265 FPS | ✅ Yes |
| VAD + VLM (Async) | 30 FPS | ✅ Yes |
| VAD + VLM + Agent | 0.19 FPS | ❌ No (이벤트 기반) |

---

## 7. 배포 구성

### 7.1 권장 하드웨어

| 구성 | 최소 사양 | 권장 사양 |
|------|---------|----------|
| GPU | RTX 3080 (10GB) | RTX 4090 (24GB) |
| CPU | 8코어 | 16코어+ |
| RAM | 16GB | 32GB |
| 저장장치 | SSD 256GB | NVMe 1TB |

### 7.2 실행 명령

```bash
# CLI 모드
python app/run.py --mode cli --source /path/to/video.mp4

# Web 모드
python app/run.py --mode web --source rtsp://camera_ip/stream

# 전체 옵션
python app/run.py \
    --mode cli \
    --source /path/to/video.mp4 \
    --vad-model mnad \
    --threshold 0.5 \
    --agent-flow sequential \
    --gpu 2
```

---

## 8. 확장 계획

### 8.1 단기 (1-2주)
- [ ] VLM 정확도 개선 (프롬프트 최적화)
- [ ] 실시간 RTSP 안정성 강화
- [ ] 메모리 최적화

### 8.2 중기 (1개월)
- [ ] REST API 추가
- [ ] 멀티 카메라 지원
- [ ] 알림 시스템 (이메일/웹훅)

### 8.3 장기 (3개월)
- [ ] 분산 처리
- [ ] 모델 파인튜닝
- [ ] 모바일 앱

---

*이 문서는 시스템 구현 상태를 반영하여 지속적으로 업데이트됩니다.*

