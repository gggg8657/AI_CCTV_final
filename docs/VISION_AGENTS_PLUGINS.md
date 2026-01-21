***

# Vision-Agents Technical Report

## PART 3: 플러그인 시스템 (Plugin Architecture)

### 3.1 플러그인 시스템 개요

Vision-Agents는 **23개 이상의 플러그인**을 통해 다양한 LLM, STT, TTS, 프로세서를 지원합니다.

```
플러그인 특징:
✓ 모듈식: 필요한 것만 설치
✓ 독립적: 각 플러그인은 독립 패키지
✓ 확장 가능: 커스텀 플러그인 개발 가능
✓ 호환성: 동일 인터페이스 제공
```

***

### 3.2 LLM 플러그인 분류

#### 3.2.1 Realtime LLM (음성/비디오 직접 처리)

**Gemini Live (Google)**
```python
from vision_agents.plugins import gemini

# 모드 1: Realtime (음성 직접 처리)
llm = gemini.Realtime(
    model="gemini-live-2.5-flash-preview",
    fps=10,  # 초당 프레임 수 (비디오)
)

# 모드 2: LLM (텍스트만)
llm = gemini.LLM("gemini-2.5-flash-lite")
```

**특징:**
```
✓ 음성 직접 스트리밍 (STT/TTS 불필요)
✓ 비디오 직접 스트리밍
✓ 자동 턴 감지 (내장)
✓ 지연시간: <100ms
✗ 비용: 높음 (스트리밍당 가격)
```

**OpenAI Realtime**
```python
from vision_agents.plugins import openai

# Realtime API (gpt-4o-realtime-preview)
llm = openai.Realtime(
    model="gpt-4o-realtime-preview"
)

# 특징
✓ 매우 빠른 응답
✓ 멀티모달 (음성+비디오)
✗ 알파 버전 (변경 가능)
```

***

#### 3.2.2 표준 LLM (텍스트 기반)

**OpenAI**
```python
from vision_agents.plugins import openai

# 표준 API
llm = openai.LLM("gpt-4o")

# ChatCompletions API (OSS 모델용)
llm = openai.ChatCompletionsVLM(
    model="qwen3vl",
    base_url="https://api.baseten.co/v1"
)

# Function Calling 지원
@llm.register_function(
    name="get_weather",
    description="날씨 조회"
)
def get_weather(city: str) -> str:
    return "맑음"
```

**Gemini (표준 API)**
```python
from vision_agents.plugins import gemini

llm = gemini.LLM(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
)

# 함수 호출
@llm.register_function(
    description="사용자 정보 조회"
)
def get_user_info(user_id: str) -> dict:
    return {"name": "John", "age": 30}
```

**Claude (Anthropic)**
```python
from vision_agents.plugins import anthropic

llm = anthropic.LLM(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
)
```

**xAI (Grok)**
```python
from vision_agents.plugins import xai

llm = xai.LLM(
    model="grok-2",  # 최신 모델
)

# 장점:
✓ 최신 지식 (Real-time)
✓ 강력한 추론
✗ 비용: 높음
```

**OpenRouter (다중 선택)**
```python
from vision_agents.plugins import openrouter

# 여러 모델 중 선택
llm = openrouter.LLM(
    model="anthropic/claude-3.5-sonnet",
    # 또는
    # model="openai/gpt-4o",
    # model="google/gemini-2-flash",
)

# 장점:
✓ 다양한 모델 비교
✓ 가격 최적화
```

***

### 3.3 STT 플러그인 (음성 인식)

#### 3.3.1 Deepgram (권장)

```python
from vision_agents.plugins import deepgram

stt = deepgram.STT(
    model="flux-general-en",
    eager_turn_detection=True,  # ⭐ 빠른 턴 감지
)

# 특징:
✓ 낮은 지연시간 (<100ms)
✓ 높은 정확도 (Flux 모델)
✓ 내장 턴 감지
✓ 한국어 지원
✓ 화자 식별 (Speaker Diarization)
```

**사용 시나리오:**
```python
agent = Agent(
    llm=openai.LLM("gpt-4o"),
    stt=deepgram.STT(eager_turn_detection=True),
    tts=elevenlabs.TTS(),
)

# eager_turn_detection=True:
# 사용자가 말을 멈추자마자 즉시 LLM 호출
# 지연시간: 낮음 (50-100ms)
```

#### 3.3.2 Fast-Whisper (로컬)

```python
from vision_agents.plugins import fast_whisper

stt = fast_whisper.STT(
    model_size="small",  # tiny, base, small, medium
    device="cuda",       # GPU 사용
    language="ko",       # 한국어
)

# 특징:
✓ 로컬 실행 (오프라인)
✓ 데이터 프라이버시
✓ 무료
✗ 지연시간: 길음 (200-500ms)
✗ GPU 필수 (CPU는 너무 느림)
```

#### 3.3.3 Fish Audio (언어 자동 감지)

```python
from vision_agents.plugins import fish

stt = fish.STT(
    language="auto",  # 자동 감지
)

# 특징:
✓ 자동 언어 감지
✓ 다국어 지원 (100+)
✓ 빠른 처리
```

***

### 3.4 TTS 플러그인 (음성 합성)

#### 3.4.1 ElevenLabs (권장)

```python
from vision_agents.plugins import elevenlabs

tts = elevenlabs.TTS(
    model_id="eleven_flash_v2_5",  # 가장 빠름
    voice_id="21m00Tcm4TlvDq8ikWAM",  # 특정 음성
)

# 음성 옵션:
elevenlabs.TTS(
    model_id="eleven_flash_v2_5",   # 빠름, 저비용
    # model_id="eleven_turbo_v2_5", # 매우 빠름
)

# 특징:
✓ 가장 자연스러운 음성
✓ 감정 표현 가능
✓ 낮은 지연시간 (<200ms)
✓ 한국어 지원
```

#### 3.4.2 Deepgram Aura (저비용)

```python
from vision_agents.plugins import deepgram

tts = deepgram.TTS(
    model="aura-2-thalia-en",  # 여성 음성
    # model="aura-2-orion-en",  # 남성 음성
    sample_rate=16000,
)

# 특징:
✓ 저비용
✓ 빠른 처리
✗ ElevenLabs만큼 자연스럽지는 않음
```

#### 3.4.3 Kokoro (로컬, 무료)

```python
from vision_agents.plugins import kokoro

tts = kokoro.TTS(
    voice="af",  # 여성
    # voice="am",  # 남성
    device="cuda",  # GPU
)

# 특징:
✓ 완전 무료
✓ 로컬 실행
✓ 빠른 처리
✗ 음질: ElevenLabs보다 낮음
```

#### 3.4.4 Cartesia (감정 표현)

```python
from vision_agents.plugins import cartesia

tts = cartesia.TTS(
    voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",
    emotion="happy",  # 감정 제어
)

# 특징:
✓ 감정 표현 가능
✓ 자연스러운 음성
```

***

### 3.5 Video Processor 플러그인

#### 3.5.1 YOLO (Ultralytics)

```python
from vision_agents.plugins import ultralytics

# 객체 감지
processor = ultralytics.YOLOObjectProcessor(
    model_path="yolov11n.pt",  # 나노 모델 (빠름)
    device="cuda",
    confidence=0.5,
)

# 포즈 감지
processor = ultralytics.YOLOPoseProcessor(
    model_path="yolo11n-pose.pt",
    device="cuda",
)

# 특징:
✓ 매우 빠름 (30+ FPS)
✓ 낮은 메모리 사용
✓ 정확도: 우수
✓ 실시간 처리 가능

# 출력:
# {
#     "people": [...],
#     "packages": [...],
#     "keypoints": [...]
# }
```

#### 3.5.2 Roboflow (커스텀 모델)

```python
from vision_agents.plugins import roboflow

processor = roboflow.RoboflowProcessor(
    project_id="my-project",
    version=2,
    confidence=0.7,
)

# 특징:
✓ 커스텀 모델 학습 가능
✓ 자동 데이터 라벨링
✓ 호스팅 포함
```

#### 3.5.3 Moondream (Vision Language Model)

```python
from vision_agents.plugins import moondream

processor = moondream.MoondreamProcessor(
    model_id="moondream2",
    device="cuda",
)

# 사용법:
# processor가 프레임을 분석하고
# "사람이 패키지를 들고 있음"처럼 설명

# 특징:
✓ 영상 이해 (VLM)
✓ 자연어 설명
✓ 정교한 분석
✗ 느림 (1-2 FPS)
```

#### 3.5.4 NVIDIA Cosmos (비디오 이해)

```python
from vision_agents.plugins import nvidia

processor = nvidia.CosmosProcessor(
    model="cosmos-1.0",
    resolution="high",
)

# 특징:
✓ 비디오 시퀀스 이해
✓ 행동 인식
✓ 장면 분석
```

***

### 3.6 Turn Detection 플러그인

#### 3.6.1 Vogent (신경망 기반)

```python
from vision_agents.plugins import vogent

turn_detection = vogent.TurnDetection(
    threshold=0.5,
)

# 특징:
✓ 신경망 모델 (정확함)
✓ 자연스러운 대화
✗ 약간의 지연시간 추가
```

#### 3.6.2 SmartTurn (고급)

```python
from vision_agents.plugins import smart_turn

turn_detection = smart_turn.SmartTurnDetection(
    mode="aggressive",  # 빠른 턴 감지
    # mode="normal",    # 균형
    # mode="conservative",  # 안정적
)

# 특징:
✓ 여러 모드 지원
✓ Silero VAD 기반
```

#### 3.6.3 Deepgram Eager (최빠름)

```python
from vision_agents.plugins import deepgram

stt = deepgram.STT(
    eager_turn_detection=True,  # STT 내장
)

# 특징:
✓ 가장 빠름 (<50ms)
✓ 추가 지연시간 없음
✓ STT와 통합
```

***

### 3.7 다른 플러그인들

#### 3.7.1 RAG (검색증강생성)

**TurboPuffer (벡터 DB)**
```python
from vision_agents.plugins import turbopuffer

processor = turbopuffer.TurboPufferRAG(
    index_name="customer_data",
    vector_size=1536,
    hybrid=True,  # 벡터 + BM25
)

# 사용 사례:
# 고객이 "주문 상태"라고 물으면
# 1. TurboPuffer에서 고객 주문 검색
# 2. LLM에 컨텍스트 제공
# 3. 개인화된 응답 생성
```

#### 3.7.2 Phone Integration

**Twilio**
```python
from vision_agents.plugins import twilio

# 전화 통합
processor = twilio.TwilioPhoneProcessor(
    account_sid="...",
    auth_token="...",
)

# 사용 사례:
# 콜센터 봇 - 전화로 상담
# 음성 입력 → LLM 처리 → 음성 출력
```

#### 3.7.3 Avatar (시각적 응답)

**HeyGen (인터랙티브 아바타)**
```python
from vision_agents.plugins import heygen

processor = heygen.HeyGenAvatarProcessor(
    avatar_id="Avatar_b622de28_fullbody",  # 아바타 ID
    voice="en-male",                        # 음성
)

agent = Agent(
    edge=getstream.Edge(),
    llm=openai.LLM("gpt-4o"),
    processors=[processor],  # 비디오 발행자
    agent_user=User(name="AI Assistant", id="bot"),
)

# 사용 사례:
# LLM 응답
# → HeyGen 텍스트 전달
# → 아바타가 제스처/표정 포함해서 말함
# → 비디오 트랙 전송
# → 클라이언트가 아바타 보면서 대화

# 특징:
✓ 시각적 인터랙션
✓ 자연스러운 제스처
✓ 표정 변화
✓ 영어/한국어 지원

# 비용:
✗ API 호출당 비용 발생
✗ 지연시간: 더 있음 (합성 시간)

**아바타 타입:**
```
- Fullbody: 전신 표현 (더 비용)
- Halfbody: 상반신만 (기본)
- Talking Head: 얼굴만 (저비용)
```

**Inworld (대화형 NPC)**
```python
from vision_agents.plugins import inworld

processor = inworld.InworldProcessor(
    character_id="...",
    api_key="...",
)

# 특징:
✓ RPG 스타일 NPC
✓ 복잡한 성격/개성
✓ 인물 관계 추적
✗ 복잡한 설정 필요
```

#### 3.7.4 Advanced Language Processing

**AWS Bedrock (멀티 모델)**
```python
from vision_agents.plugins import aws

# Speech-to-Speech (Amazon Nova)
processor = aws.BedrockS2SProcessor(
    model_id="amazon.nova-pro-v1:0",
    region="us-east-1",
)

# 특징:
✓ AWS 완전 통합
✓ 여러 모델 지원
✓ On-premises 배포 가능
✗ AWS 복잡성
```

**Qwen (중국 LLM)**
```python
from vision_agents.plugins import qwen

llm = qwen.LLM(
    model="qwen-turbo",
    api_key="...",
)

# 특징:
✓ 중국 시장 최적화
✓ 저비용
✓ 한국어도 지원
```

#### 3.7.5 Advanced STT/TTS

**Fish Audio (멀티언어)**
```python
from vision_agents.plugins import fish

# STT
stt = fish.STT(
    language="auto",  # 자동 감지
    multilingual=True,
)

# TTS
tts = fish.TTS(
    voice="female",
    emotion="neutral",
)

# 특징:
✓ 100+ 언어 지원
✓ 자동 언어 감지
✓ 음성 클로닝 가능
```

**Wizper (번역 STT)**
```python
from vision_agents.plugins import wizper

stt = wizper.STT(
    source_language="ko",
    target_language="en",  # 자동 번역
)

# 특징:
✓ 실시간 번역
✓ Whisper v3 기반
✗ 지연시간 추가
```

### 3.8 플러그인 선택 가이드

#### 시나리오별 권장 조합

**시나리오 1: 저 지연시간 (가장 빠름)**

```python
Agent(
    edge=getstream.Edge(),
    llm=gemini.Realtime(fps=10),      # 가장 빠름
    processors=[],                     # 프로세싱 없음
)

# 특징:
✓ 지연시간: 400-500ms
✗ 비용: 높음
```

**시나리오 2: 가성비 최고 (권장)**

```python
Agent(
    edge=getstream.Edge(),
    llm=openai.LLM("gpt-4o-mini"),     # 저비용
    stt=deepgram.STT(eager_turn_detection=True),
    tts=elevenlabs.TTS(model_id="eleven_flash_v2_5"),
)

# 특징:
✓ 지연시간: 600-800ms
✓ 비용: 보통
✓ 품질: 우수
```

**시나리오 3: 완전 무료 (오프라인)**

```python
Agent(
    edge=getstream.Edge(),
    llm=openai.ChatCompletionsLLM(
        model="llama2",
        base_url="http://localhost:8000"  # Ollama
    ),
    stt=fast_whisper.STT(device="cuda"),
    tts=kokoro.TTS(device="cuda"),
    processors=[ultralytics.YOLOObjectProcessor()],
)

# 특징:
✓ 비용: 0원
✓ 데이터 프라이버시: 100%
✗ 지연시간: 1-2초
✗ GPU 필수
```

**시나리오 4: 최고 품질 (비용 고려 X)**

```python
Agent(
    edge=getstream.Edge(),
    llm=openai.Realtime("gpt-4o-realtime-preview"),
    processors=[
        ultralytics.YOLOPoseProcessor(),
        moondream.MoondreamProcessor(),
    ],
    tts=elevenlabs.TTS(),
)

# 특징:
✓ 품질: 최고
✓ 지연시간: 400-600ms
✗ 비용: 매우 높음
```

**시나리오 5: 보안 카메라 (당신의 프로젝트)**

```python
Agent(
    edge=getstream.Edge(),
    llm=gemini.LLM("gemini-2.5-flash"),
    processors=[
        ultralytics.YOLOObjectProcessor(
            model_path="yolov11n.pt"
        ),
        # Custom SecurityCameraProcessor
    ],
    stt=None,  # 음성 불필요
    tts=None,
)

# 특징:
✓ 24/7 모니터링
✓ 저비용
✓ 커스텀 로직 포함
```

### 3.9 플러그인 설치 및 사용

#### 설치 방법

```bash
# 전체 플러그인 설치
pip install "vision-agents[all-plugins]"

# 선택적 설치
pip install vision-agents

# 개별 플러그인 설치
pip install vision-agents-plugins-deepgram
pip install vision-agents-plugins-openai
pip install vision-agents-plugins-gemini

# uv 사용 (권장)
uv add "vision-agents[deepgram,openai,gemini]"
```

#### 환경변수 설정

```bash
# .env 파일
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export DEEPGRAM_API_KEY="..."
export ELEVENLABS_API_KEY="..."

# 또는 Python에서
import os
os.environ["OPENAI_API_KEY"] = "..."
```

### 3.10 플러그인 구현 원리

#### 플러그인 인터페이스

**LLM 플러그인:**

```python
from vision_agents.core.llm import LLM

class CustomLLM(LLM):
    async def create_response(
        self,
        input: List[Dict],  # 메시지 히스토리
        processors: List[Processor],  # 상태 데이터
        participant: Optional[Participant] = None,
    ) -> str:
        """LLM 응답 생성"""
        # 프로세서 상태 포함
        processor_state = {
            p.name(): p.state() for p in processors
        }
        
        # OpenAI API 호출
        response = await openai_api_call(
            messages=input,
            system_state=processor_state,
        )
        
        return response.text
    
    @property
    def events(self):
        return self._event_manager
```

**STT 플러그인:**

```python
from vision_agents.core.stt import STT

class CustomSTT(STT):
    async def process_audio(
        self,
        pcm: PcmData,
        participant: Optional[Participant] = None,
    ) -> None:
        """오디오 처리"""
        
        # Deepgram API 호출
        result = await deepgram_api_call(pcm.data)
        
        # 부분 결과
        for partial in result.alternatives:
            self.events.send(
                STTPartialTranscriptEvent(
                    text=partial.transcript,
                    participant=participant,
                )
            )
        
        # 최종 결과
        self.events.send(
            STTTranscriptEvent(
                text=result.transcript,
                participant=participant,
            )
        )
```

**Processor 플러그인:**

```python
from vision_agents.core.processors import VideoProcessor

class CustomProcessor(VideoProcessor):
    async def process_video(
        self,
        track: MediaStreamTrack,
        user_id: str,
        shared_forwarder: VideoForwarder,
    ) -> None:
        """비디오 처리"""
        
        async for frame in shared_forwarder:
            # YOLO 추론
            results = self.model(frame)
            
            # 상태 업데이트
            self.state = {
                "detections": results.to_dict(),
                "timestamp": time.time(),
            }
            
            # 이벤트 발행 (선택)
            self.events.send(
                DetectionEvent(
                    results=results,
                )
            )
    
    def state(self) -> Dict:
        """LLM에 전달할 상태"""
        return self.state
