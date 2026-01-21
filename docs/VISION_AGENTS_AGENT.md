***

# Vision-Agents Technical Report

## PART 2: Agent 클래스 상세 분석

### 2.1 Agent 클래스 개요

**파일**: `agents-core/vision_agents/core/agents/agents.py`  
**크기**: 1428 lines (1203 LOC)  
**메인 클래스**: `Agent`

```python
class Agent:
    """
    Agent는 Vision-Agents의 핵심 클래스입니다.
    
    역할:
    ├─ 모든 플러그인 통합 (LLM, STT, TTS, Processors)
    ├─ 이벤트 관리 (Event Hub)
    ├─ 오디오/비디오 파이프라인 관리
    ├─ 통화 생명 주기 관리
    └─ 상태 유지 및 메트릭 수집
    
    생명 주기:
    1. __init__()        - 초기화
    2. join(call)        - 통화 참여
    3. finish()          - 통화 종료 대기
    4. close()           - 정리
    """
```

***

### 2.2 Agent 초기화 (__init__)

```python
def __init__(
    self,
    # ===== 필수 파라미터 =====
    edge: "StreamEdge",                     # 엣지 네트워크 (GetStream)
    llm: LLM | AudioLLM | VideoLLM,        # 언어 모델
    agent_user: User,                      # 에이전트 유저 정보
    
    # ===== 선택 파라미터 =====
    instructions: str = "Keep replies short",  # 시스템 프롬프트
    
    # 음성 처리 (STT/TTS 모드에서만)
    stt: Optional[STT] = None,             # 음성→텍스트
    tts: Optional[TTS] = None,             # 텍스트→음성
    turn_detection: Optional[TurnDetector] = None,  # 발화 감지
    
    # 확장 기능
    processors: Optional[List[Processor]] = None,   # 비디오/오디오 처리
    mcp_servers: Optional[List[MCPBaseServer]] = None,  # 외부 도구
    
    # 관찰성
    options: Optional[AgentOptions] = None,  # 설정
    tracer: Tracer = trace.get_tracer("agents"),  # OpenTelemetry
    profiler: Optional[Profiler] = None,  # 성능 프로파일링
):
```

#### 2.2.1 초기화 단계 (Step-by-step)

**Step 1: ID 및 기본 정보 설정**
```python
# 1. 에이전트 고유 ID 생성 (UUID4)
self._id = str(uuid4())  # 예: "f47ac10b-58cc-4372-a567-0e02b2c3d479"

# 2. 사용자 정보 설정
self.agent_user = agent_user
if not self.agent_user.id:
    self.agent_user.id = f"agent-{uuid4()}"

# 3. 상태 플래그
self._pending_turn: Optional[LLMTurn] = None      # 현재 LLM 턴
self.call: Optional[Call] = None                  # 현재 통화
self._closed = False                              # 종료 상태
```

**Step 2: 이벤트 시스템 초기화**
```python
# 이벤트 매니저 생성
self.events = EventManager()  # 중앙 이벤트 허브

# 모든 플러그인의 이벤트 등록
self.events.register_events_from_module(getstream.models, "call.")
self.events.register_events_from_module(events)          # Agent Events
self.events.register_events_from_module(sfu_events)      # SFU Events
self.events.register_events_from_module(llm_events)      # LLM Events

# 플러그인 이벤트 병합 (Merge)
for plugin in [stt, tts, turn_detection, llm, edge, profiler]:
    if plugin and hasattr(plugin, "events"):
        self.events.merge(plugin.events)  # 플러그인 이벤트 추가
```

**Step 3: 플러그인 할당**
```python
self.llm = llm                           # LLM 모델
self.stt = stt                           # 음성 인식
self.tts = tts                           # 음성 합성
self.turn_detection = turn_detection     # 발화 감지
self.processors: list[Processor] = processors or []  # 프로세서
self.mcp_servers = mcp_servers or []     # 외부 도구
self.edge = edge                         # 엣지 네트워크
```

**Step 4: 오디오 큐 초기화**
```python
# 들어오는 오디오 데이터 저장
self._incoming_audio_queue: AudioQueue = AudioQueue(
    buffer_limit_ms=8000  # 8초 버퍼 (손실 방지)
)

# 아웃풋 오디오 트랙
self._audio_track: Optional[OutputAudioTrack] = None

# 비디오 트랙 정보
self._active_video_tracks: Dict[str, TrackInfo] = {}
self._video_forwarders: List[VideoForwarder] = []
```

**Step 5: 설정 검증**
```python
def _validate_configuration(self):
    """
    에이전트 설정이 유효한지 확인
    """
    if _is_audio_llm(self.llm):
        # Realtime 모드: STT/TTS 필요 없음
        if self.stt or self.tts:
            self.logger.warning(
                "Realtime 모드 감지: STT/TTS가 무시됩니다"
            )
    else:
        # 일반 모드: LLM 필수
        if self.stt and not self.llm:
            raise ValueError("STT 사용 시 LLM 필수")
```

***

### 2.3 Agent 생명 주기: join()

```python
@asynccontextmanager
async def join(
    self, 
    call: Call,                              # 참여할 통화
    participant_wait_timeout: Optional[float] = 10.0,  # 참여자 대기
) -> AsyncIterator[None]:
    """
    통화에 에이전트가 참여하는 컨텍스트 매니저
    
    사용법:
    async with agent.join(call):
        await agent.finish()  # 통화 종료 대기
    # 자동으로 agent.close() 호출
    """
```

#### 2.3.1 join() 상세 단계

**Step 1: 중복 참여 확인**
```python
if self._call_ended_event is not None:
    raise RuntimeError("에이전트는 한 번만 참여할 수 있습니다")
```

**Step 2: 통화 정보 설정**
```python
self.call = call
self._start_tracing(call)  # OpenTelemetry 시작

# 로깅 컨텍스트 설정
self._set_call_logging_context(call.id)
```

**Step 3: 플러그인 시작**
```python
# 모든 플러그인의 start() 메서드 호출
await self._apply("start")

# 사용자 생성 (엣지에 등록)
await self.create_user()
```

**Step 4: MCP 서버 연결**
```python
if self.mcp_manager:
    await self.mcp_manager.connect_all()  # 외부 도구 연결
```

**Step 5: Realtime LLM 준비**
```python
if _is_realtime_llm(self.llm):
    await self.llm.connect()  # Gemini/OpenAI Realtime 준비
```

**Step 6: 엣지에 참여**
```python
self._connection = await self.edge.join(self, call)
self.logger.info(f"🤖 에이전트 참여: {call.id}")
```

**Step 7: 미디어 트랙 발행**
```python
# 오디오/비디오 트랙 생성 및 발행
audio_track = self._audio_track if self.publish_audio else None
video_track = self._video_track if self.publish_video else None

if audio_track or video_track:
    await self.edge.publish_tracks(audio_track, video_track)
```

**Step 8: 채팅 컨텍스트 생성**
```python
# LLM이 참조할 대화 히스토리 생성
self.conversation = await self.edge.create_conversation(
    call, 
    self.agent_user, 
    self.instructions.full_reference
)

# LLM에 컨텍스트 제공
self.llm.set_conversation(self.conversation)
```

**Step 9: 참여자 대기**
```python
if participant_wait_timeout != 0:
    await self.wait_for_participant(timeout=participant_wait_timeout)
    # 기본 10초 대기
```

**Step 10: 오디오 처리 시작**
```python
# 메인 오디오 처리 루프 시작
self._audio_consumer_task = asyncio.create_task(
    self._consume_incoming_audio()
)

# 통화 종료 신호 설정
self._call_ended_event = asyncio.Event()
self._joined_at = time.time()
```

**Step 11: 컨텍스트 양보**
```python
yield  # 여기서 with 블록 내 코드 실행
```

**Step 12: 정리**
```python
except Exception as exc:
    if self._closing or self._closed:
        logger.warning("에이전트 종료 중...")
    else:
        raise

finally:
    # 통화 종료 시 자동 정리
    await self.close()
    self._end_tracing()
    self._join_lock.release()
```

***

### 2.4 핵심 메서드: _consume_incoming_audio()

**메서드 위치**: Line 1260-1320  
**목적**: 들어오는 오디오를 20ms 간격으로 처리

```python
async def _consume_incoming_audio(self) -> None:
    """
    오디오 소비 루프 (Main Processing Loop)
    
    특징:
    ✓ 20ms 간격 (50 FPS)
    ✓ 비차단 (async/await)
    ✓ 8초 버퍼 (손실 방지)
    """
    interval_seconds = 0.02  # 20ms
    
    while self._call_ended_event and not self._call_ended_event.is_set():
        loop_start = time.perf_counter()
        
        try:
            # 1️⃣ 오디오 데이터 획득
            pcm = await asyncio.wait_for(
                self._incoming_audio_queue.get_duration(duration_ms=20),
                timeout=1.0,
            )
            
            participant = pcm.participant
            
            # 2️⃣ 에이전트 자신의 음성 제외
            if (participant and 
                getattr(participant, "user_id", None) != self.agent_user.id):
                
                # 3️⃣ 오디오 프로세서 (음성 감정 분석 등)
                for processor in self.audio_processors:
                    if processor is None:
                        continue
                    await processor.process_audio(pcm)
                
                # 4️⃣ Realtime LLM 모드
                if _is_audio_llm(self.llm):
                    await self.simple_audio_response(pcm, participant)
                
                # 5️⃣ 일반 STT 모드
                elif self.stt:
                    await self.stt.process_audio(pcm, participant)
                
                # 6️⃣ 턴 감지 (발화 끝 감지)
                if self.turn_detection is not None and participant is not None:
                    await self.turn_detection.process_audio(
                        pcm, participant, conversation=self.conversation
                    )
        
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            # 오디오 없음 - 계속
            pass
        
        # 7️⃣ 정확한 20ms 간격 유지
        elapsed = time.perf_counter() - loop_start
        sleep_time = interval_seconds - elapsed
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
```

**흐름도:**
```
┌─ while 루프 시작 (20ms 간격)
├─ PCM 데이터 대기 (20ms 청크)
│  │
│  ├─ 에이전트 자신? → 제외
│  │
│  ├─1️⃣ Audio Processors
│  │   (음성 감정/품질 분석)
│  │
│  ├─2️⃣ Realtime LLM?
│  │   예: simple_audio_response()
│  │   아니오: 다음으로
│  │
│  ├─3️⃣ STT 실행
│  │   (Deepgram STT)
│  │
│  ├─4️⃣ Turn Detection
│  │   (Vogent/SmartTurn)
│  │   발화 끝? → TurnEndedEvent 발행
│  │
│  └─ 정확한 20ms 슬립
└─ 반복
```

***

# Vision-Agents Technical Report

## PART 2-5: 이벤트 처리 설정 (Event Handling)

### 2.5 이벤트 처리 설정 (setup_event_handling) - 상세 분석

**메서드 위치**: Line 119-210  
**목적**: 모든 플러그인의 이벤트를 에이전트의 통합 이벤트 버스에 연결

```python
def setup_event_handling(self):
    """
    이벤트 처리 설정
    
    역할:
    1. 모든 플러그인 이벤트를 구독
    2. 이벤트 간 의존성 연결
    3. 콜백 함수 등록
    
    특징:
    ✓ 디커플링된 아키텍처
    ✓ 비동기 처리 (@async)
    ✓ 자동 에러 처리
    """
    
    # 1️⃣ 턴 감지 이벤트 구독
    self.events.subscribe(self._on_turn_event)
```

#### 2.5.1 턴 감지 이벤트 구독

```python
@self.events.subscribe
async def _on_turn_event(self, event: TurnStartedEvent | TurnEndedEvent) -> None:
    """
    발화 시작/종료 감지
    
    TurnStartedEvent:
    ├─ 사용자가 말하기 시작
    ├─ TTS 중단 (Barge-in)
    └─ 새 턴 준비
    
    TurnEndedEvent:
    ├─ 사용자가 말하기 종료
    ├─ 부분 텍스트 수집 완료
    └─ LLM 응답 트리거
    """
    
    # Realtime LLM 모드는 자체 처리
    if _is_audio_llm(self.llm):
        return
    
    if isinstance(event, TurnStartedEvent):
        # ❌ 사용자가 말하기 시작 → TTS 중단
        if event.participant and event.participant.user_id != self.agent_user.id:
            if self.tts:
                await self.tts.stop_audio()  # 음성 중단 (Barge-in)
                
    elif isinstance(event, TurnEndedEvent):
        # ✅ 사용자가 말하기 종료 → LLM 응답 준비
        
        # 에이전트 자신은 제외
        if not event.participant or event.participant.user_id == self.agent_user.id:
            return
        
        # 부분 텍스트 버퍼에서 최종 텍스트 추출
        buffer = self._pending_user_transcripts[event.participant.user_id]
        
        # STT가 따라잡기 위해 잠시 대기
        if not event.eager_end_of_turn:
            if self.stt:
                await self.stt.clear()
            await asyncio.sleep(0.02)  # 20ms 대기
        
        # 최종 텍스트 획득
        transcript = buffer.text
        
        if not event.eager_end_of_turn:
            buffer.reset()
        
        # 텍스트가 있으면 LLM 호출
        if transcript.strip():
            # 새 LLM 턴 생성
            if self._pending_turn is None or self._pending_turn.input != transcript:
                llm_turn = LLMTurn(
                    input=transcript,
                    participant=event.participant,
                    started_at=datetime.datetime.now(),
                    turn_finished=not event.eager_end_of_turn,
                )
                self._pending_turn = llm_turn
                
                # LLM 비동기 호출
                task = asyncio.create_task(
                    self.simple_response(transcript, event.participant)
                )
                llm_turn.task = task
```

**시각적 흐름:**
```
사용자 음성
    │
    ▼
STT: "안녕하세요"
    │
    ├─ TurnStartedEvent
    │  └─ TTS 중단 ❌
    │
    └─ (지속 음성 인식)
    
사용자 음성 종료
    │
    ▼
TurnEndedEvent
    │
    ├─ 최종 텍스트: "안녕하세요"
    ├─ LLM.simple_response() 호출
    │  └─ OpenAI/Gemini/Claude API
    │
    └─ 기다리는 중...
```

***

#### 2.5.2 LLM 응답 완료 이벤트

```python
@self.llm.events.subscribe
async def on_llm_response_send_to_tts(event: LLMResponseCompletedEvent):
    """
    LLM 응답 완료 → TTS로 음성 합성
    
    상황 1: 외부 호출 (agent.say())
    ├─ self._pending_turn이 None
    ├─ TTS에 직접 전달
    └─ 예: agent.say("도움이 되셨나요?")
    
    상황 2: 사용자 입력에 대한 응답
    ├─ self._pending_turn 존재
    ├─ 턴 정보 저장
    └─ 턴 완료 여부 확인
    """
    
    if self._pending_turn is None:
        # 외부 호출 (agent.say())
        if self.tts and event.text and event.text.strip():
            sanitized_text = self._sanitize_text(event.text)
            await self.tts.send(sanitized_text)
    else:
        # 사용자 입력 응답
        self._pending_turn.response = event
        
        if self._pending_turn.turn_finished:
            # 턴 완료 → TTS 발행
            await self._finish_llm_turn()
        else:
            # Eager 모드 - 확인 대기
            pass
```

**코드 흐름:**
```python
# 상황 1: 외부 호출
agent.say("도움이 되셨나요?")
    │
    ├─ AgentSayEvent 발행
    │
    └─ _on_agent_say() 호출
       └─ TTS.send() 호출
          └─ TTSAudioEvent 발행
             └─ OutputAudioTrack에 기록

# 상황 2: 사용자 응답
사용자: "안녕하세요"
    │
    ├─ TurnEndedEvent
    │  └─ simple_response() 호출
    │
    ├─ LLM API 호출
    │  └─ "안녕하세요! 무엇을 도와드릴까요?"
    │
    └─ LLMResponseCompletedEvent
       └─ TTS.send() 호출
          └─ 음성 합성
```

***

#### 2.5.3 TTS 오디오 출력 트랙 기록

```python
@self.events.subscribe
async def _on_tts_audio_write_to_output(event: TTSAudioEvent):
    """
    TTS 합성 오디오 → 출력 트랙에 기록
    
    처리:
    1. TTS가 오디오 청크 생성
    2. OutputAudioTrack에 기록
    3. WebRTC → 클라이언트에 전달
    """
    if self._audio_track is not None:
        # PCM 오디오 데이터를 출력 트랙에 기록
        await self._audio_track.write(event.data)
```

**오디오 흐름:**
```
TTSAudioEvent (PCM 청크)
    │
    ├─ 8kHz, 16비트, 모노
    ├─ 20ms 청크 (~160 샘플)
    │
    └─ _audio_track.write(pcm)
       │
       ├─ 내부 버퍼에 저장
       ├─ WebRTC 인코딩
       │
       └─ 클라이언트로 전송 (RTP)
```

***

#### 2.5.4 비디오 트랙 추가/제거

```python
@self.edge.events.subscribe
async def on_video_track_added(event: TrackAddedEvent | TrackRemovedEvent):
    """
    비디오 트랙 추가/제거 감지
    
    TrackAddedEvent:
    ├─ 원격 참여자의 비디오 추가
    ├─ 우선순위: ScreenShare > Camera
    └─ 비디오 프로세서에 연결
    
    TrackRemovedEvent:
    ├─ 원격 참여자 나감
    └─ 리소스 정리
    """
    
    if event.track_id is None or event.track_type is None:
        return
    
    if isinstance(event, TrackRemovedEvent):
        asyncio.create_task(
            self._on_track_removed(event.track_id, event.track_type, event.user)
        )
    else:
        asyncio.create_task(
            self._on_track_added(event.track_id, event.track_type, event.user)
        )
```

***

#### 2.5.5 오디오 수신 이벤트

```python
@self.edge.events.subscribe
async def on_audio_received(event: AudioReceivedEvent):
    """
    원격 참여자의 오디오 수신
    
    처리:
    1. PCM 데이터 추출
    2. 오디오 큐에 저장
    3. _consume_incoming_audio()에서 처리
    """
    if event.pcm_data is None:
        return
    
    # 8초 버퍼에 저장 (손실 방지)
    await self._incoming_audio_queue.put(event.pcm_data)
```

***

#### 2.5.6 통화 종료 이벤트

```python
@self.edge.events.subscribe
async def on_call_ended(event: CallEndedEvent):
    """
    통화 종료 감지
    
    처리:
    1. 종료 신호 설정
    2. 정리 작업 수행
    3. 모든 리소스 해제
    """
    if self._call_ended_event is not None:
        self._call_ended_event.set()  # 종료 신호
    
    await self.close()  # 정리
```

***

### 2.6 STT 이벤트 처리

#### 2.6.1 STT 트랜스크립트 이벤트

```python
@self.events.subscribe
async def on_stt_transcript_event_create_response(
    event: STTTranscriptEvent | STTPartialTranscriptEvent,
):
    """
    STT 결과 처리
    
    STTPartialTranscriptEvent: "안녕하"
    STTTranscriptEvent:        "안녕하세요"
    
    처리:
    1. Realtime LLM 모드 확인
    2. 부분/최종 텍스트 누적
    3. 턴 완료 신호 대기 또는 트리거
    """
    
    # Realtime LLM은 자체 처리
    if _is_audio_llm(self.llm):
        return
    
    user_id = event.user_id()
    
    if isinstance(event, STTPartialTranscriptEvent):
        self.logger.info(f"🎤 [부분]: {event.text}")
    else:
        self.logger.info(f"🎤 [완료]: {event.text}")
    
    # 사용자별 버퍼에 저장
    self._pending_user_transcripts[user_id].update(event)
    
    # 턴 감지 없으면 즉시 트리거
    if not self.turn_detection_enabled and isinstance(
        event, STTTranscriptEvent
    ):
        self.events.send(
            TurnEndedEvent(
                participant=event.participant,
            )
        )
```

**상태 다이어그램:**
```
STT 인식 진행
    │
    ├─ "안녕"         ← STTPartialTranscriptEvent
    ├─ "안녕하"       ← STTPartialTranscriptEvent
    ├─ "안녕하세"     ← STTPartialTranscriptEvent
    └─ "안녕하세요"   ← STTTranscriptEvent (최종)
       │
       ├─ 턴 감지 활성화?
       │  예: TurnEndedEvent 대기
       │  아니오: TurnEndedEvent 생성 → 즉시 LLM
       │
       └─ LLM 호출
```

***

### 2.7 LLM 응답 동기화

#### 2.7.1 대화 히스토리에 기록

```python
@self.llm.events.subscribe
async def on_llm_response_sync_conversation(event: LLMResponseCompletedEvent):
    """
    LLM 응답을 채팅 히스토리에 저장
    
    목적:
    1. 대화 컨텍스트 유지
    2. 다음 LLM 호출에 활용
    3. 감사 로그 생성
    """
    
    if event.text:
        self.logger.info(f"🤖 [LLM]: {event.text}")
    
    if self.conversation is None:
        return
    
    await self.conversation.upsert_message(
        message_id=event.item_id,
        role="assistant",
        user_id=self.agent_user.id or "agent",
        content=event.text or "",
        completed=True,
        replace=True,  # 부분 응답 덮어쓰기
    )
```

***

# Vision-Agents Technical Report

## PART 2-8: 실시간 모드 이벤트 (Realtime LLM Events)

### 2.8 Realtime 모드 전용 이벤트 처리

Realtime 모드 (Gemini Live, OpenAI Realtime)는 음성/영상을 직접 처리하므로 STT/TTS가 없습니다.

#### 2.8.1 사용자 음성 전사 이벤트

```python
@self.events.subscribe
async def on_realtime_user_speech_transcription(
    event: RealtimeUserSpeechTranscriptionEvent,
):
    """
    Realtime LLM이 사용자 음성을 인식하고 텍스트로 변환
    
    특징:
    ├─ LLM이 직접 처리 (STT 없음)
    ├─ 자동 음성 인식
    ├─ 부분 → 최종 전사
    └─ 채팅 히스토리에 저장
    
    예시:
    사용자: "날씨가 어떻게 되나요?"
         ↓
    LLM: "날씨가 어떻게 되나요?" (전사)
         ↓
    채팅에 기록
    """
    
    self.logger.info(f"🎤 [사용자 음성]: {event.text}")
    
    if self.conversation is None or not event.text:
        return
    
    if user_id := event.user_id():
        with self.span("agent.on_realtime_user_speech_transcription"):
            # 채팅 컨텍스트에 사용자 메시지 저장
            await self.conversation.upsert_message(
                message_id=str(uuid.uuid4()),
                role="user",                    # 사용자 메시지
                user_id=user_id,
                content=event.text,
                completed=True,
                replace=True,  # 부분 인식 덮어쓰기
                original=event,
            )
    else:
        self.logger.info(
            "사용자 ID가 없어 채팅에 기록하지 않음"
        )
```

**처리 흐름:**
```
사용자 음성
    │
    ▼
Gemini/OpenAI Realtime
    │
    ├─ 음성 처리 (자체 STT)
    │
    ├─ "날씨"        ← 부분 (자동 무시)
    ├─ "날씨가"      ← 부분 (자동 무시)
    └─ "날씨가 어떻게 되나요?"  ← 최종
       │
       ▼
    RealtimeUserSpeechTranscriptionEvent
       │
       ├─ 로깅
       ├─ 채팅 저장
       └─ LLM이 이미 이해함 (다음 응답 준비)
```

***

#### 2.8.2 에이전트 음성 전사 이벤트

```python
@self.events.subscribe
async def on_realtime_agent_speech_transcription(
    event: RealtimeAgentSpeechTranscriptionEvent,
):
    """
    Realtime LLM이 생성한 음성을 텍스트로 변환
    
    특징:
    ├─ LLM의 음성 합성 결과
    ├─ 실시간으로 생성됨
    ├─ 채팅 히스토리에 저장
    └─ 사용자에게 이미 재생 중
    
    예시:
    LLM이 생각 중...
         ↓
    "현재 서울 날씨는 맑고 영하 2도입니다"
         ↓
    음성으로 재생 중
         ↓
    텍스트로 저장
    """
    
    self.logger.info(f"🤖 [에이전트 음성]: {event.text}")
    
    if self.conversation is None or not event.text:
        return
    
    with self.span("agent.on_realtime_agent_speech_transcription"):
        # 채팅 컨텍스트에 에이전트 메시지 저장
        await self.conversation.upsert_message(
            message_id=str(uuid.uuid4()),
            role="assistant",                  # 에이전트 메시지
            user_id=self.agent_user.id or "",
            content=event.text,
            completed=True,
            replace=True,  # 부분 응답 덮어쓰기
            original=event,
        )
```

**Realtime LLM과의 대화:**
```
┌─ 사용자: "안녕하세요"
│
├─ LLM 처리 (내부)
│  ├─ 사용자 음성 인식
│  └─ 응답 생성 중
│
├─ RealtimeUserSpeechTranscriptionEvent
│  └─ "안녕하세요" 저장
│
├─ LLM 응답 생성 시작
│  └─ 음성 합성 시작
│
├─ RealtimeAgentSpeechTranscriptionEvent (스트리밍)
│  ├─ "안녕"      (부분)
│  ├─ "안녕하세"  (부분)
│  └─ "안녕하세요! 무엇을 도와드릴까요?"  (최종)
│     └─ 저장 (replace=True로 덮어쓰기)
│
└─ 사용자에게 음성으로 재생 중
```

***

#### 2.8.3 Realtime 오디오 출력 이벤트

```python
@self.events.subscribe
async def forward_audio(event: RealtimeAudioOutputEvent):
    """
    Realtime LLM이 생성한 오디오 → 출력 트랙
    
    처리:
    1. LLM이 음성 합성
    2. PCM 청크 생성
    3. OutputAudioTrack에 기록
    4. WebRTC → 클라이언트
    """
    if self._audio_track is not None:
        await self._audio_track.write(event.data)
```

**오디오 데이터 흐름:**
```
LLM: "안녕하세요"
    │
    ├─ 음성 합성 (TTS)
    │
    └─ PCM 청크 생성
       ├─ 청크 1: 안녕 (20ms)
       ├─ 청크 2: 하세 (20ms)
       └─ 청크 3: 요   (20ms)
          │
          ▼
       RealtimeAudioOutputEvent 발행
          │
          ├─ forward_audio() 호출
          │
          ├─ _audio_track.write(pcm)
          │
          ├─ WebRTC 인코딩
          │
          └─ 클라이언트로 전송
             └─ 사용자 스피커에서 재생
```

***

### 2.9 LLM 응답 스트리밍 처리

#### 2.9.1 LLM 응답 청크 이벤트

```python
@self.llm.events.subscribe
async def _handle_output_text_delta(event: LLMResponseChunkEvent):
    """
    LLM이 스트리밍 응답을 생성하는 동안
    부분 텍스트 청크를 받음
    
    특징:
    ├─ 실시간 스트리밍
    ├─ 최종 완성되기 전
    ├─ 채팅 업데이트
    └─ 사용자가 응답을 보는 중
    
    예시:
    사용자: "파이썬 설명해줘"
    
    청크 1: "파이썬은"
    청크 2: "파이썬은 고급"
    청크 3: "파이썬은 고급 프로그래밍 언어입니다"
    """
    
    if self.conversation is None:
        return
    
    with self.span("agent._handle_output_text_delta"):
        # 부분 응답을 채팅에 업데이트
        await self.conversation.upsert_message(
            message_id=event.item_id,
            role="assistant",
            user_id=self.agent_user.id or "agent",
            content=event.delta or "",          # 부분 텍스트
            content_index=event.content_index,
            completed=False,                    # 아직 진행 중
        )
```

**스트리밍 시각화:**
```
LLM 응답 생성 중...

시점 1 (0ms):
채팅: [Assistant] 파이썬

시점 2 (50ms):
채팅: [Assistant] 파이썬은

시점 3 (100ms):
채팅: [Assistant] 파이썬은 고급

시점 4 (150ms):
채팅: [Assistant] 파이썬은 고급 프로그래밍 언어입니다
                    ↑
              완료됨 (completed=True)
```

***

### 2.10 에러 처리

#### 2.10.1 STT 에러 이벤트

```python
@self.events.subscribe
async def on_error(event: STTErrorEvent):
    """
    STT 처리 중 발생한 에러 처리
    
    일반적인 에러:
    ├─ 네트워크 실패
    ├─ 음성 품질 낮음
    ├─ 타임아웃
    └─ API 오류
    """
    self.logger.error("STT 에러 발생: %s", event)
    
    # 에러 복구:
    # 1. 재시도 로직
    # 2. 사용자에게 알림
    # 3. 폴백 처리
```

***

### 2.11 VideoProcessor 이벤트

#### 2.11.1 비디오 트랙 처리

```python
async def _track_to_video_processors(self, track: TrackInfo):
    """
    비디오 트랙 → 프로세서 파이프라인
    
    처리:
    1. YOLO (객체/자세 감지)
    2. Roboflow (커스텀 감지)
    3. Moondream VLM (영상 이해)
    4. 결과 수집 → LLM에 전달
    """
    
    for processor in self.video_processors:
        try:
            user_id = track.participant.user_id if track.participant else None
            
            # 비디오 프로세서에 트랙 전달
            await processor.process_video(
                track.track,                    # 비디오 스트림
                user_id,
                shared_forwarder=track.forwarder  # 버퍼 공유
            )
        
        except Exception as e:
            self.logger.error(
                f"비디오 프로세서 에러 ({type(processor).__name__}): {e}"
            )
```

**비디오 처리 파이프라인:**
```
원격 참여자 비디오
    │
    ▼
VideoForwarder (30 FPS 버퍼)
    │
    ├─ YOLO 프로세서
    │  ├─ 사람 감지
    │  ├─ 자세 감지
    │  └─ {x: 100, y: 150, class: "person"} 반환
    │
    ├─ Roboflow 프로세서
    │  ├─ 패키지 감지
    │  └─ {id: 1, confidence: 0.95} 반환
    │
    └─ Moondream VLM
       ├─ 영상 이해
       └─ "사람이 패키지를 집어올리는 중" 반환
          │
          ▼
       LLM에 상태 정보 전달
          │
          └─ simple_response()에 포함
```

***

### 2.12 이벤트 체인 요약

**완전한 이벤트 체인:**

```
사용자 입력 (음성/비디오)
    │
    ├─ 🎤 AudioReceivedEvent
    │  └─ _incoming_audio_queue에 저장
    │
    ├─ _consume_incoming_audio()
    │  │
    │  ├─1️⃣ AudioProcessor
    │  │
    │  ├─2️⃣ STT.process_audio()
    │  │   └─ STTTranscriptEvent 발행
    │  │
    │  └─3️⃣ TurnDetection
    │      └─ TurnEndedEvent 발행
    │
    ├─ _on_turn_event()
    │  └─ simple_response(text, participant) 호출
    │
    ├─ 🤖 LLM API 호출
    │  ├─ OpenAI/Gemini/Claude
    │  └─ 프로세서 상태 포함
    │
    ├─ LLMResponseCompletedEvent 수신
    │  │
    │  ├─ on_llm_response_send_to_tts()
    │  │  └─ TTS.send(text) 호출
    │  │
    │  └─ on_llm_response_sync_conversation()
    │     └─ 채팅에 저장
    │
    ├─ TTSAudioEvent 수신
    │  └─ _on_tts_audio_write_to_output()
    │     └─ _audio_track.write(pcm)
    │
    └─ 🔊 클라이언트가 음성 재생
```

***

