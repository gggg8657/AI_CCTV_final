좋습니다. PART 4 예제 코드 분석을 다시 시작하겠습니다.

***

# Vision-Agents Technical Report

## PART 4: 예제 코드 상세 분석

### 4.1 Simple Agent Example (기본 예제)

**파일**: `examples/01_simple_agent_example/simple_agent_example.py`  
**목표**: 음성 입력 → LLM 처리 → 음성 출력의 기본 흐름

#### 4.1.1 에이전트 생성 함수

```python
async def create_agent(**kwargs) -> Agent:
    """
    에이전트 생성 및 구성
    
    핵심 개념:
    1. LLM 선택: gemini-2.5-flash-lite (빠르고 저비용)
    2. STT 선택: Deepgram with eager_turn_detection
    3. TTS 선택: ElevenLabs flash 모델 (빠름)
    4. 함수 등록: weather API
    """
    
    # Step 1️⃣: LLM 초기화
    llm = gemini.LLM("gemini-2.5-flash-lite")
    
    # Step 2️⃣: Agent 객체 생성
    agent = Agent(
        # 엣지 네트워크 (저 지연시간)
        edge=getstream.Edge(),
        
        # 에이전트 사용자 정보
        agent_user=User(
            name="My happy AI friend",
            id="agent"
        ),
        
        # 시스템 프롬프트
        instructions=(
            "You're a voice AI assistant. "
            "Keep responses short and conversational. "
            "Don't use special characters or formatting."
        ),
        
        # 프로세서 (비디오/오디오 처리)
        processors=[],  # 이 예제에서는 없음
        
        # LLM 모델
        llm=llm,
        
        # TTS 설정
        tts=elevenlabs.TTS(
            model_id="eleven_flash_v2_5"  # 가장 빠른 모델
        ),
        
        # STT 설정 (가장 중요)
        stt=deepgram.STT(
            model="flux-general-en",           # 기본 모델
            eager_turn_detection=True          # ⭐ 핵심 설정
        ),
        
        # 턴 감지 (선택)
        # turn_detection=vogent.TurnDetection()  # 생략 (STT 내장)
    )
    
    # Step 3️⃣: 함수 등록 (Function Calling)
    @llm.register_function(
        description="Get current weather for a location"
    )
    async def get_weather(location: str) -> Dict[str, Any]:
        """
        날씨 정보 조회
        
        LLM이 필요할 때 자동으로 호출됨:
        사용자: "서울 날씨 어때?"
        LLM: get_weather("서울") 호출
        결과: "현재 서울은 맑고 영하 2도입니다"
        """
        return await get_weather_by_location(location)
    
    # Step 4️⃣: 에이전트 반환
    return agent
```

**설정 상세 설명:**

```
eager_turn_detection=True의 의미:

일반 모드:
├─ 사용자: "안녕하세요"
├─ STT 인식: "안녕하세요"
├─ 침묵 감지 (500ms 대기)
└─ LLM 호출 (총 600ms)

Eager 모드:
├─ 사용자: "안녕하세요"
├─ STT 완료: "안녕하세요"
├─ 즉시 LLM 호출
└─ 응답 (총 300ms) ✅ 더 빠름

단점: 토큰 사용량 증가 (비용 ↑)
```

***

#### 4.1.2 통화 참여 함수

```python
async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """
    통화 참여 및 실행
    
    단계:
    1. 통화 객체 생성
    2. 에이전트 참여
    3. 초기 응답 수행
    4. 통화 종료 대기
    """
    
    # Step 1️⃣: 통화 생성
    call = await agent.create_call(call_type, call_id)
    """
    create_call() 수행:
    - call_type: "default" (기본값)
    - call_id: 고유한 통화 ID
    - created_by_id: 에이전트 ID
    """
    
    # Step 2️⃣: 에이전트 참여 (컨텍스트 매니저 사용)
    async with agent.join(call):
        """
        join() 수행:
        ├─ edge.join() - 엣지 네트워크 연결
        ├─ create_user() - 사용자 생성
        ├─ publish_tracks() - 미디어 발행
        ├─ create_conversation() - 채팅 컨텍스트
        ├─ _consume_incoming_audio() - 오디오 루프 시작
        └─ yield - with 블록 실행
        """
        
        # Step 3️⃣: 초기 응답 수행
        await agent.simple_response(
            "tell me something interesting in a short sentence"
        )
        """
        simple_response() 수행:
        1. LLM에 텍스트 전달
        2. LLM 응답 대기
        3. TTS로 음성 변환
        4. OutputAudioTrack에 기록
        5. 클라이언트가 음성 수신
        """
        
        # Step 4️⃣: 통화 종료 대기
        await agent.finish()
        """
        finish() 수행:
        - _call_ended_event 대기
        - 사용자가 나갈 때까지 대기
        - 통화 종료 시 반환
        """
    
    # Step 5️⃣: 자동 정리 (with 블록 종료)
    # agent.close() 자동 호출
    # - STT/TTS 종료
    # - 엣지 연결 해제
    # - 모든 리소스 해제
```

**실행 흐름 타이밍:**

```
시간    | 사용자          | 에이전트        | 백엔드
───────────────────────────────────────────────────────
0ms    | 통화 시작       |                |
100ms  | 웹캠 연결       | join() 시작     |
200ms  |                | 엣지 참여       | edge.join()
250ms  |                |                | create_conversation()
300ms  |                | 준비 완료 ✓     |
350ms  |                | simple_response()|  LLM API
450ms  |                |                | gemini.generate()
500ms  |                |                | "안녕하세요..."
600ms  |                | TTS 합성        | elevenlabs.synthesize()
700ms  |                | 🔊 음성 재생    | OutputAudioTrack
750ms  | 🎤 음성 입력    | 인식 중...      | STT.process_audio()
800ms  | (계속 말함)     |                | deepgram 처리
850ms  | 음성 종료       | STT 완료        | "뭔가 신기한..."
900ms  |                | LLM 호출        | LLM API
1000ms |                |                | gemini.generate()
1100ms |                | TTS 합성        | elevenlabs.synthesize()
1200ms |                | 🔊 응답 음성    | OutputAudioTrack
```

***

#### 4.1.3 Main 진입점

```python
if __name__ == "__main__":
    # Runner 사용 (자동 관리)
    Runner(AgentLauncher(
        create_agent=create_agent,
        join_call=join_call
    )).cli()
```

**Runner의 역할:**
```
Runner(AgentLauncher)
├─ CLI 인터페이스 제공
├─ 통화 관리 (생성/종료)
├─ 에러 처리
├─ 로깅
└─ HTTP 서버 제공 (웹 SDK 연결용)
```

***

### 4.2 Golf Coach Example (고급 예제)

**파일**: `examples/02_golf_coach_example/golf_coach_example.py`  
**목표**: 실시간 포즈 감지 + Gemini Live로 코칭

#### 4.2.1 구조 차이

```
Simple Example vs Golf Coach

Simple:
├─ 음성만 처리
├─ LLM 텍스트 기반
└─ processors=[]

Golf Coach:
├─ 비디오 처리 (YOLO 포즈)
├─ Gemini Realtime (음성+비디오)
├─ 프레임별 분석
└─ 실시간 피드백
```

#### 4.2.2 에이전트 구성

```python
async def create_agent(**kwargs) -> Agent:
    """
    골프 코칭 에이전트
    
    핵심: Realtime LLM + YOLO 포즈 감지
    """
    
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=agent_user,
        
        # 중요: 골프 코칭 지시사항
        instructions="Read @golf_coach.md",
        
        # 🔴 핵심: Realtime LLM (음성+비디오 직접)
        llm=gemini.Realtime(fps=10),  # 10 FPS 비디오 분석
        
        # 🔴 핵심: YOLO 포즈 감지
        processors=[
            ultralytics.YOLOPoseProcessor(
                model_path="yolo11n-pose.pt",
                device="cuda"
            )
        ],
        
        # Realtime 모드에서는 STT/TTS 불필요
        stt=None,
        tts=None,
    )
    
    return agent
```

**데이터 흐름:**

```
웹캠 입력
    │
    ├─ VideoForwarder (30 FPS 버퍼)
    │
    ├─ YOLOPoseProcessor
    │  ├─ 신체 위치 감지
    │  └─ {keypoints: [[x,y,confidence], ...]}
    │
    └─ Gemini Realtime
       ├─ 비디오 프레임 (10 FPS)
       ├─ 포즈 데이터
       ├─ "왼팔이 너무 굽혀있어요"
       └─ 🔊 음성 피드백 (실시간)
```

***

### 4.3 Security Camera Example (당신의 프로젝트)

**파일**: `examples/05_security_camera_example/security_camera_example.py`  
**목표**: 24/7 보안 감시 + AI 분석

#### 4.3.1 핵심 구성요소

```python
async def create_agent(**kwargs) -> Agent:
    """
    보안 카메라 에이전트
    
    기능:
    1. 얼굴 인식 (30분 추적)
    2. 패키지 감지
    3. 도난 감지
    4. 원포스터 생성
    5. X 자동 게시
    """
    
    llm = gemini.LLM("gemini-2.5-flash-lite")
    
    # 보안 카메라 프로세서 생성
    security_processor = SecurityCameraProcessor(
        fps=5,                          # 5 FPS
        time_window=1800,               # 30분 윈도우
        thumbnail_size=80,              # 80x80 썸네일
        detection_interval=2.0,         # 2초마다 감지
        bbox_update_interval=0.3,       # 0.3초마다 박스 업데이트
        model_path="weights_custom.pt", # YOLOv11 커스텀
        package_conf_threshold=0.7,     # 패키지 신뢰도 70%
        max_tracked_packages=1,         # 한 번에 1개 패키지만
    )
    
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Security AI", id="agent"),
        instructions="Read @instructions.md",
        
        # 프로세서 (맞춤형)
        processors=[security_processor],
        
        llm=llm,
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(eager_turn_detection=True),
    )
    
    # 프로세서 이벤트를 에이전트 이벤트에 병합
    agent.events.merge(security_processor.events)
    
    return agent
```

***

# Vision-Agents Technical Report

## PART 4-3-2: Security Camera Example - 이벤트 처리

### 4.3.2 보안 카메라 이벤트 핸들링

#### 사람 감지 이벤트

```python
@agent.events.subscribe
async def on_person_detected(event: PersonDetectedEvent):
    """
    새로운 사람 감지 또는 기존 사람 재발견
    
    event 속성:
    - face_id: 얼굴 고유 ID
    - is_new: 새로운 사람 여부
    - detection_count: 누적 감지 횟수
    - timestamp: 감지 시간
    - name: 등록된 이름 (있을 경우)
    """
    
    if event.is_new:
        # 새로운 방문자 (첫 감지)
        agent.logger.info(f"🚨 새 방문자: {event.face_id}")
        await agent.say(f"안녕하세요! 처음 뵙는데요?")
    else:
        # 재방문자 (30분 내 재감지)
        agent.logger.info(
            f"👤 재방문: {event.face_id} "
            f"({event.detection_count}번째)"
        )
        await agent.say(
            f"어서오세요! 다시 오셨네요"
        )


@agent.events.subscribe
async def on_person_disappeared(event: PersonDisappearedEvent):
    """
    사람이 프레임에서 사라짐
    
    event 속성:
    - face_id: 사라진 얼굴 ID
    - display_name: 표시 이름
    - last_seen: 마지막 감지 시간
    """
    
    display_name = event.name or event.face_id[:8]
    agent.logger.info(f"👤 방문자 퇴장: {display_name}")
    
    # 선택: 퇴장 인사
    # await agent.say("안녕히 가세요!")
```

***

#### 패키지 감지 이벤트

```python
@agent.events.subscribe
async def on_package_detected(event: PackageDetectedEvent):
    """
    새로운 패키지 감지
    
    event 속성:
    - package_id: 패키지 고유 ID
    - is_new: 새로 감지됨
    - confidence: 신뢰도 (0-1)
    - timestamp: 감지 시간
    - detection_count: 누적 감지 횟수
    """
    
    # 모든 진행 중인 도난 확인 취소
    # (이전 감지가 거짓 경보였을 가능성)
    if _pending_theft_tasks:
        cancelled_ids = list(_pending_theft_tasks.keys())
        for pkg_id in cancelled_ids:
            _pending_theft_tasks[pkg_id].cancel()
            del _pending_theft_tasks[pkg_id]
        
        agent.logger.info(
            f"📦 새 패키지 - 취소된 도난 확인: {', '.join(cancelled_ids)}"
        )
    
    # 패키지 히스토리에 기록
    if event.package_id not in _package_history:
        # 새 패키지
        _package_history[event.package_id] = {
            "package_id": event.package_id,
            "first_seen": event.timestamp.isoformat(),
            "last_seen": event.timestamp.isoformat(),
            "detection_count": 1,
            "confidence": event.confidence,
            "picked_up_by": None,
        }
    else:
        # 기존 패키지 (재감지)
        _package_history[event.package_id]["last_seen"] = (
            event.timestamp.isoformat()
        )
        _package_history[event.package_id]["detection_count"] += 1
    
    if event.is_new:
        agent.logger.info(
            f"📦 새 패키지 감지: {event.package_id} "
            f"(신뢰도: {event.confidence:.2f})"
        )
        await agent.say("📦 새 소포가 감지되었습니다!")
    else:
        agent.logger.info(
            f"📦 패키지 재감지: {event.package_id} "
            f"({event.detection_count}번째)"
        )
```

**패키지 추적 상태:**

```
시간   | 상태              | 처리
──────────────────────────────────
0s   | 패키지 감지 ✓     | 기록
     | PackageDetectedEvent
─────────────────────────────────
5s   | 여전히 보임       | 카운트 증가
─────────────────────────────────
180s | 패키지 사라짐 ❌  | 도난 확인 시작
     | PackageDisappearedEvent
     | _pending_theft_tasks에 추가
─────────────────────────────────
183s | 3초 대기 후       | 여전히 사라짐?
     | 도난으로 판정 ⚠️   |
─────────────────────────────────
185s | 원포스터 생성 📸  | handle_package_theft()
     | X 게시           | poster_generator
─────────────────────────────────
```

***

#### 패키지 사라짐 이벤트 (도난 감지)

```python
@agent.events.subscribe
async def on_package_disappeared(event: PackageDisappearedEvent):
    """
    패키지가 프레임에서 사라짐
    
    event 속성:
    - package_id: 사라진 패키지 ID
    - picker_face_id: 집어간 사람 ID
    - picker_name: 집어간 사람 이름 (등록된 경우)
    - timestamp: 사라진 시간
    
    중요: 즉시 도난으로 판정하지 않음
    → 3초 대기 후 패키지 재등장 확인
    """
    
    picker_display = event.picker_name or (
        event.picker_face_id[:8] if event.picker_face_id else "unknown"
    )
    
    agent.logger.info(
        f"📦 패키지 사라짐: {event.package_id} "
        f"(용의자: {picker_display}) - "
        f"3초 대기 중..."
    )
    
    async def delayed_theft_check():
        """
        3초 대기 후 도난 여부 확인
        
        로직:
        - 3초 후에도 패키지가 없으면 도난 판정
        - 3초 내에 재등장하면 거짓 경보 무시
        """
        await asyncio.sleep(PACKAGE_THEFT_DELAY_SECONDS)  # 3초
        
        # 3초 후에도 패키지가 사라져 있음
        del _pending_theft_tasks[event.package_id]
        
        agent.logger.info(
            f"📦 패키지 도난 확인: {event.package_id} "
            f"(용의자: {picker_display})"
        )
        
        # 패키지 히스토리 업데이트
        if event.package_id in _package_history:
            _package_history[event.package_id]["picked_up_by"] = (
                picker_display
            )
        
        # 용의자 얼굴 이미지 추출
        if event.picker_face_id:
            face_image = security_processor.get_face_image(
                event.picker_face_id
            )
            
            if face_image is not None:
                # 원포스터 생성 및 게시
                await handle_package_theft(
                    agent, 
                    face_image, 
                    picker_display, 
                    security_processor
                )
        else:
            agent.logger.warning(
                "얼굴 이미지 없음 - 원포스터 생성 불가"
            )
    
    # 비동기 작업 생성
    _pending_theft_tasks[event.package_id] = asyncio.create_task(
        delayed_theft_check()
    )
```

**도난 감지 로직:**

```
┌─ 패키지 감지 (T=0s)
│  └─ 기록됨 ✓
│
├─ 패키지 계속 보임 (T=10s, 50s, 100s)
│  └─ 정상 상태
│
├─ 패키지 사라짐 (T=180s)
│  └─ PackageDisappearedEvent 발행
│  └─ delayed_theft_check() 시작 (3초 타이머)
│
├─ 시나리오 1: 패키지 재등장 (T=182s)
│  └─ PackageDetectedEvent 발행
│  └─ delayed_theft_check() 취소 ✓
│  └─ 결과: 거짓 경보 (불필요한 경보 없음)
│
└─ 시나리오 2: 패키지 여전히 없음 (T=183s)
   └─ 타이머 만료
   └─ 도난으로 판정 ⚠️
   └─ handle_package_theft() 호출
      ├─ 원포스터 생성
      ├─ X에 게시
      └─ AI 분석 리포트
```

***

#### 원포스터 생성 및 게시

```python
async def handle_package_theft(
    agent: Agent,
    face_image: np.ndarray,           # 용의자 얼굴
    suspect_name: str,                # 용의자 이름
    processor: SecurityCameraProcessor,
) -> None:
    """
    패키지 도난 감지 후 원포스터 생성 및 게시
    
    단계:
    1. 경보 음성
    2. 원포스터 생성
    3. 영상에 표시
    4. X(Twitter) 게시
    5. 저장
    """
    
    # Step 1️⃣: 경보 음성
    await agent.say(
        f"알림! {suspect_name}이(가) 패키지를 집어갔습니다!"
    )
    
    # Step 2️⃣: 원포스터 생성 및 X 게시
    poster_bytes, tweet_url = await generate_and_post_poster(
        face_image,
        suspect_name,
        post_to_x_enabled=True,
        tweet_caption=(
            f"🚨 WANTED: {suspect_name} "
            f"caught stealing a package! "
            f"AI-powered security #VisionAgents"
        ),
    )
    
    # Step 3️⃣: 원포스터 저장
    if poster_bytes:
        # 로컬 저장
        with open(f"wanted_poster_{suspect_name}.png", "wb") as f:
            f.write(poster_bytes)
        
        agent.logger.info("✅ 원포스터 저장됨")
        
        # Step 4️⃣: 비디오 스트림에 표시 (8초)
        processor.share_image(poster_bytes, duration=8.0)
        
        await agent.say("원포스터가 영상에 표시되었습니다!")
    
    # Step 5️⃣: X 게시 확인
    if tweet_url:
        agent.logger.info(f"🐦 X에 게시됨: {tweet_url}")
        await agent.say(f"원포스터가 X에 게시되었습니다: {tweet_url}")
    else:
        agent.logger.warning("⚠️ X 게시 실패 (인증 확인 필요)")
```

**워크플로우 타이밍:**

```
도난 감지
    │
    ├─ 0ms: agent.say() 호출
    │  └─ "알림! [용의자]이(가) 패키지를..."
    │     └─ TTS → 음성 재생
    │
    ├─ 100ms: generate_and_post_poster() 호출
    │  ├─ 얼굴 이미지 처리
    │  ├─ 포스터 생성 (이미지 생성)
    │  └─ X API 호출
    │
    ├─ 500ms: 포스터 생성 완료
    │  ├─ 로컬 파일 저장
    │  └─ 비디오 스트림에 표시 (8초)
    │
    └─ 600ms: 완료
       └─ 사용자가 원포스터 봄
```

***


# Vision-Agents Technical Report

## PART 4-4: Function Calling 예제 (상세)

### 4.4 Function Calling - 보안 카메라 시스템

Function Calling은 LLM이 **필요할 때 자동으로 함수를 호출**하게 하는 기능입니다.

#### 4.4.1 등록된 함수들

```python
@llm.register_function(
    description="Get the number of unique visitors detected in the last 30 minutes."
)
async def get_visitor_count() -> Dict[str, Any]:
    """
    지난 30분간 감지된 고유 방문자 수 조회
    
    사용 사례:
    사용자: "지난 30분간 몇 명이 왔어?"
    LLM: get_visitor_count() 자동 호출
    반환: {"unique_visitors": 5, "total_detections": 12, ...}
    응답: "5명의 다른 방문자가 왔습니다"
    """
    count = security_processor.get_visitor_count()
    state = security_processor.state()
    
    return {
        "unique_visitors": count,
        "total_detections": state["total_face_detections"],
        "time_window": f"{state['time_window_minutes']} minutes",
        "last_detection": state["last_face_detection_time"],
    }


@llm.register_function(
    description="Get detailed information about all visitors including when they were first and last seen."
)
async def get_visitor_details() -> Dict[str, Any]:
    """
    모든 방문자의 상세 정보
    
    사용 사례:
    사용자: "방문자 정보 보여줘"
    LLM: get_visitor_details() 자동 호출
    반환:
    {
        "visitors": [
            {
                "face_id": "abc123",
                "name": "John",
                "first_seen": "2026-01-21T14:30:00",
                "last_seen": "2026-01-21T14:45:00",
                "detection_count": 5
            },
            ...
        ]
    }
    응답: "John은 14:30부터 14:45까지 있었고..."
    """
    details = security_processor.get_visitor_details()
    
    return {
        "visitors": details,
        "total_unique_visitors": len(details),
    }


@llm.register_function(
    description="Get package statistics including total packages seen and how many were picked up."
)
async def get_package_count() -> Dict[str, Any]:
    """
    패키지 통계 조회
    
    사용 사례:
    사용자: "패키지 현황이 어떻게 돼?"
    LLM: get_package_count() 자동 호출
    반환:
    {
        "currently_visible_packages": 2,
        "total_packages_seen": 15,
        "packages_picked_up": 13,
    }
    응답: "현재 2개, 총 15개 중 13개가 집어갔습니다"
    """
    currently_visible = security_processor.get_package_count()
    total_seen = len(_package_history)
    picked_up = sum(
        1 for p in _package_history.values() 
        if p.get("picked_up_by")
    )
    
    return {
        "currently_visible_packages": currently_visible,
        "total_packages_seen": total_seen,
        "packages_picked_up": picked_up,
    }


@llm.register_function(
    description="Get detailed history of all packages seen, including who picked them up."
)
async def get_package_details() -> Dict[str, Any]:
    """
    패키지 상세 이력
    
    사용 사례:
    사용자: "어떤 패키지들이 있었어?"
    LLM: get_package_details() 자동 호출
    반환:
    {
        "packages": [
            {
                "package_id": "pkg_001",
                "first_seen": "2026-01-21T14:00:00",
                "last_seen": "2026-01-21T14:05:00",
                "detection_count": 3,
                "picked_up_by": "face_123"
            },
            ...
        ]
    }
    """
    return {
        "packages": list(_package_history.values()),
        "total_packages_seen": len(_package_history),
    }


@llm.register_function(
    description="Get recent activity log (people arriving, packages detected). Answers 'what happened?' or 'did anyone come by?'"
)
async def get_activity_log(limit: int = 20) -> Dict[str, Any]:
    """
    최근 활동 기록
    
    사용 사례:
    사용자: "뭐가 일어났어?"
    LLM: get_activity_log(limit=20) 자동 호출
    반환: [
        {"timestamp": "2026-01-21T14:45:00", "type": "person_detected", "face_id": "abc"},
        {"timestamp": "2026-01-21T14:46:00", "type": "package_detected", "package_id": "pkg_123"},
        ...
    ]
    """
    log = security_processor.get_activity_log(limit=limit)
    
    return {
        "activity_log": log,
        "total_entries": len(log),
    }


@llm.register_function(
    description="Register the current person's face with a name so they can be recognized in the future."
)
async def remember_my_face(name: str) -> Dict[str, Any]:
    """
    얼굴 등록
    
    사용 사례:
    사용자: "나를 John이라고 기억해줘"
    LLM: remember_my_face(name="John") 자동 호출
    반환: {"success": True, "name": "John", "face_id": "abc123"}
    응답: "John을 기억했습니다. 다음에 오실 때 인식하겠습니다"
    """
    result = security_processor.register_current_face_as(name)
    return result


@llm.register_function(
    description="Get a list of all registered faces that can be recognized by name."
)
async def get_known_faces() -> Dict[str, Any]:
    """
    등록된 얼굴 목록
    
    사용 사례:
    사용자: "누가 등록되어 있어?"
    LLM: get_known_faces() 자동 호출
    반환: {
        "known_faces": ["John", "Sarah", "Mike"],
        "total_known": 3
    }
    """
    faces = security_processor.get_known_faces()
    
    return {
        "known_faces": faces,
        "total_known": len(faces),
    }
```

***

#### 4.4.2 Function Calling 워크플로우

**사용자 쿼리 → 자동 함수 호출 → LLM 응답**

```
사용자: "지난 30분간 뭐가 일어났어?"
    │
    ▼
LLM 분석:
├─ 질문 내용: 최근 활동 정보
├─ 필요한 함수: get_activity_log()
└─ 매개변수: limit=20

    │
    ▼
get_activity_log(limit=20) 실행
    │
    ├─ 30분 시간 범위 필터링
    ├─ 활동 기록 조회
    └─ 반환: [
        {timestamp: "14:45:00", type: "person_detected", ...},
        {timestamp: "14:46:00", type: "package_detected", ...},
        {timestamp: "14:50:00", type: "person_disappeared", ...},
        ...
    ]
    │
    ▼
LLM이 결과 분석:
├─ 활동 수 계산
├─ 타임라인 정렬
├─ 자연어 변환
└─ 응답 생성

    │
    ▼
응답: "지난 30분간 5명의 방문자가 있었고,
       3개의 패키지가 감지되었으며,
       그 중 2개는 집어갔습니다.
       14:50에 의심 활동이 감지되었습니다."
```

***

### 4.5 Advanced Example: 전화 + RAG (Phone and RAG)

**파일**: `examples/03_phone_and_rag_example/`

```python
"""
전화 기반 상담원 봇
- Twilio: 전화 음성 통합
- TurboPuffer: 벡터 DB (고객 정보 검색)
- RAG: 검색증강생성
"""

async def create_agent(**kwargs) -> Agent:
    """전화 상담원 에이전트"""
    
    # RAG 프로세서 (고객 정보 검색)
    rag_processor = turbopuffer.TurboPufferRAG(
        index_name="customer_db",
        vector_size=1536,
        hybrid=True,  # 벡터 + 전문 검색
    )
    
    agent = Agent(
        edge=getstream.Edge(),
        llm=openai.LLM("gpt-4o"),
        processors=[rag_processor],
        stt=deepgram.STT(),
        tts=elevenlabs.TTS(),
    )
    
    # LLM이 RAG 쿼리 가능
    @llm.register_function(
        description="Search customer database for order information"
    )
    async def search_customer_info(query: str) -> Dict:
        """고객 정보 검색"""
        results = await rag_processor.search(
            query,
            top_k=5
        )
        return {"results": results}
    
    return agent
```

**사용 사례:**
```
고객: "주문 #12345의 배송 상태는?"
    │
    ▼
STT: "주문 12345의 배송 상태는?"
    │
    ▼
LLM: search_customer_info("order #12345 status")
    │
    ├─ 벡터 임베딩: 쿼리를 벡터로 변환
    ├─ 데이터베이스 검색: 유사 문서 찾기
    └─ 반환: [
        {
            "order_id": "12345",
            "status": "in_transit",
            "tracking_number": "1Z123...",
            "estimated_delivery": "2026-01-23"
        }
    ]
    │
    ▼
LLM 응답: "귀사의 주문은 배송 중이며,
           추적번호는 1Z123...이고,
           2026년 1월 23일에 도착할 예정입니다."
    │
    ▼
TTS: 음성으로 응답 재생
    │
    ▼
고객에게 음성으로 전달
```

***

### 4.6 핵심 개념 정리

#### Simple vs Advanced 비교

```
Simple Agent (01_simple_agent_example):
├─ 음성만 처리
├─ LLM 텍스트 기반
├─ STT + LLM + TTS
├─ 지연시간: 600-800ms
└─ 사용 사례: 음성 어시스턴트

Golf Coach (02_golf_coach_example):
├─ 음성 + 비디오
├─ Gemini Realtime (multimodal)
├─ YOLO 포즈 감지
├─ 지연시간: 400-600ms
└─ 사용 사례: 실시간 코칭

Security Camera (05_security_camera_example):
├─ 24/7 모니터링
├─ 다중 프로세서
├─ 이벤트 기반
├─ 함수 호출 (Function Calling)
└─ 사용 사례: 자동화된 보안

Phone + RAG (03_phone_and_rag_example):
├─ 전화 통합 (Twilio)
├─ 벡터 데이터베이스
├─ 검색증강생성
└─ 사용 사례: 지능형 콜센터
```

***

