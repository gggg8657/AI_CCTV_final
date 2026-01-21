# Codex CLI 협업 전략

**작성일**: 2026-01-21  
**목적**: Cursor (AI Assistant)와 Codex CLI의 역할 분담 및 협업 워크플로우

---

## 역할 분담

### Codex CLI 역할 (초안 생성기)
- **코드 초안 생성**: 새로운 기능, 클래스, 함수의 기본 구조 생성
- **패치 형태 출력**: Unified diff 형식으로 변경사항 제시
- **빠른 프로토타이핑**: 아이디어를 빠르게 코드로 변환
- **반복적인 패턴**: 비슷한 구조의 코드를 여러 파일에 생성

### Cursor (AI Assistant) 역할 (검수 및 통합)
- **코드 검수**: Codex가 생성한 코드의 품질 검증
- **프로젝트 통합**: 기존 코드베이스와의 통합 및 호환성 확인
- **테스트 작성**: 단위 테스트 및 통합 테스트 작성
- **문서화**: 코드 주석, docstring, README 업데이트
- **최종 품질 보장**: 린터 에러 수정, 성능 최적화, 보안 검토

---

## 협업 워크플로우

### Phase 1: 이벤트 시스템 구현 예시

#### Step 1: Codex로 초안 생성

**Codex 입력 프롬프트**:
```
[GOAL]
EventBus 클래스 구현: Python asyncio 기반 Pub/Sub 이벤트 버스

[CONSTRAINTS]
- Output must be unified diff
- Use asyncio.Queue for event processing
- Support async and sync handlers
- Thread-safe implementation
- Include type hints
- Do not modify unrelated files

[CONTEXT]
- Project: AI_CCTV_final
- Location: src/utils/event_bus.py (new file)
- Python 3.10+
- Existing codebase uses dataclasses, typing

[INPUTS]
- Event types: AnomalyDetectedEvent, VLMAnalysisCompletedEvent, etc.
- Need subscribe/publish/unsubscribe methods
- Event history management (max 1000 events)

[OUTPUTS]
- Unified diff patch for src/utils/event_bus.py
- Commands to verify: python -m pytest tests/test_event_bus.py

[VALIDATION]
- Reproduce: Create EventBus instance, subscribe, publish event
- Verify: Check event is delivered to subscribers
```

**Codex 출력**: Unified diff 패치

#### Step 2: Cursor가 검수 및 통합

1. **패치 적용**: Codex가 생성한 diff를 적용
2. **코드 검수**:
   - 타입 힌트 확인
   - 에러 처리 확인
   - 기존 코드와의 호환성 확인
3. **통합 작업**:
   - 기존 `AnomalyEvent` 클래스와 통합
   - `E2ESystem`에 EventBus 연결
   - 테스트 작성
4. **문서화**: Docstring 추가, 사용 예시 작성

---

## Codex 호출 템플릿

### 템플릿 1: 새 파일 생성

```bash
export PROMPT=$(cat <<'EOF'
[GOAL]
새로운 클래스/모듈 생성

[CONSTRAINTS]
- Output must be unified diff
- Include type hints
- Follow existing code style
- Do not modify unrelated files

[CONTEXT]
- File path: <target_path>
- Existing patterns: <reference_files>
- Dependencies: <dependencies>

[INPUTS]
- Requirements: <requirements>
- Interface: <interface_spec>

[OUTPUTS]
- Unified diff patch
- Commands to verify

[VALIDATION]
- Reproduce: <reproduce_command>
- Verify: <verify_command>
EOF
)

codex <<'EOF'
$PROMPT
EOF
```

### 템플릿 2: 기존 파일 수정

```bash
export PROMPT=$(cat <<'EOF'
[GOAL]
기존 파일에 기능 추가/수정

[CONSTRAINTS]
- Output must be unified diff
- Do not change unrelated code
- Maintain backward compatibility
- Include commands: reproduce + verify

[CONTEXT]
- File: <file_path>
- Current implementation: <current_code>
- Related files: <related_files>

[INPUTS]
- Change required: <change_description>
- Error logs: <error_logs>

[OUTPUTS]
- Unified diff patch
- Commands to reproduce + verify

[VALIDATION]
- Reproduce: <reproduce_command>
- Verify: <verify_command>
EOF
)

codex <<'EOF'
$PROMPT
EOF
```

---

## 실제 사용 예시

### 예시 1: EventBus 클래스 생성

**Codex 호출**:
```bash
codex <<'EOF'
[GOAL]
Create EventBus class for pub/sub pattern

[CONSTRAINTS]
- Output unified diff
- Use asyncio.Queue
- Thread-safe
- Type hints required

[CONTEXT]
- New file: src/utils/event_bus.py
- Python 3.10+
- Uses dataclasses

[INPUTS]
- Methods: subscribe(event_type, handler), publish(event), unsubscribe()
- Event history: max 1000 events
- Support async handlers

[OUTPUTS]
- Unified diff for src/utils/event_bus.py
- Test command: python -c "from src.utils.event_bus import EventBus; print('OK')"

[VALIDATION]
- Reproduce: Create EventBus, subscribe, publish
- Verify: Event delivered to handler
EOF
```

**Codex 출력** (예시):
```diff
--- /dev/null
+++ b/src/utils/event_bus.py
@@ -0,0 +1,50 @@
+import asyncio
+from collections import defaultdict
+from typing import Dict, List, Callable, Type, Any
+from collections import deque
+import threading
+
+class EventBus:
+    def __init__(self, max_history: int = 1000):
+        self._subscribers: Dict[Type, List[Callable]] = defaultdict(list)
+        self._event_history: deque = deque(maxlen=max_history)
+        self._lock = threading.Lock()
+        self._queue = asyncio.Queue()
+    
+    def subscribe(self, event_type: Type, handler: Callable):
+        with self._lock:
+            self._subscribers[event_type].append(handler)
+    
+    def unsubscribe(self, event_type: Type, handler: Callable):
+        with self._lock:
+            if handler in self._subscribers[event_type]:
+                self._subscribers[event_type].remove(handler)
+    
+    async def publish(self, event: Any):
+        with self._lock:
+            self._event_history.append(event)
+            handlers = self._subscribers[type(event)].copy()
+        
+        await self._queue.put((event, handlers))
+    
+    async def _process_queue(self):
+        while True:
+            event, handlers = await self._queue.get()
+            for handler in handlers:
+                if asyncio.iscoroutinefunction(handler):
+                    await handler(event)
+                else:
+                    handler(event)
```

**Cursor 검수 및 개선**:
1. 패치 적용
2. 타입 힌트 보완
3. 에러 처리 추가
4. 기존 `AnomalyEvent`와 통합
5. 테스트 작성

---

## 분담 기준

### Codex에게 맡길 작업
- ✅ 새로운 클래스/함수 초안 생성
- ✅ 반복적인 코드 패턴 생성
- ✅ 표준 라이브러리 사용 코드
- ✅ 간단한 유틸리티 함수

### Cursor가 직접 할 작업
- ✅ 기존 코드 수정 및 통합
- ✅ 복잡한 비즈니스 로직
- ✅ 테스트 작성
- ✅ 문서화
- ✅ 디버깅 및 에러 수정
- ✅ 성능 최적화

---

## 체크리스트

### Codex 호출 전
- [ ] 목표가 명확한가?
- [ ] 제약사항이 명확한가?
- [ ] 컨텍스트가 충분한가?
- [ ] 검증 방법이 명확한가?

### Codex 출력 후
- [ ] 패치가 올바른가?
- [ ] 타입 힌트가 있는가?
- [ ] 에러 처리가 있는가?
- [ ] 기존 코드와 호환되는가?

### Cursor 통합 후
- [ ] 테스트가 통과하는가?
- [ ] 린터 에러가 없는가?
- [ ] 문서화가 완료되었는가?
- [ ] 커밋 메시지가 적절한가?

---

*이 전략은 프로젝트 진행에 따라 지속적으로 개선됩니다.*
