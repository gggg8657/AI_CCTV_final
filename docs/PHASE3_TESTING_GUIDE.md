# Phase 3 테스트 가이드

## 빠른 테스트 (Mock 데이터)

실제 비디오 없이 Phase 3 컴포넌트만 테스트:

```bash
python scripts/test_phase3_quick.py
```

**예상 결과:**
- EventBus 생성 및 시작
- PackageTracker 생성
- TheftDetector 생성
- Mock 패키지 감지 → 추적 → 사라짐 → 도난 감지
- 3개 이벤트 수신 (PackageDetectedEvent, PackageDisappearedEvent, TheftDetectedEvent)

## 실제 비디오 테스트

### 1. 비디오 파일 테스트

```bash
# 기본 설정
python scripts/test_phase3.py --source /path/to/video.mp4

# 신뢰도 임계값 조정
python scripts/test_phase3.py --source /path/to/video.mp4 --confidence 0.6

# CPU 모드 (GPU 없을 때)
python scripts/test_phase3.py --source /path/to/video.mp4 --gpu -1

# 도난 확인 시간 조정
python scripts/test_phase3.py --source /path/to/video.mp4 --theft-time 5.0
```

### 2. 웹캠 테스트

```bash
python scripts/test_phase3.py --source 0 --source-type webcam
```

### 3. RTSP 스트림 테스트

```bash
python scripts/test_phase3.py --source rtsp://192.168.1.100:554/stream --source-type rtsp
```

### 4. 패키지 감지만 테스트 (VAD/VLM/Agent 비활성화)

```bash
python scripts/test_phase3.py --source /path/to/video.mp4 --no-vad --no-vlm --no-agent
```

## 테스트 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--source`, `-s` | 비디오 소스 (필수) | - |
| `--source-type` | 소스 타입 (file/rtsp/webcam) | file |
| `--gpu`, `-g` | GPU 디바이스 ID (CPU: -1) | 2 |
| `--confidence` | 패키지 감지 신뢰도 임계값 | 0.5 |
| `--model` | YOLO 모델 경로/이름 | yolo12n.pt |
| `--theft-time` | 도난 확인 시간(초) | 3.0 |
| `--max-age` | 패키지 추적 최대 유지 시간(초) | 30 |
| `--fps` | 목표 FPS | 30 |
| `--no-vad` | VAD 비활성화 | - |
| `--no-vlm` | VLM 비활성화 | - |
| `--no-agent` | Agent 비활성화 | - |

## 예상 출력

### 정상 실행 시

```
============================================================
Phase 3 Package Detection 테스트
============================================================
비디오 소스: /path/to/video.mp4
소스 타입: file
패키지 감지: 활성화
  모델: yolo12n.pt
  신뢰도 임계값: 0.5
  도난 확인 시간: 3.0초
============================================================

[초기화] 엔진 초기화 중...
[초기화] 완료!

[이벤트] EventBus 핸들러 등록 완료
[시작] 엔진 실행 중...
Ctrl+C로 종료할 수 있습니다.

[패키지 감지] ID: pkg_0001, 위치: (100, 100, 200, 200), 신뢰도: 0.85, 카메라: 0
[패키지 감지] ID: pkg_0002, 위치: (300, 300, 400, 400), 신뢰도: 0.72, 카메라: 0

[통계] 프레임: 30, FPS: 29.5, 패키지 감지: 2, 사라짐: 0, 도난: 0

[패키지 사라짐] ID: pkg_0001, 최종 감지: 2026-01-28T02:15:30Z, 카메라: 0

============================================================
[🚨 도난 감지!] 패키지 ID: pkg_0001
   시간: 2026-01-28T02:15:33Z
   카메라: 0
   증거 영상: 1개
============================================================
```

### 테스트 완료 시

```
============================================================
테스트 완료 - 최종 통계
============================================================
처리된 프레임: 900
패키지 감지 이벤트: 5
패키지 사라짐 이벤트: 2
도난 감지 이벤트: 1

엔진 통계:
  총 프레임: 900
  평균 FPS: 29.8
  이상 감지: 12
  현재 추적 중인 패키지: 3
    - pkg_0001: stolen (감지 횟수: 45)
    - pkg_0002: present (감지 횟수: 120)
    - pkg_0003: missing (감지 횟수: 89)
============================================================
```

## 문제 해결

### 1. YOLO 모델 로드 실패

**증상:**
```
[오류] Package detector load failed
```

**해결:**
- `ultralytics` 패키지 설치 확인: `pip install ultralytics`
- 모델 파일 경로 확인
- 인터넷 연결 확인 (첫 실행 시 모델 다운로드)

### 2. GPU 메모리 부족

**증상:**
```
CUDA out of memory
```

**해결:**
- CPU 모드 사용: `--gpu -1`
- 더 작은 모델 사용: `--model yolo12n.pt` (nano 모델)

### 3. 패키지 감지 안됨

**증상:**
- 패키지 감지 이벤트가 발생하지 않음

**해결:**
- 신뢰도 임계값 낮추기: `--confidence 0.3`
- 비디오에 패키지(handbag, backpack, suitcase)가 있는지 확인
- YOLO 모델이 제대로 로드되었는지 확인

### 4. 도난 감지 안됨

**증상:**
- 패키지는 감지되지만 도난 감지가 안됨

**해결:**
- 도난 확인 시간 줄이기: `--theft-time 1.0`
- 패키지가 실제로 사라지는지 확인 (3초 이상)

## 성능 최적화

### CPU 모드 최적화

```bash
# CPU 모드 + 낮은 FPS
python scripts/test_phase3.py --source /path/to/video.mp4 --gpu -1 --fps 10
```

### GPU 메모리 절약

```bash
# VAD/VLM/Agent 비활성화 (패키지 감지만)
python scripts/test_phase3.py --source /path/to/video.mp4 --no-vad --no-vlm --no-agent
```

## 다음 단계

1. **실제 비디오로 테스트**
   - 패키지가 포함된 비디오 준비
   - 다양한 각도와 조명 조건 테스트

2. **성능 측정**
   - FPS 모니터링
   - 메모리 사용량 확인
   - GPU 사용률 확인

3. **정확도 평가**
   - False Positive/Negative 계산
   - 도난 감지 정확도 측정

4. **UI에서 확인**
   - `http://localhost:5173` 접속
   - "패키지 감지" 메뉴에서 실시간 모니터링
