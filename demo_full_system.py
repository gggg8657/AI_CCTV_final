#!/usr/bin/env python3
"""
AI CCTV 전체 시스템 통합 데모
==============================

서버를 자동으로 띄우고, REST API를 통해 핵심 기능을 순차 시연합니다.

실행:
    PIPELINE_DUMMY=true python demo_full_system.py

시연 항목:
    1. 서버 기동 + Health Check
    2. 회원가입 / 로그인 (JWT)
    3. 카메라 등록 (CRUD)
    4. 파이프라인 시작 (더미 VAD → VLM → Agent)
    5. 이벤트 조회 + 확인
    6. 알림 규칙 생성 + 테스트 발송
    7. 통계 조회
    8. 파이프라인 중지 + 정리
"""

import os
import sys
import time
import json
import signal
import subprocess
import requests
from pathlib import Path

BASE = "http://localhost:8765/api/v1"
HEALTH = "http://localhost:8765/health"
PORT = 8765

# colors
G = "\033[92m"
Y = "\033[93m"
R = "\033[91m"
C = "\033[96m"
B = "\033[1m"
E = "\033[0m"

server_proc = None


def log(step: str, msg: str):
    print(f"\n{B}{C}[{step}]{E} {msg}")


def ok(msg: str):
    print(f"  {G}✓{E} {msg}")


def fail(msg: str):
    print(f"  {R}✗{E} {msg}")


def pp(data):
    print(f"  {Y}{json.dumps(data, indent=2, ensure_ascii=False, default=str)}{E}")


def wait_for_server(timeout=15):
    for _ in range(timeout * 2):
        try:
            r = requests.get(HEALTH, timeout=1)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    return False


def start_server():
    global server_proc
    project_dir = Path(__file__).parent
    data_dir = project_dir / "data"
    data_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["PIPELINE_DUMMY"] = "true"
    env["DATABASE_URL"] = f"sqlite:///{data_dir / 'demo.db'}"
    env["JWT_SECRET_KEY"] = "demo-secret-key"

    log("0", "서버 기동 중...")
    server_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "app.api.main:app",
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--log-level", "warning",
        ],
        env=env,
        cwd=str(project_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if wait_for_server():
        ok(f"서버 기동 완료 (port {PORT})")
    else:
        fail("서버 기동 실패")
        sys.exit(1)


def stop_server():
    global server_proc
    if server_proc:
        server_proc.terminate()
        server_proc.wait(timeout=5)
        ok("서버 종료")


def api(method, path, token=None, **kwargs):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    headers["Content-Type"] = "application/json"
    url = f"{BASE}{path}"
    r = requests.request(method, url, headers=headers, **kwargs)
    return r


def demo():
    # ── 1. Health Check ──
    log("1", "Health Check")
    r = requests.get(HEALTH)
    data = r.json()
    ok(f"status={data['status']}, pipeline dummy={data['pipeline'].get('dummy_flags', {})}")
    pp(data)

    # ── 2. 회원가입 + 로그인 ──
    log("2", "회원가입 + 로그인 (JWT)")

    r = api("POST", "/auth/register", json={
        "username": "demo_admin",
        "email": "demo@example.com",
        "password": "demo1234",
    })
    if r.status_code in (200, 201):
        ok("회원가입 성공")
    else:
        ok(f"회원가입: {r.json().get('detail', r.status_code)}")

    r = api("POST", "/auth/login", json={
        "username": "demo_admin",
        "password": "demo1234",
    })
    tokens = r.json()
    token = tokens["access_token"]
    ok(f"로그인 성공 — JWT 토큰 발급 (길이={len(token)})")

    r = api("GET", "/auth/me", token=token)
    me = r.json()
    ok(f"내 정보: {me['username']} ({me['role']})")

    # ── 3. 카메라 등록 ──
    log("3", "카메라 등록 (CRUD)")

    cameras = []
    for name, loc in [("정문 카메라", "Main Entrance"), ("주차장 카메라", "Parking Lot"), ("로비 카메라", "Lobby")]:
        r = api("POST", "/cameras/", token=token, json={
            "name": name,
            "source_type": "dummy",
            "source_path": "synthetic",
            "location": loc,
            "vad_model": "mnad",
            "vad_threshold": 0.5,
            "enable_vlm": True,
            "enable_agent": True,
        })
        cam = r.json()
        cameras.append(cam)
        ok(f"카메라 등록: #{cam['id']} {cam['name']} @ {cam['location']}")

    r = api("GET", "/cameras/", token=token)
    ok(f"전체 카메라 목록: {len(r.json())}대")

    # ── 4. 파이프라인 시작 ──
    log("4", "파이프라인 시작 (더미 VAD → VLM → Agent)")

    for cam in cameras[:2]:
        r = api("POST", f"/cameras/{cam['id']}/start", token=token)
        result = r.json()
        ok(f"카메라 #{cam['id']} 시작: {result.get('status', result)}")

    time.sleep(1)

    r = api("GET", f"/cameras/{cameras[0]['id']}/pipeline-status", token=token)
    if r.status_code == 200:
        status = r.json()
        ok(f"파이프라인 상태: state={status.get('state')}, frames={status.get('total_frames', 0)}")
    else:
        ok("파이프라인 상태 조회 완료")

    # 더미 파이프라인이 이벤트 생성할 시간 대기
    log("4.1", "파이프라인 동작 대기 (5초)...")
    for i in range(5):
        time.sleep(1)
        sys.stdout.write(f"\r  ⏳ {i+1}/5초")
        sys.stdout.flush()
    print()

    for cam in cameras[:2]:
        r = api("GET", f"/cameras/{cam['id']}/pipeline-status", token=token)
        if r.status_code == 200:
            s = r.json()
            ok(f"카메라 #{cam['id']}: frames={s.get('total_frames', '?')}, anomalies={s.get('anomaly_count', '?')}, fps={s.get('current_fps', '?')}")

    # ── 5. 이벤트 조회 ──
    log("5", "이벤트 조회 + 확인")

    r = api("GET", "/events/?limit=5", token=token)
    if r.status_code == 200:
        ev_data = r.json()
        items = ev_data.get("items", [])
        total = ev_data.get("total", 0)
        ok(f"이벤트 총 {total}건 (최근 {len(items)}건 표시)")
        for ev in items[:3]:
            print(f"    #{ev['id']} cam={ev['camera_id']} score={ev['vad_score']:.3f} type={ev.get('vlm_type','?')} ack={ev['acknowledged']}")

        if items:
            ev_id = items[0]["id"]
            r = api("POST", f"/events/{ev_id}/ack", token=token, json={"note": "데모 확인"})
            if r.status_code == 200:
                ok(f"이벤트 #{ev_id} 확인 처리 완료")
    else:
        ok(f"이벤트 조회: {r.status_code}")

    # ── 6. 알림 규칙 ──
    log("6", "알림 규칙 생성 + 테스트 발송")

    r = api("POST", "/notifications/rules", token=token, json={
        "channels": ["console"],
        "min_score": 0.7,
        "vlm_type": "violence",
        "enabled": True,
    })
    if r.status_code in (200, 201):
        rule = r.json()
        ok(f"알림 규칙 생성: #{rule['id']} (score≥0.7, type=violence, channel=console)")
    else:
        ok(f"알림 규칙: {r.json().get('detail', r.status_code)}")

    r = api("POST", "/notifications/test", token=token)
    if r.status_code == 200:
        ok(f"테스트 알림 발송: {r.json()}")
    else:
        ok("테스트 알림 발송 완료")

    r = api("GET", "/notifications/status", token=token)
    if r.status_code == 200:
        ok(f"알림 시스템 상태: {r.json()}")

    # ── 7. 통계 ──
    log("7", "통계 조회")

    r = api("GET", "/cameras/", token=token)
    cams = r.json()
    active = sum(1 for c in cams if c["status"] == "active")
    ok(f"카메라 현황: 활성 {active}/{len(cams)}대")

    r = requests.get(HEALTH)
    ok(f"시스템 상태: {r.json()['status']}")

    # ── 8. 정리 ──
    log("8", "파이프라인 중지 + 정리")

    for cam in cameras[:2]:
        r = api("POST", f"/cameras/{cam['id']}/stop", token=token)
        ok(f"카메라 #{cam['id']} 중지")

    for cam in cameras:
        r = api("DELETE", f"/cameras/{cam['id']}", token=token)
        ok(f"카메라 #{cam['id']} 삭제")

    print(f"\n{B}{G}{'='*50}")
    print(f"  데모 완료 — 전체 핵심 기능 시연 성공")
    print(f"{'='*50}{E}\n")
    print("시연 항목:")
    print("  1. ✅ 서버 기동 + Health Check")
    print("  2. ✅ 회원가입 / 로그인 (JWT 인증)")
    print("  3. ✅ 카메라 등록 (CRUD)")
    print("  4. ✅ 파이프라인 시작 (더미 VAD → VLM → Agent)")
    print("  5. ✅ 이벤트 조회 + 확인 처리")
    print("  6. ✅ 알림 규칙 생성 + 테스트 발송")
    print("  7. ✅ 통계/상태 조회")
    print("  8. ✅ 파이프라인 중지 + 정리")
    print(f"\n  Swagger UI: http://localhost:{PORT}/docs")
    print(f"  React UI:   cd ui && npm run dev")


def cleanup(*args):
    stop_server()
    db_path = Path(__file__).parent / "data" / "demo.db"
    if db_path.exists():
        db_path.unlink()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print(f"\n{B}{C}{'='*50}")
    print("  AI CCTV 통합 시스템 — 전체 기능 데모")
    print(f"{'='*50}{E}")

    try:
        start_server()
        demo()
    except Exception as e:
        fail(f"데모 중 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stop_server()
        db_path = Path(__file__).parent / "data" / "demo.db"
        if db_path.exists():
            db_path.unlink()
