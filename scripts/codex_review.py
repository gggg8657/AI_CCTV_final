#!/usr/bin/env python3
"""
Codex CLI와 자동으로 통신하여 Phase 3 계획 검토 요청
pty를 사용하여 대화형 터미널 시뮬레이션
"""

import os
import sys
import pty
import select
import subprocess
import time
from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent
PROMPT_FILE = PROJECT_ROOT / "docs" / "CODEX_REVIEW_PROMPT.txt"
OUTPUT_FILE = PROJECT_ROOT / "docs" / "CODEX_REVIEW_RESPONSE.txt"


def read_codex_prompt() -> str:
    """Codex 프롬프트 파일 읽기"""
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")
    
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        return f.read()


def communicate_with_codex(prompt: str, timeout: int = 300) -> str:
    """
    pty를 사용하여 Codex CLI와 통신
    
    Args:
        prompt: Codex에 전달할 프롬프트
        timeout: 타임아웃 (초)
    
    Returns:
        Codex의 응답
    """
    # Codex CLI 경로 확인
    codex_path = "/opt/homebrew/bin/codex"
    if not os.path.exists(codex_path):
        raise FileNotFoundError(f"Codex CLI not found at: {codex_path}")
    
    # pty를 사용하여 Codex 실행
    master_fd, slave_fd = pty.openpty()
    
    try:
        # Codex 프로세스 시작
        process = subprocess.Popen(
            [codex_path],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            text=True,
            bufsize=0
        )
        
        # 프롬프트 전송
        os.write(master_fd, prompt.encode('utf-8'))
        os.write(master_fd, b'\n\n')  # 엔터 두 번 (대화 종료 신호)
        
        # 응답 수집
        response = []
        start_time = time.time()
        
        while True:
            if time.time() - start_time > timeout:
                break
            
            # 프로세스 종료 확인
            if process.poll() is not None:
                break
            
            # 읽기 가능한 데이터 확인
            if select.select([master_fd], [], [], 0.1)[0]:
                try:
                    data = os.read(master_fd, 1024)
                    if data:
                        response.append(data.decode('utf-8', errors='ignore'))
                except OSError:
                    break
        
        # 프로세스 종료 대기
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        
        return ''.join(response)
    
    finally:
        os.close(master_fd)
        os.close(slave_fd)


def main():
    """메인 함수"""
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Codex CLI와 통신 시작...")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    try:
        # 프롬프트 읽기
        print("1. 프롬프트 파일 읽기...")
        prompt = read_codex_prompt()
        print(f"   ✅ 프롬프트 길이: {len(prompt)} 문자")
        
        # Codex와 통신
        print("2. Codex CLI와 통신 중...")
        print("   (이 작업은 몇 분이 걸릴 수 있습니다...)")
        
        response = communicate_with_codex(prompt, timeout=300)
        
        # 응답 저장
        print("3. 응답 저장 중...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(response)
        
        print(f"   ✅ 응답 저장 완료: {OUTPUT_FILE}")
        print(f"   응답 길이: {len(response)} 문자")
        
        # 응답 미리보기
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("응답 미리보기 (처음 500자):")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(response[:500])
        if len(response) > 500:
            print(f"\n... (총 {len(response)} 문자, 전체 내용은 {OUTPUT_FILE} 참조)")
        
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("✅ Codex 검토 완료!")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        return 0
    
    except FileNotFoundError as e:
        print(f"❌ 오류: {e}")
        return 1
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
