#!/usr/bin/env python3
"""
Codex CLI와 자동으로 통신하여 Phase 3 계획 검토 요청
pexpect를 사용하여 대화형 터미널 시뮬레이션
"""

import sys
import pexpect
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
    pexpect를 사용하여 Codex CLI와 통신
    
    Args:
        prompt: Codex에 전달할 프롬프트
        timeout: 타임아웃 (초)
    
    Returns:
        Codex의 응답
    """
    # Codex CLI 경로 확인
    codex_path = "/opt/homebrew/bin/codex"
    
    print(f"Codex CLI 실행: {codex_path}")
    
    # Codex 프로세스 시작
    child = pexpect.spawn(
        codex_path,
        encoding='utf-8',
        timeout=timeout,
        maxread=10000
    )
    
    # 로그 파일 설정 (디버깅용)
    child.logfile = sys.stdout
    
    try:
        # Codex 프롬프트 대기 (예: ">" 또는 "codex>" 등)
        # Codex의 실제 프롬프트에 맞게 수정 필요
        child.expect([pexpect.EOF, pexpect.TIMEOUT, r'.*[>#\$]'], timeout=10)
        
        # 프롬프트 전송
        print("\n프롬프트 전송 중...")
        child.send(prompt)
        child.send('\n\n')  # 엔터
        
        # 응답 수집 (EOF 또는 타임아웃까지)
        print("응답 대기 중...")
        response_parts = []
        
        while True:
            try:
                index = child.expect([
                    pexpect.EOF,
                    pexpect.TIMEOUT,
                    r'.*',  # 어떤 텍스트든
                ], timeout=30)
                
                if index == 0:  # EOF
                    response_parts.append(child.before)
                    break
                elif index == 1:  # TIMEOUT
                    response_parts.append(child.before)
                    print("타임아웃 발생, 수집된 응답 반환")
                    break
                else:  # 텍스트 수신
                    response_parts.append(child.before)
                    # 계속 읽기
                    continue
                    
            except pexpect.TIMEOUT:
                response_parts.append(child.before)
                break
            except pexpect.EOF:
                response_parts.append(child.before)
                break
        
        return ''.join(response_parts)
    
    finally:
        child.close()


def main():
    """메인 함수"""
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Codex CLI와 통신 시작 (pexpect 사용)...")
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
        print("\n3. 응답 저장 중...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(response)
        
        print(f"   ✅ 응답 저장 완료: {OUTPUT_FILE}")
        print(f"   응답 길이: {len(response)} 문자")
        
        # 응답 미리보기
        if response:
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
