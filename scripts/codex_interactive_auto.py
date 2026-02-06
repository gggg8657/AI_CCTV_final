#!/usr/bin/env python3
"""
Codex CLI와 자동으로 대화하는 스크립트
pexpect를 사용하여 interactive mode에서 Codex와 대화
"""

import pexpect
import sys
import os
import time
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

def codex_interactive_chat(prompt: str, timeout: int = 300, wait_for_completion: bool = True):
    """
    Codex CLI와 interactive 대화
    
    Args:
        prompt: Codex에게 보낼 프롬프트
        timeout: 응답 대기 시간 (초)
        wait_for_completion: 완전한 응답을 기다릴지 여부
    
    Returns:
        Codex의 응답 텍스트
    """
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Codex CLI Interactive Session 시작")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"프로젝트 디렉토리: {PROJECT_ROOT}")
    print(f"")
    
    try:
        # Codex 실행
        print(f"[1/4] Codex CLI 실행 중...")
        child = pexpect.spawn(
            'codex',
            encoding='utf-8',
            timeout=timeout,
            cwd=str(PROJECT_ROOT)
        )
        
        # Codex가 시작될 때까지 대기 (프롬프트가 나타날 때까지)
        print(f"[2/4] Codex 프롬프트 대기 중...")
        try:
            # Codex의 프롬프트 패턴을 찾기 (일반적으로 ">" 또는 "codex>" 같은 형태)
            # 여러 패턴 시도
            child.expect([
                pexpect.EOF,
                '>',
                'codex>',
                '>>>',
                r'[\r\n]+',  # 줄바꿈
            ], timeout=10)
        except pexpect.TIMEOUT:
            # 타임아웃이 발생해도 계속 진행 (이미 준비되었을 수 있음)
            print(f"      (프롬프트 패턴을 찾지 못했지만 계속 진행합니다)")
        
        # 프롬프트 전송
        print(f"[3/4] 프롬프트 전송 중...")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"프롬프트:")
        print(f"{prompt}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"")
        
        # 프롬프트 전송
        child.sendline(prompt)
        
        # 응답 수집
        print(f"[4/4] Codex 응답 대기 중... (최대 {timeout}초)")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"")
        
        response_lines = []
        start_time = time.time()
        
        # 실시간 출력을 위한 버퍼
        buffer = ""
        
        while True:
            try:
                # 짧은 타임아웃으로 계속 읽기
                index = child.expect([
                    pexpect.EOF,
                    pexpect.TIMEOUT,
                    r'.+',  # 어떤 텍스트든
                ], timeout=2)
                
                if index == 0:  # EOF - Codex가 종료됨
                    # 남은 버퍼 읽기
                    remaining = child.before
                    if remaining:
                        buffer += remaining
                        print(remaining, end='', flush=True)
                    break
                
                elif index == 1:  # TIMEOUT
                    # 타임아웃이지만 아직 진행 중일 수 있음
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        print(f"\n[경고] 전체 타임아웃 ({timeout}초) 초과")
                        break
                    # 계속 대기
                    continue
                
                else:  # 텍스트 수신
                    # 버퍼에 추가
                    buffer += child.before + child.after
                    print(child.after, end='', flush=True)
                    
                    # 응답이 완료되었는지 확인
                    # Codex가 완료되면 보통 프롬프트가 다시 나타나거나 EOF가 발생
                    # 또는 특정 패턴으로 끝남
                    if wait_for_completion:
                        # 여러 프롬프트 패턴 확인
                        if any(pattern in buffer for pattern in ['\n>', '\ncodex>', '\n>>>']):
                            # 프롬프트가 다시 나타났다면 응답 완료
                            # 프롬프트 이전까지가 응답
                            break
            
            except KeyboardInterrupt:
                print(f"\n[중단] 사용자에 의해 중단됨")
                break
            except Exception as e:
                print(f"\n[오류] 예외 발생: {e}")
                break
        
        # 응답 정리
        response = buffer.strip()
        
        print(f"")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"응답 수집 완료")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Codex 종료
        try:
            child.close()
        except:
            pass
        
        return response
        
    except pexpect.exceptions.ExceptionPexpect as e:
        print(f"[오류] pexpect 예외: {e}")
        return None
    except Exception as e:
        print(f"[오류] 예외 발생: {e}")
        return None


def main():
    """메인 함수"""
    # 프롬프트 읽기
    prompt_file = PROJECT_ROOT / "docs" / "CODEX_REVIEW_PROMPT.txt"
    
    if prompt_file.exists():
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read()
    else:
        # 기본 프롬프트
        prompt = """Phase 3 개발 기획서를 코드 개발자 관점에서 검토해주세요.

다음 파일들을 참고해주세요:
- docs/PHASE3_DEVELOPMENT_PLAN.md
- docs/PHASE3_DESIGN_DOCUMENT.md
- docs/PHASE3_IMPLEMENTATION_PLAN.md
- docs/PHASE3_REQUIREMENTS.md

특히 다음 사항에 중점을 두어주세요:
1. 아키텍처 설계 검토
2. SOLID 원칙 준수 검토
3. 구현 복잡도 평가
4. 성능 고려사항
5. 개선 제안
6. 잠재적 문제점

구체적이고 실행 가능한 피드백 부탁드립니다."""
    
    # Codex와 대화
    response = codex_interactive_chat(prompt, timeout=600)  # 10분 타임아웃
    
    if response:
        # 응답 저장
        output_file = PROJECT_ROOT / "docs" / "CODEX_INTERACTIVE_RESPONSE.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            f.write("Codex Interactive Session Response\n")
            f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            f.write(f"\n프롬프트:\n{prompt}\n\n")
            f.write(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            f.write(f"\n응답:\n{response}\n")
        
        print(f"")
        print(f"✅ 응답이 저장되었습니다: {output_file}")
        return 0
    else:
        print(f"")
        print(f"❌ 응답을 받지 못했습니다.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
