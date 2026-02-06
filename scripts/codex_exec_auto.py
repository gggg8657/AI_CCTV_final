#!/usr/bin/env python3
"""
Codex CLI exec 명령어를 사용하여 자동으로 대화
non-interactive 모드이지만 더 안정적
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def codex_exec_chat(prompt: str) -> str:
    """
    codex exec 명령어를 사용하여 대화
    
    Args:
        prompt: Codex에게 보낼 프롬프트
    
    Returns:
        Codex의 응답
    """
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Codex CLI exec 모드 실행")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"프로젝트 디렉토리: {PROJECT_ROOT}")
    print(f"")
    
    try:
        # codex exec 실행
        print(f"[1/2] Codex exec 실행 중...")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"프롬프트:")
        print(f"{prompt[:200]}...")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"")
        
        # codex exec 실행 (실시간 출력)
        print(f"[2/2] Codex 응답 (실시간):")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"")
        
        # 실시간 출력을 위해 Popen 사용
        process = subprocess.Popen(
            ['codex', 'exec', prompt],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1,  # 라인 버퍼링
            universal_newlines=True
        )
        
        # 실시간으로 출력하면서 수집
        response_lines = []
        for line in process.stdout:
            print(line, end='', flush=True)  # 실시간 출력
            response_lines.append(line)
        
        # 프로세스 종료 대기
        process.wait()
        
        response = ''.join(response_lines)
        
        print(f"")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"응답 수집 완료 (return code: {process.returncode})")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        if process.returncode == 0:
            return response
        else:
            print(f"⚠️  오류 발생 (return code: {process.returncode})")
            return response
        
    except subprocess.TimeoutExpired:
        print(f"[오류] 타임아웃 발생 (10분 초과)")
        return None
    except Exception as e:
        print(f"[오류] 예외 발생: {e}")
        return None


def main():
    """메인 함수"""
    # 프롬프트 읽기 (코드 리뷰용)
    prompt_file = PROJECT_ROOT / "docs" / "CODEX_CODE_REVIEW_PROMPT.txt"
    
    # 코드 리뷰 프롬프트가 없으면 기본 프롬프트 사용
    if not prompt_file.exists():
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
    response = codex_exec_chat(prompt)
    
    if response:
        # 응답 저장
        output_file = PROJECT_ROOT / "docs" / "CODEX_EXEC_RESPONSE.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            f.write("Codex exec Response\n")
            f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            f.write(f"\n프롬프트:\n{prompt}\n\n")
            f.write(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            f.write(f"\n응답:\n{response}\n")
        
        print(f"")
        print(f"✅ 응답이 저장되었습니다: {output_file}")
        print(f"응답 길이: {len(response)} 문자")
        print(f"")
        print(f"응답 미리보기 (처음 500자):")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(response[:500])
        if len(response) > 500:
            print(f"\n... (총 {len(response)} 문자, 전체 내용은 {output_file} 참조)")
        return 0
    else:
        print(f"")
        print(f"❌ 응답을 받지 못했습니다.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
