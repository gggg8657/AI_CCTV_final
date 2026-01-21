"""
LLM Wrapper for Function Calling
=================================

llama.cpp의 create_chat_completion을 Function Calling 지원으로 래핑
"""

from typing import Any, Dict, List, Optional


def create_chat_completion_with_tools(
    llm,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    **kwargs
) -> Dict[str, Any]:
    """
    llama.cpp의 create_chat_completion을 tools 지원으로 래핑
    
    Args:
        llm: llama.cpp Llama 인스턴스
        messages: 대화 메시지 리스트
        tools: Function Calling 도구 정의 리스트
        tool_choice: 도구 선택 모드 ("auto", "none", "required")
        **kwargs: 기타 create_chat_completion 파라미터
    
    Returns:
        OpenAI-compatible 응답 딕셔너리
    """
    # tools가 없으면 일반 호출
    if not tools:
        return llm.create_chat_completion(messages=messages, **kwargs)
    
    # llama.cpp의 create_chat_completion이 tools를 지원하는지 확인
    # Qwen3는 tools 파라미터를 지원함
    try:
        # tools를 OpenAI 형식으로 변환
        # llama.cpp는 tools를 직접 지원할 수 있음
        response = llm.create_chat_completion(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs
        )
        return response
    except TypeError:
        # tools 파라미터를 지원하지 않는 경우
        # 메시지에 tools 정보를 포함시켜서 처리
        # 이는 fallback 방식이며, 실제로는 llama.cpp가 tools를 지원해야 함
        system_message = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_message = messages.pop(i)
                break
        
        # System 메시지에 tools 정보 추가
        if system_message:
            tools_desc = "\n".join([
                f"- {tool['function']['name']}: {tool['function']['description']}"
                for tool in tools
            ])
            system_content = system_message.get("content", "")
            system_message["content"] = f"{system_content}\n\nAvailable tools:\n{tools_desc}"
            messages.insert(0, system_message)
        else:
            tools_desc = "\n".join([
                f"- {tool['function']['name']}: {tool['function']['description']}"
                for tool in tools
            ])
            messages.insert(0, {
                "role": "system",
                "content": f"Available tools:\n{tools_desc}"
            })
        
        # 일반 호출 (LLM이 tools를 자연어로 이해)
        return llm.create_chat_completion(messages=messages, **kwargs)
