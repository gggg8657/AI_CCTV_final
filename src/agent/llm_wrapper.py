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
    model_name: Optional[str] = None,
    is_api_mode: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    LLM의 create_chat_completion을 tools 지원으로 래핑
    Local 모드 (llama.cpp)와 API 모드 (OpenAI-compatible) 모두 지원
    
    Args:
        llm: llama.cpp Llama 인스턴스 또는 OpenAI 클라이언트
        messages: 대화 메시지 리스트
        tools: Function Calling 도구 정의 리스트
        tool_choice: 도구 선택 모드 ("auto", "none", "required")
        model_name: API 모드에서 사용할 모델 이름
        is_api_mode: API 모드 여부
        **kwargs: 기타 create_chat_completion 파라미터
    
    Returns:
        OpenAI-compatible 응답 딕셔너리
    """
    if is_api_mode:
        # API 모드: OpenAI-compatible API 사용
        try:
            if tools:
                response = llm.chat.completions.create(
                    model=model_name or "Qwen/Qwen3-8B",
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    **kwargs
                )
            else:
                response = llm.chat.completions.create(
                    model=model_name or "Qwen/Qwen3-8B",
                    messages=messages,
                    **kwargs
                )
            # OpenAI 응답을 dict 형식으로 변환
            return {
                "choices": [{
                    "message": {
                        "content": response.choices[0].message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in (response.choices[0].message.tool_calls or [])
                        ]
                    }
                }]
            }
        except Exception as e:
            # API 호출 실패 시 fallback
            logging.warning(f"API 호출 실패, fallback: {e}")
            if tools:
                # tools 정보를 메시지에 포함
                tools_desc = "\n".join([
                    f"- {tool['function']['name']}: {tool['function']['description']}"
                    for tool in tools
                ])
                messages.insert(0, {
                    "role": "system",
                    "content": f"Available tools:\n{tools_desc}"
                })
            response = llm.chat.completions.create(
                model=model_name or "Qwen/Qwen3-8B",
                messages=messages,
                **kwargs
            )
            return {
                "choices": [{
                    "message": {
                        "content": response.choices[0].message.content
                    }
                }]
            }
    else:
        # Local 모드: llama.cpp 사용
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
