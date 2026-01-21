"""
Function Calling support for Agent flows.
"""

import json
from typing import Any, Dict, List, Optional

from ..function_calling import FunctionRegistry, register_core_functions
from ..llm_wrapper import create_chat_completion_with_tools


class FunctionCallingSupport:
    """Handle function calling with the text LLM and registered tools."""

    def __init__(self, llm_manager, e2e_system=None):
        self.llm_manager = llm_manager
        self.e2e_system = e2e_system
        self.registry = FunctionRegistry()
        self._ready = False
        if e2e_system is not None:
            register_core_functions(self.registry, e2e_system)
            self._ready = True

    def set_e2e_system(self, e2e_system) -> None:
        self.e2e_system = e2e_system
        self.registry = FunctionRegistry()
        self._ready = False
        if e2e_system is not None:
            register_core_functions(self.registry, e2e_system)
            self._ready = True

    def process_query(
        self,
        query: str,
        conversation: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not self._ready:
            return {
                "success": False,
                "error": "Function registry is not initialized",
                "response": "",
                "tool_calls": [],
                "tool_results": [],
            }

        if not self.llm_manager or not self.llm_manager.text_llm:
            return {
                "success": False,
                "error": "Text LLM is not available",
                "response": "",
                "tool_calls": [],
                "tool_results": [],
            }

        messages: List[Dict[str, Any]] = []
        if conversation:
            messages.extend(conversation)

        system_prompt = (
            "You are a security monitoring assistant. Use the provided tools to "
            "answer questions about system status and recent events when needed. "
            "Respond concisely in Korean after tool results are available."
        )
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        try:
            # Function Calling 지원 래퍼 사용
            response = create_chat_completion_with_tools(
                self.llm_manager.text_llm,
                messages=messages,
                tools=self.registry.list_functions(),
                tool_choice="auto",
                temperature=0.2,
                max_tokens=512,
            )
        except Exception as exc:
            # Fallback: tools 없이 일반 호출
            fallback = self.llm_manager.text_llm.create_chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=256,
            )
            content = fallback["choices"][0]["message"].get("content", "").strip()
            return {
                "success": True,
                "response": content,
                "tool_calls": [],
                "tool_results": [],
                "warning": f"Tool calling unavailable: {exc}",
            }

        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls")
        if not tool_calls and message.get("function_call"):
            tool_calls = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": message.get("function_call", {}),
                }
            ]

        tool_results: List[Dict[str, Any]] = []
        response_text = (message.get("content") or "").strip()

        if tool_calls:
            messages.append(message)
            for index, tool_call in enumerate(tool_calls):
                function_call = tool_call.get("function", {})
                name = function_call.get("name")
                raw_args = function_call.get("arguments")
                parsed_args = self._parse_arguments(raw_args)

                try:
                    result = self.registry.call(name, parsed_args)
                except Exception as exc:
                    result = {"ok": False, "error": str(exc)}

                tool_results.append(
                    {
                        "name": name,
                        "arguments": parsed_args,
                        "result": result,
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", f"call_{index}"),
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

            try:
                # Function Calling 결과를 바탕으로 최종 응답 생성
                follow_up = self.llm_manager.text_llm.create_chat_completion(
                    messages=messages,
                    temperature=0.2,
                    max_tokens=512,
                )
                response_text = (
                    follow_up["choices"][0]["message"].get("content", "").strip()
                )
            except Exception as exc:
                response_text = json.dumps(tool_results, ensure_ascii=False)
                return {
                    "success": False,
                    "error": f"Failed to generate final response: {exc}",
                    "response": response_text,
                    "tool_calls": tool_calls,
                    "tool_results": tool_results,
                }

        return {
            "success": True,
            "response": response_text,
            "tool_calls": tool_calls or [],
            "tool_results": tool_results,
        }

    @staticmethod
    def _parse_arguments(raw_args: Any) -> Dict[str, Any]:
        if raw_args is None:
            return {}
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            if not raw_args.strip():
                return {}
            try:
                return json.loads(raw_args)
            except json.JSONDecodeError:
                return {}
        return {}
