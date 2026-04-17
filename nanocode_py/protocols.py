import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable
from .llm_client import LLMResponse
from .tools import ToolRegistry

class LLMProtocol(ABC):
    @abstractmethod
    def build_payload(self, model: str, system_prompt: str, messages: List[Dict[str, Any]], stream: bool) -> Dict[str, Any]:
        pass

    @abstractmethod
    def normalize_response(self, raw_resp: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def create_sse_handlers(self, on_chunk: Callable[[str], None]):
        pass

class OpenAIProtocol(LLMProtocol):
    def build_payload(self, model: str, system_prompt: str, messages: List[Dict[str, Any]], stream: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": model}
        if stream:
            payload["stream"] = True
        
        # Tools translation
        openai_tools = []
        for tool_schema in ToolRegistry.instance().get_all_schemas():
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool_schema["name"],
                    "description": tool_schema["description"],
                    "parameters": tool_schema["input_schema"]
                }
            })
        payload["tools"] = openai_tools
        
        # Messages translation
        msgs: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "user":
                if isinstance(content, str):
                    msgs.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "tool_result":
                            msgs.append({
                                "role": "tool",
                                "tool_call_id": item["tool_use_id"],
                                "content": str(item["content"])
                            })
            elif role == "assistant":
                if isinstance(content, list):
                    text_content = ""
                    tool_calls = []
                    for block in content:
                        if block["type"] == "text":
                            text_content += block["text"]
                        elif block["type"] == "tool_use":
                            tool_calls.append({
                                "id": block["id"],
                                "type": "function",
                                "function": {
                                    "name": block["name"],
                                    "arguments": json.dumps(block["input"])
                                }
                            })
                    asst_msg: Dict[str, Any] = {"role": "assistant"}
                    if text_content:
                        asst_msg["content"] = text_content
                    if tool_calls:
                        asst_msg["tool_calls"] = tool_calls
                    msgs.append(asst_msg)
                else:
                    msgs.append({"role": "assistant", "content": content})
        payload["messages"] = msgs
        return payload

    def normalize_response(self, raw_resp: Dict[str, Any]) -> Dict[str, Any]:
        content_blocks = []
        if "choices" in raw_resp and raw_resp["choices"]:
            message = raw_resp["choices"][0]["message"]
            if message.get("content"):
                content_blocks.append({"type": "text", "text": message["content"]})
            if message.get("tool_calls"):
                for tc in message["tool_calls"]:
                    if tc["type"] == "function":
                        func = tc["function"]
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": func["name"],
                            "input": json.loads(func["arguments"])
                        })
        return {"content": content_blocks}

    def create_sse_handlers(self, on_chunk: Callable[[str], None]):
        state: Dict[str, Any] = {
            "final_text": "",
            "tool_calls": [],
            "current_tool": None,
            "current_tool_args": ""
        }
        
        def on_line(data_str: str):
            if data_str == "[DONE]":
                return
            try:
                obj = json.loads(data_str)
            except json.JSONDecodeError:
                return
            
            if "choices" not in obj or not obj["choices"]:
                return
            delta = obj["choices"][0].get("delta", {})
            
            if "content" in delta and delta["content"]:
                text = delta["content"]
                state["final_text"] += text
                on_chunk(text)
            
            if "tool_calls" in delta:
                for tc in delta["tool_calls"]:
                    if "id" in tc:
                        if state["current_tool"]:
                            state["current_tool"]["function"]["arguments"] = state["current_tool_args"]
                            state["tool_calls"].append(state["current_tool"])
                        state["current_tool"] = {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["function"]["name"], "arguments": ""}
                        }
                        state["current_tool_args"] = ""
                    if "function" in tc and "arguments" in tc["function"]:
                        state["current_tool_args"] += tc["function"]["arguments"]

        def finalize() -> LLMResponse:
            if state["current_tool"]:
                state["current_tool"]["function"]["arguments"] = state["current_tool_args"]
                state["tool_calls"].append(state["current_tool"])
            
            msg: Dict[str, Any] = {"role": "assistant"}
            if state["final_text"]:
                msg["content"] = state["final_text"]
            if state["tool_calls"]:
                msg["tool_calls"] = state["tool_calls"]
            return LLMResponse({"choices": [{"message": msg}]})

        return on_line, finalize

class AnthropicProtocol(LLMProtocol):
    def build_payload(self, model: str, system_prompt: str, messages: List[Dict[str, Any]], stream: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": 8192,
            "system": system_prompt,
            "tools": ToolRegistry.instance().get_all_schemas(),
            "messages": messages
        }
        if stream:
            payload["stream"] = True
        return payload

    def normalize_response(self, raw_resp: Dict[str, Any]) -> Dict[str, Any]:
        return raw_resp

    def create_sse_handlers(self, on_chunk: Callable[[str], None]):
        state: Dict[str, Any] = {
            "final_text": "",
            "anthropic_content": [],
            "current_tool": None,
            "current_tool_args": ""
        }
        
        def on_line(data_str: str):
            try:
                obj = json.loads(data_str)
            except json.JSONDecodeError:
                return
            
            t = obj.get("type")
            if t == "content_block_start":
                block = obj["content_block"]
                if block["type"] == "tool_use":
                    state["current_tool"] = {
                        "type": "tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "input": {}
                    }
                    state["current_tool_args"] = ""
            elif t == "content_block_delta":
                delta = obj["delta"]
                if delta["type"] == "text_delta":
                    text = delta["text"]
                    state["final_text"] += text
                    on_chunk(text)
                elif delta["type"] == "input_json_delta":
                    state["current_tool_args"] += delta["partial_json"]
            elif t == "content_block_stop":
                if state["current_tool"]:
                    if state["current_tool_args"]:
                        state["current_tool"]["input"] = json.loads(state["current_tool_args"])
                    state["anthropic_content"].append(state["current_tool"])
                    state["current_tool"] = None

        def finalize() -> LLMResponse:
            content = []
            if state["final_text"]:
                content.append({"type": "text", "text": state["final_text"]})
            content.extend(state["anthropic_content"])
            return LLMResponse({"content": content})

        return on_line, finalize
