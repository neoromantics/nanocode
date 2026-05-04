import json
from typing import List, Dict, Any, Protocol
from .config import AgentConfig
from .model import create_model
from .llm_client import send_request
from .tools import ToolRegistry, ToolResult
from .provider import AnthropicProvider, OpenAIProvider, OpenRouterProvider

class AgentObserver(Protocol):
    def on_text_chunk(self, chunk: str): ...
    def on_tool_start(self, name: str, args_preview: str): ...
    def on_tool_result(self, result_preview: str): ...
    def on_error(self, error_msg: str): ...
    def on_thought_start(self): ...
    def on_thought_end(self): ...

class Agent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.current_model = create_model(config.initial_model, config)
        self.messages: List[Dict[str, Any]] = []
        self.system_prompt = config.system_prompt

    async def process_message(self, user_input: str, observer: AgentObserver):
        self.messages.append({"role": "user", "content": user_input})
        await self.run_agentic_loop(observer)

    def clear_history(self):
        self.messages = []

    def save_session(self, path: str):
        data: Dict[str, Any] = {
            "model": self.current_model.id,
            "messages": self.messages
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_session(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "model" in data:
            self.current_model = create_model(data["model"], self.config)
        if "messages" in data:
            self.messages = data["messages"]

    async def list_available_models(self) -> List[str]:
        providers = []
        if self.config.anthropic_key:
            providers.append(AnthropicProvider(self.config.anthropic_key))
        if self.config.openai_key:
            providers.append(OpenAIProvider(self.config.openai_key, "https://api.openai.com/v1/chat/completions"))
        if self.config.gemini_key:
            url = f"https://generativelanguage.googleapis.com/v1beta/openai/chat/completions?key={self.config.gemini_key}"
            providers.append(OpenAIProvider(self.config.gemini_key, url, "Gemini"))
        if self.config.openrouter_key:
            providers.append(OpenRouterProvider(self.config.openrouter_key))

        all_models = []
        for p in providers:
            try:
                models = await p.fetch_available_models()
                for m in models:
                    all_models.append(f"{p.name()}:{m}")
            except (RuntimeError, ValueError, Exception):
                # Silently skip providers that are unreachable or misconfigured
                continue
        return all_models

    async def switch_model(self, model_id: str) -> str:
        if not model_id:
            raise ValueError("Model ID cannot be empty")
        self.current_model = create_model(model_id, self.config)
        return self.current_model.id

    async def run_agentic_loop(self, observer: AgentObserver):
        while True:
            observer.on_thought_start()
            
            provider = self.current_model.provider
            payload = provider.build_payload(
                self.current_model.id, 
                self.system_prompt, 
                self.messages, 
                True
            )
            
            on_line, finalize = provider.create_sse_handlers(observer.on_text_chunk)
            
            try:
                response = await send_request(
                    provider.api_url(),
                    payload,
                    provider.setup_headers,
                    on_chunk=observer.on_text_chunk,
                    on_line=on_line,
                    finalize=finalize
                )
            except (RuntimeError, ValueError, Exception) as e:
                observer.on_thought_end()
                observer.on_error(str(e))
                break

            observer.on_thought_end()
            raw_resp = response.raw_json
            
            if "error" in raw_resp:
                observer.on_error(json.dumps(raw_resp["error"]))
                break
            
            content_data = provider.normalize_response(raw_resp)
            content_blocks = content_data.get("content", [])
            tool_results: List[Dict[str, Any]] = []

            for block in content_blocks:
                if block.get("type") == "tool_use":
                    tool_name = block["name"]
                    tool_args = block["input"]
                    
                    arg_preview = ""
                    if tool_args:
                        arg_preview = json.dumps(next(iter(tool_args.values())))[:50]
                    
                    observer.on_tool_start(tool_name, arg_preview)
                    
                    tool = ToolRegistry.instance().get_tool(tool_name)
                    if tool:
                        res = await tool.execute(tool_args)
                    else:
                        res = ToolResult(error=f"error: unknown tool {tool_name}")
                    
                    res_str = res.value if res.value is not None else (res.error if res.error is not None else "unknown error")
                    
                    # Truncate preview
                    preview_lines = []
                    for line in res_str.splitlines()[:5]:
                        if len(line) > 120:
                            preview_lines.append(line[:117] + "...")
                        else:
                            preview_lines.append(line)
                    if len(res_str.splitlines()) > 5:
                        preview_lines.append("  ... (truncated output)")
                    
                    observer.on_tool_result("\n".join(preview_lines))
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": res_str
                    })

            self.messages.append({"role": "assistant", "content": content_blocks})
            
            if not tool_results:
                break
            self.messages.append({"role": "user", "content": tool_results})
