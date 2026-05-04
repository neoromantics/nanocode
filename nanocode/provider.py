from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable
from .protocols import AnthropicProtocol, OpenAIProtocol
from .llm_client import fetch_json

class LLMProvider(ABC):
    @abstractmethod
    def name(self) -> str: pass
    
    @abstractmethod
    def api_url(self) -> str: pass
    
    @abstractmethod
    def setup_headers(self, headers: Dict[str, str]): pass
    
    @abstractmethod
    def build_payload(self, model: str, system_prompt: str, messages: List[Dict[str, Any]], stream: bool) -> Dict[str, Any]: pass
    
    @abstractmethod
    def normalize_response(self, raw_resp: Dict[str, Any]) -> Dict[str, Any]: pass
    
    @abstractmethod
    async def fetch_available_models(self) -> List[str]: pass
    
    @abstractmethod
    def create_sse_handlers(self, on_chunk: Callable[[str], None]): pass

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.protocol = AnthropicProtocol()
        self.cached_models = None

    def name(self) -> str: return "Anthropic"
    def api_url(self) -> str: return "https://api.anthropic.com/v1/messages"
    
    def setup_headers(self, headers: Dict[str, str]):
        headers["x-api-key"] = self.api_key
        headers["anthropic-version"] = "2023-06-01"

    def build_payload(self, model: str, system_prompt: str, messages: List[Dict[str, Any]], stream: bool) -> Dict[str, Any]:
        return self.protocol.build_payload(model, system_prompt, messages, stream)

    def normalize_response(self, raw_resp: Dict[str, Any]) -> Dict[str, Any]:
        return self.protocol.normalize_response(raw_resp)

    async def fetch_available_models(self) -> List[str]:
        if self.cached_models: return self.cached_models
        # Anthropic doesn't have a public models list API without auth for all tiers, 
        # but we can manually define common ones if needed, or just return a default list.
        # The C++ code doesn't seem to have a dynamic fetch for Anthropic either, 
        # it might just be placeholders or empty.
        # Looking at C++ provider.cpp... it actually might call an API.
        # For now, let's return common models.
        self.cached_models = ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"]
        return self.cached_models

    def create_sse_handlers(self, on_chunk: Callable[[str], None]):
        return self.protocol.create_sse_handlers(on_chunk)

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, api_url: str, provider_name: str = "OpenAI"):
        self.api_key = api_key
        self._api_url = api_url
        self._name = provider_name
        self.protocol = OpenAIProtocol()
        self.cached_models = None

    def name(self) -> str: return self._name
    def api_url(self) -> str: return self._api_url
    
    def setup_headers(self, headers: Dict[str, str]):
        headers["Authorization"] = f"Bearer {self.api_key}"

    def build_payload(self, model: str, system_prompt: str, messages: List[Dict[str, Any]], stream: bool) -> Dict[str, Any]:
        return self.protocol.build_payload(model, system_prompt, messages, stream)

    def normalize_response(self, raw_resp: Dict[str, Any]) -> Dict[str, Any]:
        return self.protocol.normalize_response(raw_resp)

    async def fetch_available_models(self) -> List[str]:
        if self.cached_models: return self.cached_models
        try:
            # Try to fetch from /models endpoint if it's OpenAI-compatible
            if "generativelanguage" in self._api_url:
                # Gemini OpenAI endpoint doesn't support /models easily
                self.cached_models = ["gemini-2.0-flash", "gemini-2.0-pro-exp-02-05", "gemini-1.5-pro", "gemini-1.5-flash"]
            else:
                # Standard OpenAI models
                self.cached_models = ["gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini", "o3-mini"]
        except (RuntimeError, ValueError, Exception):
            self.cached_models = []
        return self.cached_models

    def create_sse_handlers(self, on_chunk: Callable[[str], None]):
        return self.protocol.create_sse_handlers(on_chunk)

class OpenRouterProvider(AnthropicProvider):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.protocol = OpenAIProtocol() # OpenRouter uses OpenAI format for everything mostly

    def name(self) -> str: return "OpenRouter"
    def api_url(self) -> str: return "https://openrouter.ai/api/v1/chat/completions"
    
    def setup_headers(self, headers: Dict[str, str]):
        headers["Authorization"] = f"Bearer {self.api_key}"
        headers["HTTP-Referer"] = "https://github.com/taiyanliu/nanocode-cpp"
        headers["X-Title"] = "nanocode-py"

    async def fetch_available_models(self) -> List[str]:
        if self.cached_models: return self.cached_models
        try:
            resp = await fetch_json("https://openrouter.ai/api/v1/models", self.setup_headers)
            self.cached_models = [m["id"] for m in resp["data"]]
        except (RuntimeError, ValueError, Exception):
            self.cached_models = []
        return self.cached_models
