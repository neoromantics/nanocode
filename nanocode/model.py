from .provider import LLMProvider, AnthropicProvider, OpenAIProvider, OpenRouterProvider
from .config import AgentConfig

class Model:
    def __init__(self, model_id: str, provider: LLMProvider):
        self._id = model_id
        self._provider = provider

    @property
    def id(self) -> str: return self._id
    
    @property
    def provider(self) -> LLMProvider: return self._provider

def create_model(model_id: str, config: AgentConfig) -> Model:
    # Logic to map model_id to provider
    if model_id.startswith("claude-"):
        return Model(model_id, AnthropicProvider(config.anthropic_key or ""))
    elif "gemini" in model_id:
        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        if config.gemini_key:
            url += f"?key={config.gemini_key}"
        return Model(model_id, OpenAIProvider(config.gemini_key or "", url, "Gemini"))
    elif model_id.startswith("gpt-") or model_id.startswith("o1-") or model_id.startswith("o3-"):
        return Model(model_id, OpenAIProvider(config.openai_key or "", "https://api.openai.com/v1/chat/completions", "OpenAI"))
    elif "/" in model_id: # Likely OpenRouter model e.g. "anthropic/claude-3.5-sonnet"
        return Model(model_id, OpenRouterProvider(config.openrouter_key or ""))
    else:
        # Default to Gemini if unknown and key exists, or just pick one
        if config.gemini_key:
            url = f"https://generativelanguage.googleapis.com/v1beta/openai/chat/completions?key={config.gemini_key}"
            return Model(model_id, OpenAIProvider(config.gemini_key, url, "Gemini"))
        return Model(model_id, OpenAIProvider(config.openai_key or "", "https://api.openai.com/v1/chat/completions", "OpenAI"))
