import os
import argparse
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class AgentConfig:
    gemini_key: Optional[str] = None
    anthropic_key: Optional[str] = None
    openrouter_key: Optional[str] = None
    openai_key: Optional[str] = None
    initial_model: str = "gemini-2.5-flash"
    system_prompt: str = "Concise coding assistant."

    @classmethod
    def load(cls) -> "AgentConfig":
        load_dotenv()
        
        parser = argparse.ArgumentParser(description="nanocode-py: AI coding assistant")
        parser.add_argument("--model", type=str, help="Initial model to use")
        parser.add_argument("--system", type=str, help="System prompt")
        args = parser.parse_args()

        return cls(
            gemini_key=os.getenv("GEMINI_API_KEY"),
            anthropic_key=os.getenv("ANTHROPIC_API_KEY"),
            openrouter_key=os.getenv("OPENROUTER_API_KEY"),
            openai_key=os.getenv("OPENAI_API_KEY"),
            initial_model=args.model or os.getenv("DEFAULT_MODEL", "gemini-2.0-flash"),
            system_prompt=args.system or "Concise coding assistant."
        )
