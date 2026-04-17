import asyncio
import sys
from .config import AgentConfig
from .agent import Agent
from .cli import CLI

async def main():
    config = AgentConfig.load()
    agent = Agent(config)
    cli = CLI(agent)
    await cli.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
