import asyncio
import sys
from .config import AgentConfig
from .agent import Agent
from .cli import CLI

async def _main():
    config = AgentConfig.load()
    agent = Agent(config)
    cli = CLI(agent)
    await cli.run()

def main():
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    main()
