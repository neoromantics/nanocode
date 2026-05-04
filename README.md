# nanocode

Minimal AI coding assistant CLI. Supports Anthropic, Gemini, OpenAI, and OpenRouter.

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Setup

```bash
git clone https://github.com/neoromantics/nanocode
cd nanocode
uv sync
```

Create a `.env` file with at least one API key:

```
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
OPENAI_API_KEY=...
OPENROUTER_API_KEY=...
```

## Run

```bash
uv run nanocode

# optional flags
uv run nanocode --model claude-opus-4-5
uv run nanocode --system "You are a helpful assistant."
```

## Install as a global command

```bash
uv tool install .
nanocode
```

Upgrade after pulling changes:

```bash
uv tool upgrade nanocode
```

## Build a standalone binary

Requires the dev dependencies:

```bash
uv sync --dev
uv run pyinstaller --onefile --name nanocode nanocode/__main__.py
```

The binary is written to `dist/nanocode` (or `dist/nanocode.exe` on Windows). You must run the build on each target platform — the output is not cross-compiled.

## Development

```bash
uv sync --dev

# lint
uv run ruff check .

# type check
uv run pyright

# tests
uv run pytest
```
