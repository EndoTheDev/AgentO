# AgentO

A lightweight AI agent that chats with Ollama and can use tools. ~300 lines of Python.

## ⚠️ Security Warning

Tools can **execute any shell command** and **read/write any file**. Only run this with models you trust, in environments you control.

## Quick Start

1. Install: `uv sync`
2. Copy `.env.example` → `.env` and set your model

## Usage

### Interactive Mode

Run: `uv run main.py` (interactive mode)

### One-Shot Mode

Run: `uv run main.py --message "Your question here"` (one-shot)

## What It Does

- **Streaming responses** from Ollama in real-time
- **Tool calls**: bash commands, file read/write
- **Interactive mode** with persistent context (`/clear` to reset)

## Configuration

| Variable                 | Default              | Description                                                 |
| ------------------------ | -------------------- | ----------------------------------------------------------- |
| `AGENTO_OLLAMA_MODEL`    | _required_           | Which model to use (e.g. `kimi-k2.5`)                       |
| `AGENTO_OLLAMA_BASE_URL` | `https://ollama.com` | Ollama server URL                                           |
| `AGENTO_OLLAMA_API_KEY`  | _empty_              | request [Ollama API Keys](https://ollama.com/settings/keys) |

## Tech Stack

- `httpx` for HTTP streaming
- `python-dotenv` for config
- Native Ollama API (no extra SDKs)
