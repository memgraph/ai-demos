#!/bin/bash

curl -LsSf https://astral.sh/uv/install.sh | sh
uv init || true
uv add "mcp[cli]"
uv add ollama

uv run mcp dev server.py
