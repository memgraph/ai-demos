#!/bin/bash

curl -LsSf https://astral.sh/uv/install.sh | sh

uv init || true
uv add mcp[cli]
uv add ollama
uv add colorama
uv add langchain_core
uv add langgraph
uv add langchain_openai
uv add langchain_community
uv add neo4j
uv add black

uv run mcp dev server.py

# NOTE: uv run black *.py
