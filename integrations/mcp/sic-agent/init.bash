#!/bin/bash

set -euo pipefail

curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run python sic_classification.py
