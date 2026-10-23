#!/usr/bin/env bash
# Lint, format check, unit tests with coverage (HTML in docs/coverage). Add --integration to hit the real API.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov --cov-report=term-missing --cov-report=html
if [[ "${1:-}" == "--integration" ]]; then uv run pytest -m integration -v; fi
