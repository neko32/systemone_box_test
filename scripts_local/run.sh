#!/usr/bin/env bash
# Usage: scripts_local/run.sh [live|offline] [extra neko-box args...]
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-live}"; shift || true
if [[ -z "${API_KEY_JEV:-}" && ! -f .env ]]; then
  echo "API_KEY_JEV is not set (export it or create .env from .env.example)" >&2
  exit 1
fi
uv sync
uv run neko-box --mode "$MODE" "$@"
