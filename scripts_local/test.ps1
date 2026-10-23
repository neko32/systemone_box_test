# Lint, format check, unit tests with coverage (HTML in docs/coverage). Add -Integration to hit the real API.
param([switch]$Integration)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
uv run ruff check .; if ($LASTEXITCODE) { exit $LASTEXITCODE }
uv run ruff format --check .; if ($LASTEXITCODE) { exit $LASTEXITCODE }
uv run pytest --cov --cov-report=term-missing --cov-report=html; if ($LASTEXITCODE) { exit $LASTEXITCODE }
if ($Integration) { uv run pytest -m integration -v; exit $LASTEXITCODE }
