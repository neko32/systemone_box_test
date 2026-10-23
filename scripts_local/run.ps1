# Usage: scripts_local/run.ps1 [-Mode live|offline] [extra neko-box args...]
param(
    [ValidateSet("live", "offline")][string]$Mode = "live",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (-not $env:API_KEY_JEV -and -not (Test-Path .env)) {
    Write-Error "API_KEY_JEV is not set (set it or create .env from .env.example)"
}
uv sync
uv run neko-box --mode $Mode @Rest
