# ─────────────────────────────────────────────────────────────
# SDDj — Version Bump (Single Source of Truth)
#
# Usage:  .\bump_version.ps1 0.9.54
#
# Updates version in all 3 canonical locations:
#   1. server/pyproject.toml       (Python package)
#   2. extension/package.json      (Aseprite extension)
#   3. extension/scripts/sddj_state.lua  (Lua runtime)
#
# Python __init__.py reads from importlib.metadata at runtime,
# so it does NOT need manual bumping.
# ─────────────────────────────────────────────────────────────

param(
    [Parameter(Mandatory=$true, Position=0)]
    [ValidatePattern('^\d+\.\d+\.\d+(-\w+)?$')]
    [string]$Version
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# ── 1. pyproject.toml ────────────────────────────────────────
$pyproject = Join-Path $root "server/pyproject.toml"
$content = Get-Content $pyproject -Raw
$content = $content -replace '(?m)^version\s*=\s*"[^"]+"', "version = ""$Version"""
Set-Content $pyproject $content -NoNewline
Write-Host "[OK] pyproject.toml -> $Version" -ForegroundColor Green

# ── 2. extension/package.json ────────────────────────────────
$pkgjson = Join-Path $root "extension/package.json"
$json = Get-Content $pkgjson -Raw | ConvertFrom-Json
$json.version = $Version
$json | ConvertTo-Json -Depth 10 | Set-Content $pkgjson
Write-Host "[OK] package.json   -> $Version" -ForegroundColor Green

# ── 3. extension/scripts/sddj_state.lua ─────────────────────
$lua = Join-Path $root "extension/scripts/sddj_state.lua"
$content = Get-Content $lua -Raw
$content = $content -replace 'PT\.VERSION\s*=\s*"[^"]+"', "PT.VERSION = ""$Version"""
Set-Content $lua $content -NoNewline
Write-Host "[OK] sddj_state.lua -> $Version" -ForegroundColor Green

# ── 4. uv.lock (regenerate) ─────────────────────────────────
Push-Location (Join-Path $root "server")
try {
    uv lock --quiet 2>$null
    Write-Host "[OK] uv.lock        -> synced" -ForegroundColor Green
} catch {
    Write-Host "[WARN] uv lock failed (run manually)" -ForegroundColor Yellow
}
Pop-Location

Write-Host ""
Write-Host "Version bumped to $Version across all files." -ForegroundColor Cyan
Write-Host "Next: update CHANGELOG.md, then: git add -A && git commit -m 'v$Version' && git tag v$Version && git push --follow-tags"
