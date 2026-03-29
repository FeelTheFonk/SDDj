#!/usr/bin/env pwsh
#Requires -Version 7.0

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true, Position=0)]
    [ValidatePattern('^\d+\.\d+\.\d+(([ab]|rc)\d+)?$')]
    [string]$Version
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = [System.IO.Path]::GetFullPath($PSScriptRoot)

# ── Validate all targets exist before modifying any ──
$PyProject = Join-Path $Root "server\pyproject.toml"
$PkgJson   = Join-Path $Root "extension\package.json"
$Lua       = Join-Path $Root "extension\scripts\sddj_state.lua"
$Readme    = Join-Path $Root "README.md"
$ServerDir = Join-Path $Root "server"
$Changelog = Join-Path $Root "CHANGELOG.md"

foreach ($f in @($PyProject, $PkgJson, $Lua, $Readme, $ServerDir)) {
    if (-not (Test-Path $f)) {
        Write-Host "[FAIL] Required path not found: $f" -ForegroundColor Red
        exit 1
    }
}

# ── 1. pyproject.toml ──
$Content = Get-Content $PyProject -Raw
$Content = $Content -replace '(?m)^version\s*=\s*"[^"]+"', "version = ""$Version"""
Set-Content $PyProject -Value $Content -NoNewline
Write-Host "[OK] pyproject.toml -> $Version" -ForegroundColor Green

# ── 2. package.json ──
$Content = Get-Content $PkgJson -Raw
$Content = $Content -replace '(?m)^(\s*"version"\s*:\s*)"[^"]+"(.*)$', "`$1""$Version""`$2"
Set-Content $PkgJson -Value $Content -NoNewline
Write-Host "[OK] package.json   -> $Version" -ForegroundColor Green

# ── 3. sddj_state.lua ──
$Content = Get-Content $Lua -Raw
$Content = $Content -replace 'PT\.VERSION\s*=\s*"[^"]+"', "PT.VERSION = ""$Version"""
Set-Content $Lua -Value $Content -NoNewline
Write-Host "[OK] sddj_state.lua -> $Version" -ForegroundColor Green

# ── 4. README.md ──
$Content = Get-Content $Readme -Raw
$Content = $Content -replace '(?m)^(# SDDj v)\S+', "`${1}$Version"
Set-Content $Readme -Value $Content -NoNewline
Write-Host "[OK] README.md      -> $Version" -ForegroundColor Green

# ── 5. CHANGELOG.md template ──
if (Test-Path $Changelog) {
    $clContent = Get-Content $Changelog -Raw
    if ($clContent -notmatch [regex]::Escape("## [$Version]")) {
        $date = Get-Date -Format "yyyy-MM"
        $template = "## [$Version] — $date`n### [Title]`n`n#### Added`n- `n`n#### Fixed`n- `n`n"
        $clContent = $clContent -replace '(# Changelog\r?\n)', "`$1$template"
        Set-Content $Changelog -Value $clContent -NoNewline
        Write-Host "[OK] CHANGELOG.md   -> template inserted" -ForegroundColor Green
    } else {
        Write-Host "[OK] CHANGELOG.md   -> entry exists" -ForegroundColor Green
    }
}

# ── 6. uv.lock ──
Push-Location $ServerDir
try {
    $lockOut = uv lock --quiet 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] uv lock failed: $($lockOut | Out-String)" -ForegroundColor Yellow
    } else {
        Write-Host "[OK] uv.lock        -> synced" -ForegroundColor Green
    }
} catch {
    Write-Host "[WARN] uv lock failed (run manually)" -ForegroundColor Yellow
} finally {
    Pop-Location
}

# ── 7. Post-bump verification ──
$esc = [regex]::Escape($Version)
$verifyPy   = (Get-Content $PyProject -Raw) -match "version\s*=\s*`"$esc`""
$verifyPkg  = (Get-Content $PkgJson -Raw)   -match "`"version`"\s*:\s*`"$esc`""
$verifyLua  = (Get-Content $Lua -Raw)        -match "PT\.VERSION\s*=\s*`"$esc`""
$verifyRead = (Get-Content $Readme -Raw)     -match "# SDDj v$esc"

if (-not ($verifyPy -and $verifyPkg -and $verifyLua -and $verifyRead)) {
    Write-Host "[WARN] Version replacement may have failed in one or more files. Verify manually." -ForegroundColor Yellow
}

Write-Host "`nVersion bumped to $Version across all files." -ForegroundColor Cyan
Write-Host "Next: update CHANGELOG.md, then: git add -A && git commit -m 'v$Version' && git tag v$Version && git push --follow-tags"
