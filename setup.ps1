#!/usr/bin/env pwsh
#Requires -Version 7.0

[CmdletBinding()]
param(
    [switch]$SkipModels
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSStyle.OutputRendering = "Ansi"

# --- UI ---
$e = [char]27
$R  = "$e[0m";  $D = "$e[90m"; $W = "$e[97m"; $B = "$e[1m"
$G  = "$e[92m"; $Re = "$e[91m"; $Y = "$e[93m"; $C = "$e[96m"
function Step($n, $total, $msg) { Write-Host "  ${D}[$n/$total]${R} $msg${D}...${R}" }
function Ok($msg)   { Write-Host "  ${G}OK${R}  $D$msg$R" }
function Fail($msg) { Write-Host "  ${Re}FAIL${R}  $msg"; Read-Host "`n  Press Enter to exit"; exit 1 }
function Warn($msg) { Write-Host "  ${Y}!${R}  $msg" }
try { $Host.UI.RawUI.WindowTitle = "SDDj - Setup" } catch {}

$Root = [System.IO.Path]::GetFullPath($PSScriptRoot)
Set-Location -Path $Root

Write-Host "`n  ${B}${W}SDDj${R}  ${D}Setup${R}`n  ${D}$('-' * 36)${R}`n"

# --- Paths ---
$ServerDir  = Join-Path $Root "server"
$ScriptsDir = Join-Path $Root "scripts"
$VenvPython = Join-Path $ServerDir ".venv\Scripts\python.exe"

# --- 1. Check uv ---
Step 1 8 "Checking uv package manager"
if (-not (Get-Command -Name "uv" -ErrorAction Ignore)) {
    Fail "uv not found — install from https://docs.astral.sh/uv/"
}
$uvVer = (uv --version 2>$null) -join ""
Ok $uvVer

# --- 2. Install dependencies ---
Step 2 8 "Installing dependencies"
Push-Location $ServerDir
try {
    $uvOut = uv sync --locked 2>&1
    if ($LASTEXITCODE -ne 0) {
        Warn "Locked sync failed — retrying with resolution..."
        $uvOut = uv sync 2>&1
        if ($LASTEXITCODE -ne 0) {
            Fail "Dependency install failed:`n$($uvOut | Out-String)"
        }
    }
} finally {
    Pop-Location
}
Ok "Dependencies installed"

# --- 3. Validate environment ---
Step 3 8 "Validating environment"
if (-not (Test-Path $VenvPython)) {
    Fail "Python executable not found in .venv"
}
Ok "Virtual environment valid"

# --- 4. Download models ---
Step 4 8 "Provisioning models"
if ($SkipModels) {
    Ok "Skipped (-SkipModels)"
} else {
    $dlScript = Join-Path $ScriptsDir "download_models.py"
    # Execute natively to preserve \r buffer flushing and prevent console newline spam
    Write-Host ""
    & $VenvPython $dlScript --all
    Write-Host ""
    if ($LASTEXITCODE -ne 0) {
        Warn "Model download had errors. Re-run: $VenvPython $dlScript --all"
    } else {
        Ok "Models ready"
    }
}

# --- 5. Build extension ---
Step 5 8 "Building Aseprite extension"
$buildScript = Join-Path $ScriptsDir "build_extension.py"
$buildOut = & $VenvPython $buildScript 2>&1
if ($LASTEXITCODE -ne 0) { Fail "Extension build failed:`n$($buildOut | Out-String)" }
Ok "Extension built"

# --- 6. Deploy extension ---
Step 6 8 "Deploying extension"
if (-not $env:APPDATA) { Fail "APPDATA not set (required for Aseprite extension deploy)" }
$AseData    = Join-Path $env:APPDATA "Aseprite"
$AseExt     = Join-Path $AseData "extensions\sddj"
$AseScripts = Join-Path $AseData "scripts"

# Remove legacy flat scripts (pre-extension era)
foreach ($f in @("sddj.lua", "json.lua")) {
    $p = Join-Path $AseScripts $f
    if (Test-Path $p) { Remove-Item $p -Force }
}

# Remove old extension (guarded: Aseprite file locks cause access-denied)
if (Test-Path $AseExt) {
    try {
        Remove-Item $AseExt -Recurse -Force
    } catch {
        Fail "Cannot remove old extension — close Aseprite first"
    }
}

$null = New-Item -Path (Join-Path $AseExt "scripts") -ItemType Directory -Force
$null = New-Item -Path (Join-Path $AseExt "keys") -ItemType Directory -Force

$ExtSrc = Join-Path $Root "extension"
Copy-Item (Join-Path $ExtSrc "package.json") $AseExt -Force
Copy-Item (Join-Path $ExtSrc "scripts\*.lua") (Join-Path $AseExt "scripts") -Force
Copy-Item (Join-Path $ExtSrc "keys\*") (Join-Path $AseExt "keys") -Force

$luaFiles = @(Get-ChildItem -Path (Join-Path $AseExt "scripts") -Filter "*.lua" -ErrorAction SilentlyContinue)
if ($luaFiles.Count -eq 0) { Fail "No Lua files deployed — extension source may be missing" }
Ok "$($luaFiles.Count) Lua files deployed"

# --- 7. Environment config ---
Step 7 8 "Checking environment config"
$EnvFile    = Join-Path $ServerDir ".env"
$EnvExample = Join-Path $ServerDir ".env.example"

if (-not (Test-Path $EnvFile)) {
    if (Test-Path $EnvExample) {
        Copy-Item $EnvExample $EnvFile -Force
        Ok "Created .env from template"
    } else {
        Ok "Using default settings"
    }
} else {
    Ok ".env exists — keeping config"
}

# --- 8. Verify installation ---
Step 8 8 "Verifying installation"
try {
    # Metadata-only check (avoids torch/CUDA import chain, ~50ms vs ~500ms)
    $ver = & $VenvPython -c "from importlib.metadata import version; print(version('sddj-server'))" 2>$null
    if ($LASTEXITCODE -eq 0 -and $ver) { Ok "SDDj v$ver" } else { Warn "Package verification failed" }
} catch {
    Warn "Package verification failed"
}

Write-Host "`n  ${D}$('-' * 36)${R}`n  ${G}${B}Setup complete.${R}`n"
Write-Host "  ${W}Next:${R}  Run ${C}.\start.ps1${R} to launch SDDj"
Write-Host "  ${W}Edit:${R}  ${D}server\.env${R} to customize settings`n"
Read-Host "  Press Enter to exit"
