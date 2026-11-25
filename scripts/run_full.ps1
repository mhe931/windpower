<#
.SYNOPSIS
  Full real-data run (Windows PowerShell + Conda): env setup, credential checks, all stages.
.DESCRIPTION
  1. checks that conda is available (or -UseActivePython to use the active Python instead)
  2. checks credentials WITHOUT printing them:
       FINGRID_API_KEY       environment variable or a line in an untracked .env
       CDS (ERA5 download)   CDSAPI_URL + CDSAPI_KEY, or ~\.cdsapirc
     The ArcGIS/Overpass fleet inventories need no credentials (network only).
  3. creates/updates the conda env from environment.yml
  4. runs `python -m windpower all --mode full` (ERA5 download via cdsapi, Fingrid fetch with rate limiting,
     fleet fetch/cache, simulation, calibration, validation, manifest, site export)

  Missing inputs are listed and the script exits with code 2 (distinct from validation failure = 1);
  it never substitutes synthetic data.
.PARAMETER CheckOnly
  (alias -DryRun) only report what is available and what would run; exit 2 if anything is unavailable.
.PARAMETER Refresh
  Re-download raw inputs even if cached (needs credentials and network).
.PARAMETER SkipEnv
  Do not create/update the conda env (use the existing one).
.PARAMETER UseActivePython
  Skip conda entirely and use the active/`python` on PATH (e.g. a venv with requirements.txt installed).
.PARAMETER EnvFile
  Path of the untracked .env to search for FINGRID_API_KEY (default <repo>\.env).
#>
[CmdletBinding()]
param(
    [Alias("DryRun")][switch]$CheckOnly,
    [switch]$Refresh,
    [switch]$SkipEnv,
    [switch]$UseActivePython,
    [string]$EnvName = "windpower",
    [string]$DataDir = "",
    [string]$EnvFile = "",
    [string]$Start = "",
    [string]$End = ""
)

$repo = Split-Path -Parent $PSScriptRoot
if (-not $EnvFile) { $EnvFile = Join-Path $repo ".env" }
$unavailable = New-Object System.Collections.Generic.List[string]

function Test-DotEnvKey([string]$path, [string]$name) {
    if (-not (Test-Path $path)) { return $false }
    foreach ($line in Get-Content $path) {
        if ($line -match "^\s*$name\s*=\s*\S+") { return $true }
    }
    return $false
}

Write-Host "== windpower full run: input check =="

# conda
$conda = $null
if (-not $UseActivePython) {
    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if ($conda) { Write-Host "[ok] conda: $($conda.Source)" }
    else { $unavailable.Add("conda: not found on PATH (install Miniconda/Miniforge, or re-run with -UseActivePython)"); Write-Host "[UNAVAILABLE] conda" -ForegroundColor Yellow }
}

# Fingrid key (value is never printed)
if ($env:FINGRID_API_KEY -or (Test-DotEnvKey $EnvFile "FINGRID_API_KEY")) { Write-Host "[ok] FINGRID_API_KEY is set (environment or $EnvFile)" }
else { $unavailable.Add("FINGRID_API_KEY: not set in the environment or in $EnvFile (copy .env.example to .env and fill it in)"); Write-Host "[UNAVAILABLE] FINGRID_API_KEY" -ForegroundColor Yellow }

# CDS credentials
$cdsRc = if ($env:CDSAPI_RC) { $env:CDSAPI_RC } else { Join-Path $env:USERPROFILE ".cdsapirc" }
if (($env:CDSAPI_URL -and $env:CDSAPI_KEY) -or ((Test-Path $cdsRc) -and (Get-Item $cdsRc).Length -gt 0)) { Write-Host "[ok] CDS credentials (CDSAPI_URL/CDSAPI_KEY or $cdsRc)" }
else { $unavailable.Add("CDS credentials: set CDSAPI_URL and CDSAPI_KEY, or create $cdsRc (https://cds.climate.copernicus.eu)"); Write-Host "[UNAVAILABLE] CDS credentials" -ForegroundColor Yellow }

Write-Host "[info] fleet inventories (OSM Overpass + ArcGIS) need network access but no credentials; cached under data\cache\real\fleet"

$allArgs = @("-m", "windpower", "all", "--mode", "full")
if ($Refresh) { $allArgs += "--refresh" }
if ($DataDir) { $allArgs += @("--data-dir", $DataDir) }
if ($Start) { $allArgs += @("--start", $Start) }
if ($End) { $allArgs += @("--end", $End) }

if ($CheckOnly) {
    Write-Host "== check only: nothing is installed or run =="
    if ($UseActivePython) { Write-Host "would run: python $($allArgs -join ' ')" }
    else {
        if (-not $SkipEnv) { Write-Host "would run: conda env create/update -n $EnvName -f environment.yml" }
        Write-Host "would run: conda run -n $EnvName --no-capture-output python $($allArgs -join ' ')"
    }
}

if ($unavailable.Count -gt 0) {
    Write-Host ""
    Write-Host "INPUTS UNAVAILABLE - the full run cannot start (no synthetic data is substituted):" -ForegroundColor Red
    foreach ($u in $unavailable) { Write-Host "  - $u" -ForegroundColor Red }
    exit 2
}
if ($CheckOnly) { Write-Host "All inputs available."; exit 0 }

Set-Location $repo
if ($UseActivePython) {
    & python @allArgs
    exit $LASTEXITCODE
}

if (-not $SkipEnv) {
    $envs = & conda env list 2>$null | Out-String
    if ($envs -match "(?m)^\s*$([regex]::Escape($EnvName))\s") {
        Write-Host "Updating conda env '$EnvName' from environment.yml"
        & conda env update -n $EnvName -f (Join-Path $repo "environment.yml") --prune
    } else {
        Write-Host "Creating conda env '$EnvName' from environment.yml"
        & conda env create -n $EnvName -f (Join-Path $repo "environment.yml")
    }
    if ($LASTEXITCODE -ne 0) { Write-Host "conda env setup failed (exit $LASTEXITCODE)" -ForegroundColor Red; exit 3 }
}

& conda run -n $EnvName --no-capture-output python @allArgs
exit $LASTEXITCODE
