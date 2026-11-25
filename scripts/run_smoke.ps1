<#
.SYNOPSIS
  Offline end-to-end smoke run on a SYNTHETIC fixture (no network, no credentials).
.DESCRIPTION
  Uses the active Python (virtualenv/conda) if one is active, else .venv\Scripts\python.exe, else `python`
  on PATH. Runs `python -m windpower smoke` and exits with its exit code (0 = all hard checks passed).
.PARAMETER DataDir
  Optional root for cache/interim/outputs (default: <repo>\data, which is git-ignored).
#>
[CmdletBinding()]
param(
    [string]$DataDir = ""
)

$repo = Split-Path -Parent $PSScriptRoot
$venvPy = Join-Path $repo ".venv\Scripts\python.exe"

$py = $null
if ($env:VIRTUAL_ENV -or $env:CONDA_PREFIX) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $py = $cmd.Source }
}
if (-not $py -and (Test-Path $venvPy)) { $py = $venvPy }
if (-not $py) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $py = $cmd.Source }
}
if (-not $py) {
    Write-Host "No Python found: activate an environment, create .venv, or install Python 3.11+." -ForegroundColor Red
    exit 2
}

Set-Location $repo
# Work even if the package is not pip-installed (editable install is still recommended).
& $py -c "import windpower" 2>$null
if ($LASTEXITCODE -ne 0) {
    $env:PYTHONPATH = (Join-Path $repo "src") + $(if ($env:PYTHONPATH) { ";" + $env:PYTHONPATH } else { "" })
}

$argList = @("-m", "windpower", "smoke")
if ($DataDir) { $argList += @("--data-dir", $DataDir) }
Write-Host "Python: $py"
& $py @argList
$code = $LASTEXITCODE
if ($code -eq 0) { Write-Host "run_smoke.ps1: PASS" -ForegroundColor Green } else { Write-Host "run_smoke.ps1: FAIL (exit $code)" -ForegroundColor Red }
exit $code
