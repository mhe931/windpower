<#
.SYNOPSIS
  Start the local Finland wind simulator website (Python stdlib server; no network at runtime).
.DESCRIPTION
  Uses the active Python (virtualenv/conda) if one is active, else .venv\Scripts\python.exe, else `python` on PATH,
  and runs `python -m windpower serve`. If the site artifact does not exist the command prints how to generate it
  and exits with code 2; nothing is synthesised silently. -Generate runs `python -m windpower demo` first
  (SYNTHETIC, clearly labelled data) when the demo artifact is missing.
.PARAMETER Mode
  demo | full | smoke. Default: the first existing artifact of full, demo, smoke.
.PARAMETER Port
  TCP port (default 8000). Binds to 127.0.0.1 unless -BindHost is given.
.PARAMETER SiteDir
  Serve this site/v1 directory instead of data\outputs\<mode>\site\v1.
#>
[CmdletBinding()]
param(
    [ValidateSet("", "demo", "full", "smoke")][string]$Mode = "",
    [int]$Port = 8000,
    [string]$BindHost = "127.0.0.1",
    [string]$SiteDir = "",
    [string]$DataDir = "",
    [switch]$Generate
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
if (-not $py) { Write-Host "No Python found: activate an environment, create .venv, or install Python 3.11+." -ForegroundColor Red; exit 2 }

Set-Location $repo
& $py -c "import windpower" 2>$null
if ($LASTEXITCODE -ne 0) { $env:PYTHONPATH = (Join-Path $repo "src") + $(if ($env:PYTHONPATH) { ";" + $env:PYTHONPATH } else { "" }) }

$serveArgs = @("-m", "windpower", "serve", "--port", "$Port", "--host", $BindHost)
if ($Mode) { $serveArgs += @("--mode", $Mode) }
if ($SiteDir) { $serveArgs += @("--site-dir", $SiteDir) }
if ($DataDir) { $serveArgs += @("--data-dir", $DataDir) }

if ($Generate) {
    $demoArgs = @("-m", "windpower", "demo", "--quiet")
    if ($DataDir) { $demoArgs += @("--data-dir", $DataDir) }
    $dataRoot = if ($DataDir) { $DataDir } else { Join-Path $repo "data" }
    if (-not (Test-Path (Join-Path $dataRoot "outputs\demo\site\v1\manifest.json")) -and -not $SiteDir) {
        Write-Host "Generating the SYNTHETIC demo artifact (python -m windpower demo) ..."
        & $py @demoArgs
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

& $py @serveArgs
exit $LASTEXITCODE
