from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from windpower import cli, config as C, era5, fingrid, validate
from windpower.pipeline import MODE_SPECS, Run

REPO = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")


def clean_env(tmp_path, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FINGRID", "CDSAPI"))}
    env["USERPROFILE"] = env["HOME"] = str(tmp_path)
    env.update(extra)
    return env


@pytest.fixture()
def no_credentials(monkeypatch, tmp_path):
    """Neither the env nor ~/.cdsapirc nor a repo .env may provide credentials in these tests."""
    monkeypatch.setattr(era5, "cds_credentials_available", lambda env=None, home=None: False)
    monkeypatch.setattr(fingrid, "REPO_ROOT", tmp_path)


# --- smoke -----------------------------------------------------------------------------------------
def test_smoke_stage_passes_end_to_end(smoke_run):
    code, out, data = smoke_run
    assert code == 0
    assert "SYNTHETIC" in out and "SMOKE PASS" in out and "CHECKS: 11 pass, 0 fail, 1 skipped" in out
    assert "forward_check: coverage 21.0 days < 60 days required" in out
    o = data / "outputs" / "smoke"
    v = json.loads((o / "validation.json").read_text())
    assert v["summary"] == {"passed": 11, "failed": 0, "skipped": 1} and v["synthetic"] is True
    assert (o / "manifest.json").is_file() and (o / "site" / "v1" / "manifest.json").is_file()
    assert (data / "cache" / "synthetic_smoke" / "era5").is_dir() and not (data / "cache" / "real").exists()   # never mixes with real data


def test_smoke_generated_data_stays_under_the_data_dir(smoke_run):
    _, _, data = smoke_run
    assert REPO not in data.parents                                                        # tests write to tmp only
    # and the default location is git-ignored
    if shutil.which("git") is None:
        pytest.skip("git not available")
    for rel in ("data/outputs/smoke/manifest.json", "data/outputs/demo/site/v1/wind_effective.bin",
                "data/cache/synthetic_smoke/era5/era5_2025_07.nc", "data/cache/real/fingrid/fingrid_raw_181_wind_power_mw.csv",
                "data/interim/smoke/run_info.json"):
        r = subprocess.run(["git", "check-ignore", "-q", rel], cwd=REPO)
        assert r.returncode == 0, rel


def test_stages_are_independent_in_synthetic_mode(tmp_path):
    assert cli.main(["fleet", "--mode", "smoke", "--data-dir", str(tmp_path), "--quiet"]) == 0
    out = tmp_path / "outputs" / "smoke"
    assert (out / "fleet_turbines.csv").is_file() and (out / "capacity_matrix.csv").is_file() and not (out / "manifest.json").exists()
    assert cli.main(["simulate", "--mode", "smoke", "--data-dir", str(tmp_path), "--quiet"]) == 0
    assert (out / "archetypes.csv").is_file() and (out / "power_curves.json").is_file()


def test_validation_failure_gives_exit_code_1(smoke_run, monkeypatch, capsys, tmp_path):
    _, _, shared = smoke_run
    data = tmp_path / "copy"
    shutil.copytree(shared, data)                                                         # never touch the shared smoke outputs
    failing = validate.CheckResult("injected", validate.FAIL, "deliberate failure")
    monkeypatch.setattr(validate, "run_all_checks", lambda ev: [validate.CheckResult("ok", validate.PASS, "fine"), failing])
    code = cli.main(["validate", "--mode", "smoke", "--data-dir", str(data), "--quiet"])
    out = capsys.readouterr().out
    assert code == cli.EXIT_VALIDATION == 1 and "FAIL injected: deliberate failure" in out and "VALIDATE[smoke] FAIL" in out
    v = json.loads((data / "outputs" / "smoke" / "validation.json").read_text())
    assert v["summary"]["failed"] == 1                                                      # failure is recorded, not hidden


# --- full mode without credentials --------------------------------------------------------------------
def test_full_mode_without_credentials_exits_2_and_lists_inputs(tmp_path, no_credentials, capsys):
    code = cli.main(["all", "--mode", "full", "--data-dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert code == cli.EXIT_UNAVAILABLE == 2 and code != cli.EXIT_VALIDATION
    assert "INPUTS UNAVAILABLE (full mode)" in err and "no synthetic data was substituted" in err
    for needle in ("ERA5 (Copernicus CDS)", "CDSAPI_URL", "fleet inventories", "ingest-fleet --refresh", "Fingrid API", "FINGRID_API_KEY"):
        assert needle in err
    man = json.loads((tmp_path / "outputs" / "full" / "manifest.json").read_text())
    assert man["status"] == "incomplete_inputs_unavailable" and {u["input"] for u in man["unavailable"]} >= {"ERA5 (Copernicus CDS)", "Fingrid API"}
    assert not (tmp_path / "cache").exists() or not any((tmp_path / "cache").rglob("*.nc"))   # nothing synthetic was generated


@pytest.mark.parametrize("stage", ["ingest-era5", "ingest-fingrid", "ingest-fleet", "fleet", "simulate", "calibrate", "validate", "export-site"])
def test_every_full_stage_reports_unavailable_instead_of_synthetic_fallback(stage, tmp_path, no_credentials, capsys):
    code = cli.main([stage, "--mode", "full", "--data-dir", str(tmp_path), "--quiet"])
    assert code == cli.EXIT_UNAVAILABLE
    assert "INPUTS UNAVAILABLE (full mode)" in capsys.readouterr().err
    assert not any((tmp_path / "cache").rglob("era5_*.nc")) if (tmp_path / "cache").exists() else True


def test_full_mode_offline_flag_is_respected(tmp_path, no_credentials, monkeypatch, capsys):
    monkeypatch.setattr(era5, "cds_credentials_available", lambda env=None, home=None: True)        # even with credentials...
    code = cli.main(["all", "--mode", "full", "--offline", "--data-dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert code == 2 and "--offline" in err and "cache incomplete" in err
    code = cli.main(["ingest-fleet", "--mode", "full", "--offline", "--refresh", "--data-dir", str(tmp_path)])
    assert code == 2 and "--refresh needs the network but --offline" in capsys.readouterr().err


def test_preflight_lists_only_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(era5, "cds_credentials_available", lambda env=None, home=None: True)
    monkeypatch.setattr(fingrid, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("FINGRID_API_KEY", "placeholder")
    run = Run("full", tmp_path, say=lambda m: None)
    items = {i["input"] for i in run.preflight()}
    assert items == {"fleet inventories (OSM + ArcGIS)"}                                      # credentials present; fleet cache missing
    assert Run("smoke", tmp_path).preflight() == [] and Run("demo", tmp_path).preflight() == []


def test_usage_error_exit_code_is_distinct():
    with pytest.raises(SystemExit) as e:
        cli.main(["not-a-stage"])
    assert e.value.code == cli.EXIT_USAGE == 64
    with pytest.raises(SystemExit) as e:
        cli.main(["smoke", "--mode", "nope"])
    assert e.value.code == 64


def test_modes_and_paths(tmp_path):
    assert C.RUN_MODES == ("smoke", "full", "demo") and C.SYNTHETIC_MODES == ("smoke", "demo")
    assert C.cache_dir("demo", tmp_path) == tmp_path / "cache" / "synthetic_demo" and C.cache_dir("full", tmp_path) == tmp_path / "cache" / "real"
    assert C.output_dir("smoke", tmp_path) == tmp_path / "outputs" / "smoke" and C.interim_dir("full", tmp_path) == tmp_path / "interim" / "full"
    with pytest.raises(ValueError):
        C.output_dir("prod", tmp_path)
    assert MODE_SPECS["demo"].grid is C.FINLAND_GRID and (MODE_SPECS["demo"].grid.n_lat, MODE_SPECS["demo"].grid.n_lon) == (45, 52)
    assert MODE_SPECS["smoke"].hours >= 24 * 10
    assert Run("smoke", tmp_path).offline is True and Run("demo", tmp_path).offline is True


def test_module_entry_point_via_subprocess(tmp_path):
    r = subprocess.run([sys.executable, "-m", "windpower", "all", "--mode", "full", "--data-dir", str(tmp_path / "d")], cwd=REPO,
                       env=clean_env(tmp_path), capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "INPUTS UNAVAILABLE" in r.stderr
    r = subprocess.run([sys.executable, "-m", "windpower", "--help"], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "smoke" in r.stdout and "--offline" in r.stdout
    r = subprocess.run([sys.executable, "-m", "windpower", "nonsense"], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert r.returncode == 64


# --- PowerShell scripts ----------------------------------------------------------------------------------
needs_ps = pytest.mark.skipif(POWERSHELL is None, reason="PowerShell not available")


def ps(args, env, timeout=300):
    # cwd is a temp dir: with USERPROFILE redirected, PowerShell drops a module-analysis cache into its cwd
    return subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", *args], cwd=env.get("USERPROFILE") or tempfile.gettempdir(),
                          env=env, capture_output=True, text=True, timeout=timeout)


@needs_ps
@pytest.mark.parametrize("script", ["run_smoke.ps1", "run_full.ps1"])
def test_scripts_parse(script):
    cmd = (f"$e=$null;$t=$null;[void][System.Management.Automation.Language.Parser]::ParseFile('{REPO / 'scripts' / script}',[ref]$t,[ref]$e);"
           "if($e.Count){$e|%{$_.Message};exit 1}")
    r = ps(["-Command", cmd], os.environ.copy())
    assert r.returncode == 0, r.stdout + r.stderr


@needs_ps
def test_run_full_check_only_reports_missing_credentials(tmp_path):
    r = ps(["-File", str(REPO / "scripts" / "run_full.ps1"), "-CheckOnly", "-UseActivePython", "-EnvFile", str(tmp_path / "none.env")], clean_env(tmp_path))
    out = r.stdout + r.stderr
    assert r.returncode == 2, out
    assert "FINGRID_API_KEY" in out and "CDS credentials" in out and "INPUTS UNAVAILABLE" in out and "no synthetic data is substituted" in out
    assert "conda" not in out.split("INPUTS UNAVAILABLE")[1]                                 # -UseActivePython skips the conda requirement


@needs_ps
def test_run_full_check_only_passes_when_credentials_exist_and_never_prints_them(tmp_path):
    secret = "dummy-secret-value-123"
    env = clean_env(tmp_path, FINGRID_API_KEY=secret, CDSAPI_URL="https://example.invalid/api", CDSAPI_KEY="dummy-cds-key-456")
    r = ps(["-File", str(REPO / "scripts" / "run_full.ps1"), "-DryRun", "-UseActivePython", "-EnvFile", str(tmp_path / "none.env")], env)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "All inputs available." in out and "would run: python -m windpower all --mode full" in out
    assert secret not in out and "dummy-cds-key-456" not in out


@needs_ps
def test_run_full_reads_dotenv_key_without_printing_it(tmp_path):
    envfile = tmp_path / ".env"
    envfile.write_text("FINGRID_API_KEY=file-secret-789\n")
    (tmp_path / ".cdsapirc").write_text("url: https://example.invalid\nkey: abc\n")
    r = ps(["-File", str(REPO / "scripts" / "run_full.ps1"), "-CheckOnly", "-UseActivePython", "-EnvFile", str(envfile)], clean_env(tmp_path))
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "file-secret-789" not in out and "[ok] FINGRID_API_KEY" in out


@needs_ps
def test_run_smoke_script_exits_0_and_prints_pass(tmp_path):
    scripts_dir = str(Path(sys.executable).parent)
    env = clean_env(tmp_path, PATH=scripts_dir + os.pathsep + os.environ.get("PATH", ""), VIRTUAL_ENV=str(Path(sys.executable).parents[1]))
    r = ps(["-File", str(REPO / "scripts" / "run_smoke.ps1"), "-DataDir", str(tmp_path / "data")], env, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "SMOKE PASS" in out and "run_smoke.ps1: PASS" in out and "SYNTHETIC" in out
    assert (tmp_path / "data" / "outputs" / "smoke" / "manifest.json").is_file()
