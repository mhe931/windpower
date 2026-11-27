"""Solar pipeline stages, CLI exit codes, partial real-mode runs and the site artifact (all offline)."""

from __future__ import annotations

import contextlib
import io
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from windpower import cli
from windpower import config as C
from windpower import fingrid as FG
from windpower import fingrid_solar as FS
from windpower import fixtures as F
from windpower import fixtures_solar as FP
from windpower import solar_public as SP
from windpower import solar_validate as SV
from windpower.export_site_solar import CF_SCALE, SOLAR_SITE_DIR
from conftest import SOLAR_END, SOLAR_START

REAL_WINDOW = ["--window-start", "2025-07-30", "--window-end", "2025-07-31"]


def run(argv, capsys=None):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


# --- synthetic end to end ------------------------------------------------------------------------------------------------------------
def test_solar_smoke_passes_all_hard_checks_offline(solar_smoke_run):
    code, out, data = solar_smoke_run
    assert code == 0 and "SOLAR-SMOKE PASS" in out and "CHECKS: 11 pass, 0 fail, 0 skipped (SYNTHETIC data)" in out and "SYNTHETIC" in out.splitlines()[0]
    o = data / "outputs" / "solar-smoke"
    for name in ("validation.csv", "validation.json", "reconciliation.csv", "reconciliation.json", "metrics.json", "manifest.json", "national_hourly.csv", "daily.csv",
                 "cf_cubes.nc", "cf_cell_summary.csv", "allocation_cells.csv", "allocation_plants.csv", "allocation_report.json", "era5_solar_hourly.nc",
                 "irradiance_comparison.json", "sarah_hourly_era5grid.nc", "sarah_coverage_cells.csv"):
        assert (o / name).is_file(), name
    assert not (data / "outputs" / "smoke").exists() and not (data / "outputs" / "solar").exists()             # nothing leaks into wind / real folders


def test_manifest_has_provenance_units_licences_and_synthetic_flags(solar_smoke_run):
    _, _, data = solar_smoke_run
    m = json.loads((data / "outputs" / "solar-smoke" / "manifest.json").read_text())
    assert m["kind"] == "solar" and m["run_mode"] == "solar-smoke" and m["synthetic"] is True and m["demo"] is False and m["status"] == "complete"
    assert m["label"] == "SYNTHETIC - not real data" and m["model_version"].startswith("windpower-solar-") and m["unavailable"] == []
    assert m["window"] == {"first_hour_utc": "2025-07-30T00:00:00Z", "last_hour_utc": "2025-07-31T23:00:00Z", "hours": 48, "time_label": "start of the hour, UTC"}
    assert m["grid"]["n_lat"] == 7 and m["pv"]["panel"]["inverter_efficiency"] == 0.9 and m["era5_solar"]["convention"]["convention_used"] == "end"
    assert m["validation"]["summary"]["failed"] == 0 and m["validation"]["summary"]["passed"] == 11 and len(m["reconciliation"]) == len(SV.TARGETS)
    assert {r["status"] for r in m["reconciliation"]} == {"not_reproducible_synthetic"}
    assert {"cm_saf_sarah3", "era5", "fingrid", "statistics_finland", "suomen_uusiutuvat"} <= set(m["licences"]) and "SYNTHETIC" in m["licences"]["_note"]
    assert set(m["units"]) >= {"era5_only_mw", "surplus_362_kwh", "capacity_mwp", "energy_mwh", "cf_era5"}
    paths = {f["path"] for f in m["files"]}
    assert "national_hourly.csv" in paths and any(p.startswith("site/solar/v1/") for p in paths) and all(len(f["sha256"]) == 64 for f in m["files"])
    assert m["fingrid_solar"]["verification"]["overall"] == "verified" and [s["id"] for s in m["fingrid_solar"]["series"]] == [267, 248, 362]
    assert m["public"]["plants"]["report"]["operational_plants"] == 30 and m["allocation"]["large_total_mw"] == pytest.approx(250.66)
    cols = {f["path"]: f["columns"] for f in m["files"] if f["path"] == "national_hourly.csv"}["national_hourly.csv"]
    assert cols["era5_only_mw"]["unit"] == "MW" and cols["surplus_362_kwh"]["unit"] == "kWh"
    assert "FINGRID_API_KEY" not in json.dumps(m) and F.FAKE_API_KEY not in json.dumps(m)


def test_national_outputs_energy_and_unit_consistency(solar_smoke_run):
    _, _, data = solar_smoke_run
    o = data / "outputs" / "solar-smoke"
    n = pd.read_csv(o / "national_hourly.csv", index_col="time_utc", parse_dates=True)
    assert len(n) == 48 and n.index.is_unique and n.index[0] == pd.Timestamp("2025-07-30T00:00", tz="UTC")
    assert {"era5_only_mw", "hybrid_mw", "capacity_267_mw", "forecast_248_mw", "surplus_362_mw", "surplus_362_kwh", "surplus_362_count"} <= set(n.columns)
    assert (n["era5_only_mw"] >= 0).all() and (n["era5_only_mw"] <= n["capacity_267_mw"]).all() and (n["forecast_248_mw"] < 0).any()    # negative night values kept
    np.testing.assert_allclose(n["surplus_362_mw"], n["surplus_362_kwh"] / 1000.0, atol=1e-6)
    d = pd.read_csv(o / "daily.csv", index_col="date_utc")
    assert d["hours"].tolist() == [24, 24] and d["era5_only_gwh"].sum() == pytest.approx(n["era5_only_mw"].sum() / 1000.0)
    met = json.loads((o / "metrics.json").read_text())
    assert met["models"]["era5_only_mw"]["energy"]["gwh"] == pytest.approx(d["era5_only_gwh"].sum()) and met["models"]["era5_only_mw"]["energy"]["hours"] == 48
    v = met["models"]["era5_only_mw"]["vs_fingrid_248"]
    assert -1 <= v["pearson_r"] <= 1 and "regression R2" in v["label"] and met["references"]["fingrid_362_av06"]["cycle_centre"]["hhmm_utc"]
    rec = json.loads((o / "allocation_report.json").read_text())
    assert rec["weights_sum"] == pytest.approx(1.0, abs=1e-12) and rec["inhabitants_outside_grid"] == 450


def test_site_artifact_schema_encoding_and_labels(solar_smoke_run):
    _, _, data = solar_smoke_run
    o = data / "outputs" / "solar-smoke"
    site = o / SOLAR_SITE_DIR
    assert site == o / "site" / "solar" / "v1"
    assert sorted(p.name for p in site.iterdir()) == ["cf_era5.bin", "cf_hybrid.bin", "coverage.json", "grid.json", "manifest.json", "meta.json", "national_hourly.json", "source_hybrid.bin"]
    man = json.loads((site / "manifest.json").read_text())
    assert man["run_mode"] == "solar-smoke" and man["synthetic"] and man["label"] == "SYNTHETIC - not real data" and man["artifact_schema_version"] == "1.0"
    assert man["modes"]["era5"]["available"] and man["modes"]["hybrid"]["available"] and man["window"]["hours"] == 48
    assert man["panel"]["panel"] == "CSi" and man["panel"]["slope_deg"] == 40.0 and man["panel"]["azimuth_deg"] == 180.0
    assert man["grid"]["n_lat"] == 7 and man["capacity_factor_encoding"] == {"dtype": "uint16", "scale": CF_SCALE}
    actual = sum(p.stat().st_size for p in site.iterdir())
    assert man["total_bytes"] < 15_000_000 and abs(man["total_bytes"] - actual) < 4096 and man["total_bytes"] >= sum(f["bytes"] for f in man["files"])
    meta = json.loads((site / "meta.json").read_text())
    assert meta["files"]["cf_era5.bin"]["shape"] == [48, 7, 9] and meta["files"]["cf_era5.bin"]["scale"] == 10000 and meta["files"]["source_hybrid.bin"]["codes"]["0"] == "satellite"
    n_cells = 48 * 7 * 9
    assert (site / "cf_era5.bin").stat().st_size == n_cells * 2 and (site / "cf_hybrid.bin").stat().st_size == n_cells * 2 and (site / "source_hybrid.bin").stat().st_size == n_cells
    cf = np.fromfile(site / "cf_era5.bin", dtype="<u2").reshape(48, 7, 9) / CF_SCALE
    with xr.open_dataset(o / "cf_cubes.nc") as ds:
        np.testing.assert_allclose(cf, ds["cf_era5"].values, atol=0.5 / CF_SCALE + 1e-6)             # rounding error <= 5e-5
        hyb = np.fromfile(site / "cf_hybrid.bin", dtype="<u2").reshape(48, 7, 9) / CF_SCALE
        np.testing.assert_allclose(hyb, ds["cf_hybrid"].values, atol=0.5 / CF_SCALE + 1e-6)
        src = np.fromfile(site / "source_hybrid.bin", dtype="u1").reshape(48, 7, 9)
        np.testing.assert_array_equal(src, ds["source_code"].values)
    with xr.open_dataset(o / "era5_solar_hourly.nc") as e:
        night = e["solar_altitude"].values <= 0
    assert (cf[night] == 0).all() and cf.max() <= 1.0 and cf.max() > 0.3                                    # night stays exactly 0 after encoding
    grid = json.loads((site / "grid.json").read_text())
    assert len(grid["grid"]["lat"]) == 7 and len(grid["cells"]["satellite_covered"]) == 63 and sum(grid["cells"]["satellite_covered"]) == 17
    assert grid["cells"]["population_share"] and sum(grid["cells"]["population_share"]) == pytest.approx(1.0, abs=1e-6) and grid["label"] == man["label"]
    cov = json.loads((site / "coverage.json").read_text())
    assert cov["available"] and cov["summary"]["fully_covered_cells"] == 17 and len(cov["pixel_map"]["values"]) == 32 * 44 and cov["synthetic"]


def test_site_national_series_match_the_csv_and_carry_scope_and_units(solar_smoke_run):
    _, _, data = solar_smoke_run
    o = data / "outputs" / "solar-smoke"
    nat = json.loads((o / SOLAR_SITE_DIR / "national_hourly.json").read_text())
    csv = pd.read_csv(o / "national_hourly.csv")
    assert nat["hours"][0] == "2025-07-30T00:00:00Z" and len(nat["hours"]) == 48 and nat["synthetic"] is True
    s = nat["series"]
    assert set(s) == {"era5_only_mw", "hybrid_mw", "era5_only_popw_mw", "hybrid_popw_mw", "fingrid_248_forecast_mw", "fingrid_362_av06_small_scale_surplus_mw", "fingrid_267_capacity_mw"}
    assert all(v["available"] and v["reason"] is None and v["unit"] == "MW" for v in s.values())
    np.testing.assert_allclose(s["era5_only_mw"]["values"], csv["era5_only_mw"], atol=1e-4)
    np.testing.assert_allclose(s["fingrid_362_av06_small_scale_surplus_mw"]["values_native_kwh"], csv["surplus_362_kwh"], atol=0.1)
    assert s["fingrid_248_forecast_mw"]["native_unit"] == "MWh/h" and "forecast, not measured" in s["fingrid_248_forecast_mw"]["scope"]
    assert "surplus only" in s["fingrid_362_av06_small_scale_surplus_mw"]["scope"] and "not official" in s["fingrid_267_capacity_mw"]["scope"]
    assert s["fingrid_362_av06_small_scale_surplus_mw"]["native_unit"] == "kWh"


def test_wind_site_artifact_is_never_touched_and_stages_can_run_one_by_one(tmp_path, capsys):
    data = tmp_path / "d"
    wind_site = data / "outputs" / "solar-smoke" / "site" / "v1"
    wind_site.mkdir(parents=True)
    sentinel = wind_site / "manifest.json"
    sentinel.write_text('{"wind": "untouched"}', encoding="utf-8")
    for stage in ("ingest-era5-solar", "ingest-solar-public", "ingest-fingrid-solar", "solar-hourly", "solar-pv", "solar-national", "export-site-solar"):
        code, out, err = run([stage, "--mode", "solar-smoke", "--data-dir", str(data), "--quiet"])
        assert code == 0, (stage, out, err)
        assert "SYNTHETIC" in out.splitlines()[0]
    code, out, err = run(["solar-validate", "--mode", "solar-smoke", "--data-dir", str(data), "--quiet"])
    assert code == 0 and "SOLAR-VALIDATE[solar-smoke] PASS" in out
    assert sentinel.read_text(encoding="utf-8") == '{"wind": "untouched"}' and sorted(p.name for p in wind_site.iterdir()) == ["manifest.json"]
    assert (data / "outputs" / "solar-smoke" / "site" / "solar" / "v1" / "manifest.json").is_file()
    man = json.loads((data / "outputs" / "solar-smoke" / "manifest.json").read_text())
    assert any(f["path"].startswith("site/solar/v1/") for f in man["files"]) and man["validation"]["summary"]["failed"] == 0                 # stage-by-stage manifest is complete


def test_export_refuses_artifacts_above_15_mb(solar_smoke_run, monkeypatch):
    from windpower import export_site_solar as ESS
    from windpower.export_site import ExportSizeError
    from windpower.solar_pipeline import SolarRun

    _, _, data = solar_smoke_run
    assert ESS.MAX_BYTES == 15_000_000
    run_ = SolarRun("solar-smoke", data, say=lambda m: None)
    monkeypatch.setattr(ESS, "MAX_BYTES", 5_000)
    with pytest.raises(ExportSizeError, match="MB"):
        run_.export_site_solar()


def test_failed_hard_check_exits_1(tmp_path, monkeypatch):
    from windpower.validate import CheckResult

    monkeypatch.setattr(SV, "check_units", lambda ev: CheckResult("unit_consistency", FAIL_STATUS, "forced failure"))
    code, out, err = run(["solar-smoke", "--data-dir", str(tmp_path), "--quiet"])
    assert code == cli.EXIT_VALIDATION == 1 and "FAIL unit_consistency: forced failure" in out and "SOLAR-SMOKE FAIL" in out


FAIL_STATUS = "fail"


def test_solar_demo_runs_on_the_full_grid_and_passes():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        code, out, err = run(["solar-demo", "--data-dir", d, "--quiet"])
        assert code == 0, err + out
        assert "SOLAR-DEMO PASS" in out and "DEMO" not in err
        site = json.loads((__import__("pathlib").Path(d) / "outputs" / "solar-demo" / SOLAR_SITE_DIR / "manifest.json").read_text())
        assert site["label"] == "DEMO / SYNTHETIC - not real data" and site["demo"] and site["grid"]["n_lat"] == 44 and site["grid"]["n_lon"] == 53
        assert site["total_bytes"] < 15_000_000 and site["modes"]["hybrid"]["available"]


# --- real mode with synthetic stand-ins for the inputs (full 44 x 53 grid) --------------------------------------------------------------------
@pytest.fixture(scope="session")
def real_inputs(tmp_path_factory, sarah_smoke):
    """Inputs for REAL-mode runs, produced by the fake transports: ERA5 file, public cache, Fingrid cache (grid = FINLAND_SOLAR)."""
    base = tmp_path_factory.mktemp("real_inputs")
    era5 = F.write_netcdf(F.synthetic_era5_solar(C.FINLAND_SOLAR, SOLAR_START, 48, seed=9).raw, base / "era5" / "era5_solar_2025_07.nc")
    pub = base / "public"
    SP.fetch_public(pub, FP.FakePublicSession(FP.synthetic_public(C.FINLAND_SOLAR, seed=5)), sleep=lambda s: None)
    fgd = base / "fingrid"
    fx = FP.synthetic_solar_fingrid(pd.date_range(SOLAR_START, SOLAR_END, freq="h"), seed=4)
    client = FG.FingridClient(F.FAKE_API_KEY, F.FakeFingridSession(fx), request_delay=0, sleep=lambda s: None, now=lambda: F.FIXED_ISO)
    FS.load_or_fetch_solar(fgd, SOLAR_START, SOLAR_END, client=client)
    return {"era5": era5, "public": pub, "fingrid": fgd, "sis": sarah_smoke.sis_dir, "sid": sarah_smoke.sid_dir}


def real_args(tmp_path, ri, *, sarah=True, era5=True, public=True, fingrid=True):
    data = tmp_path / "data"
    cache = data / "cache" / "solar"
    if public:
        shutil.copytree(ri["public"], cache / "public")
    if fingrid:
        shutil.copytree(ri["fingrid"], cache / "fingrid")
    argv = ["--data-dir", str(data), *REAL_WINDOW]
    if sarah:
        argv += ["--sarah-sis-dir", str(ri["sis"]), "--sarah-sid-dir", str(ri["sid"])]
    if era5:
        argv += ["--era5-solar-path", str(ri["era5"])]
    return data, argv


def test_real_mode_with_every_input_completes_and_compares_with_the_report(tmp_path, real_inputs):
    data, argv = real_args(tmp_path, real_inputs)
    code, out, err = run(["solar-all", "--quiet", *argv])
    assert code == 0, out + err
    assert "STATUS" not in out and "SOLAR-ALL[solar] PASS" in out and "SYNTHETIC" not in out
    o = data / "outputs" / "solar"
    m = json.loads((o / "manifest.json").read_text())
    assert m["status"] == "complete" and m["synthetic"] is False and m["label"] == "real-data run" and m["unavailable"] == []
    status = {r["metric"]: r["status"] for r in m["reconciliation"]}
    assert set(status.values()) <= {"match", "differs"} and status["large_plants_mw"] in ("match", "differs") and status["hourly_values"] == "differs"   # 48 h, not 744
    rec = {r["metric"]: r for r in m["reconciliation"]}
    assert rec["hourly_values"]["observed"] == 48 and rec["grid_cells"]["status"] == "match" and rec["large_plants_mw"]["status"] == "match"
    site = json.loads((o / SOLAR_SITE_DIR / "manifest.json").read_text())
    assert site["label"].startswith("Retrospective") and not site["synthetic"] and all(v["available"] for v in site["modes"].values())
    assert all(v["available"] for v in site["national_series"].values())


def test_real_mode_sarah_and_public_only_is_partial_with_unavailable_ERA5_and_Fingrid(tmp_path, real_inputs):
    data, argv = real_args(tmp_path, real_inputs, era5=False, fingrid=False)
    code, out, err = run(["solar-all", *argv])
    assert code == 0, out + err
    assert "STATUS partial_inputs_unavailable" in out and "SOLAR-ALL[solar] PASS" in out and "skipped capacity_reconciliation" in out
    o = data / "outputs" / "solar"
    m = json.loads((o / "manifest.json").read_text())
    assert m["status"] == "partial_inputs_unavailable" and {u["input"] for u in m["unavailable"]} >= {"era5", "fingrid_267", "fingrid_248", "fingrid_362", "national"}
    assert any("FINGRID_API_KEY" in u["reason"] for u in m["unavailable"]) and any("ERA5" in u["reason"] for u in m["unavailable"])
    by = {r["metric"]: r for r in m["reconciliation"]}
    assert by["fully_covered_cells"]["status"] in ("match", "differs") and by["large_plants"]["status"] == "match"       # verifiable from SARAH + public data
    assert by["energy_era5_gwh"]["status"] == "unavailable" and "ERA5 solar data" in by["energy_era5_gwh"]["note"]
    assert by["energy_fingrid_248_gwh"]["status"] == "unavailable" and "FINGRID_API_KEY" in by["energy_fingrid_248_gwh"]["note"]
    site = json.loads((o / SOLAR_SITE_DIR / "manifest.json").read_text())
    assert not site["modes"]["era5"]["available"] and not site["modes"]["hybrid"]["available"] and "ERA5" in site["modes"]["era5"]["reason"]
    assert sorted(p.name for p in (o / SOLAR_SITE_DIR).iterdir()) == ["coverage.json", "grid.json", "manifest.json", "meta.json", "national_hourly.json"]   # no .bin
    nat = json.loads((o / SOLAR_SITE_DIR / "national_hourly.json").read_text())
    for key, s in nat["series"].items():
        assert s["available"] is False and "values" not in s and s["reason"]
        assert "FINGRID_API_KEY" in s["reason"] if key.startswith("fingrid") else ("ERA5" in s["reason"] or "FINGRID_API_KEY" in s["reason"])
    assert "ERA5" in nat["series"]["era5_only_mw"]["reason"]
    grid = json.loads((o / SOLAR_SITE_DIR / "grid.json").read_text())
    assert len(grid["cells"]["satellite_covered"]) == 2332 and "population_share" in grid["cells"]
    assert not (o / "national_hourly.csv").exists() and not (o / "cf_cubes.nc").exists()                 # nothing is fabricated for the missing inputs
    v = json.loads((o / "validation.json").read_text())
    assert v["summary"]["failed"] == 0 and v["summary"]["skipped"] >= 4


def test_real_mode_without_sarah_gives_era5_only_and_marks_hybrid_unavailable(tmp_path, real_inputs):
    data, argv = real_args(tmp_path, real_inputs, sarah=False)
    code, out, err = run(["solar-all", "--quiet", *argv])
    assert code == 0, out + err
    o = data / "outputs" / "solar"
    site = json.loads((o / SOLAR_SITE_DIR / "manifest.json").read_text())
    assert site["modes"]["era5"]["available"] and not site["modes"]["hybrid"]["available"]
    ser = json.loads((o / SOLAR_SITE_DIR / "national_hourly.json").read_text())["series"]
    assert ser["era5_only_mw"]["available"] and ser["fingrid_267_capacity_mw"]["available"]
    assert not ser["hybrid_mw"]["available"] and "SARAH-3" in ser["hybrid_mw"]["reason"]
    n = pd.read_csv(o / "national_hourly.csv")
    assert "hybrid_mw" not in n.columns and "era5_only_mw" in n.columns


def test_real_mode_with_nothing_available_exits_2(tmp_path):
    code, out, err = run(["solar-all", "--data-dir", str(tmp_path), "--quiet"])
    assert code == cli.EXIT_UNAVAILABLE and "INPUTS UNAVAILABLE" in err and "no synthetic data was substituted" in err
    assert not (tmp_path / "outputs" / "solar" / "site").exists()


def test_stage_exit_codes_in_real_mode(tmp_path, real_inputs):
    d1 = tmp_path / "a"
    code, out, err = run(["ingest-era5-solar", "--data-dir", str(d1)])
    assert code == 2 and "ERA5" in err and ("cdsapirc" in err or "CDS" in err)
    code, out, err = run(["ingest-fingrid-solar", "--data-dir", str(d1)])
    assert code == 2 and "FINGRID_API_KEY" in err and "Fingrid 267" in err
    code, out, err = run(["solar-pv", "--data-dir", str(d1), *REAL_WINDOW, "--sarah-sis-dir", str(real_inputs["sis"]), "--sarah-sid-dir", str(real_inputs["sid"])])
    assert code == 2 and "ERA5" in err                                           # the stage's own required input is missing
    code, out, err = run(["solar-national", "--data-dir", str(d1)])
    assert code == 2 and "public" in err                                         # needs the public cache
    code, out, err = run(["ingest-solar-public", "--data-dir", str(d1), "--offline"])
    assert code == 2 and "public inputs" in err and "ingest-solar-public" in err
    data, argv = real_args(tmp_path / "b", real_inputs, sarah=False, era5=False, fingrid=False)
    code, out, err = run(["solar-national", *argv])                               # public data present: allocation is done, national series are partial
    assert code == 0 and "OK (partial)" in out and "national" in out and (data / "outputs" / "solar" / "allocation_report.json").is_file()
    code, out, err = run(["solar-all", "--offline", "--data-dir", str(tmp_path / "c")])
    assert code == 2


def test_era5_convention_ambiguity_is_a_contract_failure_and_override_works(tmp_path, real_inputs):
    flat = F.synthetic_era5_solar(C.FINLAND_SOLAR, SOLAR_START, 48, seed=9).prepared.copy(deep=True)
    flat["influx_toa"] = flat["influx_toa"] * 0 + 300.0
    path = F.write_netcdf(flat, tmp_path / "flat" / "prepared.nc")
    data, argv = real_args(tmp_path, real_inputs, era5=False)
    argv += ["--era5-solar-path", str(path)]
    code, out, err = run(["ingest-era5-solar", *argv])
    assert code == cli.EXIT_VALIDATION and "AMBIGUOUS CONVENTION" in err and "--era5-convention" in err
    code, out, err = run(["ingest-era5-solar", *argv, "--era5-convention", "start"])
    assert code == 0 and "convention start (detected ambiguous)" in out


def test_a_foreign_grid_is_rejected(tmp_path, real_inputs):
    bad = F.write_netcdf(F.synthetic_era5_solar(C.SOLAR_SMOKE_GRID, SOLAR_START, 48, seed=9).raw, tmp_path / "small" / "era5.nc")
    data, argv = real_args(tmp_path, real_inputs, era5=False)
    code, out, err = run(["ingest-era5-solar", *argv, "--era5-solar-path", str(bad)])
    assert code == cli.EXIT_VALIDATION and "does not match finland_solar" in err
