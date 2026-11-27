"""`ingest-sarah` stage: solar run modes, input paths (flags / env / defaults), exit codes, written products."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from windpower import cli
from windpower import config as C
from windpower import fixtures as F
from windpower import sarah as S
from windpower import solar_pipeline as SP


def run(argv, capsys):
    code = cli.main(argv)
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def test_solar_modes_follow_the_existing_path_conventions(tmp_path):
    assert C.RUN_MODES == ("smoke", "full", "demo") and C.SYNTHETIC_MODES == ("smoke", "demo")       # wind contract untouched
    assert C.SOLAR_RUN_MODES == ("solar-smoke", "solar-demo", "solar") and C.SOLAR_SYNTHETIC_MODES == ("solar-smoke", "solar-demo")
    assert C.cache_dir("solar-smoke", tmp_path) == tmp_path / "cache" / "synthetic_solar-smoke"
    assert C.cache_dir("solar-demo", tmp_path) == tmp_path / "cache" / "synthetic_solar-demo"
    assert C.cache_dir("solar", tmp_path) == tmp_path / "cache" / "solar" != C.cache_dir("full", tmp_path)
    assert C.output_dir("solar", tmp_path) == tmp_path / "outputs" / "solar" and C.interim_dir("solar-demo", tmp_path) == tmp_path / "interim" / "solar-demo"
    assert C.is_synthetic_mode("solar-smoke") and C.is_synthetic_mode("demo") and not C.is_synthetic_mode("solar") and not C.is_synthetic_mode("full")
    with pytest.raises(ValueError):
        C.output_dir("solar-big", tmp_path)


def test_parse_window_dates_are_inclusive_days():
    h = SP.parse_window(None, None)
    assert len(h) == 744 and h[0] == pd.Timestamp("2025-07-01T00:00") and h[-1] == pd.Timestamp("2025-07-31T23:00")
    h = SP.parse_window("2025-07-30T05:00", "2025-07-30T07:00")
    assert list(h.hour) == [5, 6, 7]
    with pytest.raises(ValueError):
        SP.parse_window("2025-07-02", "2025-07-01")


def test_ingest_sarah_solar_smoke_writes_hourly_and_coverage_products(tmp_path, capsys):
    code, out, err = run(["ingest-sarah", "--mode", "solar-smoke", "--data-dir", str(tmp_path)], capsys)
    assert code == 0, err
    assert "SYNTHETIC" in out and "48 hourly steps" in out and "fully covered cells" in out and "report target 777 of 2332" in out
    o = tmp_path / "outputs" / "solar-smoke"
    for name in (SP.HOURLY_FILE, SP.CELLS_FILE, SP.PIXELS_FILE, SP.INVENTORY_FILE):
        assert (o / name).is_file()
    with xr.open_dataset(o / SP.HOURLY_FILE) as ds:
        assert ds.sizes == {"time": 48, "latitude": 7, "longitude": 9} and ds.attrs["synthetic"] == "True"
        assert {"ghi_wm2", "direct_wm2", "diffuse_wm2", "valid_area_fraction", "source_code"} <= set(ds.data_vars)
        assert ds["time"].values[0] == np.datetime64("2025-07-30T00:00") and ds["valid_area_fraction"].max() <= 1.0
        assert ds["ghi_wm2"].isnull().any()                          # missing stays NaN
    cells = pd.read_csv(o / SP.CELLS_FILE)
    assert len(cells) == 63 and cells["fully_covered"].sum() == 17 and cells["static_covered_fraction"].between(0, 1).all()
    inv = json.loads((o / SP.INVENTORY_FILE).read_text())
    assert inv["synthetic"] and inv["coverage"]["hours"] == 48 and inv["coverage"]["fully_covered_cells"] == 17
    assert [s["meaning"] for s in inv["inventory"]["SIS"]["window_flagged_steps"]] == ["bad_quality", "void"]
    info = json.loads((tmp_path / "interim" / "solar-smoke" / "run_info.json").read_text())
    assert info["sarah"]["source"] == "SYNTHETIC fixture" and info["sarah"]["coverage"]["hours"] == 48
    assert (tmp_path / "cache" / "synthetic_solar-smoke" / "sarah" / "SYNTHETIC_SIS").is_dir()    # fixtures live below the data dir only


def test_real_mode_reads_explicit_folders_in_place_and_uses_the_full_solar_grid(tmp_path, capsys, sarah_smoke):
    before = sorted((p.name, p.stat().st_mtime_ns) for p in sarah_smoke.sis_dir.iterdir())
    code, out, err = run(["ingest-sarah", "--data-dir", str(tmp_path), "--sarah-sis-dir", str(sarah_smoke.sis_dir),
                          "--sarah-sid-dir", str(sarah_smoke.sid_dir), "--window-start", "2025-07-30", "--window-end", "2025-07-31"], capsys)
    assert code == 0, err
    assert "INGEST-SARAH[solar] OK: 48 hours" in out and "SYNTHETIC" not in out.splitlines()[-1]
    cells = pd.read_csv(tmp_path / "outputs" / "solar" / SP.CELLS_FILE)
    assert len(cells) == 2332 and cells["fully_covered"].sum() == 17           # same synthetic order seen on the 44 x 53 grid
    info = json.loads((tmp_path / "interim" / "solar" / "run_info.json").read_text())
    assert info["sarah"]["inputs"]["SIS"] == str(sarah_smoke.sis_dir) and len(info["sarah"]["sha256"]["SIS"]) == 3        # file hashes recorded
    assert before == sorted((p.name, p.stat().st_mtime_ns) for p in sarah_smoke.sis_dir.iterdir())                         # inputs untouched
    assert not any(p.suffix == ".nc" for p in (tmp_path / "cache").rglob("*")) if (tmp_path / "cache").exists() else True   # nothing copied


def test_environment_variables_select_the_folders(tmp_path, capsys, monkeypatch, sarah_smoke):
    monkeypatch.setenv(C.SARAH_SIS_ENV, str(sarah_smoke.sis_dir))
    monkeypatch.setenv(C.SARAH_SID_ENV, str(sarah_smoke.sid_dir))
    code, out, err = run(["ingest-sarah", "--data-dir", str(tmp_path), "--window-start", "2025-07-30", "--window-end", "2025-07-30", "--quiet"], capsys)
    assert code == 0, err and "OK: 24 hours" in out


def test_missing_folders_exit_2_with_a_clear_reason_and_no_synthetic_fallback(tmp_path, capsys):
    code, out, err = run(["ingest-sarah", "--data-dir", str(tmp_path)], capsys)
    assert code == cli.EXIT_UNAVAILABLE == 2
    assert "INPUTS UNAVAILABLE" in err and "no synthetic data was substituted" in err
    assert C.SARAH_SIS_ENV in err and "--sarah-sid-dir" in err and "ORD73128" in err
    manifest = json.loads((tmp_path / "outputs" / "solar" / "manifest.json").read_text())
    assert manifest["status"] == "incomplete_inputs_unavailable" and manifest["run_mode"] == "solar" and manifest["synthetic"] is False
    assert [p.name for p in (tmp_path / "outputs" / "solar").iterdir()] == ["manifest.json"]


def test_window_outside_the_order_exits_2(tmp_path, capsys, sarah_smoke):
    code, out, err = run(["ingest-sarah", "--data-dir", str(tmp_path), "--sarah-sis-dir", str(sarah_smoke.sis_dir),
                          "--sarah-sid-dir", str(sarah_smoke.sid_dir), "--window-start", "2025-09-01", "--window-end", "2025-09-02"], capsys)
    assert code == 2 and "SARAH-3 files" in err


def test_unreadable_or_non_conforming_files_exit_1(tmp_path, capsys, sarah_smoke):
    bad = tmp_path / "bad"
    bad.mkdir()
    ds = xr.Dataset({"SIS": (("time", "lat", "lon"), np.zeros((48, 2, 2), np.float32))},
                    coords={"time": pd.date_range("2025-07-30", periods=48, freq="30min"), "lat": [60.0, 60.05], "lon": [21.0, 21.05]})
    ds.to_netcdf(bad / "SISin202507300000004UD10001I1UD.nc")              # no record_status, no bounds
    code, out, err = run(["ingest-sarah", "--data-dir", str(tmp_path), "--sarah-sis-dir", str(bad), "--sarah-sid-dir", str(sarah_smoke.sid_dir),
                          "--window-start", "2025-07-30", "--window-end", "2025-07-30"], capsys)
    assert code == cli.EXIT_VALIDATION and "INPUT CONTRACT FAILURE" in err and "record_status" in err


def test_mode_usage_errors(tmp_path, capsys):
    code, _, err = run(["ingest-sarah", "--mode", "full"], capsys)
    assert code == cli.EXIT_USAGE and "solar mode" in err
    code, _, err = run(["simulate", "--mode", "solar-smoke"], capsys)
    assert code == cli.EXIT_USAGE and "solar mode" in err
    with pytest.raises(SystemExit) as exc:
        cli.main(["ingest-sarah", "--era5-convention", "sideways"])
    assert exc.value.code == cli.EXIT_USAGE
    capsys.readouterr()
    code, _, err = run(["ingest-sarah", "--mode", "solar", "--window-start", "2025-07-02", "--window-end", "2025-07-01", "--data-dir", str(tmp_path)], capsys)
    assert code == cli.EXIT_USAGE and "invalid options" in err
    assert "ingest-sarah" in cli.STAGE_CHOICES and cli.build_parser().parse_args(["ingest-sarah", "--era5-convention", "end"]).era5_convention == "end"


def test_solar_demo_covers_the_full_grid_and_a_month_boundary(tmp_path, capsys):
    code, out, err = run(["ingest-sarah", "--mode", "solar-demo", "--data-dir", str(tmp_path), "--quiet"], capsys)
    assert code == 0, err and "OK: 96 hours, " in out and "of 2332 cells fully covered (SYNTHETIC data)" in out
    cells = pd.read_csv(tmp_path / "outputs" / "solar-demo" / SP.CELLS_FILE)
    assert len(cells) == 2332 and 0 < cells["fully_covered"].sum() < 2332
    assert cells.loc[cells["lat"] > 65.0, "fully_covered"].sum() == 0          # nothing north of the order is ever "covered"


def test_default_folders_are_only_used_when_present(tmp_path):
    with pytest.raises(S.SarahError, match="does not exist"):
        S.resolve_sarah_dir("SIS", None, {}, tmp_path)
    (tmp_path / "interim" / "smoke" / "ORD73128").mkdir(parents=True)
    assert S.resolve_sarah_dir("SIS", None, {}, tmp_path) == tmp_path / "interim" / "smoke" / "ORD73128"
    with pytest.raises(S.SarahError):
        S.resolve_sarah_dir("SID", None, {}, tmp_path)                    # SID default (ORD73129) still absent
