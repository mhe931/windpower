from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from windpower import calibrate as K
from windpower import export_site as E
from windpower import simulate as S
from windpower.io_utils import sha256_file

JSON_FILES = ("manifest.json", "grid.json", "archetypes.json", "national_hourly.json", "wind_effective.meta.json")


@pytest.fixture(scope="module")
def site(smoke_run):
    code, _, data = smoke_run
    assert code == 0
    d = data / "outputs" / "smoke" / "site" / "v1"
    return d, {n: json.loads((d / n).read_text(encoding="utf-8")) for n in JSON_FILES}


def test_site_files_and_size(site):
    d, _ = site
    assert sorted(p.name for p in d.iterdir()) == sorted([*JSON_FILES, "dataset.json", "wind_effective.bin", "roughness_mean.bin"])
    assert sum(p.stat().st_size for p in d.iterdir()) < 15_000_000


def test_every_file_is_labelled(site):
    _, js = site
    for name, obj in js.items():
        assert obj["artifact_schema_version"] == "1.0", name
        assert obj["model_version"].startswith("windpower-") and obj["run_mode"] == "smoke", name
        assert obj["synthetic"] is True and obj["demo"] is False and "SYNTHETIC" in obj["label"], name
        assert obj["units"], name


def test_manifest_content_and_hashes(site):
    d, js = site
    m = js["manifest.json"]
    assert m["calibration_applied_to_site_simulation"] is False and m["total_bytes"] < 15_000_000
    assert m["grid"]["n_lat"] == 13 and m["grid"]["n_lon"] == 17 and m["grid"]["cell_size_deg"] == 0.25
    assert m["time_range"] == {"first_utc": "2025-07-22T00:00:00Z", "last_utc": "2025-08-11T23:00:00Z", "hours": 504}
    assert m["data_coverage"]["observed_hours"] == 501 and len(m["data_coverage"]["missing_observed_hours"]) == 3
    assert m["licences"]["atlite"]["name"] == "MIT" and "SYNTHETIC" in m["attribution"]
    assert {f["path"] for f in m["files"]} == {p.name for p in d.iterdir()} - {"manifest.json"}
    for f in m["files"]:
        assert f["sha256"] == sha256_file(d / f["path"]) and f["bytes"] == (d / f["path"]).stat().st_size


def test_grid_json(site):
    _, js = site
    g = js["grid.json"]
    assert g["lat_order"] == "ascending" and np.all(np.diff(g["lats"]) > 0) and np.all(np.diff(g["lons"]) > 0)
    assert np.array(g["supported"]).shape == (13, 17) == tuple(g["shape"]) and g["index_order"] == "[lat_index][lon_index]"
    cap = np.array(g["fleet_capacity_mw"])
    assert cap.shape == (13, 17) and cap.sum() == pytest.approx(g["fleet_capacity_total_mw"], abs=1e-6)
    assert "no land/sea" in g["supported_definition"]


def test_archetypes_json_has_61_point_curves(site):
    _, js = site
    a = js["archetypes.json"]
    assert a["curve_points"] == 61 and 1 <= len(a["archetypes"]) <= 9
    for arch in a["archetypes"]:
        c = arch["power_curve"]
        assert len(c["v_ms"]) == len(c["power_mw"]) == len(c["power_per_unit"]) == 61
        assert c["v_ms"][0] == 0 and c["v_ms"][-1] == 30 and max(c["power_per_unit"]) == pytest.approx(1.0, abs=1e-6)
        assert c["power_mw"][0] == 0 and c["power_mw"][-1] == 0
        assert arch["mean_hub_height_m"] > 0 and arch["installed_mw"] > 0


def test_wind_binary_matches_meta_and_source(site, smoke_run):
    d, js = site
    meta = js["wind_effective.meta.json"]["arrays"]
    w = meta["wind_effective"]
    assert w["dtype"] == "float16" and w["byte_order"] == "little" and w["shape"] == [504, 13, 17] and w["axes"] == ["time", "lat", "lon"]
    arr = np.frombuffer((d / "wind_effective.bin").read_bytes(), dtype="<f2").reshape(w["shape"])
    assert np.isfinite(arr).all() and arr.min() >= 0
    z = np.frombuffer((d / "roughness_mean.bin").read_bytes(), dtype="<f2").reshape(meta["roughness_mean"]["shape"])
    assert z.min() >= 9e-5 and z.max() < 2
    # compare with an independent computation from the synthetic ERA5 cache
    from windpower import pipeline

    run = pipeline.Run("smoke", smoke_run[2], say=lambda m: None)
    ds = run.era5()
    t, i, j = 100, 4, 9
    expect = S.effective_wind_speed(float(ds["speed_100"].values[t, i, j]), float(ds["air_density_kgm3"].values[t, i, j]))
    assert float(arr[t, i, j]) == pytest.approx(float(expect), rel=2e-3)                 # float16 rounding
    assert js["wind_effective.meta.json"]["calibration_applied"] is False and "ln(h/z0)" in js["wind_effective.meta.json"]["hub_height_extrapolation"]


def test_national_hourly(site, smoke_run):
    _, js = site
    n = js["national_hourly.json"]
    assert len(n["utc_hours"]) == len(n["theoretical_mw"]) == len(n["observed_mw"]) == len(n["observed_available"]) == len(n["inference_only_mw"]) == 504
    assert n["utc_hours"][0] == "2025-07-22T00:00:00Z" and n["utc_hours"][-1] == "2025-08-11T23:00:00Z"
    assert n["observed_mw"].count(None) == 3 == n["coverage"]["era5_hours"] - n["coverage"]["observed_hours"]
    assert [i for i, a in enumerate(n["observed_available"]) if not a] == [i for i, v in enumerate(n["observed_mw"]) if v is None]
    assert [n["utc_hours"][i] for i in (10, 11)] == n["coverage"]["missing_observed_hours"][:2]
    assert min(n["theoretical_mw"]) >= 0 and max(n["theoretical_mw"]) <= n["installed_mw"] + 1e-6
    assert all(0 <= v <= n["installed_mw"] + 1e-6 for v in n["inference_only_mw"]) and "hour of day only" in n["inference_only_note"]


def test_demo_flag_and_labels(tmp_path, pipeline, fleet_result, joined):
    calib = K.run_calibration(joined, pipeline.sim.installed_mw, synthetic=True)
    m = E.export_site(tmp_path, ds=pipeline.ds, sim=pipeline.sim, joined=joined, fleet=fleet_result, calib=calib, run_mode="demo", synthetic=True)
    assert m["demo"] is True and m["label"] == "DEMO / SYNTHETIC - not real data" and m["run_mode"] == "demo"
    for name in JSON_FILES:
        obj = json.loads((tmp_path / "site" / "v1" / name).read_text(encoding="utf-8"))
        assert obj["demo"] is True and obj["label"].startswith("DEMO / SYNTHETIC"), name


def test_export_without_calibration_omits_inference_series(tmp_path, pipeline, fleet_result, joined):
    E.export_site(tmp_path, ds=pipeline.ds, sim=pipeline.sim, joined=joined, fleet=fleet_result, calib=None, run_mode="smoke", synthetic=True)
    nat = json.loads((tmp_path / "site" / "v1" / "national_hourly.json").read_text())
    assert "inference_only_mw" not in nat and len(nat["theoretical_mw"]) == len(pipeline.sim.hourly)


def test_size_budget_for_full_finland_q3_and_guard(tmp_path, pipeline, fleet_result, joined, monkeypatch):
    assert E.estimate_wind_bin_bytes(2208, 45, 52) <= 15_000_000                       # 10.3 MB for the real Q3 window
    assert E.estimate_wind_bin_bytes(2208, 45, 52) == 10_333_440
    monkeypatch.setattr(E, "MAX_BYTES", 10_000)
    with pytest.raises(E.ExportSizeError, match="limit"):
        E.export_site(tmp_path, ds=pipeline.ds, sim=pipeline.sim, joined=joined, fleet=fleet_result, calib=None, run_mode="smoke", synthetic=True)


def test_site_labels_for_real_runs():
    assert E.site_label("full", False) == "Retrospective ERA5 simulation" and E.site_label("smoke", True) == "SYNTHETIC - not real data"
