from __future__ import annotations

import json

import pandas as pd
import pytest

from windpower import manifest as M
from windpower.io_utils import sha256_file


@pytest.fixture(scope="module")
def man(smoke_run):
    code, _, data = smoke_run
    assert code == 0
    out = data / "outputs" / "smoke"
    return json.loads((out / "manifest.json").read_text(encoding="utf-8")), out


def test_manifest_top_level_schema(man):
    m, _ = man
    assert m["manifest_schema_version"] == "1.0" and m["status"] == "complete"
    assert m["run_mode"] == "smoke" and m["synthetic"] is True and m["demo"] is False and "SYNTHETIC" in m["label"]
    assert m["model_version"].startswith("windpower-")
    for key in ("environment", "source_dates", "era5", "fleet", "fingrid", "calibration", "time_coverage", "files", "validation",
                "reconciliation", "metrics_statement", "licences", "unavailable", "notes"):
        assert key in m, key


def test_manifest_file_hashes_and_schemas(man):
    m, out = man
    assert m["files"] and "manifest.json" not in {f["path"] for f in m["files"]}
    paths = {f["path"] for f in m["files"]}
    assert {"hourly_joined.csv", "daily.csv", "fleet_turbines.csv", "grid.csv", "grid.parquet", "metrics.csv", "reconciliation.json",
            "validation.json", "site/v1/wind_effective.bin", "site/v1/manifest.json"} <= paths
    for f in m["files"]:
        p = out / f["path"]
        assert f["sha256"] == sha256_file(p) and f["bytes"] == p.stat().st_size, f["path"]
    hourly = next(f for f in m["files"] if f["path"] == "hourly_joined.csv")
    assert hourly["rows"] == 504 and hourly["columns"]["theoretical_wind_mw"] == {"dtype": "float64", "unit": "MW"}
    assert hourly["columns"]["imbalance_price_eur_mwh"]["unit"] == "EUR MWh-1" and hourly["columns"]["theoretical_cf"]["unit"] == "1"
    daily = next(f for f in m["files"] if f["path"] == "daily.csv")
    assert daily["columns"]["theoretical_gwh"]["unit"] == "GWh"
    fleet = next(f for f in m["files"] if f["path"] == "fleet_turbines.csv")
    assert fleet["columns"]["capacity_mw"]["unit"] == "MW" and fleet["columns"]["capacity_imputed"]["dtype"] == "bool"
    assert next(f for f in m["files"] if f["path"] == "site/v1/grid.json")["columns"] is None


def test_manifest_era5_fleet_fingrid_blocks(man):
    m, _ = man
    e = m["era5"]
    assert e["window"] == {"start_utc": "2025-07-22T00:00:00Z", "end_utc": "2025-08-11T23:00:00Z", "expected_hours": 504, "present_hours": 504}
    assert (e["grid"]["n_lat"], e["grid"]["n_lon"], e["grid"]["res_deg"]) == (13, 17, 0.25) and e["grid"]["source_lat_order"] == "descending"
    assert len(e["combined_sha256"]) == 64 and [f["name"] for f in e["files"]] == ["era5_2025_07.nc", "era5_2025_08.nc"]
    assert "SYNTHETIC" in e["source"]
    f = m["fleet"]
    assert f["snapshot"]["note"] == "current inventory may differ from the 2025 fleet" and f["snapshot"]["osm_base_timestamp_utc"]
    assert set(f["imputation_flag_counts"]) == {"capacity", "hub", "rotor", "year"} and f["imputation_flag_counts"]["capacity"]["imputed"] >= 1
    assert f["column_mapping"]["capacity"] == "Teho_MW" and sum(f["source_counts"].values()) == f["stats"]["total_turbines"]
    g = m["fingrid"]
    assert g["verification"]["overall"] == "verified" and "SYNTHETIC" in g["verification"]["note"]
    assert {d["dataset_id"]: d["reported_unit"] for d in g["verification"]["datasets"]} == {181: "MW", 248: "MW", 124: "MW", 319: "EUR/MWh"}
    assert len(g["reports"]["wind_power_mw"]["missing_hours"]) == 3 and "mean of all samples" in g["hourly_rule"]
    assert g["series"]["imbalance_price_eur_mwh"] == "EUR/MWh"
    tc = m["time_coverage"]
    assert (tc["expected_hours"], tc["present_hours"], tc["comparable_hours"]) == (504, 504, 501) and len(tc["missing_hours"]["wind_power_mw"]) == 3


def test_manifest_licences_versions_git_unavailable(man):
    m, _ = man
    lic = m["licences"]
    assert "Copernicus Climate Change Service information" in lic["era5"]["attribution"] and lic["osm"]["name"].startswith("Open Database Licence")
    assert "to be verified" in lic["fingrid"]["status"] and "CC BY 4.0" in lic["fingrid"]["name"]
    assert lic["atlite"]["name"] == "MIT" and "unavailable" in lic["arcgis_layer"]["status"] and "SYNTHETIC" in lic["_note"]
    env = m["environment"]
    assert env["libraries"]["atlite"] and env["libraries"]["windpower"] and set(env["git"]) == {"commit", "dirty"}
    assert env["python"].startswith("3.")
    inputs = {u["input"] for u in m["unavailable"]}
    assert {"real ERA5 data", "real fleet snapshot (OSM, ArcGIS)", "documented-target reconciliation"} <= inputs


def test_manifest_validation_reconciliation_and_metric_statement(man):
    m, _ = man
    assert m["validation"]["summary"]["failed"] == 0 and m["validation"]["synthetic"] is True
    assert {r["status"] for r in m["reconciliation"]} == {"not_reproducible_no_real_data"}
    combos = {(s["target"], s["split"], s["model"]) for s in m["metrics_statement"]}
    assert ("reconstructed_clean_actual", "test", "quasi_linear_spline") in combos and ("observed_actual", "test", "inference_only") in combos
    assert all("not a forecast error" in s["meaning"] for s in m["metrics_statement"] if s["target"] == "reconstructed_clean_actual")


def test_manifest_holds_no_credentials(man):
    _, out = man
    text = (out / "manifest.json").read_text(encoding="utf-8").lower()
    for forbidden in ("x-api-key", "synthetic-test-key", "cdsapi_key=", "fingrid_api_key="):
        assert forbidden not in text


def test_describe_outputs_skips_manifest_and_partial_files(tmp_path):
    (tmp_path / "a.csv").write_text("x,theoretical_wind_mw\n1,2.5\n3,4.5\n")
    (tmp_path / "b.json").write_text("{}")
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "c.csv.part").write_text("junk")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.bin").write_bytes(b"\x00\x01")
    files = {f["path"]: f for f in M.describe_outputs(tmp_path)}
    assert set(files) == {"a.csv", "b.json", "sub/d.bin"}
    assert files["a.csv"]["rows"] == 2 and files["a.csv"]["columns"]["theoretical_wind_mw"]["unit"] == "MW"
    assert files["a.csv"]["columns"]["x"] == {"dtype": "int64", "unit": None}
    assert files["sub/d.bin"]["sha256"] == sha256_file(tmp_path / "sub" / "d.bin") and files["b.json"]["columns"] is None


def test_unavailable_manifest_and_licence_block_for_real_runs(tmp_path):
    items = [{"input": "Fingrid API", "reason": "credential unavailable"}]
    m = M.unavailable_manifest("full", items, tmp_path)
    assert m["status"] == "incomplete_inputs_unavailable" and m["unavailable"] == items
    assert json.loads((tmp_path / "manifest.json").read_text())["run_mode"] == "full"
    lic = M.licence_block(False, {"copyright_text": "(c) Example registry", "license_info": "CC BY"})
    assert lic["arcgis_layer"]["copyright_text"] == "(c) Example registry" and "captured" in lic["arcgis_layer"]["status"]
    assert "have not been checked" in lic["_note"]


def test_git_info_shape_and_versions(tmp_path):
    g = M.git_info()
    assert set(g) == {"commit", "dirty"} and (g["commit"] is None or len(g["commit"]) == 40)
    assert M.git_info(repo=tmp_path) == {"commit": None, "dirty": None}               # not a repository: no exception
    v = M.package_versions()
    assert v["numpy"] and v["windpower"] == __import__("windpower").__version__


def test_metric_statement_labels():
    s = M.metric_statement([{"target": "observed_actual", "split": "test", "model": "inference_only"},
                            {"target": "observed_actual", "split": "test", "model": "inference_only"}])
    assert len(s) == 1 and "Fingrid-reported" in s[0]["meaning"]


def test_jsonable_handles_numpy_nan_and_timestamps():
    import numpy as np

    from windpower.io_utils import jsonable

    out = jsonable({"a": np.float64("nan"), "b": np.int64(3), "c": pd.Timestamp("2025-07-01"), "d": np.array([1.0, np.inf]), "e": {2, 1}, "f": np.bool_(True)})
    assert out == {"a": None, "b": 3, "c": "2025-07-01T00:00:00", "d": [1.0, None], "e": [1, 2], "f": True}
    json.dumps(out, allow_nan=False)
