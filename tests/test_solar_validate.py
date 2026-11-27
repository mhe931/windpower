"""Solar hard checks and the reconciliation table (statuses, tolerances, exact missing-input reasons)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from windpower import solar_validate as SV
from windpower.validate import FAIL, PASS, SKIPPED
from test_solar_national import tiny_alloc

HOURS = pd.date_range("2025-07-30", periods=48, freq="h")
ALL_THERE = {k: "" for k in ("sarah", "era5", "public", "fingrid_267", "fingrid_248", "fingrid_362")}


def row(df, metric):
    return df[df["metric"] == metric].iloc[0]


def test_targets_carry_the_documented_values_of_the_report():
    doc = {t.key: t.documented for t in SV.TARGETS}
    assert doc["hourly_values"] == 744 and doc["fully_covered_cells"] == 777 and doc["grid_cells"] == 2332
    assert doc["energy_era5_gwh"] == 214.0 and doc["energy_hybrid_gwh"] == 216.2 and doc["energy_fingrid_248_gwh"] == 248.2
    assert doc["energy_ratio_era5_to_248"] == 0.862 and doc["energy_ratio_hybrid_to_248"] == 0.871
    assert (doc["pearson_era5_vs_248"], doc["pearson_hybrid_vs_248"], doc["pearson_era5_vs_362"], doc["pearson_hybrid_vs_362"]) == (0.981, 0.983, 0.978, 0.985)
    assert (doc["cycle_centre_era5_min"], doc["cycle_centre_hybrid_min"], doc["cycle_centre_248_min"], doc["cycle_centre_362_min"]) == (621, 617, 624, 632)
    assert doc["large_plants"] == 30 and doc["large_plants_mw"] == 250.66 and doc["largest_plant_mw"] == 103.2
    assert doc["inhabited_cells"] == 96904 and doc["inhabitants"] == 5573552 and doc["mean_cf_era5_all_cells"] == 0.195
    assert doc["ghi_sarah_covered_wm2"] == 242.5 and doc["ghi_era5_covered_wm2"] == 241.9 and doc["energy_hybrid_popw_gwh"] == 215.9
    assert len({t.key for t in SV.TARGETS}) == len(SV.TARGETS)


def test_match_differs_and_the_stated_tolerance():
    obs = {"hourly_values": 744, "fully_covered_cells": 776, "energy_era5_gwh": 214.9, "energy_hybrid_gwh": 217.5, "pearson_era5_vs_248": 0.9812,
           "cycle_centre_era5_min": 622.5, "never_covered_source_pixels_pct": 25.39}
    r = SV.reconciliation(obs, False, ALL_THERE)
    assert row(r, "hourly_values")["status"] == "match" and row(r, "hourly_values")["tolerance"] == "exact"
    assert row(r, "fully_covered_cells")["status"] == "differs" and row(r, "fully_covered_cells")["diff"] == -1     # counts are exact
    assert row(r, "energy_era5_gwh")["status"] == "match" and row(r, "energy_era5_gwh")["tolerance"] == "abs 1.07"   # 0.5 % of 214.0
    assert row(r, "energy_hybrid_gwh")["status"] == "differs" and row(r, "energy_hybrid_gwh")["diff"] == pytest.approx(1.3)
    assert row(r, "pearson_era5_vs_248")["status"] == "match" and row(r, "cycle_centre_era5_min")["status"] == "match"
    assert set(r.columns) == set(SV.RECON_COLUMNS) and len(r) == len(SV.TARGETS)


def test_missing_inputs_give_unavailable_with_the_exact_input_named():
    avail = {**ALL_THERE, "era5": "no CDS credentials", "fingrid_267": "no key"}
    r = SV.reconciliation({"hourly_values": 744, "energy_era5_gwh": 200.0, "inhabitants": 5573552}, False, avail)
    e = row(r, "energy_era5_gwh")
    assert e["status"] == "unavailable" and e["observed"] == 200.0                       # a value without its inputs is not compared
    assert "ERA5 solar data" in e["note"] and "CDS" in e["note"] and "FINGRID_API_KEY" not in "".join(row(r, "inhabitants")["note"])
    assert row(r, "inhabitants")["status"] == "match"
    n = row(r, "energy_fingrid_248_gwh")
    assert n["status"] == "unavailable" and n["observed"] is None or pd.isna(n["observed"])
    assert "Fingrid dataset 248 (FINGRID_API_KEY)" in n["note"]
    assert "ERA5 solar data" in row(r, "mean_cf_era5_all_cells")["note"]
    assert (r["status"] == "unavailable").sum() > 15


def test_synthetic_runs_are_never_compared_with_the_report():
    r = SV.reconciliation({"hourly_values": 744, "energy_era5_gwh": 214.0}, True, ALL_THERE)
    assert set(r["status"]) == {"not_reproducible_synthetic"} and row(r, "energy_era5_gwh")["observed"] == 214.0        # even an exact hit
    assert set(r["status"]).isdisjoint({"match", "differs"})


def ev(**kw) -> SV.SolarEvidence:
    return SV.SolarEvidence(mode="solar-smoke", synthetic=True, hours=HOURS, **kw)


def test_checks_are_skipped_with_the_missing_input_not_failed():
    res = {r.name: r for r in SV.run_checks(ev())}
    assert res["hourly_index:window"].status == PASS and res["mode_labelling"].status == PASS
    for name in ("sarah_alignment", "era5_units_and_shift", "cf_night_and_range", "allocation_weights", "capacity_reconciliation", "coverage_threshold",
                 "unit_consistency", "fingrid_definitions"):
        assert res[name].status == SKIPPED and res[name].detail, name
    assert "FINGRID_API_KEY" in res["capacity_reconciliation"].detail and "ERA5" in res["cf_night_and_range"].detail


def test_hourly_index_expects_the_window_length():
    assert SV.run_checks(ev(), expected_hours=744)[0].status == FAIL
    assert SV.run_checks(SV.SolarEvidence(mode="solar", synthetic=False, hours=HOURS[[0, 1, 3]]))[0].status == FAIL        # a gap in the hours


def test_allocation_and_capacity_checks_catch_inconsistencies():
    a = tiny_alloc()
    a.report.update(large_total_mw=a.large_mw, large_outside_grid_mw=0.0, cells_with_population=4, cells_with_large_plants=2, plants_outside_grid=[],
                    inhabitants_outside_grid=0.0)
    assert SV.check_allocation(ev(allocation=a)).status == PASS
    bad = tiny_alloc(share=((0.1, 0.2), (0.3, 0.3)))
    bad.report.update(a.report)
    assert SV.check_allocation(ev(allocation=bad)).status == FAIL
    assert "sum to" in SV.check_allocation(ev(allocation=bad)).detail
    off = tiny_alloc()
    off.report.update(a.report)
    assert SV.check_allocation(ev(allocation=off, plants_report={"summary_sheet": {"cumulative_mw": 99.0}})).status == FAIL    # plant list vs summary sheet
    ok = {"ok": True, "hours_checked": 5, "hours_nan": 0, "max_abs_diff_mw": 1e-13, "tolerance_mw": 1e-9}
    assert SV.check_capacity_reconciliation(ev(reconciliation=ok)).status == PASS
    assert SV.check_capacity_reconciliation(ev(reconciliation={**ok, "ok": False, "max_abs_diff_mw": 0.5})).status == FAIL


def test_coverage_threshold_check_detects_a_tampered_source_map(sarah_smoke):
    g = sarah_smoke.ongrid
    assert SV.check_coverage_threshold(ev(ongrid=g)).status == PASS
    code = g.source_code.copy()
    code[0, 3, 3] = 0 if code[0, 3, 3] != 0 else 3                           # flip one cell-hour
    assert SV.check_coverage_threshold(ev(ongrid=dataclasses.replace(g, source_code=code))).status == FAIL
    assert SV.check_coverage_threshold(ev(ongrid=dataclasses.replace(g, threshold=0.99))).status == FAIL
    fc = g.fully_covered.copy()
    fc[0, 0] = ~fc[0, 0]
    assert SV.check_coverage_threshold(ev(ongrid=dataclasses.replace(g, fully_covered=fc))).status == FAIL


def test_cf_and_era5_checks():
    good = {"cf_checks": {"era5": {"min": 0.0, "max": 0.8, "night_values": 10, "night_nonzero": 0}}}
    assert SV.check_cf(ev(pv_info=good)).status == PASS
    assert SV.check_cf(ev(pv_info={"cf_checks": {"era5": {"min": 0.0, "max": 1.2, "night_nonzero": 0}}})).status == FAIL
    assert SV.check_cf(ev(pv_info={"cf_checks": {"era5": {"min": -0.1, "max": 0.5, "night_nonzero": 0}}})).status == FAIL
    assert SV.check_cf(ev(pv_info={"cf_checks": {"hybrid": {"min": 0.0, "max": 0.5, "night_values": 4, "night_nonzero": 4}}})).status == FAIL
    era5 = {"convention": {"convention_used": "end", "detected": "end", "evidence": {"x": 1}, "shift_applied": "relabel"}, "source_detail": {"units": {"tisr": "J"}}}
    assert SV.check_era5_units_and_shift(ev(era5_info=era5)).status == PASS
    amb = {**era5, "convention": {**era5["convention"], "detected": "ambiguous"}}
    assert SV.check_era5_units_and_shift(ev(era5_info=amb)).status == FAIL                       # ambiguous and no override
    assert SV.check_era5_units_and_shift(ev(era5_info={**amb, "convention": {**amb["convention"], "override": "end"}})).status == PASS
    contradict = {**era5, "convention": {**era5["convention"], "override": "start", "override_contradicts_detection": True}}
    assert SV.check_era5_units_and_shift(ev(era5_info=contradict)).status == FAIL


def test_mode_labelling_unit_and_fingrid_checks():
    assert SV.check_mode_labelling(ev(sources={"era5": "real ERA5"})).status == FAIL              # synthetic run, real-labelled source
    assert SV.check_mode_labelling(SV.SolarEvidence(mode="solar", synthetic=False, hours=HOURS, sources={"sarah": "SYNTHETIC fixture"})).status == FAIL
    assert SV.check_mode_labelling(SV.SolarEvidence(mode="solar", synthetic=True, hours=HOURS)).status == FAIL     # flag contradicts the mode name
    idx = pd.date_range("2025-07-30", periods=3, freq="h", name="time_utc")
    t = pd.DataFrame({"era5_only_mw": [1.0, 2.0, 3.0], "capacity_267_mw": [10.0, 10.0, 10.0], "surplus_362_kwh": [1000.0, 2000.0, 3000.0],
                      "surplus_362_mw": [1.0, 2.0, 3.0]}, index=idx)
    assert SV.check_units(ev(national=t)).status == PASS
    assert SV.check_units(ev(national=t.assign(surplus_362_mw=[1.0, 2.0, 30.0]))).status == FAIL     # kWh/MW mix-up
    assert SV.check_units(ev(national=t.assign(era5_only_mw=[1.0, 2.0, 30.0]))).status == FAIL        # MW above the capacity
    assert SV.check_units(ev(national=t.assign(era5_only_mw=[1.0, -2.0, 3.0]))).status == FAIL
    v = {"verification": {"overall": "verified", "datasets": [{"dataset_id": 267, "status": "verified", "reported_unit": "MW"}]}}
    assert SV.check_fingrid_definitions(ev(fingrid_info=v)).status == PASS
    v["verification"]["datasets"][0] = {"dataset_id": 248, "status": "mismatch", "reason": "reported unit 'MW' != expected 'MWh/h'"}
    assert SV.check_fingrid_definitions(ev(fingrid_info=v)).status == FAIL
    assert SV.check_fingrid_definitions(ev(fingrid_info={"verification": {"overall": "unverified", "datasets": []}})).status == SKIPPED


def test_licences_name_every_source_and_flag_what_is_not_verified():
    lic = SV.solar_licences(False, {"public": {"population": {"licence": {"verified": False}}}})
    assert {"cm_saf_sarah3", "era5", "fingrid", "statistics_finland", "suomen_uusiutuvat", "atlite"} <= set(lic)
    assert "CC BY 4.0" in lic["fingrid"]["name"] and "verified" in lic["fingrid"]["status"] and "NOT stated" in lic["statistics_finland"]["status"]
    assert "SYNTHETIC" in SV.solar_licences(True, {})["_note"]
    assert "Copernicus" in lic["era5"]["attribution"] and "EUM_SAF_CM" in lic["cm_saf_sarah3"]["attribution"]
