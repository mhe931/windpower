from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pandas as pd
import pytest

from windpower import calibrate as K
from windpower import cli, validate as V
from windpower import simulate as S


@pytest.fixture(scope="module")
def calib(joined, pipeline):
    return K.run_calibration(joined, pipeline.sim.installed_mw, synthetic=True)


@pytest.fixture(scope="module")
def ev(pipeline, fleet_result, joined, smoke_axes, calib):
    lats, lons = smoke_axes
    n_comp = int(joined["actual_wind_mw"].notna().sum())
    counts = {"era5_hours": len(joined), "joined_rows": len(joined), "comparable_hours": n_comp,
              "dropped_missing_actual": len(joined) - n_comp, "join_coverage": n_comp / len(joined)}
    return V.Evidence(
        era5_index=pipeline.sim.hourly.index, expected_hours=len(joined), sim=pipeline.sim, joined=joined, join_counts=counts,
        capacity_matrix=fleet_result.capacity_matrix, turbines=fleet_result.turbines, lats=lats, lons=lons, totals=fleet_result.totals,
        grid_summary_mw=float(fleet_result.pixels["installed_capacity_mw"].sum()), specs_mw=float(fleet_result.specs["cap_mw"].sum()),
        calib=calib, daily=S.daily_summary(joined))


def by_name(results):
    return {r.name: r for r in results}


# --- the whole suite on good data ----------------------------------------------------------------
def test_all_checks_pass_on_good_data(ev):
    results = V.run_all_checks(ev)
    assert [r.name for r in results] == ["hourly_index:era5", "hourly_index:theoretical", "hourly_index:joined", "hourly_index:calibration_rows",
                                         "join_coverage", "grid_orientation", "capacity_totals", "split_isolation", "units",
                                         "frontier_bounds", "metrics_labels", "forward_check"]
    assert V.failed_checks(results) == [], [r.detail for r in V.failed_checks(results)]
    assert all(r.status == V.PASS for r in results)                         # 70 days -> forward check runs
    assert all(r.detail for r in results) and set(V.results_frame(results).columns) == {"name", "status", "detail"}


def test_results_are_structured():
    r = V.CheckResult("x", V.FAIL, "why", {"k": 1})
    assert r.failed and r.as_dict() == {"name": "x", "status": "fail", "detail": "why", "data": {"k": 1}}


def test_failed_hard_check_gives_nonzero_exit_code():
    ok = V.CheckResult("a", V.PASS, "fine")
    bad = V.CheckResult("b", V.FAIL, "broken")
    assert cli.summarise([ok, V.CheckResult("c", V.SKIPPED, "n/a")], "T", True) == 0
    assert cli.summarise([ok, bad], "T", True) == cli.EXIT_VALIDATION == 1


# --- 1. hourly index ----------------------------------------------------------------------------------
def test_hourly_index_ok_and_expected_count():
    idx = pd.date_range("2025-07-01", periods=48, freq="h")
    assert V.check_hourly_index(idx, "t", 48).status == V.PASS
    assert V.check_hourly_index(idx.tz_localize("UTC"), "t", 48).status == V.PASS


def test_hourly_index_duplicate_hour_fails():
    idx = pd.DatetimeIndex(list(pd.date_range("2025-07-01", periods=5, freq="h")) + [pd.Timestamp("2025-07-01 03:00")]).sort_values()
    r = V.check_hourly_index(idx, "t")
    assert r.failed and "duplicate timestamp" in r.detail


@pytest.mark.parametrize("make, text", [
    (lambda: pd.DatetimeIndex(["2025-07-01 00:00", "2025-07-01 02:00"]), "not strictly hourly"),
    (lambda: pd.DatetimeIndex(["2025-07-01 01:00", "2025-07-01 00:00"]), "not monotonic"),
    (lambda: pd.DatetimeIndex(["2025-07-01 00:30", "2025-07-01 01:30"]), "not aligned to the hour"),
    (lambda: pd.date_range("2025-07-01", periods=3, freq="h", tz="Europe/Helsinki"), "is not UTC"),
])
def test_hourly_index_faults(make, text):
    r = V.check_hourly_index(make(), "t")
    assert r.failed and text in r.detail


def test_hourly_index_wrong_count_and_irregular_allowed_when_not_regular():
    idx = pd.date_range("2025-07-01", periods=48, freq="h")
    assert "48 hours, expected 49" in V.check_hourly_index(idx, "t", 49).detail
    gappy = idx.delete([5, 6])
    assert V.check_hourly_index(gappy, "t").failed
    assert V.check_hourly_index(gappy, "t", regular=False).status == V.PASS


# --- 2. join coverage ---------------------------------------------------------------------------------
def test_join_coverage_threshold():
    ok = {"era5_hours": 2208, "comparable_hours": 2202, "dropped_missing_actual": 6, "joined_rows": 2208}
    r = V.check_join_coverage(ok)
    assert r.status == V.PASS and "2202 comparable of 2208" in r.detail and r.data["coverage"] == pytest.approx(2202 / 2208)
    low = dict(ok, comparable_hours=1000, dropped_missing_actual=1208)
    assert V.check_join_coverage(low).failed
    assert V.check_join_coverage(low, min_coverage=0.4).status == V.PASS          # configurable


# --- 3. grid orientation ------------------------------------------------------------------------------------
def test_grid_orientation_good_and_flipped(ev):
    args = (ev.capacity_matrix, ev.turbines, ev.lats, ev.lons)
    assert V.check_grid_orientation(*args).status == V.PASS
    flipped = V.check_grid_orientation(ev.capacity_matrix[::-1], ev.turbines, ev.lats, ev.lons)
    assert flipped.failed and "differ from the turbines inside them" in flipped.detail
    assert V.check_grid_orientation(ev.capacity_matrix[:, ::-1], ev.turbines, ev.lats, ev.lons).failed


def test_grid_orientation_transposed_and_descending_axes(ev):
    assert "matrix shape" in V.check_grid_orientation(ev.capacity_matrix.T, ev.turbines, ev.lats, ev.lons).detail
    r = V.check_grid_orientation(ev.capacity_matrix[::-1], ev.turbines, ev.lats[::-1], ev.lons)
    assert r.failed and "latitude axis is not strictly ascending" in r.detail


def test_grid_orientation_swapped_pixel_indices(ev):
    t = ev.turbines.copy()
    t[["pixel_i", "pixel_j"]] = t[["pixel_j", "pixel_i"]].to_numpy()
    r = V.check_grid_orientation(ev.capacity_matrix, t.assign(pixel_i=t["pixel_i"].clip(upper=12)), ev.lats, ev.lons)
    assert r.failed and "pixel_i/pixel_j do not index" in r.detail


def test_grid_orientation_edge_turbine_is_not_double_counted(ev):
    expected = V.expected_capacity_matrix(ev.turbines, ev.lats, ev.lons)
    np.testing.assert_allclose(expected, ev.capacity_matrix, rtol=1e-9)
    assert ev.turbines["capacity_mw"].sum() > expected.sum()                          # edge + outside turbines are unassigned


# --- 4. capacity totals ---------------------------------------------------------------------------------------
def totals_args(ev):
    return dict(totals=ev.totals, grid_summary_mw=ev.grid_summary_mw, specs_mw=ev.specs_mw,
                archetype_mw=float(ev.sim.archetypes["installed_mw"].sum()), n_archetypes=len(ev.sim.archetypes))


def test_capacity_totals_pass_and_report_unassigned(ev):
    r = V.check_capacity_totals(**totals_args(ev))
    assert r.status == V.PASS and "unassigned" in r.detail and r.data["unassigned_mw"] > 0


@pytest.mark.parametrize("key, delta, text", [
    ("matrix_mw", 5.0, "matrix+unassigned"),
    ("pixel_agg_mw", -3.0, "pixel_agg+unassigned"),
    ("grid_summary_mw", 1.0, "grid_summary+unassigned"),
    ("specs_mw", 2.0, "spec_table"),
    ("archetype_mw", 40.0, "archetype table"),
])
def test_capacity_totals_mismatch_fails(ev, key, delta, text):
    args = totals_args(ev)
    if key in args["totals"]:
        args["totals"] = {**args["totals"], key: args["totals"][key] + delta}
    else:
        args[key] = args[key] + delta
    r = V.check_capacity_totals(**args)
    assert r.failed and text in r.detail


def test_capacity_totals_unassigned_must_balance(ev):
    args = totals_args(ev)
    args["totals"] = {**args["totals"], "unassigned_mw": args["totals"]["unassigned_mw"] + 1.0}
    assert V.check_capacity_totals(**args).failed


def test_documented_capacity_is_not_a_pass_fail_criterion(ev):
    """9,790.18 MW is a reconciliation target; synthetic totals of ~177 MW must still pass the consistency check."""
    assert V.check_capacity_totals(**totals_args(ev)).status == V.PASS


# --- 5. split isolation ----------------------------------------------------------------------------------------
def test_split_isolation_passes(ev, pipeline):
    r = V.check_split_isolation(ev.calib.table, ev.calib.models, clip_max_mw=pipeline.sim.installed_mw)
    assert r.status == V.PASS and "identical under test-label perturbation" in r.detail
    assert r.data["anchors_train"] <= r.data["anchors_all_rows"]


def test_split_isolation_block_straddling_fails(ev):
    t = ev.calib.table.copy()
    test_block = t.loc[t["split"] == "test", "block_id"].iloc[0]
    t.loc[t.index[(t["block_id"] == test_block)][0], "split"] = "train"          # one hour of a test block moves to train
    r = V.check_split_isolation(t)
    assert r.failed and "split between train and test" in r.detail


def test_split_isolation_non_contiguous_blocks_and_duplicates_fail(ev):
    t = ev.calib.table.copy()
    t["block_id"] = np.arange(len(t)) % 7                                           # interleaved "blocks"
    assert "contiguous" in V.check_split_isolation(t).detail
    dup = pd.concat([ev.calib.table, ev.calib.table.iloc[:1]])
    assert "duplicate timestamps" in V.check_split_isolation(dup).detail


def test_split_isolation_detects_a_leaky_fit(ev, pipeline):
    def leaky(df, train_index, clip):                                              # ignores the train index: fits on everything
        return K.fit_models(df.assign(split="train"), df.index, clip)

    r = V.check_split_isolation(ev.calib.table, fit_fn=leaky, clip_max_mw=pipeline.sim.installed_mw)
    assert r.failed and "leakage" in r.detail


def test_split_isolation_detects_anchor_leak(ev, pipeline):
    models = copy.copy(ev.calib.models)
    models.frontier = dataclasses.replace(models.frontier, n_anchors=models.frontier.n_anchors + 25)   # as if anchors came from test rows
    r = V.check_split_isolation(ev.calib.table, models, clip_max_mw=pipeline.sim.installed_mw)
    assert r.failed and "anchors leak" in r.detail


def test_split_isolation_detects_stored_models_not_matching_train_refit(ev, pipeline):
    models = copy.copy(ev.calib.models)
    models.frontier = dataclasses.replace(models.frontier, beta=models.frontier.beta + 0.5)
    r = V.check_split_isolation(ev.calib.table, models, clip_max_mw=pipeline.sim.installed_mw)
    assert r.failed and "do not equal a refit" in r.detail


def test_split_isolation_single_split_is_skipped(ev):
    t = ev.calib.table.assign(split="train")
    assert V.check_split_isolation(t).status == V.SKIPPED


# --- 6. units ------------------------------------------------------------------------------------------------------
def test_units_pass(ev):
    r = V.check_units(ev.sim, ev.joined, ev.calib.table, ev.daily)
    assert r.status == V.PASS and 0 <= r.data["cf_min"] <= r.data["cf_max"] <= 1 + 1e-9


def test_units_cf_above_one_fails(ev):
    small = dataclasses.replace(ev.sim, installed_mw=ev.sim.installed_mw / 2)         # CF would reach 2
    r = V.check_units(small, ev.joined)
    assert r.failed and "capacity factor outside [0, 1]" in r.detail


def test_units_negative_power_fails(ev):
    bad = ev.joined.copy()
    bad.iloc[3, bad.columns.get_loc("actual_wind_mw")] = -25.0
    assert "negative actual_wind_mw" in V.check_units(ev.sim, bad).detail
    hourly = ev.sim.hourly.copy()
    hourly.iloc[0, 0] = -1.0
    r = V.check_units(dataclasses.replace(ev.sim, hourly=hourly), ev.joined)
    assert r.failed and "negative simulated power" in r.detail
    tbl = ev.calib.table.copy()
    tbl.iloc[0, tbl.columns.get_loc("pred_inference_mw")] = -3.0
    assert "negative pred_inference_mw" in V.check_units(ev.sim, ev.joined, tbl).detail


def test_units_daily_energy_must_equal_mw_times_hours(ev):
    wrong = ev.daily.copy()
    wrong["theoretical_gwh"] = wrong["theoretical_gwh"] * 1000.0                       # MWh mislabelled as GWh
    r = V.check_units(ev.sim, ev.joined, daily=wrong)
    assert r.failed and "daily theoretical_gwh" in r.detail


def test_frontier_bounds_check(ev):
    assert V.check_frontier_bounds(ev.calib.models.frontier).status == V.PASS
    out = dataclasses.replace(ev.calib.models.frontier, beta=np.array([0.0, 2.0, 0.0, 0.0, 0.0]))
    assert V.check_frontier_bounds(out).failed


# --- 7. metrics table and forward -------------------------------------------------------------------------------------
def test_metrics_labels_good_and_faulty(ev):
    good = ev.calib.metrics
    assert V.check_metrics_labels(good).status == V.PASS
    assert V.check_metrics_labels([]).failed
    broken = [dict(good[0], target="whatever")] + good[1:]
    assert "bad label" in V.check_metrics_labels(broken).detail
    no_unit = [dict(good[0], unit="")] + good[1:]
    assert V.check_metrics_labels(no_unit).failed
    missing_key = [{k: v for k, v in good[0].items() if k != "split"}] + good[1:]
    assert "record keys" in V.check_metrics_labels(missing_key).detail
    misplaced = [dict(m, target=K.REC) if m["model"] == "inference_only" else m for m in good]
    assert "inference_only metric attached to a non-observed target" in V.check_metrics_labels(misplaced).detail
    only_obs = [m for m in good if m["target"] == K.OBS]
    assert "targets present" in V.check_metrics_labels(only_obs).detail
    no_reg = [m for m in good if m["metric"] != "regression_r2"]
    assert "pearson_r2 reported without regression_r2" in V.check_metrics_labels(no_reg).detail
    nan = [dict(good[0], value=float("nan"))] + good[1:]
    assert V.check_metrics_labels(nan).failed


def test_metrics_frame_is_labelled_and_sorted(ev):
    f = V.metrics_frame(ev.calib.metrics)
    assert list(f.columns) == ["target", "split", "model", "metric", "value", "unit", "n"] and len(f) == len(ev.calib.metrics)
    assert f.equals(f.sort_values(["target", "split", "model", "metric"], kind="stable").reset_index(drop=True))


def test_forward_check_statuses(ev):
    assert V.check_forward(ev.calib.forward).status == V.PASS
    assert V.check_forward({"status": "skipped", "reason": "coverage 5.0 days < 60 days required"}).status == V.SKIPPED
    bad = dict(ev.calib.forward, train_end_utc="2025-09-01T00:00:00Z", test_start_utc="2025-08-01T00:00:00Z")
    assert V.check_forward(bad).failed


# --- reconciliation ---------------------------------------------------------------------------------------------------------
OBSERVED = {"era5_hours": 2208, "turbines": 2228, "osm_kept": 1859, "arcgis_kept": 2080, "capacity_mw": 9790.18, "active_cells": 208,
            "year_coverage_pct": 28.5, "mean_speed_100": 5.46, "mean_air_density": 1.202, "icing_share_pct": 0.07}


def test_reconciliation_contains_documented_targets(ev):
    r = V.full_reconciliation(OBSERVED, ev.calib, synthetic=True)
    assert list(r.columns) == V.RECON_COLUMNS
    doc = {(m, t, s, mo): d for m, d, t, s, mo in zip(r["metric"], r["documented"], r["target"], r["split"], r["model"])}
    assert doc[("era5_hours", "n/a", "n/a", "n/a")] == 2208 and doc[("comparable_hours", "n/a", "n/a", "n/a")] == 2202
    assert doc[("turbines", "n/a", "n/a", "n/a")] == 2228 and doc[("capacity_mw", "n/a", "n/a", "n/a")] == 9790.18
    assert doc[("active_cells", "n/a", "n/a", "n/a")] == 208
    assert doc[("train_hours", "n/a", "n/a", "n/a")] == 1699 and doc[("test_hours", "n/a", "n/a", "n/a")] == 503
    assert doc[("rmse", K.OBS, "test", "atlite_raw")] == 1166.7
    assert doc[("rmse", K.REC, "test", "quasi_linear_spline")] == 323.2
    assert doc[("pearson_r", K.OBS, "all", "atlite_raw")] == 0.9296
    assert (r["status"] == "not_reproducible_no_real_data").all()
    assert r["metric"].is_unique is False                                               # rmse appears per split/model, labelled


def test_reconciliation_matching_synthetic_numbers_are_still_not_reproducible(ev):
    r = V.full_reconciliation({k: v for k, v in OBSERVED.items()}, ev.calib, synthetic=True)
    assert set(r["status"]) == {"not_reproducible_no_real_data"} and r["diff"].notna().sum() > 0


def test_reconciliation_status_rules_for_real_data(ev):
    r = V.data_reconciliation(OBSERVED, synthetic=False)
    assert set(r["status"]) == {"match"}
    off = V.data_reconciliation({**OBSERVED, "turbines": 2229, "capacity_mw": 9900.0, "mean_speed_100": 5.47, "era5_hours": None}, synthetic=False)
    s = dict(zip(off["metric"], off["status"]))
    assert s["turbines"] == "differs"            # counts are exact
    assert s["capacity_mw"] == "differs"         # > 1 %
    assert s["mean_speed_100"] == "match"        # within 1 %
    assert s["era5_hours"] == "unavailable"
    assert off.loc[off["metric"] == "turbines", "diff"].item() == 1.0


def test_reconciliation_synthetic_observed_values_present(ev):
    r = V.full_reconciliation({**OBSERVED, "turbines": 50, "capacity_mw": 177.3}, ev.calib, synthetic=True)
    row = r[r["metric"] == "capacity_mw"].iloc[0]
    assert row["observed"] == 177.3 and row["diff"] == pytest.approx(177.3 - 9790.18)
