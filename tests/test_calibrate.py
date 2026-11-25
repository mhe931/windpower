from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from windpower import calibrate as K


def frame(theo, actual, price, cons, start="2025-07-01T00:00:00"):
    n = len(theo)
    df = pd.DataFrame({"theoretical_wind_mw": theo, "actual_wind_mw": actual, "imbalance_price_eur_mwh": price,
                       "total_consumption_mw": cons}, index=pd.date_range(start, periods=n, freq="h", name="valid_time"))
    return K.prepare(df)[0]


@pytest.fixture(scope="module")
def calib(joined, pipeline):
    return K.run_calibration(joined, pipeline.sim.installed_mw, synthetic=True)


@pytest.fixture(scope="module")
def prepared(joined):
    return K.prepare(joined)[0]


@pytest.fixture(scope="module")
def split_df(prepared):
    return K.assign_blocks(prepared)[0]


# --- preparation -----------------------------------------------------------------------------
def test_prepare_drops_incomplete_rows_and_adds_columns(joined):
    out, counts = K.prepare(joined)
    assert counts == {"input_rows": len(joined), "usable_rows": len(joined) - 3, "dropped_missing_inputs": 3}
    assert out[["actual_wind_mw", "imbalance_price_eur_mwh", "total_consumption_mw"]].notna().all().all()
    h = out.index[7].hour
    assert out["hour_sin"].iloc[7] == pytest.approx(np.sin(2 * np.pi * h / 24)) and out["hour_cos"].iloc[7] == pytest.approx(np.cos(2 * np.pi * h / 24))
    np.testing.assert_allclose(out["wind_to_cons_ratio"], out["theoretical_wind_mw"] / out["total_consumption_mw"])
    np.testing.assert_allclose(out["residual_raw_mw"], out["theoretical_wind_mw"] - out["actual_wind_mw"])


def test_prepare_errors_and_consumption_floor():
    with pytest.raises(K.CalibrationError, match="missing required column"):
        K.prepare(pd.DataFrame({"theoretical_wind_mw": [1.0]}, index=pd.date_range("2025-07-01", periods=1, freq="h")))
    dup = pd.DataFrame({c: [1.0, 1.0] for c in K.REQUIRED_COLUMNS}, index=pd.DatetimeIndex(["2025-07-01", "2025-07-01"]))
    with pytest.raises(K.CalibrationError, match="duplicate timestamps"):
        K.prepare(dup)
    f = frame([100.0], [50.0], [30.0], [0.0])
    assert f["wind_to_cons_ratio"].item() == 100.0                    # consumption floored at 1 MW


# --- block split ---------------------------------------------------------------------------------
def test_split_is_deterministic_and_seed_sensitive(prepared):
    a, ba = K.assign_blocks(prepared)
    b, bb = K.assign_blocks(prepared)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ba, bb)
    other, _ = K.assign_blocks(prepared, seed=2027)
    assert not (other["split"] == a["split"]).all()


def test_split_blocks_are_whole_disjoint_and_72h(split_df):
    assert set(split_df["split"]) == {"train", "test"}
    assert (split_df.groupby("block_id")["split"].nunique() == 1).all()          # no block straddles the split
    expected_id = ((split_df.index - split_df.index.min()) // pd.Timedelta(hours=72)).astype(int)
    assert (split_df["block_id"].to_numpy() == expected_id.to_numpy()).all()
    spans = split_df.groupby("block_id").apply(lambda g: g.index.max() - g.index.min(), include_groups=False)
    assert (spans <= pd.Timedelta(hours=71)).all() and spans.iloc[:-2].eq(pd.Timedelta(hours=71)).sum() >= len(spans) - 4
    tr, te = split_df.index[split_df["split"] == "train"], split_df.index[split_df["split"] == "test"]
    assert len(tr.intersection(te)) == 0 and len(tr) + len(te) == len(split_df)


def test_split_is_stratified_by_block_mean_wind_quantile(split_df, prepared):
    _, blocks = K.assign_blocks(prepared)
    assert blocks["wind_quartile"].nunique() == min(4, len(blocks) // 2)
    n_test_expected = sum(max(1, int(round(0.2 * len(g)))) if len(g) >= 2 else 0 for _, g in blocks.groupby("wind_quartile"))
    assert (blocks["split"] == "test").sum() == n_test_expected
    for _, g in blocks.groupby("wind_quartile"):
        assert (g["split"] == "test").any() and (g["split"] == "train").any()   # every stratum is in both sets
    m = blocks.groupby("wind_quartile")["theoretical_wind_mw"].mean()
    assert m.is_monotonic_increasing                                            # strata really are wind-ordered
    assert 0.1 < (split_df["split"] == "test").mean() < 0.35


def test_31_blocks_give_24_train_7_test_blocks():
    idx = pd.date_range("2025-07-01", periods=2208, freq="h")
    theo = pd.DataFrame({"theoretical_wind_mw": np.random.default_rng(1).random(2208) * 1000}, index=idx)
    out, blocks = K.assign_blocks(theo)
    assert len(blocks) == 31 and (blocks["split"] == "test").sum() == 7 and (blocks["split"] == "train").sum() == 24
    assert blocks["wind_quartile"].nunique() == 4 and out["block_id"].max() == 30


def test_rng_order_is_pinned():
    """8 blocks with ascending wind -> quartiles of 2; seed 2026, groups shuffled in quartile order."""
    idx = pd.date_range("2025-07-01", periods=8 * 72, freq="h")
    df = pd.DataFrame({"theoretical_wind_mw": ((np.arange(len(idx)) // 72) + 1) * 100.0}, index=idx)
    _, blocks = K.assign_blocks(df)
    assert sorted(blocks.loc[blocks["split"] == "test", "block_id"]) == [1, 3, 4, 6]
    # independent re-statement of the notebook loop
    rng = np.random.default_rng(2026)
    test = []
    for q in range(4):
        b = np.array([2 * q, 2 * q + 1])
        rng.shuffle(b)
        test.append(int(b[0]))
    assert sorted(test) == [1, 3, 4, 6]


def test_split_uses_no_labels(prepared):
    perturbed = prepared.copy()
    perturbed["actual_wind_mw"] *= 7
    perturbed["imbalance_price_eur_mwh"] = -50.0
    pd.testing.assert_series_equal(K.assign_blocks(prepared)[0]["split"], K.assign_blocks(perturbed)[0]["split"])


def test_tiny_inputs_split_gracefully():
    idx = pd.date_range("2025-07-01", periods=100, freq="h")
    out, blocks = K.assign_blocks(pd.DataFrame({"theoretical_wind_mw": np.arange(100.0)}, index=idx))
    assert len(blocks) == 2 and (blocks["split"] == "test").sum() >= 1
    single, sb = K.assign_blocks(pd.DataFrame({"theoretical_wind_mw": np.arange(50.0)}, index=idx[:50]))
    assert len(sb) == 1 and set(single["split"]) == {"train"}


# --- anchors, frontier, reconstruction -----------------------------------------------------------------
def test_anchor_thresholds_are_strict():
    f = frame(theo=[1000, 1000, 1000, 3800, 3799], actual=[900] * 5, price=[25, 20, 25, 25, 25], cons=[5000, 5000, 2500, 20000, 20000])
    # rows: ok | price == 20 | ratio == 0.40 | theo == 3800 | ok
    assert K.anchor_mask(f).tolist() == [True, False, False, False, True]


def test_frontier_recovers_known_truth_and_respects_bounds():
    n = 24 * 20
    rng = np.random.default_rng(0)
    theo = rng.uniform(100, 3000, n)
    idx = pd.date_range("2025-07-01", periods=n, freq="h", name="valid_time")
    hs, hc = K.hour_harmonics(idx)
    actual = -20.0 + 0.9 * theo - 15.0 * (theo / 1000) ** 2 + 30 * hs - 10 * hc
    f = frame(theo, actual, np.full(n, 60.0), np.full(n, 12000.0))
    f["split"] = "train"
    fit = K.fit_frontier(f, f.index)
    np.testing.assert_allclose(fit.beta, [-20.0, 0.9, 15.0, 30.0, -10.0], atol=0.05)
    assert fit.within_bounds and fit.n_anchors == n and fit.success


def test_frontier_needs_enough_anchors():
    f = frame(np.linspace(100, 900, 50), np.linspace(90, 800, 50), np.full(50, 5.0), np.full(50, 10000.0))   # price < 20
    f["split"] = "train"
    with pytest.raises(K.CalibrationError, match="train anchor hours"):
        K.fit_frontier(f, f.index)


def test_gate_and_reconstruction_by_hand():
    beta = np.array([0.0, 0.9, 0.0, 0.0, 0.0])
    f = frame(theo=[1000, 1000, 1000, 1000], actual=[700, 700, 700, 950],
              price=[-5, 15, 7.5, 40], cons=[2000, 1818.1818181818182, 1818.1818181818182, 2000])
    # ratios: .5, .55, .55, .5 ; price scores: 1, 0, .5, 0 ; grid scores: 1/3, .5, .5, 1/3
    r = K.reconstruct(f, beta)
    np.testing.assert_allclose(r["score_price"], [1.0, 0.0, 0.5, 0.0])
    np.testing.assert_allclose(r["score_grid"], [1 / 3, 0.5, 0.5, 1 / 3], atol=1e-9)
    np.testing.assert_allclose(r["curtail_gate_w"], [1.0, 0.5, 0.75, 1 / 3], atol=1e-9)
    np.testing.assert_allclose(r["p_base_uncurtailed_mw"], 900.0)
    np.testing.assert_allclose(r["excess_shortfall_mw"], [200, 200, 200, 0])                 # max(0, p_base - actual)
    np.testing.assert_allclose(r["estimated_curtailment_mw"], [200, 100, 150, 0], atol=1e-6)
    np.testing.assert_allclose(r["actual_wind_clean_mw"], [900, 800, 850, 950], atol=1e-6)
    np.testing.assert_allclose(r["residual_clean_mw"], [100, 200, 150, 50], atol=1e-6)


def test_baseline_is_clipped_to_zero_and_1p15_theoretical():
    f = frame([1000.0, 1000.0, 1000.0], [500.0] * 3, [50.0] * 3, [9000.0] * 3)
    hi = K.reconstruct(f, np.array([5000.0, 1.0, 0.0, 0.0, 0.0]))
    lo = K.reconstruct(f, np.array([-5000.0, 1.0, 0.0, 0.0, 0.0]))
    assert (hi["p_base_uncurtailed_mw"] == 1150.0).all() and (lo["p_base_uncurtailed_mw"] == 0.0).all()


def test_frontier_ignores_everything_outside_train_index(split_df):
    train_idx = split_df.index[split_df["split"] == "train"]
    ref = K.fit_frontier(split_df, train_idx)
    blind = split_df.copy()
    blind.loc[blind["split"] == "test", ["actual_wind_mw", "imbalance_price_eur_mwh", "total_consumption_mw", "theoretical_wind_mw"]] = np.nan
    np.testing.assert_array_equal(K.fit_frontier(blind, train_idx).beta, ref.beta)
    assert ref.n_anchors == int(K.anchor_mask(split_df[split_df["split"] == "train"]).sum()) and ref.n_train_rows == len(train_idx)
    assert ref.within_bounds


def test_fitting_on_non_train_rows_is_refused(split_df):
    test_idx = split_df.index[split_df["split"] == "test"]
    with pytest.raises(K.LeakageError, match="non-train row"):
        K.fit_frontier(split_df, test_idx)
    with pytest.raises(K.LeakageError):
        K.fit_models(split_df, split_df.index)                                  # everything, incl. test rows
    with pytest.raises(K.CalibrationError, match="not in the table"):
        K.fit_frontier(split_df, pd.DatetimeIndex(["1999-01-01"]))


# --- leakage perturbation ------------------------------------------------------------------------------
def _coefs(m: K.ModelSet):
    return [m.frontier.beta, m.linear.coef_, np.atleast_1d(m.linear.intercept_), m.spline[-1].coef_,
            np.atleast_1d(m.spline[-1].intercept_), m.inference.pipeline[-1].coef_, np.atleast_1d(m.inference.pipeline[-1].intercept_)]


def test_perturbing_test_rows_leaves_every_fit_unchanged(split_df):
    train_idx = split_df.index[split_df["split"] == "train"]
    ref = K.fit_models(split_df, train_idx, 200.0)
    rng = np.random.default_rng(5)
    bad = split_df.copy()
    t = bad["split"] == "test"
    n = int(t.sum())
    bad.loc[t, "actual_wind_mw"] = rng.uniform(0, 1e4, n)
    bad.loc[t, "imbalance_price_eur_mwh"] = rng.uniform(-100, 100, n)
    bad.loc[t, "total_consumption_mw"] = rng.uniform(1, 1e5, n)
    bad.loc[t, "theoretical_wind_mw"] = rng.uniform(0, 1e4, n)
    again = K.fit_models(bad, train_idx, 200.0)
    for a, b in zip(_coefs(ref), _coefs(again)):
        np.testing.assert_array_equal(a, b)


def test_perturbing_train_rows_does_change_the_fit(split_df):
    train_idx = split_df.index[split_df["split"] == "train"]
    ref = K.fit_models(split_df, train_idx, 200.0)
    bad = split_df.copy()
    bad.loc[train_idx, "actual_wind_mw"] = bad.loc[train_idx, "actual_wind_mw"] * 0.5
    changed = K.fit_models(bad, train_idx, 200.0)
    assert not np.allclose(ref.frontier.beta, changed.frontier.beta)             # the guard test is sensitive


def test_end_to_end_run_is_label_blind_on_test_rows(joined, pipeline, calib):
    test_hours = calib.table.index[calib.table["split"] == "test"]
    bad = joined.copy()
    bad.loc[test_hours, "actual_wind_mw"] = np.random.default_rng(9).uniform(0, 5000, len(test_hours))
    bad.loc[test_hours, "imbalance_price_eur_mwh"] = -99.0
    bad.loc[test_hours, "total_consumption_mw"] = 1.0
    again = K.run_calibration(bad, pipeline.sim.installed_mw, synthetic=True)
    pd.testing.assert_series_equal(again.table["split"], calib.table["split"])
    for a, b in zip(_coefs(calib.models), _coefs(again.models)):
        np.testing.assert_array_equal(a, b)
    train = lambda r: [m for m in r.metrics if m["split"] == "train"]            # noqa: E731
    assert train(again) == train(calib)


# --- inference-only model -----------------------------------------------------------------------------------
def test_inference_model_uses_only_theoretical_and_hour(calib, split_df, pipeline):
    m = calib.models.inference
    assert K.InferenceOnlyModel.features == ("theoretical_wind_mw", "hour_sin", "hour_cos")
    assert list(m.pipeline.feature_names_in_) == ["theoretical_wind_mw", "hour_sin", "hour_cos"]
    forbidden = {"imbalance_price_eur_mwh", "total_consumption_mw", "solar_power_mw", "net_load_mw", "actual_wind_mw", "wind_to_cons_ratio"}
    assert not forbidden & set(m.pipeline.feature_names_in_)
    pred = m.predict(np.array([0.0, 50.0, 100.0, 1e6]), np.array([3, 12, 18, 12]))
    assert (pred >= 0).all() and (pred <= pipeline.sim.installed_mw).all()
    capped = K.InferenceOnlyModel(clip_max_mw=10.0).fit(split_df, split_df.index[split_df["split"] == "train"])
    assert capped.predict(np.array([0.0, 100.0, 200.0]), np.array([3, 12, 18])).max() <= 10.0
    with pytest.raises(TypeError):
        m.predict(np.array([1.0]), np.array([1]), np.array([30.0]))               # no price/consumption argument exists


def test_inference_model_beats_raw_theoretical_on_test(calib):
    rows = {(r["metric"], r["model"]): r["value"] for r in calib.metrics if r["split"] == "test" and r["target"] == "observed_actual"}
    assert rows[("rmse", "inference_only")] < rows[("rmse", "atlite_raw")]
    assert rows[("regression_r2", "inference_only")] > rows[("regression_r2", "atlite_raw")]


# --- metrics -----------------------------------------------------------------------------------------------------
def test_pearson_r2_is_not_regression_r2():
    theo, act = np.array([1.0, 2, 3, 4]), np.array([2.0, 4, 6, 8])
    assert K.pearson_r(theo, act) == pytest.approx(1.0)
    assert K.pearson_r(theo, act) ** 2 == pytest.approx(1.0)
    assert K.regression_r2(theo, act) == pytest.approx(1 - 30 / 20)                # SSE 30, SST 20 -> -0.5
    assert K.regression_r2(act, act) == 1.0


def test_metric_records_are_fully_labelled(calib):
    keys = {"metric", "value", "unit", "target", "split", "n", "model"}
    assert calib.metrics and all(set(m) == keys for m in calib.metrics)
    assert {m["target"] for m in calib.metrics} == {"observed_actual", "reconstructed_clean_actual"}
    assert {m["split"] for m in calib.metrics} == {"all", "train", "test", "forward_test"}
    assert all(np.isfinite(m["value"]) and isinstance(m["n"], int) and m["n"] > 0 and m["unit"] for m in calib.metrics)
    by_target = {t: {m["model"] for m in calib.metrics if m["target"] == t} for t in ("observed_actual", "reconstructed_clean_actual")}
    assert by_target["observed_actual"] == {"atlite_raw", "inference_only"}
    assert by_target["reconstructed_clean_actual"] == {"clean_residual_before", "linear", "quasi_linear_spline", "frontier_gate"}


def test_metric_values_match_direct_computation(calib):
    t = calib.table
    get = lambda metric, target, split, model: next(m["value"] for m in calib.metrics if (m["metric"], m["target"], m["split"], m["model"]) == (metric, target, split, model))  # noqa: E731
    test = t[t["split"] == "test"]
    assert get("rmse", "observed_actual", "test", "atlite_raw") == pytest.approx(np.sqrt(((test["theoretical_wind_mw"] - test["actual_wind_mw"]) ** 2).mean()))
    assert get("mean_residual", "observed_actual", "all", "atlite_raw") == pytest.approx((t["theoretical_wind_mw"] - t["actual_wind_mw"]).mean())
    assert get("rmse", "reconstructed_clean_actual", "test", "quasi_linear_spline") == pytest.approx(np.sqrt((test["res_after_quasi_mw"] ** 2).mean()))
    assert get("curtailed_energy", "reconstructed_clean_actual", "all", "frontier_gate") == pytest.approx(t["estimated_curtailment_mw"].sum() / 1000.0)
    assert get("curtailed_energy", "reconstructed_clean_actual", "train", "frontier_gate") + get("curtailed_energy", "reconstructed_clean_actual", "test", "frontier_gate") \
        == pytest.approx(get("curtailed_energy", "reconstructed_clean_actual", "all", "frontier_gate"))
    r = get("pearson_r", "observed_actual", "all", "atlite_raw")
    assert get("pearson_r2", "observed_actual", "all", "atlite_raw") == pytest.approx(r**2)
    assert get("regression_r2", "observed_actual", "all", "atlite_raw") <= r**2 + 1e-12   # a fixed predictor can't beat the best line
    assert get("rmse", "reconstructed_clean_actual", "train", "linear") <= get("rmse", "reconstructed_clean_actual", "train", "clean_residual_before")
    n = {m["split"]: m["n"] for m in calib.metrics if m["metric"] == "rmse" and m["model"] == "atlite_raw"}
    assert n["train"] + n["test"] == n["all"] == calib.counts["usable_rows"]


def test_reconstructed_actual_never_below_observed(calib):
    t = calib.table
    assert (t["estimated_curtailment_mw"] >= 0).all() and (t["actual_wind_clean_mw"] >= t["actual_wind_mw"]).all()
    assert t["curtail_gate_w"].between(0, 1).all() and t["curtail_gate_w"].max() > 0     # the synthetic curtailment is detected
    assert t["p_base_uncurtailed_mw"].between(0, 1.15 * t["theoretical_wind_mw"] + 1e-9).all()


# --- forward check -------------------------------------------------------------------------------------------------
def test_forward_check_skipped_below_60_days(prepared):
    short = prepared.iloc[: 24 * 30]
    fwd = K.forward_check(short)
    assert fwd["status"] == "skipped" and re.search(r"coverage 30\.\d days < 60 days", fwd["reason"]) and fwd["metrics"] == []


def test_forward_check_threshold_is_inclusive_at_60_days():
    n = 24 * 60
    idx = pd.date_range("2025-07-01", periods=n, freq="h", name="valid_time")
    rng = np.random.default_rng(2)
    f = K.prepare(pd.DataFrame({"theoretical_wind_mw": rng.uniform(50, 150, n), "actual_wind_mw": rng.uniform(40, 120, n),
                                "total_consumption_mw": 400.0, "imbalance_price_eur_mwh": 50.0}, index=idx))[0]
    assert K.coverage_days(f.index) == 60.0
    assert K.forward_check(f)["status"] == "done"
    assert K.forward_check(f.iloc[:-1])["status"] == "skipped"


def test_forward_check_is_chronological_and_labelled(calib):
    fwd = calib.forward
    assert fwd["status"] == "done" and fwd["coverage_days"] >= 60
    assert pd.Timestamp(fwd["train_end_utc"]) < pd.Timestamp(fwd["test_start_utc"])          # no shuffling: train precedes test
    assert fwd["train_blocks"] + fwd["test_blocks"] == calib.counts["n_blocks"] and fwd["train_blocks"] == round(0.7 * calib.counts["n_blocks"])
    fm = [m for m in calib.metrics if m["split"] == "forward_test"]
    assert fm and {m["n"] for m in fm} == {fwd["test_hours"]}
    models = {(m["target"], m["model"]) for m in fm}
    assert ("observed_actual", "inference_only") in models and ("reconstructed_clean_actual", "quasi_linear_spline") in models
    assert ("reconstructed_clean_actual", "linear") in models


def test_run_skips_forward_for_short_coverage(joined, pipeline):
    short = joined.iloc[: 24 * 20]
    res = K.run_calibration(short, pipeline.sim.installed_mw, synthetic=True)
    assert res.forward["status"] == "skipped" and any("forward check skipped" in n for n in res.notes)
    assert not [m for m in res.metrics if m["split"] == "forward_test"]
    assert res.counts["train_hours"] + res.counts["test_hours"] == res.counts["usable_rows"]


# --- reconciliation -----------------------------------------------------------------------------------------------------
def test_reconciliation_synthetic_status(calib):
    r = calib.reconciliation
    assert set(r["status"]) == {"not_reproducible_no_real_data"}
    assert list(r.columns) == ["metric", "documented", "observed", "diff", "unit", "status", "target", "split", "model", "note"]
    row = r[(r["metric"] == "rmse") & (r["split"] == "test") & (r["model"] == "quasi_linear_spline")].iloc[0]
    assert row["documented"] == 323.2 and row["observed"] == pytest.approx(row["documented"] + row["diff"])
    assert row["target"] == "reconstructed_clean_actual"
    pr2 = r[r["metric"] == "pearson_r2"].iloc[0]
    assert pr2["documented"] == 0.8641 and "Pearson" in pr2["note"] and pr2["observed"] is not None
    assert {"frontier_intercept_mw", "frontier_linear_slope", "train_hours", "anchor_train_hours", "n_blocks"} <= set(r["metric"])
    assert r.loc[r["metric"] == "frontier_concave_quad_mw_per_gw2", "documented"].item() == 37.72
    assert r.loc[r["metric"] == "curtailed_energy", "documented"].tolist() == [108.90, 86.13, 22.77]
    assert r.loc[r["metric"] == "train_hours", "observed"].item() == calib.counts["train_hours"]


def test_reconciliation_statuses_for_real_data():
    doc = {(m, t, s, mod): v for m, t, s, mod, v, _ in K.DOCUMENTED}
    metrics = [K.rec(m, v, "MW", t, s, 1, mod) for (m, t, s, mod), v in doc.items()]
    fit = K.FrontierFit(np.array(K.DOCUMENTED_BETA), 735, 1699, True)
    r = K.reconcile(metrics, fit, K.DOCUMENTED_COUNTS, synthetic=False)
    assert set(r["status"]) == {"match"} and (r["diff"].abs() < 1e-9).all()
    off = [dict(m, value=m["value"] * 1.2) if m["metric"] == "rmse" else m for m in metrics]
    r2 = K.reconcile(off, fit, K.DOCUMENTED_COUNTS, synthetic=False)
    assert (r2.loc[r2["metric"] == "rmse", "status"] == "differs").all() and (r2.loc[r2["metric"] == "mae", "status"] == "match").all()
    r3 = K.reconcile([], None, {}, synthetic=False)
    assert set(r3["status"]) == {"unavailable"} and r3["observed"].isna().all()


def test_counts_and_notes(calib, joined):
    c = calib.counts
    assert c["input_rows"] == len(joined) and c["dropped_missing_inputs"] == 3
    assert c["train_hours"] + c["test_hours"] == c["usable_rows"] == c["comparable_hours"]
    assert c["train_blocks"] + c["test_blocks"] == c["n_blocks"] == 24
    assert c["anchor_train_hours"] == calib.models.frontier.n_anchors > 0
    assert calib.models.frontier.within_bounds and calib.synthetic is True and calib.notes == []
