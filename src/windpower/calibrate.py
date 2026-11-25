"""Notebook-2 calibration: block split, uncurtailed frontier, curtailment reconstruction, de-biasing.

Two kinds of results are kept strictly apart and labelled in every metric record:
  * target="reconstructed_clean_actual" - retrospective benchmark: Fingrid actual plus an estimated
    curtailment (needs price and consumption, so it is NOT a forecast error);
  * target="observed_actual" - error against what Fingrid actually reported; the website-facing
    `InferenceOnlyModel` lives here and sees only theoretical wind MW and hour-of-day harmonics.

Every fit function takes the full table plus the *train index* and touches only those rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import SplineTransformer

SPLIT_SEED = 2026
BLOCK_HOURS = 72
TEST_FRACTION = 0.20
MAX_QUANTILE_BINS = 4

ANCHOR_MIN_PRICE_EUR = 20.0
ANCHOR_MAX_WIND_TO_CONS = 0.40
ANCHOR_MAX_THEORETICAL_MW = 3800.0
MIN_ANCHORS = 10

BETA_NAMES = ("intercept_mw", "linear_slope", "concave_quad_mw_per_gw2", "sin_hour_mw", "cos_hour_mw")
BETA_X0 = np.array([0.0, 0.92, 10.0, 0.0, 0.0])
BETA_LB = np.array([-500.0, 0.50, 0.0, -500.0, -500.0])
BETA_UB = np.array([500.0, 1.30, 80.0, 500.0, 500.0])
FRONTIER_F_SCALE = 150.0
P_BASE_MAX_FACTOR = 1.15
PRICE_GATE_EUR = (0.0, 15.0)        # score 1 at <= 0, 0 at >= 15 EUR/MWh
GRID_GATE_RATIO = (0.40, 0.70)      # score 0 below 40 %, 1 above 70 % wind-to-consumption

FORWARD_MIN_DAYS = 60
FORWARD_TRAIN_FRACTION = 0.70
SPLINE_KNOTS, SPLINE_DEGREE, RIDGE_ALPHA = 4, 2, 10.0

REQUIRED_COLUMNS = ("theoretical_wind_mw", "actual_wind_mw", "total_consumption_mw", "imbalance_price_eur_mwh")
INFERENCE_FEATURES = ("theoretical_wind_mw", "hour_sin", "hour_cos")


class CalibrationError(ValueError):
    pass


class LeakageError(CalibrationError):
    """A fit was asked to use rows that are not part of the training index."""


# --- table preparation ----------------------------------------------------------
def hour_harmonics(index: pd.DatetimeIndex) -> "tuple[np.ndarray, np.ndarray]":
    h = np.asarray(index.hour)
    return np.sin(2.0 * np.pi * h / 24.0), np.cos(2.0 * np.pi * h / 24.0)


def prepare(df: pd.DataFrame) -> "tuple[pd.DataFrame, dict[str, int]]":
    """Keep rows with all required inputs, sorted by unique UTC hour; add harmonics, ratio, raw residual."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise CalibrationError(f"missing required column(s): {missing}")
    out = df.sort_index()
    if out.index.has_duplicates:
        raise CalibrationError("duplicate timestamps in calibration table")
    n_in = len(out)
    out = out.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    out["hour_sin"], out["hour_cos"] = hour_harmonics(out.index)
    out["wind_to_cons_ratio"] = out["theoretical_wind_mw"] / np.maximum(out["total_consumption_mw"], 1.0)
    out["residual_raw_mw"] = out["theoretical_wind_mw"] - out["actual_wind_mw"]
    return out, {"input_rows": n_in, "usable_rows": len(out), "dropped_missing_inputs": n_in - len(out)}


# --- block split ------------------------------------------------------------------
def assign_blocks(df: pd.DataFrame, seed: int = SPLIT_SEED, block_hours: int = BLOCK_HOURS,
                  test_fraction: float = TEST_FRACTION) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Whole `block_hours` blocks, stratified by quantile of block-mean theoretical wind, shuffled with
    default_rng(seed) group by group (notebook order). Uses only the theoretical wind (a model input).

    Returns (df with block_id/split columns, block table).
    """
    out = df.copy()
    out["block_id"] = ((out.index - out.index.min()) // pd.Timedelta(hours=block_hours)).astype(int)
    stats = out.groupby("block_id")["theoretical_wind_mw"].mean().reset_index()
    n_bins = min(MAX_QUANTILE_BINS, max(1, len(stats) // 2))
    stats["wind_quartile"] = (pd.qcut(stats["theoretical_wind_mw"], q=n_bins, labels=False, duplicates="drop")
                              if n_bins > 1 else 0)
    rng = np.random.default_rng(seed)
    test_blocks: list[int] = []
    for _, grp in stats.groupby("wind_quartile"):
        blocks = grp["block_id"].to_numpy().copy()
        rng.shuffle(blocks)
        n_test = max(1, int(round(test_fraction * len(blocks)))) if len(blocks) >= 2 else 0
        test_blocks.extend(blocks[:n_test].tolist())
    if not test_blocks and len(stats) >= 2:
        test_blocks = [int(stats["block_id"].iloc[-1])]
    out["split"] = np.where(out["block_id"].isin(test_blocks), "test", "train")
    stats["split"] = np.where(stats["block_id"].isin(test_blocks), "test", "train")
    return out, stats


def _train_rows(df: pd.DataFrame, train_index: pd.Index) -> pd.DataFrame:
    if not train_index.isin(df.index).all():
        raise CalibrationError("train_index contains labels that are not in the table")
    sub = df.loc[train_index]
    if "split" in sub and (sub["split"] != "train").any():
        raise LeakageError(f"{int((sub['split'] != 'train').sum())} non-train row(s) passed to a fitting function")
    return sub


# --- frontier -----------------------------------------------------------------------
def _design(sub: pd.DataFrame) -> np.ndarray:
    p_mw = sub["theoretical_wind_mw"].to_numpy(float)
    return np.column_stack([np.ones(len(sub)), p_mw, -((p_mw / 1000.0) ** 2),
                            sub["hour_sin"].to_numpy(float), sub["hour_cos"].to_numpy(float)])


@dataclass
class FrontierFit:
    beta: np.ndarray
    n_anchors: int
    n_train_rows: int
    success: bool

    def as_dict(self) -> dict[str, float]:
        return {n: float(b) for n, b in zip(BETA_NAMES, self.beta)}

    @property
    def within_bounds(self) -> bool:
        return bool(np.all(self.beta >= BETA_LB - 1e-9) and np.all(self.beta <= BETA_UB + 1e-9))


def anchor_mask(sub: pd.DataFrame) -> pd.Series:
    """Hours believed uncurtailed: price > 20 EUR/MWh, wind/consumption < 0.40, theoretical < 3,800 MW."""
    return ((sub["imbalance_price_eur_mwh"] > ANCHOR_MIN_PRICE_EUR)
            & (sub["wind_to_cons_ratio"] < ANCHOR_MAX_WIND_TO_CONS)
            & (sub["theoretical_wind_mw"] < ANCHOR_MAX_THEORETICAL_MW))


def fit_frontier(df: pd.DataFrame, train_index: pd.Index) -> FrontierFit:
    """Robust (soft-L1) bounded fit of actual MW on TRAIN anchors only."""
    train = _train_rows(df, train_index)
    anchors = train[anchor_mask(train)]
    if len(anchors) < MIN_ANCHORS:
        raise CalibrationError(f"only {len(anchors)} train anchor hours (< {MIN_ANCHORS}); frontier not identifiable")
    x, y = _design(anchors), anchors["actual_wind_mw"].to_numpy(float)
    res = least_squares(lambda b: x @ b - y, BETA_X0, bounds=(BETA_LB, BETA_UB), loss="soft_l1", f_scale=FRONTIER_F_SCALE)
    return FrontierFit(res.x, len(anchors), len(train), bool(res.success))


def reconstruct(df: pd.DataFrame, beta: np.ndarray) -> pd.DataFrame:
    """Baseline, soft curtailment gate and reconstructed clean actual (row-wise; uses beta only)."""
    out = df.copy()
    theo = out["theoretical_wind_mw"].to_numpy(float)
    out["p_base_uncurtailed_mw"] = np.clip(_design(out) @ beta, 0.0, theo * P_BASE_MAX_FACTOR)
    lo, hi = PRICE_GATE_EUR
    out["score_price"] = np.clip((hi - out["imbalance_price_eur_mwh"]) / (hi - lo), 0.0, 1.0)
    g0, g1 = GRID_GATE_RATIO
    out["score_grid"] = np.clip((out["wind_to_cons_ratio"] - g0) / (g1 - g0), 0.0, 1.0)
    out["curtail_gate_w"] = 1.0 - (1.0 - out["score_price"]) * (1.0 - out["score_grid"])
    out["excess_shortfall_mw"] = np.maximum(0.0, out["p_base_uncurtailed_mw"] - out["actual_wind_mw"])
    out["estimated_curtailment_mw"] = out["curtail_gate_w"] * out["excess_shortfall_mw"]
    out["actual_wind_clean_mw"] = out["actual_wind_mw"] + out["estimated_curtailment_mw"]
    out["residual_clean_mw"] = out["theoretical_wind_mw"] - out["actual_wind_clean_mw"]
    return out


# --- de-biasing models -----------------------------------------------------------------
class InferenceOnlyModel:
    """Website-facing candidate: observed actual MW from theoretical MW and UTC hour-of-day only.

    Spline(theoretical) + hour harmonics -> Ridge. No price, consumption or Fingrid value is used at
    inference. Predictions are clipped to [0, clip_max_mw].
    """

    features = INFERENCE_FEATURES

    def __init__(self, clip_max_mw: "float | None" = None):
        self.clip_max_mw = clip_max_mw
        self.pipeline: Pipeline = Pipeline([
            ("features", ColumnTransformer([
                ("spline", SplineTransformer(n_knots=SPLINE_KNOTS, degree=SPLINE_DEGREE, include_bias=False), ["theoretical_wind_mw"]),
                ("hour", "passthrough", ["hour_sin", "hour_cos"]),
            ])),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ])

    def _frame(self, theoretical_mw: Any, hour: Any) -> pd.DataFrame:
        h = np.asarray(hour, dtype=float)
        return pd.DataFrame({"theoretical_wind_mw": np.asarray(theoretical_mw, dtype=float),
                             "hour_sin": np.sin(2 * np.pi * h / 24.0), "hour_cos": np.cos(2 * np.pi * h / 24.0)})

    def fit(self, df: pd.DataFrame, train_index: pd.Index) -> "InferenceOnlyModel":
        train = _train_rows(df, train_index)
        self.pipeline.fit(self._frame(train["theoretical_wind_mw"], train.index.hour), train["actual_wind_mw"].to_numpy(float))
        return self

    def predict(self, theoretical_mw: Any, hour: Any) -> np.ndarray:
        pred = self.pipeline.predict(self._frame(theoretical_mw, hour))
        return np.clip(pred, 0.0, self.clip_max_mw)


@dataclass
class ModelSet:
    frontier: FrontierFit
    linear: LinearRegression
    spline: Pipeline
    inference: InferenceOnlyModel


def fit_models(df: pd.DataFrame, train_index: pd.Index, clip_max_mw: "float | None" = None) -> ModelSet:
    """Fit everything on the train index only; `df` rows outside it are never read for fitting."""
    frontier = fit_frontier(df, train_index)
    train = reconstruct(_train_rows(df, train_index), frontier.beta)
    x, y = train[["theoretical_wind_mw"]], train["residual_clean_mw"].to_numpy(float)
    linear = LinearRegression().fit(x, y)
    spline = make_pipeline(SplineTransformer(n_knots=SPLINE_KNOTS, degree=SPLINE_DEGREE, include_bias=False),
                           Ridge(alpha=RIDGE_ALPHA)).fit(x, y)
    return ModelSet(frontier, linear, spline, InferenceOnlyModel(clip_max_mw).fit(df, train_index))


def apply_models(df: pd.DataFrame, models: ModelSet) -> pd.DataFrame:
    out = reconstruct(df, models.frontier.beta)
    x = out[["theoretical_wind_mw"]]
    out["bias_linear_mw"] = models.linear.predict(x)
    out["res_after_linear_mw"] = out["residual_clean_mw"] - out["bias_linear_mw"]
    out["bias_quasi_mw"] = models.spline.predict(x)
    out["res_after_quasi_mw"] = out["residual_clean_mw"] - out["bias_quasi_mw"]
    out["pred_inference_mw"] = models.inference.predict(out["theoretical_wind_mw"], out.index.hour)
    return out


# --- metrics ---------------------------------------------------------------------------
def pearson_r(x: Any, y: Any) -> float:
    return float(np.corrcoef(np.asarray(x, float), np.asarray(y, float))[0, 1])


def regression_r2(predictor: Any, observed: Any) -> float:
    """1 - SSE/SST with `predictor` used as-is as the prediction of `observed` (not a fitted line)."""
    p, o = np.asarray(predictor, float), np.asarray(observed, float)
    return float(1.0 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2))


def rec(metric: str, value: float, unit: str, target: str, split: str, n: int, model: str) -> dict[str, Any]:
    return {"metric": metric, "value": float(value), "unit": unit, "target": target, "split": split, "n": int(n), "model": model}


def _error_stats(r: np.ndarray, target: str, split: str, model: str, full: bool) -> list[dict[str, Any]]:
    n = len(r)
    out = [rec("mean_residual", r.mean(), "MW", target, split, n, model),
           rec("mae", np.abs(r).mean(), "MW", target, split, n, model),
           rec("rmse", np.sqrt((r**2).mean()), "MW", target, split, n, model)]
    if full:
        out += [rec("median_residual", np.median(r), "MW", target, split, n, model),
                rec("iqr", np.percentile(r, 75) - np.percentile(r, 25), "MW", target, split, n, model),
                rec("p95_abs_error", np.percentile(np.abs(r), 95), "MW", target, split, n, model)]
    return out


OBS, REC = "observed_actual", "reconstructed_clean_actual"


def compute_metrics(df: pd.DataFrame, splits: "Mapping[str, pd.Series] | None" = None) -> list[dict[str, Any]]:
    """Metric records for every target/split/model. Residual = estimate - reference (positive = over-estimate).

    `df` must have passed `apply_models`. Splits default to all/train/test from the `split` column.
    """
    if splits is None:
        splits = {"all": pd.Series(True, index=df.index), "train": df["split"] == "train", "test": df["split"] == "test"}
    out: list[dict[str, Any]] = []
    for name, mask in splits.items():
        sub = df[mask]
        if sub.empty:
            continue
        n = len(sub)
        theo, act = sub["theoretical_wind_mw"].to_numpy(float), sub["actual_wind_mw"].to_numpy(float)
        # target = observed Fingrid actual
        out += _error_stats(theo - act, OBS, name, "atlite_raw", False)
        out += [rec("mean_theoretical", theo.mean(), "MW", OBS, name, n, "atlite_raw"),
                rec("mean_actual", act.mean(), "MW", OBS, name, n, "atlite_raw")]
        if n > 2:
            r = pearson_r(theo, act)
            out += [rec("pearson_r", r, "1", OBS, name, n, "atlite_raw"),
                    rec("pearson_r2", r**2, "1", OBS, name, n, "atlite_raw"),          # squared correlation
                    rec("regression_r2", regression_r2(theo, act), "1", OBS, name, n, "atlite_raw")]  # 1 - SSE/SST
        pred = sub["pred_inference_mw"].to_numpy(float)
        out += _error_stats(pred - act, OBS, name, "inference_only", False)
        if n > 2:
            out += [rec("pearson_r", pearson_r(pred, act), "1", OBS, name, n, "inference_only"),
                    rec("regression_r2", regression_r2(pred, act), "1", OBS, name, n, "inference_only")]
        # target = reconstructed clean actual (retrospective benchmark)
        for model, col in (("clean_residual_before", "residual_clean_mw"), ("linear", "res_after_linear_mw"),
                           ("quasi_linear_spline", "res_after_quasi_mw")):
            out += _error_stats(sub[col].to_numpy(float), REC, name, model, True)
        out.append(rec("curtailed_energy", sub["estimated_curtailment_mw"].sum() / 1000.0, "GWh", REC, name, n, "frontier_gate"))
        out.append(rec("curtailment_hours_gt_50mw", (sub["estimated_curtailment_mw"] > 50.0).sum(), "h", REC, name, n, "frontier_gate"))
    return out


# --- forward-in-time check -----------------------------------------------------------------
def coverage_days(index: pd.DatetimeIndex) -> float:
    return (index.max() - index.min() + pd.Timedelta(hours=1)) / pd.Timedelta(days=1) if len(index) else 0.0


def forward_check(df: pd.DataFrame, *, min_days: float = FORWARD_MIN_DAYS, train_fraction: float = FORWARD_TRAIN_FRACTION,
                  block_hours: int = BLOCK_HOURS, clip_max_mw: "float | None" = None) -> dict[str, Any]:
    """Chronological check: earliest ~70 % of whole blocks train, the rest test (no shuffling).

    Skipped (with an explicit reason) when coverage is below `min_days`.
    """
    days = coverage_days(df.index)
    if days < min_days:
        return {"status": "skipped", "reason": f"coverage {days:.1f} days < {min_days:g} days required", "coverage_days": days, "metrics": []}
    blocks = ((df.index - df.index.min()) // pd.Timedelta(hours=block_hours)).astype(int)
    ids = np.sort(np.unique(blocks))
    n_train = min(len(ids) - 1, max(1, int(round(train_fraction * len(ids)))))
    cutoff = ids[n_train - 1]
    fwd = df.copy()
    fwd["split"] = np.where(blocks <= cutoff, "train", "test")
    train_idx = fwd.index[fwd["split"] == "train"]
    try:
        models = fit_models(fwd, train_idx, clip_max_mw)
    except CalibrationError as exc:
        return {"status": "skipped", "reason": str(exc), "coverage_days": days, "metrics": []}
    scored = apply_models(fwd, models)
    mask = scored["split"] == "test"
    metrics = compute_metrics(scored, {"forward_test": mask})
    return {"status": "done", "coverage_days": days, "train_blocks": int(n_train), "test_blocks": int(len(ids) - n_train),
            "train_hours": int((~mask).sum()), "test_hours": int(mask.sum()), "frontier": models.frontier.as_dict(),
            "train_end_utc": fwd.index[~mask.to_numpy()].max().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "test_start_utc": fwd.index[mask.to_numpy()].min().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "metrics": metrics}


# --- documented targets and reconciliation ---------------------------------------------------
# (metric, target, split, model) -> documented value from the notebooks/report.
# beta_2 is stored as the fitted parameter (+37.72); the notebook printed its negation (-37.72).
DOCUMENTED: list[tuple[str, str, str, str, float, str]] = [
    ("pearson_r", OBS, "all", "atlite_raw", 0.9296, "1"),
    ("pearson_r2", OBS, "all", "atlite_raw", 0.8641, "1"),   # report calls this R^2; it is Pearson r^2, not regression R^2
    ("mean_theoretical", OBS, "all", "atlite_raw", 2298.7, "MW"),
    ("mean_actual", OBS, "all", "atlite_raw", 1746.4, "MW"),
    ("mean_residual", OBS, "all", "atlite_raw", 552.4, "MW"),
    ("mae", OBS, "all", "atlite_raw", 640.6, "MW"),
    ("rmse", OBS, "all", "atlite_raw", 1095.0, "MW"),
    ("rmse", OBS, "test", "atlite_raw", 1166.7, "MW"),
    ("rmse", REC, "test", "clean_residual_before", 1098.7, "MW"),
    ("rmse", REC, "test", "linear", 356.5, "MW"),
    ("rmse", REC, "test", "quasi_linear_spline", 323.2, "MW"),
    ("curtailed_energy", REC, "all", "frontier_gate", 108.90, "GWh"),
    ("curtailed_energy", REC, "train", "frontier_gate", 86.13, "GWh"),
    ("curtailed_energy", REC, "test", "frontier_gate", 22.77, "GWh"),
]
DOCUMENTED_BETA = (-42.31, 0.8860, 37.72, -74.78, 71.76)
DOCUMENTED_COUNTS = {"comparable_hours": 2202, "train_hours": 1699, "test_hours": 503, "train_blocks": 24,
                     "test_blocks": 7, "n_blocks": 31, "anchor_train_hours": 735}
RECON_REL_TOL = 0.01


def reconcile(metrics: Sequence[Mapping[str, Any]], frontier: "FrontierFit | None", counts: Mapping[str, int],
              synthetic: bool) -> pd.DataFrame:
    """Documented vs observed table; status is `not_reproducible_no_real_data` for synthetic runs."""
    lookup = {(m["metric"], m["target"], m["split"], m["model"]): m["value"] for m in metrics}
    rows: list[dict[str, Any]] = []

    def add(name: str, target: str, split: str, model: str, documented: float, observed: "float | None", unit: str, note: str = ""):
        diff = None if observed is None else observed - documented
        if synthetic:
            status = "not_reproducible_no_real_data"
        elif observed is None:
            status = "unavailable"
        else:
            status = "match" if abs(diff) <= max(RECON_REL_TOL * abs(documented), 0.01) else "differs"
        rows.append({"metric": name, "documented": documented, "observed": observed, "diff": diff, "unit": unit,
                     "status": status, "target": target, "split": split, "model": model, "note": note})

    for metric, target, split, model, doc, unit in DOCUMENTED:
        note = "Pearson r^2 (not regression R^2)" if metric == "pearson_r2" else ""
        add(metric, target, split, model, doc, lookup.get((metric, target, split, model)), unit, note)
    for i, (name, doc) in enumerate(zip(BETA_NAMES, DOCUMENTED_BETA)):
        add(f"frontier_{name}", "observed_actual", "train", "frontier_train_anchors", doc,
            None if frontier is None else float(frontier.beta[i]), "MW" if i != 1 else "1",
            "notebook printed the concave term as -beta_2" if i == 2 else "")
    for name, doc in DOCUMENTED_COUNTS.items():
        add(name, "n/a", "n/a", "n/a", float(doc), None if name not in counts else float(counts[name]), "count")
    return pd.DataFrame(rows)


# --- orchestration ------------------------------------------------------------------------------
@dataclass
class CalibrationResult:
    table: pd.DataFrame
    blocks: pd.DataFrame
    models: ModelSet
    metrics: list[dict[str, Any]]
    reconciliation: pd.DataFrame
    forward: dict[str, Any]
    counts: dict[str, int]
    synthetic: bool
    notes: list[str] = field(default_factory=list)


def run_calibration(df: pd.DataFrame, installed_mw: "float | None" = None, *, synthetic: bool, seed: int = SPLIT_SEED,
                    forward_min_days: float = FORWARD_MIN_DAYS) -> CalibrationResult:
    prepared, counts = prepare(df)
    split_df, blocks = assign_blocks(prepared, seed)
    train_idx = split_df.index[split_df["split"] == "train"]
    models = fit_models(split_df, train_idx, installed_mw)
    table = apply_models(split_df, models)
    metrics = compute_metrics(table)
    forward = forward_check(prepared, min_days=forward_min_days, clip_max_mw=installed_mw)
    metrics += forward["metrics"]
    counts.update(
        train_hours=int((table["split"] == "train").sum()), test_hours=int((table["split"] == "test").sum()),
        n_blocks=int(len(blocks)), train_blocks=int((blocks["split"] == "train").sum()),
        test_blocks=int((blocks["split"] == "test").sum()), anchor_train_hours=models.frontier.n_anchors,
        comparable_hours=counts["usable_rows"],
    )
    notes = []
    if not models.frontier.within_bounds:
        notes.append("frontier beta outside configured bounds")
    if forward["status"] == "skipped":
        notes.append(f"forward check skipped: {forward['reason']}")
    return CalibrationResult(table, blocks, models, metrics, reconcile(metrics, models.frontier, counts, synthetic),
                             forward, counts, synthetic, notes)
