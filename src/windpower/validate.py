"""Hard checks and the reconciliation table.

Each check returns a CheckResult (pass / fail / skipped + detail). Hard failures make the pipeline
exit non-zero. Documented notebook values are *reconciliation targets only*: they are compared and
reported, never forced, and never decide pass/fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import calibrate as K
from . import config as C
from . import simulate as S
from .io_utils import UNITS

PASS, FAIL, SKIPPED = "pass", "fail", "skipped"
ALLOWED_TARGETS = (K.OBS, K.REC)
ALLOWED_SPLITS = ("all", "train", "test", "forward_test")
MIN_JOIN_COVERAGE = 0.95
CAPACITY_REL_TOL = 1e-6


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status == FAIL

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail, "data": self.data}


def _result(name: str, problems: Sequence[str], ok_detail: str, data: "dict[str, Any] | None" = None) -> CheckResult:
    return CheckResult(name, FAIL if problems else PASS, "; ".join(problems) if problems else ok_detail, data or {})


def failed_checks(results: Sequence[CheckResult]) -> list[CheckResult]:
    return [r for r in results if r.failed]


# --- 1. hourly UTC index -----------------------------------------------------------------
def check_hourly_index(index: pd.DatetimeIndex, label: str, expected_hours: "int | None" = None, regular: bool = True) -> CheckResult:
    """Unique, increasing, UTC (tz-naive or UTC), hour-aligned; strictly hourly + expected count if `regular`."""
    problems: list[str] = []
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None and str(idx.tz) != "UTC":
        problems.append(f"timezone {idx.tz} is not UTC")
    if idx.has_duplicates:
        problems.append(f"{int(idx.duplicated().sum())} duplicate timestamp(s), first {idx[idx.duplicated()][0]}")
    if not idx.is_monotonic_increasing:
        problems.append("index is not monotonic increasing")
    if len(idx) and (idx != idx.floor("h")).any():
        problems.append("timestamps not aligned to the hour")
    if regular and len(idx) > 1 and not problems:
        naive = idx.tz_convert(None) if idx.tz is not None else idx
        gaps = np.flatnonzero(np.diff(naive.to_numpy()) != np.timedelta64(1, "h"))
        if gaps.size:
            problems.append(f"not strictly hourly: {idx[gaps[0]]} -> {idx[gaps[0] + 1]} ({gaps.size} irregular step(s))")
    if expected_hours is not None and len(idx) != expected_hours:
        problems.append(f"{len(idx)} hours, expected {expected_hours}")
    return _result(f"hourly_index:{label}", problems, f"{len(idx)} unique {'strictly hourly ' if regular else ''}UTC hours",
                   {"n": len(idx), "expected": expected_hours})


# --- 2. join coverage ----------------------------------------------------------------------
def check_join_coverage(counts: Mapping[str, Any], min_coverage: float = MIN_JOIN_COVERAGE) -> CheckResult:
    cov = counts["comparable_hours"] / counts["era5_hours"] if counts["era5_hours"] else 0.0
    problems = [] if cov >= min_coverage else [f"coverage {cov:.3f} < required {min_coverage:.3f}"]
    return _result("join_coverage", problems,
                   f"{counts['comparable_hours']} comparable of {counts['era5_hours']} ERA5 hours ({100 * cov:.2f} %), "
                   f"{counts['dropped_missing_actual']} dropped for missing actual wind",
                   {**counts, "coverage": cov, "threshold": min_coverage})


# --- 3. grid orientation ---------------------------------------------------------------------
def expected_capacity_matrix(turbines: pd.DataFrame, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Capacity per cell from first principles: nearest centre, and strictly inside the 0.25 deg box."""
    half = C.GRID_RES_DEG / 2
    lat, lon = turbines["lat"].to_numpy(float), turbines["lon"].to_numpy(float)
    i = np.abs(lats[:, None] - lat[None, :]).argmin(axis=0)
    j = np.abs(lons[:, None] - lon[None, :]).argmin(axis=0)
    inside = (np.abs(lat - lats[i]) < half) & (np.abs(lon - lons[j]) < half)
    out = np.zeros((lats.size, lons.size))
    np.add.at(out, (i[inside], j[inside]), turbines["capacity_mw"].to_numpy(float)[inside])
    return out


def check_grid_orientation(capacity_matrix: np.ndarray, turbines: pd.DataFrame, lats: Sequence[float], lons: Sequence[float]) -> CheckResult:
    """capacity_matrix[i, j] must be the capacity inside the cell centred at (lats[i], lons[j]); axes ascending."""
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    problems: list[str] = []
    if not np.all(np.diff(lats) > 0):
        problems.append("latitude axis is not strictly ascending")
    if not np.all(np.diff(lons) > 0):
        problems.append("longitude axis is not strictly ascending")
    if capacity_matrix.shape != (lats.size, lons.size):
        problems.append(f"matrix shape {capacity_matrix.shape} != (n_lat, n_lon) = {(lats.size, lons.size)}")
    if not problems:
        exp = expected_capacity_matrix(turbines, lats, lons)
        bad = ~np.isclose(capacity_matrix, exp, rtol=1e-9, atol=1e-9)
        if bad.any():
            i, j = np.argwhere(bad)[0]
            problems.append(f"{int(bad.sum())} cell(s) differ from the turbines inside them; first at [{i},{j}] "
                            f"(lat {lats[i]}, lon {lons[j]}): matrix {capacity_matrix[i, j]:.4f} vs expected {exp[i, j]:.4f}")
        assigned = turbines.dropna(subset=["pixel_i", "pixel_j"])
        if len(assigned):
            pi, pj = assigned["pixel_i"].astype(int).to_numpy(), assigned["pixel_j"].astype(int).to_numpy()
            if not (np.allclose(lats[pi], assigned["era5_lat"].to_numpy(float)) and np.allclose(lons[pj], assigned["era5_lon"].to_numpy(float))):
                problems.append("pixel_i/pixel_j do not index the era5_lat/era5_lon columns (axes swapped or flipped)")
    return _result("grid_orientation", problems, f"matrix[{len(lats)},{len(lons)}] = cell (lat[i], lon[j]); axes ascending")


# --- 4. capacity totals -------------------------------------------------------------------------
def check_capacity_totals(totals: Mapping[str, float], grid_summary_mw: float, specs_mw: float, archetype_mw: float,
                          n_archetypes: int, rel_tol: float = CAPACITY_REL_TOL) -> CheckResult:
    """turbine table == pixel aggregation + unassigned == matrix + unassigned == grid summary + unassigned == spec table;
    the archetype table (rounded to 0.1 MW per row) must agree within its rounding."""
    t = totals["turbine_table_mw"]
    un = totals["unassigned_mw"]
    pairs = {"pixel_agg+unassigned": totals["pixel_agg_mw"] + un, "matrix+unassigned": totals["matrix_mw"] + un,
             "grid_summary+unassigned": grid_summary_mw + un, "spec_table": specs_mw}
    problems = [f"{k} = {v:.6f} MW != turbine table {t:.6f} MW" for k, v in pairs.items() if abs(v - t) > rel_tol * max(abs(t), 1.0)]
    if abs(archetype_mw - specs_mw) > 0.05 * max(n_archetypes, 1) + rel_tol * abs(specs_mw):
        problems.append(f"archetype table {archetype_mw:.3f} MW != spec table {specs_mw:.3f} MW")
    return _result("capacity_totals", problems,
                   f"{t:.2f} MW in the turbine table = pixel agg + {un:.2f} MW unassigned ({int(totals['unassigned_turbines'])} turbine(s)) "
                   f"= matrix + unassigned; {int(totals['active_cells'])} active cells",
                   {**{k: float(v) for k, v in totals.items()}, "grid_summary_mw": grid_summary_mw, "specs_mw": specs_mw,
                    "archetype_mw": archetype_mw})


# --- 5. split isolation ---------------------------------------------------------------------------
def model_coefficients(m: K.ModelSet) -> list[np.ndarray]:
    return [m.frontier.beta, m.linear.coef_, np.atleast_1d(m.linear.intercept_), m.spline[-1].coef_, np.atleast_1d(m.spline[-1].intercept_),
            m.inference.pipeline[-1].coef_, np.atleast_1d(m.inference.pipeline[-1].intercept_)]


FitFn = Callable[[pd.DataFrame, pd.Index, "float | None"], K.ModelSet]


def check_split_isolation(table: pd.DataFrame, models: "K.ModelSet | None" = None, fit_fn: FitFn = K.fit_models,
                          clip_max_mw: "float | None" = None, block_hours: int = K.BLOCK_HOURS, seed: int = 0) -> CheckResult:
    """Disjoint sets, whole contiguous blocks, anchors from train only, and fits unchanged under test-label perturbation."""
    problems: list[str] = []
    if table.index.has_duplicates:
        problems.append("duplicate timestamps (an hour could be in both sets)")
    if not set(table["split"]) <= {"train", "test"}:
        problems.append(f"unexpected split labels {sorted(set(table['split']))}")
    straddle = table.groupby("block_id")["split"].nunique()
    if (straddle > 1).any():
        problems.append(f"{int((straddle > 1).sum())} block(s) are split between train and test (first block {int(straddle[straddle > 1].index[0])})")
    expected_id = ((table.index - table.index.min()) // pd.Timedelta(hours=block_hours)).astype(int)
    if not (table["block_id"].to_numpy() == expected_id.to_numpy()).all():
        problems.append(f"block_id is not the contiguous {block_hours} h block of the timestamp")
    if not set(table["split"]) >= {"train", "test"}:
        return CheckResult("split_isolation", FAIL if problems else SKIPPED, "; ".join(problems) or "only one split present; leakage test skipped")
    train_idx = table.index[table["split"] == "train"]
    train = table.loc[train_idx]
    n_anchor_train, n_anchor_all = int(K.anchor_mask(train).sum()), int(K.anchor_mask(table).sum())
    if models is not None and models.frontier.n_anchors != n_anchor_train:
        problems.append(f"frontier used {models.frontier.n_anchors} anchors, train has {n_anchor_train} (all rows: {n_anchor_all}) - anchors leak")
    # leakage perturbation: change every label/input of the test rows, refit, compare
    rng = np.random.default_rng(seed)
    bad = table.copy()
    test = bad["split"] == "test"
    n = int(test.sum())
    bad.loc[test, "actual_wind_mw"] = rng.uniform(0, 5.0 * max(table["actual_wind_mw"].max(), 1.0), n)
    bad.loc[test, "imbalance_price_eur_mwh"] = rng.uniform(-100, 100, n)
    bad.loc[test, "total_consumption_mw"] = rng.uniform(1, 10.0 * max(table["total_consumption_mw"].max(), 1.0), n)
    bad.loc[test, "theoretical_wind_mw"] = rng.uniform(0, 5.0 * max(table["theoretical_wind_mw"].max(), 1.0), n)
    bad["wind_to_cons_ratio"] = bad["theoretical_wind_mw"] / np.maximum(bad["total_consumption_mw"], 1.0)
    try:
        ref, pert = fit_fn(table, train_idx, clip_max_mw), fit_fn(bad, train_idx, clip_max_mw)
        same = all(np.array_equal(a, b) for a, b in zip(model_coefficients(ref), model_coefficients(pert)))
        if not same:
            problems.append("fitted parameters changed when test-row labels were perturbed (leakage)")
        if models is not None and not all(np.array_equal(a, b) for a, b in zip(model_coefficients(ref), model_coefficients(models))):
            problems.append("stored models do not equal a refit on the train index")
    except K.CalibrationError as exc:
        problems.append(f"refit failed: {exc}")
    return _result("split_isolation", problems,
                   f"{len(train_idx)} train / {n} test hours in {table['block_id'].nunique()} whole {block_hours} h blocks; "
                   f"{n_anchor_train} anchors (train) of {n_anchor_all}; fits identical under test-label perturbation",
                   {"train_hours": len(train_idx), "test_hours": n, "anchors_train": n_anchor_train, "anchors_all_rows": n_anchor_all})


# --- 6. units -------------------------------------------------------------------------------------------
def check_units(sim: S.SimulationResult, joined: pd.DataFrame, calib_table: "pd.DataFrame | None" = None,
                daily: "pd.DataFrame | None" = None) -> CheckResult:
    """CF = MW / installed MW in [0, 1]; no negative power; daily GWh = sum(MW x 1 h) / 1000; MWh = MW x 1 h."""
    problems: list[str] = []
    theo = sim.theoretical_mw
    cf = S.capacity_factor(theo, sim.installed_mw)
    if cf.min() < -1e-9 or cf.max() > 1.0 + 1e-6:
        problems.append(f"theoretical capacity factor outside [0, 1]: [{cf.min():.4f}, {cf.max():.4f}]")
    neg = {c: int((sim.hourly[c] < -1e-9).sum()) for c in sim.hourly.columns if (sim.hourly[c] < -1e-9).any()}
    if neg:
        problems.append(f"negative simulated power in {neg}")
    for col in ("actual_wind_mw", "solar_power_mw", "total_consumption_mw"):
        if col in joined and (joined[col] < -1e-6).any():
            problems.append(f"negative {col}: {int((joined[col] < -1e-6).sum())} hour(s)")
    if calib_table is not None:
        for col in ("pred_inference_mw", "p_base_uncurtailed_mw", "actual_wind_clean_mw"):
            if col in calib_table and (calib_table[col] < -1e-9).any():
                problems.append(f"negative {col}")
    if daily is not None:
        exp = joined["theoretical_wind_mw"].groupby(joined.index.floor("D")).sum() / 1000.0
        got = daily["theoretical_gwh"]
        if len(exp) != len(got) or not np.allclose(exp.to_numpy(), got.to_numpy(), rtol=1e-9, atol=1e-9):
            problems.append("daily theoretical_gwh != sum(MW x 1 h)/1000")
        mwh = float(S.energy_mwh(joined["theoretical_wind_mw"]).sum())
        if not np.isclose(mwh, float(got.sum()) * 1000.0, rtol=1e-9):
            problems.append(f"total MWh {mwh:.3f} != daily GWh x 1000 {float(got.sum()) * 1000.0:.3f}")
    unregistered = [c for c in joined.columns if c in ("actual_wind_mw", "theoretical_wind_mw") and c not in UNITS]
    if unregistered:
        problems.append(f"no unit registered for {unregistered}")
    return _result("units", problems, f"MW/MWh/CF consistent (installed {sim.installed_mw:.2f} MW, CF in [{cf.min():.3f}, {cf.max():.3f}], no negative power)",
                   {"installed_mw": sim.installed_mw, "cf_min": float(cf.min()), "cf_max": float(cf.max())})


def check_frontier_bounds(frontier: K.FrontierFit) -> CheckResult:
    problems = [] if frontier.within_bounds else [f"beta {np.round(frontier.beta, 4).tolist()} outside bounds"]
    return _result("frontier_bounds", problems, f"beta within bounds {K.BETA_LB.tolist()}..{K.BETA_UB.tolist()}", {"beta": frontier.beta.tolist()})


# --- 7. metrics table -------------------------------------------------------------------------------------
def check_metrics_labels(metrics: Sequence[Mapping[str, Any]]) -> CheckResult:
    keys = {"metric", "value", "unit", "target", "split", "n", "model"}
    problems: list[str] = []
    if not metrics:
        return CheckResult("metrics_labels", FAIL, "no metrics")
    for m in metrics:
        if set(m) != keys:
            problems.append(f"record keys {sorted(set(m) ^ keys)} differ from schema")
            break
        if m["target"] not in ALLOWED_TARGETS or m["split"] not in ALLOWED_SPLITS:
            problems.append(f"bad label target={m['target']!r} split={m['split']!r}")
            break
        if not m["unit"] or not isinstance(m["n"], (int, np.integer)) or m["n"] <= 0 or not np.isfinite(m["value"]):
            problems.append(f"bad value/unit/n in {m}")
            break
        if m["model"] == "inference_only" and m["target"] != K.OBS:
            problems.append("inference_only metric attached to a non-observed target")
            break
    targets = {m["target"] for m in metrics if isinstance(m, Mapping) and "target" in m}
    if not problems and targets != set(ALLOWED_TARGETS):
        problems.append(f"targets present {sorted(targets)}, expected both {list(ALLOWED_TARGETS)}")
    scopes = {(m["split"], m["model"]) for m in metrics if not problems and m["metric"] == "pearson_r2"}
    r2 = {(m["split"], m["model"]) for m in metrics if not problems and m["metric"] == "regression_r2"}
    if not problems and not scopes <= r2:
        problems.append("pearson_r2 reported without regression_r2 for the same scope")
    return _result("metrics_labels", problems, f"{len(metrics)} records, each with target/split/model/unit/n")


def metrics_frame(metrics: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    cols = ["target", "split", "model", "metric", "value", "unit", "n"]
    return pd.DataFrame(list(metrics), columns=cols).sort_values(cols[:4], kind="stable").reset_index(drop=True)


def check_forward(forward: Mapping[str, Any]) -> CheckResult:
    if forward["status"] == "skipped":
        return CheckResult("forward_check", SKIPPED, forward["reason"])
    problems = []
    if not pd.Timestamp(forward["train_end_utc"]) < pd.Timestamp(forward["test_start_utc"]):
        problems.append("forward-test starts before the training period ends")
    return _result("forward_check", problems, f"train to {forward['train_end_utc']}, test from {forward['test_start_utc']} ({forward['coverage_days']:.0f} days)")


# --- reconciliation ---------------------------------------------------------------------------------------------
RECON_COLUMNS = ["metric", "documented", "observed", "diff", "unit", "status", "target", "split", "model", "note"]


def _status(documented: float, observed: "float | None", synthetic: bool, exact: bool) -> str:
    if synthetic:
        return "not_reproducible_no_real_data"
    if observed is None:
        return "unavailable"
    tol = 0.0 if exact else max(K.RECON_REL_TOL * abs(documented), 0.01)
    return "match" if abs(observed - documented) <= tol + 1e-9 else "differs"


def data_reconciliation(observed: Mapping[str, "float | None"], synthetic: bool) -> pd.DataFrame:
    """Rows for the ingest/fleet targets in config.TARGETS (comparable_hours etc. come from calibrate)."""
    spec = [("era5_hours", "count", True, ""), ("turbines", "count", True, ""), ("osm_kept", "count", True, ""),
            ("arcgis_kept", "count", True, ""), ("capacity_mw", "MW", False, "calibrated active capacity"),
            ("active_cells", "count", True, "ERA5 cells with turbines (within join)"), ("year_coverage_pct", "%", False, ""),
            ("mean_speed_100", "m s-1", False, "mean over the window and grid"), ("mean_air_density", "kg m-3", False, ""),
            ("icing_share_pct", "%", False, "share of space-time cells flagged")]
    rows = []
    for name, unit, exact, note in spec:
        doc, obs = float(C.TARGETS[name]), observed.get(name)
        rows.append({"metric": name, "documented": doc, "observed": None if obs is None else float(obs),
                     "diff": None if obs is None else float(obs) - doc, "unit": unit, "status": _status(doc, obs, synthetic, exact),
                     "target": "n/a", "split": "n/a", "model": "n/a", "note": note})
    return pd.DataFrame(rows, columns=RECON_COLUMNS)


def full_reconciliation(observed: Mapping[str, "float | None"], calib: K.CalibrationResult, synthetic: bool) -> pd.DataFrame:
    return pd.concat([data_reconciliation(observed, synthetic), calib.reconciliation[RECON_COLUMNS]], ignore_index=True)


# --- orchestration ------------------------------------------------------------------------------------------------
@dataclass
class Evidence:
    era5_index: pd.DatetimeIndex
    expected_hours: int
    sim: S.SimulationResult
    joined: pd.DataFrame
    join_counts: Mapping[str, Any]
    capacity_matrix: np.ndarray
    turbines: pd.DataFrame
    lats: np.ndarray
    lons: np.ndarray
    totals: Mapping[str, float]
    grid_summary_mw: float
    specs_mw: float
    calib: K.CalibrationResult
    daily: pd.DataFrame
    min_join_coverage: float = MIN_JOIN_COVERAGE


def run_all_checks(ev: Evidence) -> list[CheckResult]:
    arch_mw = float(ev.sim.archetypes["installed_mw"].sum())
    return [
        check_hourly_index(ev.era5_index, "era5", ev.expected_hours),
        check_hourly_index(ev.sim.hourly.index, "theoretical", ev.expected_hours),
        check_hourly_index(ev.joined.index, "joined", ev.expected_hours),
        check_hourly_index(ev.calib.table.index, "calibration_rows", None, regular=False),
        check_join_coverage(ev.join_counts, ev.min_join_coverage),
        check_grid_orientation(ev.capacity_matrix, ev.turbines, ev.lats, ev.lons),
        check_capacity_totals(ev.totals, ev.grid_summary_mw, ev.specs_mw, arch_mw, len(ev.sim.archetypes)),
        check_split_isolation(ev.calib.table, ev.calib.models, clip_max_mw=ev.sim.installed_mw),
        check_units(ev.sim, ev.joined, ev.calib.table, ev.daily),
        check_frontier_bounds(ev.calib.models.frontier),
        check_metrics_labels(ev.calib.metrics),
        check_forward(ev.calib.forward),
    ]


def results_frame(results: Sequence[CheckResult]) -> pd.DataFrame:
    return pd.DataFrame([{"name": r.name, "status": r.status, "detail": r.detail} for r in results])
