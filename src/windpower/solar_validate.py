"""Solar hard checks, July reconciliation table and the solar manifest (patterns of validate.py / manifest.py).

Checks return `validate.CheckResult` (pass / fail / skipped with a reason); a failing check makes the CLI exit 1. Targets of
the solar report are compared, never forced: every reconciliation row gets one of
`match` (within the stated tolerance) | `differs` | `unavailable` (the exact missing input is named) | `not_reproducible_synthetic`.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import __version__
from . import config as C
from .io_utils import atomic_write_json, utc_now_iso
from .manifest import describe_outputs, git_info, package_versions
from .sarah_regrid import SRC_SATELLITE, SarahOnGrid
from .validate import FAIL, PASS, SKIPPED, CheckResult, check_hourly_index, _result

MANIFEST_NAME = "manifest.json"
MODEL_VERSION = f"windpower-solar-{__version__}"
SOLAR_UNITS: dict[str, str] = {
    "era5_only_mw": "MW", "hybrid_mw": "MW", "era5_only_popw_mw": "MW", "hybrid_popw_mw": "MW", "capacity_267_mw": "MW", "forecast_248_mw": "MW",
    "surplus_362_mw": "MW", "surplus_362_kwh": "kWh", "surplus_362_count": "1", "era5_only_gwh": "GWh", "hybrid_gwh": "GWh", "capacity_factor": "1",
    "cf_era5": "1", "cf_hybrid": "1", "capacity_mwp": "MWp", "energy_mwh": "MWh", "ghi_wm2": "W m-2", "valid_area_fraction": "1",
    "inhabitants": "1", "large_capacity_mw": "MW", "population_share": "1", "satellite_coverage_fraction": "1",
    "mean_cf_era5": "1", "mean_cf_hybrid": "1", "forecast_248_gwh": "GWh", "surplus_362_gwh": "GWh",
}
NEEDS_TEXT = {
    "sarah": "SARAH-3 SIS/SID files (--sarah-sis-dir/--sarah-sid-dir or WINDPOWER_SARAH_SIS_DIR/_SID_DIR)",
    "era5": "ERA5 solar data (CDS credentials via ~/.cdsapirc or --era5-solar-path / WINDPOWER_ERA5_SOLAR)",
    "fingrid_267": "Fingrid dataset 267 (FINGRID_API_KEY)", "fingrid_248": "Fingrid dataset 248 (FINGRID_API_KEY)",
    "fingrid_362": "Fingrid dataset 362 (FINGRID_API_KEY)",
    "public": "public inputs (plant list, population grid, municipalities: run `ingest-solar-public`)",
}


# --- checks -------------------------------------------------------------------------------------------------------------------
@dataclass
class SolarEvidence:
    mode: str
    synthetic: bool
    hours: pd.DatetimeIndex
    sarah_info: "Mapping[str, Any] | None" = None
    ongrid: "SarahOnGrid | None" = None
    era5_info: "Mapping[str, Any] | None" = None
    pv_info: "Mapping[str, Any] | None" = None
    allocation: Any = None                                   # solar_national.Allocation
    plants_report: "Mapping[str, Any] | None" = None
    reconciliation: "Mapping[str, Any] | None" = None        # check_reconciliation() result for the hourly capacity cube
    national: "pd.DataFrame | None" = None
    fingrid_info: "Mapping[str, Any] | None" = None
    sources: Mapping[str, str] = field(default_factory=dict)


def _skip(name: str, why: str) -> CheckResult:
    return CheckResult(name, SKIPPED, why)


def check_sarah_alignment(ev: SolarEvidence) -> CheckResult:
    if not ev.sarah_info:
        return _skip("sarah_alignment", "SARAH-3 not ingested: " + NEEDS_TEXT["sarah"])
    inv = ev.sarah_info["inventory"]
    a, b = inv["SIS"], inv["SID"]
    problems = []
    for key in ("first_utc", "last_utc", "n_steps"):
        if a["time"][key] != b["time"][key]:
            problems.append(f"SIS and SID differ in time {key}: {a['time'][key]} vs {b['time'][key]}")
    if a["grid"]["n_lat"] != b["grid"]["n_lat"] or a["grid"]["n_lon"] != b["grid"]["n_lon"]:
        problems.append("SIS and SID grids differ")
    for k, i in (("SIS", a), ("SID", b)):
        if not i["time"]["unique"] or i["time"]["n_gaps"]:
            problems.append(f"{k}: time axis has {i['time']['n_gaps']} gap(s) / duplicates")
        if i["standard_name"] != i["expected_standard_name"]:
            problems.append(f"{k}: standard_name {i['standard_name']!r} != {i['expected_standard_name']!r}")
    cov = ev.sarah_info["coverage"]
    if cov.get("missing_half_hour_steps"):
        problems.append(f"{cov['missing_half_hour_steps']} half-hour step(s) absent in the window")
    first, last = ev.hours[0], ev.hours[-1] + pd.Timedelta(hours=1)
    if not (pd.Timestamp(a["time"]["first_utc"]).tz_localize(None) <= first and pd.Timestamp(a["time"]["last_utc"]).tz_localize(None) >= last):
        problems.append("files do not cover the window plus the h+1:00 sample of the last hour")
    flagged = [s["time_utc"] for s in a.get("window_flagged_steps", [])]
    return _result("sarah_alignment", problems, f"SIS/SID time axes and grids identical ({a['time']['n_steps']} steps), no absent steps; "
                                                f"flagged steps in window: {flagged or 'none'}", {"flagged_steps": flagged})


def check_era5_units_and_shift(ev: SolarEvidence) -> CheckResult:
    if not ev.era5_info:
        return _skip("era5_units_and_shift", "ERA5 solar not available: " + NEEDS_TEXT["era5"])
    conv, src = ev.era5_info["convention"], ev.era5_info["source_detail"]
    problems = []
    if conv.get("convention_used") not in ("end", "start"):
        problems.append("no time convention recorded")
    if conv.get("detected") == "ambiguous" and not conv.get("override"):
        problems.append("ambiguous convention used without an override")
    if conv.get("override_contradicts_detection"):
        problems.append(f"--era5-convention {conv['override']} contradicts the detected {conv['detected']}")
    if not conv.get("evidence"):
        problems.append("no convention evidence recorded")
    if not src.get("units"):
        problems.append("no unit evidence recorded")
    shifted_once = conv.get("convention_used") == "end" or conv.get("shift_applied", "").startswith("none")
    if not shifted_once:
        problems.append("shift bookkeeping inconsistent")
    return _result("era5_units_and_shift", problems, f"convention {conv.get('convention_used')} (detected {conv.get('detected')}), shift: {conv.get('shift_applied')}",
                   {"convention": conv.get("convention_used"), "detected": conv.get("detected")})


def check_cf(ev: SolarEvidence) -> CheckResult:
    if not ev.pv_info:
        return _skip("cf_night_and_range", "PV not computed (needs ERA5 solar data)")
    problems, parts = [], []
    for mode, s in ev.pv_info["cf_checks"].items():
        if s["min"] < 0 or s["max"] > 1.0 + 1e-9:
            problems.append(f"{mode}: CF outside [0, 1] ({s['min']}, {s['max']})")
        if s.get("night_nonzero"):
            problems.append(f"{mode}: {s['night_nonzero']} night values != 0")
        parts.append(f"{mode}: [{s['min']:.4g}, {s['max']:.4g}], night values {s.get('night_values')} all 0")
    return _result("cf_night_and_range", problems, "; ".join(parts), ev.pv_info["cf_checks"])


def check_allocation(ev: SolarEvidence) -> CheckResult:
    if ev.allocation is None:
        return _skip("allocation_weights", "public inputs not available: " + NEEDS_TEXT["public"])
    a, rep = ev.allocation, ev.plants_report or {}
    problems = []
    if abs(float(a.pop_share.sum()) - 1.0) > 1e-12:
        problems.append(f"population shares sum to {a.pop_share.sum()!r}")
    if (a.pop_share < 0).any() or (a.large_by_cell < 0).any():
        problems.append("negative allocation")
    if abs(float(a.large_by_cell.sum()) - a.large_mw) > 1e-9:
        problems.append("L_c does not sum to L")
    if abs(a.report["large_total_mw"] - a.large_mw - a.report["large_outside_grid_mw"]) > 1e-6:
        problems.append("plants inside + outside the grid do not add up to the plant list")
    summ = (rep.get("summary_sheet") or {}).get("cumulative_mw")
    if summ is not None and abs(summ - a.report["large_total_mw"]) > 1e-6:
        problems.append(f"plant list sums to {a.report['large_total_mw']} MW but its summary sheet states {summ} MW")
    return _result("allocation_weights", problems,
                   f"w_c sums to 1 over {a.report['cells_with_population']} cells; L = {a.large_mw:.2f} MW in {a.report['cells_with_large_plants']} cells "
                   f"({len(a.report['plants_outside_grid'])} plant(s) and {a.report['inhabitants_outside_grid']:.0f} inhabitants outside the grid)",
                   {"large_mw": a.large_mw, "weights_sum": float(a.pop_share.sum())})


def check_capacity_reconciliation(ev: SolarEvidence) -> CheckResult:
    if ev.reconciliation is None:
        return _skip("capacity_reconciliation", "no hourly capacity C(t): " + NEEDS_TEXT["fingrid_267"])
    r = ev.reconciliation
    return _result("capacity_reconciliation", [] if r["ok"] else [f"max |sum_c capacity - C(t)| = {r['max_abs_diff_mw']:.3g} MW > {r['tolerance_mw']:g}"],
                   f"sum over cells == C(t) for {r['hours_checked']} hours (max diff {r['max_abs_diff_mw']:.2e} MW, {r['hours_nan']} hours without C(t) skipped)", dict(r))


def check_coverage_threshold(ev: SolarEvidence) -> CheckResult:
    g = ev.ongrid
    if g is None:
        return _skip("coverage_threshold", "SARAH-3 not ingested: " + NEEDS_TEXT["sarah"])
    problems = []
    sat = g.source_code == SRC_SATELLITE
    hi = g.valid_area_fraction >= g.threshold
    if not np.array_equal(sat, hi):
        problems.append(f"{int((sat != hi).sum())} cell-hour(s) where satellite use disagrees with valid_area_fraction >= {g.threshold}")
    if g.threshold != C.SATELLITE_MIN_AREA_FRACTION:
        problems.append(f"threshold {g.threshold} != {C.SATELLITE_MIN_AREA_FRACTION}")
    if not np.all((g.valid_area_fraction >= 0) & (g.valid_area_fraction <= 1.0 + 1e-6)):
        problems.append("valid_area_fraction outside [0, 1]")
    if np.isnan(g.direct[sat]).any() or np.isnan(g.diffuse[sat]).any():
        problems.append("satellite cell-hours with NaN direct/diffuse")
    if (g.fully_covered != (g.static_covered_fraction >= g.threshold)).any():
        problems.append("fully_covered set disagrees with static_covered_fraction")
    return _result("coverage_threshold", problems, f"satellite use == (valid area fraction >= {g.threshold}) for all {sat.size} cell-hours; {int(g.fully_covered.sum())} fully covered cells")


def check_mode_labelling(ev: SolarEvidence) -> CheckResult:
    problems = []
    if ev.synthetic != C.is_synthetic_mode(ev.mode):
        problems.append(f"synthetic flag {ev.synthetic} contradicts mode {ev.mode}")
    for name, src in ev.sources.items():
        if ev.synthetic != ("SYNTHETIC" in str(src)):
            problems.append(f"source {name!r} ({src!r}) is {'not ' if ev.synthetic else ''}labelled SYNTHETIC")
    return _result("mode_labelling", problems, f"mode {ev.mode}: {'SYNTHETIC' if ev.synthetic else 'real'}; {len(ev.sources)} source label(s) consistent")


def check_units(ev: SolarEvidence) -> CheckResult:
    n = ev.national
    if n is None:
        return _skip("unit_consistency", "no national table (needs ERA5 solar data and Fingrid 267)")
    problems = []
    for c in ("era5_only_mw", "hybrid_mw"):
        if c in n and (n[c].dropna() < 0).any():
            problems.append(f"{c} has negative MW")
    if "surplus_362_kwh" in n and "surplus_362_mw" in n:
        both = n[["surplus_362_kwh", "surplus_362_mw"]].dropna()
        if len(both) and not np.allclose(both["surplus_362_kwh"] / 1000.0, both["surplus_362_mw"]):
            problems.append("362 MW != kWh / 1000")
    if "capacity_267_mw" in n and "era5_only_mw" in n:
        both = n[["capacity_267_mw", "era5_only_mw"]].dropna()
        if len(both) and (both["era5_only_mw"] > both["capacity_267_mw"] + 1e-6).any():
            problems.append("national power above the capacity C(t): MW/MWp mix-up?")
    return _result("unit_consistency", problems, "MW non-negative and below C(t); 362 kWh -> MW = /1000; energy GWh = sum(MW x 1 h)/1000")


def check_fingrid_definitions(ev: SolarEvidence) -> CheckResult:
    v = (ev.fingrid_info or {}).get("verification")
    if not v:
        return _skip("fingrid_definitions", "Fingrid solar series not ingested: " + NEEDS_TEXT["fingrid_267"])
    bad = [r for r in v["datasets"] if r["status"] == "mismatch"]
    if v["overall"] == "unverified":
        return _skip("fingrid_definitions", "unverified (no key or --offline); cached data used as is")
    return _result("fingrid_definitions", [f"dataset {r['dataset_id']}: {r.get('reason')}" for r in bad],
                   "units verified: " + ", ".join(f"{r['dataset_id']}={r.get('reported_unit')}" for r in v["datasets"]))


def run_checks(ev: SolarEvidence, expected_hours: "int | None" = None) -> list[CheckResult]:
    out = [check_hourly_index(ev.hours, "window", expected_hours if expected_hours is not None else len(ev.hours))]
    if ev.national is not None:
        out.append(check_hourly_index(ev.national.index, "national", len(ev.hours)))
    out += [check_sarah_alignment(ev), check_era5_units_and_shift(ev), check_cf(ev), check_allocation(ev), check_capacity_reconciliation(ev),
            check_coverage_threshold(ev), check_mode_labelling(ev), check_units(ev), check_fingrid_definitions(ev)]
    return out


# --- reconciliation -------------------------------------------------------------------------------------------------------------------
RECON_COLUMNS = ["metric", "documented", "observed", "diff", "unit", "status", "tolerance", "needs", "note"]


@dataclass(frozen=True)
class Target:
    key: str
    documented: float
    unit: str
    needs: tuple[str, ...]
    abs_tol: float
    rel_tol: float = 0.0
    note: str = ""


TARGETS: tuple[Target, ...] = (
    Target("hourly_values", 744, "count", (), 0.0, note="hours in the July window"),
    Target("fully_covered_cells", 777, "count", ("sarah",), 0.0, note="static: >= 1 fully valid daytime hour per source pixel, area share >= 99.9 %"),
    Target("grid_cells", 2332, "count", (), 0.0),
    Target("never_covered_source_pixels_pct", 25.4, "%", ("sarah",), 0.05, note="share of 0.05 deg cells that never had daytime data"),
    Target("large_plants", 30, "count", ("public",), 0.0),
    Target("large_plants_mw", 250.66, "MW", ("public",), 0.005),
    Target("largest_plant_mw", 103.2, "MW", ("public",), 0.005, note="Isosuo, Utajarvi"),
    Target("largest_plant_share_pct", 41.0, "%", ("public",), 0.5),
    Target("inhabited_cells", 96904, "count", ("public",), 0.0),
    Target("inhabitants", 5573552, "count", ("public",), 0.0),
    Target("ghi_sarah_covered_wm2", 242.5, "W m-2", ("sarah", "era5"), 0.0, 0.005, "Table 2, hourly means incl. night, 777 covered cells"),
    Target("ghi_era5_covered_wm2", 241.9, "W m-2", ("sarah", "era5"), 0.0, 0.005, "Table 2"),
    Target("mean_cf_era5_all_cells", 0.195, "1", ("era5",), 0.002, note="mean CF over all cells"),
    Target("mean_cf_hybrid_all_cells", 0.195, "1", ("era5", "sarah"), 0.002),
    Target("mean_cf_era5_covered_cells", 0.2027, "1", ("era5", "sarah"), 0.0005),
    Target("mean_cf_hybrid_covered_cells", 0.2022, "1", ("era5", "sarah"), 0.0005),
    Target("energy_era5_gwh", 214.0, "GWh", ("era5", "fingrid_267"), 0.0, 0.005),
    Target("energy_hybrid_gwh", 216.2, "GWh", ("era5", "sarah", "fingrid_267"), 0.0, 0.005),
    Target("energy_fingrid_248_gwh", 248.2, "GWh", ("fingrid_248",), 0.0, 0.005, "forecast, not measured"),
    Target("energy_ratio_era5_to_248", 0.862, "1", ("era5", "fingrid_267", "fingrid_248"), 0.005),
    Target("energy_ratio_hybrid_to_248", 0.871, "1", ("era5", "sarah", "fingrid_267", "fingrid_248"), 0.005),
    Target("mean_cf_vs_267_era5", 0.1965, "1", ("era5", "fingrid_267"), 0.002, note="national MW / C(t), mean of hours"),
    Target("mean_cf_vs_267_hybrid", 0.1984, "1", ("era5", "sarah", "fingrid_267"), 0.002),
    Target("pearson_era5_vs_248", 0.981, "1", ("era5", "fingrid_267", "fingrid_248"), 0.005, note="Pearson r of hourly values, not regression R2"),
    Target("pearson_hybrid_vs_248", 0.983, "1", ("era5", "sarah", "fingrid_267", "fingrid_248"), 0.005),
    Target("pearson_era5_vs_362", 0.978, "1", ("era5", "fingrid_267", "fingrid_362"), 0.005),
    Target("pearson_hybrid_vs_362", 0.985, "1", ("era5", "sarah", "fingrid_267", "fingrid_362"), 0.005),
    Target("cycle_centre_era5_min", 10 * 60 + 21, "min UTC", ("era5", "fingrid_267"), 2.0, note="10:21 UTC"),
    Target("cycle_centre_hybrid_min", 10 * 60 + 17, "min UTC", ("era5", "sarah", "fingrid_267"), 2.0, note="10:17 UTC"),
    Target("cycle_centre_248_min", 10 * 60 + 24, "min UTC", ("fingrid_248",), 2.0, note="10:24 UTC"),
    Target("cycle_centre_362_min", 10 * 60 + 32, "min UTC", ("fingrid_362",), 2.0, note="10:32 UTC"),
    Target("energy_era5_popw_gwh", 213.3, "GWh", ("era5", "fingrid_267"), 0.0, 0.005, "capacity by population only"),
    Target("energy_hybrid_popw_gwh", 215.9, "GWh", ("era5", "sarah", "fingrid_267"), 0.0, 0.005, "capacity by population only"),
)


def reconciliation(observed: Mapping[str, "float | None"], synthetic: bool, available: Mapping[str, str]) -> pd.DataFrame:
    """One row per documented target. `available` maps an input key (sarah, era5, fingrid_267, ...) to "" when present,
    else the reason it is missing. Synthetic runs show the observed value with status not_reproducible_synthetic."""
    rows = []
    for t in TARGETS:
        obs = observed.get(t.key)
        missing = [k for k in t.needs if available.get(k, "missing") != ""]
        tol = max(t.abs_tol, t.rel_tol * abs(t.documented))
        if synthetic:
            status, note = "not_reproducible_synthetic", t.note
        elif obs is None or missing:
            status = "unavailable"
            note = "; ".join(filter(None, [t.note, "missing input: " + " + ".join(NEEDS_TEXT.get(k, k) for k in (missing or t.needs))]))
        else:
            status, note = ("match" if abs(obs - t.documented) <= tol + 1e-9 else "differs"), t.note
        rows.append({"metric": t.key, "documented": t.documented, "observed": None if obs is None else float(obs),
                     "diff": None if obs is None else float(obs) - t.documented, "unit": t.unit, "status": status,
                     "tolerance": f"abs {tol:g}" if tol else "exact", "needs": ",".join(t.needs), "note": note})
    return pd.DataFrame(rows, columns=RECON_COLUMNS)


# --- licences + manifest -----------------------------------------------------------------------------------------------------------------
def solar_licences(synthetic: bool, info: Mapping[str, Any]) -> dict[str, Any]:
    sarah_meta = (((info.get("sarah") or {}).get("inventory") or {}).get("SIS") or {}).get("metadata") or {}
    pub = info.get("public") or {}

    def lic(kind: str) -> Any:
        return ((pub.get(kind) or {}).get("licence")) or None

    out = {
        "cm_saf_sarah3": {"name": "CM SAF / EUMETSAT licence as stated in the file headers", "attribution": "Pfeifroth et al. (2023), CM SAF SARAH-3 ICDR, DOI 10.5676/EUM_SAF_CM/SARAH/V003",
                          "file_header_license": sarah_meta.get("license"), "status": "text copied from the NetCDF attribute `license` (not interpreted)"},
        "era5": {"name": "Copernicus Climate Change Service (ERA5) licence", "attribution": "Contains modified Copernicus Climate Change Service information (2026)",
                 "status": "assumed - verify at https://cds.climate.copernicus.eu (licence accepted per CDS account)"},
        "fingrid": {"name": "CC BY 4.0", "attribution": "Fingrid Oyj open data (datasets 267, 248, 362)", "status": "verified on the public dataset pages by the lead on 2026-11-27"},
        "statistics_finland": {"name": "CC BY 4.0 (assumed)", "attribution": "Statistics Finland (population grid vaki2025_1km, municipalities kunta1000k_2025)",
                               "status": "NOT stated by the WFS service or its metadata record on retrieval; taken from the solar report", "wfs_findings": {"population": lic("population"), "municipalities": lic("municipalities")}},
        "suomen_uusiutuvat": {"name": "no licence stated in the workbook", "attribution": "Suomen Uusiutuvat ry (2025), list of solar PV farms, status 30.6.2025", "status": "check with the owner before redistribution"},
        "atlite": {"name": "MIT", "status": "see package metadata"},
        "_note": ("SYNTHETIC run: no third-party data is contained in the outputs; entries describe the real sources." if synthetic else
                  "Statuses marked assumed / not stated have not been confirmed against the live licence text."),
    }
    return out


def build_solar_manifest(mode: str, synthetic: bool, info: Mapping[str, Any], out_dir: Path, *, status: str, hours: pd.DatetimeIndex,
                         validation: "Mapping[str, Any] | Sequence[CheckResult] | None", recon: "pd.DataFrame | Sequence[Mapping[str, Any]] | None",
                         unavailable: Sequence[Mapping[str, Any]], metrics: "Mapping[str, Any] | None") -> dict[str, Any]:
    label = ("DEMO / SYNTHETIC - not real data" if mode == "solar-demo" else "SYNTHETIC - not real data") if synthetic else "real-data run"
    return {
        "manifest_schema_version": "1.0", "kind": "solar", "status": status, "generated_utc": utc_now_iso(), "run_mode": mode, "synthetic": synthetic,
        "demo": mode == "solar-demo", "label": label, "model_version": MODEL_VERSION,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "libraries": package_versions(), "git": git_info()},
        "window": {"first_hour_utc": hours[0].strftime("%Y-%m-%dT%H:%M:%SZ"), "last_hour_utc": hours[-1].strftime("%Y-%m-%dT%H:%M:%SZ"), "hours": int(len(hours)),
                   "time_label": "start of the hour, UTC"},
        "grid": info.get("grid"), "sarah": info.get("sarah"), "era5_solar": info.get("era5_solar"), "pv": info.get("pv"), "public": info.get("public"),
        "allocation": info.get("allocation"), "fingrid_solar": info.get("fingrid_solar"), "units": SOLAR_UNITS,
        "validation": validation if (validation is None or isinstance(validation, Mapping)) else {
            "passed": sum(r.status == PASS for r in validation), "failed": sum(r.status == FAIL for r in validation),
            "skipped": sum(r.status == SKIPPED for r in validation), "results": [r.as_dict() for r in validation]},
        "reconciliation": recon.to_dict(orient="records") if isinstance(recon, pd.DataFrame) else recon,
        "metrics": metrics, "unavailable": list(unavailable), "licences": solar_licences(synthetic, info),
        "files": describe_outputs(out_dir, exclude=(MANIFEST_NAME,), units=SOLAR_UNITS),
        "notes": ["Report targets are compared, never forced.", "Fingrid 248 is a forecast and 362 is partial-scope surplus: neither is measured total national generation.",
                  "Hours without Fingrid 267 stay missing; nothing is filled or invented."],
    }


def write_solar_manifest(out_dir: Path, **kw: Any) -> dict[str, Any]:
    m = build_solar_manifest(out_dir=out_dir, **kw)
    atomic_write_json(Path(out_dir) / MANIFEST_NAME, m)
    return m


__all__ = ["SolarEvidence", "run_checks", "reconciliation", "write_solar_manifest", "TARGETS", "NEEDS_TEXT", "SOLAR_UNITS"]
