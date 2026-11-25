"""Run manifest: provenance, schemas + units, hashes, versions, licences and what was unavailable."""

from __future__ import annotations

import importlib.metadata as md
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from . import __version__
from . import config as C
from .io_utils import UNITS, atomic_write_json, sha256_file, utc_now_iso

MANIFEST_SCHEMA_VERSION = "1.0"
MANIFEST_NAME = "manifest.json"
LIBRARIES = ("numpy", "pandas", "xarray", "netCDF4", "h5netcdf", "scipy", "scikit-learn", "geopandas", "pyogrio",
             "shapely", "pyproj", "pyarrow", "requests", "atlite", "cdsapi")

# Units for output-table columns that are not in io_utils.UNITS.
EXTRA_UNITS: dict[str, str] = {
    "theoretical_cf": "1", "actual_cf": "1", "residual_mw": "MW", "residual_cf": "1", "residual_raw_mw": "MW",
    "theoretical_gwh": "GWh", "actual_gwh": "GWh", "residual_gwh": "GWh", "mean_theoretical_mw": "MW",
    "mean_actual_mw": "MW", "mean_imbalance_eur": "EUR MWh-1", "capture_ratio_pct": "%", "hours": "h",
    "p_base_uncurtailed_mw": "MW", "estimated_curtailment_mw": "MW", "actual_wind_clean_mw": "MW",
    "residual_clean_mw": "MW", "res_after_linear_mw": "MW", "res_after_quasi_mw": "MW", "pred_inference_mw": "MW",
    "excess_shortfall_mw": "MW", "bias_linear_mw": "MW", "bias_quasi_mw": "MW", "mean_gen_mw": "MW",
    "mean_hub_m": "m", "mean_sp_wm2": "W m-2", "mean_rated_mw": "MW", "installed_mw": "MW",
    "mean_hub_height_m": "m", "mean_rotor_diam_m": "m", "mean_commissioning_year": "year",
    "era5_lat": "degrees_north", "era5_lon": "degrees_east", "wind_to_cons_ratio": "1", "hour_sin": "1", "hour_cos": "1",
    "curtail_gate_w": "1", "score_price": "1", "score_grid": "1", "match_dist_m": "m", "turbine_count": "1",
    "theoretical_wind_mw": "MW",
}

LICENCES_STATIC = {
    "era5": {"name": "Copernicus Climate Change Service (ERA5) licence", "attribution": "Contains modified Copernicus Climate Change Service information",
             "status": "assumed - verify at https://cds.climate.copernicus.eu (licence accepted per CDS account)"},
    "osm": {"name": "Open Database Licence (ODbL) 1.0", "attribution": "(c) OpenStreetMap contributors",
            "status": "assumed - verify at https://www.openstreetmap.org/copyright"},
    "fingrid": {"name": "CC BY 4.0", "attribution": "Fingrid Oyj open data", "status": "to be verified at https://data.fingrid.fi"},
}


def package_versions() -> dict[str, "str | None"]:
    out: dict[str, "str | None"] = {}
    for lib in LIBRARIES:
        try:
            out[lib] = md.version(lib)
        except md.PackageNotFoundError:
            out[lib] = None
    out["windpower"] = __version__
    return out


def git_info(repo: "Path | None" = None) -> dict[str, Any]:
    """HEAD commit and dirty flag, or None values when git/the repo is unavailable."""
    repo = repo or C.REPO_ROOT

    def run(*args: str) -> "str | None":
        try:
            r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain") if commit else None
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def licence_block(synthetic: bool, arcgis_meta: "Mapping[str, Any] | None") -> dict[str, Any]:
    arc_meta = arcgis_meta or {}
    text = arc_meta.get("copyright_text") or arc_meta.get("license_info")
    try:
        meta = md.metadata("atlite")
        atlite_mit = "MIT" in " ".join(filter(None, [meta.get("License"), meta.get("License-Expression"), *(meta.get_all("Classifier") or [])]))
    except md.PackageNotFoundError:
        atlite_mit = False
    out = {k: dict(v) for k, v in LICENCES_STATIC.items()}
    out["arcgis_layer"] = {"name": "Finnish wind turbine layer (ArcGIS FeatureServer)", "copyright_text": arc_meta.get("copyright_text"),
                           "license_info": arc_meta.get("license_info"),
                           "status": "captured from layer metadata at fetch time" if text and not synthetic else
                                     "unavailable - layer licence text not captured (no real fetch)"}
    out["atlite"] = {"name": "MIT", "status": "verified from installed package metadata" if atlite_mit else "assumed - verify at https://github.com/PyPSA/atlite"}
    out["_note"] = ("SYNTHETIC run: no third-party data is contained in the outputs; the entries describe the real sources the full run would use."
                    if synthetic else "Statuses marked assumed/to be verified have not been checked against the live licence text.")
    return out


# --- output scan --------------------------------------------------------------------------------------
def _unit_for(col: str, units: Mapping[str, str]) -> "str | None":
    return units.get(col) or EXTRA_UNITS.get(col) or UNITS.get(col)


def describe_outputs(out_dir: Path, exclude: Sequence[str] = (MANIFEST_NAME,), units: "Mapping[str, str] | None" = None) -> list[dict[str, Any]]:
    """{path, bytes, sha256, rows, columns{name: {dtype, unit}}} for every file below `out_dir`."""
    units = units or {}
    files: list[dict[str, Any]] = []
    for p in sorted(out_dir.rglob("*")):
        rel = p.relative_to(out_dir).as_posix()
        if not p.is_file() or rel in exclude or p.name.endswith(".part"):
            continue
        entry: dict[str, Any] = {"path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p), "rows": None, "columns": None}
        if p.suffix == ".csv":
            df = pd.read_csv(p)
            entry["rows"] = int(len(df))
            entry["columns"] = {str(c): {"dtype": str(df[c].dtype), "unit": _unit_for(str(c), units)} for c in df.columns}
        files.append(entry)
    return files


def metric_statement(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Which target/split/model every reported metric uses (unique combinations)."""
    combos = sorted({(m["target"], m["split"], m["model"]) for m in metrics})
    desc = {"observed_actual": "error vs the Fingrid-reported (curtailed) actual wind output",
            "reconstructed_clean_actual": "retrospective benchmark vs observed + estimated curtailment; not a forecast error"}
    return [{"target": t, "split": s, "model": m, "meaning": desc[t]} for t, s, m in combos]


def _read(out_dir: Path, name: str) -> Any:
    p = out_dir / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def build_manifest(run_mode: str, synthetic: bool, info: Mapping[str, Any], out_dir: Path, *, status: str = "complete",
                   units: "Mapping[str, str] | None" = None) -> dict[str, Any]:
    """Assemble the manifest from run info (stage by stage) and the files currently in `out_dir`."""
    validation = _read(out_dir, "validation.json")
    reconciliation = _read(out_dir, "reconciliation.json")
    metrics = _read(out_dir, "metrics.json") or []
    fleet = info.get("fleet", {})
    era5 = info.get("era5", {})
    fingrid = info.get("fingrid", {})
    label = ("DEMO / SYNTHETIC - not real data" if run_mode == "demo" else "SYNTHETIC - not real data") if synthetic else "real-data run"
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "status": status,
        "generated_utc": utc_now_iso(),
        "run_mode": run_mode,
        "synthetic": synthetic,
        "demo": run_mode == "demo",
        "label": label,
        "model_version": f"windpower-{__version__}",
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "libraries": package_versions(),
                        "git": git_info()},
        "source_dates": {
            "era5_window": era5.get("window"), "era5_retrieval": era5.get("retrieval"),
            "fleet_snapshot": fleet.get("snapshot"),
            "fingrid_fetched_at": {k: v.get("fetched_at_utc") for k, v in (fingrid.get("cache_meta") or {}).items()},
        },
        "era5": era5,
        "fleet": fleet,
        "fingrid": fingrid,
        "calibration": info.get("calibration", {}),
        "time_coverage": info.get("time_coverage", {}),
        "files": describe_outputs(out_dir, units=units),
        "validation": validation,
        "reconciliation": reconciliation,
        "metrics_statement": metric_statement(metrics) if metrics else [],
        "licences": licence_block(synthetic, (fleet.get("arcgis_meta") or {})),
        "unavailable": info.get("unavailable", []),
        "notes": ["Documented notebook values are reconciliation targets only; they are compared, never forced.",
                  "Retrospective (reconstructed_clean_actual) and website-facing (inference_only, observed_actual) metrics are reported separately."],
    }


def write_manifest(run_mode: str, synthetic: bool, info: Mapping[str, Any], out_dir: Path, **kw: Any) -> dict[str, Any]:
    manifest = build_manifest(run_mode, synthetic, info, out_dir, **kw)
    atomic_write_json(out_dir / MANIFEST_NAME, manifest)
    return manifest


def unavailable_manifest(run_mode: str, items: Sequence[Mapping[str, str]], out_dir: Path) -> dict[str, Any]:
    """Minimal manifest for a run that stopped because inputs were unavailable."""
    manifest = {"manifest_schema_version": MANIFEST_SCHEMA_VERSION, "status": "incomplete_inputs_unavailable",
                "generated_utc": utc_now_iso(), "run_mode": run_mode, "synthetic": False, "demo": False,
                "environment": {"python": sys.version.split()[0], "libraries": package_versions(), "git": git_info()},
                "unavailable": list(items), "files": []}
    atomic_write_json(out_dir / MANIFEST_NAME, manifest)
    return manifest
