"""Solar website artifact: `<outputs>/site/solar/v1/` (the wind artifact `site/v1/` is never touched).

Files (all labelled with model version, units, run mode and the SYNTHETIC/DEMO flag):
  manifest.json        what is in the artifact: window, grid, modes (available / unavailable + reason), units, convention evidence,
                       panel / orientation, allocation and coverage summaries, licences, file hashes
  grid.json            axes, cell size and per-cell arrays (row-major, latitude ascending): satellite_covered, coverage fraction, domain
                       fraction, population share, large-plant capacity
  coverage.json        satellite coverage summary and the 0.05 deg pixel coverage map (percent, uint8 range)
  cf_era5.bin          uint16 [time, lat, lon] capacity factor x 10000 (rounding error <= 5e-5), C order, little endian
  cf_hybrid.bin        same for the hybrid input
  source_hybrid.bin    uint8 [time, lat, lon] source / fallback reason code per cell-hour
  meta.json            shape / dtype / scale / axes of the .bin files and the source-code table
  national_hourly.json UTC hour starts and one entry per series with availability, reason, unit, native unit and scope;
                       unavailable series carry `available: false` and no values (never zero-filled)

Modes or series that cannot be built from the available inputs are marked unavailable with the exact reason; the artifact is
still written (e.g. SARAH-3 + public data only: grid.json, coverage.json and manifest.json).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import config as C
from .export_site import MAX_BYTES, SITE_DIR as WIND_SITE_DIR, ExportSizeError
from .io_utils import atomic_write_bytes, atomic_write_json, iso_utc, redact_paths
from .manifest import describe_outputs
from .sarah_regrid import SOURCE_CODES, SarahOnGrid
from .solar_pv import CfCubes, panel_record
from .solar_validate import MODEL_VERSION, SOLAR_UNITS, solar_licences

SOLAR_SITE_DIR = Path("site") / "solar" / "v1"
ARTIFACT_SCHEMA_VERSION = "1.0"
CF_SCALE = 10_000
# column -> (export key, title, unit, native unit, scope, input that must be available)
NATIONAL_SERIES: dict[str, tuple[str, str, str, str, str, str]] = {
    "era5_only_mw": ("era5_only_mw", "ERA5-only national PV", "MW", "MW", "model estimate from ERA5 irradiance scaled with Fingrid 267 capacity; no calibration", "era5"),
    "hybrid_mw": ("hybrid_mw", "Hybrid (SARAH-3 + ERA5) national PV", "MW", "MW",
                  "model estimate; satellite direct/diffuse where covered >= 99.9 % else ERA5; no calibration", "hybrid"),
    "era5_only_popw_mw": ("era5_only_popw_mw", "ERA5-only, capacity by population only", "MW", "MW", "sensitivity case without the large-plant list", "era5"),
    "hybrid_popw_mw": ("hybrid_popw_mw", "Hybrid, capacity by population only", "MW", "MW", "sensitivity case without the large-plant list", "hybrid"),
    "forecast_248_mw": ("fingrid_248_forecast_mw", "Fingrid solar generation forecast (248)", "MW", "MWh/h", "forecast, not measured generation", "fingrid_248"),
    "surplus_362_mw": ("fingrid_362_av06_small_scale_surplus_mw", "Fingrid small-scale surplus, solar AV06 (362)", "MW", "kWh",
                       "small-scale surplus only (after own consumption), not total generation; kWh/h -> MW = /1000", "fingrid_362"),
    "capacity_267_mw": ("fingrid_267_capacity_mw", "Fingrid capacity used in the solar forecast (267)", "MW", "MW",
                        "capacity used in the forecast, not official capacity", "fingrid_267"),
}


def site_label(mode: str, synthetic: bool) -> str:
    if mode == "solar-demo":
        return "DEMO / SYNTHETIC - not real data"
    return "SYNTHETIC - not real data" if synthetic else "Retrospective SARAH-3 / ERA5 PV simulation (no national calibration)"


def _flat(a: np.ndarray, decimals: int = 6) -> list:
    return [None if not np.isfinite(v) else round(float(v), decimals) for v in np.asarray(a, dtype=float).ravel()]


def _series_reason(col: str, need: str, unavailable: Mapping[str, str]) -> str:
    # model series also need Fingrid 267 for the national scaling
    base = unavailable.get(need) or (unavailable.get("era5") if need == "hybrid" else None)
    if base:
        return base
    if col.endswith("mw") and need in ("era5", "hybrid") and "fingrid_267" in unavailable:
        return unavailable["fingrid_267"]
    return "not computed"


def export_site_solar(out_root: "str | Path", *, mode: str, synthetic: bool, hours: pd.DatetimeIndex, grid: C.GridSpec, ongrid: "SarahOnGrid | None",
                      cubes: "CfCubes | None", allocation: Any, national: "pd.DataFrame | None", info: Mapping[str, Any],
                      unavailable: Mapping[str, str], coverage_summary: "Mapping[str, Any] | None" = None) -> dict[str, Any]:
    """Write site/solar/v1 under `out_root`; returns its manifest. `unavailable` maps keys (era5, hybrid, sarah, public, fingrid_267, ...) to reasons."""
    out = Path(out_root) / SOLAR_SITE_DIR
    unavailable = {k: redact_paths(str(v)) for k, v in unavailable.items()}      # the artifact is served to browsers: no home-directory paths
    if out.resolve() == (Path(out_root) / WIND_SITE_DIR).resolve():
        raise RuntimeError("refusing to write the solar artifact into the wind site directory")
    n_t, n_y, n_x = len(hours), grid.n_lat, grid.n_lon
    modes: dict[str, Any] = {}
    for m in ("era5", "hybrid"):
        cube = None if cubes is None else (cubes.era5 if m == "era5" else cubes.hybrid)
        modes[m] = {"available": cube is not None, "reason": None if cube is not None else (unavailable.get(m) or unavailable.get("era5") or "not computed"),
                    **({"file": f"cf_{m}.bin"} if cube is not None else {})}
    est = n_t * n_y * n_x * (2 * sum(1 for v in modes.values() if v["available"]) + (1 if modes["hybrid"]["available"] else 0))
    if est > MAX_BYTES:
        raise ExportSizeError(f"solar artifact bins would be {est / 1e6:.1f} MB (> {MAX_BYTES / 1e6:.0f} MB limit)")
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("cf_era5.bin", "cf_hybrid.bin", "source_hybrid.bin"):     # a re-export must not leave files of an earlier, fuller run
        (out / stale).unlink(missing_ok=True)
    bins: dict[str, Any] = {}
    for m in ("era5", "hybrid"):
        if modes[m]["available"]:
            cube = cubes.era5 if m == "era5" else cubes.hybrid     # type: ignore[union-attr]
            arr = np.rint(np.clip(cube.transpose("time", "latitude", "longitude").values, 0.0, 1.0) * CF_SCALE).astype("<u2")
            atomic_write_bytes(out / f"cf_{m}.bin", arr.tobytes(order="C"))
            bins[f"cf_{m}.bin"] = {"dtype": "uint16", "endianness": "little", "shape": [n_t, n_y, n_x], "axes": ["time", "latitude", "longitude"],
                                   "scale": CF_SCALE, "meaning": "capacity factor (fraction) = value / scale", "max_rounding_error": 0.5 / CF_SCALE}
    if modes["hybrid"]["available"]:
        code = cubes.irradiance["hybrid"]["source_code"].transpose("time", "y", "x").values.astype("u1")    # type: ignore[union-attr]
        atomic_write_bytes(out / "source_hybrid.bin", code.tobytes(order="C"))
        bins["source_hybrid.bin"] = {"dtype": "uint8", "shape": [n_t, n_y, n_x], "axes": ["time", "latitude", "longitude"],
                                     "codes": {str(k): v for k, v in SOURCE_CODES.items()},
                                     "meaning": "0 = satellite (SARAH-3 direct/diffuse), others = ERA5 fallback reason"}
    atomic_write_json(out / "meta.json", {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "files": bins,
                                          "time_axis": {"first": iso_utc(hours[0]), "last": iso_utc(hours[-1]), "n": n_t, "label": "start of the hour, UTC"}})

    lats, lons = np.array(grid.lats()), np.array(grid.lons())
    cells: dict[str, Any] = {}
    if ongrid is not None:
        cells.update(satellite_covered=[int(v) for v in ongrid.fully_covered.ravel()], satellite_coverage_fraction=_flat(ongrid.satellite_coverage_fraction, 4),
                     domain_fraction=_flat(ongrid.domain_fraction, 4), static_covered_fraction=_flat(ongrid.static_covered_fraction, 4))
    else:
        cells["satellite"] = {"available": False, "reason": unavailable.get("sarah", "SARAH-3 not ingested")}
    if allocation is not None:
        cells.update(population_share=_flat(allocation.pop_share, 9), large_capacity_mw=_flat(allocation.large_by_cell, 4))
    else:
        cells["allocation"] = {"available": False, "reason": unavailable.get("public", "public inputs not ingested")}
    label = site_label(mode, synthetic)
    flags = {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "model_version": MODEL_VERSION, "run_mode": mode, "synthetic": synthetic,
             "demo": mode == "solar-demo", "label": label, "units": SOLAR_UNITS}
    atomic_write_json(out / "grid.json", {**flags, "grid": {"name": grid.name, "n_lat": n_y, "n_lon": n_x, "res_deg": grid.res, "cell_bounds": "centre +- res/2",
                                                            "lat": lats.tolist(), "lon": lons.tolist(),
                                                            "order": "row-major, latitude ascending then longitude ascending"}, "cells": cells})

    cov: dict[str, Any] = {**flags}
    if ongrid is not None:
        pm = ongrid.pixel_daytime_valid_share
        cov.update(available=True, summary=dict(coverage_summary or {}), threshold=ongrid.threshold,
                   pixel_map={"lat0": float(ongrid.src_lat[0]), "lon0": float(ongrid.src_lon[0]), "res_deg": C.SARAH_RES_DEG, "shape": list(pm.shape),
                              "unit": "percent of daytime hours with a fully valid hourly value (255 = no daytime hours)",
                              "values": [255 if not np.isfinite(v) else int(round(100 * float(v))) for v in pm.ravel()]})
    else:
        cov.update(available=False, reason=unavailable.get("sarah", "SARAH-3 not ingested"))
    atomic_write_json(out / "coverage.json", cov)

    series: dict[str, Any] = {}
    for col, (key, title, unit, native, scope, need) in NATIONAL_SERIES.items():
        rec: dict[str, Any] = {"title": title, "unit": unit, "native_unit": native, "scope": scope}
        if national is not None and col in national and national[col].notna().any():
            rec |= {"available": True, "reason": None, "values": _flat(national[col], 4), "hours_missing": int(national[col].isna().sum())}
            if col == "surplus_362_mw" and "surplus_362_kwh" in national:
                rec["values_native_kwh"] = _flat(national["surplus_362_kwh"], 1)
        else:
            rec |= {"available": False, "reason": _series_reason(col, need, unavailable)}
        series[key] = rec
    atomic_write_json(out / "national_hourly.json", {**flags, "hours": [iso_utc(h) for h in hours], "series": series,
                                                     "note": "Fingrid 248 is a forecast and 362 is partial-scope surplus: neither is measured total generation"})

    manifest = {**flags, "kind": "solar-site", "content": "solar PV capacity factors and national series",
                "window": {"first_hour_utc": iso_utc(hours[0]), "last_hour_utc": iso_utc(hours[-1]), "hours": n_t, "time_label": "start of the hour, UTC"},
                "grid": {"name": grid.name, "n_lat": n_y, "n_lon": n_x, "res_deg": grid.res, "lat_range": [float(lats[0]), float(lats[-1])],
                         "lon_range": [float(lons[0]), float(lons[-1])]},
                "modes": modes, "panel": panel_record(), "capacity_factor_encoding": {"dtype": "uint16", "scale": CF_SCALE},
                "convention": (info.get("era5_solar") or {}).get("convention"),
                "allocation": None if allocation is None else allocation.report, "coverage": None if coverage_summary is None else dict(coverage_summary),
                "national_series": {k: {"available": v["available"], "reason": v["reason"]} for k, v in series.items()},
                "unavailable": dict(unavailable), "licences": solar_licences(synthetic, info),
                "notes": ["No national calibration is applied.", "Hours without Fingrid 267 are missing, never filled."]}
    atomic_write_json(out / "manifest.json", manifest)
    files = describe_outputs(out, exclude=("manifest.json",))
    total = sum(f["bytes"] for f in files) + (out / "manifest.json").stat().st_size
    if total > MAX_BYTES:
        raise ExportSizeError(f"solar artifact is {total / 1e6:.1f} MB (> {MAX_BYTES / 1e6:.0f} MB limit)")
    manifest |= {"files": files, "total_bytes": total}
    atomic_write_json(out / "manifest.json", manifest)
    return manifest
