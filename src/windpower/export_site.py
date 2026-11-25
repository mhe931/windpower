"""Versioned compact artifact for the website: `<outputs>/site/v1/`.

Files (all labelled with model version, units, run mode and the SYNTHETIC/DEMO flag):
  manifest.json            what is in the artifact, provenance, coverage, licences, file hashes
  grid.json                axes, cell size, supported-cell mask, fleet capacity per cell
  archetypes.json          the archetype specs with their 61-point power curves
  wind_effective.bin       float16 [time, lat, lon] density-adjusted 100 m wind speed (m/s), C order, little endian
  wind_effective.meta.json shape/dtype/axes/units of the .bin and of the roughness array
  roughness_mean.bin       float16 [lat, lon] time-mean roughness length (m)
  national_hourly.json     UTC hours, theoretical MW, observed MW (null where missing), coverage flags

No national calibration is applied to the site-level simulation: clients convert wind to power with the
archetype curves and the log profile (v_hub = v_eff ln(h/z0)/ln(100/z0)) only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from . import __version__
from . import calibrate as K
from . import config as C
from . import simulate as S
from .fleet import FleetResult
from .io_utils import atomic_write_bytes, atomic_write_json, iso_utc, sha256_file, utc_now_iso
from .manifest import licence_block

ARTIFACT_SCHEMA_VERSION = "1.0"
SITE_DIR = Path("site") / "v1"
MAX_BYTES = 15_000_000


class ExportSizeError(RuntimeError):
    pass


def estimate_wind_bin_bytes(n_hours: int, n_lat: int, n_lon: int) -> int:
    return n_hours * n_lat * n_lon * 2


def site_label(run_mode: str, synthetic: bool) -> str:
    if run_mode == "demo":
        return "DEMO / SYNTHETIC - not real data"
    return "SYNTHETIC - not real data" if synthetic else "Retrospective ERA5 simulation"


def _meta(run_mode: str, synthetic: bool, units: Mapping[str, str], what: str) -> dict[str, Any]:
    return {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "model_version": f"windpower-{__version__}", "run_mode": run_mode,
            "synthetic": synthetic, "demo": run_mode == "demo", "label": site_label(run_mode, synthetic), "units": dict(units),
            "content": what}


def export_site(
    out_root: "str | Path", *, ds: xr.Dataset, sim: S.SimulationResult, joined: pd.DataFrame, fleet: FleetResult,
    calib: "K.CalibrationResult | None", run_mode: str, synthetic: bool, provenance: "Mapping[str, Any] | None" = None,
    arcgis_meta: "Mapping[str, Any] | None" = None,
) -> dict[str, Any]:
    """Write site/v1 under `out_root` and return its manifest. Raises ExportSizeError above 15 MB."""
    out = Path(out_root) / SITE_DIR
    out.mkdir(parents=True, exist_ok=True)
    wind = S.prepare_wind_dataset(ds)
    lats, lons = wind["y"].values, wind["x"].values
    t_index = pd.DatetimeIndex(wind["time"].values)
    n_t, n_y, n_x = wind["wnd100m"].shape
    est = estimate_wind_bin_bytes(n_t, n_y, n_x)
    if est > MAX_BYTES:
        raise ExportSizeError(f"wind_effective.bin would be {est / 1e6:.1f} MB (> {MAX_BYTES / 1e6:.0f} MB limit)")
    unit_all = {"wind_speed": "m s-1", "roughness": "m", "power": "MW", "capacity": "MW", "energy": "MWh", "capacity_factor": "1"}
    first, last = iso_utc(t_index[0]), iso_utc(t_index[-1])
    flags = _meta(run_mode, synthetic, unit_all, "")

    # wind field
    w16 = wind["wnd100m"].values.astype("<f2")
    if not np.isfinite(w16).all():
        raise ValueError("non-finite wind speed after float16 conversion")
    atomic_write_bytes(out / "wind_effective.bin", w16.tobytes(order="C"))
    z16 = wind["roughness"].mean("time").values.astype("<f2")
    atomic_write_bytes(out / "roughness_mean.bin", z16.tobytes(order="C"))
    atomic_write_json(out / "wind_effective.meta.json", {
        **_meta(run_mode, synthetic, {"wind_effective": "m s-1", "roughness": "m"}, "binary arrays; see 'arrays'"),
        "arrays": {
            "wind_effective": {"file": "wind_effective.bin", "dtype": "float16", "byte_order": "little", "order": "C",
                               "shape": [n_t, n_y, n_x], "axes": ["time", "lat", "lon"], "units": "m s-1",
                               "description": "100 m wind speed adjusted to standard air density: v*clip(rho/1.225,0.7,1.3)^(1/3)"},
            "roughness_mean": {"file": "roughness_mean.bin", "dtype": "float16", "byte_order": "little", "order": "C",
                               "shape": [n_y, n_x], "axes": ["lat", "lon"], "units": "m",
                               "description": "time-mean forecast surface roughness length, clipped to >= 1e-4 m"}},
        "time": {"first_utc": first, "last_utc": last, "step": "1h", "count": n_t},
        "lat_order": "ascending", "lon_order": "ascending",
        "hub_height_extrapolation": "v_hub = v_eff * ln(h/z0) / ln(100/z0), z0 = roughness_mean (or hourly roughness if added later)",
        "calibration_applied": False})

    # grid
    cap = fleet.capacity_matrix
    atomic_write_json(out / "grid.json", {
        **_meta(run_mode, synthetic, {"lat": "degrees_north", "lon": "degrees_east", "fleet_capacity": "MW"}, "grid axes and masks"),
        "lats": lats.tolist(), "lons": lons.tolist(), "cell_size_deg": C.GRID_RES_DEG, "shape": [n_y, n_x],
        "lat_order": "ascending", "lon_order": "ascending", "index_order": "[lat_index][lon_index]",
        "supported": np.ones((n_y, n_x), dtype=int).tolist(),
        "supported_definition": "all ERA5 cells with finite data; no land/sea or border mask is applied",
        "fleet_capacity_mw": np.round(cap, 3).tolist(), "fleet_capacity_total_mw": float(np.round(cap.sum(), 3)),
    })

    # archetypes
    arch = []
    for _, row in sim.archetypes.iterrows():
        c = sim.curves[row["archetype"]]
        arch.append({"name": row["archetype"], "turbines": int(row["turbines"]), "installed_mw": float(c["installed_mw"]),
                     "mean_rated_mw": float(c["P"]), "mean_hub_height_m": float(c["hub_height"]),
                     "mean_specific_power_wm2": float(c["specific_power_wm2"]),
                     "power_curve": {"v_ms": np.round(c["V"], 3).tolist(), "power_mw": np.round(c["POW"], 6).tolist(),
                                     "power_per_unit": np.round(c["POW"] / c["P"], 6).tolist()}})
    atomic_write_json(out / "archetypes.json", {
        **_meta(run_mode, synthetic, {"power_curve.v_ms": "m s-1", "power_curve.power_mw": "MW per turbine of mean rated size",
                                       "power_curve.power_per_unit": "1"}, "nine SP x hub archetypes with 61-point curves (0..30 m/s)"),
        "curve_points": 61, "cut_in_ms": S.CUT_IN, "cut_out_ms": S.CUT_OUT, "archetypes": arch,
        "specific_power_bins_wm2": [0, 265, 340, None], "hub_bins_m": [0, 100, 140, None]})

    # national hourly
    j = joined.reindex(t_index)
    obs = j["actual_wind_mw"]
    avail = obs.notna()
    nat: dict[str, Any] = {
        **_meta(run_mode, synthetic, {"theoretical_mw": "MW", "observed_mw": "MW", "inference_only_mw": "MW"}, "national hourly series, UTC hour start"),
        "utc_hours": [iso_utc(t) for t in t_index],
        "theoretical_mw": [round(float(v), 3) for v in sim.theoretical_mw.reindex(t_index)],
        "observed_mw": [None if np.isnan(v) else round(float(v), 3) for v in obs],
        "observed_available": avail.astype(int).tolist(),
        "installed_mw": float(sim.installed_mw),
        "coverage": {"era5_hours": int(len(t_index)), "observed_hours": int(avail.sum()),
                     "missing_observed_hours": [iso_utc(t) for t in t_index[~avail.to_numpy()]]},
    }
    if calib is not None:
        hours = t_index.hour
        pred = calib.models.inference.predict(sim.theoretical_mw.reindex(t_index).to_numpy(), hours)
        nat["inference_only_mw"] = [round(float(v), 3) for v in pred]
        nat["inference_only_note"] = ("website-facing candidate: observed-actual target, inputs = theoretical MW + hour of day only; "
                                      "trained on train blocks, see metrics for out-of-sample error")
    atomic_write_json(out / "national_hourly.json", nat)

    files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(out.iterdir())
             if p.is_file() and p.name != "manifest.json"]
    total = sum(f["bytes"] for f in files)
    if total > MAX_BYTES:
        raise ExportSizeError(f"site/v1 is {total / 1e6:.1f} MB (> {MAX_BYTES / 1e6:.0f} MB limit)")
    manifest = {
        **flags, "content": "artifact manifest", "generated_utc": utc_now_iso(),
        "grid": {"n_lat": n_y, "n_lon": n_x, "cell_size_deg": C.GRID_RES_DEG, "lat_range": [float(lats[0]), float(lats[-1])],
                 "lon_range": [float(lons[0]), float(lons[-1])]},
        "time_range": {"first_utc": first, "last_utc": last, "hours": n_t},
        "data_coverage": nat["coverage"] | {"theoretical_hours": n_t},
        "provenance": dict(provenance or {}),
        "calibration_applied_to_site_simulation": False,
        "licences": licence_block(synthetic, arcgis_meta),
        "files": files, "total_bytes": total,
        "attribution": "Contains modified Copernicus Climate Change Service information; (c) OpenStreetMap contributors"
                       if not synthetic else "SYNTHETIC data: nothing here is derived from Copernicus, OpenStreetMap, Fingrid or Finnish registry data",
    }
    atomic_write_json(out / "manifest.json", manifest)
    return manifest
