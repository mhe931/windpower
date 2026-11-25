"""Nine-archetype wind simulation (notebook 1) plus wind/power-curve helpers shared with the site API.

Units: power in MW, energy in MWh (= MW x 1 h for hourly data), capacity factor = MW / installed MW.
Two interchangeable engines: `atlite` (a Cutout built from the validated ERA5 data, no download) and a
`direct` NumPy implementation of the same conversion; tests check they agree.
"""

from __future__ import annotations

import logging
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from . import config as C

log = logging.getLogger("windpower.simulate")

V_GRID = np.linspace(0.0, 30.0, 61)
CUT_IN, CUT_OUT = 3.0, 25.0
REF_HEIGHT_M = 100.0
MIN_ROUGHNESS_M = 1e-4


# --- helpers shared with the site API -----------------------------------------------
def effective_wind_speed(speed_100: Any, air_density: Any) -> Any:
    """IEC 61400-12-1 style density normalisation: v * clip(rho/1.225, 0.7, 1.3)^(1/3)."""
    return np.asarray(speed_100) * np.clip(np.asarray(air_density) / C.RHO_0, 0.7, 1.3) ** (1.0 / 3.0)


def hub_height_speed(wnd_100: Any, roughness: Any, hub_height_m: float, ref_height_m: float = REF_HEIGHT_M) -> Any:
    """Logarithmic extrapolation (atlite's method): v_hub = v_ref * ln(h/z0) / ln(h_ref/z0).

    Roughness is clipped to >= 1e-4 m; the result is floored at 0 for heights at or below z0.
    """
    z0 = np.maximum(np.asarray(roughness, dtype=float), MIN_ROUGHNESS_M)
    ratio = np.log(max(hub_height_m, 1e-9) / z0) / np.log(ref_height_m / z0)
    return np.maximum(np.asarray(wnd_100) * ratio, 0.0)


def build_power_curve(rated_mw: float, specific_power_wm2: float, hub_height_m: float) -> dict[str, Any]:
    """61-point (0..30 m/s, 0.5 m/s) aerodynamic curve: P = min(P_rated, 0.5 rho A Cp v^3) for 3 <= v <= 25 m/s.

    A = P_rated / clip(SP, 170, 450); Cp = clip(0.50 - 0.00018 SP, 0.40, 0.48) with the unclipped SP.
    Returns the atlite turbine-config keys V, POW (MW), hub_height (m) and P (MW).
    """
    area = rated_mw * 1e6 / np.clip(specific_power_wm2, *C.SP_CLIP)
    cp = np.clip(0.50 - 0.00018 * specific_power_wm2, 0.40, 0.48)
    aero = 0.5 * C.RHO_0 * area * cp * V_GRID**3 / 1e6
    pow_mw = np.where((V_GRID >= CUT_IN) & (V_GRID <= CUT_OUT), np.minimum(rated_mw, aero), 0.0)
    return {"V": V_GRID.copy(), "POW": pow_mw, "hub_height": float(hub_height_m), "P": float(rated_mw)}


def power_from_speed(curve: Mapping[str, Any], speed: Any, per_unit: bool = False) -> Any:
    """Piecewise-linear power-curve lookup (MW, or MW/MW_rated if per_unit); 0 beyond the last point."""
    out = np.interp(np.asarray(speed, dtype=float), curve["V"], curve["POW"], right=0.0)
    return out / curve["P"] if per_unit else out


def capacity_factor(power_mw: Any, installed_mw: float) -> Any:
    return np.asarray(power_mw) / installed_mw


def energy_mwh(power_mw: Any, hours: float = 1.0) -> Any:
    return np.asarray(power_mw) * hours


# --- ERA5 -> atlite-style dataset ---------------------------------------------------
def prepare_wind_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Density-adjusted 100 m speed and clipped roughness on (time, y, x), ascending axes.

    `ds` must be canonical and carry the engineered `speed_100` and `air_density_kgm3`.
    """
    dims = ("valid_time", "latitude", "longitude")
    w = effective_wind_speed(ds["speed_100"].transpose(*dims).values, ds["air_density_kgm3"].transpose(*dims).values)
    z0 = np.maximum(ds["fsr"].transpose(*dims).values, MIN_ROUGHNESS_M)
    out = xr.Dataset(
        {"wnd100m": (("time", "y", "x"), w), "roughness": (("time", "y", "x"), z0)},
        coords={"time": pd.to_datetime(ds["valid_time"].values), "y": ds["latitude"].values.astype(np.float64),
                "x": ds["longitude"].values.astype(np.float64)},
    ).sortby(["y", "x", "time"])
    out.attrs.update(module="era5", prepared_features=["wind"],
                     dx=float(np.round(np.median(np.diff(out["x"].values)), 6)),
                     dy=float(np.round(np.median(np.diff(out["y"].values)), 6)))
    return out


def build_cutout(wind_ds: xr.Dataset, tmp_dir: "str | None" = None) -> Any:
    """atlite.Cutout wrapping an in-memory dataset (offline: nothing is downloaded or written)."""
    import atlite

    path = os.path.join(tmp_dir or tempfile.gettempdir(), "windpower_cutout_unused.nc")
    return atlite.Cutout(path=path, data=wind_ds)


# --- archetypes -----------------------------------------------------------------------
@dataclass
class LayoutReport:
    """Nearest-centre layout bookkeeping for one archetype (the simulation quirk vs `within`)."""

    turbines: int
    capacity_mw: float
    outside_cell_box_turbines: int      # nearest centre is >= half a cell away in lat or lon (edge / outside grid)
    outside_cell_box_mw: float
    outside_grid_turbines: int          # beyond the outer cell edges of the grid
    outside_grid_mw: float


def nearest_centre_layout(lat: np.ndarray, lon: np.ndarray, cap_mw: np.ndarray, lats: np.ndarray, lons: np.ndarray
                          ) -> "tuple[np.ndarray, LayoutReport]":
    """Capacity (MW) per cell by argmin distance to cell centres (notebook behaviour).

    Turbines on a cell edge or outside the grid are still assigned to the nearest cell; they are
    counted in the report and logged so the quirk stays visible.
    """
    iy = np.abs(lats[:, None] - lat[None, :]).argmin(axis=0)
    ix = np.abs(lons[:, None] - lon[None, :]).argmin(axis=0)
    mat = np.zeros((lats.size, lons.size))
    np.add.at(mat, (iy, ix), cap_mw)
    half = C.GRID_RES_DEG / 2
    off_box = (np.abs(lat - lats[iy]) >= half) | (np.abs(lon - lons[ix]) >= half)
    off_grid = (lat <= lats.min() - half) | (lat >= lats.max() + half) | (lon <= lons.min() - half) | (lon >= lons.max() + half)
    rep = LayoutReport(len(lat), float(cap_mw.sum()), int(off_box.sum()), float(cap_mw[off_box].sum()),
                       int(off_grid.sum()), float(cap_mw[off_grid].sum()))
    return mat, rep


@dataclass
class SimulationResult:
    hourly: pd.DataFrame                   # index valid_time (UTC naive); one MW column per archetype + theoretical_wind_mw
    archetypes: pd.DataFrame               # notebook summary table (+ rated/curve columns)
    curves: dict[str, dict[str, Any]]      # archetype -> 61-point curve (V, POW, hub_height, P) + sp, installed_mw
    layout: dict[str, LayoutReport]
    installed_mw: float
    engine: str
    notes: list[str] = field(default_factory=list)

    @property
    def theoretical_mw(self) -> pd.Series:
        return self.hourly["theoretical_wind_mw"]


def archetype_groups(specs: pd.DataFrame) -> "list[tuple[str, pd.DataFrame]]":
    """(name, turbines) per non-empty SP x hub group in notebook order (SP outer, hub inner)."""
    out = []
    for (sp_grp, hub_grp), grp in specs.groupby(["sp_bin", "hub_bin"], observed=False):
        if not grp.empty:
            out.append((f"{sp_grp}_{hub_grp}", grp))
    return out


def _simulate_direct(wind_ds: xr.Dataset, curve: Mapping[str, Any], layout_mw: np.ndarray) -> np.ndarray:
    """Same conversion as atlite.convert_wind + layout sum, in NumPy: returns MW per time step."""
    w = hub_height_speed(wind_ds["wnd100m"].values, wind_ds["roughness"].values, curve["hub_height"])
    cf = power_from_speed(curve, w, per_unit=True)
    return np.einsum("tyx,yx->t", cf, layout_mw)


def _simulate_atlite(cutout: Any, curve: Mapping[str, Any], layout_mw: np.ndarray, wind_ds: xr.Dataset) -> np.ndarray:
    layout = xr.DataArray(layout_mw, coords={"y": wind_ds["y"].values, "x": wind_ds["x"].values}, dims=["y", "x"],
                          name="installed_capacity_mw")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        gen = cutout.wind(turbine=dict(curve), layout=layout, per_unit=False, show_progress=False)
    return np.asarray(gen.squeeze().values, dtype=np.float64).ravel()


def simulate_archetypes(ds: xr.Dataset, specs: pd.DataFrame, engine: str = "atlite") -> SimulationResult:
    """Hourly national theoretical MW per archetype and in total.

    `ds`: canonical ERA5 with engineered features (see era5.engineer_features); `specs`: build_turbine_spec_table().
    """
    if engine not in ("atlite", "direct"):
        raise ValueError("engine must be 'atlite' or 'direct'")
    wind_ds = prepare_wind_dataset(ds)
    lats, lons = wind_ds["y"].values, wind_ds["x"].values
    index = pd.DatetimeIndex(wind_ds["time"].values, name="valid_time")
    cutout = build_cutout(wind_ds) if engine == "atlite" else None
    hourly = pd.DataFrame(index=index)
    total = np.zeros(len(index))
    rows: list[dict[str, Any]] = []
    curves: dict[str, dict[str, Any]] = {}
    layouts: dict[str, LayoutReport] = {}
    for name, grp in archetype_groups(specs):
        layout_mw, rep = nearest_centre_layout(grp["lat"].to_numpy(float), grp["lon"].to_numpy(float),
                                               grp["cap_mw"].to_numpy(float), lats, lons)
        mean_rated, mean_sp, mean_hub = (float(grp["cap_mw"].mean()), float(grp["specific_power_wm2"].mean()),
                                         float(grp["hub_m"].mean()))
        curve = build_power_curve(mean_rated, mean_sp, mean_hub)
        gen = (_simulate_atlite(cutout, curve, layout_mw, wind_ds) if engine == "atlite"
               else _simulate_direct(wind_ds, curve, layout_mw))
        hourly[name] = gen
        total += gen
        if rep.outside_cell_box_turbines:
            log.warning("%s: %d turbine(s) (%.1f MW) lie outside their nearest cell's box (edge/outside grid); "
                        "nearest-centre layout still assigns them (%d beyond the grid edge)",
                        name, rep.outside_cell_box_turbines, rep.outside_cell_box_mw, rep.outside_grid_turbines)
        curves[name] = {**curve, "specific_power_wm2": mean_sp, "installed_mw": float(grp["cap_mw"].sum())}
        layouts[name] = rep
        rows.append({"archetype": name, "turbines": len(grp), "installed_mw": round(float(grp["cap_mw"].sum()), 1),
                     "mean_hub_m": round(mean_hub, 1), "mean_sp_wm2": round(mean_sp, 1),
                     "mean_gen_mw": round(float(gen.mean()), 1), "mean_rated_mw": mean_rated})
    hourly["theoretical_wind_mw"] = total
    installed = float(specs["cap_mw"].sum())
    notes = []
    off = sum(r.outside_cell_box_turbines for r in layouts.values())
    if off:
        notes.append(f"{off} turbine(s) not strictly inside their nearest cell box were still placed by argmin")
    return SimulationResult(hourly, pd.DataFrame(rows), curves, layouts, installed, engine, notes)


# --- aggregation ------------------------------------------------------------------------
def daily_summary(df: pd.DataFrame, theoretical: str = "theoretical_wind_mw", actual: "str | None" = "actual_wind_mw",
                  price: "str | None" = "imbalance_price_eur_mwh") -> pd.DataFrame:
    """Per UTC day: hours, GWh (= sum(MW x 1 h)/1000), mean MW and (if available) capture ratio %.

    `hours` counts hours with a non-missing actual value when `actual` is given, else all hours.
    """
    g = df.resample("1D")
    out = pd.DataFrame({
        "hours": g[actual].count() if actual and actual in df else g[theoretical].count(),
        "theoretical_gwh": g[theoretical].sum() / 1000.0,
        "mean_theoretical_mw": g[theoretical].mean(),
    })
    if actual and actual in df:
        out["actual_gwh"] = g[actual].sum() / 1000.0
        out["residual_gwh"] = out["theoretical_gwh"] - out["actual_gwh"]
        out["mean_actual_mw"] = g[actual].mean()
        out["capture_ratio_pct"] = 100.0 * out["actual_gwh"] / out["theoretical_gwh"]
    if price and price in df:
        out["mean_imbalance_eur"] = g[price].mean()
    return out
