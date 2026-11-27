"""National PV: capacity allocation on the ERA5 solar cells, national power/energy, comparison metrics.

    P(t)[MW] = sum_c CF(t, c) * [ (C(t) - L) * w_c + L_c ]

C(t) = Fingrid 267 capacity (MW), L = capacity of the large plants inside the grid (250.66 MW expected), L_c = large capacity in
the ERA5 cell containing the centroid of the plant's municipality, w_c = cell share of the inhabitants inside the grid
(1 km cell centroid, EPSG:3067 -> WGS84 -> ERA5 cell bounds). The alternative "population only" distribution uses C(t) * w_c.
Hours without C(t) stay NaN (never filled); without Fingrid 267 no national MW exists and none is invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from . import config as C
from .sarah_regrid import SRC_SATELLITE, SarahOnGrid

RECONCILIATION_TOL_MW = 1e-9


class AllocationError(ValueError):
    """Capacity allocation is impossible or inconsistent."""


def cell_indices(lat: np.ndarray, lon: np.ndarray, grid: C.GridSpec = C.FINLAND_SOLAR) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """(i_lat, i_lon, inside) of the cell whose bounds contain each point; an edge belongs to the northern / eastern cell."""
    half = grid.res / 2
    lat, lon = np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)
    lo_lat, hi_lat, lo_lon, hi_lon = grid.lat_min - half, grid.lat_max + half, grid.lon_min - half, grid.lon_max + half
    inside = (lat >= lo_lat) & (lat <= hi_lat) & (lon >= lo_lon) & (lon <= hi_lon)
    i = np.minimum(np.floor((lat - lo_lat) / grid.res + 1e-9), grid.n_lat - 1).astype(int)
    j = np.minimum(np.floor((lon - lo_lon) / grid.res + 1e-9), grid.n_lon - 1).astype(int)
    return i, j, inside


def to_wgs84(x: np.ndarray, y: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    from pyproj import Transformer

    lon, lat = Transformer.from_crs("EPSG:3067", "EPSG:4326", always_xy=True).transform(np.asarray(x, float), np.asarray(y, float))
    return np.asarray(lat), np.asarray(lon)


@dataclass
class Allocation:
    grid: C.GridSpec
    plants: pd.DataFrame            # name, nominal_mw, municipality, lat, lon, i_lat, i_lon, inside_grid
    large_mw: float                 # L: capacity of the plants inside the grid (the formula's L)
    large_by_cell: np.ndarray       # (Y, X) L_c in MW
    pop_share: np.ndarray           # (Y, X) w_c, sums to 1 over the grid
    pop_by_cell: np.ndarray         # (Y, X) inhabitants
    report: dict[str, Any]


def allocate(plants_placed: pd.DataFrame, population: pd.DataFrame, grid: C.GridSpec = C.FINLAND_SOLAR) -> Allocation:
    """Build L_c and w_c. `plants_placed` needs name, nominal_mw, municipality, lat, lon (see solar_public.place_plants)."""
    pl = plants_placed.copy()
    i, j, inside = cell_indices(pl["lat"].to_numpy(), pl["lon"].to_numpy(), grid)
    pl["i_lat"], pl["i_lon"], pl["inside_grid"] = i, j, inside
    large = np.zeros((grid.n_lat, grid.n_lon))
    np.add.at(large, (i[inside], j[inside]), pl.loc[inside, "nominal_mw"].to_numpy(float))

    pop = population[population["inhabitants"] > 0]
    lat, lon = to_wgs84(pop["x"].to_numpy() + 500.0, pop["y"].to_numpy() + 500.0)       # x/y are SW corners of the 1 km cells
    pi, pj, pin = cell_indices(lat, lon, grid)
    inh = pop["inhabitants"].to_numpy(float)
    by_cell = np.zeros((grid.n_lat, grid.n_lon))
    np.add.at(by_cell, (pi[pin], pj[pin]), inh[pin])
    total_in = float(by_cell.sum())
    if total_in <= 0:
        raise AllocationError("no inhabitants inside the solar grid")
    share = by_cell / total_in
    big = pl.loc[pl["nominal_mw"].idxmax()] if len(pl) else None
    rep = {
        "plants": int(len(pl)), "plants_inside_grid": int(inside.sum()), "plants_outside_grid": pl.loc[~inside, ["name", "municipality", "nominal_mw"]].to_dict("records"),
        "large_total_mw": round(float(pl["nominal_mw"].sum()), 9), "large_inside_grid_mw": round(float(large.sum()), 9),
        "large_outside_grid_mw": round(float(pl.loc[~inside, "nominal_mw"].sum()), 9),
        "largest_plant": None if big is None else {"name": big["name"], "nominal_mw": float(big["nominal_mw"]), "municipality": big["municipality"],
                                                   "inside_grid": bool(big["inside_grid"])},
        "largest_plant_share_of_large_capacity": None if big is None else float(big["nominal_mw"] / pl["nominal_mw"].sum()),
        "population_cells": int(len(pop)), "population_cells_inside_grid": int(pin.sum()), "population_cells_outside_grid": int((~pin).sum()),
        "inhabitants_total": float(inh.sum()), "inhabitants_inside_grid": total_in, "inhabitants_outside_grid": float(inh[~pin].sum()),
        "cells_with_population": int((by_cell > 0).sum()), "cells_with_large_plants": int((large > 0).sum()), "weights_sum": float(share.sum()),
        "cell_assignment": "plants: centroid of the municipality (EPSG:3067 shapely centroid -> WGS84); population: centre of the 1 km cell",
    }
    return Allocation(grid, pl, float(large.sum()), large, share, by_cell, rep)


def capacity_cube(capacity_mw: pd.Series, alloc: Allocation, population_only: bool = False) -> np.ndarray:
    """Installed MW per hour and cell, (T, Y, X): (C - L) w_c + L_c, or C w_c when `population_only`. NaN hours stay NaN."""
    c = np.asarray(capacity_mw, dtype=np.float64)
    if population_only:
        return c[:, None, None] * alloc.pop_share[None]
    if np.isfinite(c).any() and np.nanmin(c) < alloc.large_mw:
        raise AllocationError(f"capacity C(t) falls below the large-plant capacity L = {alloc.large_mw:.2f} MW (min {np.nanmin(c):.2f})")
    return (c - alloc.large_mw)[:, None, None] * alloc.pop_share[None] + alloc.large_by_cell[None]


def check_reconciliation(cube: np.ndarray, capacity_mw: pd.Series, tol: float = RECONCILIATION_TOL_MW) -> dict[str, Any]:
    """Per hour: sum over cells == C(t) (NaN hours skipped and counted)."""
    c = np.asarray(capacity_mw, dtype=np.float64)
    ok = np.isfinite(c)
    diff = np.abs(cube[ok].sum(axis=(1, 2)) - c[ok]) if ok.any() else np.array([0.0])
    return {"hours_checked": int(ok.sum()), "hours_nan": int((~ok).sum()), "max_abs_diff_mw": float(diff.max()), "tolerance_mw": tol,
            "ok": bool(diff.max() <= tol)}


def national_power(cf: xr.DataArray, cube: np.ndarray) -> np.ndarray:
    """MW per hour = sum_c CF(t, c) * installed MW(t, c); NaN where the capacity is NaN."""
    return np.einsum("tyx,tyx->t", cf.transpose("time", "latitude", "longitude").values, cube)


# --- metrics --------------------------------------------------------------------------------------------------------------------
def energy_gwh(mw: "pd.Series | np.ndarray") -> dict[str, Any]:
    v = np.asarray(mw, dtype=np.float64)
    ok = np.isfinite(v)
    return {"gwh": float(v[ok].sum() / 1000.0) if ok.any() else None, "hours": int(ok.sum()), "hours_missing": int((~ok).sum())}


def pearson(a: pd.Series, b: pd.Series) -> dict[str, Any]:
    """Pearson correlation of hourly values on the common non-missing hours (NOT a regression R^2)."""
    df = pd.concat([a, b], axis=1).dropna()
    ok = len(df) > 2 and df.iloc[:, 0].std() > 0 and df.iloc[:, 1].std() > 0
    return {"pearson_r": float(np.corrcoef(df.iloc[:, 0], df.iloc[:, 1])[0, 1]) if ok else None, "n_hours": int(len(df)),
            "label": "Pearson correlation of hourly values (not regression R2)"}


def daily_cycle_centre(mw: pd.Series) -> dict[str, Any]:
    """Output-weighted mean hour of the mean daily cycle (UTC): mean per hour of day, weighted at the hour's mid-point."""
    s = mw.dropna()
    if s.empty or s.sum() <= 0:
        return {"minutes_utc": None, "hhmm_utc": None, "n_hours": int(len(s))}
    by = s.groupby(s.index.hour).mean()
    m = float(((by.index.to_numpy() + 0.5) * 60.0 * by.to_numpy()).sum() / by.sum())
    total = int(round(m))
    return {"minutes_utc": m, "hhmm_utc": f"{total // 60:02d}:{total % 60:02d}", "n_hours": int(len(s)),
            "rule": "mean daily cycle weighted at hour mid-points (h + 30 min), UTC"}


def mean_cf(cf: xr.DataArray, mask: "np.ndarray | None" = None) -> float:
    v = cf.transpose("time", "latitude", "longitude").values
    return float(v[:, mask].mean()) if mask is not None else float(v.mean())


def ghi_table(era5_hourly: xr.Dataset, sarah: SarahOnGrid) -> dict[str, Any]:
    """Table-2 style means over the satellite-covered cell-hours: SARAH vs ERA5 GHI, direct, diffuse (W m-2)."""
    sat = (sarah.source_code == SRC_SATELLITE) & sarah.fully_covered[None]
    if not sat.any():
        return {"cell_hours": 0}
    e_dir = era5_hourly["influx_direct"].transpose("time", "y", "x").values
    e_dif = era5_hourly["influx_diffuse"].transpose("time", "y", "x").values
    out = {"cell_hours": int(sat.sum()), "cells": int(sarah.fully_covered.sum()),
           "sarah_ghi_wm2": float(sarah.ghi[sat].mean()), "era5_ghi_wm2": float((e_dir + e_dif)[sat].mean()),
           "sarah_direct_wm2": float(sarah.direct[sat].mean()), "era5_direct_wm2": float(e_dir[sat].mean()),
           "sarah_diffuse_wm2": float(sarah.diffuse[sat].mean()), "era5_diffuse_wm2": float(e_dif[sat].mean())}
    out["sarah_daily_ghi_kwh_m2"] = out["sarah_ghi_wm2"] * 24 / 1000.0
    out["era5_daily_ghi_kwh_m2"] = out["era5_ghi_wm2"] * 24 / 1000.0
    return out


_REFS = {"forecast_248_mw": "fingrid_248", "surplus_362_mw": "fingrid_362_av06"}


def comparison_metrics(national: pd.DataFrame) -> dict[str, Any]:
    """Energy, ratios, Pearson r and daily-cycle centres of the model series vs the Fingrid references present in `national`."""
    out: dict[str, Any] = {"models": {}, "references": {}}
    refs = {c: lab for c, lab in _REFS.items() if c in national and national[c].notna().any()}
    for col in ("era5_only_mw", "hybrid_mw", "era5_only_popw_mw", "hybrid_popw_mw"):
        if col in national and national[col].notna().any():
            out["models"][col] = {"energy": energy_gwh(national[col]), "cycle_centre": daily_cycle_centre(national[col])}
    for col, label in refs.items():
        out["references"][label] = {"energy": energy_gwh(national[col]), "cycle_centre": daily_cycle_centre(national[col])}
    for col, m in out["models"].items():
        for col_ref, label in refs.items():
            common = national[[col, col_ref]].dropna()
            m[f"vs_{label}"] = {**pearson(national[col], national[col_ref]), "hours": int(len(common)),
                                "energy_ratio_model_over_reference": float(common[col].sum() / common[col_ref].sum()) if len(common) and common[col_ref].sum() else None,
                                "note": "surplus only: use for timing, not energy" if label == "fingrid_362_av06" else None}
    return out


def mean_cf_vs_capacity(power_mw: pd.Series, capacity_mw: pd.Series) -> "float | None":
    r = (power_mw / capacity_mw).replace([np.inf, -np.inf], np.nan).dropna()
    return float(r.mean()) if len(r) else None
