"""Solar PV conversion: ERA5-only and hybrid (SARAH-3 + ERA5) irradiance builders, atlite CSi baseline, CF cubes.

Units: site capacity in MWp, modelled output in MW (= CF x MWp), energy in MWh (= MW x 1 h), capacity factor as a fraction.
The PV model is atlite's `convert_pv` (Huld 2010 `CSi` panel, inverter efficiency 0.90 inside the panel config, fixed
orientation slope 40 deg / azimuth 180 deg = south) applied to an in-memory cutout; nothing is downloaded.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from . import config as C
from . import simulate
from .sarah_regrid import SOURCE_CODES, SRC_PARTIAL, SRC_SATELLITE, SarahOnGrid

IRRADIANCE_VARS = ("influx_toa", "influx_direct", "influx_diffuse", "albedo", "temperature", "solar_altitude", "solar_azimuth")
MODES = ("era5", "hybrid")
CF_TOLERANCE = 1e-9


class PvError(ValueError):
    """The irradiance input or the PV result violates a contract."""


# --- panel / orientation ---------------------------------------------------------------------------------------------
def panel_config() -> dict[str, Any]:
    from atlite.resource import get_solarpanelconfig

    return dict(get_solarpanelconfig(C.PV_PANEL))


def panel_record() -> dict[str, Any]:
    """Panel and orientation as used (for manifests / API headers); the inverter efficiency is read from atlite, not assumed."""
    import atlite

    pc = panel_config()
    eff = pc.get("inverter_efficiency")
    if eff is None or abs(float(eff) - C.PV_INVERTER_EFFICIENCY) > 1e-12:
        raise PvError(f"atlite {C.PV_PANEL} inverter_efficiency is {eff!r}, the report assumes {C.PV_INVERTER_EFFICIENCY}")
    return {"panel": C.PV_PANEL, "model": pc.get("model"), "source": pc.get("source"), "inverter_efficiency": float(eff),
            "inverter_efficiency_verified_in_atlite": True, "slope_deg": C.PV_SLOPE_DEG, "azimuth_deg": C.PV_AZIMUTH_DEG,
            "orientation_note": "azimuth 180 = south; slope 40 deg equals atlite's latitude_optimal rule above 50 N",
            "temperature_unit": "K (CSi: r_tmod 298 K, c_temp_amb 1)", "atlite_version": atlite.__version__,
            "tracking": None, "trigon_model": "simple", "clearsky_model": "not used (direct and diffuse provided)"}


def orientation() -> dict[str, float]:
    return {"slope": C.PV_SLOPE_DEG, "azimuth": C.PV_AZIMUTH_DEG}


# --- irradiance builders -------------------------------------------------------------------------------------------------
def _check_irradiance(ds: xr.Dataset) -> None:
    missing = [v for v in IRRADIANCE_VARS if v not in ds]
    if missing:
        raise PvError(f"irradiance dataset lacks {missing}")
    for v in IRRADIANCE_VARS:
        if set(ds[v].dims) != {"time", "y", "x"}:
            raise PvError(f"{v} has dims {ds[v].dims}, expected (time, y, x)")


def era5_irradiance(hourly: xr.Dataset) -> xr.Dataset:
    """ERA5-only input: the hour-start dataset of `era5_solar.build_hourly` (direct = fdir, diffuse = ssrd - fdir)."""
    _check_irradiance(hourly)
    out = hourly[list(IRRADIANCE_VARS)].copy()
    out.attrs = {**hourly.attrs, "irradiance_mode": "era5"}
    return out


def _check_alignment(hourly: xr.Dataset, sarah: SarahOnGrid) -> None:
    t = pd.DatetimeIndex(hourly["time"].values)
    if len(t) != len(sarah.hours) or not t.equals(pd.DatetimeIndex(sarah.hours)):
        raise PvError("SARAH hours and ERA5 hours differ (both must be the same hour-start axis)")
    if not (np.allclose(hourly["y"].values, sarah.lat, atol=1e-6) and np.allclose(hourly["x"].values, sarah.lon, atol=1e-6)):
        raise PvError("SARAH and ERA5 solar grids differ")


def hybrid_irradiance(hourly: xr.Dataset, sarah: SarahOnGrid) -> xr.Dataset:
    """Direct = SID and diffuse = SIS - SID of SARAH-3 on cell-hours covered >= threshold, ERA5 elsewhere.

    TOA, temperature, albedo and solar position stay ERA5 in both modes. `source_code` (uint8, see
    sarah_regrid.SOURCE_CODES) records satellite use or the fallback reason per cell-hour.
    """
    _check_irradiance(hourly)
    _check_alignment(hourly, sarah)
    covered = sarah.source_code == SRC_SATELLITE
    use = covered & np.isfinite(sarah.direct) & np.isfinite(sarah.diffuse)
    # a covered cell-hour without finite satellite values cannot happen by construction; keep it visible if it does
    code = np.where(use, SRC_SATELLITE, np.where(covered, SRC_PARTIAL, sarah.source_code)).astype(np.uint8)
    out = hourly[list(IRRADIANCE_VARS)].copy()
    dims = ("time", "y", "x")
    out["influx_direct"] = (dims, np.where(use, sarah.direct, hourly["influx_direct"].values))
    out["influx_diffuse"] = (dims, np.where(use, sarah.diffuse, hourly["influx_diffuse"].values))
    out["source_code"] = (dims, code)
    out["source_code"].attrs["codes"] = {str(k): v for k, v in SOURCE_CODES.items()}
    out.attrs = {**hourly.attrs, "irradiance_mode": "hybrid", "satellite_min_area_fraction": sarah.threshold}
    return out


def hybrid_diagnostics(irr: xr.Dataset) -> dict[str, Any]:
    """Counts for the manifest: satellite cell-hours, fallback reasons, satellite GHI above ERA5 TOA."""
    code = irr["source_code"].values
    ghi = irr["influx_direct"].values + irr["influx_diffuse"].values
    sat = code == SRC_SATELLITE
    return {"cell_hours": int(code.size), "satellite_cell_hours": int(sat.sum()), "satellite_share": float(sat.mean()),
            "reasons": {SOURCE_CODES[int(k)]: int((code == k).sum()) for k in np.unique(code)},
            "satellite_ghi_above_era5_toa_cell_hours": int((sat & (ghi > irr["influx_toa"].values + 1e-6)).sum())}


# --- conversion ----------------------------------------------------------------------------------------------------------
def _cutout_dataset(irr: xr.Dataset) -> xr.Dataset:
    ds = irr[list(IRRADIANCE_VARS)].copy()
    ds = ds.assign_coords(lon=("x", ds["x"].values), lat=("y", ds["y"].values))   # atlite cutouts carry lon/lat aliases
    ds.attrs.update(module="era5", prepared_features=["influx", "temperature"],
                    dx=float(np.round(np.median(np.diff(ds["x"].values)), 6)), dy=float(np.round(np.median(np.diff(ds["y"].values)), 6)))
    return ds.sortby(["y", "x", "time"])


def capacity_factors(irr: xr.Dataset) -> xr.DataArray:
    """Hourly PV capacity factor per cell, dims (time, latitude, longitude), hour-start UTC.

    atlite's `cutout.pv` reads only the variables it needs: this cutout has no wind variable and none is requested. Night
    (sun below atlite's 1 deg threshold or zero direct + diffuse irradiance) is exactly 0 in atlite; it is enforced again
    here so that a stray tiny negative value can never be reported.
    """
    _check_irradiance(irr)
    cutout = simulate.build_cutout(_cutout_dataset(irr))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            cf = cutout.pv(panel=C.PV_PANEL, orientation=orientation(), aggregate_time=None, show_progress=False)
        except TypeError:   # atlite < 0.5 has no aggregate_time
            cf = cutout.pv(panel=C.PV_PANEL, orientation=orientation(), capacity_factor_timeseries=True, show_progress=False)
    cf = cf.transpose("time", "y", "x").reindex(time=irr["time"].values, y=irr["y"].values, x=irr["x"].values)
    dark = ((irr["influx_direct"] + irr["influx_diffuse"]) <= 0).transpose(*cf.dims)
    cf = cf.where(~dark.values, 0.0).clip(min=0.0).astype(np.float64)
    cf = cf.rename({"y": "latitude", "x": "longitude"})
    cf.name = "capacity_factor"
    cf.attrs = {"units": "1", "long_name": "PV capacity factor (fraction of nominal power)", "panel": C.PV_PANEL,
                "slope_deg": C.PV_SLOPE_DEG, "azimuth_deg": C.PV_AZIMUTH_DEG, "irradiance_mode": irr.attrs.get("irradiance_mode", "unknown")}
    return cf


def check_cf(cf: xr.DataArray, solar_altitude: "xr.DataArray | None" = None) -> dict[str, Any]:
    """Hard checks: finite, within [0, 1], exactly 0 while the sun is below the horizon. Returns a summary."""
    v = cf.values
    out: dict[str, Any] = {"n": int(v.size), "n_nan": int(np.isnan(v).sum()), "min": float(np.nanmin(v)), "max": float(np.nanmax(v)),
                           "mean": float(np.nanmean(v))}
    if out["n_nan"]:
        raise PvError(f"{out['n_nan']} NaN capacity factor value(s)")
    if out["min"] < 0.0 or out["max"] > 1.0 + CF_TOLERANCE:
        raise PvError(f"capacity factor outside [0, 1]: min {out['min']:.6g}, max {out['max']:.6g}")
    if solar_altitude is not None:
        night = solar_altitude.values <= 0.0
        bad = int((v[night] != 0.0).sum()) if night.any() else 0
        out.update(night_values=int(night.sum()), night_nonzero=bad)
        if bad:
            raise PvError(f"{bad} night-time capacity factor value(s) are not exactly 0")
    return out


@dataclass
class CfCubes:
    era5: xr.DataArray
    hybrid: "xr.DataArray | None"
    irradiance: dict[str, xr.Dataset]

    def get(self, mode: str) -> xr.DataArray:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        cube = self.era5 if mode == "era5" else self.hybrid
        if cube is None:
            raise PvError("hybrid capacity factors are not available (no SARAH-3 input)")
        return cube


def _alt(irr: xr.Dataset) -> xr.DataArray:
    return irr["solar_altitude"].transpose("time", "y", "x")


def compute_cubes(hourly: xr.Dataset, sarah: "SarahOnGrid | None" = None) -> CfCubes:
    """CF cubes (time, latitude, longitude) for ERA5-only and, if SARAH-3 on the grid is given, hybrid."""
    irr_e = era5_irradiance(hourly)
    cube_e = capacity_factors(irr_e)
    check_cf(cube_e, _alt(irr_e))
    if sarah is None:
        return CfCubes(cube_e, None, {"era5": irr_e})
    irr_h = hybrid_irradiance(hourly, sarah)
    cube_h = capacity_factors(irr_h)
    check_cf(cube_h, _alt(irr_h))
    return CfCubes(cube_e, cube_h, {"era5": irr_e, "hybrid": irr_h})


# --- units ---------------------------------------------------------------------------------------------------------------
def _capacity(mwp: float) -> float:
    if not np.isfinite(mwp) or mwp <= 0:
        raise ValueError("capacity must be a positive number of MWp")
    return float(mwp)


def power_mw(cf: Any, capacity_mwp: float) -> Any:
    """Modelled output in MW = capacity factor (fraction) x site capacity (MWp)."""
    return np.asarray(cf, dtype=np.float64) * _capacity(capacity_mwp)


def energy_mwh(power: Any, hours: float = 1.0) -> Any:
    """Energy in MWh = MW x hours (hourly values: x 1 h)."""
    return np.asarray(power, dtype=np.float64) * hours


def capacity_factor_from_power(power: Any, capacity_mwp: float) -> Any:
    return np.asarray(power, dtype=np.float64) / _capacity(capacity_mwp)


# --- per-cell lookup -----------------------------------------------------------------------------------------------------
class OutsideGridError(ValueError):
    """The point lies outside the solar grid (outside the outer cell edges)."""


@dataclass(frozen=True)
class CellRef:
    i_lat: int
    i_lon: int
    lat: float
    lon: float
    lat_bounds: "tuple[float, float]"
    lon_bounds: "tuple[float, float]"


def locate_cell(lat: float, lon: float, grid: C.GridSpec = C.FINLAND_SOLAR) -> CellRef:
    """The ERA5 cell whose bounds (centre +- res/2) contain the point; a shared edge belongs to the northern / eastern cell."""
    half = grid.res / 2
    if not (np.isfinite(lat) and np.isfinite(lon)):
        raise OutsideGridError("latitude and longitude must be finite numbers")
    lo_lat, hi_lat, lo_lon, hi_lon = grid.lat_min - half, grid.lat_max + half, grid.lon_min - half, grid.lon_max + half
    if not (lo_lat <= lat <= hi_lat and lo_lon <= lon <= hi_lon):
        raise OutsideGridError(f"({lat}, {lon}) is outside the grid {grid.name}: lat {lo_lat}..{hi_lat}, lon {lo_lon}..{hi_lon}")
    i = min(int(np.floor((lat - lo_lat) / grid.res + 1e-9)), grid.n_lat - 1)
    j = min(int(np.floor((lon - lo_lon) / grid.res + 1e-9)), grid.n_lon - 1)
    c_lat, c_lon = grid.lat_min + i * grid.res, grid.lon_min + j * grid.res
    return CellRef(i, j, c_lat, c_lon, (c_lat - half, c_lat + half), (c_lon - half, c_lon + half))


def cell_series(cf: xr.DataArray, lat: float, lon: float, grid: C.GridSpec = C.FINLAND_SOLAR) -> "tuple[CellRef, pd.Series]":
    """Hourly CF series (fraction) of the cell containing (lat, lon), indexed by hour-start UTC."""
    ref = locate_cell(lat, lon, grid)
    vals = cf.isel(latitude=ref.i_lat, longitude=ref.i_lon).values
    return ref, pd.Series(vals, index=pd.DatetimeIndex(cf["time"].values, name="time_utc"), name="capacity_factor")
