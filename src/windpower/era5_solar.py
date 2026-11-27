"""ERA5 solar inputs: raw CDS files or a prepared atlite-style dataset -> hour-start W m-2 fields for the PV model.

Time convention (solar report sec. 3.1): ERA5 radiation is accumulated over the hour ENDING at its stamp (J m-2), so the
value stamped h+1 belongs to hour h; 2 m temperature is instantaneous, the hour-h value is the mean of the stamps h and h+1;
the solar position is evaluated at h+30 min. Because a prepared dataset may already be hour-start, the convention is
DETECTED from the data (TOA irradiance vs. solar geometry) before anything is shifted, evidence is recorded, the shift is
applied at most once, and an ambiguous result raises `ConventionAmbiguousError` unless the user passes an override.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from . import config as C
from . import era5
from . import solar_geometry as G
from .io_utils import CredentialUnavailableError, atomic_write_with, to_utc_naive_index, utc_hour_range

TIME, LAT, LON = era5.TIME, era5.LAT, era5.LON
RAW_VARS = ("ssrd", "ssr", "fdir", "tisr", "t2m")
PREPARED_VARS = ("influx_toa", "influx_direct", "influx_diffuse", "albedo", "temperature")
HOURLY_VARS = (*PREPARED_VARS, "solar_altitude", "solar_azimuth")
CONVENTIONS = ("end", "start")
HINT_ATTR = "era5_time_convention"
J_TO_W = 1.0 / 3600.0
MAX_TOA_WM2 = 1500.0
_J_UNITS = {"j m**-2", "j m-2", "j/m2", "j m^-2", "j m**-2 "}
_W_UNITS = {"w m**-2", "w m-2", "w/m2", "w m^-2"}
CDS_SOLAR_VARIABLES = (
    "surface_solar_radiation_downwards", "surface_net_solar_radiation", "total_sky_direct_solar_radiation_at_surface",
    "toa_incident_solar_radiation", "2m_temperature",
)


class SolarEra5Error(ValueError):
    """The ERA5 solar input violates its contract; `check` names the failed check."""

    def __init__(self, check: str, message: str):
        super().__init__(f"[{check}] {message}")
        self.check = check
        self.detail = message


class ConventionAmbiguousError(SolarEra5Error):
    """The time convention (end-stamped vs hour-start) cannot be established from the data; pass --era5-convention."""

    def __init__(self, message: str, evidence: "Mapping[str, Any] | None" = None):
        super().__init__("time_convention", message)
        self.evidence = dict(evidence or {})


class WindowError(SolarEra5Error):
    """Required ERA5 stamps for the requested window are missing."""

    def __init__(self, message: str):
        super().__init__("window", message)


class UnitsError(SolarEra5Error):
    def __init__(self, message: str):
        super().__init__("units", message)


# --- opening ---------------------------------------------------------------------------------------------------------
def _canonical(ds: xr.Dataset) -> xr.Dataset:
    """valid_time/latitude/longitude names (atlite x/y/time included), ascending axes, de-duplicated time."""
    if "x" in ds.dims and "y" in ds.dims:
        ds = ds.drop_vars([c for c in ("lon", "lat") if c in ds.coords and c not in ds.dims]).rename({"x": LON, "y": LAT})
    return era5.canonicalise(ds)


def open_solar_source(path: "str | Path") -> xr.Dataset:
    """A NetCDF file, a CDS ZIP bundle (instant + accum members) or a directory of them -> one canonical dataset."""
    p = Path(path)
    if p.is_dir():
        files = sorted([*p.glob("*.nc"), *p.glob("*.zip")])
        if not files:
            raise FileNotFoundError(f"no .nc/.zip ERA5 solar files in {p}")
    elif p.is_file():
        files = [p]
    else:
        raise FileNotFoundError(f"ERA5 solar input not found: {p}")
    parts = [_canonical(era5.open_cds_file(f)) for f in files]
    groups: dict[frozenset[str], list[xr.Dataset]] = {}
    for part in parts:
        groups.setdefault(frozenset(part.data_vars), []).append(part)
    merged = [xr.concat(g, dim=TIME, data_vars="minimal", coords="minimal", compat="override", join="exact") if len(g) > 1 else g[0]
              for g in groups.values()]
    ds = merged[0] if len(merged) == 1 else xr.merge(merged, join="exact", compat="override")
    return _canonical(ds)


def classify(ds: xr.Dataset) -> str:
    """'raw' (CDS ssrd/ssr/fdir/tisr/t2m) or 'prepared' (atlite-style influx_*, albedo, temperature)."""
    have = set(ds.data_vars)
    if set(RAW_VARS) <= have:
        return "raw"
    if set(PREPARED_VARS) <= have:
        return "prepared"
    raise SolarEra5Error("variables", f"neither raw CDS ({RAW_VARS}) nor prepared ({PREPARED_VARS}) variables present; have {sorted(have)}")


# --- units --------------------------------------------------------------------------------------------------------------
def _unit_scale(da: xr.Dataset, name: str, default: str) -> "tuple[float, str]":
    """(factor to W m-2, how the unit was established) for a radiation variable."""
    raw = da[name].attrs.get("units")
    if raw is None:
        return (J_TO_W if default == "J" else 1.0), f"assumed {default} (no units attribute)"
    u = str(raw).strip().lower()
    if u in _J_UNITS:
        return J_TO_W, f"attribute {raw!r}: J m-2 over 1 h -> W m-2 (/3600)"
    if u in _W_UNITS:
        return 1.0, f"attribute {raw!r}: already W m-2"
    raise UnitsError(f"{name}: unexpected units attribute {raw!r} (expected J m**-2 or W m**-2)")


def _check_magnitudes(toa_wm2: xr.DataArray, temperature_k: xr.DataArray) -> None:
    hi, lo = float(toa_wm2.max()), float(toa_wm2.min())
    if hi > MAX_TOA_WM2 or lo < -50.0:
        raise UnitsError(f"TOA irradiance range [{lo:.4g}, {hi:.4g}] W m-2 after unit handling is implausible (expected 0..{MAX_TOA_WM2:g}); "
                         "J m-2 not converted, or wrong variable?")
    tlo, thi = float(temperature_k.min()), float(temperature_k.max())
    if not (C.RANGE_T_K[0] <= tlo and thi <= C.RANGE_T_K[1]):
        raise UnitsError(f"temperature range [{tlo:.4g}, {thi:.4g}] outside plausible {C.RANGE_T_K} K (Celsius instead of Kelvin?)")


def to_wm2_fields(ds: xr.Dataset, kind: str) -> "tuple[xr.Dataset, dict[str, str]]":
    """Stamped (still unshifted) fields in atlite naming and units: influx_* W m-2 (clipped >= 0), albedo, temperature K.

    Raw: influx_toa = tisr, influx_direct = fdir, influx_diffuse = ssrd - fdir (J m-2 / 3600), albedo = (ssrd - ssr) / ssrd
    (0 where ssrd == 0), exactly as atlite's `_process_influx`.
    """
    notes: dict[str, str] = {}
    if kind == "raw":
        sc = {}
        for v in ("ssrd", "ssr", "fdir", "tisr"):
            sc[v], notes[v] = _unit_scale(ds, v, "J")
        ssrd, ssr, fdir, tisr = (ds[v].astype(np.float64) * sc[v] for v in ("ssrd", "ssr", "fdir", "tisr"))
        albedo = ((ssrd - ssr) / ssrd.where(ssrd != 0)).fillna(0.0)
        out = xr.Dataset({"influx_toa": tisr, "influx_direct": fdir, "influx_diffuse": ssrd - fdir, "albedo": albedo,
                          "temperature": ds["t2m"].astype(np.float64)})
        notes["t2m"] = "K (instantaneous)"
    else:
        out = xr.Dataset({v: ds[v].astype(np.float64) for v in PREPARED_VARS})
        for v in ("influx_toa", "influx_direct", "influx_diffuse"):
            s, notes[v] = _unit_scale(ds, v, "W")
            out[v] = out[v] * s
        notes["temperature"] = "K"
    _check_magnitudes(out["influx_toa"], out["temperature"])
    for v in ("influx_toa", "influx_direct", "influx_diffuse"):
        out[v] = out[v].clip(min=0.0)   # atlite sanitize_influx
    return out, notes


# --- window --------------------------------------------------------------------------------------------------------------
def required_stamps(start: Any, end: Any, convention: str) -> pd.DatetimeIndex:
    """end-stamped: radiation h+1 and temperature h, h+1 for h in [start, end] -> start..end+1h; hour-start: start..end."""
    hours = utc_hour_range(start, end)
    return hours.append(pd.DatetimeIndex([hours[-1] + pd.Timedelta(hours=1)])) if convention == "end" else hours


def validate_window(stamps: pd.DatetimeIndex, start: Any, end: Any, convention: str) -> dict[str, Any]:
    need = required_stamps(start, end, convention)
    have = to_utc_naive_index(stamps)
    if have.has_duplicates:
        raise WindowError(f"duplicate ERA5 time stamps, first {have[have.duplicated()][0]}")
    missing = need.difference(have)
    if len(missing):
        raise WindowError(f"{len(missing)} of {len(need)} required ERA5 stamps are missing for {convention}-stamped data "
                          f"({need[0]:%Y-%m-%dT%H:%M} .. {need[-1]:%Y-%m-%dT%H:%M}); first missing {missing[0]:%Y-%m-%dT%H:%M}")
    return {"convention": convention, "required_stamps": int(len(need)), "first_required": f"{need[0]:%Y-%m-%dT%H:%M:%SZ}",
            "last_required": f"{need[-1]:%Y-%m-%dT%H:%M:%SZ}", "hours": int(len(utc_hour_range(start, end))),
            "note": ("the stamp at the first window hour is only used for the temperature endpoint mean (its radiation covers the previous hour)"
                     if convention == "end" else "hour-start data: one stamp per hour")}


# --- convention detection -----------------------------------------------------------------------------------------------------
_MAX_DETECT_CELLS = 400


def _detect_subset(lat: np.ndarray, lon: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    step_y = max(1, int(np.ceil(np.sqrt(_MAX_DETECT_CELLS * lat.size / lon.size))))
    step_x = max(1, int(np.ceil(np.sqrt(_MAX_DETECT_CELLS * lon.size / lat.size))))
    return np.arange(0, lat.size, step_y), np.arange(0, lon.size, step_x)


def _com_offsets(x: np.ndarray, times: pd.DatetimeIndex, lon: np.ndarray, label_to_mid_min: float) -> np.ndarray:
    """Centre-of-mass minus solar noon (minutes) per complete cell-day whose first and last label hour are dark."""
    offs: list[np.ndarray] = []
    dates = times.normalize()
    for d in dates.unique():
        idx = np.flatnonzero(dates == d)
        if len(idx) != 24 or (np.diff(idx) != 1).any():
            continue
        day = x[idx]                                     # (24, Y, X)
        ok = (day[0] <= 0) & (day[-1] <= 0) & (day.sum(axis=0) > 0)
        if not ok.any():
            continue
        mins = (np.asarray(times[idx].hour) * 60.0 + label_to_mid_min)[:, None, None]
        com = (day * mins).sum(axis=0) / np.where(ok, day.sum(axis=0), np.nan)
        noon = G.solar_noon_utc_minutes([d], lon)[0][None, :]
        offs.append((com - noon)[ok])
    return np.concatenate(offs) if offs else np.array([])


def detect_convention(toa_wm2: xr.DataArray, min_values: int = 24) -> dict[str, Any]:
    """Which labelling fits the TOA daily cycle: 'end' (value over [L-1h, L]) or 'start' ([L, L+1h]).

    Primary evidence: relative RMSE between the TOA series (scaled) and the interval-mean of max(cos zenith, 0) from the NOAA
    equations under both hypotheses. Supporting evidence: centre of mass of the daily cycle minus NOAA solar noon (minutes) on
    complete dark-to-dark cell-days. Ambiguous when data are insufficient, neither fits (rel. RMSE > 0.25) or the better
    hypothesis is not clearly better (RMSE ratio > 0.7).
    """
    t = to_utc_naive_index(toa_wm2[TIME].values)
    iy, ix = _detect_subset(toa_wm2[LAT].values, toa_wm2[LON].values)
    lat, lon = toa_wm2[LAT].values[iy], toa_wm2[LON].values[ix]
    x = np.asarray(toa_wm2.transpose(TIME, LAT, LON).values, dtype=np.float64)[:, iy][:, :, ix]
    ev: dict[str, Any] = {"n_stamps": int(len(t)), "n_cells_used": int(lat.size * lon.size), "n_positive_values": int((x > 0).sum()),
                          "method": "TOA vs NOAA interval-mean cos(zenith); centre of mass vs NOAA solar noon"}
    if len(t) < 24 or ev["n_positive_values"] < min_values:
        return {"detected": "ambiguous", "reason": "insufficient data (need >= 24 stamps with daylight)", **ev}
    hyp = {"start": (0.0, 30.0), "end": (-60.0, -30.0)}   # (interval start offset from label, label -> interval midpoint) minutes
    for name, (start_off, mid_off) in hyp.items():
        model = G.interval_mean_cos_zenith(t + pd.Timedelta(minutes=start_off), lat, lon)
        denom = float((model * model).sum())
        alpha = float((x * model).sum() / denom) if denom > 0 else np.nan
        rel = float(np.sqrt(np.mean((x - alpha * model) ** 2)) / max(float(x.mean()), 1e-9))
        corr = float(np.corrcoef(x.ravel(), model.ravel())[0, 1])
        off = _com_offsets(x, t, lon, mid_off)
        ev[name] = {"relative_rmse": rel, "correlation": corr, "scale_vs_cos_zenith": alpha,
                    "com_minus_solar_noon_min_mean": float(off.mean()) if off.size else None,
                    "com_minus_solar_noon_min_mean_abs": float(np.abs(off).mean()) if off.size else None, "n_cell_days": int(off.size)}
    best, other = sorted(CONVENTIONS, key=lambda k: ev[k]["relative_rmse"])
    ratio = ev[best]["relative_rmse"] / max(ev[other]["relative_rmse"], 1e-12)
    ev["rmse_ratio_best_over_other"] = float(ratio)
    if ev[best]["relative_rmse"] > 0.25 or ratio > 0.7:
        return {"detected": "ambiguous", "reason": "neither labelling fits the solar geometry clearly", **ev}
    return {"detected": best, **ev}


@dataclass(frozen=True)
class ConventionResult:
    convention: str            # 'end' | 'start' actually used
    shift_hours: int           # -1 for end-stamped input (relabel h+1 -> h), 0 otherwise; applied exactly once
    record: dict[str, Any]     # {detected, evidence, shift_applied, override, ...} for the manifest


def resolve_convention(toa_wm2: xr.DataArray, override: "str | None" = None, hint: "str | None" = None) -> ConventionResult:
    if override is not None and override not in CONVENTIONS:
        raise ValueError(f"--era5-convention must be one of {CONVENTIONS}")
    ev = detect_convention(toa_wm2)
    detected = ev["detected"]
    if override is None:
        if detected == "ambiguous":
            raise ConventionAmbiguousError(f"cannot establish the ERA5 time convention ({ev.get('reason')}); "
                                           "pass --era5-convention end|start", ev)
        if hint in CONVENTIONS and hint != detected:
            raise ConventionAmbiguousError(f"dataset attribute {HINT_ATTR}={hint!r} contradicts the data (detected {detected!r}); "
                                           "pass --era5-convention end|start", ev)
        used = detected
    else:
        used = override
    record = {"detected": detected, "evidence": {k: v for k, v in ev.items() if k != "detected"}, "override": override,
              "override_contradicts_detection": bool(override is not None and detected not in ("ambiguous", override)),
              "dataset_hint": hint, "convention_used": used,
              "shift_applied": "relabel end-stamp h+1 -> hour start h (-1 h), temperature = mean(h, h+1)" if used == "end" else "none (hour-start input)"}
    return ConventionResult(used, -1 if used == "end" else 0, record)


# --- hourly conversion ------------------------------------------------------------------------------------------------------------
def solar_position(hours: pd.DatetimeIndex, lat: np.ndarray, lon: np.ndarray, offset: str = "30 minutes") -> "tuple[np.ndarray, np.ndarray]":
    """atlite's solar altitude/azimuth (rad), shape (T, Y, X), evaluated at hour start + `offset` (default h+30 min)."""
    from atlite.pv.solar_position import SolarPosition

    frame = xr.Dataset(coords={"time": pd.DatetimeIndex(hours), "y": np.asarray(lat, dtype=np.float64), "x": np.asarray(lon, dtype=np.float64)})
    frame = frame.assign_coords(lon=("x", frame["x"].values), lat=("y", frame["y"].values))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sp = SolarPosition(frame, time_shift=pd.to_timedelta(offset))
    sp = sp.transpose("time", "y", "x").compute()
    return sp["altitude"].values, sp["azimuth"].values


def build_hourly(fields: xr.Dataset, start: Any, end: Any, convention: str) -> xr.Dataset:
    """Hour-start dataset (time, y, x) with atlite variable names; applies the end-stamp relabel / endpoint mean exactly once."""
    hours = utc_hour_range(start, end)
    f = fields.assign_coords({TIME: to_utc_naive_index(fields[TIME].values).astype("datetime64[ns]")})
    if convention == "end":
        nxt = hours + pd.Timedelta(hours=1)
        out = {v: f[v].sel({TIME: nxt}).assign_coords({TIME: hours.values}) for v in ("influx_toa", "influx_direct", "influx_diffuse", "albedo")}
        out["temperature"] = 0.5 * (f["temperature"].sel({TIME: hours}).assign_coords({TIME: hours.values})
                                    + f["temperature"].sel({TIME: nxt}).assign_coords({TIME: hours.values}))
    else:
        out = {v: f[v].sel({TIME: hours}) for v in PREPARED_VARS}
    ds = xr.Dataset(out).transpose(TIME, LAT, LON)
    alt, az = solar_position(hours, ds[LAT].values, ds[LON].values)
    ds = ds.rename({TIME: "time", LAT: "y", LON: "x"})
    ds["solar_altitude"] = (("time", "y", "x"), alt)
    ds["solar_azimuth"] = (("time", "y", "x"), az)
    for v, unit in (("influx_toa", "W m**-2"), ("influx_direct", "W m**-2"), ("influx_diffuse", "W m**-2"), ("albedo", "(0 - 1)"),
                    ("temperature", "K"), ("solar_altitude", "rad"), ("solar_azimuth", "rad")):
        ds[v].attrs["units"] = unit
    ds["time"] = pd.DatetimeIndex(ds["time"].values)
    ds.attrs.update({HINT_ATTR: "start", "time_label": "start of the hour, UTC", "solar_position": "atlite SolarPosition at hour start + 30 min",
                     "module": "era5_solar", "prepared_features": ["influx", "temperature"]})
    return ds


@dataclass
class Era5SolarResult:
    ds: xr.Dataset             # hour-start (time, y, x), atlite names; see HOURLY_VARS
    convention: dict[str, Any]
    window: dict[str, Any]
    source: dict[str, Any]


def load_solar_era5(source: "str | Path | xr.Dataset", start: Any, end: Any, convention: "str | None" = None) -> Era5SolarResult:
    """Open, unit-check, detect the time convention, validate the window and return hour-start fields.

    `source`: a CDS NetCDF/ZIP, a directory of them, a prepared atlite-style dataset file, or an in-memory dataset.
    `convention`: 'end' | 'start' forces the interpretation (recorded; contradiction with the detection is noted).
    """
    ds = _canonical(source) if isinstance(source, xr.Dataset) else open_solar_source(source)
    kind = classify(ds)
    fields, unit_notes = to_wm2_fields(ds, kind)
    res = resolve_convention(fields["influx_toa"], convention, ds.attrs.get(HINT_ATTR))
    window = validate_window(to_utc_naive_index(ds[TIME].values), start, end, res.convention)
    hourly = build_hourly(fields, start, end, res.convention)
    src = {"kind": kind, "path": None if isinstance(source, xr.Dataset) else str(source), "variables": sorted(ds.data_vars),
           "units": unit_notes, "grid": {"n_lat": int(ds[LAT].size), "n_lon": int(ds[LON].size),
                                         "lat_range": [float(ds[LAT].min()), float(ds[LAT].max())],
                                         "lon_range": [float(ds[LON].min()), float(ds[LON].max())]},
           "atlite_temperature_unit": "K (atlite's CSi model uses r_tmod = 298 K and c_temp_amb = 1; Celsius would be wrong)"}
    return Era5SolarResult(hourly, res.record, window, src)


# --- CDS download (never called by tests) ----------------------------------------------------------------------------------------------
def build_cds_requests(start: Any, end: Any, grid: C.GridSpec = C.FINLAND_SOLAR) -> "list[tuple[str, dict[str, Any]]]":
    """One (file name, request body) per month for the stamps start .. end+1h (end-stamped radiation needs the next hour)."""
    import calendar

    stamps = required_stamps(start, end, "end")
    out: list[tuple[str, dict[str, Any]]] = []
    for (year, month), idx in pd.Series(stamps, index=stamps).groupby([stamps.year, stamps.month]):
        days = sorted({int(d) for d in idx.index.day})
        out.append((f"era5_solar_{year}_{month:02d}.nc", {
            "product_type": ["reanalysis"], "variable": list(CDS_SOLAR_VARIABLES), "year": [str(year)], "month": [f"{month:02d}"],
            "day": [f"{d:02d}" for d in days if d <= calendar.monthrange(year, month)[1]],
            "time": [f"{h:02d}:00" for h in range(24)], "data_format": "netcdf", "download_format": "unarchived", "area": grid.cds_area}))
    return out


def download_era5_solar(cache_dir: "str | Path", start: Any, end: Any, grid: C.GridSpec = C.FINLAND_SOLAR, *,
                        env: "Mapping[str, str] | None" = None, home: "str | Path | None" = None, client: Any = None) -> list[Path]:
    """Download the missing monthly files via cdsapi. Credentials only from the environment or ~/.cdsapirc; nothing else is read."""
    if client is None:
        if not era5.cds_credentials_available(env, home):
            raise CredentialUnavailableError("CDS credentials unavailable: set CDSAPI_URL and CDSAPI_KEY or create ~/.cdsapirc")
        import cdsapi

        e = os.environ if env is None else env
        client = cdsapi.Client(url=e.get("CDSAPI_URL"), key=e.get("CDSAPI_KEY")) if e.get("CDSAPI_KEY") else cdsapi.Client()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, body in build_cds_requests(start, end, grid):
        target = cache / name
        if not (target.is_file() and target.stat().st_size > era5.MIN_FILE_BYTES):
            atomic_write_with(target, lambda tmp, body=body: client.retrieve(C.CDS_DATASET, body, str(tmp)))
        paths.append(target)
    return paths
