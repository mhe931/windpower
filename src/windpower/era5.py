"""ERA5 ingest: open CDS files, canonicalise, validate, engineer features."""

from __future__ import annotations

import calendar
import glob
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from . import config as C
from .io_utils import (
    CredentialUnavailableError,
    atomic_write_with,
    to_utc_naive_index,
    utc_hour_range,
)

TIME, LAT, LON = "valid_time", "latitude", "longitude"
_TIME_ALIASES = ("valid_time", "time")
_LAT_ALIASES = ("latitude", "lat")
_LON_ALIASES = ("longitude", "lon")
_ENGINES = ("h5netcdf", "netcdf4", "scipy", None)
MIN_FILE_BYTES = 1024

# Accepted `units` attribute spellings per variable (CDS uses "m s**-1" and "(0 - 1)").
_UNIT_SPELLINGS: dict[str, set[str]] = {
    **{v: {"m s**-1", "m s-1", "m/s", "m s^-1"} for v in ("u10", "v10", "u100", "v100")},
    "t2m": {"k"}, "d2m": {"k"}, "sp": {"pa"}, "fsr": {"m"}, "tp": {"m"},
    "lcc": {"(0 - 1)", "1", "fraction", "dimensionless"},
}


class ValidationError(ValueError):
    """Raised when ERA5 data violates a contract; `check` names the failed check."""

    def __init__(self, check: str, message: str):
        super().__init__(f"[{check}] {message}")
        self.check = check
        self.detail = message


# --- opening files ---------------------------------------------------------
def _open_nc(path: "str | Path") -> xr.Dataset:
    last: "Exception | None" = None
    for eng in _ENGINES:
        try:
            ds = xr.open_dataset(path, engine=eng) if eng else xr.open_dataset(path)
            return _loaded(ds)
        except Exception as exc:  # engine cannot read this flavour; try the next
            last = exc
    raise RuntimeError(f"Could not open {path}: {last}")


def _loaded(ds: xr.Dataset) -> xr.Dataset:
    """Read into memory and release the file handle (Windows cannot delete open files)."""
    out = ds.load()
    ds.close()
    return out


def open_cds_file(filepath: "str | Path") -> xr.Dataset:
    """Open a CDS download whether NetCDF4, NetCDF3 or a zipped bundle of *.nc files.

    Zip members are read in memory-safe fashion (no path from the archive is trusted) and
    merged with compat="override" so instant/accum streams share coordinates.
    """
    filepath = Path(filepath)
    if not zipfile.is_zipfile(filepath):
        return _open_nc(filepath)
    parts: list[xr.Dataset] = []
    with zipfile.ZipFile(filepath) as zf, tempfile.TemporaryDirectory(prefix="era5_zip_") as tmp:
        members = sorted(n for n in zf.namelist() if n.lower().endswith(".nc"))
        if not members:
            raise ValueError(f"Zip archive {filepath} contains no .nc files.")
        for i, name in enumerate(members):
            info = zf.getinfo(name)
            if info.file_size < MIN_FILE_BYTES:
                continue
            target = Path(tmp) / f"{i:02d}_{Path(name).name}"
            target.write_bytes(zf.read(name))
            parts.append(_open_nc(target))
    if not parts:
        raise ValueError(f"Zip archive {filepath} has no usable .nc member (all < {MIN_FILE_BYTES} B).")
    return xr.merge(parts, compat="override")


# --- canonicalisation ------------------------------------------------------
def _rename_first(ds: xr.Dataset, aliases: tuple[str, ...], target: str, what: str) -> xr.Dataset:
    for name in aliases:
        if name in ds.dims or name in ds.coords:
            return ds if name == target else ds.rename({name: target})
    raise ValidationError("dims", f"no {what} coordinate; looked for {aliases}, found {list(ds.dims)}")


def canonicalise(ds: xr.Dataset) -> xr.Dataset:
    """Names valid_time/latitude/longitude; ascending, de-duplicated time; ascending lat and lon.

    Records the source latitude order in attrs["source_lat_order"] (idempotent: an existing
    record is kept so a second call cannot overwrite the original orientation).
    """
    ds = _rename_first(ds, _TIME_ALIASES, TIME, "time")
    ds = _rename_first(ds, _LAT_ALIASES, LAT, "latitude")
    ds = _rename_first(ds, _LON_ALIASES, LON, "longitude")
    drop = [c for c in ("number", "expver", "surface") if c in ds.coords and c not in ds.dims]
    ds = ds.drop_vars(drop)

    ds = ds.assign_coords({TIME: to_utc_naive_index(ds[TIME].values).astype("datetime64[ns]")})
    t_index = pd.DatetimeIndex(ds[TIME].values)
    if t_index.has_duplicates:
        ds = ds.isel({TIME: ~t_index.duplicated(keep="first")})
    ds = ds.sortby(TIME)

    lat_vals = ds[LAT].values
    order = "descending" if lat_vals.size > 1 and lat_vals[0] > lat_vals[-1] else "ascending"
    out = ds.sortby([LAT, LON])
    out.attrs = dict(ds.attrs)
    out.attrs.setdefault("source_lat_order", order)
    return out


def load_era5_window(
    cache_dir: "str | Path",
    start: Any,
    end: Any,
    pattern: str = "era5_*.nc",
) -> "LoadedEra5":
    """Load every completed monthly file in `cache_dir`, canonicalise, concat, cut to [start, end]."""
    files = sorted(f for f in glob.glob(os.path.join(str(cache_dir), pattern)) if os.path.getsize(f) > MIN_FILE_BYTES)
    if not files:
        raise FileNotFoundError(f"No completed ERA5 files matching {pattern!r} (> {MIN_FILE_BYTES} B) in {cache_dir}")
    loaded: list[str] = []
    skipped: dict[str, str] = {}
    parts: list[xr.Dataset] = []
    for f in files:
        try:
            parts.append(canonicalise(open_cds_file(f)))
            loaded.append(os.path.basename(f))
        except Exception as exc:
            skipped[os.path.basename(f)] = str(exc)
    if not parts:
        raise RuntimeError(f"ERA5 files exist in {cache_dir} but none could be opened: {skipped}")
    ds = xr.concat(parts, dim=TIME, data_vars="minimal", coords="minimal", compat="override", join="exact") \
        if len(parts) > 1 else parts[0]
    ds = canonicalise(ds)
    s, e = utc_hour_range(start, end)[[0, -1]]
    ds = ds.sel({TIME: slice(s, e)})
    return LoadedEra5(ds, loaded, skipped)


@dataclass
class LoadedEra5:
    ds: xr.Dataset
    loaded_files: list[str]
    skipped_files: "dict[str, str]" = field(default_factory=dict)


# --- validation ------------------------------------------------------------
def _fmt_ts(ts: Any) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M")


def _check_axis(vals: np.ndarray, name: str) -> None:
    if vals.size < 2:
        raise ValidationError("grid", f"{name} has {vals.size} point(s); need at least 2 for a regular grid")
    steps = np.diff(vals.astype(np.float64))
    bad = np.flatnonzero(np.abs(steps - C.GRID_RES_DEG) > C.GRID_TOL)
    if bad.size:
        k = int(bad[0])
        raise ValidationError(
            "grid",
            f"{name} not regular at {C.GRID_RES_DEG} deg: step {steps[k]:.6f} between index {k} "
            f"({vals[k]}) and {k + 1} ({vals[k + 1]}); {bad.size} irregular step(s)",
        )
    frac = vals / C.GRID_RES_DEG
    off = np.flatnonzero(np.abs(frac - np.round(frac)) * C.GRID_RES_DEG > C.GRID_TOL)
    if off.size:
        k = int(off[0])
        raise ValidationError(
            "grid",
            f"{name} not aligned to the {C.GRID_RES_DEG} deg lattice: value {vals[k]} at index {k} "
            f"is not a multiple of {C.GRID_RES_DEG}",
        )


def validate_era5(
    ds: xr.Dataset,
    start: Any = None,
    end: Any = None,
    *,
    grid: "C.GridSpec | None" = None,
    require_ascending: bool = True,
) -> dict[str, Any]:
    """Raise ValidationError on the first violated contract; return a summary otherwise.

    `ds` must already be canonical (see `canonicalise`). start/end are the requested UTC window.
    """
    for dim in (TIME, LAT, LON):
        if dim not in ds.dims:
            raise ValidationError("dims", f"missing dimension {dim!r}; have {list(ds.dims)}")

    # time: unique, strictly increasing, hourly, expected count
    t = to_utc_naive_index(ds[TIME].values)
    if t.has_duplicates:
        raise ValidationError("time", f"duplicate timestamps, first: {_fmt_ts(t[t.duplicated()][0])}")
    if not t.is_monotonic_increasing:
        raise ValidationError("time", "time index is not monotonic increasing")
    if len(t) > 1:
        gaps = np.flatnonzero(np.diff(t.values) != np.timedelta64(1, "h"))
        if gaps.size:
            k = int(gaps[0])
            raise ValidationError(
                "time",
                f"index is not strictly hourly: {_fmt_ts(t[k])} -> {_fmt_ts(t[k + 1])} "
                f"({gaps.size} irregular step(s))",
            )
    if start is not None and end is not None:
        want = utc_hour_range(start, end)
        if len(t) != len(want):
            raise ValidationError("time", f"expected {len(want)} hours for {_fmt_ts(want[0])}..{_fmt_ts(want[-1])}, found {len(t)}")
        if t[0] != want[0] or t[-1] != want[-1]:
            raise ValidationError("time", f"window is {_fmt_ts(t[0])}..{_fmt_ts(t[-1])}, expected {_fmt_ts(want[0])}..{_fmt_ts(want[-1])}")

    # grid
    lat, lon = ds[LAT].values, ds[LON].values
    if require_ascending:
        for name, v in ((LAT, lat), (LON, lon)):
            if v.size > 1 and not np.all(np.diff(v) > 0):
                raise ValidationError("grid", f"{name} is not strictly ascending; canonicalise first")
    _check_axis(lat, LAT)
    _check_axis(lon, LON)
    if grid is not None:
        exp_lat, exp_lon = np.array(grid.lats()), np.array(grid.lons())
        if lat.shape != exp_lat.shape or not np.allclose(lat, exp_lat, atol=C.GRID_TOL):
            raise ValidationError("grid", f"latitude axis {lat[0]}..{lat[-1]} (n={lat.size}) != grid {grid.name} {exp_lat[0]}..{exp_lat[-1]} (n={exp_lat.size})")
        if lon.shape != exp_lon.shape or not np.allclose(lon, exp_lon, atol=C.GRID_TOL):
            raise ValidationError("grid", f"longitude axis {lon[0]}..{lon[-1]} (n={lon.size}) != grid {grid.name} {exp_lon[0]}..{exp_lon[-1]} (n={exp_lon.size})")

    # variables, dims, units, ranges
    missing = [v for v in C.REQUIRED_VARS if v not in ds.data_vars]
    if missing:
        raise ValidationError("variables", f"missing required variable(s) {missing}; have {sorted(ds.data_vars)}")
    for v in C.REQUIRED_VARS:
        da = ds[v]
        if set(da.dims) != {TIME, LAT, LON}:
            raise ValidationError("variables", f"{v} has dims {da.dims}, expected ({TIME}, {LAT}, {LON})")
        unit = da.attrs.get("units")
        if unit is not None and str(unit).strip().lower() not in {s.lower() for s in _UNIT_SPELLINGS[v]}:
            raise ValidationError("units", f"{v} units attribute {unit!r}, expected one of {sorted(_UNIT_SPELLINGS[v])}")
        vals = da.values
        n_nan = int(np.isnan(vals).sum())
        if n_nan:
            raise ValidationError("range", f"{v} contains {n_nan} NaN value(s)")
        _check_range(v, float(vals.min()), float(vals.max()))

    return {
        "n_times": len(t),
        "first_utc": _fmt_ts(t[0]) + "Z" if len(t) else None,
        "last_utc": _fmt_ts(t[-1]) + "Z" if len(t) else None,
        "n_lat": int(lat.size),
        "n_lon": int(lon.size),
        "source_lat_order": ds.attrs.get("source_lat_order", "unknown"),
    }


def _check_range(v: str, lo: float, hi: float) -> None:
    def bad(msg: str) -> ValidationError:
        return ValidationError("range", f"{v} range [{lo:.6g}, {hi:.6g}] {msg}")

    if v in ("t2m", "d2m") and not (C.RANGE_T_K[0] <= lo and hi <= C.RANGE_T_K[1]):
        raise bad(f"outside plausible {C.RANGE_T_K} K (Kelvin expected; Celsius?)")
    if v == "sp" and not (C.RANGE_SP_PA[0] <= lo and hi <= C.RANGE_SP_PA[1]):
        raise bad(f"outside plausible {C.RANGE_SP_PA} Pa (hPa instead of Pa?)")
    if v == "lcc" and not (C.RANGE_LCC[0] <= lo and hi <= C.RANGE_LCC[1]):
        raise bad(f"outside fraction range {C.RANGE_LCC}")
    if v == "tp" and not (lo >= C.TP_MIN_M and hi < C.TP_MAX_M):
        raise bad(f"outside plausible [{C.TP_MIN_M}, {C.TP_MAX_M}) m/h (mm instead of m?)")
    if v in ("u10", "v10", "u100", "v100") and not (max(abs(lo), abs(hi)) < C.WIND_ABS_MAX):
        raise bad(f"has |wind| >= {C.WIND_ABS_MAX} m/s")


# --- feature engineering ---------------------------------------------------
def _magnus(t_c: xr.DataArray) -> xr.DataArray:
    return 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))


def engineer_features(ds: xr.Dataset) -> xr.Dataset:
    """Add the notebook-1 derived variables (returns a new dataset; input untouched)."""
    ds = ds.copy()
    ds["speed_100"] = np.hypot(ds["u100"], ds["v100"])
    ds["speed_10"] = np.hypot(ds["u10"], ds["v10"])
    s100 = np.maximum(ds["speed_100"], 0.1)
    s10 = np.maximum(ds["speed_10"], 0.1)
    ds["wind_shear_alpha"] = np.clip(np.log(s100 / s10) / np.log(100.0 / 10.0), 0.0, C.SHEAR_ALPHA_MAX)

    ds["temp_2m_c"] = ds["t2m"] - 273.15
    ds["dewpoint_2m_c"] = ds["d2m"] - 273.15
    ds["dewpoint_spread_c"] = ds["temp_2m_c"] - ds["dewpoint_2m_c"]
    es, e = _magnus(ds["temp_2m_c"]), _magnus(ds["dewpoint_2m_c"])
    ds["relative_humidity_pct"] = np.clip(100.0 * (e / es), 0.0, 100.0)
    ds["air_density_kgm3"] = ds["sp"] / (C.R_SPECIFIC_AIR * ds["t2m"])

    # Icing: supercooled water needs -15..+1 C, near-saturated air and low cloud or rain.
    window = (ds["temp_2m_c"] >= C.ICING_T_MIN_C) & (ds["temp_2m_c"] <= C.ICING_T_MAX_C)
    saturated = ds["dewpoint_spread_c"] <= C.ICING_SPREAD_MAX_C
    wet = ds["tp"] > C.ICING_TP_MIN_M
    moisture = (ds["lcc"] >= C.ICING_LCC_MIN) | wet
    ds["icing_flag"] = (window & saturated & moisture).astype(np.float32)
    temp_factor = np.clip(1.0 - np.abs(ds["temp_2m_c"] + 4.0) / 11.0, 0.0, 1.0)
    humid_factor = np.clip(1.0 - ds["dewpoint_spread_c"] / 2.5, 0.0, 1.0)
    cloud_factor = np.clip(np.maximum(ds["lcc"], wet.astype(float)), 0.0, 1.0)
    ds["icing_severity"] = (temp_factor * humid_factor * cloud_factor).astype(np.float32)
    return ds


def summary_stats(ds: xr.Dataset) -> dict[str, float]:
    """Headline numbers printed by notebook 1 (icing share in percent of space-time cells)."""
    return {
        "mean_speed_100": float(ds["speed_100"].mean()),
        "mean_air_density": float(ds["air_density_kgm3"].mean()),
        "icing_share_pct": float(ds["icing_flag"].mean()) * 100.0,
    }


# --- CDS download (never called by tests) ----------------------------------
def cds_credentials_available(env: "Mapping[str, str] | None" = None, home: "str | Path | None" = None) -> bool:
    env = os.environ if env is None else env
    if env.get("CDSAPI_URL") and env.get("CDSAPI_KEY"):
        return True
    rc = Path(env.get("CDSAPI_RC") or (Path(home) if home else Path.home()) / ".cdsapirc")
    return rc.is_file() and rc.stat().st_size > 0


def build_cds_requests(
    start: Any, end: Any, grid: C.GridSpec = C.FINLAND_GRID
) -> "list[tuple[str, dict[str, Any]]]":
    """One (target filename, request body) per calendar month touching [start, end]."""
    hours = utc_hour_range(start, end)
    out: list[tuple[str, dict[str, Any]]] = []
    for (year, month), idx in pd.Series(hours, index=hours).groupby([hours.year, hours.month]):
        days = sorted({int(d) for d in idx.index.day})
        out.append((
            f"era5_{year}_{month:02d}.nc",
            {
                "product_type": ["reanalysis"],
                "variable": list(C.CDS_VARIABLES),
                "year": [str(year)],
                "month": [f"{month:02d}"],
                "day": [f"{d:02d}" for d in days if d <= calendar.monthrange(year, month)[1]],
                "time": [f"{h:02d}:00" for h in range(24)],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": grid.cds_area,
            },
        ))
    return out


def download_era5_months(
    cache_dir: "str | Path",
    start: Any,
    end: Any,
    grid: C.GridSpec = C.FINLAND_GRID,
    *,
    env: "Mapping[str, str] | None" = None,
    home: "str | Path | None" = None,
    client: Any = None,
) -> list[Path]:
    """Download missing monthly files through cdsapi. Credentials come only from env / ~/.cdsapirc."""
    if client is None:
        if not cds_credentials_available(env, home):
            raise CredentialUnavailableError(
                "CDS credentials unavailable: set CDSAPI_URL and CDSAPI_KEY or create ~/.cdsapirc"
            )
        import cdsapi  # imported lazily: the offline path must not need a configured client

        e = os.environ if env is None else env
        client = cdsapi.Client(url=e.get("CDSAPI_URL"), key=e.get("CDSAPI_KEY")) if e.get("CDSAPI_KEY") else cdsapi.Client()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, body in build_cds_requests(start, end, grid):
        target = cache / name
        if target.is_file() and target.stat().st_size > MIN_FILE_BYTES:
            paths.append(target)
            continue

        def fetch(tmp: Path, body: dict[str, Any] = body) -> None:
            client.retrieve(C.CDS_DATASET, body, str(tmp))

        atomic_write_with(target, fetch)
        paths.append(target)
    return paths
