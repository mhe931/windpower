"""CM SAF SARAH-3 (ICDR) daily-file reader: validity masks, trapezoid hourly means, SIS/SID pairing, inventory.

Input contract (checked against the real order, see docs/assumptions.md S1): one NetCDF per UTC day and variable,
30-min instantaneous steps (`time: point`), ascending `lat`/`lon` centres with `lat_bnds`/`lon_bnds`, int16 data with
fill -999, and `record_status(time)` = 0 ok / 1 void / 2 bad_quality. Units W m-2.

Hourly means are labelled by the START of the hour (UTC) and use the trapezoid rule over the three instantaneous
samples 0.25*G(h:00) + 0.50*G(h:30) + 0.25*G(h+1:00); the last sample may come from the next day's file. Every sample must
be valid or the hourly value is NaN. Missing data is never turned into zero.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from . import config as C
from .io_utils import sha256_file, to_utc_naive_index

KINDS: dict[str, str] = {
    "SIS": "surface_downwelling_shortwave_flux_in_air",
    "SID": "surface_direct_downwelling_shortwave_flux_in_air",
}
STATUS_OK, STATUS_VOID, STATUS_BAD, STATUS_NO_STEP = 0, 1, 2, -1
STATUS_NAMES = {STATUS_OK: "ok", STATUS_VOID: "void", STATUS_BAD: "bad_quality", STATUS_NO_STEP: "no_file_or_step"}
STEP = pd.Timedelta(minutes=30)
TRAPEZOID = (0.25, 0.50, 0.25)
BOUNDS_DECIMALS = 6   # bounds are float32 on disk (59.45 is stored as 59.45000076)

# Per-sample / per-hour cause of an invalid value (uint8 codes).
CAUSE_VALID, CAUSE_STEP, CAUSE_SIS, CAUSE_SID, CAUSE_NEG_DIFFUSE = 0, 1, 2, 3, 4
CAUSE_NAMES = {CAUSE_VALID: "valid", CAUSE_STEP: "missing_or_flagged_step", CAUSE_SIS: "sis_missing",
               CAUSE_SID: "sid_missing", CAUSE_NEG_DIFFUSE: "negative_diffuse"}


class SarahError(ValueError):
    """The SARAH-3 input violates the format contract (or is absent); `check` names the failed check."""

    def __init__(self, check: str, message: str):
        super().__init__(f"[{check}] {message}")
        self.check = check
        self.detail = message


# --- locating inputs ------------------------------------------------------------------------------------
def default_sarah_dir(kind: str, base: "str | Path | None" = None) -> Path:
    return C.data_dir(base) / "interim" / "smoke" / C.SARAH_DEFAULT_DIRS[kind]


def resolve_sarah_dir(kind: str, explicit: "str | Path | None" = None, env: "Mapping[str, str] | None" = None,
                      base: "str | Path | None" = None) -> Path:
    """Precedence: CLI flag > environment variable > default folder (only if it exists). Never creates or copies anything."""
    import os

    env = os.environ if env is None else env
    var = C.SARAH_SIS_ENV if kind == "SIS" else C.SARAH_SID_ENV
    if explicit:
        path, origin = Path(explicit), "flag"
    elif env.get(var):
        path, origin = Path(env[var]), f"environment variable {var}"
    else:
        path, origin = default_sarah_dir(kind, base), "default location"
        if not path.is_dir():
            raise SarahError("input_missing", f"no {kind} folder: the default {path} does not exist; pass "
                                              f"--sarah-{kind.lower()}-dir or set {var}")
    if not path.is_dir():
        raise SarahError("input_missing", f"{kind} folder from {origin} does not exist: {path}")
    return path


_NAME_RE = re.compile(r"^(SIS|SID)in(\d{8})")


def list_day_files(directory: "str | Path", kind: str) -> "dict[pd.Timestamp, Path]":
    """{UTC date: path} for `<kind>in<YYYYMMDD>*.nc` files (the date in the name selects files; the content is authoritative)."""
    out: dict[pd.Timestamp, Path] = {}
    for p in sorted(Path(directory).glob(f"{kind}in*.nc")):
        m = _NAME_RE.match(p.name)
        if m and m.group(1) == kind:
            out[pd.Timestamp(m.group(2))] = p
    if not out:
        raise SarahError("input_missing", f"no {kind}in<YYYYMMDD>*.nc files in {directory}")
    return out


# --- one file ---------------------------------------------------------------------------------------------
@dataclass
class DayFile:
    path: Path
    kind: str
    times: pd.DatetimeIndex
    values: np.ndarray            # (time, lat, lon) float32; NaN where the file holds fill
    record_status: np.ndarray     # (time,) int8
    lat: np.ndarray               # float64 centres, ascending
    lon: np.ndarray
    lat_bnds: np.ndarray          # (n, 2) float64 [lo, hi], rounded to 1e-6 deg
    lon_bnds: np.ndarray
    attrs: dict[str, Any] = field(default_factory=dict)
    var_attrs: dict[str, Any] = field(default_factory=dict)
    n_fill: int = 0
    bounds_from_file: bool = True


def _bounds(ds: xr.Dataset, name: str, centres: np.ndarray) -> "tuple[np.ndarray, bool]":
    if f"{name}_bnds" in ds:
        b = np.sort(np.asarray(ds[f"{name}_bnds"].values, dtype=np.float64), axis=1)
        return np.round(b, BOUNDS_DECIMALS), True
    half = C.SARAH_RES_DEG / 2
    return np.round(np.stack([centres - half, centres + half], axis=1), BOUNDS_DECIMALS), False


def read_day_file(path: "str | Path", kind: str) -> DayFile:
    """Read one daily SIS or SID file. The raw int16 fill value is masked to NaN; nothing else is altered."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {tuple(KINDS)}")
    path = Path(path)
    try:
        with xr.open_dataset(path, mask_and_scale=False) as ds:
            ds = ds.load()
    except Exception as exc:  # unreadable / not NetCDF
        raise SarahError("unreadable", f"{path.name}: {exc}") from exc
    for needed in (kind, "record_status", "time", "lat", "lon"):
        if needed not in ds.variables:
            raise SarahError("variables", f"{path.name}: missing variable {needed!r}; have {sorted(ds.variables)}")
    da = ds[kind]
    if da.dims != ("time", "lat", "lon"):
        raise SarahError("dims", f"{path.name}: {kind} has dims {da.dims}, expected ('time', 'lat', 'lon')")
    fill = da.attrs.get("_FillValue", da.attrs.get("missing_value", C.SARAH_FILL))
    raw = da.values
    fill_mask = raw == fill
    vals = raw.astype(np.float32)
    if "scale_factor" in da.attrs or "add_offset" in da.attrs:
        vals = vals * np.float32(da.attrs.get("scale_factor", 1.0)) + np.float32(da.attrs.get("add_offset", 0.0))
    vals[fill_mask | ~np.isfinite(vals)] = np.nan
    lat = np.asarray(ds["lat"].values, dtype=np.float64)
    lon = np.asarray(ds["lon"].values, dtype=np.float64)
    lat_b, lat_ok = _bounds(ds, "lat", lat)
    lon_b, lon_ok = _bounds(ds, "lon", lon)
    if lat.size > 1 and lat[0] > lat[-1]:   # always return ascending latitude
        lat, lat_b, vals = lat[::-1], lat_b[::-1], vals[:, ::-1, :]
    if lon.size > 1 and lon[0] > lon[-1]:
        lon, lon_b, vals = lon[::-1], lon_b[::-1], vals[:, :, ::-1]
    times = to_utc_naive_index(ds["time"].values)
    status = np.asarray(ds["record_status"].values).astype(np.int8)
    if status.shape != (len(times),):
        raise SarahError("dims", f"{path.name}: record_status shape {status.shape} != time length {len(times)}")
    return DayFile(path, kind, times, vals, status, lat, lon, lat_b, lon_b, dict(ds.attrs), dict(da.attrs),
                   int(fill_mask.sum()), lat_ok and lon_ok)


# --- a time series of files ----------------------------------------------------------------------------------
@dataclass
class SarahSeries:
    kind: str
    times: pd.DatetimeIndex       # complete 30-min grid first..last (steps without a file stay NaN, status -1)
    values: np.ndarray            # (T, y, x) float32, NaN = fill / absent
    record_status: np.ndarray     # (T,) int8; -1 = step not present in any file
    lat: np.ndarray
    lon: np.ndarray
    lat_bnds: np.ndarray
    lon_bnds: np.ndarray
    files: list[str]
    units: "str | None" = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def step_ok(self) -> np.ndarray:
        return self.record_status == STATUS_OK


def _same_grid(a: DayFile, ref: DayFile) -> None:
    for name in ("lat", "lon", "lat_bnds", "lon_bnds"):
        x, y = getattr(a, name), getattr(ref, name)
        if x.shape != y.shape or not np.allclose(x, y, atol=1e-5):
            raise SarahError("grid", f"{a.path.name}: {name} differs from {ref.path.name}")


def load_series(directory: "str | Path", kind: str, first_step: Any, last_step: Any) -> SarahSeries:
    """Concatenate the daily files that cover the 30-min steps first_step..last_step (inclusive) on a gap-free grid."""
    first, last = to_utc_naive_index([first_step])[0], to_utc_naive_index([last_step])[0]
    times = pd.date_range(first, last, freq=STEP)
    days = list_day_files(directory, kind)
    wanted = [d for d in pd.date_range(first.normalize(), last.normalize(), freq="D") if d in days]
    if not wanted:
        raise SarahError("input_missing", f"no {kind} file covers {first}..{last} in {directory}")
    values: "np.ndarray | None" = None
    status = np.full(len(times), STATUS_NO_STEP, dtype=np.int8)
    ref: "DayFile | None" = None
    seen = np.zeros(len(times), dtype=bool)
    files: list[str] = []
    for d in wanted:
        f = read_day_file(days[d], kind)
        if ref is None:
            ref = f
            values = np.full((len(times), f.lat.size, f.lon.size), np.nan, dtype=np.float32)
        else:
            _same_grid(f, ref)
        files.append(f.path.name)
        pos = times.get_indexer(f.times)
        inside = pos >= 0
        if seen[pos[inside]].any():
            raise SarahError("time", f"{f.path.name}: time steps already provided by another {kind} file")
        seen[pos[inside]] = True
        values[pos[inside]] = f.values[inside]  # type: ignore[index]
        status[pos[inside]] = f.record_status[inside]
    assert ref is not None and values is not None
    return SarahSeries(kind, times, values, status, ref.lat, ref.lon, ref.lat_bnds, ref.lon_bnds, files,
                       ref.var_attrs.get("units"), {k: ref.attrs.get(k) for k in ("title", "id", "institution", "license", "history")})


# --- validity and hourly means ---------------------------------------------------------------------------------------
def _in_range(v: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.isfinite(v) & (v >= 0.0) & (v <= C.SARAH_MAX_WM2)


@dataclass
class SampleMasks:
    """Per-sample (30-min step) validity; SID needs SIS valid in the same sample, diffuse is derived from the pair."""

    sis_ok: np.ndarray
    sid_ok: np.ndarray
    diffuse: np.ndarray       # float32; NaN where not computable or below -1 W m-2; [-1, 0) clipped to 0
    diffuse_ok: np.ndarray
    negative_diffuse: np.ndarray   # SIS and SID valid but SIS - SID < -1 W m-2


def sample_masks(sis: np.ndarray, sid: np.ndarray, sis_step_ok: np.ndarray, sid_step_ok: np.ndarray) -> SampleMasks:
    """Masks for sample arrays (T, y, x) and per-step record_status==ok flags (T,)."""
    sis_ok = _in_range(sis) & sis_step_ok[:, None, None]
    sid_ok = _in_range(sid) & sid_step_ok[:, None, None] & sis_ok     # SID without SIS in the same sample is never usable
    diff = np.where(sid_ok, sis - sid, np.nan).astype(np.float32)
    with np.errstate(invalid="ignore"):
        negative = sid_ok & (diff < -C.SARAH_DIFFUSE_TOL_WM2)
    diffuse_ok = sid_ok & ~negative
    diff = np.where(diffuse_ok, np.maximum(diff, 0.0), np.nan).astype(np.float32)
    return SampleMasks(sis_ok, sid_ok, diff, diffuse_ok, negative)


def trapezoid(g0: np.ndarray, g30: np.ndarray, g60: np.ndarray) -> np.ndarray:
    """0.25*G(h:00) + 0.50*G(h:30) + 0.25*G(h+1:00); NaN in any sample gives NaN."""
    return (TRAPEZOID[0] * g0 + TRAPEZOID[1] * g30 + TRAPEZOID[2] * g60).astype(np.float32)


@dataclass
class SarahHourly:
    hours: pd.DatetimeIndex       # hour START, UTC
    lat: np.ndarray
    lon: np.ndarray
    lat_bnds: np.ndarray
    lon_bnds: np.ndarray
    ghi: np.ndarray               # (T, y, x) float32 W m-2, NaN = invalid (from SIS only)
    direct: np.ndarray            # SID; valid only where the SIS samples are valid too
    diffuse: np.ndarray           # SIS - SID per sample, trapezoid
    cause: np.ndarray             # uint8 CAUSE_* per hour and pixel; 0 means ghi, direct and diffuse are all valid
    hour_step_ok: np.ndarray      # (T,) bool: all three samples have record_status ok in both files

    @property
    def valid(self) -> np.ndarray:
        return self.cause == CAUSE_VALID


def check_pair(sis: SarahSeries, sid: SarahSeries) -> None:
    if not sis.times.equals(sid.times):
        raise SarahError("time", "SIS and SID time axes differ")
    for name in ("lat", "lon", "lat_bnds", "lon_bnds"):
        if not np.allclose(getattr(sis, name), getattr(sid, name), atol=1e-5):
            raise SarahError("grid", f"SIS and SID {name} differ")


def hourly_means(sis: SarahSeries, sid: SarahSeries, hours: pd.DatetimeIndex, block_hours: int = 96) -> SarahHourly:
    """Trapezoid hourly SIS / SID / diffuse for the hour starts in `hours` (needs the step at h+1:00 of the last hour)."""
    check_pair(sis, sid)
    hours = to_utc_naive_index(hours)
    i0 = sis.times.get_indexer(hours)
    i2 = sis.times.get_indexer(hours + pd.Timedelta(hours=1))
    if (i0 < 0).any() or (i2 < 0).any():
        raise SarahError("time", "series does not contain all samples h:00 .. h+1:00 for the requested hours")
    ny, nx = sis.values.shape[1:]
    n = len(hours)
    ghi = np.full((n, ny, nx), np.nan, dtype=np.float32)
    direct, diffuse = ghi.copy(), ghi.copy()
    cause = np.zeros((n, ny, nx), dtype=np.uint8)
    hour_step_ok = np.zeros(n, dtype=bool)
    step_ok = sis.step_ok & sid.step_ok
    for b in range(0, n, block_hours):
        sl = slice(b, min(b + block_hours, n))
        lo, hi = int(i0[sl].min()), int(i2[sl].max()) + 1
        m = sample_masks(sis.values[lo:hi], sid.values[lo:hi], sis.step_ok[lo:hi], sid.step_ok[lo:hi])
        s_sis = np.where(m.sis_ok, sis.values[lo:hi], np.nan)
        s_sid = np.where(m.sid_ok, sid.values[lo:hi], np.nan)
        a, b1, c = i0[sl] - lo, i0[sl] + 1 - lo, i2[sl] - lo
        ghi[sl] = trapezoid(s_sis[a], s_sis[b1], s_sis[c])
        direct[sl] = trapezoid(s_sid[a], s_sid[b1], s_sid[c])
        diffuse[sl] = trapezoid(m.diffuse[a], m.diffuse[b1], m.diffuse[c])
        any3 = lambda ok: ~ok[a] | ~ok[b1] | ~ok[c]  # noqa: E731 - "at least one of the three samples fails"
        step_bad = ~(step_ok[lo:hi][a] & step_ok[lo:hi][b1] & step_ok[lo:hi][c])
        hour_step_ok[sl] = ~step_bad
        cz = np.zeros((len(a), ny, nx), dtype=np.uint8)
        # ok sets are nested (diffuse_ok <= sid_ok <= sis_ok): assign the weakest cause first, stronger ones override
        cz[any3(m.diffuse_ok)] = CAUSE_NEG_DIFFUSE
        cz[any3(m.sid_ok)] = CAUSE_SID
        cz[any3(m.sis_ok)] = CAUSE_SIS
        cz[np.broadcast_to(step_bad[:, None, None], cz.shape)] = CAUSE_STEP
        cause[sl] = cz
    return SarahHourly(hours, sis.lat, sis.lon, sis.lat_bnds, sis.lon_bnds, ghi, direct, diffuse, cause, hour_step_ok)


# --- inventory ------------------------------------------------------------------------------------------------------
def inventory(directory: "str | Path", kind: str, hash_files: bool = False,
              first_date: "Any | None" = None, last_date: "Any | None" = None) -> dict[str, Any]:
    """Read-only description of a SARAH-3 order: variable, units, time coverage, grid, flag counts, fill share, hashes.

    `first_date`/`last_date` restrict the scan to files whose date lies in that (inclusive) range.
    """
    days = list_day_files(directory, kind)
    if first_date is not None or last_date is not None:
        lo = pd.Timestamp(first_date) if first_date is not None else min(days)
        hi = pd.Timestamp(last_date) if last_date is not None else max(days)
        days = {d: p for d, p in days.items() if lo <= d <= hi}
        if not days:
            raise SarahError("input_missing", f"no {kind} files between {lo.date()} and {hi.date()} in {directory}")
    times: list[pd.DatetimeIndex] = []
    flags = {name: 0 for name in ("ok", "void", "bad_quality", "other")}
    flagged: list[dict[str, Any]] = []
    n_fill = n_values = n_neg = n_over = 0
    vmin, vmax = np.inf, -np.inf
    ref: "DayFile | None" = None
    hashes: dict[str, str] = {}
    meta: dict[str, Any] = {}
    grid_mismatch: list[str] = []
    for p in days.values():
        f = read_day_file(p, kind)
        if ref is None:
            ref = f
            meta = {k: f.attrs.get(k) for k in ("title", "id", "institution", "creator_name", "license", "history", "Conventions", "instrument")}
            meta |= {"variable_attrs": {k: (v.item() if hasattr(v, "item") else v) for k, v in f.var_attrs.items()}}
        else:
            try:
                _same_grid(f, ref)
            except SarahError as exc:
                grid_mismatch.append(exc.detail)
        times.append(f.times)
        for t, s in zip(f.times, f.record_status):
            name = STATUS_NAMES.get(int(s), "other")
            flags[name if name in flags else "other"] += 1
            if s != STATUS_OK:
                flagged.append({"time_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "record_status": int(s), "meaning": STATUS_NAMES.get(int(s), "other"),
                                "file": f.path.name})
        v = f.values
        finite = np.isfinite(v)
        n_values += v.size
        n_fill += f.n_fill
        if finite.any():
            vmin, vmax = min(vmin, float(np.nanmin(v))), max(vmax, float(np.nanmax(v)))
        n_neg += int((v[finite] < 0).sum())
        n_over += int((v[finite] > C.SARAH_MAX_WM2).sum())
        if hash_files:
            hashes[p.name] = sha256_file(p)
    assert ref is not None
    idx = times[0].append(times[1:]) if len(times) > 1 else times[0]
    diffs = np.diff(idx.values) if len(idx) > 1 else np.array([], dtype="timedelta64[ns]")
    step = np.timedelta64(int(STEP.value), "ns")
    gaps = [{"after_utc": pd.Timestamp(idx[i]).strftime("%Y-%m-%dT%H:%M:%SZ"), "gap_minutes": float(d / np.timedelta64(1, "m"))}
            for i, d in enumerate(diffs) if d != step][:20]
    return {
        "kind": kind, "directory": str(directory), "standard_name": ref.var_attrs.get("standard_name"),
        "expected_standard_name": KINDS[kind], "units": ref.var_attrs.get("units"),
        "n_files": len(days), "file_dates": [min(days).strftime("%Y-%m-%d"), max(days).strftime("%Y-%m-%d")],
        "time": {"first_utc": idx.min().strftime("%Y-%m-%dT%H:%M:%SZ"), "last_utc": idx.max().strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "n_steps": int(len(idx)), "step_minutes": 30, "unique": bool(idx.is_unique), "monotonic": bool(idx.is_monotonic_increasing),
                 "gaps": gaps, "n_gaps": len([1 for d in diffs if d != step])},
        "grid": {"n_lat": int(ref.lat.size), "n_lon": int(ref.lon.size), "res_deg": float(np.round(np.median(np.diff(ref.lat)), 4)),
                 "lat_centres": [float(ref.lat[0]), float(ref.lat[-1])], "lon_centres": [float(ref.lon[0]), float(ref.lon[-1])],
                 "lat_bounds": [float(ref.lat_bnds[0, 0]), float(ref.lat_bnds[-1, 1])], "lon_bounds": [float(ref.lon_bnds[0, 0]), float(ref.lon_bnds[-1, 1])],
                 "bounds_from_file": ref.bounds_from_file, "mismatches": grid_mismatch},
        "record_status": {"counts": flags, "flagged_steps": flagged},
        "values": {"n_total": n_values, "n_fill": n_fill, "fill_share": (n_fill / n_values) if n_values else None,
                   "min_non_fill": None if not np.isfinite(vmin) else vmin, "max_non_fill": None if not np.isfinite(vmax) else vmax,
                   "n_negative_non_fill": n_neg, f"n_above_{int(C.SARAH_MAX_WM2)}": n_over},
        "metadata": meta, "sha256": hashes if hash_files else None,
    }


def flagged_steps_in_window(inv: Mapping[str, Any], first: Any, last: Any) -> "list[dict[str, Any]]":
    lo, hi = to_utc_naive_index([first])[0], to_utc_naive_index([last])[0]
    return [s for s in inv["record_status"]["flagged_steps"] if lo <= pd.Timestamp(s["time_utc"]).tz_localize(None) <= hi]
