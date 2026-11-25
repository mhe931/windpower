"""Atomic writes, hashing, UTC helpers and the schema/units registry."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

SYNTHETIC_STAMP = "SYNTHETIC"


class CredentialUnavailableError(RuntimeError):
    """A required credential (API key, CDS account) is not configured."""


# --- atomic writes ---------------------------------------------------------
def atomic_write_bytes(path: "str | Path", data: bytes) -> Path:
    """Write via a temp file in the target directory, then os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def atomic_write_text(path: "str | Path", text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars and containers to JSON-safe values (NaN/inf -> None)."""
    import math

    import numpy as np

    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, np.ndarray, pd.Index, pd.Series)):
        return [jsonable(v) for v in (sorted(obj) if isinstance(obj, set) else list(obj))]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if obj is pd.NaT:
        return None
    return str(obj)


def atomic_write_json(path: "str | Path", obj: Any) -> Path:
    return atomic_write_text(path, json.dumps(jsonable(obj), ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_with(path: "str | Path", writer: Callable[[Path], None]) -> Path:
    """Let `writer(tmp_path)` produce a file (e.g. to_netcdf), then atomically move it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".part")
    os.close(fd)
    try:
        writer(Path(tmp))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


# --- hashing ---------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: "str | Path", chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# --- UTC helpers -----------------------------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_utc_naive_index(values: Any) -> pd.DatetimeIndex:
    """Return a tz-naive DatetimeIndex in UTC. Naive input is taken to be UTC already."""
    idx = pd.DatetimeIndex(values)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


def utc_hour_range(start: Any, end: Any) -> pd.DatetimeIndex:
    """Inclusive hourly range, tz-naive UTC."""
    s = to_utc_naive_index([start])[0]
    e = to_utc_naive_index([end])[0]
    return pd.date_range(s, e, freq="h")


def iso_utc(ts: Any) -> str:
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def epoch_ms_to_iso(ms: "int | float | None") -> "str | None":
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- schema / units registry ----------------------------------------------
UNITS: dict[str, str] = {
    # ERA5 raw
    "u10": "m s-1", "v10": "m s-1", "u100": "m s-1", "v100": "m s-1",
    "t2m": "K", "d2m": "K", "sp": "Pa", "fsr": "m", "lcc": "1", "tp": "m",
    # ERA5 engineered
    "speed_100": "m s-1", "speed_10": "m s-1", "wind_shear_alpha": "1",
    "temp_2m_c": "degC", "dewpoint_2m_c": "degC", "dewpoint_spread_c": "K",
    "relative_humidity_pct": "%", "air_density_kgm3": "kg m-3",
    "icing_flag": "1", "icing_severity": "1",
    # fleet
    "lat": "degrees_north", "lon": "degrees_east",
    "capacity_mw": "MW", "cap_mw": "MW", "installed_capacity_mw": "MW",
    "hub_height_m": "m", "hub_m": "m", "rotor_diameter_m": "m", "rotor_m": "m",
    "specific_power_wm2": "W m-2", "commissioning_year": "year", "year_clean": "year",
    # power / price (later stages)
    "theoretical_wind_mw": "MW", "actual_wind_mw": "MW", "wind_power_mw": "MW",
    "solar_power_mw": "MW", "total_consumption_mw": "MW", "net_load_mw": "MW",
    "imbalance_price_eur_mwh": "EUR MWh-1",
}


def unit_of(name: str) -> "str | None":
    return UNITS.get(name)


def table_schema(df: pd.DataFrame, units: "Mapping[str, str] | None" = None) -> dict[str, dict[str, "str | None"]]:
    """{column: {dtype, unit}} for manifest.json; unit is None when unregistered."""
    reg = UNITS if units is None else {**UNITS, **units}
    return {str(c): {"dtype": str(df[c].dtype), "unit": reg.get(str(c))} for c in df.columns}


def is_synthetic(meta: Mapping[str, Any]) -> bool:
    return any(SYNTHETIC_STAMP in str(v) for v in meta.values())
