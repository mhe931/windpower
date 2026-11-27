"""NOAA solar-position equations (GML "General solar position calculations", Spencer-series form).

Used for objective checks only (ERA5 time-convention detection, daytime masks, fixture generation). The PV conversion
itself uses atlite's own `SolarPosition` so that the baseline of the solar report is reproduced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .io_utils import to_utc_naive_index

SOLAR_CONSTANT_WM2 = 1361.0


def _fractional_year(times: pd.DatetimeIndex) -> np.ndarray:
    doy = np.asarray(times.dayofyear, dtype=np.float64)
    hour = np.asarray(times.hour + times.minute / 60.0 + times.second / 3600.0, dtype=np.float64)
    return 2.0 * np.pi / 365.0 * (doy - 1.0 + (hour - 12.0) / 24.0)


def equation_of_time_min(times: Any) -> np.ndarray:
    g = _fractional_year(to_utc_naive_index(times))
    return 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))


def declination_rad(times: Any) -> np.ndarray:
    g = _fractional_year(to_utc_naive_index(times))
    return (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))


def cos_zenith(times: Any, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """cos(solar zenith angle) at the given UTC instants for a lat/lon grid: shape (T, n_lat, n_lon)."""
    t = to_utc_naive_index(times)
    minutes = np.asarray(t.hour * 60 + t.minute + t.second / 60.0, dtype=np.float64)
    eq, dec = equation_of_time_min(t), declination_rad(t)
    lat_r = np.radians(np.asarray(lat, dtype=np.float64))[None, :, None]
    tst = minutes[:, None, None] + eq[:, None, None] + 4.0 * np.asarray(lon, dtype=np.float64)[None, None, :]
    ha = np.radians(tst / 4.0 - 180.0)
    d = dec[:, None, None]
    return np.sin(lat_r) * np.sin(d) + np.cos(lat_r) * np.cos(d) * np.cos(ha)


def elevation_deg(times: Any, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return np.degrees(np.arcsin(np.clip(cos_zenith(times, lat, lon), -1.0, 1.0)))


def interval_mean_cos_zenith(starts: Any, lat: np.ndarray, lon: np.ndarray, minutes: float = 60.0, n: int = 6) -> np.ndarray:
    """Mean of max(cos z, 0) over [start, start + minutes) from n mid-point samples: what a TOA accumulation sees."""
    t = to_utc_naive_index(starts)
    acc = np.zeros((len(t), len(np.atleast_1d(lat)), len(np.atleast_1d(lon))))
    for k in range(n):
        off = pd.Timedelta(minutes=minutes * (k + 0.5) / n)
        acc += np.maximum(cos_zenith(t + off, lat, lon), 0.0)
    return acc / n


def solar_noon_utc_minutes(dates: Any, lon: np.ndarray) -> np.ndarray:
    """Solar noon as minutes after 00:00 UTC for each date (T,) and longitude (X,): shape (T, X)."""
    d = to_utc_naive_index(dates).normalize() + pd.Timedelta(hours=12)
    return 720.0 - 4.0 * np.asarray(lon, dtype=np.float64)[None, :] - equation_of_time_min(d)[:, None]
