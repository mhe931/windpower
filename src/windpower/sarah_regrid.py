"""Conservative regridding of SARAH-3 (0.05 deg) onto the ERA5 solar grid with exact area weights.

Weights come from cell bounds: the overlap of a source cell with a target cell is dlon * (sin(lat_hi) - sin(lat_lo)), i.e.
proportional to spherical area. `valid_area_fraction` is the summed weight of valid sources divided by the FULL target-cell
area (it is not renormalised over the valid sources), so parts of a target cell outside the satellite order count as
not covered. A cell-hour uses the satellite only if that fraction reaches `SATELLITE_MIN_AREA_FRACTION` (99.9 %).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import config as C
from . import sarah as S
from . import solar_geometry as G

# Per cell-hour source / fallback reason codes (uint8).
SRC_SATELLITE, SRC_OUTSIDE_DOMAIN, SRC_NO_COVERAGE, SRC_PARTIAL, SRC_STEP, SRC_SIS, SRC_SID, SRC_NEG_DIFFUSE = range(8)
SOURCE_CODES: dict[int, str] = {
    SRC_SATELLITE: "satellite",
    SRC_OUTSIDE_DOMAIN: "era5_fallback:outside_satellite_domain",
    SRC_NO_COVERAGE: "era5_fallback:no_coverage",
    SRC_PARTIAL: "era5_fallback:partial_coverage",
    SRC_STEP: "era5_fallback:missing_or_flagged_step",
    SRC_SIS: "era5_fallback:sis_missing",
    SRC_SID: "era5_fallback:sid_missing",
    SRC_NEG_DIFFUSE: "era5_fallback:negative_diffuse",
}
_CAUSE_TO_SRC = {S.CAUSE_SIS: SRC_SIS, S.CAUSE_SID: SRC_SID, S.CAUSE_NEG_DIFFUSE: SRC_NEG_DIFFUSE}
_BLOCK = 48


def cell_bounds(centres: np.ndarray, res: float) -> np.ndarray:
    c = np.asarray(centres, dtype=np.float64)
    return np.stack([c - res / 2, c + res / 2], axis=1)


def overlap_matrix(target_bnds: np.ndarray, source_bnds: np.ndarray, spherical_lat: bool) -> np.ndarray:
    """(n_target, n_source) overlap measure: sin(lat) difference for latitude, radians for longitude."""
    lo = np.maximum(target_bnds[:, 0][:, None], source_bnds[:, 0][None, :])
    hi = np.minimum(target_bnds[:, 1][:, None], source_bnds[:, 1][None, :])
    pos = hi > lo
    if spherical_lat:
        return np.where(pos, np.sin(np.radians(hi)) - np.sin(np.radians(lo)), 0.0)
    return np.where(pos, np.radians(hi - lo), 0.0)


@dataclass(frozen=True)
class RegridPlan:
    grid: C.GridSpec
    lat: np.ndarray
    lon: np.ndarray
    lat_bnds: np.ndarray
    lon_bnds: np.ndarray
    w_lat: np.ndarray            # (n_lat, n_src_lat)
    w_lon: np.ndarray            # (n_lon, n_src_lon)
    cell_area: np.ndarray        # (n_lat, n_lon): full target-cell area in the units of the weights
    domain_fraction: np.ndarray  # (n_lat, n_lon): share of the cell area inside the satellite order

    def weighted_sum(self, x: np.ndarray) -> np.ndarray:
        """sum over sources of weight * x for (..., n_src_lat, n_src_lon) arrays -> (..., n_lat, n_lon)."""
        return (self.w_lat @ np.asarray(x, dtype=np.float64)) @ self.w_lon.T

    def area_fraction(self, x: np.ndarray) -> np.ndarray:
        """Share of each target-cell AREA carrying x (x in [0, 1] per source cell); not renormalised."""
        return self.weighted_sum(x) / self.cell_area


def build_plan(grid: C.GridSpec, src_lat_bnds: np.ndarray, src_lon_bnds: np.ndarray) -> RegridPlan:
    lat, lon = np.array(grid.lats()), np.array(grid.lons())
    lat_b, lon_b = cell_bounds(lat, grid.res), cell_bounds(lon, grid.res)
    w_lat = overlap_matrix(lat_b, np.asarray(src_lat_bnds, dtype=np.float64), True)
    w_lon = overlap_matrix(lon_b, np.asarray(src_lon_bnds, dtype=np.float64), False)
    full_lat = np.sin(np.radians(lat_b[:, 1])) - np.sin(np.radians(lat_b[:, 0]))
    full_lon = np.radians(lon_b[:, 1] - lon_b[:, 0])
    area = np.outer(full_lat, full_lon)
    domain = np.clip(np.outer(w_lat.sum(axis=1), w_lon.sum(axis=1)) / area, 0.0, 1.0)
    return RegridPlan(grid, lat, lon, lat_b, lon_b, w_lat, w_lon, area, domain)


def regrid_blocks(plan: RegridPlan, values: np.ndarray, valid: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Area-weighted mean over VALID sources and the valid area fraction, for (T, y, x) inputs -> (T, Y, X) float32 each."""
    n = values.shape[0]
    mean = np.full((n, plan.lat.size, plan.lon.size), np.nan, dtype=np.float32)
    frac = np.zeros_like(mean)
    for i in range(0, n, _BLOCK):
        sl = slice(i, min(i + _BLOCK, n))
        ok = valid[sl]
        wsum = plan.weighted_sum(ok.astype(np.float64))
        num = plan.weighted_sum(np.where(ok, values[sl], 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            mean[sl] = np.where(wsum > 0, num / wsum, np.nan)
        frac[sl] = np.clip(wsum / plan.cell_area, 0.0, 1.0)
    return mean, frac


@dataclass
class SarahOnGrid:
    """SARAH-3 hourly irradiance on the ERA5 solar grid plus coverage bookkeeping (all hour-start labelled)."""

    grid: C.GridSpec
    hours: pd.DatetimeIndex
    lat: np.ndarray
    lon: np.ndarray
    ghi: np.ndarray                      # (T, Y, X) W m-2 area-weighted mean over valid sources, NaN if none valid
    direct: np.ndarray
    diffuse: np.ndarray
    valid_area_fraction: np.ndarray      # (T, Y, X) float32 in [0, 1]
    source_code: np.ndarray              # (T, Y, X) uint8, see SOURCE_CODES
    threshold: float
    domain_fraction: np.ndarray          # (Y, X) static: share of the cell inside the order
    static_covered_fraction: np.ndarray  # (Y, X) static: area share with at least one valid daytime hour
    fully_covered: np.ndarray            # (Y, X) bool: static_covered_fraction >= threshold
    satellite_coverage_fraction: np.ndarray  # (Y, X): share of daytime hours with source == satellite (NaN without daytime)
    daytime_hours_per_cell: np.ndarray   # (Y, X) int
    src_lat: np.ndarray                  # source-grid centres for the pixel map
    src_lon: np.ndarray
    pixel_daytime_valid_share: np.ndarray  # (y, x): share of daytime hours with a fully valid hourly value
    pixel_covered: np.ndarray            # (y, x) bool: at least one valid daytime hour

    @property
    def satellite_mask(self) -> np.ndarray:
        return self.source_code == SRC_SATELLITE


def pixel_daytime_stats(hourly: S.SarahHourly) -> "tuple[np.ndarray, np.ndarray]":
    """Per source pixel: (valid daytime hours, daytime hours); daytime = solar elevation > 5 deg at h+30 min."""
    n_valid = np.zeros(hourly.ghi.shape[1:], dtype=np.int64)
    n_day = np.zeros_like(n_valid)
    mid = hourly.hours + pd.Timedelta(minutes=30)
    for i in range(0, len(mid), _BLOCK):
        sl = slice(i, min(i + _BLOCK, len(mid)))
        day = G.elevation_deg(mid[sl], hourly.lat, hourly.lon) > C.SARAH_DAYTIME_MIN_ELEVATION_DEG
        n_day += day.sum(axis=0)
        n_valid += (day & hourly.valid[sl]).sum(axis=0)
    return n_valid, n_day


def source_codes(plan: RegridPlan, hourly: S.SarahHourly, frac: np.ndarray, never: np.ndarray, threshold: float) -> np.ndarray:
    """Per cell-hour source / fallback reason. Precedence: satellite (>= threshold) > outside domain > flagged step >
    partial coverage (some but < threshold) > dominant cause by area when no source is valid."""
    code = np.full(frac.shape, SRC_PARTIAL, dtype=np.uint8)
    outside_area = plan.cell_area * (1.0 - plan.domain_fraction)
    static_none = plan.weighted_sum(never.astype(np.float64)) + outside_area   # never-covered pixels + area outside the order
    for i in range(0, frac.shape[0], _BLOCK):
        sl = slice(i, min(i + _BLOCK, frac.shape[0]))
        cause = hourly.cause[sl]
        areas = [np.broadcast_to(static_none, (cause.shape[0], *static_none.shape))]
        codes = [SRC_NO_COVERAGE]
        for c, src in _CAUSE_TO_SRC.items():
            areas.append(plan.weighted_sum((cause == c) & ~never[None]))
            codes.append(src)
        dominant = np.asarray(codes, dtype=np.uint8)[np.argmax(np.stack(areas), axis=0)]
        code[sl] = np.where(frac[sl] <= 0.0, dominant, code[sl])
    sat = frac >= threshold
    outside = plan.domain_fraction <= 0.0
    code[~hourly.hour_step_ok[:, None, None] & ~sat] = SRC_STEP
    code[:, outside] = SRC_OUTSIDE_DOMAIN
    code[sat] = SRC_SATELLITE
    return code


def regrid_hourly(hourly: S.SarahHourly, grid: C.GridSpec = C.FINLAND_SOLAR, threshold: float = C.SATELLITE_MIN_AREA_FRACTION,
                  progress: "Callable[[str], None] | None" = None) -> SarahOnGrid:
    """Regrid hourly SIS/SID/diffuse, derive valid area fractions, source codes and the static coverage map."""
    say = progress or (lambda m: None)
    plan = build_plan(grid, hourly.lat_bnds, hourly.lon_bnds)
    valid = hourly.valid          # ghi, direct and diffuse all valid
    ghi, frac = regrid_blocks(plan, hourly.ghi, valid)
    direct, _ = regrid_blocks(plan, hourly.direct, valid)
    diffuse, _ = regrid_blocks(plan, hourly.diffuse, valid)
    say("[regrid] area-weighted means and valid area fractions done")

    n_valid_px, n_day_px = pixel_daytime_stats(hourly)
    pixel_covered = n_valid_px > 0
    share = np.where(n_day_px > 0, n_valid_px / np.maximum(n_day_px, 1), np.nan).astype(np.float32)
    static_frac = np.clip(plan.area_fraction(pixel_covered.astype(np.float64)), 0.0, 1.0)   # rounding can give 1 + 1e-16
    code = source_codes(plan, hourly, frac, ~pixel_covered, threshold)
    sat = code == SRC_SATELLITE

    mid = hourly.hours + pd.Timedelta(minutes=30)
    day_cell = np.zeros(plan.cell_area.shape, dtype=np.int64)
    sat_day = np.zeros_like(day_cell)
    for i in range(0, len(mid), _BLOCK):
        sl = slice(i, min(i + _BLOCK, len(mid)))
        day = G.elevation_deg(mid[sl], plan.lat, plan.lon) > C.SARAH_DAYTIME_MIN_ELEVATION_DEG
        day_cell += day.sum(axis=0)
        sat_day += (day & sat[sl]).sum(axis=0)
    cov = np.where(day_cell > 0, sat_day / np.maximum(day_cell, 1), np.nan)
    return SarahOnGrid(grid, hourly.hours, plan.lat, plan.lon, ghi, direct, diffuse, frac, code, threshold, plan.domain_fraction,
                       static_frac, static_frac >= threshold, cov, day_cell, hourly.lat, hourly.lon, share, pixel_covered)


def source_counts(g: SarahOnGrid) -> dict[str, int]:
    vals, cnt = np.unique(g.source_code, return_counts=True)
    return {SOURCE_CODES[int(v)]: int(c) for v, c in zip(vals, cnt)}


def coverage_summary(g: SarahOnGrid) -> dict[str, Any]:
    """Headline numbers for the manifest / CLI; the report's targets (777 of 2,332) are compared, never forced."""
    frac = g.valid_area_fraction
    day = g.daytime_hours_per_cell > 0
    return {
        "hours": int(len(g.hours)),
        "grid": {"name": g.grid.name, "n_lat": int(g.lat.size), "n_lon": int(g.lon.size), "n_cells": int(g.fully_covered.size)},
        "threshold": g.threshold,
        "fully_covered_cells": int(g.fully_covered.sum()),
        "target_fully_covered_cells": C.SOLAR_REPORT_TARGETS["covered_cells"],
        "target_grid_cells": C.SOLAR_REPORT_TARGETS["grid_cells"],
        "fully_covered_definition": "cells whose static covered area share (source pixels with >= 1 fully valid daytime hour; "
                                    "daytime = solar elevation > 5 deg at h+30 min) is >= threshold of the ERA5 cell area",
        "cells_with_any_satellite_hour": int(g.satellite_mask.any(axis=0).sum()),
        "cells_satellite_every_daytime_hour": int(((g.satellite_coverage_fraction >= 1.0) & day).sum()),
        "source_pixels": {"n": int(g.pixel_covered.size), "never_covered_share": float(1.0 - g.pixel_covered.mean())},
        "valid_area_fraction": {
            "mean": float(frac.mean()),
            "mean_in_fully_covered_cells": float(frac[:, g.fully_covered].mean()) if g.fully_covered.any() else None,
            "min": float(frac.min()), "max": float(frac.max()),
            "share_cell_hours_ge_threshold": float((frac >= g.threshold).mean())},
        "source_counts": source_counts(g),
        "mean_satellite_coverage_fraction_daytime": float(np.nanmean(g.satellite_coverage_fraction)) if day.any() else None,
        "cell_hours_total": int(g.source_code.size),
    }
