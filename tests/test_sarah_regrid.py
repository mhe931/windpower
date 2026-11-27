"""Conservative regrid: exact area weights, valid area fractions, the 99.9 % rule, fallback reason codes, static coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from windpower import config as C
from windpower import fixtures as F
from windpower import sarah as S
from windpower import sarah_regrid as R


def real_like_bounds():
    """Source bounds of the real order: lat 59.4..65.0 (112 rows), lon 19.0..32.0 (260 columns), 0.05 deg."""
    lat_e, lon_e = 59.4 + 0.05 * np.arange(113), 19.0 + 0.05 * np.arange(261)
    return np.stack([lat_e[:-1], lat_e[1:]], axis=1), np.stack([lon_e[:-1], lon_e[1:]], axis=1)


def test_weights_sum_to_cell_area_inside_and_partial_at_the_edges():
    lat_b, lon_b = real_like_bounds()
    plan = R.build_plan(C.FINLAND_SOLAR, lat_b, lon_b)
    assert plan.cell_area.shape == (44, 53)
    inner = plan.domain_fraction[1:22, 1:52]
    np.testing.assert_allclose(inner, 1.0, atol=1e-12)                         # weights of an interior cell sum to its area
    full = np.outer(plan.w_lat.sum(axis=1), plan.w_lon.sum(axis=1))[1:22, 1:52]
    np.testing.assert_allclose(full, plan.cell_area[1:22, 1:52], rtol=1e-12)
    # south row 59.375-59.625: the order starts at 59.4, so 0.025 of 0.25 deg is outside -> not covered
    lat0 = np.sin(np.radians(59.625)) - np.sin(np.radians(59.4))
    assert plan.domain_fraction[0, 10] == pytest.approx(lat0 / (np.sin(np.radians(59.625)) - np.sin(np.radians(59.375))))
    assert plan.domain_fraction[0, 10] == pytest.approx(0.9, abs=1e-3)
    assert plan.domain_fraction[22, 10] == pytest.approx(0.5, abs=2e-3)       # centre 65.0: half of the cell is north of the order
    assert (plan.domain_fraction[23:] == 0).all()                              # entirely north of 65 N
    assert plan.domain_fraction[10, 0] == pytest.approx(0.5) and plan.domain_fraction[10, 52] == pytest.approx(0.5)   # lon 19.0 / 32.0 edge columns
    assert (plan.w_lat >= 0).all() and (plan.w_lon >= 0).all()


def test_latitude_weights_are_spherical_area_not_degrees():
    lat_b, lon_b = real_like_bounds()
    plan = R.build_plan(C.FINLAND_SOLAR, lat_b, lon_b)
    i = 10                                              # cell 62.0 N: bounds 61.875-62.125 -> source rows overlapping
    row = plan.w_lat[i]
    nz = np.flatnonzero(row)
    assert row.sum() == pytest.approx(np.sin(np.radians(62.125)) - np.sin(np.radians(61.875)), rel=1e-12)
    interior = nz[1:-1]
    centres = lat_b[interior].mean(axis=1)
    ratio = row[interior] / row[interior][0]
    np.testing.assert_allclose(ratio, np.cos(np.radians(centres)) / np.cos(np.radians(centres[0])), rtol=1e-4)   # smaller toward the pole
    assert row[interior][-1] < row[interior][0]
    # the same 0.25 deg of longitude is a smaller ground area further north: cell area scales like cos(lat)
    assert plan.cell_area[40, 0] < plan.cell_area[0, 0]
    assert plan.cell_area[0, 0] / plan.cell_area[40, 0] == pytest.approx(np.cos(np.radians(59.5)) / np.cos(np.radians(69.5)), rel=2e-3)


def test_area_weighted_mean_is_not_renormalised_in_the_fraction():
    lat_b, lon_b = real_like_bounds()
    plan = R.build_plan(C.FINLAND_SOLAR, lat_b, lon_b)
    vals = np.full((1, 112, 260), 100.0)
    valid = np.ones_like(vals, dtype=bool)
    valid[0, :, 8:] = False                              # only the first 8 source columns (19.0-19.4) carry data
    mean, frac = R.regrid_blocks(plan, vals, valid)
    assert np.nanmax(np.abs(mean[np.isfinite(mean)] - 100.0)) < 1e-4            # mean over valid sources
    assert frac[0, 3, 0] == pytest.approx(plan.domain_fraction[3, 0])           # column 0 (18.875-19.125): half outside the order, rest valid
    assert frac[0, 3, 1] == pytest.approx(1.0)                                   # 19.125-19.375 lies inside the valid columns
    assert frac[0, 3, 2] == pytest.approx(0.1, abs=1e-6)                         # 19.375-19.4 of 19.375-19.625 is valid; NOT renormalised to 1
    assert frac[0, 3, 4] == 0 and np.isnan(mean[0, 3, 4])                        # no valid source at all: NaN, not 0
    assert frac.max() <= 1.0


def _single_cell_case(n_invalid: int):
    """One 0.25 deg ERA5 cell tiled by 100 x 100 sources; `n_invalid` of them invalid. Returns the valid area fraction."""
    grid = C.GridSpec("one_cell", 60.0, 60.0, 21.0, 21.0)
    e_lat, e_lon = 59.875 + 0.0025 * np.arange(101), 20.875 + 0.0025 * np.arange(101)
    lat_b, lon_b = np.stack([e_lat[:-1], e_lat[1:]], axis=1), np.stack([e_lon[:-1], e_lon[1:]], axis=1)
    plan = R.build_plan(grid, lat_b, lon_b)
    valid = np.ones((1, 100, 100), dtype=bool)
    valid[0, 50, :n_invalid] = False                    # near the middle latitude, so every pixel is ~1e-4 of the area
    _, frac = R.regrid_blocks(plan, np.ones(valid.shape), valid)
    return float(frac[0, 0, 0])


def test_threshold_is_0_999_of_the_cell_area_on_both_sides():
    assert C.SATELLITE_MIN_AREA_FRACTION == 0.999
    above, below = _single_cell_case(5), _single_cell_case(20)      # ~0.9995 and ~0.998 of the area valid
    assert above == pytest.approx(0.9995, abs=2e-5) and below == pytest.approx(0.998, abs=2e-5)
    assert above >= C.SATELLITE_MIN_AREA_FRACTION > below


def _manual_hourly(valid_by_hour: list[np.ndarray]) -> S.SarahHourly:
    e_lat, e_lon = 59.875 + 0.0025 * np.arange(101), 20.875 + 0.0025 * np.arange(101)
    lat_b, lon_b = np.stack([e_lat[:-1], e_lat[1:]], axis=1), np.stack([e_lon[:-1], e_lon[1:]], axis=1)
    hours = pd.date_range("2025-07-01T10:00", periods=len(valid_by_hour), freq="h")
    valid = np.stack(valid_by_hour)
    shape = valid.shape
    ghi = np.where(valid, 400.0, np.nan).astype(np.float32)
    cause = np.where(valid, S.CAUSE_VALID, S.CAUSE_SIS).astype(np.uint8)
    return S.SarahHourly(hours, lat_b.mean(axis=1), lon_b.mean(axis=1), lat_b, lon_b, ghi, ghi * 0.6, ghi * 0.4, cause, np.ones(shape[0], bool))


def test_regrid_hourly_applies_the_threshold_per_cell_hour():
    def mask(n_bad):
        m = np.ones((100, 100), dtype=bool)
        m[50, :n_bad] = False
        return m

    h = _manual_hourly([mask(5), mask(20), mask(0)])
    g = R.regrid_hourly(h, C.GridSpec("one_cell", 60.0, 60.0, 21.0, 21.0))
    code = g.source_code[:, 0, 0]
    assert list(code) == [R.SRC_SATELLITE, R.SRC_PARTIAL, R.SRC_SATELLITE]
    assert g.valid_area_fraction[0, 0, 0] >= 0.999 > g.valid_area_fraction[1, 0, 0] > 0.99
    assert bool(g.fully_covered[0, 0]) and g.static_covered_fraction[0, 0] == pytest.approx(1.0)   # every pixel valid in some daytime hour
    # satellite values are the means over valid sources, even when the cell fell back
    np.testing.assert_allclose(g.ghi[:, 0, 0], 400.0, atol=1e-3)


def test_northern_fallback_reason_codes_on_synthetic_order(tmp_path_factory, sarah_spec):
    base = tmp_path_factory.mktemp("sarah_ext")
    sis_dir, sid_dir = F.write_sarah_cache(base, "2025-07-30", 3, sarah_spec)
    hours = pd.date_range("2025-07-30T00:00", "2025-07-31T23:00", freq="h")
    last = hours[-1] + pd.Timedelta(hours=1)
    hourly = S.hourly_means(S.load_series(sis_dir, "SIS", hours[0], last), S.load_series(sid_dir, "SID", hours[0], last), hours)
    grid = C.GridSpec("ext", 59.5, 62.0, 20.5, 23.5)            # extends beyond the fixture order (59.9-61.5 N) on both sides
    g = R.regrid_hourly(hourly, grid)
    lats = list(grid.lats())
    codes = g.source_code
    south, north = lats.index(59.5), lats.index(62.0)
    assert (codes[:, south, :] == R.SRC_OUTSIDE_DOMAIN).all() and (codes[:, north, :] == R.SRC_OUTSIDE_DOMAIN).all()   # no overlap with the order
    i61_25 = lats.index(61.25)                                   # inside the order but north of the cutoff (61.0): never has data
    c = codes[:, i61_25, 3:8]
    assert np.isin(c, [R.SRC_NO_COVERAGE, R.SRC_STEP]).all() and (c == R.SRC_NO_COVERAGE).sum() > 0
    i60_0 = lats.index(60.0)                                     # south edge of the order (59.9) cuts the 59.875-60.125 cell: partial, never satellite
    assert (codes[:, i60_0, 4] != R.SRC_SATELLITE).all() and (codes[:, i60_0, 4] != R.SRC_OUTSIDE_DOMAIN).all()
    assert np.isin(codes[:, i60_0, 4], [R.SRC_PARTIAL, R.SRC_STEP]).all()
    # a cell fully inside the order and clear of the patch: satellite except in hours with a flagged record
    i60_75, j = lats.index(60.75), list(grid.lons()).index(22.0)
    col = codes[:, i60_75, j]
    flagged = ~hourly.hour_step_ok
    assert (col[~flagged] == R.SRC_SATELLITE).all() and (col[flagged] == R.SRC_STEP).all()
    assert flagged.sum() == 3      # bad 07-30T10:30 touches hour 10 only (as h:30); void 07-31T12:00 touches hours 11 (h+1:00) and 12 (h:00)
    assert set(np.unique(codes)) <= set(R.SOURCE_CODES)
    # reason strings follow the spec's vocabulary
    assert R.SOURCE_CODES[R.SRC_SATELLITE] == "satellite" and R.SOURCE_CODES[R.SRC_OUTSIDE_DOMAIN] == "era5_fallback:outside_satellite_domain"
    assert {R.SOURCE_CODES[c] for c in (R.SRC_NO_COVERAGE, R.SRC_PARTIAL, R.SRC_STEP, R.SRC_SIS, R.SRC_SID, R.SRC_NEG_DIFFUSE)} == {
        f"era5_fallback:{r}" for r in ("no_coverage", "partial_coverage", "missing_or_flagged_step", "sis_missing", "sid_missing", "negative_diffuse")}


def test_patch_cells_are_partial_and_not_fully_covered(sarah_smoke):
    g = sarah_smoke.ongrid
    lats, lons = list(C.SOLAR_SMOKE_GRID.lats()), list(C.SOLAR_SMOKE_GRID.lons())
    patched = [(lats.index(a), lons.index(b)) for a in (60.25, 60.5) for b in (21.5, 21.75)]   # patch 60.3-60.45 N, 21.5-21.7 E touches all four
    clean = (lats.index(60.75), lons.index(22.0))
    for cell in patched:
        assert not g.fully_covered[cell] and 0 < g.static_covered_fraction[cell] < C.SATELLITE_MIN_AREA_FRACTION
    assert g.fully_covered[clean] and g.static_covered_fraction[clean] >= 0.999
    assert not g.fully_covered[lats.index(60.0), :].any()       # south edge row is cut by the order
    assert not g.fully_covered[lats.index(61.0):, :].any()      # at and north of the 61.0 cutoff
    assert not g.fully_covered[:, lons.index(21.0)].any() and not g.fully_covered[:, lons.index(23.0)].any()   # 20.9 / 23.1 order edges cut these columns
    day = g.daytime_hours_per_cell > 0
    assert (g.satellite_coverage_fraction[g.fully_covered] > 0.85).all()        # full except the hours around the flagged steps
    assert (g.satellite_coverage_fraction[~g.fully_covered & day] == 0).all()
    s = R.coverage_summary(g)
    assert s["fully_covered_cells"] == int(g.fully_covered.sum()) and s["grid"]["n_cells"] == 63 and s["target_fully_covered_cells"] == 777
    assert s["valid_area_fraction"]["share_cell_hours_ge_threshold"] == pytest.approx((g.valid_area_fraction >= 0.999).mean())
    assert sum(s["source_counts"].values()) == s["cell_hours_total"]
