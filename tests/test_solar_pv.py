"""PV conversion: atlite CSi baseline, ERA5-only and hybrid cubes, night/range checks, unit helpers, per-cell lookup."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from windpower import config as C
from windpower import era5_solar as E
from windpower import sarah_regrid as R
from windpower import solar_pv as P
from conftest import SOLAR_END, SOLAR_START


@pytest.fixture(scope="module")
def cubes(era5_solar_result, sarah_smoke):
    return P.compute_cubes(era5_solar_result.ds, sarah_smoke.ongrid)


def test_panel_and_orientation_record_comes_from_atlite():
    rec = P.panel_record()
    assert rec["panel"] == "CSi" and rec["slope_deg"] == 40.0 and rec["azimuth_deg"] == 180.0
    assert rec["inverter_efficiency"] == 0.9 and rec["inverter_efficiency_verified_in_atlite"] and rec["model"] == "huld"
    assert P.orientation() == {"slope": 40.0, "azimuth": 180.0} and "K" in rec["temperature_unit"]


def test_cubes_have_time_lat_lon_axes_and_cf_in_unit_interval(cubes, era5_solar_result):
    for mode, cube in (("era5", cubes.era5), ("hybrid", cubes.hybrid)):
        assert cube.dims == ("time", "latitude", "longitude") and cube.shape == (48, 7, 9)
        assert pd.DatetimeIndex(cube["time"].values).equals(pd.date_range(SOLAR_START, SOLAR_END, freq="h"))
        assert np.all(np.diff(cube["latitude"].values) > 0) and cube.attrs["irradiance_mode"] == mode
        s = P.check_cf(cube)
        assert 0.0 <= s["min"] and s["max"] <= 1.0 and 0.1 < s["max"] and 0.05 < s["mean"] < 0.5
    assert cubes.get("era5") is cubes.era5 and cubes.get("hybrid") is cubes.hybrid
    with pytest.raises(P.PvError):
        P.compute_cubes(era5_solar_result.ds, None).get("hybrid")
    with pytest.raises(ValueError):
        cubes.get("both")


def test_night_capacity_factor_is_exactly_zero(cubes):
    for mode, irr in cubes.irradiance.items():
        cube = cubes.get(mode)
        alt = irr["solar_altitude"].transpose("time", "y", "x").values
        night = alt <= 0.0
        assert night.sum() > 100
        assert (cube.values[night] == 0.0).all()                       # exactly 0, not merely small
        dark = (irr["influx_direct"] + irr["influx_diffuse"]).transpose("time", "y", "x").values <= 0
        assert (cube.values[dark] == 0.0).all() and not (cube.values < 0).any()
        assert (cube.values[alt > np.radians(20)] > 0).all()
    s = P.check_cf(cubes.era5, cubes.irradiance["era5"]["solar_altitude"])
    assert s["night_nonzero"] == 0 and s["night_values"] > 0


def test_check_cf_rejects_out_of_range_and_non_zero_night(cubes):
    cf = cubes.era5.copy(deep=True)
    bad = cf.copy()
    bad[0, 0, 0] = 1.2
    with pytest.raises(P.PvError, match=r"outside \[0, 1\]"):
        P.check_cf(bad)
    neg = cf.copy()
    neg[0, 0, 0] = -1e-3
    with pytest.raises(P.PvError):
        P.check_cf(neg)
    nan = cf.copy()
    nan[0, 0, 0] = np.nan
    with pytest.raises(P.PvError, match="NaN"):
        P.check_cf(nan)
    alt = cubes.irradiance["era5"]["solar_altitude"].transpose("time", "y", "x")
    night_cf = cf.copy(deep=True)
    night_cf.values[alt.values <= 0] = 0.01
    with pytest.raises(P.PvError, match="night"):
        P.check_cf(night_cf, alt)


def _atlite_reference(raw: xr.Dataset, constant_temperature: bool = False) -> xr.DataArray:
    """atlite's own ERA5 processing (end-stamped, position at -30 min) -> CF labelled by the hour START (stamp - 1 h)."""
    import atlite

    try:
        from atlite.datasets.era5 import _process_influx, sanitize_influx
    except ImportError:                                               # pragma: no cover - atlite internals moved
        pytest.skip("atlite.datasets.era5._process_influx not available in this atlite version")
    r = raw.rename({"valid_time": "time", "latitude": "y", "longitude": "x"}).sortby(["y", "x"])
    r = r.assign_coords(lon=("x", r["x"].values), lat=("y", r["y"].values))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = sanitize_influx(_process_influx(r[["ssrd", "ssr", "fdir", "tisr"]].astype("float64")))
    ds["temperature"] = r["t2m"].astype("float64") * (0 if constant_temperature else 1) + (288.0 if constant_temperature else 0)
    ds.attrs.update(module="era5", prepared_features=["influx", "temperature"], dx=0.25, dy=0.25)
    cutout = atlite.Cutout(path="unused_atlite_reference.nc", data=ds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        ref = cutout.pv(panel="CSi", orientation={"slope": 40.0, "azimuth": 180.0}, aggregate_time=None, show_progress=False)
    return ref.assign_coords(time=ref["time"] - pd.Timedelta(hours=1)).transpose("time", "y", "x")


def test_era5_only_cf_matches_atlites_own_end_stamped_processing(cubes, era5_solar_fx):
    ref = _atlite_reference(era5_solar_fx.raw).sel(time=cubes.era5["time"].values)
    ours = cubes.era5.rename({"latitude": "y", "longitude": "x"}).transpose("time", "y", "x")
    diff = np.abs(ours.values - ref.values)
    assert diff.mean() < 3e-4                                          # report: "within 0.0003 on average"
    assert ref.values.mean() > 0.1 and (ref.values > 0).sum() == pytest.approx((ours.values > 0).sum(), rel=0.05)


def test_cross_check_is_exact_when_temperature_cannot_differ(era5_solar_fx):
    """Constant temperature isolates the time handling (relabel, position at h+30, clipping): agreement to rounding."""
    raw = era5_solar_fx.raw.copy(deep=True)
    raw["t2m"] = raw["t2m"] * 0 + 288.0
    res = E.load_solar_era5(raw, SOLAR_START, SOLAR_END)
    ours = P.capacity_factors(P.era5_irradiance(res.ds)).rename({"latitude": "y", "longitude": "x"}).transpose("time", "y", "x")
    ref = _atlite_reference(raw).sel(time=ours["time"].values)
    assert np.abs(ours.values - ref.values).max() < 1e-5


def test_hybrid_replaces_direct_and_diffuse_only_in_satellite_cell_hours(cubes, era5_solar_result, sarah_smoke):
    irr_e, irr_h = cubes.irradiance["era5"], cubes.irradiance["hybrid"]
    g = sarah_smoke.ongrid
    sat = g.source_code == R.SRC_SATELLITE
    assert sat.any() and (~sat).any()
    for v in ("influx_toa", "albedo", "temperature", "solar_altitude", "solar_azimuth"):    # ERA5 in both modes
        np.testing.assert_array_equal(irr_h[v].values, irr_e[v].values)
    np.testing.assert_allclose(irr_h["influx_direct"].values[sat], g.direct[sat])           # satellite: SID and SIS - SID
    np.testing.assert_allclose(irr_h["influx_diffuse"].values[sat], g.diffuse[sat])
    np.testing.assert_array_equal(irr_h["influx_direct"].values[~sat], irr_e["influx_direct"].values[~sat])   # fallback: untouched ERA5
    np.testing.assert_array_equal(irr_h["influx_diffuse"].values[~sat], irr_e["influx_diffuse"].values[~sat])
    code = irr_h["source_code"].values
    assert ((code == 0) == sat).all() and set(np.unique(code[~sat])) <= set(range(1, 8))      # fallback reasons are kept per cell-hour
    np.testing.assert_array_equal(code[~sat], g.source_code[~sat])
    # CF: identical where ERA5 was used, different where the satellite replaced the split
    same = cubes.hybrid.values[~sat] == cubes.era5.values[~sat]
    assert same.all()
    daytime_sat = sat & (irr_e["influx_toa"].transpose("time", "y", "x").values > 50)
    assert np.abs(cubes.hybrid.values - cubes.era5.values)[daytime_sat].max() > 1e-3
    d = P.hybrid_diagnostics(irr_h)
    assert d["satellite_cell_hours"] == int(sat.sum()) and sum(d["reasons"].values()) == d["cell_hours"] == sat.size
    assert "satellite" in d["reasons"] and any(k.startswith("era5_fallback:") for k in d["reasons"])


def test_hybrid_rejects_misaligned_inputs(era5_solar_result, sarah_smoke):
    shifted = era5_solar_result.ds.assign_coords(time=pd.DatetimeIndex(era5_solar_result.ds["time"].values) + pd.Timedelta(hours=1))
    with pytest.raises(P.PvError, match="hours differ"):
        P.hybrid_irradiance(shifted, sarah_smoke.ongrid)
    moved = era5_solar_result.ds.assign_coords(x=era5_solar_result.ds["x"].values + 0.25)
    with pytest.raises(P.PvError, match="grids differ"):
        P.hybrid_irradiance(moved, sarah_smoke.ongrid)
    with pytest.raises(P.PvError, match="lacks"):
        P.era5_irradiance(era5_solar_result.ds.drop_vars("albedo"))


def test_unit_helpers_mwp_mw_mwh_and_cf():
    cf = np.array([0.0, 0.25, 0.5])
    mw = P.power_mw(cf, 4.0)
    np.testing.assert_allclose(mw, [0.0, 1.0, 2.0])                        # MW = CF x MWp
    assert P.energy_mwh(mw).sum() == pytest.approx(3.0)                    # hourly MW x 1 h
    assert P.energy_mwh(mw, hours=0.5).sum() == pytest.approx(1.5)
    np.testing.assert_allclose(P.capacity_factor_from_power(mw, 4.0), cf)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            P.power_mw(cf, bad)


def test_locate_cell_uses_cell_bounds_and_rejects_points_outside():
    g = C.SOLAR_SMOKE_GRID                                                  # centres 60.0..61.5 / 21.0..23.0
    c = P.locate_cell(60.3, 21.9, g)
    assert (c.lat, c.lon) == (60.25, 22.0) and c.lat_bounds == (60.125, 60.375) and c.lon_bounds == (21.875, 22.125)
    assert P.locate_cell(60.125, 21.875, g).lat == 60.25 and P.locate_cell(60.125, 21.875, g).lon == 22.0   # shared edge -> north / east cell
    assert (P.locate_cell(59.875, 20.875, g).i_lat, P.locate_cell(59.875, 20.875, g).i_lon) == (0, 0)       # outer edge is inside
    top = P.locate_cell(61.625, 23.125, g)
    assert (top.i_lat, top.i_lon) == (6, 8)
    for lat, lon in ((59.87, 22.0), (61.63, 22.0), (60.5, 20.87), (60.5, 23.13), (float("nan"), 22.0)):
        with pytest.raises(P.OutsideGridError):
            P.locate_cell(lat, lon, g)
    fin = P.locate_cell(60.1699, 24.9384)                                   # Helsinki on the default Finland solar grid
    assert (fin.lat, fin.lon) == (60.25, 25.0)


def test_cell_series_returns_that_cells_hourly_cf(cubes):
    ref, s = P.cell_series(cubes.era5, 60.8, 22.1, C.SOLAR_SMOKE_GRID)
    assert (ref.lat, ref.lon) == (60.75, 22.0) and len(s) == 48 and s.index.name == "time_utc"
    np.testing.assert_array_equal(s.values, cubes.era5.sel(latitude=60.75, longitude=22.0).values)
    assert s.max() > 0.1 and (s >= 0).all() and s.index.is_unique
