from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from windpower import config as C
from windpower import era5, fleet
from windpower import simulate as S

LATS, LONS = np.array([60.0, 60.25]), np.array([21.0, 21.25, 21.5])
T0 = "2025-07-01T00:00:00"


def grid_ds(speed, rho=1.225, z0=0.05, hours=1):
    """Canonical dataset with just what the simulator needs; scalars or (y, x) arrays broadcast over time."""
    shape = (hours, LATS.size, LONS.size)

    def full(v):
        return np.broadcast_to(np.asarray(v, float), shape).copy()

    dims = ("valid_time", "latitude", "longitude")
    return xr.Dataset(
        {"speed_100": (dims, full(speed)), "air_density_kgm3": (dims, full(rho)), "fsr": (dims, full(z0))},
        coords={"valid_time": pd.date_range(T0, periods=hours, freq="h"), "latitude": LATS, "longitude": LONS})


def specs_for(turbines):
    """turbines: [(lat, lon, cap_mw, specific_power, hub_m)] -> df_t via the real spec-table builder."""
    rows = [{"lat": la, "lon": lo, "capacity_mw": c, "rotor_diameter_m": 2 * math.sqrt(c * 1e6 / (math.pi * sp)),
             "hub_height_m": h, "commissioning_year": 2020.0} for la, lo, c, sp, h in turbines]
    return fleet.build_turbine_spec_table(pd.DataFrame(rows))


def hand_power(v, rated=3.0, sp=300.0):
    """Independent re-statement of the aerodynamic curve at one speed (MW)."""
    cp = min(max(0.50 - 0.00018 * sp, 0.40), 0.48)
    area = rated * 1e6 / min(max(sp, 170.0), 450.0)
    return min(rated, 0.5 * 1.225 * area * cp * v**3 / 1e6) if 3.0 <= v <= 25.0 else 0.0


# --- power curve ---------------------------------------------------------------------------
def test_power_curve_has_61_points_and_cut_limits():
    c = S.build_power_curve(3.0, 300.0, 120.0)
    assert len(c["V"]) == len(c["POW"]) == 61 and c["V"][0] == 0.0 and c["V"][-1] == 30.0
    np.testing.assert_allclose(np.diff(c["V"]), 0.5)
    assert c["hub_height"] == 120.0 and c["P"] == 3.0
    assert (c["POW"][c["V"] < 3.0] == 0).all() and (c["POW"][c["V"] > 25.0] == 0).all()
    assert c["POW"][c["V"] == 25.0].item() == 3.0 and c["POW"].max() == 3.0       # 25 m/s is still in service
    assert c["POW"][c["V"] == 3.0].item() > 0                                       # 3 m/s is cut-in (inclusive)
    assert (np.diff(c["POW"][(c["V"] >= 3.0) & (c["V"] <= 25.0)]) >= 0).all()


@pytest.mark.parametrize("sp, v", [(300.0, 8.0), (300.0, 6.5), (240.0, 9.0), (100.0, 7.0), (500.0, 7.0)])
def test_power_curve_matches_hand_calculation(sp, v):
    c = S.build_power_curve(3.0, sp, 100.0)
    got = c["POW"][np.flatnonzero(c["V"] == v)[0]]
    assert got == pytest.approx(hand_power(v, 3.0, sp), rel=1e-12)


def test_power_curve_sp_clip_and_cp_clip():
    # SP below 170: area uses 170 W/m2 but Cp still follows the unclipped SP (clipped to 0.48 at SP <= 111)
    c = S.build_power_curve(3.0, 100.0, 100.0)
    assert c["POW"][c["V"] == 6.0].item() == pytest.approx(0.5 * 1.225 * (3e6 / 170.0) * 0.48 * 216 / 1e6)
    c2 = S.build_power_curve(3.0, 600.0, 100.0)
    assert c2["POW"][c2["V"] == 6.0].item() == pytest.approx(0.5 * 1.225 * (3e6 / 450.0) * 0.40 * 216 / 1e6)


def test_power_curve_interpolation():
    c = S.build_power_curve(3.0, 300.0, 100.0)
    mid = S.power_from_speed(c, 7.25)
    assert mid == pytest.approx(0.5 * (hand_power(7.0) + hand_power(7.5)))        # linear between grid points
    assert S.power_from_speed(c, 8.0) == pytest.approx(hand_power(8.0))
    out = S.power_from_speed(c, np.array([[2.0, 12.0], [26.0, 40.0]]))
    assert out.shape == (2, 2) and out[0, 1] == 3.0 and out[1, 0] == 0.0 and out[1, 1] == 0.0 and out[0, 0] == 0.0
    assert S.power_from_speed(c, 12.0, per_unit=True) == 1.0
    assert 0.0 <= S.power_from_speed(c, 8.2, per_unit=True) <= 1.0


# --- wind helpers ----------------------------------------------------------------------------
def test_effective_wind_speed_density_adjustment():
    assert S.effective_wind_speed(8.0, 1.225) == pytest.approx(8.0)
    assert S.effective_wind_speed(8.0, 1.0) == pytest.approx(8.0 * (1.0 / 1.225) ** (1 / 3))
    assert S.effective_wind_speed(8.0, 0.5) == pytest.approx(8.0 * 0.7 ** (1 / 3))     # clipped at 0.7
    assert S.effective_wind_speed(8.0, 2.0) == pytest.approx(8.0 * 1.3 ** (1 / 3))     # clipped at 1.3


def test_hub_height_log_extrapolation():
    assert S.hub_height_speed(8.0, 0.05, 150.0) == pytest.approx(8.0 * math.log(150 / 0.05) / math.log(100 / 0.05))
    assert S.hub_height_speed(8.0, 0.05, 100.0) == pytest.approx(8.0)
    assert S.hub_height_speed(8.0, 0.05, 60.0) < 8.0
    assert S.hub_height_speed(8.0, 0.0, 120.0) == pytest.approx(8.0 * math.log(120 / 1e-4) / math.log(100 / 1e-4))  # z0 floor
    np.testing.assert_allclose(S.hub_height_speed(np.array([4.0, 8.0]), np.array([0.1, 0.1]), 140.0),
                               np.array([4.0, 8.0]) * math.log(1400) / math.log(1000))
    assert S.hub_height_speed(8.0, 5.0, 3.0) == 0.0                                      # hub below roughness length


def test_units_helpers():
    assert S.capacity_factor(np.array([0.0, 50.0, 100.0]), 100.0).tolist() == [0.0, 0.5, 1.0]
    assert S.energy_mwh(np.array([10.0, 20.0])).tolist() == [10.0, 20.0]             # MW x 1 h
    assert S.energy_mwh(10.0, hours=0.25) == 2.5


def test_prepare_wind_dataset_orientation_and_clipping():
    ds = grid_ds(8.0, rho=0.8, z0=0.0, hours=2)
    out = S.prepare_wind_dataset(ds.isel(latitude=slice(None, None, -1)))        # descending input is re-sorted
    assert list(out["y"].values) == sorted(LATS) and out.attrs["dx"] == out.attrs["dy"] == 0.25
    assert out["wnd100m"].dims == ("time", "y", "x") and out["wnd100m"].shape == (2, 2, 3)
    np.testing.assert_allclose(out["wnd100m"].values, 8.0 * 0.7 ** (1 / 3))        # rho/1.225 = 0.65 -> clipped to 0.7
    assert (out["roughness"].values == 1e-4).all()


# --- layout ------------------------------------------------------------------------------------
def test_nearest_centre_layout_orientation_and_border_report():
    lats, lons = np.array(C.SMOKE_GRID.lats()), np.array(C.SMOKE_GRID.lons())
    lat = np.array([lats[2], lats[3] + 0.125, lats[-1] + 0.5, lats[5] + 0.01])
    lon = np.array([lons[5], lons[4], lons[1], lons[6] - 0.01])
    cap = np.array([1.0, 2.0, 4.0, 8.0])
    mat, rep = S.nearest_centre_layout(lat, lon, cap, lats, lons)
    assert mat.shape == (lats.size, lons.size) and mat.sum() == cap.sum()
    assert mat[2, 5] == 1.0 and mat[5, 6] == 8.0 and mat[lats.size - 1, 1] == 4.0           # [lat_i, lon_j]
    assert mat[3, 4] + mat[4, 4] == 2.0                                                       # edge point lands on one neighbour
    assert rep.turbines == 4 and rep.outside_cell_box_turbines == 2 and rep.outside_cell_box_mw == 6.0
    assert rep.outside_grid_turbines == 1 and rep.outside_grid_mw == 4.0


def test_border_turbines_are_logged(caplog, fleet_result):
    ds = era5.engineer_features(era5.canonicalise(_small_era5()))
    with caplog.at_level(logging.WARNING, logger="windpower.simulate"):
        res = S.simulate_archetypes(ds, fleet_result.specs, engine="direct")
    assert any("outside their nearest cell's box" in r.message for r in caplog.records)
    assert res.notes and "not strictly inside" in res.notes[0]
    assert sum(r.outside_cell_box_turbines for r in res.layout.values()) >= 2          # edge + outside-grid fixture turbines


# --- single turbine / single cell ----------------------------------------------------------------
@pytest.mark.parametrize("engine", ["atlite", "direct"])
def test_single_turbine_single_cell_matches_hand_value(engine):
    ds = grid_ds(np.array([[20.0, 20.0, 20.0], [20.0, 20.0, 20.0]]), hours=1)
    ds["speed_100"].values[0, 1, 1] = 8.0                                            # only the turbine's cell has 8 m/s
    res = S.simulate_archetypes(ds, specs_for([(60.25, 21.25, 3.0, 300.0, 100.0)]), engine=engine)
    assert res.installed_mw == 3.0 and list(res.archetypes["archetype"]) == ["MidSP_LowHub"]
    assert res.theoretical_mw.iloc[0] == pytest.approx(hand_power(8.0), rel=1e-6)    # 1.3987 MW
    assert res.hourly["MidSP_LowHub"].iloc[0] == res.theoretical_mw.iloc[0]
    assert res.archetypes["mean_sp_wm2"].item() == 300.0 and res.archetypes["installed_mw"].item() == 3.0


@pytest.mark.parametrize("engine", ["atlite", "direct"])
def test_two_turbines_scale_with_capacity_and_rated_cap(engine):
    spec = specs_for([(60.0, 21.0, 3.0, 300.0, 100.0), (60.0, 21.0, 3.0, 300.0, 100.0)])
    res = S.simulate_archetypes(grid_ds([[8.0, 1, 1], [1, 1, 1]], hours=1), spec, engine=engine)
    assert res.theoretical_mw.iloc[0] == pytest.approx(2 * hand_power(8.0), rel=1e-6)
    gale = S.simulate_archetypes(grid_ds(15.0), spec, engine=engine)                  # above rated everywhere
    assert gale.theoretical_mw.iloc[0] == pytest.approx(6.0, rel=1e-6)                 # = installed, CF 1
    out = S.simulate_archetypes(grid_ds(26.0), spec, engine=engine)                    # above cut-out
    assert out.theoretical_mw.iloc[0] == 0.0
    calm = S.simulate_archetypes(grid_ds(2.5), spec, engine=engine)                    # below cut-in grid point (linear ramp 2.5->3.0)
    assert calm.theoretical_mw.iloc[0] == 0.0


@pytest.mark.parametrize("engine", ["atlite", "direct"])
def test_hub_height_extrapolation_in_simulation(engine):
    ds = grid_ds(8.0, z0=0.05)
    res = S.simulate_archetypes(ds, specs_for([(60.0, 21.0, 3.0, 300.0, 150.0)]), engine=engine)
    v_hub = 8.0 * math.log(150 / 0.05) / math.log(100 / 0.05)                           # 8.4264 m/s
    lo, hi = 8.0, 8.5
    expected = hand_power(lo) + (hand_power(hi) - hand_power(lo)) * (v_hub - lo) / (hi - lo)
    assert res.theoretical_mw.iloc[0] == pytest.approx(expected, rel=1e-5)


def test_density_normalisation_in_simulation():
    thin = S.simulate_archetypes(grid_ds(8.0, rho=1.0), specs_for([(60.0, 21.0, 3.0, 300.0, 100.0)]), engine="direct")
    v_eff = 8.0 * (1.0 / 1.225) ** (1 / 3)
    assert thin.theoretical_mw.iloc[0] == pytest.approx(S.power_from_speed(thin.curves["MidSP_LowHub"], v_eff), rel=1e-9)
    assert thin.theoretical_mw.iloc[0] < hand_power(8.0)


# --- archetypes -------------------------------------------------------------------------------------
def test_nine_archetype_names_and_order():
    combos = [(61.0, 22.0, 3.0, sp, hub) for sp in (200.0, 300.0, 400.0) for hub in (60.0, 120.0, 160.0)]
    names = [n for n, _ in S.archetype_groups(specs_for(combos))]
    assert names == [f"{a}_{b}" for a in ("LowSP_Modern", "MidSP", "HighSP_Legacy") for b in ("LowHub", "MidHub", "HighHub")]


def test_sp_clip_group_has_170_mean():
    spec = specs_for([(60.0, 21.0, 2.0, 100.0, 60.0)])                                  # SP < 170 -> clipped to 170
    assert spec["specific_power_wm2"].item() == 170.0 and spec["sp_bin"].astype(str).item() == "LowSP_Modern"


# --- fixture-level unit checks ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sim48(fleet_result):
    ds = era5.engineer_features(era5.canonicalise(_small_era5()))
    return ds, S.simulate_archetypes(ds, fleet_result.specs, engine="direct")


def _small_era5():
    from windpower import fixtures as F

    return F.synthetic_era5(C.SMOKE_GRID, T0, 48, seed=13)


def test_atlite_and_direct_engines_agree(sim48, fleet_result):
    ds, direct = sim48
    viaatlite = S.simulate_archetypes(ds, fleet_result.specs, engine="atlite")
    assert list(viaatlite.hourly.columns) == list(direct.hourly.columns)
    np.testing.assert_allclose(viaatlite.hourly.to_numpy(), direct.hourly.to_numpy(), rtol=1e-4, atol=1e-4)


def test_totals_capacity_factor_and_energy(sim48, fleet_result):
    _, res = sim48
    h = res.hourly
    np.testing.assert_allclose(h["theoretical_wind_mw"], h.drop(columns="theoretical_wind_mw").sum(axis=1))
    assert res.installed_mw == pytest.approx(fleet_result.specs["cap_mw"].sum())
    assert res.archetypes["installed_mw"].sum() == pytest.approx(res.installed_mw, abs=0.05 * len(res.archetypes))
    cf = S.capacity_factor(res.theoretical_mw, res.installed_mw)
    assert (cf >= 0).all() and (cf <= 1.0 + 1e-9).all() and cf.max() > 0.0
    for name, curve in res.curves.items():                                         # each archetype stays below its installed MW
        assert (h[name] <= curve["installed_mw"] + 1e-9).all() and len(curve["V"]) == 61 and len(curve["POW"]) == 61
    assert h.index.is_monotonic_increasing and h.index.name == "valid_time" and len(h) == 48
    assert S.energy_mwh(h["theoretical_wind_mw"]).sum() == pytest.approx(h["theoretical_wind_mw"].sum())   # MW x 1 h


def test_daily_summary_units(sim48):
    _, res = sim48
    df = res.hourly[["theoretical_wind_mw"]].copy()
    df["actual_wind_mw"] = df["theoretical_wind_mw"] * 0.8
    df.iloc[3, 1] = np.nan
    d = S.daily_summary(df)
    day1 = df.iloc[:24]
    assert d["theoretical_gwh"].iloc[0] == pytest.approx(day1["theoretical_wind_mw"].sum() / 1000.0)    # GWh = sum(MW x 1 h)/1000
    assert d["hours"].tolist() == [23, 24]                                                           # hours with an actual value
    assert d["actual_gwh"].iloc[1] == pytest.approx(df.iloc[24:]["actual_wind_mw"].sum() / 1000.0)
    assert d["capture_ratio_pct"].iloc[1] == pytest.approx(80.0)
    assert d["mean_theoretical_mw"].iloc[0] == pytest.approx(day1["theoretical_wind_mw"].mean())
    plain = S.daily_summary(df[["theoretical_wind_mw"]], actual=None)
    assert list(plain.columns) == ["hours", "theoretical_gwh", "mean_theoretical_mw"] and plain["hours"].tolist() == [24, 24]


def test_invalid_engine(fleet_result):
    with pytest.raises(ValueError, match="engine"):
        S.simulate_archetypes(grid_ds(5.0), fleet_result.specs, engine="magic")
