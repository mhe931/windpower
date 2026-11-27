"""National allocation, capacity reconciliation, national power / energy and comparison metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import Transformer

from windpower import config as C
from windpower import solar_national as N
from windpower import solar_public as SP
from windpower import solar_pv as P

GRID2 = C.GridSpec("t2", 60.0, 60.25, 21.0, 21.25)


def tiny_alloc(large=((10.0, 0.0), (0.0, 5.0)), share=((0.1, 0.2), (0.3, 0.4))) -> N.Allocation:
    l, w = np.array(large), np.array(share)
    return N.Allocation(GRID2, pd.DataFrame(), float(l.sum()), l, w, w * 1000, {})


@pytest.fixture(scope="module")
def alloc(public_cache):
    pub = SP.load_public(public_cache)
    return N.allocate(SP.place_plants(pub.plants, pub.municipalities), pub.population, C.SOLAR_SMOKE_GRID), pub


def test_allocation_on_the_fixture_matches_the_spec(alloc, public_fx):
    a, pub = alloc
    r = a.report
    assert a.large_mw == pytest.approx(250.66, abs=1e-9) and r["plants"] == 30 and r["plants_inside_grid"] == 30 and r["plants_outside_grid"] == []
    assert a.large_by_cell.sum() == pytest.approx(250.66) and r["largest_plant"]["name"] == "SYNTHETIC Plant 00" and r["largest_plant"]["nominal_mw"] == 103.2
    assert a.pop_share.sum() == pytest.approx(1.0, abs=1e-12) and (a.pop_share >= 0).all() and a.pop_share.shape == (7, 9)
    assert r["weights_sum"] == pytest.approx(1.0, abs=1e-12)
    # inhabitants outside the grid are reported, and excluded from the shares
    assert r["inhabitants_outside_grid"] == public_fx.outside_inhabitants == 450 and r["population_cells_outside_grid"] == 3
    assert r["inhabitants_inside_grid"] + r["inhabitants_outside_grid"] == r["inhabitants_total"] == a.pop_by_cell.sum() + 450


def test_plants_go_to_the_cell_containing_their_municipality_centroid(alloc, public_fx):
    a, pub = alloc
    to4326 = Transformer.from_crs("EPSG:3067", "EPSG:4326", always_xy=True)
    code_of = {r["name"]: r["code"] for r in public_fx.plant_rows}
    by_code = {m["properties"]["kunta"]: m["geometry"]["coordinates"][0] for m in public_fx.municipalities}
    for _, p in a.plants.iterrows():
        ring = by_code[code_of[p["name"]]]
        lon, lat = to4326.transform((ring[0][0] + ring[2][0]) / 2, (ring[0][1] + ring[2][1]) / 2)          # centre of the square = its centroid
        ref = P.locate_cell(lat, lon, C.SOLAR_SMOKE_GRID)
        assert (p["i_lat"], p["i_lon"]) == (ref.i_lat, ref.i_lon) and p["municipality_code"] == code_of[p["name"]]
    for (i, j), mw in np.ndenumerate(a.large_by_cell):
        assert mw == pytest.approx(a.plants[(a.plants["i_lat"] == i) & (a.plants["i_lon"] == j)]["nominal_mw"].sum())


def test_population_cells_use_their_centre_not_the_corner(public_fx):
    """x/y are SW corners: a cell whose corner lies just outside the grid edge but whose centre is inside belongs to the grid."""
    g = C.GridSpec("edge", 60.0, 60.0, 21.0, 21.0)                      # one cell: lat 59.875-60.125, lon 20.875-21.125
    to3067 = Transformer.from_crs("EPSG:4326", "EPSG:3067", always_xy=True)
    x, y = to3067.transform(21.0, 60.0)
    x0, y0 = int(x // 1000) * 1000, int(y // 1000) * 1000
    pop = pd.DataFrame({"x": [x0, x0 + 400_000, x0], "y": [y0, y0, y0 + 400_000], "inhabitants": [10, 99, 99]})
    placed = pd.DataFrame({"name": ["p"], "nominal_mw": [2.0], "municipality": ["m"], "lat": [60.0], "lon": [21.0]})
    a = N.allocate(placed, pop, g)
    assert a.pop_by_cell[0, 0] == 10 and a.report["inhabitants_outside_grid"] == 198 and a.pop_share[0, 0] == 1.0 and a.large_by_cell[0, 0] == 2.0


def test_plant_outside_the_grid_is_reported_and_excluded_from_L():
    placed = pd.DataFrame({"name": ["in", "out"], "nominal_mw": [10.0, 40.0], "municipality": ["a", "b"], "lat": [60.1, 70.0], "lon": [21.1, 40.0]})
    pop = pd.DataFrame({"x": [0.0], "y": [0.0], "inhabitants": [1]})
    with pytest.raises(N.AllocationError):                              # (0, 0) EPSG:3067 is far outside: no inhabitants inside the grid
        N.allocate(placed, pop, GRID2)
    to3067 = Transformer.from_crs("EPSG:4326", "EPSG:3067", always_xy=True)
    x, y = to3067.transform(21.1, 60.1)
    a = N.allocate(placed, pd.DataFrame({"x": [int(x // 1000) * 1000], "y": [int(y // 1000) * 1000], "inhabitants": [7]}), GRID2)
    assert a.large_mw == 10.0 and a.report["large_outside_grid_mw"] == 40.0 and a.report["plants_outside_grid"][0]["name"] == "out"
    assert a.report["large_total_mw"] == 50.0


def test_capacity_cube_is_exactly_the_spec_formula_and_reconciles():
    a = tiny_alloc()
    c = pd.Series([100.0, 200.0, np.nan, 15.0])
    cube = N.capacity_cube(c, a)
    manual = np.array([(100 - 15) * w + l for w, l in zip(a.pop_share.ravel(), a.large_by_cell.ravel())]).reshape(2, 2)
    np.testing.assert_allclose(cube[0], manual, atol=1e-12)
    assert np.isnan(cube[2]).all() and np.allclose(cube[3], a.large_by_cell)                      # C == L: everything is the large plants
    rep = N.check_reconciliation(cube, c)
    assert rep["ok"] and rep["hours_checked"] == 3 and rep["hours_nan"] == 1 and rep["max_abs_diff_mw"] <= 1e-9
    pop = N.capacity_cube(c, a, population_only=True)
    np.testing.assert_allclose(pop[1], 200.0 * a.pop_share)
    assert N.check_reconciliation(pop, c)["ok"]
    bad = cube.copy()
    bad[0, 0, 0] += 1e-6
    assert not N.check_reconciliation(bad, c)["ok"]                                              # a 1e-6 MW leak is caught at 1e-9
    with pytest.raises(N.AllocationError, match="below the large-plant capacity"):
        N.capacity_cube(pd.Series([10.0]), a)


def test_reconciliation_holds_on_the_real_fixture_for_every_hour(alloc):
    a, _ = alloc
    c = pd.Series(1400.0 + 0.4 * np.arange(48))
    for pop_only in (False, True):
        assert N.check_reconciliation(N.capacity_cube(c, a, population_only=pop_only), c)["max_abs_diff_mw"] <= 1e-9


def test_national_power_is_sum_of_cf_times_installed_mw_and_keeps_nan_hours():
    a = tiny_alloc()
    cf = xr.DataArray(np.array([[[0.5, 0.0], [1.0, 0.2]], [[0.1, 0.1], [0.1, 0.1]], [[0.3, 0.3], [0.3, 0.3]]]), dims=("time", "latitude", "longitude"),
                      coords={"time": pd.date_range("2025-07-01", periods=3, freq="h"), "latitude": [60.0, 60.25], "longitude": [21.0, 21.25]})
    cap = pd.Series([100.0, 200.0, np.nan])
    p = N.national_power(cf, N.capacity_cube(cap, a))
    cube = N.capacity_cube(cap, a)
    assert p[0] == pytest.approx(0.5 * cube[0, 0, 0] + 0.0 + 1.0 * cube[0, 1, 0] + 0.2 * cube[0, 1, 1])
    assert p[1] == pytest.approx(0.1 * 200.0) and np.isnan(p[2])                                   # CF 0.1 everywhere = 10 % of C(t); missing C stays NaN


def test_energy_pearson_and_daily_cycle_centre():
    idx = pd.date_range("2025-07-01", periods=72, freq="h")
    mw = pd.Series(np.maximum(0, np.cos((idx.hour + 0.5 - 10.5) / 24 * 2 * np.pi)), index=idx) * 100       # symmetric around 10:30 -> centre 10:30
    e = N.energy_gwh(mw)
    assert e["gwh"] == pytest.approx(mw.sum() / 1000.0) and e["hours"] == 72 and e["hours_missing"] == 0
    assert N.energy_gwh(mw.where(mw.index.hour != 3))["hours_missing"] == 3
    c = N.daily_cycle_centre(mw)
    assert c["minutes_utc"] == pytest.approx(10 * 60 + 30, abs=0.5) and c["hhmm_utc"] == "10:30"
    assert N.daily_cycle_centre(pd.Series(0.0, index=idx))["hhmm_utc"] is None
    r = N.pearson(mw, mw * 2 + 5)
    assert r["pearson_r"] == pytest.approx(1.0) and "not regression R2" in r["label"] and r["n_hours"] == 72
    assert N.pearson(mw, pd.Series(1.0, index=idx))["pearson_r"] is None                          # constant series: undefined, not NaN/0


def test_comparison_metrics_label_references_and_energy_ratio():
    idx = pd.date_range("2025-07-01", periods=48, freq="h")
    base = pd.Series(np.maximum(0, np.sin((idx.hour - 4) / 16 * np.pi)), index=idx) * 100
    t = pd.DataFrame({"era5_only_mw": base, "hybrid_mw": base * 1.1, "forecast_248_mw": base * 1.2, "surplus_362_mw": base * 0.1})
    m = N.comparison_metrics(t)
    assert set(m["models"]) == {"era5_only_mw", "hybrid_mw"} and set(m["references"]) == {"fingrid_248", "fingrid_362_av06"}
    v = m["models"]["era5_only_mw"]["vs_fingrid_248"]
    assert v["pearson_r"] == pytest.approx(1.0) and v["energy_ratio_model_over_reference"] == pytest.approx(1 / 1.2) and "regression R2" in v["label"]
    assert m["models"]["hybrid_mw"]["vs_fingrid_248"]["energy_ratio_model_over_reference"] == pytest.approx(1.1 / 1.2)
    assert "surplus only" in m["models"]["era5_only_mw"]["vs_fingrid_362_av06"]["note"]
    assert N.comparison_metrics(t[["era5_only_mw"]])["references"] == {}                           # absent references are simply absent


def test_ghi_table_and_mean_cf_use_only_covered_cells(sarah_smoke, era5_solar_result, ):
    t = N.ghi_table(era5_solar_result.ds, sarah_smoke.ongrid)
    g = sarah_smoke.ongrid
    sat = (g.source_code == 0) & g.fully_covered[None]
    assert t["cell_hours"] == int(sat.sum()) and t["cells"] == int(g.fully_covered.sum()) > 0
    assert t["sarah_ghi_wm2"] == pytest.approx(float(g.ghi[sat].mean())) and t["sarah_daily_ghi_kwh_m2"] == pytest.approx(t["sarah_ghi_wm2"] * 0.024)
    d = era5_solar_result.ds
    assert t["era5_ghi_wm2"] == pytest.approx(float(((d["influx_direct"] + d["influx_diffuse"]).transpose("time", "y", "x").values)[sat].mean()))
    cf = xr.DataArray(np.arange(2 * 7 * 9, dtype=float).reshape(2, 7, 9) / 200, dims=("time", "latitude", "longitude"))
    assert N.mean_cf(cf) == pytest.approx(float(cf.mean())) and N.mean_cf(cf, g.fully_covered) == pytest.approx(float(cf.values[:, g.fully_covered].mean()))
    assert N.mean_cf_vs_capacity(pd.Series([10.0, 20.0]), pd.Series([100.0, 100.0])) == pytest.approx(0.15)
