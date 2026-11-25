from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from windpower import config as C
from windpower import fixtures as F
from windpower import fleet

X0, Y0 = 400_000.0, 6_800_000.0  # EPSG:3067 origin for metric test layouts


def lonlat(xs, ys):
    """EPSG:3067 metres -> (lon, lat) arrays."""
    pts = gpd.GeoSeries([Point(x, y) for x, y in zip(xs, ys)], crs=C.CRS_METRIC).to_crs(C.CRS_GEO)
    return pts.x.to_numpy(), pts.y.to_numpy()


def make_off(xs, ys, **cols) -> gpd.GeoDataFrame:
    lon, lat = lonlat(xs, ys)
    n = len(xs)
    d = {"off_name": [f"Farm{i}" for i in range(n)], "off_status": "Toiminnassa", "off_capacity_mw": np.nan,
         "off_hub_height_m": np.nan, "off_rotor_diam_m": np.nan, "off_manufacturer": None, "off_model": None,
         "off_year": np.nan, "off_lat": lat, "off_lon": lon}
    d.update(cols)
    return gpd.GeoDataFrame(pd.DataFrame(d), geometry=gpd.points_from_xy(lon, lat), crs=C.CRS_GEO)


def make_osm(xs, ys, **cols) -> gpd.GeoDataFrame:
    lon, lat = lonlat(xs, ys)
    n = len(xs)
    d = {"osm_id": [str(100 + i) for i in range(n)], "name": [f"osm{i}" for i in range(n)], "lat": lat, "lon": lon,
         "osm_capacity_mw": np.nan, "osm_hub_height_m": np.nan, "osm_rotor_diam_m": np.nan, "osm_manufacturer": None,
         "osm_model": None, "osm_year": np.nan}
    d.update(cols)
    return gpd.GeoDataFrame(pd.DataFrame(d), geometry=gpd.points_from_xy(lon, lat), crs=C.CRS_GEO)


# --- parse helpers -------------------------------------------------------------
@pytest.mark.parametrize("val, expected", [
    ("3,6 MW", 3.6), ("3.6MW", 3.6), ("3600 kW", 3.6), ("0.0036 GW", 3.6), ("4", 4.0), (2, 2.0),
    ("Nimellisteho 2,5 mw", 2.5), (None, np.nan), (np.nan, np.nan), ("n/a", np.nan),
])
def test_parse_mw(val, expected):
    got = fleet.parse_mw(val)
    assert (math.isnan(got) and math.isnan(expected)) or got == pytest.approx(expected)


def test_parse_mw_default_unit():
    assert fleet.parse_mw("3600", default_unit="kw") == pytest.approx(3.6)
    assert fleet.parse_mw("3600 MW", default_unit="kw") == 3600.0  # explicit unit wins


@pytest.mark.parametrize("val, expected", [("120 m", 120.0), ("98,5", 98.5), (112, 112.0), ("x", np.nan), (None, np.nan)])
def test_parse_meters(val, expected):
    got = fleet.parse_meters(val)
    assert (math.isnan(got) and math.isnan(expected)) or got == expected


@pytest.mark.parametrize("val, expected", [
    ("2012-05-01", 2012), ("since 1999", 1999), (1985, 1985), ("1975", np.nan), ("2045", np.nan), ("2039", 2039),
    (None, np.nan), ("n/a", np.nan),
])
def test_parse_year(val, expected):
    got = fleet.parse_year(val)
    assert (math.isnan(got) and math.isnan(expected)) or got == expected


def test_find_col_keyword_priority():
    cols = ["OBJECTID", "Hankkeen_nimi", "Name_alt", "Teho_MW", "Napakorkeus_m", "geometry"]
    assert fleet.find_col(cols, ["name", "nimi"]) == "Name_alt"      # keyword order beats column order
    assert fleet.find_col(cols, ["nimi", "name"]) == "Hankkeen_nimi"
    assert fleet.find_col(cols, ["hub", "napakorkeus"]) == "Napakorkeus_m"
    assert fleet.find_col(cols, ["rotor"]) is None


@pytest.mark.parametrize("val, expected", [
    ("Suunnitteilla", True), ("Under CONSTRUCTION", True), ("Rakenteilla", True), ("Purettu", True),
    ("ei käytössä", True), ("Toiminnassa", False), ("Operational", False), (None, False), (np.nan, False), ("", False),
])
def test_is_inactive_text(val, expected):
    assert fleet.is_inactive_text(val) is expected


# --- cleaning -----------------------------------------------------------------
def test_clean_osm_filters_inactive_and_bad_values():
    base = {"power": "generator", "generator:source": "wind"}
    els = [
        {"type": "node", "id": 1, "lat": 61.0, "lon": 22.0, "tags": dict(base, **{"generator:output:electricity": "3 MW"})},
        {"type": "node", "id": 2, "lat": 61.1, "lon": 22.0, "tags": dict(base, status="proposed")},
        {"type": "node", "id": 3, "lat": 61.2, "lon": 22.0, "tags": dict(base, **{"disused:power": "generator"})},
        {"type": "way", "id": 4, "center": {"lat": 61.3, "lon": 22.0}, "tags": dict(base, start_date="2031-01-01")},
        {"type": "way", "id": 5, "center": {"lat": 61.4, "lon": 22.0}, "tags": dict(base, **{"generator:output:electricity": "40 MW", "height:hub": "120 m"})},
        {"type": "node", "id": 6, "tags": base},  # no coordinates
    ]
    gdf, stats = fleet.clean_osm({"elements": els})
    assert stats == {"elements": 6, "kept": 2, "skipped_inactive": 3}
    assert list(gdf["osm_id"]) == ["1", "5"]
    assert gdf["osm_capacity_mw"].iloc[0] == 3.0 and math.isnan(gdf["osm_capacity_mw"].iloc[1])  # 40 MW out of bounds
    assert gdf["osm_hub_height_m"].iloc[1] == 120.0
    assert gdf.crs.to_string() == C.CRS_GEO
    assert fleet.clean_osm({"elements": []})[0].empty


def test_clean_arcgis_fixture_stats_and_mapping(fleet_cache, fleet_fx):
    raw, meta = fleet.load_arcgis_cache(fleet_cache)
    gdf, mapping, stats = fleet.clean_arcgis(raw)
    assert mapping == {"name": "Nimi", "status": "Tila", "capacity": "Teho_MW", "hub": "Napakorkeus_m",
                       "rotor": "Roottorin_halkaisija_m", "manufacturer": "Valmistaja", "model": "Turbiinimalli",
                       "year": "Kayttoonottovuosi"}
    assert stats["kept"] == fleet_fx.expected["arcgis_kept"]
    assert stats["skipped_inactive_or_future"] == fleet_fx.expected["arcgis_filtered"]
    assert stats["capacity_out_of_range"] == fleet_fx.expected["arcgis_capacity_out_of_range"]
    assert stats["status_counts"]["Suunnitteilla"] == 1
    assert gdf["off_capacity_mw"].dropna().between(C.MIN_TURBINE_MW, C.MAX_TURBINE_MW).all()
    assert not gdf["off_status"].apply(fleet.is_inactive_text).any()
    assert (gdf["off_year"].dropna() <= C.CURRENT_MAX_YEAR).all()
    assert meta["retrieved_utc"]


def test_clean_arcgis_kw_column_and_missing_fields():
    raw = gpd.GeoDataFrame({"Nimi": ["a", "b"], "Teho_kW": [3600, 2500]}, geometry=[Point(23, 61), Point(23.1, 61)], crs=C.CRS_GEO)
    gdf, mapping, stats = fleet.clean_arcgis(raw)
    assert list(gdf["off_capacity_mw"]) == [3.6, 2.5]            # unit-less numbers follow the "kw" column name
    assert mapping["status"] is None and (gdf["off_status"] == "Operational").all()
    assert gdf["off_hub_height_m"].isna().all() and stats["kept"] == 2


# --- 150 m matching -----------------------------------------------------------
@pytest.mark.parametrize("dist, matched", [(149.0, True), (151.0, False), (0.0, True)])
def test_match_radius_boundaries(dist, matched):
    off = make_off([X0], [Y0], off_capacity_mw=3.0)
    osm = make_osm([X0 + dist], [Y0], osm_capacity_mw=2.0)
    out, stats = fleet.match_and_merge(off, osm)
    assert (stats["matched_osm_unique"] == 1) is matched
    assert len(out) == (1 if matched else 2)
    if matched:
        assert out["source"].tolist() == ["arcgis+osm"] and out["match_dist_m"].iloc[0] == pytest.approx(dist, abs=0.01)
    else:
        assert sorted(out["source"]) == ["arcgis", "osm"] and stats["osm_only_appended"] == 1


def test_match_uses_nearest_osm_point_and_each_osm_once():
    off = make_off([X0], [Y0])
    osm = make_osm([X0 + 100, X0 + 50], [Y0, Y0], osm_capacity_mw=[1.0, 2.0])
    out, stats = fleet.match_and_merge(off, osm)
    assert stats["matched_osm_unique"] == 1 and stats["osm_only_appended"] == 1
    assert out.loc[out["source"] == "arcgis+osm", "turbine_id"].item() == "101"   # the 50 m one
    assert len(out) == 2


def test_arcgis_fields_preferred_osm_fills_gaps():
    off = make_off([X0, X0 + 1000], [Y0, Y0], off_name=["ArcFarm", None], off_capacity_mw=[3.0, np.nan],
                   off_hub_height_m=[np.nan, 100.0], off_year=[2015.0, np.nan], off_manufacturer=["ArcMan", None])
    osm = make_osm([X0 + 10, X0 + 1010], [Y0, Y0], name=["OsmFarm", "OsmFarm2"], osm_capacity_mw=[2.5, 2.2],
                   osm_hub_height_m=[90.0, 80.0], osm_year=[2010.0, 2011.0], osm_manufacturer=["OsmMan", "OsmMan2"])
    out, _ = fleet.match_and_merge(off, osm)
    a, b = out.iloc[0], out.iloc[1]
    assert (a.farm_name, a.capacity_mw, a.hub_height_m, a.commissioning_year, a.manufacturer) == ("ArcFarm", 3.0, 90.0, 2015.0, "ArcMan")
    assert (b.farm_name, b.capacity_mw, b.hub_height_m, b.commissioning_year, b.manufacturer) == ("OsmFarm2", 2.2, 100.0, 2011.0, "OsmMan2")
    assert a.official_capacity_mw == 3.0 and a.osm_capacity_mw == 2.5 and math.isnan(b.official_capacity_mw)
    # notebook behaviour: the matched record takes the OSM coordinates
    assert a.lon == pytest.approx(osm.geometry.x.iloc[0])


def test_merge_with_empty_osm():
    out, stats = fleet.match_and_merge(make_off([X0], [Y0], off_capacity_mw=3.0), fleet.clean_osm({"elements": []})[0])
    assert len(out) == 1 and out["source"].item() == "arcgis" and stats["matched_osm_unique"] == 0


# --- imputation ---------------------------------------------------------------
def _imputation_frame():
    xs = [0, 50_000, 2_900, 51_500, 100_000, 4_800]
    lon, lat = lonlat([X0 + x for x in xs], [Y0] * 6)
    nan = np.nan
    return pd.DataFrame({
        "lat": lat, "lon": lon,
        "capacity_mw": [2.0, 4.0, nan, nan, nan, nan],
        "hub_height_m": [80.0, 120.0, nan, nan, nan, nan],
        "rotor_diameter_m": [90.0, 110.0, nan, nan, nan, nan],
        "commissioning_year": [2010.0, 2015.0, nan, nan, nan, nan],
    })


def test_imputation_radii_values_and_flags():
    out = fleet.impute_attributes(_imputation_frame())
    K1, K2, U1, U2, U3, U4 = (out.iloc[i] for i in range(6))
    for k in (K1, K2):  # observed rows are never flagged
        assert not (k.capacity_imputed or k.hub_imputed or k.rotor_imputed or k.year_imputed)
        assert k.capacity_impute_method == ""
    # U1: 2.9 km from K1 -> cap/hub/rotor imputed (<=5/3/3 km); year radius 2 km exceeded -> stays missing
    assert (U1.capacity_mw, U1.hub_height_m, U1.rotor_diameter_m) == (2.0, 80.0, 90.0)
    assert U1.capacity_impute_method == "nearest_known_mean_within_5000m"
    assert U1.hub_impute_method == "nearest_known_mean_within_3000m"
    assert U1.rotor_impute_method == "nearest_known_mean_within_3000m"
    assert math.isnan(U1.commissioning_year) and not U1.year_imputed and U1.year_impute_method == ""
    # U2: 1.5 km from K2 -> everything incl. year (<=2 km)
    assert (U2.capacity_mw, U2.hub_height_m, U2.rotor_diameter_m, U2.commissioning_year) == (4.0, 120.0, 110.0, 2015.0)
    assert U2.year_impute_method == "nearest_known_median_within_2000m" and U2.year_imputed
    # U3: far from everything -> fleet-median fallback. Capacity median includes neighbour-imputed U1,U2,U4:
    # median(2,4,2,4,2) = 2; hub median(80,120,80,120) = 100; rotor median(90,110,90,110) = 100
    assert U3.capacity_mw == 2.0 and U3.capacity_impute_method == "fleet_median_fallback" and U3.capacity_imputed
    assert U3.hub_height_m == 100.0 and U3.rotor_diameter_m == 100.0
    assert math.isnan(U3.commissioning_year)
    # U4: 4.8 km from K1 -> capacity by neighbour (<=5 km), hub/rotor beyond 3 km -> median fallback
    assert U4.capacity_mw == 2.0 and U4.capacity_impute_method == "nearest_known_mean_within_5000m"
    assert U4.hub_height_m == 100.0 and U4.hub_impute_method == "fleet_median_fallback"


def test_imputation_just_outside_radius_falls_back():
    df = _imputation_frame().iloc[[0, 1]].copy()
    lon, lat = lonlat([X0 + 5_100], [Y0])  # 5.1 km from K1 (capacity radius 5 km)
    df.loc[2] = [lat[0], lon[0], np.nan, np.nan, np.nan, np.nan]
    out = fleet.impute_attributes(df)
    assert out.loc[2, "capacity_impute_method"] == "fleet_median_fallback" and out.loc[2, "capacity_mw"] == 3.0


def test_imputation_does_not_mutate_input():
    df = _imputation_frame()
    before = df.copy()
    fleet.impute_attributes(df)
    pd.testing.assert_frame_equal(df, before)


# --- grid / aggregation ---------------------------------------------------------
def test_grid_polygons_orientation(smoke_axes):
    lats, lons = smoke_axes
    g = fleet.grid_cell_polygons(lats, lons)
    assert len(g) == lats.size * lons.size
    for i, j in [(0, 0), (2, 5), (lats.size - 1, lons.size - 1), (12, 0)]:
        row = g[(g.pixel_i == i) & (g.pixel_j == j)].iloc[0]
        assert (row.era5_lat, row.era5_lon) == (lats[i], lons[j])
        c = row.geometry.centroid
        assert (c.y, c.x) == pytest.approx((lats[i], lons[j]))
        assert row.geometry.bounds == pytest.approx((lons[j] - 0.125, lats[i] - 0.125, lons[j] + 0.125, lats[i] + 0.125))


def _handmade(points, caps):
    return pd.DataFrame({"turbine_id": [f"t{k}" for k in range(len(points))], "lat": [p[0] for p in points],
                         "lon": [p[1] for p in points], "capacity_mw": caps, "hub_height_m": 100.0,
                         "rotor_diameter_m": 100.0, "commissioning_year": 2015.0})


def test_capacity_matrix_orientation_non_square(smoke_axes):
    lats, lons = smoke_axes                     # 13 x 17: a transposed matrix would not even fit
    grid = fleet.grid_cell_polygons(lats, lons)
    pts = [(lats[2] + 0.03, lons[5] - 0.04), (lats[2], lons[5]), (lats[12], lons[16]), (lats[0], lons[0])]
    df = fleet.assign_cells(_handmade(pts, [1.0, 2.0, 4.0, 8.0]), grid)
    agg, pixels, m = fleet.aggregate_pixels(df, grid, (lats.size, lons.size))
    assert m.shape == (13, 17)
    assert m[2, 5] == 3.0 and m[12, 16] == 4.0 and m[0, 0] == 8.0 and m.sum() == 15.0
    assert pixels.loc[(pixels.pixel_i == 2) & (pixels.pixel_j == 5), "turbine_count"].item() == 2
    assert fleet.capacity_totals(df, agg, m)["active_cells"] == 3


def test_edge_and_outside_points_are_unassigned(smoke_axes):
    lats, lons = smoke_axes
    grid = fleet.grid_cell_polygons(lats, lons)
    df = fleet.assign_cells(_handmade([(lats[3] + 0.125, lons[3]), (lats[3], lons[3] + 0.125), (75.0, 25.0),
                                       (lats[3] + 0.1249, lons[3])], [1.0, 1.0, 1.0, 1.0]), grid)
    assert df["pixel_i"].isna().tolist() == [True, True, True, False]
    assert df.loc[3, "pixel_i"] == 3
    agg, _, m = fleet.aggregate_pixels(df, grid, (lats.size, lons.size))
    t = fleet.capacity_totals(df, agg, m)
    assert t["unassigned_turbines"] == 3 and t["unassigned_mw"] == 3.0 and t["matrix_mw"] == 1.0


# --- full offline build on the fixture --------------------------------------------
def test_fixture_counts_match_generator(fleet_result, fleet_fx):
    s, e = fleet_result.stats, fleet_fx.expected
    assert s["arcgis"]["kept"] == e["arcgis_kept"] and s["osm"]["kept"] == e["osm_kept"]
    assert s["osm"]["skipped_inactive"] == e["osm_skipped_inactive"]
    assert s["merge"]["matched_osm_unique"] == e["osm_matched_unique"]
    assert s["total_turbines"] == e["total_turbines"] == len(fleet_result.turbines)
    assert fleet_result.turbines["turbine_id"].is_unique
    assert set(fleet_result.turbines["source"]) == {"arcgis", "arcgis+osm", "osm"}


def test_capacity_totals_agree_across_representations(fleet_result):
    r, t = fleet_result, fleet_result.totals
    assert t["unassigned_turbines"] >= 2 and t["unassigned_mw"] > 0      # cell-edge + outside-grid turbines
    assert t["turbine_table_mw"] == pytest.approx(t["pixel_agg_mw"] + t["unassigned_mw"], rel=1e-9)
    assert t["turbine_table_mw"] == pytest.approx(t["matrix_mw"] + t["unassigned_mw"], rel=1e-9)
    assert r.pixels["installed_capacity_mw"].sum() == pytest.approx(t["pixel_agg_mw"], rel=1e-9)
    assert r.pixels["turbine_count"].sum() + t["unassigned_turbines"] == len(r.turbines)
    assert t["turbine_table_mw"] == pytest.approx(r.specs["cap_mw"].sum(), rel=1e-9)
    assert t["active_cells"] == len(r.pixel_agg) == int((r.capacity_matrix > 0).sum())


def test_capacity_matrix_matches_independent_cell_assignment(fleet_result, smoke_axes):
    lats, lons = smoke_axes
    expected = np.zeros((lats.size, lons.size))
    for row in fleet_result.turbines.itertuples():
        # independent of shapely: nearest centre, and strictly inside the 0.25 deg box
        i, j = int(np.abs(lats - row.lat).argmin()), int(np.abs(lons - row.lon).argmin())
        inside = abs(row.lat - lats[i]) < 0.125 and abs(row.lon - lons[j]) < 0.125
        if inside:
            expected[i, j] += row.capacity_mw
    np.testing.assert_allclose(fleet_result.capacity_matrix, expected, rtol=1e-9)
    for row in fleet_result.turbines.dropna(subset=["pixel_i"]).itertuples():
        assert (row.era5_lat, row.era5_lon) == (lats[row.pixel_i], lons[row.pixel_j])


def test_cell_edge_turbine_is_unassigned(fleet_result):
    edge = fleet_result.turbines[fleet_result.turbines["farm_name"] == "SYNTHETIC Edge"]
    assert len(edge) == 1 and edge["pixel_i"].isna().all() and edge["source"].item() == "arcgis+osm"
    outside = fleet_result.turbines[fleet_result.turbines["farm_name"] == "SYNTHETIC Outside"]
    assert outside["pixel_i"].isna().all()


def test_duplicate_location_rows_kept_separate(fleet_result):
    dups = fleet_result.turbines[fleet_result.turbines["farm_name"].str.startswith("SYNTHETIC Duplicate", na=False)]
    assert len(dups) == 2 and dups["lat"].nunique() == 1 and dups["lon"].nunique() == 1


def test_fixture_provenance_columns(fleet_result):
    t = fleet_result.turbines
    for p in ("capacity", "hub", "rotor", "year"):
        assert t[f"{p}_imputed"].dtype == bool and f"{p}_impute_method" in t
        assert (t.loc[t[f"{p}_imputed"], f"{p}_impute_method"] != "").all()
        assert (t.loc[~t[f"{p}_imputed"], f"{p}_impute_method"] == "").all()
    assert t["capacity_imputed"].any() and t["hub_imputed"].any()  # fixture drops some values on purpose
    assert t[["capacity_mw", "hub_height_m", "rotor_diameter_m"]].notna().all().all()
    assert fleet_result.snapshot["note"] == "current inventory may differ from the 2025 fleet"
    assert fleet_result.snapshot["osm_base_timestamp_utc"] == F.FIXED_ISO


def test_offline_build_is_deterministic(fleet_cache, smoke_axes, fleet_result):
    again = fleet.build_fleet_from_cache(fleet_cache, *smoke_axes)
    pd.testing.assert_frame_equal(again.turbines, fleet_result.turbines)
    np.testing.assert_array_equal(again.capacity_matrix, fleet_result.capacity_matrix)


def test_missing_cache_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="refresh"):
        fleet.load_osm_cache(tmp_path)
    with pytest.raises(FileNotFoundError, match="refresh"):
        fleet.load_arcgis_cache(tmp_path)


# --- per-turbine spec table (df_t) -----------------------------------------------
def _spec_in(**cols):
    n = len(next(iter(cols.values())))
    d = {"lat": [61.0] * n, "lon": [23.0] * n}
    d.update(cols)
    return pd.DataFrame(d)


def test_spec_table_basic_values_and_bins():
    df = fleet.build_turbine_spec_table(_spec_in(capacity_mw=[3.0, 4.0], hub_height_m=[100.0, 150.0],
                                                 rotor_diameter_m=[100.0, 150.0], commissioning_year=[2015.0, 2020.0]))
    assert df["specific_power_wm2"].iloc[0] == pytest.approx(3e6 / (math.pi * 50**2))
    assert df["sp_bin"].astype(str).tolist() == ["HighSP_Legacy", "LowSP_Modern"]   # 381.97 and 226.4
    assert df["hub_bin"].astype(str).tolist() == ["LowHub", "HighHub"]               # 100 -> (0,100], 150 -> (140,inf)
    assert not df[["cap_era_default", "rotor_inferred", "hub_era_default"]].any().any()


def test_spec_table_era_defaults():
    df = fleet.build_turbine_spec_table(_spec_in(
        capacity_mw=[np.nan] * 4, hub_height_m=[np.nan] * 4, rotor_diameter_m=[np.nan] * 4,
        commissioning_year=[2005.0, 2012.0, 2018.0, 2024.0]))
    assert df["cap_mw"].tolist() == [1.8, 3.2, 4.3, 5.6]
    assert df["hub_m"].tolist() == [75.0, 115.0, 138.0, 152.0]
    sp = [390.0, 320.0, 275.0, 245.0]
    np.testing.assert_allclose(df["rotor_m"], [2 * math.sqrt(c * 1e6 / (math.pi * s)) for c, s in zip(df["cap_mw"], sp)])
    np.testing.assert_allclose(df["specific_power_wm2"], sp)
    assert df[["cap_era_default", "rotor_inferred", "hub_era_default"]].all().all()


def test_spec_table_year_clip_and_median_fill():
    df = fleet.build_turbine_spec_table(_spec_in(capacity_mw=[3.0] * 4, commissioning_year=[1950.0, 2040.0, 2000.0, np.nan]))
    assert df["year_clean"].tolist() == [1990.0, 2026.0, 2000.0, 2000.0]      # NaN -> median of clipped = 2000
    allnan = fleet.build_turbine_spec_table(_spec_in(capacity_mw=[np.nan], commissioning_year=[np.nan]))
    assert allnan["cap_mw"].item() == 5.6                                      # no year information -> newest era


def test_spec_table_tip_height_correction():
    df = fleet.build_turbine_spec_table(_spec_in(capacity_mw=[5.0, 5.0, 5.0], rotor_diameter_m=[150.0, 150.0, 150.0],
                                                 hub_height_m=[200.0, 180.0, 166.0], commissioning_year=[2022.0] * 3))
    assert df["hub_m"].tolist() == [125.0, 180.0, 166.0]    # > 180 m is a tip height; exactly 180 is kept
    assert df["hub_raw_m"].tolist() == [200.0, 180.0, 166.0]
    assert df["hub_tip_corrected"].tolist() == [True, False, False]


def test_spec_table_specific_power_clip():
    df = fleet.build_turbine_spec_table(_spec_in(capacity_mw=[3.0, 8.0], rotor_diameter_m=[300.0, 60.0],
                                                 hub_height_m=[100.0, 100.0], commissioning_year=[2020.0, 2020.0]))
    assert df["specific_power_wm2"].tolist() == [170.0, 450.0]


def test_spec_table_kilowatt_capacity_detection():
    kw = fleet.build_turbine_spec_table(_spec_in(capacity_kw=[3000.0, 2000.0]))
    assert kw["cap_mw"].tolist() == [3.0, 2.0]
    big = fleet.build_turbine_spec_table(_spec_in(capacity_mw=[3000.0, 4000.0]))  # median > 100 -> kW
    assert big["cap_mw"].tolist() == [3.0, 4.0]


def test_spec_table_requires_coordinates():
    with pytest.raises(ValueError, match="latitude/longitude"):
        fleet.build_turbine_spec_table(pd.DataFrame({"capacity_mw": [1.0]}))


def test_fixture_spec_table_bins_cover_all_rows(fleet_result):
    s = fleet_result.specs
    assert s["sp_bin"].notna().all() and s["hub_bin"].notna().all()
    assert s["specific_power_wm2"].between(*C.SP_CLIP).all()
    assert len(s) == len(fleet_result.turbines)


# --- snapshot ------------------------------------------------------------------
def test_fleet_snapshot_block():
    snap = fleet.fleet_snapshot({"retrieved_utc": "a", "last_edit_utc": "b"}, {"retrieved_utc": "c", "base_timestamp_utc": "d"})
    assert snap == {"arcgis_retrieved_utc": "a", "arcgis_last_edit_utc": "b", "osm_base_timestamp_utc": "d",
                    "osm_retrieved_utc": "c", "note": "current inventory may differ from the 2025 fleet"}


# --- fetchers with injected fake transport (no network) ----------------------------
class Resp:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, handler):
        self.handler, self.calls = handler, []

    def get(self, url, params=None, timeout=None):
        self.calls.append(("GET", url, params))
        return self.handler("GET", url, params or {})

    def post(self, url, data=None, timeout=None):
        self.calls.append(("POST", url, data))
        return self.handler("POST", url, data or {})


def test_fetch_osm_mirror_fallback_and_snapshot(fleet_fx):
    sleeps = []

    def handler(method, url, data):
        if "overpass-api" in url:
            return Resp({}, status=504)
        return Resp(fleet_fx.osm)

    sess = FakeSession(handler)
    body, meta = fleet.fetch_osm(sess, attempts=2, sleep=sleeps.append)
    assert body is fleet_fx.osm and meta["mirror"] == C.OSM_MIRRORS[1]
    assert meta["base_timestamp_utc"] == F.FIXED_ISO and meta["element_count"] == len(fleet_fx.osm["elements"])
    assert len(sleeps) == 1                                   # one backoff between the two attempts on mirror 1
    assert sum(1 for c in sess.calls if "overpass-api" in c[1]) == 2


def test_fetch_osm_all_mirrors_fail():
    sess = FakeSession(lambda *a: Resp({}, status=500))
    with pytest.raises(RuntimeError, match="All Overpass mirrors failed"):
        fleet.fetch_osm(sess, attempts=1, sleep=lambda s: None)


def test_fetch_osm_rejects_in_band_overpass_error():
    sess = FakeSession(lambda *a: Resp({"elements": [], "remark": "runtime error: Query timed out"}))
    with pytest.raises(RuntimeError, match="timed out"):
        fleet.fetch_osm(sess, mirrors=["https://m"], attempts=1, sleep=lambda s: None)


def _arcgis_handler(n_ids, geojson):
    def handler(method, url, p):
        if method == "GET" and url.endswith("FeatureServer"):
            return Resp({"layers": [{"id": 0}], "copyrightText": "svc"})
        if method == "GET":
            return Resp({"name": "Tuulivoimalat", "supportedQueryFormats": "JSON, geoJSON" if geojson else "JSON",
                         "editingInfo": {"dataLastEditDate": 1_759_276_800_000}, "copyrightText": "cr", "licenseInfo": "lic"})
        if p.get("returnIdsOnly"):
            return Resp({"objectIds": list(range(n_ids, 0, -1))})
        ids = [int(i) for i in p["objectIds"].split(",")]
        if geojson:
            return Resp({"features": [{"type": "Feature", "properties": {"OBJECTID": i}, "geometry": {"type": "Point", "coordinates": [23.0, 61.0]}} for i in ids]})
        return Resp({"features": [{"attributes": {"OBJECTID": i}, "geometry": {"x": 23.0, "y": 61.0}} for i in ids]})
    return handler


@pytest.mark.parametrize("geojson", [True, False])
def test_fetch_arcgis_pages_by_500_and_records_snapshot(geojson):
    sess = FakeSession(_arcgis_handler(1200, geojson))
    fc, meta = fleet.fetch_arcgis(sess, sleep=lambda s: None)
    pages = [c for c in sess.calls if c[0] == "POST" and "objectIds" in c[2]]
    assert [len(c[2]["objectIds"].split(",")) for c in pages] == [500, 500, 200]
    assert len(fc["features"]) == 1200 and fc["features"][0]["properties"]["OBJECTID"] == 1   # sorted ids
    assert fc["features"][0]["geometry"] == {"type": "Point", "coordinates": [23.0, 61.0]}
    assert meta["query_format"] == ("geojson" if geojson else "esri-json")
    assert meta["last_edit_utc"] == "2025-10-01T00:00:00Z" and meta["copyright_text"] == "cr" and meta["license_info"] == "lic"
    assert meta["retrieved_utc"].endswith("Z") and meta["object_count"] == 1200


def test_refresh_then_offline_roundtrip(tmp_path, fleet_fx):
    def handler(method, url, p):
        if "interpreter" in url:
            return Resp(fleet_fx.osm)
        return _arcgis_handler(3, True)(method, url, p)

    fleet.refresh_fleet_cache(tmp_path, FakeSession(handler), sleep=lambda s: None)
    osm, osm_meta = fleet.load_osm_cache(tmp_path)
    arc, arc_meta = fleet.load_arcgis_cache(tmp_path)
    assert len(osm["elements"]) == len(fleet_fx.osm["elements"]) and osm_meta["base_timestamp_utc"] == F.FIXED_ISO
    assert len(arc) == 3 and arc_meta["last_edit_utc"] == "2025-10-01T00:00:00Z"
    assert not list(tmp_path.glob("*.part"))
