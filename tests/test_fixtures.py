from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from windpower import config as C
from windpower import fixtures as F


def test_era5_fixture_is_deterministic():
    a = F.synthetic_era5(C.SMOKE_GRID, "2025-07-01T00:00:00", 24, seed=3)
    b = F.synthetic_era5(C.SMOKE_GRID, "2025-07-01T00:00:00", 24, seed=3)
    c = F.synthetic_era5(C.SMOKE_GRID, "2025-07-01T00:00:00", 24, seed=4)
    xr.testing.assert_identical(a, b)
    assert not np.array_equal(a["u100"].values, c["u100"].values)


def test_era5_fixture_shape_and_stamp():
    ds = F.synthetic_era5(C.SMOKE_GRID, hours=6)
    assert ds["u100"].dims == ("valid_time", "latitude", "longitude")
    assert ds["u100"].shape == (6, C.SMOKE_GRID.n_lat, C.SMOKE_GRID.n_lon) == (6, 13, 17)
    assert ds["latitude"].values[0] > ds["latitude"].values[-1]  # CDS: descending
    assert "SYNTHETIC" in ds.attrs["title"] and ds.attrs["synthetic"] == "SYNTHETIC"
    assert ds["u100"].dtype == np.float32


def test_finland_grid_is_45_by_52():
    ds = F.synthetic_era5(C.FINLAND_GRID, hours=2)
    assert ds["t2m"].shape == (2, 45, 52)
    assert float(ds["latitude"].max()) == 70.5 and float(ds["longitude"].min()) == 19.0
    assert float(ds["longitude"].max()) == 31.75


def test_fleet_fixture_deterministic_and_stamped():
    a = F.synthetic_fleet(C.SMOKE_GRID, seed=5)
    b = F.synthetic_fleet(C.SMOKE_GRID, seed=5)
    assert a.arcgis == b.arcgis and a.osm == b.osm and a.expected == b.expected
    assert a.arcgis["synthetic"] == "SYNTHETIC" and a.osm["synthetic"] == "SYNTHETIC"
    assert all(f["properties"]["Nimi"].startswith("SYNTHETIC") for f in a.arcgis["features"])
    assert a.osm["osm3s"]["timestamp_osm_base"]


def test_fleet_fixture_edge_cases_present(fleet_fx):
    statuses = {f["properties"]["Tila"] for f in fleet_fx.arcgis["features"]}
    assert {"Suunnitteilla", "Rakenteilla", "Purettu"} <= statuses
    coords = [tuple(f["geometry"]["coordinates"]) for f in fleet_fx.arcgis["features"]]
    assert len(coords) != len(set(coords))  # duplicate-location entries
    assert any("construction:power" in e["tags"] for e in fleet_fx.osm["elements"])
    assert {"node", "way"} == {e["type"] for e in fleet_fx.osm["elements"]}


def test_cache_writers_files_and_zip(tmp_path):
    import zipfile

    paths = F.write_era5_cache(tmp_path, C.SMOKE_GRID, "2025-07-31T00:00:00", 48)
    assert [p.name for p in paths] == ["era5_2025_07.nc", "era5_2025_08.nc"]
    assert not zipfile.is_zipfile(paths[0]) and zipfile.is_zipfile(paths[1])
    with zipfile.ZipFile(paths[1]) as zf:
        assert sorted(zf.namelist()) == ["data_stream-oper_stepType-accum.nc", "data_stream-oper_stepType-instant.nc"]


def test_offset_deg_distance():
    import math

    lat, lon = F.offset_deg(61.0, 23.0, 100.0, 0.0)
    d = (lon - 23.0) * 111_320.0 * math.cos(math.radians(61.0))
    assert d == pytest.approx(100.0, rel=1e-9) and lat == 61.0
