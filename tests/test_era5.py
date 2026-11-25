from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from windpower import config as C
from windpower import era5
from windpower import fixtures as F
from windpower.era5 import ValidationError
from windpower.io_utils import CredentialUnavailableError

from conftest import END, HOURS, START, tiny_ds


# --- ingest ------------------------------------------------------------------
def test_zip_and_netcdf_ingest_are_equivalent(tmp_path, era5_ds):
    nc = F.write_netcdf(era5_ds, tmp_path / "plain.nc")
    zp = F.write_cds_zip(era5_ds, tmp_path / "bundle.nc")  # CDS saves zips under the .nc name
    a = era5.canonicalise(era5.open_cds_file(nc))
    b = era5.canonicalise(era5.open_cds_file(zp))
    assert set(b.data_vars) == set(C.REQUIRED_VARS)
    xr.testing.assert_equal(a[list(C.REQUIRED_VARS)], b[list(C.REQUIRED_VARS)])
    assert "number" not in b.coords and "expver" not in b.coords


def test_zip_without_nc_members_fails(tmp_path):
    import zipfile

    p = tmp_path / "empty.nc"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("readme.txt", "x")
    with pytest.raises(ValueError, match="no .nc files"):
        era5.open_cds_file(p)


def test_descending_latitude_is_canonicalised(era5_ds):
    assert era5_ds["latitude"].values[0] > era5_ds["latitude"].values[-1]
    can = era5.canonicalise(era5_ds)
    assert np.all(np.diff(can["latitude"].values) > 0) and np.all(np.diff(can["longitude"].values) > 0)
    assert can.attrs["source_lat_order"] == "descending"
    # values follow their coordinate, not their position
    pt = dict(latitude=61.5, longitude=23.0, valid_time=era5_ds["valid_time"].values[5])
    assert float(can["u100"].sel(**pt)) == float(era5_ds["u100"].sel(**pt))
    again = era5.canonicalise(can)  # idempotent, keeps the original order record
    assert again.attrs["source_lat_order"] == "descending"


def test_alias_names_and_duplicate_times(era5_ds):
    legacy = era5_ds.rename(valid_time="time", latitude="lat", longitude="lon")
    dup = xr.concat([legacy, legacy.isel(time=[3])], dim="time")
    can = era5.canonicalise(dup)
    assert {"valid_time", "latitude", "longitude"} <= set(can.dims)
    assert can.sizes["valid_time"] == HOURS
    assert pd.DatetimeIndex(can["valid_time"].values).is_monotonic_increasing


def test_load_window_across_plain_and_zip_months(tmp_path):
    F.write_era5_cache(tmp_path, C.SMOKE_GRID, "2025-07-31T00:00:00", 48, seed=9)
    res = era5.load_era5_window(tmp_path, "2025-07-31T00:00:00", "2025-08-01T23:00:00")
    assert res.loaded_files == ["era5_2025_07.nc", "era5_2025_08.nc"] and not res.skipped_files
    info = era5.validate_era5(res.ds, "2025-07-31T00:00:00", "2025-08-01T23:00:00", grid=C.SMOKE_GRID)
    assert info["n_times"] == 48 and info["source_lat_order"] == "descending"
    ref = F.synthetic_era5(C.SMOKE_GRID, "2025-07-31T00:00:00", 48, seed=9)
    xr.testing.assert_equal(res.ds["t2m"], era5.canonicalise(ref)["t2m"])


def test_load_window_skips_tiny_and_broken_files(tmp_path):
    F.write_era5_cache(tmp_path, C.SMOKE_GRID, START, HOURS, zip_months=())
    (tmp_path / "era5_2025_09.nc").write_bytes(b"x" * 10)            # < 1 KiB: ignored
    (tmp_path / "era5_2025_08.nc").write_bytes(b"not netcdf" * 200)  # unreadable: reported
    res = era5.load_era5_window(tmp_path, START, END)
    assert res.loaded_files == ["era5_2025_07.nc"]
    assert list(res.skipped_files) == ["era5_2025_08.nc"]


def test_load_window_without_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        era5.load_era5_window(tmp_path, START, END)


# --- validation --------------------------------------------------------------
def _ok(ds):
    return era5.validate_era5(era5.canonicalise(ds), START, END, grid=C.SMOKE_GRID)


def test_valid_dataset_passes(era5_ds):
    info = _ok(era5_ds)
    assert info["n_times"] == HOURS and info["first_utc"] == "2025-07-01T00:00Z" and info["last_utc"] == "2025-07-02T23:00Z"


def test_time_gap_is_reported_precisely(era5_ds):
    gap = era5.canonicalise(era5_ds).isel(valid_time=[i for i in range(HOURS) if i != 10])
    with pytest.raises(ValidationError, match=r"\[time\] index is not strictly hourly: 2025-07-01T09:00 -> 2025-07-01T11:00") as ei:
        era5.validate_era5(gap)
    assert ei.value.check == "time"


def test_wrong_count_and_window(era5_ds):
    can = era5.canonicalise(era5_ds)
    with pytest.raises(ValidationError, match="expected 72 hours for 2025-07-01T00:00..2025-07-03T23:00, found 48"):
        era5.validate_era5(can, START, "2025-07-03T23:00:00")
    with pytest.raises(ValidationError, match="window is 2025-07-01T00:00..2025-07-02T23:00, expected 2025-07-01T01:00"):
        era5.validate_era5(can, "2025-07-01T01:00:00", "2025-07-03T00:00:00")


def test_duplicate_and_unsorted_times(era5_ds):
    can = era5.canonicalise(era5_ds)
    dup = can.isel(valid_time=[0, 0, 1])
    with pytest.raises(ValidationError, match="duplicate timestamps, first: 2025-07-01T00:00"):
        era5.validate_era5(dup)
    rev = can.isel(valid_time=slice(None, None, -1))
    with pytest.raises(ValidationError, match="not monotonic"):
        era5.validate_era5(rev)


def test_grid_misaligned_and_irregular(era5_ds):
    can = era5.canonicalise(era5_ds)
    shifted = can.assign_coords(latitude=can["latitude"].values + 0.1)
    with pytest.raises(ValidationError, match=r"latitude not aligned to the 0.25 deg lattice"):
        era5.validate_era5(shifted)
    lat = can["latitude"].values.copy()
    lat[4:] += 0.25
    with pytest.raises(ValidationError, match=r"latitude not regular at 0.25 deg: step 0.500000 between index 3"):
        era5.validate_era5(can.assign_coords(latitude=lat))


def test_grid_extent_mismatch_and_orientation(era5_ds):
    can = era5.canonicalise(era5_ds)
    with pytest.raises(ValidationError, match="longitude axis 21.0..25.0 .* != grid finland 19.0..23.0"):
        era5.validate_era5(can, grid=C.GridSpec("finland", 60.0, 63.0, 19.0, 23.0))
    with pytest.raises(ValidationError, match="latitude is not strictly ascending"):
        era5.validate_era5(era5_ds)  # raw CDS orientation must be canonicalised first


def test_missing_variable_and_dims(era5_ds):
    can = era5.canonicalise(era5_ds)
    with pytest.raises(ValidationError, match=r"missing required variable\(s\) \['d2m'\]"):
        era5.validate_era5(can.drop_vars("d2m"))
    with pytest.raises(ValidationError, match="missing dimension 'valid_time'"):
        era5.validate_era5(can.isel(valid_time=0, drop=True))


@pytest.mark.parametrize(
    "var, transform, message",
    [
        ("t2m", lambda a: a - 273.15, r"t2m range .* outside plausible .* K"),
        ("sp", lambda a: a / 100.0, r"sp range .* outside plausible .* Pa"),
        ("lcc", lambda a: a * 100.0, r"lcc range .* outside fraction range"),
        ("tp", lambda a: a * 1000.0 + 1.0, r"tp range .* outside plausible"),
        ("u100", lambda a: a * 0 + 85.0, r"u100 range .* \|wind\| >= 80"),
        ("v10", lambda a: a * 0 - 90.0, r"v10 range .* \|wind\| >= 80"),
    ],
)
def test_unit_and_range_failures(era5_ds, var, transform, message):
    can = era5.canonicalise(era5_ds)
    bad = can.copy()
    bad[var] = transform(can[var]).astype(np.float32)
    bad[var].attrs = can[var].attrs
    with pytest.raises(ValidationError, match=message) as ei:
        era5.validate_era5(bad)
    assert ei.value.check == "range"


def test_units_attribute_mismatch_and_nan(era5_ds):
    can = era5.canonicalise(era5_ds)
    bad = can.copy(deep=True)
    bad["t2m"].attrs["units"] = "C"
    with pytest.raises(ValidationError, match=r"t2m units attribute 'C'"):
        era5.validate_era5(bad)
    nan = can.copy(deep=True)
    nan["sp"].values[0, 0, 0] = np.nan
    with pytest.raises(ValidationError, match=r"sp contains 1 NaN"):
        era5.validate_era5(nan)


# --- features ----------------------------------------------------------------
def _f(da: xr.DataArray) -> float:
    return float(da.values.ravel()[0])


def test_feature_formulas_match_hand_values():
    f = era5.engineer_features(tiny_ds(u10=1.5, v10=2.0, u100=3.0, v100=4.0, t2m=273.15, d2m=272.15, lcc=0.7, tp=0.0))
    v = lambda n: _f(f[n])  # noqa: E731
    assert v("speed_100") == pytest.approx(5.0) and v("speed_10") == pytest.approx(2.5)
    assert v("wind_shear_alpha") == pytest.approx(math.log10(2.0))
    assert v("temp_2m_c") == pytest.approx(0.0) and v("dewpoint_2m_c") == pytest.approx(-1.0)
    assert v("dewpoint_spread_c") == pytest.approx(1.0)
    e = 6.112 * math.exp(17.67 * -1.0 / (-1.0 + 243.5))
    assert v("relative_humidity_pct") == pytest.approx(100 * e / 6.112)
    assert v("air_density_kgm3") == pytest.approx(101325.0 / (287.05 * 273.15))
    assert v("icing_flag") == 1.0
    assert v("icing_severity") == pytest.approx((1 - 4.0 / 11.0) * (1 - 1.0 / 2.5) * 0.7)


def test_shear_alpha_clipped_and_rh_bounds():
    hi = era5.engineer_features(tiny_ds(u10=0.0, v10=0.0, u100=10.0, v100=0.0))
    assert _f(hi["wind_shear_alpha"]) == 0.6          # log10(10/0.1)=2 -> clipped
    lo = era5.engineer_features(tiny_ds(u10=10.0, v10=0.0, u100=5.0, v100=0.0))
    assert _f(lo["wind_shear_alpha"]) == 0.0
    sat = era5.engineer_features(tiny_ds(d2m=280.0, t2m=273.15))  # dewpoint above temperature
    assert _f(sat["relative_humidity_pct"]) == 100.0


@pytest.mark.parametrize(
    "kw, flag, severity",
    [
        (dict(t2m=273.15, d2m=272.15, lcc=0.7, tp=0.0), 1.0, (1 - 4 / 11) * 0.6 * 0.7),
        (dict(t2m=273.15, d2m=272.15, lcc=0.3, tp=0.0), 0.0, None),       # dry sky, no rain
        (dict(t2m=273.15, d2m=272.15, lcc=0.0, tp=0.0002), 1.0, (1 - 4 / 11) * 0.6 * 1.0),  # rain counts as cloud 1
        (dict(t2m=273.15, d2m=270.65, lcc=0.9, tp=0.0), 0.0, 0.0),        # spread 2.5 K: not near saturation, humid_factor 0
        (dict(t2m=273.15 + 1.5, d2m=273.15 + 1.0, lcc=0.9, tp=0.0), 0.0, None),  # warmer than +1 C
        (dict(t2m=273.15 - 15.5, d2m=273.15 - 16.0, lcc=0.9, tp=0.0), 0.0, None),  # colder than -15 C
        (dict(t2m=273.15 - 4.0, d2m=273.15 - 4.0, lcc=1.0, tp=0.0), 1.0, 1.0),   # optimum
        (dict(t2m=273.15 - 14.9, d2m=273.15 - 16.3, lcc=0.6, tp=0.0), 1.0, None),  # lcc threshold is inclusive
    ],
)
def test_icing_rules(kw, flag, severity):
    f = era5.engineer_features(tiny_ds(**kw))
    assert _f(f["icing_flag"]) == flag
    if severity is not None:
        assert _f(f["icing_severity"]) == pytest.approx(severity, abs=1e-6)


def test_engineer_does_not_mutate_input(era5_ds):
    before = set(era5_ds.data_vars)
    era5.engineer_features(era5_ds)
    assert set(era5_ds.data_vars) == before


def test_summary_stats(era5_ds):
    ds = era5.engineer_features(era5.canonicalise(era5_ds))
    s = era5.summary_stats(ds)
    assert s["mean_speed_100"] == pytest.approx(float(np.hypot(ds["u100"], ds["v100"]).mean()))
    assert s["mean_air_density"] == pytest.approx(float((ds["sp"] / (287.05 * ds["t2m"])).mean()), rel=1e-6)
    assert s["icing_share_pct"] == pytest.approx(100 * float(ds["icing_flag"].mean()))
    assert 0.9 < s["mean_air_density"] < 1.4


def test_cold_humid_fixture_triggers_icing():
    ds = era5.engineer_features(era5.canonicalise(F.synthetic_era5(C.SMOKE_GRID, START, 24, seed=1, base_temp_c=-3.0)))
    assert era5.summary_stats(ds)["icing_share_pct"] > 0.0


# --- CDS helper (offline only) --------------------------------------------------
def test_cds_requests_one_per_month():
    reqs = era5.build_cds_requests("2025-07-01T00:00:00", "2025-09-30T23:00:00")
    assert [n for n, _ in reqs] == ["era5_2025_07.nc", "era5_2025_08.nc", "era5_2025_09.nc"]
    body = reqs[2][1]
    assert len(body["day"]) == 30 and body["area"] == [70.5, 19.0, 59.5, 31.75]
    assert len(body["variable"]) == 10 and len(body["time"]) == 24
    assert len(reqs[0][1]["day"]) == 31


def test_cds_credentials_detection(tmp_path):
    assert not era5.cds_credentials_available({}, home=tmp_path)
    assert era5.cds_credentials_available({"CDSAPI_URL": "u", "CDSAPI_KEY": "k"}, home=tmp_path)
    (tmp_path / ".cdsapirc").write_text("url: x\nkey: y\n")
    assert era5.cds_credentials_available({}, home=tmp_path)


def test_download_fails_clearly_without_credentials(tmp_path):
    with pytest.raises(CredentialUnavailableError, match="CDS credentials unavailable"):
        era5.download_era5_months(tmp_path / "c", START, END, env={}, home=tmp_path)
    assert not (tmp_path / "c").exists()


def test_download_with_fake_client_is_atomic_and_resumable(tmp_path):
    calls = []

    class Fake:
        def retrieve(self, dataset, body, target):
            calls.append((dataset, body["month"], target))
            from pathlib import Path

            Path(target).write_bytes(b"z" * 2048)

    paths = era5.download_era5_months(tmp_path, "2025-07-31T00:00:00", "2025-08-01T23:00:00", client=Fake())
    assert [p.name for p in paths] == ["era5_2025_07.nc", "era5_2025_08.nc"]
    era5.download_era5_months(tmp_path, "2025-07-31T00:00:00", "2025-08-01T23:00:00", client=Fake())
    assert len(calls) == 2 and calls[0][0] == "reanalysis-era5-single-levels"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["era5_2025_07.nc", "era5_2025_08.nc"]
