"""ERA5 solar inputs: units, time-convention detection (shift exactly once), temperature, solar position, windows, CDS helper."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from windpower import config as C
from windpower import era5_solar as E
from windpower import fixtures as F
from windpower import solar_geometry as G
from windpower.io_utils import CredentialUnavailableError
from conftest import SOLAR_END, SOLAR_START


def test_joules_per_m2_become_watts_per_m2(era5_solar_fx, era5_solar_result):
    raw, h = era5_solar_fx.raw, era5_solar_result.ds
    t = pd.Timestamp("2025-07-30T10:00")
    cell = dict(latitude=60.5, longitude=22.0)
    j = float(raw["tisr"].sel(valid_time=t + pd.Timedelta(hours=1), **cell))
    assert j > 1e6                                                   # J m-2 over one hour
    assert float(h["influx_toa"].sel(time=t, y=60.5, x=22.0)) == pytest.approx(j / 3600.0, rel=1e-6)
    fd, ssrd = (float(raw[v].sel(valid_time=t + pd.Timedelta(hours=1), **cell)) for v in ("fdir", "ssrd"))
    assert float(h["influx_direct"].sel(time=t, y=60.5, x=22.0)) == pytest.approx(fd / 3600.0, rel=1e-6)
    assert float(h["influx_diffuse"].sel(time=t, y=60.5, x=22.0)) == pytest.approx((ssrd - fd) / 3600.0, rel=1e-5)
    assert h["influx_toa"].attrs["units"] == "W m**-2" and h["temperature"].attrs["units"] == "K"
    assert era5_solar_result.convention["evidence"] is not None
    assert "J m-2" in era5_solar_result.source["units"]["tisr"]


def test_unit_errors_are_typed():
    base = lambda **a: xr.Dataset({v: (("valid_time", "latitude", "longitude"), np.full((1, 1, 1), 1.0e6, np.float32), a.get(v, {}))
                                   for v in ("ssrd", "ssr", "fdir", "tisr")} | {"t2m": (("valid_time", "latitude", "longitude"), np.full((1, 1, 1), 280.0, np.float32))},
                                  coords={"valid_time": [pd.Timestamp("2025-07-01")], "latitude": [60.0], "longitude": [21.0]})
    with pytest.raises(E.UnitsError):
        E.to_wm2_fields(base(tisr={"units": "MJ m**-2"}), "raw")
    with pytest.raises(E.UnitsError):                                # labelled W m-2 but joule-sized numbers
        E.to_wm2_fields(base(tisr={"units": "W m**-2"}), "raw")
    ds = base(**{v: {"units": "J m**-2"} for v in ("ssrd", "ssr", "fdir", "tisr")})
    ds["t2m"] = ds["t2m"] * 0 + 15.0                                  # Celsius instead of Kelvin
    with pytest.raises(E.UnitsError):
        E.to_wm2_fields(ds, "raw")


def test_albedo_diffuse_and_clipping_follow_atlite():
    dims = ("valid_time", "latitude", "longitude")
    vals = {"ssrd": [7.2e6, 0.0, 3.6e6], "ssr": [6.12e6, 0.0, 3.6e6], "fdir": [3.6e6, 0.0, 4.5e6], "tisr": [4.5e6, 0.0, 5.0e6]}
    ds = xr.Dataset({k: (dims, np.array(v, np.float64).reshape(3, 1, 1), {"units": "J m**-2"}) for k, v in vals.items()}
                    | {"t2m": (dims, np.full((3, 1, 1), 285.0))},
                    coords={"valid_time": pd.date_range("2025-07-01", periods=3, freq="h"), "latitude": [60.0], "longitude": [21.0]})
    f, _ = E.to_wm2_fields(ds, "raw")
    assert f["albedo"].values.ravel() == pytest.approx([0.15, 0.0, 0.0])   # (ssrd - ssr) / ssrd; 0 where ssrd == 0
    assert f["influx_diffuse"].values.ravel() == pytest.approx([1000.0, 0.0, 0.0])   # ssrd - fdir = -250 W m-2 is clipped to 0
    assert f["influx_direct"].values.ravel() == pytest.approx([1000.0, 0.0, 1250.0]) and (f["influx_toa"] >= 0).all()


def test_end_stamp_relabel_h_plus_1_to_h_exactly_once(era5_solar_fx, era5_solar_result):
    raw, h = era5_solar_fx.raw, era5_solar_result.ds
    assert era5_solar_result.convention["detected"] == "end" and era5_solar_result.convention["convention_used"] == "end"
    assert pd.DatetimeIndex(h["time"].values).equals(pd.date_range(SOLAR_START, SOLAR_END, freq="h")) and h.sizes["time"] == 48
    for hour in ("2025-07-30T00:00", "2025-07-31T23:00"):
        t = pd.Timestamp(hour)
        want = raw["fdir"].sel(valid_time=t + pd.Timedelta(hours=1)).sortby("latitude").values / 3600.0
        np.testing.assert_allclose(h["influx_direct"].sel(time=t).values, want, rtol=1e-6)
    prepared = era5_solar_fx.prepared                                  # the truth, already labelled by hour start
    np.testing.assert_allclose(h["influx_toa"].values, prepared["influx_toa"].values, rtol=2e-6, atol=1e-3)
    np.testing.assert_allclose(h["temperature"].values, prepared["temperature"].values, atol=1e-4)


def test_prepared_hour_start_input_is_not_shifted_and_processing_is_idempotent(era5_solar_fx, era5_solar_result):
    res_p = E.load_solar_era5(era5_solar_fx.prepared, SOLAR_START, SOLAR_END)
    assert res_p.convention["detected"] == "start" and res_p.convention["shift_applied"].startswith("none")
    for v in E.PREPARED_VARS:
        np.testing.assert_allclose(res_p.ds[v].values, era5_solar_result.ds[v].values, rtol=2e-6, atol=1e-3)   # raw (shifted once) == prepared (shift 0)
    again = E.load_solar_era5(era5_solar_result.ds, SOLAR_START, SOLAR_END)    # feeding our own output back in must change nothing
    assert again.convention["detected"] == "start" and again.convention["dataset_hint"] == "start"
    for v in E.HOURLY_VARS:
        np.testing.assert_array_equal(again.ds[v].values, era5_solar_result.ds[v].values)


def test_detection_evidence_matches_the_report_pattern(era5_solar_result):
    ev = era5_solar_result.convention["evidence"]
    assert ev["end"]["relative_rmse"] < 0.01 < 0.2 < ev["start"]["relative_rmse"]
    assert abs(ev["end"]["com_minus_solar_noon_min_mean"]) < 2.0                    # report: 0.4 min with the shift ...
    assert 55.0 < ev["start"]["com_minus_solar_noon_min_mean_abs"] < 65.0            # ... vs 58.9 min without it
    assert ev["end"]["n_cell_days"] > 0 and ev["start"]["n_cell_days"] > 0
    assert ev["rmse_ratio_best_over_other"] < 0.7


def test_ambiguous_convention_raises_unless_overridden(era5_solar_fx):
    flat = era5_solar_fx.prepared.copy(deep=True)
    flat["influx_toa"] = flat["influx_toa"] * 0 + 300.0                  # no daily cycle at all: cannot tell
    with pytest.raises(E.ConventionAmbiguousError) as e:
        E.load_solar_era5(flat, SOLAR_START, SOLAR_END)
    assert e.value.check == "time_convention" and e.value.evidence["detected"] == "ambiguous"
    forced = E.load_solar_era5(flat, SOLAR_START, "2025-07-31T22:00", convention="end")   # 'end' needs the next stamp: 48 stamps cover 47 hours
    assert forced.convention["override"] == "end" and forced.convention["detected"] == "ambiguous" and forced.ds.sizes["time"] == 47
    with pytest.raises(E.ConventionAmbiguousError):                      # too short to judge
        E.load_solar_era5(era5_solar_fx.prepared.isel(time=slice(0, 10)), SOLAR_START, "2025-07-30T09:00")


def test_override_is_recorded_and_contradiction_is_flagged(era5_solar_fx):
    res = E.load_solar_era5(era5_solar_fx.prepared, SOLAR_START, SOLAR_END, convention="start")
    assert res.convention["override"] == "start" and not res.convention["override_contradicts_detection"]
    wrong = E.load_solar_era5(era5_solar_fx.raw, SOLAR_START, "2025-07-31T22:00", convention="start")   # user insists: no shift
    assert wrong.convention["override_contradicts_detection"] and wrong.convention["detected"] == "end"
    with pytest.raises(ValueError):
        E.resolve_convention(era5_solar_fx.prepared["influx_toa"].rename({"time": "valid_time", "y": "latitude", "x": "longitude"}), "sideways")


def test_dataset_hint_contradicting_the_data_is_rejected(era5_solar_fx):
    ds = era5_solar_fx.prepared.copy()
    ds.attrs[E.HINT_ATTR] = "end"
    with pytest.raises(E.ConventionAmbiguousError, match="contradicts"):
        E.load_solar_era5(ds, SOLAR_START, SOLAR_END)


def test_temperature_is_the_mean_of_the_two_endpoint_stamps(era5_solar_fx, era5_solar_result):
    raw, h = era5_solar_fx.raw, era5_solar_result.ds
    for hour in ("2025-07-30T00:00", "2025-07-30T13:00", "2025-07-31T23:00"):
        t = pd.Timestamp(hour)
        a = raw["t2m"].sel(valid_time=t).sortby("latitude").values.astype(np.float64)
        b = raw["t2m"].sel(valid_time=t + pd.Timedelta(hours=1)).sortby("latitude").values.astype(np.float64)
        np.testing.assert_allclose(h["temperature"].sel(time=t).values, 0.5 * (a + b), atol=1e-9)
    assert float(h["temperature"].min()) > 200 and E.PREPARED_VARS[-1] == "temperature"


def test_solar_position_is_evaluated_at_h_plus_30_minutes(era5_solar_result):
    from atlite.pv.solar_position import SolarPosition

    h = era5_solar_result.ds
    t = pd.DatetimeIndex(h["time"].values)
    lats, lons = h["y"].values, h["x"].values
    frame = xr.Dataset(coords={"time": t, "y": lats, "x": lons}).assign_coords(lon=("x", lons), lat=("y", lats))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mid = SolarPosition(frame, time_shift=pd.Timedelta(minutes=30)).transpose("time", "y", "x").compute()
        start = SolarPosition(frame).transpose("time", "y", "x").compute()
        # atlite's own end-stamped path: stamp h+1 evaluated at -30 min
        end_frame = xr.Dataset(coords={"time": t + pd.Timedelta(hours=1), "y": lats, "x": lons}).assign_coords(lon=("x", lons), lat=("y", lats))
        own = SolarPosition(end_frame, time_shift=pd.Timedelta(minutes=-30)).transpose("time", "y", "x").compute()
    np.testing.assert_array_equal(h["solar_altitude"].values, mid["altitude"].values)
    np.testing.assert_allclose(h["solar_altitude"].values, own["altitude"].values, atol=1e-12)
    np.testing.assert_allclose(h["solar_azimuth"].values, own["azimuth"].values, atol=1e-12)
    # independent check against the NOAA equations at h+30, and clearly different from the hour-start position
    noaa = np.radians(G.elevation_deg(t + pd.Timedelta(minutes=30), lats, lons))
    day = noaa > np.radians(10)
    assert np.abs(h["solar_altitude"].values - noaa)[day].max() < np.radians(0.5)
    morning = (t.hour == 5)[:, None, None] & (noaa > np.radians(3))
    assert np.degrees(np.abs(start["altitude"].values - h["solar_altitude"].values))[morning].mean() > 1.0


def test_required_stamps_for_july_is_745_and_window_errors_are_typed():
    need = E.required_stamps("2025-07-01", "2025-07-31T23:00", "end")
    assert len(need) == 745 and need[0] == pd.Timestamp("2025-07-01T00:00") and need[-1] == pd.Timestamp("2025-08-01T00:00")
    assert len(E.required_stamps("2025-07-01", "2025-07-31T23:00", "start")) == 744
    info = E.validate_window(need, "2025-07-01", "2025-07-31T23:00", "end")
    assert info["required_stamps"] == 745 and "previous hour" in info["note"]
    with pytest.raises(E.WindowError, match="1 of 745"):
        E.validate_window(need[:-1], "2025-07-01", "2025-07-31T23:00", "end")      # Aug 1 00:00 missing
    with pytest.raises(E.WindowError):
        E.validate_window(need.delete(100), "2025-07-01", "2025-07-31T23:00", "end")
    E.validate_window(need[:-1], "2025-07-01", "2025-07-31T23:00", "start")           # hour-start data needs no extra stamp


def test_raw_window_missing_last_stamp_fails_in_the_loader(era5_solar_fx):
    short = era5_solar_fx.raw.isel(valid_time=slice(0, 48))      # 48 stamps = 47 hours of end-stamped data
    with pytest.raises(E.WindowError):
        E.load_solar_era5(short, SOLAR_START, SOLAR_END)


def test_sources_netcdf_zip_directory_and_prepared_file_agree(tmp_path, era5_solar_fx, era5_solar_result):
    raw = era5_solar_fx.raw
    nc = F.write_netcdf(raw, tmp_path / "single" / "era5_solar.nc")
    zp = F.write_era5_solar_zip(raw, tmp_path / "zip" / "era5_solar_2025_07.nc")           # CDS bundle: instant + accum members
    split = tmp_path / "split"
    F.write_netcdf(raw.isel(valid_time=slice(0, 30)), split / "a.nc")
    F.write_era5_solar_zip(raw.isel(valid_time=slice(30, None)), split / "b.zip")             # two files, one of them a bundle
    prep = F.write_netcdf(era5_solar_fx.prepared, tmp_path / "prep" / "prepared.nc")
    split_streams = tmp_path / "streams"
    for name, names in (("instant", ["t2m"]), ("accum", ["ssrd", "ssr", "fdir", "tisr"])):
        F.write_netcdf(raw[names], split_streams / f"{name}.nc")                              # separate files per stream in one directory
    for src in (nc, zp, split, prep, split_streams):
        res = E.load_solar_era5(src, SOLAR_START, SOLAR_END)
        for v in E.PREPARED_VARS:
            np.testing.assert_allclose(res.ds[v].values, era5_solar_result.ds[v].values, rtol=2e-6, atol=1e-3, err_msg=str(src))
        assert res.source["kind"] == ("prepared" if src == prep else "raw") and res.source["grid"]["n_lat"] == 7
    with pytest.raises(FileNotFoundError):
        E.open_solar_source(tmp_path / "missing")
    with pytest.raises(E.SolarEra5Error, match="neither raw"):
        E.load_solar_era5(raw[["ssrd", "t2m"]], SOLAR_START, SOLAR_END)


def test_cds_request_builder_and_download_failure_without_credentials(tmp_path, monkeypatch):
    reqs = E.build_cds_requests("2025-07-01", "2025-07-31T23:00")
    assert [n for n, _ in reqs] == ["era5_solar_2025_07.nc", "era5_solar_2025_08.nc"]
    july, aug = reqs[0][1], reqs[1][1]
    assert len(july["day"]) == 31 and aug["day"] == ["01"] and july["area"] == [70.25, 19.0, 59.5, 32.0]
    assert len(july["time"]) == 24 and "toa_incident_solar_radiation" in july["variable"] and "2m_temperature" in july["variable"]
    assert set(july["variable"]) == set(E.CDS_SOLAR_VARIABLES) and len(E.CDS_SOLAR_VARIABLES) == 5
    # no ~/.cdsapirc and no environment variables: clear failure, nothing contacted, nothing written
    with pytest.raises(CredentialUnavailableError, match="cdsapirc"):
        E.download_era5_solar(tmp_path / "cache", "2025-07-01", "2025-07-31T23:00", env={}, home=tmp_path)
    assert not list((tmp_path / "cache").glob("*"))

    class FakeClient:
        calls: list = []

        def retrieve(self, dataset, body, target):
            self.calls.append((dataset, body["month"], target))
            with open(target, "wb") as fh:
                fh.write(b"x" * 4096)

    c = FakeClient()
    paths = E.download_era5_solar(tmp_path / "cache", "2025-07-01", "2025-07-31T23:00", client=c)
    assert [p.name for p in paths] == ["era5_solar_2025_07.nc", "era5_solar_2025_08.nc"] and len(c.calls) == 2
    E.download_era5_solar(tmp_path / "cache", "2025-07-01", "2025-07-31T23:00", client=c)    # cached: no second request
    assert len(c.calls) == 2


def test_solar_grid_differs_from_the_wind_grid():
    s, w = C.FINLAND_SOLAR, C.FINLAND_GRID
    assert (s.n_lat, s.n_lon, s.n_lat * s.n_lon) == (44, 53, 2332) and (s.lat_min, s.lat_max, s.lon_min, s.lon_max) == (59.5, 70.25, 19.0, 32.0)
    assert (w.n_lat, w.n_lon) == (45, 52) and (w.lat_max, w.lon_max) == (70.5, 31.75)     # the wind grid is untouched
    assert s.cds_area == [70.25, 19.0, 59.5, 32.0] and s.res == 0.25 and C.GRIDS["finland"] is w and "finland_solar" not in C.GRIDS
