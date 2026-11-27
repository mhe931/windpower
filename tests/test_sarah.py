"""CM SAF SARAH-3 reader: fill/flag masking, trapezoid hourly means across day/month files, SIS/SID pairing, inventory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from windpower import config as C
from windpower import fixtures as F
from windpower import sarah as S

LAT_B = np.array([[60.00, 60.05], [60.05, 60.10]])
LON_B = np.array([[21.00, 21.05], [21.05, 21.10]])
NAN = np.nan


def g_sis(ts: pd.Timestamp) -> np.ndarray:
    """Distinct value per 30-min step and pixel (so a wrong weight or a wrong step cannot hide)."""
    k = ts.day * 20 + ts.hour * 8 + ts.minute // 30 * 4
    return np.array([[k + 1.0, k + 2.0], [k + 3.0, k + 4.0]])


def sid_of(ts: pd.Timestamp) -> np.ndarray:
    return g_sis(ts) - 5.0            # integer W m-2 (the files hold int16), so diffuse is 5 everywhere


def make_order(tmp_path: Path, dates, sis_fn=g_sis, sid_fn=sid_of, status_fn=lambda ts: 0, name="order"):
    """Write SYNTHETIC SIS and SID daily files with explicit per-step arrays on a 2 x 2 pixel grid."""
    out = {}
    for kind, fn in (("SIS", sis_fn), ("SID", sid_fn)):
        d = tmp_path / name / kind
        for date in dates:
            times = pd.date_range(pd.Timestamp(date), periods=48, freq="30min")
            data = np.stack([fn(t) for t in times])
            status = np.array([status_fn(t) for t in times], dtype=np.int8)
            F.write_sarah_arrays(d, kind, date, data, status, LAT_B, LON_B)
        out[kind] = d
    return out


def hourly_for(dirs, hours):
    hours = pd.DatetimeIndex(hours)
    last = hours[-1] + pd.Timedelta(hours=1)
    sis = S.load_series(dirs["SIS"], "SIS", hours[0], last)
    sid = S.load_series(dirs["SID"], "SID", hours[0], last)
    return S.hourly_means(sis, sid, hours), sis, sid


def trap(fn, h: pd.Timestamp) -> np.ndarray:
    return 0.25 * fn(h) + 0.50 * fn(h + pd.Timedelta(minutes=30)) + 0.25 * fn(h + pd.Timedelta(hours=1))


def test_trapezoid_weights_and_day_boundary_jun30_to_jul1(tmp_path):
    dirs = make_order(tmp_path, ["2025-06-30", "2025-07-01"])
    hours = ["2025-06-30T22:00", "2025-06-30T23:00", "2025-07-01T00:00"]
    h, _, _ = hourly_for(dirs, hours)
    for i, t in enumerate(pd.DatetimeIndex(hours)):
        np.testing.assert_allclose(h.ghi[i], trap(g_sis, t), rtol=0, atol=1e-4)
        np.testing.assert_allclose(h.direct[i], trap(sid_of, t), rtol=0, atol=1e-4)
        np.testing.assert_allclose(h.diffuse[i], 5.0, atol=1e-4)
    # 23:00 uses the first step of the NEXT day's file with weight 0.25 (and not the 23:30 step twice)
    t = pd.Timestamp("2025-06-30T23:00")
    expect = 0.25 * g_sis(t) + 0.5 * g_sis(t + pd.Timedelta(minutes=30)) + 0.25 * g_sis(pd.Timestamp("2025-07-01T00:00"))
    np.testing.assert_allclose(h.ghi[1], expect, atol=1e-4)
    assert S.TRAPEZOID == (0.25, 0.50, 0.25)


def test_month_boundary_jul31_23_uses_aug1_midnight(tmp_path):
    bump = {"on": False}

    def sis(ts):
        v = g_sis(ts)
        return v + 400.0 if (bump["on"] and ts == pd.Timestamp("2025-08-01T00:00")) else v

    dirs = make_order(tmp_path, ["2025-07-31", "2025-08-01"], sis_fn=lambda t: sis(t), name="a")
    base, _, _ = hourly_for(dirs, ["2025-07-31T23:00"])
    bump["on"] = True
    dirs2 = make_order(tmp_path, ["2025-07-31", "2025-08-01"], sis_fn=lambda t: sis(t), name="b")
    bumped, _, _ = hourly_for(dirs2, ["2025-07-31T23:00"])
    np.testing.assert_allclose(bumped.ghi - base.ghi, 0.25 * 400.0, atol=1e-4)   # only that sample moved, with weight 1/4


def test_series_needs_last_sample_else_error(tmp_path):
    dirs = make_order(tmp_path, ["2025-07-31"])   # no Aug 1 file
    sis = S.load_series(dirs["SIS"], "SIS", "2025-07-31T00:00", "2025-08-01T00:00")
    sid = S.load_series(dirs["SID"], "SID", "2025-07-31T00:00", "2025-08-01T00:00")
    assert sis.record_status[-1] == S.STATUS_NO_STEP and np.isnan(sis.values[-1]).all()   # absent step stays NaN, never 0
    h = S.hourly_means(sis, sid, pd.date_range("2025-07-31T22:00", "2025-07-31T23:00", freq="h"))
    assert np.isfinite(h.ghi[0]).all()                       # 22:00 has all three samples
    assert np.isnan(h.ghi[1]).all() and (h.cause[1] == S.CAUSE_STEP).all() and not h.hour_step_ok[1]
    with pytest.raises(S.SarahError):
        S.hourly_means(sis, sid, pd.date_range("2025-07-31T23:00", "2025-08-01T00:00", freq="h"))   # beyond the loaded steps


def test_fill_is_nan_and_missing_is_preserved_not_zero(tmp_path):
    def sis(ts):
        v = g_sis(ts)
        if ts == pd.Timestamp("2025-07-01T12:00"):
            v[0, 1] = NAN                                    # fill in exactly one sample of one pixel
        return v

    dirs = make_order(tmp_path, ["2025-07-01"], sis_fn=sis)
    hours = pd.date_range("2025-07-01T10:00", "2025-07-01T13:00", freq="h")
    h, sis_s, _ = hourly_for(dirs, hours)
    assert sis_s.values.dtype == np.float32
    assert np.isnan(sis_s.values[sis_s.times.get_loc(pd.Timestamp("2025-07-01T12:00")), 0, 1])
    # the sample 12:00 is used by hours 11 (as h+1:00) and 12 (as h:00) only
    assert np.isnan(h.ghi[[1, 2], 0, 1]).all() and np.isfinite(h.ghi[[0, 3], 0, 1]).all()
    assert np.isfinite(h.ghi[:, 0, 0]).all()                 # other pixels untouched
    assert (h.cause[[1, 2], 0, 1] == S.CAUSE_SIS).all() and (h.cause[[0, 3], 0, 1] == S.CAUSE_VALID).all()


def test_record_status_void_and_bad_quality_invalidate_the_steps(tmp_path):
    flags = {pd.Timestamp("2025-07-01T05:00"): 1, pd.Timestamp("2025-07-01T09:30"): 2}
    dirs = make_order(tmp_path, ["2025-07-01"], status_fn=lambda ts: flags.get(ts, 0),
                      sis_fn=lambda ts: np.full((2, 2), np.nan) if flags.get(ts) == 1 else g_sis(ts))   # void: all fill; bad: values kept
    hours = pd.date_range("2025-07-01T03:00", "2025-07-01T11:00", freq="h")
    h, sis, _ = hourly_for(dirs, hours)
    assert list(sis.record_status[[sis.times.get_loc(t) for t in flags]]) == [1, 2]
    for i, t in enumerate(hours):
        flagged = any(t + pd.Timedelta(minutes=m) in flags for m in (0, 30, 60))   # a flagged step among h:00, h:30, h+1:00
        assert h.hour_step_ok[i] == (not flagged)
        assert np.isnan(h.ghi[i]).all() == flagged
        if flagged:
            assert (h.cause[i] == S.CAUSE_STEP).all()
    assert not h.hour_step_ok.all() and h.hour_step_ok.any()
    # the bad_quality step still holds numbers on disk; they must not leak into any hourly value
    assert np.isfinite(sis.values[sis.times.get_loc(pd.Timestamp("2025-07-01T09:30"))]).all()


def test_sid_without_sis_is_never_reported_as_ghi(tmp_path):
    def sis(ts):
        v = g_sis(ts)
        if ts == pd.Timestamp("2025-07-01T12:30"):
            v[1, 0] = NAN
        return v

    dirs = make_order(tmp_path, ["2025-07-01"], sis_fn=sis)       # SID is valid at that sample
    h, sis_s, sid_s = hourly_for(dirs, ["2025-07-01T12:00"])
    i = sid_s.times.get_loc(pd.Timestamp("2025-07-01T12:30"))
    assert np.isfinite(sid_s.values[i, 1, 0])                     # raw SID is there ...
    assert np.isnan(h.direct[0, 1, 0]) and np.isnan(h.ghi[0, 1, 0]) and np.isnan(h.diffuse[0, 1, 0])   # ... but unusable
    assert h.cause[0, 1, 0] == S.CAUSE_SIS
    m = S.sample_masks(np.array([[[np.nan]]]), np.array([[[80.0]]]), np.array([True]), np.array([True]))
    assert not m.sid_ok[0, 0, 0] and not m.sis_ok[0, 0, 0] and np.isnan(m.diffuse[0, 0, 0])


def test_diffuse_sign_handling(tmp_path):
    cases = {  # (SIS, SID) -> expected diffuse; NaN = invalid
        (100, 40): 60.0, (100, 100): 0.0, (100, 101): 0.0,   # exactly -1 W m-2 is clipped to 0
        (100, 102): NAN,                                      # -2 W m-2: invalid
    }
    keys = list(cases)

    def per_pixel(idx):
        return lambda ts: np.array([[keys[0][idx], keys[1][idx]], [keys[2][idx], keys[3][idx]]], dtype=float)

    dirs = make_order(tmp_path, ["2025-07-01"], sis_fn=per_pixel(0), sid_fn=per_pixel(1))
    h, _, _ = hourly_for(dirs, ["2025-07-01T12:00"])
    got = h.diffuse[0].ravel()
    for g, key in zip(got, keys):
        if np.isnan(cases[key]):
            assert np.isnan(g)
        else:
            assert g == pytest.approx(cases[key])
    assert h.cause[0, 1, 1] == S.CAUSE_NEG_DIFFUSE and (h.cause[0].ravel()[:3] == S.CAUSE_VALID).all()
    assert h.ghi[0, 1, 1] == pytest.approx(100.0) and h.direct[0, 1, 1] == pytest.approx(102.0)   # GHI and SID themselves stay valid


def test_out_of_range_samples_are_invalid(tmp_path):
    dirs = make_order(tmp_path, ["2025-07-01"], sis_fn=lambda ts: np.array([[2000.0, -5.0], [10.0, 1500.0]]),
                      sid_fn=lambda ts: np.array([[1.0, 1.0], [1.0, 1.0]]))
    h, _, _ = hourly_for(dirs, ["2025-07-01T06:00"])
    assert np.isnan(h.ghi[0, 0, 0]) and np.isnan(h.ghi[0, 0, 1]) and h.ghi[0, 1, 0] == pytest.approx(10.0) and h.ghi[0, 1, 1] == pytest.approx(1500.0)


def test_read_day_file_masks_int16_fill_and_orders_latitude_ascending(tmp_path):
    data = np.stack([g_sis(t) for t in pd.date_range("2025-07-01", periods=48, freq="30min")])
    data[3, 0, 0] = NAN
    p = F.write_sarah_arrays(tmp_path / "d", "SIS", "2025-07-01", data[:, ::-1, :], np.zeros(48), LAT_B[::-1], LON_B)   # descending latitude on disk
    import xarray as xr

    with xr.open_dataset(p, mask_and_scale=False) as ds:
        assert ds["SIS"].dtype == np.int16 and int(ds["SIS"].values.min()) == C.SARAH_FILL
        assert ds["SIS"].attrs["_FillValue"] == C.SARAH_FILL
    f = S.read_day_file(p, "SIS")
    assert f.values.dtype == np.float32 and np.isnan(f.values[3, 0, 0]) and int(f.n_fill) == 1   # the flip restores the original orientation
    assert np.all(np.diff(f.lat) > 0) and f.lat_bnds[0, 0] == pytest.approx(60.0)
    np.testing.assert_allclose(f.values, data, equal_nan=True)


def test_read_day_file_rejects_missing_variable(tmp_path):
    p = F.write_sarah_arrays(tmp_path, "SIS", "2025-07-01", np.zeros((48, 2, 2)), np.zeros(48), LAT_B, LON_B)
    with pytest.raises(S.SarahError) as e:
        S.read_day_file(p, "SID")
    assert e.value.check == "variables"


def test_inventory_reports_variables_flags_grid_and_hashes(tmp_path):
    spec = F.SarahFixtureSpec(void_steps=("2025-07-31T12:00",), bad_steps=("2025-07-30T10:30",))
    sis_dir, sid_dir = F.write_sarah_cache(tmp_path, "2025-07-30", 3, spec)
    inv = S.inventory(sis_dir, "SIS", hash_files=True)
    assert inv["standard_name"] == inv["expected_standard_name"] == "surface_downwelling_shortwave_flux_in_air" and inv["units"] == "W m-2"
    assert inv["n_files"] == 3 and inv["time"]["n_steps"] == 144 and inv["time"]["n_gaps"] == 0 and inv["time"]["unique"]
    assert inv["time"]["first_utc"] == "2025-07-30T00:00:00Z" and inv["time"]["last_utc"] == "2025-08-01T23:30:00Z"
    assert inv["record_status"]["counts"] == {"ok": 142, "void": 1, "bad_quality": 1, "other": 0}
    assert [(s["time_utc"], s["meaning"]) for s in inv["record_status"]["flagged_steps"]] == [
        ("2025-07-30T10:30:00Z", "bad_quality"), ("2025-07-31T12:00:00Z", "void")]
    assert (inv["grid"]["n_lat"], inv["grid"]["n_lon"]) == (32, 44) and inv["grid"]["res_deg"] == pytest.approx(0.05, abs=1e-4)
    assert inv["grid"]["lat_bounds"][0] == pytest.approx(59.9, abs=1e-5) and inv["grid"]["bounds_from_file"]
    assert 0.0 < inv["values"]["fill_share"] < 1.0 and inv["values"]["n_negative_non_fill"] == 0
    assert set(inv["sha256"]) == {p.name for p in sis_dir.glob("*.nc")} and all(len(v) == 64 for v in inv["sha256"].values())
    assert S.inventory(sid_dir, "SID")["sha256"] is None
    sub = S.inventory(sis_dir, "SIS", first_date="2025-07-31", last_date="2025-07-31")
    assert sub["n_files"] == 1 and [s["time_utc"] for s in S.flagged_steps_in_window(inv, "2025-07-31", "2025-07-31T23:30")] == ["2025-07-31T12:00:00Z"]


def test_resolve_sarah_dir_precedence_and_clear_errors(tmp_path, monkeypatch):
    a, b = tmp_path / "flag", tmp_path / "env"
    a.mkdir(), b.mkdir()
    env = {C.SARAH_SIS_ENV: str(b)}
    assert S.resolve_sarah_dir("SIS", a, env, tmp_path) == a                      # flag wins
    assert S.resolve_sarah_dir("SIS", None, env, tmp_path) == b                   # then the environment variable
    with pytest.raises(S.SarahError) as e:                                        # default location absent -> clear error naming both options
        S.resolve_sarah_dir("SID", None, {}, tmp_path)
    assert e.value.check == "input_missing" and C.SARAH_SID_ENV in str(e.value) and "--sarah-sid-dir" in str(e.value)
    default = tmp_path / "interim" / "smoke" / "ORD73129"
    default.mkdir(parents=True)
    assert S.resolve_sarah_dir("SID", None, {}, tmp_path) == default
    with pytest.raises(S.SarahError):
        S.resolve_sarah_dir("SIS", tmp_path / "nope", {}, tmp_path)


def test_list_day_files_requires_matching_names(tmp_path):
    (tmp_path / "x.nc").write_bytes(b"")
    with pytest.raises(S.SarahError):
        S.list_day_files(tmp_path, "SIS")


REAL_SIS = S.default_sarah_dir("SIS")
REAL_SID = S.default_sarah_dir("SID")


@pytest.mark.skipif(not (REAL_SIS.is_dir() and REAL_SID.is_dir()),
                    reason=f"real CM SAF SARAH-3 folders not present ({REAL_SIS} / {REAL_SID}); this check only runs where the order is available")
def test_real_folder_inventory_matches_the_documented_order():
    for kind, folder in (("SIS", REAL_SIS), ("SID", REAL_SID)):
        inv = S.inventory(folder, kind, first_date="2025-07-01", last_date="2025-07-02")
        assert inv["units"] == "W m-2" and inv["standard_name"] == S.KINDS[kind]
        assert inv["time"]["n_steps"] == 96 and inv["time"]["n_gaps"] == 0 and inv["time"]["unique"]
        assert (inv["grid"]["n_lat"], inv["grid"]["n_lon"]) == (112, 260)
        assert inv["grid"]["lat_bounds"][0] == pytest.approx(59.4, abs=1e-4) and inv["grid"]["lon_bounds"] == [pytest.approx(19.0), pytest.approx(32.0)]
        assert inv["record_status"]["counts"]["ok"] == 96 and inv["values"]["n_negative_non_fill"] == 0
        assert 0.2 < inv["values"]["fill_share"] < 0.3
