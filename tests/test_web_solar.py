"""Solar web API (in-process server on the solar-demo artifact), CSV/JSON agreement, error paths, partial artifacts, serve options, UI hygiene."""

from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

import test_web as TW
from windpower import cli
from windpower import config as C
from windpower import fixtures_solar as FP
from windpower import solar_public as SP
from windpower.export_site_solar import SOLAR_SITE_DIR
from windpower.web import data as D
from windpower.web import server as W
from windpower.web import solar_data as SD

NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))
SOUTH = "lat=60.17&lon=24.94"          # Helsinki area: cell centre 60.25 N 25.0 E, fully covered by the (fixture) satellite order
NORTH = "lat=66.5&lon=25.72"           # Rovaniemi area: north of the satellite order
MID_NORTH = "lat=64.0&lon=27.0"        # inside the order but north of the fixture's 63.5 N cutoff
WIN = "start=2025-07-28&end=2025-07-31"


class Live:
    def __init__(self, solar_dir=None, wind_dir=None):
        self.solar = SD.SolarSiteData.load(solar_dir) if solar_dir else None
        self.wind = D.SiteData.load(wind_dir) if wind_dir else None
        self.server = W.make_server(self.wind, port=0, quiet=True, solar=self.solar)
        self.thread = W.start_in_thread(self.server)
        self.base = W.url_for(self.server).rstrip("/")

    def get(self, path, method="GET"):
        req = urllib.request.Request(self.base + path, method=method)
        try:
            with NO_PROXY.open(req, timeout=30) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    def json(self, path):
        status, headers, body = self.get(path)
        return status, json.loads(body.decode("utf-8")), headers

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture(scope="session")
def demo_data(tmp_path_factory):
    import contextlib
    import io

    d = tmp_path_factory.mktemp("solar_demo_data")
    with contextlib.redirect_stdout(io.StringIO()):
        code = cli.main(["solar-demo", "--data-dir", str(d), "--quiet"])
    assert code == 0
    return d


@pytest.fixture(scope="session")
def demo_dir(demo_data) -> Path:
    return demo_data / "outputs" / "solar-demo" / SOLAR_SITE_DIR


@pytest.fixture(scope="module")
def live(demo_dir):
    lv = Live(demo_dir)
    yield lv
    lv.close()


@pytest.fixture(scope="module")
def both(demo_dir, tmp_path_factory):
    wind = TW.write_hand_site(tmp_path_factory.mktemp("hand_wind"))
    lv = Live(demo_dir, wind)
    yield lv
    lv.close()


@pytest.fixture(scope="module")
def partial_dir(tmp_path_factory, sarah_smoke):
    """A real-mode PARTIAL artifact built by the pipeline itself: SARAH-3 (synthetic stand-in files) + public data, no ERA5 and no Fingrid key."""
    base = tmp_path_factory.mktemp("partial")
    data = base / "data"
    SP.fetch_public(data / "cache" / "solar" / "public", FP.FakePublicSession(FP.synthetic_public(C.FINLAND_SOLAR, seed=5)), sleep=lambda s: None)
    with contextlib.redirect_stdout(io.StringIO()):
        code = cli.main(["solar-all", "--quiet", "--data-dir", str(data), "--window-start", "2025-07-30", "--window-end", "2025-07-31",
                         "--sarah-sis-dir", str(sarah_smoke.sis_dir), "--sarah-sid-dir", str(sarah_smoke.sid_dir)])
    assert code == 0
    return data / "outputs" / "solar" / SOLAR_SITE_DIR


@pytest.fixture(scope="module")
def partial(partial_dir):
    lv = Live(partial_dir)
    yield lv
    lv.close()


def raw_cf(demo_dir, mode, lat_idx, lon_idx):
    meta = json.loads((demo_dir / "meta.json").read_text())
    shape = tuple(meta["files"][f"cf_{mode}.bin"]["shape"])
    return np.fromfile(demo_dir / f"cf_{mode}.bin", dtype="<u2").reshape(shape)[:, lat_idx, lon_idx] / 10000.0


# ----------------------------------------------------------------------------------------------------------- meta / grid / coverage
def test_meta_describes_modes_panel_units_scope_and_labels(live):
    s, m, h = live.json("/api/v1/solar/meta")
    assert s == 200 and h["Content-Type"].startswith("application/json") and h["Cache-Control"] == "no-store"
    assert m["synthetic"] and m["demo"] and m["run_mode"] == "solar-demo" and m["label"] == "DEMO / SYNTHETIC - not real data" and "DEMO - SYNTHETIC" in m["notice"]
    assert m["partial"] is False and m["api_version"] == "v1" and m["kind"] == "solar"
    g = m["grid"]
    assert (g["n_lat"], g["n_lon"], g["resolution_deg"]) == (44, 53, 0.25) and g["lat_range"] == [59.5, 70.25] and g["lon_range"] == [19.0, 32.0]
    assert 11 < g["cell_km_at_mid_latitude"]["east_west"] < 15 and g["cell_km_at_mid_latitude"]["north_south"] == pytest.approx(27.8, abs=0.1)
    assert m["panel"] == {"panel": "CSi", "model": "huld", "slope_deg": 40.0, "azimuth_deg": 180.0, "inverter_efficiency": 0.9, "source": "Huld 2010", "tracking": None}
    assert m["limits"]["capacity_mwp"] == {"min": 0.001, "max": 1000.0, "default": 1.0} and m["defaults"]["mode"] == "hybrid"
    assert m["modes"]["era5"] == {"title": "ERA5 only", "available": True, "reason": None, "reason_short": None} and m["modes"]["hybrid"]["available"]
    assert m["time_range"] == {"first_utc": "2025-07-28T00:00:00Z", "last_utc": "2025-07-31T23:00:00Z", "hours": 96, "step": "1h", "label": "start of the hour, UTC"}
    assert m["units"]["power"] == "MW" and m["units"]["capacity"] == "MWp" and m["units"]["energy"] == "MWh"
    assert set(m["series"]) >= {"era5_only_mw", "hybrid_mw", "fingrid_248_forecast_mw", "fingrid_362_av06_small_scale_surplus_mw", "fingrid_267_capacity_mw"}
    assert "forecast, not measured" in m["series"]["fingrid_248_forecast_mw"]["scope"] and "surplus only" in m["series"]["fingrid_362_av06_small_scale_surplus_mw"]["scope"]
    assert m["series"]["fingrid_248_forecast_mw"]["native_unit"] == "MWh/h" and m["series"]["fingrid_362_av06_small_scale_surplus_mw"]["native_unit"] == "kWh"
    assert "No national calibration" in m["statements"]["calibration"] and "north" in m["statements"]["northern_fallback"].lower()
    assert {"cm_saf_sarah3", "era5", "fingrid", "statistics_finland"} <= set(m["licences"]) and "CC BY 4.0" in m["licences"]["fingrid"]["name"]
    assert m["coverage"]["available"] and m["coverage"]["fully_covered_cells"] == 756 and m["unavailable"] == {"modes": {}, "series": {}, "causes": []}
    assert "mainland" in m["region"]["outlines_lonlat"]


def test_grid_and_coverage_endpoints(live, demo_dir):
    s, g, _ = live.json("/api/v1/solar/grid")
    assert s == 200 and g["grid"]["n_lat"] == 44 and len(g["cells"]["satellite_covered"]) == 44 * 53 and g["synthetic"]
    assert len(g["cells"]["population_share"]) == 44 * 53 and sum(g["cells"]["population_share"]) == pytest.approx(1.0, abs=1e-6)
    s, c, _ = live.json("/api/v1/solar/coverage")
    assert s == 200 and c["available"] and c["threshold"] == 0.999 and c["summary"]["fully_covered_cells"] == 756
    cov = np.array(c["cells"]["satellite_covered"]).reshape(44, 53)
    assert cov.sum() == 756 and not cov[30:].any()                                  # nothing north of ~65 N is ever covered
    frac = c["cells"]["satellite_coverage_fraction"]
    assert len(frac) == 44 * 53 and all(f is None or 0 <= f <= 1 for f in frac) and c["pixel_map"]["shape"] == [112, 260]
    i, j = 3, 24                                                                    # 60.25 N, 25.0 E
    assert cov[i, j] and frac[i * 53 + j] > 0.9
    assert live.json("/api/v1/solar/meta?x=1")[0] == 400                            # no parameters accepted


# ----------------------------------------------------------------------------------------------------------- simulate: arithmetic
def test_simulate_arithmetic_matches_the_stored_capacity_factors(live, demo_dir):
    s, r, _ = live.json(f"/api/v1/solar/simulate?{SOUTH}&{WIN}&capacity_mwp=2.5&mode=hybrid")
    assert s == 200 and r["cell"]["lat_index"] == 3 and r["cell"]["lon_index"] == 24 and (r["cell"]["center_lat"], r["cell"]["center_lon"]) == (60.25, 25.0)
    cf = raw_cf(demo_dir, "hybrid", 3, 24)                                         # read straight from the .bin file
    d = r["data"]
    assert len(d["time_utc"]) == 96 and d["time_utc"][0] == "2025-07-28T00:00:00Z" and d["time_utc"][-1] == "2025-07-31T23:00:00Z"
    np.testing.assert_allclose(d["capacity_factor"], np.round(cf, 6), atol=1e-12)
    np.testing.assert_allclose(d["power_mw"], np.round(cf * 2.5, 6), atol=1e-9)    # MW = CF x MWp
    np.testing.assert_allclose(d["energy_mwh"], d["power_mw"], atol=1e-9)           # MWh = MW x 1 h
    sm = r["summary"]
    assert sm["total_mwh"] == pytest.approx(float(np.round(cf * 2.5, 6).sum()), abs=1e-5) and sm["hours"] == 96
    assert sm["mean_capacity_factor"] == pytest.approx(sm["total_mwh"] / (2.5 * 96), abs=1e-6) and sm["full_load_hours"] == pytest.approx(sm["total_mwh"] / 2.5, abs=1e-3)
    assert sm["peak_mw"] == pytest.approx(cf.max() * 2.5, abs=1e-6) and sm["mean_mw"] == pytest.approx(sm["total_mwh"] / 96, abs=1e-5)
    k = int(np.argmax(cf))                                                          # one explicit hour, by hand
    assert d["power_mw"][k] == pytest.approx(round(cf[k] * 2.5, 6)) and 0 < d["capacity_factor"][k] <= 1
    assert r["units"]["capacity"] == "MWp" and "none" in r["calibration"] and r["panel"]["slope_deg"] == 40.0 and r["request"]["capacity_mwp"] == 2.5


def test_capacity_scaling_default_and_limits(live):
    one = live.json(f"/api/v1/solar/simulate?{SOUTH}&{WIN}&mode=era5")[1]                       # default capacity 1 MWp
    big = live.json(f"/api/v1/solar/simulate?{SOUTH}&{WIN}&mode=era5&capacity_mwp=1000")[1]
    tiny = live.json(f"/api/v1/solar/simulate?{SOUTH}&{WIN}&mode=era5&capacity_mwp=0.001")[1]
    assert one["request"]["capacity_mwp"] == 1.0
    np.testing.assert_allclose(big["data"]["power_mw"], np.array(one["data"]["power_mw"]) * 1000, rtol=1e-5, atol=1e-6)
    assert big["data"]["capacity_factor"] == one["data"]["capacity_factor"] == tiny["data"]["capacity_factor"]       # CF is independent of the capacity
    assert tiny["summary"]["total_mwh"] == pytest.approx(one["summary"]["total_mwh"] / 1000, abs=1e-6)


def test_cf_range_and_night_hours_are_exactly_zero(live, demo_dir, demo_data):
    import xarray as xr

    with xr.open_dataset(demo_data / "outputs" / "solar-demo" / "era5_solar_hourly.nc") as e:
        alt = e["solar_altitude"].values[:, 3, 24]                                  # solar altitude (rad) at the south cell, h+30 min
    for mode in ("era5", "hybrid"):
        r = live.json(f"/api/v1/solar/simulate?{SOUTH}&{WIN}&mode={mode}&capacity_mwp=7")[1]
        p, cf = np.array(r["data"]["power_mw"]), np.array(r["data"]["capacity_factor"])
        assert (cf >= 0).all() and (cf <= 1).all() and (p >= 0).all() and (alt <= 0).sum() > 20
        assert (p[alt <= 0] == 0).all() and (cf[alt <= 0] == 0).all()               # night = exactly 0 MW
        assert r["summary"]["zero_output_hours"] >= int((alt <= 0).sum()) and (cf[alt > np.radians(20)] > 0).all()


# ----------------------------------------------------------------------------------------------------------- satellite vs fallback
def test_covered_site_uses_the_satellite_in_hybrid_mode(live):
    r = live.json(f"/api/v1/solar/simulate?{SOUTH}&{WIN}&mode=hybrid")[1]
    sm, d = r["summary"], r["data"]
    assert r["cell"]["satellite_covered"] is True and r["cell"]["satellite_coverage_fraction"] > 0.9 and "fully covered" in r["mode"]["explanation"]
    assert sm["satellite_hours"] + sm["fallback_hours"] == sm["hours"] == 96 and sm["satellite_hours"] >= 90 and sm["era5_hours"] == 0
    assert set(d["irradiance_source"]) <= {"satellite", "era5_fallback"} and "satellite" in d["irradiance_source"]
    assert set(sm["fallback_reasons"]) <= {"missing_or_flagged_step"} and sum(sm["fallback_reasons"].values()) == sm["fallback_hours"] > 0   # the fixture's flagged steps
    for src, why in zip(d["irradiance_source"], d["fallback_reason"]):
        assert (why is None) == (src == "satellite")
    assert "missing_or_flagged_step" in sm["fallback_explanations"]


def test_northern_site_falls_back_to_era5_with_a_reason(live):
    for q, reason in ((NORTH, "outside_satellite_domain"), (MID_NORTH, "no_coverage")):
        r = live.json(f"/api/v1/solar/simulate?{q}&{WIN}&mode=hybrid")[1]
        sm = r["summary"]
        assert r["cell"]["satellite_covered"] is False and sm["satellite_hours"] == 0 and sm["fallback_hours"] == 96 and sm["fallback_reasons"][reason] >= 90
        assert set(sm["fallback_reasons"]) <= {reason, "missing_or_flagged_step"} and sum(sm["fallback_reasons"].values()) == 96   # flagged steps are reported as such
        assert set(r["data"]["irradiance_source"]) == {"era5_fallback"} and set(r["data"]["fallback_reason"]) <= {reason, "missing_or_flagged_step"} and reason in r["data"]["fallback_reason"]
        assert "Northern" in r["mode"]["explanation"] and "ERA5" in r["mode"]["explanation"] and reason in sm["fallback_explanations"]
    # a fallback cell gives exactly the ERA5-only numbers, a covered cell does not
    north_h = live.json(f"/api/v1/solar/simulate?{NORTH}&{WIN}&mode=hybrid&capacity_mwp=3")[1]["data"]["power_mw"]
    north_e = live.json(f"/api/v1/solar/simulate?{NORTH}&{WIN}&mode=era5&capacity_mwp=3")[1]["data"]["power_mw"]
    assert north_h == north_e
    sh = live.json(f"/api/v1/solar/simulate?{SOUTH}&{WIN}&mode=hybrid&capacity_mwp=3")[1]
    se = live.json(f"/api/v1/solar/simulate?{SOUTH}&{WIN}&mode=era5&capacity_mwp=3")[1]
    assert sh["data"]["power_mw"] != se["data"]["power_mw"] and sh["summary"]["total_mwh"] != se["summary"]["total_mwh"]
    assert set(se["data"]["irradiance_source"]) == {"era5"} and se["summary"]["satellite_hours"] == 0 and set(se["data"]["fallback_reason"]) == {None}
    assert "ERA5-only" in se["mode"]["explanation"] and se["summary"]["era5_hours"] == 96


def test_default_mode_is_hybrid_when_available_and_dates_can_be_a_sub_range(live):
    r = live.json(f"/api/v1/solar/simulate?{SOUTH}&start=2025-07-30T05:00Z&end=2025-07-30T08:00Z")[1]
    assert r["request"]["mode"] == "hybrid" and r["data"]["time_utc"] == [f"2025-07-30T0{h}:00:00Z" for h in (5, 6, 7, 8)] and r["summary"]["hours"] == 4
    whole = live.json(f"/api/v1/solar/simulate?{SOUTH}")[1]["request"]
    assert (whole["start"], whole["end"]) == ("2025-07-28T00:00:00Z", "2025-07-31T23:00:00Z")


# ----------------------------------------------------------------------------------------------------------- errors
@pytest.mark.parametrize("query, status, code, param", [
    ("lon=25", 400, "missing_parameter", "lat"), ("lat=60.2", 400, "missing_parameter", "lon"),
    ("lat=abc&lon=25", 400, "invalid_number", "lat"), ("lat=60.2&lon=nan", 400, "invalid_number", "lon"),
    ("lat=60.2&lon=25&capacity_mwp=x", 400, "invalid_number", "capacity_mwp"),
    ("lat=60.2&lon=25&capacity_mwp=0.0009", 422, "capacity_out_of_range", "capacity_mwp"),
    ("lat=60.2&lon=25&capacity_mwp=1000.5", 422, "capacity_out_of_range", "capacity_mwp"),
    ("lat=60.2&lon=25&capacity_mwp=-1", 422, "capacity_out_of_range", "capacity_mwp"),
    ("lat=60.2&lon=25&mode=sun", 422, "unknown_mode", "mode"),
    ("lat=40&lon=25", 422, "outside_grid", "lat"), ("lat=60.2&lon=5", 422, "outside_grid", "lat"), ("lat=71&lon=25", 422, "outside_grid", "lat"),
    ("lat=59.5&lon=25", 422, "outside_finland", "lat"), ("lat=64&lon=19.5", 422, "outside_finland", "lat"),
    ("lat=60.2&lon=25&start=2025-06-01", 422, "date_out_of_range", "start"), ("lat=60.2&lon=25&end=2025-09-01", 422, "date_out_of_range", "end"),
    ("lat=60.2&lon=25&start=2025-07-30&end=2025-07-29", 422, "end_before_start", "end"),
    ("lat=60.2&lon=25&start=2025-07-30T05:30Z", 400, "bad_date", "start"), ("lat=60.2&lon=25&start=yesterday", 400, "bad_date", "start"),
    ("lat=60.2&lon=25&foo=1", 400, "unknown_parameter", "foo"), ("lat=60.2&lat=61&lon=25", 400, "duplicate_parameter", "lat"),
])
def test_simulate_errors_are_4xx_json_with_code_message_and_param(live, query, status, code, param):
    for path in ("/api/v1/solar/simulate", "/api/v1/solar/simulate.csv"):
        s, body, _ = live.json(f"{path}?{query}")
        assert s == status and body["error"]["code"] == code and body["error"]["param"] == param and body["error"]["message"]
        assert body["data_label"]["synthetic"] is True and body["data_label"]["run_mode"] == "solar-demo"


def test_national_errors(live):
    for q, code in (("start=2025-06-01", "date_out_of_range"), ("start=2025-07-30&end=2025-07-29", "end_before_start"), ("x=1", "unknown_parameter"), ("start=zzz", "bad_date")):
        s, body, _ = live.json(f"/api/v1/solar/national?{q}")
        assert s in (400, 422) and body["error"]["code"] == code


# ----------------------------------------------------------------------------------------------------------- CSV == JSON
def parse_csv(text):
    head = [ln[2:] for ln in text.splitlines() if ln.startswith("# ")]
    body = [ln for ln in text.splitlines() if not ln.startswith("#")]
    header = body[0].split(",")
    return head, header, [dict(zip(header, ln.split(","))) for ln in body[1:]]


def test_simulate_csv_equals_json_and_carries_every_header_item(live):
    q = f"{SOUTH}&{WIN}&capacity_mwp=3.5&mode=hybrid"
    r = live.json(f"/api/v1/solar/simulate?{q}")[1]
    s, h, body = live.get(f"/api/v1/solar/simulate.csv?{q}")
    assert s == 200 and h["Content-Type"] == "text/csv; charset=utf-8"
    assert h["Content-Disposition"] == 'attachment; filename="solar_simulation_hybrid_2025-07-28_2025-07-31_DEMO-SYNTHETIC.csv"'
    head, cols, rows = parse_csv(body.decode("utf-8"))
    assert cols == ["time_utc", "power_mw", "capacity_factor", "energy_mwh", "irradiance_source", "fallback_reason"] and len(rows) == 96
    d = r["data"]
    assert [x["time_utc"] for x in rows] == d["time_utc"] and [float(x["power_mw"]) for x in rows] == d["power_mw"]
    assert [float(x["capacity_factor"]) for x in rows] == d["capacity_factor"] and [float(x["energy_mwh"]) for x in rows] == d["energy_mwh"]
    assert [x["irradiance_source"] for x in rows] == d["irradiance_source"] and [x["fallback_reason"] or None for x in rows] == d["fallback_reason"]
    text = "\n".join(head)
    for needle in ("DEMO - SYNTHETIC DATA", "synthetic: true", "demo: true", "model_version: windpower-solar-", "run_mode: solar-demo", "mode: hybrid", "CSi (atlite), tilt 40.0 deg, azimuth 180.0 deg",
                   "inverter efficiency 0.9", "capacity_mwp: 3.5", "grid_resolution_deg: 0.25", "cell_center_lat_deg_n: 60.25", "cell_center_lon_deg_e: 25.0",
                   "cell_satellite_covered: true", "cell_satellite_coverage_fraction_daytime:", "no national calibration", "no curtailment", "fallback_reasons:",
                   "source_cm_saf_sarah3: Pfeifroth", "source_era5: Contains modified Copernicus", "source_fingrid: Fingrid Oyj open data", "CC BY 4.0",
                   f"total_energy_mwh: {r['summary']['total_mwh']}", "units: time_utc=ISO 8601 UTC hour start; power_mw=MW = capacity_factor x capacity_mwp"):
        assert needle in text, needle


def test_national_series_have_availability_unit_scope_and_csv_matches(live):
    s, n, _ = live.json("/api/v1/solar/national?start=2025-07-29&end=2025-07-30")
    assert s == 200 and len(n["data"]["time_utc"]) == 48 and n["request"] == {"start": "2025-07-29T00:00:00Z", "end": "2025-07-30T23:00:00Z"}
    ser = n["series"]
    assert all(v["available"] and v["reason"] is None and len(v["values"]) == 48 for v in ser.values())
    f248, s362, c267 = ser["fingrid_248_forecast_mw"], ser["fingrid_362_av06_small_scale_surplus_mw"], ser["fingrid_267_capacity_mw"]
    assert f248["unit"] == "MW" and f248["native_unit"] == "MWh/h" and "forecast, not measured" in f248["scope"] and min(f248["values"]) < 0       # negative night kept
    assert s362["native_unit"] == "kWh" and "not total generation" in s362["scope"] and len(s362["values_native_kwh"]) == 48
    assert np.allclose(np.array(s362["values_native_kwh"]) / 1000, s362["values"], atol=1e-3) and "not official" in c267["scope"] and "energy_gwh" not in c267["summary"]
    e = ser["era5_only_mw"]
    assert e["summary"]["energy_gwh"] == pytest.approx(sum(e["values"]) / 1000, abs=1e-4) and e["summary"]["hours_missing"] == 0 and e["summary"]["coverage_pct"] == 100.0
    assert all(x <= c for x, c in zip(e["values"], c267["values"])) and n["calibration"].startswith("none")
    s, h, body = live.get("/api/v1/solar/national.csv?start=2025-07-29&end=2025-07-30")
    assert s == 200 and 'filename="national_solar_2025-07-29_2025-07-30_DEMO-SYNTHETIC.csv"' in h["Content-Disposition"]
    head, cols, rows = parse_csv(body.decode("utf-8"))
    assert cols[0] == "time_utc" and {"era5_only_mw", "hybrid_mw", "fingrid_248_forecast_mw", "fingrid_362_av06_surplus_mw", "fingrid_362_av06_surplus_kwh", "fingrid_267_capacity_mw"} <= set(cols)
    assert [x["time_utc"] for x in rows] == n["data"]["time_utc"] and [float(x["era5_only_mw"]) for x in rows] == e["values"]
    assert [float(x["fingrid_362_av06_surplus_kwh"]) for x in rows] == s362["values_native_kwh"]
    text = "\n".join(head)
    assert "forecast, not measured generation" in text and "small-scale surplus only" in text and "synthetic: true" in text and "source_fingrid" in text and "never zero-filled" in text


# ----------------------------------------------------------------------------------------------------------- partial artifact
def test_partial_artifact_reports_exact_gaps_and_refuses_unavailable_modes(partial):
    s, m, _ = partial.json("/api/v1/solar/meta")
    assert s == 200 and m["partial"] is True and m["synthetic"] is False and m["label"].startswith("Retrospective") and "PARTIAL RUN" in m["notice"]
    reason = m["modes"]["era5"]["reason"]
    assert "ERA5 solar" in reason and "--era5-solar-path" in reason and "ingest-era5-solar" in reason
    assert m["modes"]["era5"]["available"] is False and m["modes"]["hybrid"]["available"] is False and m["modes"]["hybrid"]["reason"] and m["defaults"]["mode"] is None
    assert m["unavailable"]["modes"]["era5"] == reason and "FINGRID_API_KEY" in m["unavailable"]["series"]["fingrid_248_forecast_mw"]
    # grouped by cause: one entry per distinct cause with everything it affects, short texts for tiles / radio buttons
    causes = {c["id"]: c for c in m["unavailable"]["causes"]}
    assert list(causes) == ["era5", "fingrid_key"]
    assert causes["era5"]["short"] == "ERA5 solar data not ingested" and causes["fingrid_key"]["short"] == "FINGRID_API_KEY not set"
    assert [(a["kind"], a["key"]) for a in causes["era5"]["affects"]][:2] == [("mode", "era5"), ("mode", "hybrid")]
    assert {a["key"] for a in causes["era5"]["affects"] if a["kind"] == "series"} == {"era5_only_mw", "era5_only_popw_mw", "hybrid_mw", "hybrid_popw_mw"}
    assert {a["key"] for a in causes["fingrid_key"]["affects"]} == {"fingrid_248_forecast_mw", "fingrid_267_capacity_mw", "fingrid_362_av06_small_scale_surplus_mw"}
    assert causes["era5"]["detail"] == reason and sum(len(c["affects"]) for c in causes.values()) == 9
    assert m["modes"]["hybrid"]["reason_short"] == "ERA5 solar data not ingested" and m["series"]["fingrid_248_forecast_mw"]["reason_short"] == "FINGRID_API_KEY not set"
    assert m["series"]["era5_only_mw"]["reason_short"] == "ERA5 solar data not ingested"
    assert m["coverage"]["available"] and m["coverage"]["fully_covered_cells"] > 0 and m["grid"]["n_lat"] == 44 and m["grid"]["n_lon"] == 53 and m["time_range"]["hours"] == 48
    for mode in ("era5", "hybrid"):
        for ep in ("simulate", "simulate.csv"):
            s, body, _ = partial.json(f"/api/v1/solar/{ep}?lat=60.75&lon=22.0&mode={mode}")
            assert s == 422 and body["error"]["code"] == "mode_unavailable" and body["error"]["param"] == "mode" and m["modes"][mode]["reason"] in body["error"]["message"]
            assert body["data_label"]["partial"] is True
    s, body, _ = partial.json("/api/v1/solar/simulate?lat=60.75&lon=22.0")          # no mode given and none available
    assert s == 422 and body["error"]["code"] == "mode_unavailable" and "No irradiance mode is available" in body["error"]["message"]
    assert body["error"]["message"].count("no ERA5 solar files") == 1                # the shared cause is stated once, not once per mode
    assert partial.json("/api/v1/solar/simulate?lat=60.75&lon=22.0&mode=sun")[1]["error"]["code"] == "unknown_mode"


def test_partial_artifact_coverage_and_national_states(partial):
    s, c, _ = partial.json("/api/v1/solar/coverage")
    cov = np.array(c["cells"]["satellite_covered"]).reshape(44, 53)
    assert s == 200 and c["available"] and cov.sum() == partial.json("/api/v1/solar/meta")[1]["coverage"]["fully_covered_cells"] > 0 and c["synthetic"] is False
    s, n, _ = partial.json("/api/v1/solar/national")
    assert s == 200 and all(v["available"] is False and "values" not in v and v["reason"] for v in n["series"].values())
    assert "FINGRID_API_KEY" in n["series"]["fingrid_267_capacity_mw"]["reason"] and "ERA5" in n["series"]["era5_only_mw"]["reason"]
    assert n["series"]["hybrid_mw"]["reason"] and set(n["unavailable"]) == set(n["series"])
    s, h, body = partial.get("/api/v1/solar/national.csv")
    head, cols, rows = parse_csv(body.decode("utf-8"))
    assert s == 200 and cols == ["time_utc"] and "UNAVAILABLE: Fingrid credential unavailable: FINGRID_API_KEY" in "\n".join(head) and "partial_run: true" in "\n".join(head)
    assert "synthetic: false" in "\n".join(head) and not any("DEMO" in ln for ln in head)
    assert len(rows) == 48 and all(r["time_utc"] for r in rows)                    # only the time axis, no zero-filled series


# ----------------------------------------------------------------------------------------------------------- combined / single servers
def test_no_absolute_local_paths_in_anything_served_for_a_partial_artifact(partial, partial_dir):
    texts = []
    for ep in ("meta", "grid", "coverage", "national", "national.csv", "simulate?lat=60.75&lon=22.0&mode=era5", "simulate.csv?lat=60.75&lon=22.0", "simulate?lat=60.75&lon=22.0"):
        texts.append(partial.get(f"/api/v1/solar/{ep}")[2].decode("utf-8"))
    blob = "\n".join(texts)
    assert "data\\cache\\solar\\era5" in blob or "solar\\era5" in blob or "solar/era5" in blob           # the useful part of the location survives
    assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/](?!/)", blob) and str(partial_dir.parents[4]) not in blob and Path.home().name not in blob and "pytest-of" not in blob
    stored = (partial_dir / "manifest.json").read_text(encoding="utf-8") + (partial_dir / "national_hourly.json").read_text(encoding="utf-8")
    assert "pytest-of" not in stored and Path.home().name not in stored                                 # and the exporter no longer writes them


def test_server_strips_paths_from_artifacts_written_before_the_exporter_did(partial_dir, tmp_path):
    old = tmp_path / "old"
    shutil.copytree(partial_dir, old)
    dirty = "no ERA5 solar files in C:\\Users\\someone\\proj\\data\\cache\\solar\\era5 and no --era5-solar-path; see /home/someone/proj/x. (needs CDS credentials)"
    man = json.loads((old / "manifest.json").read_text())
    nat = json.loads((old / "national_hourly.json").read_text())
    for m in man["modes"].values():
        m["reason"] = dirty
    for k, v in nat["series"].items():
        if k in ("era5_only_mw", "hybrid_mw"):
            v["reason"] = dirty
    (old / "manifest.json").write_text(json.dumps(man))
    (old / "national_hourly.json").write_text(json.dumps(nat))
    lv = Live(old)
    try:
        blob = "".join(lv.get(f"/api/v1/solar/{ep}")[2].decode("utf-8") for ep in ("meta", "national", "national.csv", "simulate?lat=60.75&lon=22.0&mode=era5"))
        assert "someone" not in blob and "C:\\" not in blob and "/home/" not in blob
        assert "no ERA5 solar files in ...\\solar\\era5 and no --era5-solar-path" in blob or "no ERA5 solar files in ..." in blob
        assert "(needs CDS credentials)" in blob
    finally:
        lv.close()


def test_redact_paths_and_display_path_unit():
    from windpower.io_utils import display_path, redact_paths

    inside = C.REPO_ROOT / "data" / "cache" / "solar" / "era5"
    assert display_path(inside) == str(Path("data") / "cache" / "solar" / "era5")
    assert redact_paths(f"no files in {inside} and no flag") == f"no files in {Path('data') / 'cache' / 'solar' / 'era5'} and no flag"
    assert redact_paths(f"see {inside}.") == f"see {Path('data') / 'cache' / 'solar' / 'era5'}."                 # sentence period kept
    out = redact_paths("in C:\\Users\\bob\\work\\data\\era5; also /home/bob/work/era5, done")
    assert "bob" not in out and "C:\\" not in out and "/home" not in out and "; also" in out and ", done" in out
    for text in ("kWh/h and MWh/h", "--era5-solar-path / WINDPOWER_ERA5_SOLAR; run `ingest-era5-solar`", "ERA5/Copernicus licence", "(needs CDS credentials)"):
        assert redact_paths(text) == text


def test_cause_grouping_unit():
    assert SD.cause_of("no ERA5 solar files in data\\cache and no flag")[0] == "era5"
    assert SD.cause_of("SARAH-3 unavailable: no files")[0] == "sarah"
    assert SD.cause_of("Fingrid credential unavailable: FINGRID_API_KEY is not set in the environment")[1] == "FINGRID_API_KEY not set"
    cid, short = SD.cause_of("something else happened; more detail")
    assert cid.startswith("other:") and short == "something else happened"


# ----------------------------------------------------------------------------------------------------------- combined / single servers (continued)
def test_wind_routes_still_work_next_to_solar(both):
    s, m, _ = both.json("/api/v1/meta")
    assert s == 200 and m["synthetic"] and m["run_mode"] == "demo"
    s, r, _ = both.json(TW_QUERY)
    assert s == 200 and r["summary"]["hours"] > 0 and r["request"]["capacity_mw"] == 10
    assert both.json("/api/v1/archetypes")[0] == 200 and both.json("/api/v1/national")[0] == 200 and both.get("/api/v1/simulate.csv?" + TW_QUERY.split("?")[1])[0] == 200
    s, sm, _ = both.json("/api/v1/solar/meta")
    assert s == 200 and sm["run_mode"] == "solar-demo" and sm["kind"] == "solar" and m["run_mode"] != sm["run_mode"]
    s, body, _ = both.json("/api/v1/simulate?lat=abc&lon=25")                        # wind errors carry the WIND label, solar errors the solar one
    assert body["data_label"]["run_mode"] == "demo"
    assert both.json("/api/v1/solar/simulate?lat=abc&lon=25")[1]["data_label"]["run_mode"] == "solar-demo"
    s, h, body = both.get("/")
    assert s == 200 and b"solar-simulate" in body and h["Content-Security-Policy"].startswith("default-src 'none'")


TW_QUERY = "/api/v1/simulate?lat=61.0&lon=24.25&start=2025-07-01&end=2025-07-02&archetype=A_hub100&capacity_mw=10"


def test_missing_artifact_states(demo_dir, tmp_path_factory):
    wind_only = Live(wind_dir=TW.write_hand_site(tmp_path_factory.mktemp("wind_only")))
    solar_only = Live(solar_dir=demo_dir)
    try:
        s, body, _ = wind_only.json("/api/v1/solar/meta")
        assert s == 404 and body["error"]["code"] == "solar_artifact_not_built" and "python -m windpower solar-demo" in body["error"]["message"]
        assert "python -m windpower solar-all" in body["error"]["message"] and body["data_label"]["run_mode"] == "demo"          # wind label still attached
        for ep in ("grid", "coverage", "simulate?lat=60&lon=25", "simulate.csv?lat=60&lon=25", "national", "national.csv"):
            assert wind_only.json(f"/api/v1/solar/{ep}")[1]["error"]["code"] == "solar_artifact_not_built"
        assert wind_only.json("/api/v1/meta")[0] == 200 and wind_only.get("/")[0] == 200            # wind-only still serves wind and the page
        s, body, _ = solar_only.json("/api/v1/meta")
        assert s == 404 and body["error"]["code"] == "wind_artifact_not_built" and "python -m windpower demo" in body["error"]["message"] and body["data_label"]["run_mode"] == "solar-demo"
        assert solar_only.json("/api/v1/solar/meta")[0] == 200 and solar_only.json("/api/v1/simulate?lat=60&lon=25")[0] == 404
    finally:
        wind_only.close()
        solar_only.close()


def test_static_and_traversal_protection_with_solar(live):
    for path in ("/../README.md", "/%2e%2e/%2e%2e/pyproject.toml", "/js/../../server.py", "/.git/config", "/api/v1/solar/../meta", "/api/v1/solar/nothing"):
        s, _, _ = live.get(path)
        assert s in (404,)
    s, _, body = live.get("/api/v1/solar/nothing")
    assert b"/api/v1/solar/simulate.csv" in body                                    # the 404 lists the available endpoints, solar ones included
    for js in ("main.js", "solar.js", "ui.js"):
        s, h, _ = live.get(f"/js/{js}")
        assert s == 200 and h["Content-Type"].startswith("text/javascript") and "Content-Security-Policy" in h
    assert live.get("/api/v1/solar/meta", method="POST")[0] == 405


def test_loader_rejects_broken_solar_artifacts(demo_dir, tmp_path):
    bad = tmp_path / "a"
    shutil.copytree(demo_dir, bad)
    (bad / "cf_era5.bin").write_bytes(b"\x00\x01")
    with pytest.raises(D.SiteDataError, match="cf_era5.bin"):
        SD.SolarSiteData.load(bad)
    bad2 = tmp_path / "b"
    shutil.copytree(demo_dir, bad2)
    m = json.loads((bad2 / "manifest.json").read_text())
    m["kind"] = "wind-site"
    (bad2 / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(D.SiteDataError, match="not a solar site artifact"):
        SD.SolarSiteData.load(bad2)
    (bad2 / "grid.json").unlink()
    with pytest.raises(D.SiteDataError, match="missing"):
        SD.SolarSiteData.load(bad2)


# ----------------------------------------------------------------------------------------------------------- serve command options
def test_serve_resolves_wind_and_solar_artifacts(demo_dir, tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(W, "run", lambda site_dir, host, port, say=print, **kw: calls.append((site_dir and Path(site_dir), port, kw)) or 0)
    hand = TW.write_hand_site(tmp_path / "hand")
    root = tmp_path / "root"
    shutil.copytree(hand, root / "outputs" / "demo" / "site" / "v1")
    shutil.copytree(demo_dir, root / "outputs" / "solar-demo" / "site" / "solar" / "v1")
    wind_demo, solar_demo = root / "outputs" / "demo" / "site" / "v1", root / "outputs" / "solar-demo" / "site" / "solar" / "v1"
    assert cli.main(["serve", "--data-dir", str(root)]) == 0 and calls[-1] == (wind_demo, 8000, {"solar_dir": solar_demo})                  # both, auto-discovered
    assert cli.main(["serve", "--data-dir", str(root), "--mode", "demo", "--solar-mode", "solar-demo"]) == 0 and calls[-1] == (wind_demo, 8000, {"solar_dir": solar_demo})
    assert cli.main(["serve", "--data-dir", str(root), "--mode", "solar-demo"]) == 0 and calls[-1][2] == {"solar_dir": solar_demo}
    assert cli.main(["serve", "--data-dir", str(root), "--solar-site-dir", str(demo_dir)]) == 0 and calls[-1] == (None, 8000, {"solar_dir": demo_dir})   # solar only
    assert cli.main(["serve", "--site-dir", str(hand), "--solar-site-dir", str(demo_dir), "--port", "9"]) == 0 and calls[-1] == (hand, 9, {"solar_dir": demo_dir})
    assert cli.main(["serve", "--site-dir", str(hand)]) == 0 and calls[-1] == (hand, 8000, {})                                               # custom wind dir: no solar discovery
    only_wind = tmp_path / "only_wind"
    shutil.copytree(hand, only_wind / "outputs" / "demo" / "site" / "v1")
    assert cli.main(["serve", "--data-dir", str(only_wind)]) == 0 and calls[-1][2] == {}                                                    # wind-only keeps working


def test_serve_prints_generation_commands_for_missing_artifacts(demo_dir, tmp_path, capsys):
    assert cli.main(["serve", "--data-dir", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "SITE ARTIFACT NOT FOUND" in err and "python -m windpower demo" in err and "python -m windpower solar-demo" in err and "python -m windpower solar-all" in err
    assert cli.main(["serve", "--data-dir", str(tmp_path), "--solar-mode", "solar"]) == 2
    err = capsys.readouterr().err
    assert "solar-all" in err and "python -m windpower demo" not in err                                                                      # only what was asked for
    assert cli.main(["serve", "--data-dir", str(tmp_path), "--mode", "solar-demo"]) == 2 and "python -m windpower solar-demo" in capsys.readouterr().err
    assert cli.main(["serve", "--data-dir", str(tmp_path), "--mode", "demo"]) == 2 and "python -m windpower solar" not in capsys.readouterr().err
    bad = tmp_path / "outputs" / "solar-demo" / "site" / "solar" / "v1"
    shutil.copytree(demo_dir, bad)
    (bad / "source_hybrid.bin").write_bytes(b"")
    assert cli.main(["serve", "--data-dir", str(tmp_path), "--solar-mode", "solar-demo"]) == 2
    err = capsys.readouterr().err
    assert "SITE ARTIFACT INVALID" in err and "python -m windpower solar-demo" in err and "Traceback" not in err


def test_serve_via_subprocess_serves_wind_and_solar(smoke_run, demo_dir):
    """The real command line with both artifacts: / , /api/v1/meta and /api/v1/solar/meta answer 200 over loopback; the process is stopped."""
    _, _, wind_data = smoke_run
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    proc = subprocess.Popen([sys.executable, "-u", "-m", "windpower", "serve", "--site-dir", str(wind_data / "outputs" / "smoke" / "site" / "v1"),
                             "--solar-site-dir", str(demo_dir), "--port", str(port)], cwd=TW.REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        first = proc.stdout.readline() + proc.stdout.readline() + proc.stdout.readline()
        assert f"http://127.0.0.1:{port}/" in first and "Serving wind" in first and "Serving solar" in first and "DEMO / SYNTHETIC" in first
        for path in ("/", "/api/v1/meta", "/api/v1/solar/meta"):
            with NO_PROXY.open(f"http://127.0.0.1:{port}{path}", timeout=20) as r:
                assert r.status == 200
    finally:
        TW.stop_tree(proc)
    assert proc.poll() is not None


# ----------------------------------------------------------------------------------------------------------- static UI hygiene
def test_solar_ui_structure_routes_and_accessibility_hooks():
    html = (W.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for needle in ('id="nav-wind"', 'id="nav-solar"', 'href="#solar-simulate"', 'href="#solar-overview"', 'href="#simulate"', 'href="#overview"', 'aria-label="Energy source"',
                   'id="view-solar-simulate"', 'id="view-solar-overview"', '<fieldset', '<legend>Irradiance input</legend>', 'name="sol_mode"', 'id="sol-overlay"',
                   'id="sov-chart"', 'id="sov-chart2"', 'id="solar-missing"', 'id="wind-missing"', "Small-scale surplus only (kWh/h shown as MW,", "not total generation",
                   "Northern fallback", "Fixed panel", "40° tilt", "0.25°", "sol-map-dialog", 'aria-live="polite"',
                   'id="sol-run"', 'id="sol-run-note"', 'id="sol-notice-list"', 'id="sov-notice-list"', 'id="sov-note1"', 'id="sov-note2"', 'id="sov-more-empty"'):
        assert needle in html, needle
    assert html.count("<h1") == 1 and html.count('role="status"') >= 4
    js = (W.STATIC_DIR / "js" / "solar.js").read_text(encoding="utf-8")
    assert "Fingrid forecast (not measured generation)" in js and "/api/v1/solar" in js and "disabled = !info.available" in js and "Unavailable:" in js
    assert "run.disabled = firstAvailable === null" in js and "reason_short" in js and "Download CSV: none available" in js and "aria-disabled" in js
    assert "innerHTML" not in js and "eval(" not in js
    assert "widest" in (W.STATIC_DIR / "js" / "chart.js").read_text(encoding="utf-8")                      # direct-label margin follows the label length
    css = (W.STATIC_DIR / "style.css").read_text(encoding="utf-8")
    for needle in (".tile.unavailable", "hatch-line", "stroke-dasharray: 2 4", "stroke-dasharray: 8 4", "stroke-dasharray: 10 3 2 3", "prefers-color-scheme: dark"):
        assert needle in css, needle
    chart = (W.STATIC_DIR / "js" / "chart.js").read_text(encoding="utf-8")
    assert "gapHatch" in chart and "yMin" in chart                                                                  # hatching + negative forecast values
