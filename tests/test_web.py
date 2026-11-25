from __future__ import annotations

import csv
import io
import json
import math
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest

from windpower import cli
from windpower import simulate as S
from windpower.web import data as D
from windpower.web import server as W

REPO = Path(__file__).resolve().parents[1]
LATS = [60.0 + 0.25 * k for k in range(6)]          # 60.0 .. 61.25
LONS = [21.0 + 0.25 * k for k in range(15)]         # 21.0 .. 24.5
I_HAND, J_HAND = 4, 13                               # cell centred on 61.0 N, 24.25 E (inside Finland)
Z0 = 0.0625                                          # exactly representable in float16
FLAGS = {"artifact_schema_version": "1.0", "model_version": "windpower-test", "run_mode": "demo", "synthetic": True, "demo": True,
         "label": "DEMO / SYNTHETIC - not real data", "units": {"power": "MW"}}


# --------------------------------------------------------------------------------- hand-made artifact
def write_hand_site(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    n_t = 48
    wind = np.full((n_t, len(LATS), len(LONS)), 8.0, dtype="<f2")
    wind[:24, I_HAND, J_HAND] = 4.0
    wind[24:, I_HAND, J_HAND] = 12.0
    rough = np.full((len(LATS), len(LONS)), Z0, dtype="<f2")
    (d / "wind_effective.bin").write_bytes(wind.tobytes())
    (d / "roughness_mean.bin").write_bytes(rough.tobytes())
    hours = [f"2025-07-{1 + h // 24:02d}T{h % 24:02d}:00:00Z" for h in range(n_t)]
    j = lambda name, obj: (d / name).write_text(json.dumps(obj), encoding="utf-8")  # noqa: E731
    j("wind_effective.meta.json", {**FLAGS, "arrays": {
        "wind_effective": {"file": "wind_effective.bin", "dtype": "float16", "byte_order": "little", "order": "C", "shape": list(wind.shape), "axes": ["time", "lat", "lon"]},
        "roughness_mean": {"file": "roughness_mean.bin", "dtype": "float16", "byte_order": "little", "order": "C", "shape": list(rough.shape), "axes": ["lat", "lon"]}},
        "time": {"first_utc": hours[0], "last_utc": hours[-1], "step": "1h", "count": n_t}})
    j("grid.json", {**FLAGS, "lats": LATS, "lons": LONS, "cell_size_deg": 0.25, "shape": [len(LATS), len(LONS)], "index_order": "[lat_index][lon_index]",
                    "supported": np.ones((len(LATS), len(LONS)), dtype=int).tolist()})
    archs = []
    for name, hub in (("A_hub100", 100.0), ("B_hub150", 150.0)):
        c = S.build_power_curve(3.0, 300.0, hub)
        archs.append({"name": name, "turbines": 10 if name[0] == "A" else 5, "installed_mw": 30.0 if name[0] == "A" else 15.0, "mean_rated_mw": 3.0,
                      "mean_hub_height_m": hub, "mean_specific_power_wm2": 300.0,
                      "power_curve": {"v_ms": np.round(c["V"], 3).tolist(), "power_mw": np.round(c["POW"], 6).tolist(),
                                      "power_per_unit": np.round(c["POW"] / c["P"], 6).tolist()}})
    j("archetypes.json", {**FLAGS, "curve_points": 61, "archetypes": archs})
    theo = [10.0 * h for h in range(n_t)]
    obs = [None if h in (5, 6, 30) else round(0.8 * theo[h], 3) for h in range(n_t)]
    j("national_hourly.json", {**FLAGS, "utc_hours": hours, "theoretical_mw": theo, "observed_mw": obs,
                               "observed_available": [int(v is not None) for v in obs], "installed_mw": 500.0,
                               "coverage": {"era5_hours": n_t, "observed_hours": n_t - 3, "missing_observed_hours": [hours[5], hours[6], hours[30]]}})
    j("manifest.json", {**FLAGS, "data_coverage": {"era5_hours": n_t, "observed_hours": n_t - 3}, "licences": {"atlite": {"name": "MIT"}},
                        "attribution": "SYNTHETIC test data"})
    return d


@pytest.fixture(scope="module")
def hand_dir(tmp_path_factory):
    return write_hand_site(tmp_path_factory.mktemp("hand_site"))


NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # loopback only, never via a system proxy


class Live:
    def __init__(self, site_dir):
        self.site = D.SiteData.load(site_dir)
        self.server = W.make_server(self.site, port=0, quiet=True)
        self.thread = W.start_in_thread(self.server)
        self.base = W.url_for(self.server).rstrip("/")

    def get(self, path, method="GET", data=None):
        req = urllib.request.Request(self.base + path, method=method, data=data)
        try:
            with NO_PROXY.open(req, timeout=20) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    def json(self, path):
        status, headers, body = self.get(path)
        return status, json.loads(body.decode("utf-8")), headers

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture(scope="module")
def live(hand_dir):
    lv = Live(hand_dir)
    yield lv
    lv.close()


@pytest.fixture(scope="module")
def smoke_live(smoke_run):
    code, _, data = smoke_run
    assert code == 0
    lv = Live(data / "outputs" / "smoke" / "site" / "v1")
    yield lv
    lv.close()


def curve_pu(speed, hub=100.0, rated=3.0, sp=300.0):
    """Independent re-statement of the aerodynamic power curve, interpolated linearly on the 0.5 m/s grid."""
    def pu_at(v):
        cp = min(max(0.50 - 0.00018 * sp, 0.40), 0.48)
        area = rated * 1e6 / min(max(sp, 170.0), 450.0)
        return min(rated, 0.5 * 1.225 * area * cp * v**3 / 1e6) / rated if 3.0 <= v <= 25.0 else 0.0
    lo = math.floor(speed * 2) / 2
    hi = lo + 0.5
    return pu_at(lo) + (pu_at(hi) - pu_at(lo)) * (speed - lo) / 0.5


Q = "lat=61.0&lon=24.25&start=2025-07-01&end=2025-07-02&archetype=A_hub100&capacity_mw=10"


# --------------------------------------------------------------------------------- meta and archetypes
def test_meta_labels_and_content(live):
    s, m, h = live.json("/api/v1/meta")
    assert s == 200 and m["api_version"] == "v1" and m["demo"] is True and m["synthetic"] is True and m["run_mode"] == "demo"
    assert m["model_version"] == "windpower-test" and "DEMO" in m["notice"] and "SYNTHETIC" in m["notice"]
    assert m["time_range"] == {"first_utc": "2025-07-01T00:00:00Z", "last_utc": "2025-07-02T23:00:00Z", "hours": 48, "step": "1h"}
    g = m["grid"]
    assert (g["n_lat"], g["n_lon"], g["resolution_deg"]) == (6, 15, 0.25) and g["lat_range"] == [60.0, 61.25]
    assert 27 < g["cell_km_at_mid_latitude"]["north_south"] < 29 and 12 < g["cell_km_at_mid_latitude"]["east_west"] < 15
    assert set(m["region"]["outlines_lonlat"]) == {"mainland", "aland"} and len(m["region"]["outlines_lonlat"]["mainland"]) == 37
    assert m["limits"]["capacity_mw"] == {"min": 0.01, "max": 1000.0, "default": 3.0}
    assert "No national calibration" in m["statements"]["calibration"] and "0.25 degrees" in m["statements"]["resolution"]
    assert "SYNTHETIC" in m["statements"]["real_data"]
    assert h["Cache-Control"] == "no-store" and h["Content-Type"].startswith("application/json")


def test_archetypes_endpoint(live):
    s, a, _ = live.json("/api/v1/archetypes")
    assert s == 200 and a["demo"] is True and a["curve_points"] == 61 and a["default"] == "A_hub100"   # most fleet MW
    assert [x["name"] for x in a["archetypes"]] == ["A_hub100", "B_hub150"]
    for x in a["archetypes"]:
        assert len(x["power_curve"]["v_ms"]) == 61 == len(x["power_curve"]["power_per_unit"])


# --------------------------------------------------------------------------------- arithmetic
def test_simulate_hand_computed_values(live):
    s, r, _ = live.json("/api/v1/simulate?" + Q)
    assert s == 200
    assert r["cell"]["lat_index"] == I_HAND and r["cell"]["lon_index"] == J_HAND
    assert (r["cell"]["center_lat"], r["cell"]["center_lon"], r["cell"]["resolution_deg"]) == (61.0, 24.25, 0.25)
    d = r["data"]
    assert len(d["time_utc"]) == 48 and d["time_utc"][0] == "2025-07-01T00:00:00Z" and d["time_utc"][-1] == "2025-07-02T23:00:00Z"
    # hub height = reference height -> log extrapolation factor 1, so speed in = speed used.
    # 4 m/s: 0.5*1.225*(3e6/300)*0.446*4^3/1e6 = 0.174832 MW per 3 MW turbine; 8 m/s: 1.398656 MW; 12 m/s: rated.
    pu4 = 0.174832 / 3
    assert d["power_mw"][0] == pytest.approx(10 * pu4, abs=2e-3)
    assert d["power_mw"][23] == pytest.approx(0.58277, abs=1e-3)
    assert set(d["power_mw"][24:]) == {10.0}                                           # 12 m/s: rated power x capacity
    assert d["capacity_factor"][30] == 1.0
    # other cells see 8 m/s
    s2, r2, _ = live.json("/api/v1/simulate?" + Q.replace("lat=61.0&lon=24.25", "lat=60.5&lon=22.0"))
    assert set(r2["data"]["power_mw"]) == {round(10 * 0.466219, 6)} and r2["data"]["power_mw"][0] == pytest.approx(4.66219, abs=1e-4)


def test_totals_cf_and_energy_identities(live):
    s, r, _ = live.json("/api/v1/simulate?" + Q)
    d, sm, cap = r["data"], r["summary"], r["request"]["capacity_mw"]
    assert all(0.0 <= c <= 1.0 for c in d["capacity_factor"])
    assert d["energy_mwh"] == d["power_mw"]                                           # MWh = MW x 1 h
    assert sm["total_mwh"] == pytest.approx(sum(d["power_mw"]), abs=1e-6) == pytest.approx(sum(d["energy_mwh"]), abs=1e-6)
    assert sm["hours"] == 48 and sm["mean_mw"] == pytest.approx(sum(d["power_mw"]) / 48, abs=1e-3)
    assert sm["peak_mw"] == max(d["power_mw"]) == 10.0
    assert sm["mean_capacity_factor"] == pytest.approx(sm["total_mwh"] / (cap * 48), abs=1e-6)
    assert sm["full_load_hours"] == pytest.approx(sm["total_mwh"] / cap, abs=0.01)
    for p, c in zip(d["power_mw"], d["capacity_factor"]):
        assert c == pytest.approx(p / cap, abs=1e-6)


def test_output_scales_linearly_with_capacity(live):
    _, a, _ = live.json("/api/v1/simulate?" + Q.replace("capacity_mw=10", "capacity_mw=1"))
    _, b, _ = live.json("/api/v1/simulate?" + Q.replace("capacity_mw=10", "capacity_mw=100"))
    assert np.allclose(np.array(b["data"]["power_mw"]), 100 * np.array(a["data"]["power_mw"]), atol=1e-4)
    assert a["data"]["capacity_factor"] == pytest.approx(b["data"]["capacity_factor"], abs=1e-6)


def test_hub_height_log_extrapolation_is_applied(live):
    q = "lat=60.5&lon=22.0&start=2025-07-01&end=2025-07-01&archetype=B_hub150&capacity_mw=3"
    _, r, _ = live.json("/api/v1/simulate?" + q)
    v_hub = 8.0 * math.log(150 / Z0) / math.log(100 / Z0)
    assert v_hub > 8.0
    assert r["data"]["power_mw"][0] == pytest.approx(3 * curve_pu(v_hub), abs=2e-3)
    _, a, _ = live.json("/api/v1/simulate?" + q.replace("B_hub150", "A_hub100"))
    assert r["data"]["power_mw"][0] > a["data"]["power_mw"][0]


def test_no_calibration_statement_in_responses(live):
    _, r, _ = live.json("/api/v1/simulate?" + Q)
    assert r["calibration"].startswith("none") and r["units"]["energy_mwh"].startswith("MWh")


def test_location_snaps_to_cell_and_defaults(live):
    _, a, _ = live.json("/api/v1/simulate?lat=61.1&lon=24.3&start=2025-07-01&end=2025-07-01")
    assert (a["cell"]["lat_index"], a["cell"]["lon_index"]) == (I_HAND, J_HAND)
    assert a["request"]["archetype"] == "A_hub100" and a["request"]["capacity_mw"] == 3.0       # defaults
    _, b, _ = live.json("/api/v1/simulate?lat=61.2&lon=24.25&start=2025-07-01&end=2025-07-01")
    assert b["cell"]["lat_index"] == I_HAND + 1
    s, _, _ = live.json("/api/v1/simulate?lat=61.3749&lon=24.25")                              # inside the last cell's edge (61.25 + 0.125)
    assert s == 200


def test_date_ranges(live):
    _, r, _ = live.json("/api/v1/simulate?lat=61.0&lon=24.25&start=2025-07-02&end=2025-07-02")
    assert r["request"]["start"] == "2025-07-02T00:00:00Z" and r["request"]["end"] == "2025-07-02T23:00:00Z" and r["summary"]["hours"] == 24
    _, h, _ = live.json("/api/v1/simulate?lat=61.0&lon=24.25&start=2025-07-01T10:00:00Z&end=2025-07-01T12:00Z")
    assert h["data"]["time_utc"] == ["2025-07-01T10:00:00Z", "2025-07-01T11:00:00Z", "2025-07-01T12:00:00Z"]
    _, one, _ = live.json("/api/v1/simulate?lat=61.0&lon=24.25&start=2025-07-01T05&end=2025-07-01T05")
    assert one["summary"]["hours"] == 1
    _, full, _ = live.json("/api/v1/simulate?lat=61.0&lon=24.25")
    assert full["summary"]["hours"] == 48


# --------------------------------------------------------------------------------- errors
BAD = [
    ("lon=24.25", 400, "missing_parameter", "lat"),
    ("lat=61.0", 400, "missing_parameter", "lon"),
    ("lat=abc&lon=24.25", 400, "invalid_number", "lat"),
    ("lat=nan&lon=24.25", 400, "invalid_number", "lat"),
    ("lat=inf&lon=24.25", 400, "invalid_number", "lat"),
    ("lat=61.0&lon=1e999", 400, "invalid_number", "lon"),
    ("lat=61.0&lon=24.25&capacity_mw=ten", 400, "invalid_number", "capacity_mw"),
    ("lat=0&lon=0", 422, "outside_grid", "lat"),
    ("lat=61.0&lon=40", 422, "outside_grid", "lat"),
    ("lat=61.4&lon=24.25", 422, "outside_grid", "lat"),                                   # beyond the last cell edge (61.375)
    ("lat=61.0&lon=24.25&capacity_mw=0", 422, "capacity_out_of_range", "capacity_mw"),
    ("lat=61.0&lon=24.25&capacity_mw=-5", 422, "capacity_out_of_range", "capacity_mw"),
    ("lat=61.0&lon=24.25&capacity_mw=1000.01", 422, "capacity_out_of_range", "capacity_mw"),
    ("lat=61.0&lon=24.25&archetype=Nope", 422, "unknown_archetype", "archetype"),
    ("lat=61.0&lon=24.25&start=2025-06-30", 422, "date_out_of_range", "start"),
    ("lat=61.0&lon=24.25&end=2025-07-03", 422, "date_out_of_range", "end"),
    ("lat=61.0&lon=24.25&start=2025-07-02&end=2025-07-01", 422, "end_before_start", "end"),
    ("lat=61.0&lon=24.25&start=2025-07-01T10:00&end=2025-07-01T09:00", 422, "end_before_start", "end"),
    ("lat=61.0&lon=24.25&start=yesterday", 400, "bad_date", "start"),
    ("lat=61.0&lon=24.25&start=2025-13-45", 400, "bad_date", "start"),
    ("lat=61.0&lon=24.25&end=01/07/2025", 400, "bad_date", "end"),
    ("lat=61.0&lon=24.25&start=2025-07-01T05:30:00Z", 400, "bad_date", "start"),
    ("lat=61.0&lon=24.25&start=2025-07-01T05:00:00%2B03:00", 400, "bad_date", "start"),
    ("lat=61.0&lon=24.25&lat=62", 400, "duplicate_parameter", "lat"),
    ("lat=61.0&lon=24.25&capacity=5", 400, "unknown_parameter", "capacity"),
]


@pytest.mark.parametrize("query, status, code, param", BAD)
def test_bad_requests_give_4xx_json(live, query, status, code, param):
    s, body, headers = live.json("/api/v1/simulate?" + query)
    assert (s, body["error"]["code"], body["error"]["param"]) == (status, code, param)
    assert body["error"]["message"] and "Traceback" not in json.dumps(body) and "File \"" not in json.dumps(body)
    assert body["data_label"]["demo"] is True and headers["Content-Type"].startswith("application/json")
    sc, hdr, csv_body = live.get("/api/v1/simulate.csv?" + query)                         # CSV endpoint fails the same way
    assert sc == status and hdr["Content-Type"].startswith("application/json") and json.loads(csv_body)["error"]["code"] == code


def test_outside_finland_even_inside_grid(smoke_live):
    # smoke grid spans 60-63 N, 21-25 E; (62.0, 21.0) is open sea (Gulf of Bothnia), more than 0.25 deg from the outline
    s, body, _ = smoke_live.json("/api/v1/simulate?lat=62.0&lon=21.0")
    assert s == 422 and body["error"]["code"] == "outside_finland"
    s, ok, _ = smoke_live.json("/api/v1/simulate?lat=61.5&lon=23.0")
    assert s == 200 and ok["synthetic"] is True and ok["demo"] is False and ok["run_mode"] == "smoke"


def test_national_bad_requests(live):
    for q, code in (("start=2025-08-01", "date_out_of_range"), ("start=2025-07-02&end=2025-07-01", "end_before_start"), ("start=x", "bad_date"),
                    ("lat=1", "unknown_parameter")):
        s, body, _ = live.json("/api/v1/national?" + q)
        assert 400 <= s < 500 and body["error"]["code"] == code
    s, body, _ = live.json("/api/v1/meta?x=1")
    assert s == 400 and body["error"]["code"] == "unknown_parameter"


# --------------------------------------------------------------------------------- CSV
def parse_csv(text):
    lines = text.splitlines()
    head = [ln for ln in lines if ln.startswith("# ")]
    rows = list(csv.reader(ln for ln in lines if not ln.startswith("# ")))
    return head, rows[0], rows[1:]


def test_simulate_csv_matches_json_and_carries_labels(live):
    s, r, _ = live.json("/api/v1/simulate?" + Q)
    sc, hdr, body = live.get("/api/v1/simulate.csv?" + Q)
    assert sc == 200 and hdr["Content-Type"] == "text/csv; charset=utf-8"
    assert hdr["Content-Disposition"].startswith("attachment; filename=") and hdr["Content-Disposition"].endswith('DEMO-SYNTHETIC.csv"')
    head, cols, rows = parse_csv(body.decode("utf-8"))
    assert cols == ["time_utc", "power_mw", "capacity_factor", "energy_mwh"]
    d = r["data"]
    assert len(rows) == len(d["time_utc"])
    for row, t, p, c, e in zip(rows, d["time_utc"], d["power_mw"], d["capacity_factor"], d["energy_mwh"]):
        assert row[0] == t and t.endswith("Z") and float(row[1]) == p and float(row[2]) == c and float(row[3]) == e
    text = "\n".join(head)
    assert head[0].startswith("# *** DEMO - SYNTHETIC DATA") and "demo: true" in text and "synthetic: true" in text
    for needle in ("model_version: windpower-test", "run_mode: demo", "cell_index_lat_lon: 4,13", "cell_center_lat_deg_n: 61.0", "cell_center_lon_deg_e: 24.25",
                   "grid_resolution_deg: 0.25", "cell_source:", "archetype: A_hub100", "installed_capacity_mw: 10.0", "period_utc: 2025-07-01T00:00:00Z .. 2025-07-02T23:00:00Z",
                   "units: time_utc=ISO 8601 UTC", "power_mw=MW", "energy_mwh=MW x 1 h", "calibration: none", "not a micro-sited estimate"):
        assert needle in text, needle
    assert f"total_energy_mwh: {r['summary']['total_mwh']}" in text


def test_national_json_coverage_numbers(live):
    s, n, _ = live.json("/api/v1/national")
    assert s == 200 and n["demo"] is True and n["installed_mw"] == 500.0
    sm = n["summary"]
    assert (sm["hours"], sm["observed_hours"], sm["missing_hours"], sm["observed_pct"]) == (48, 45, 3, 93.75)
    assert n["missing_ranges"] == [{"start": "2025-07-01T05:00:00Z", "end": "2025-07-01T06:00:00Z", "hours": 2},
                                   {"start": "2025-07-02T06:00:00Z", "end": "2025-07-02T06:00:00Z", "hours": 1}]
    assert n["missing_hours_utc"] == ["2025-07-01T05:00:00Z", "2025-07-01T06:00:00Z", "2025-07-02T06:00:00Z"] and n["missing_hours_truncated"] is False
    theo = [10.0 * h for h in range(48)]
    assert sm["theoretical_mean_mw"] == pytest.approx(sum(theo) / 48) and sm["theoretical_peak_mw"] == 470.0 and sm["theoretical_total_mwh"] == sum(theo)
    obs = [v for v in n["data"]["observed_mw"] if v is not None]
    assert len(obs) == 45 and sm["observed_mean_mw"] == pytest.approx(sum(obs) / 45, abs=1e-3) and sm["observed_total_mwh"] == pytest.approx(sum(obs), abs=1e-2)
    assert n["data"]["observed_mw"][5] is None and n["data"]["observed_available"][5] == 0 and n["data"]["observed_available"][4] == 1
    both = [t for t, o in zip(theo, n["data"]["observed_mw"]) if o is not None]
    assert sm["theoretical_mean_mw_on_observed_hours"] == pytest.approx(sum(both) / 45, abs=1e-3)
    assert "uncalibrated" in n["series"]["theoretical"] and "curtailed" in n["series"]["observed"]


def test_national_range_restricts_coverage(live):
    s, n, _ = live.json("/api/v1/national?start=2025-07-01T04:00:00Z&end=2025-07-01T07:00:00Z")
    assert n["summary"]["hours"] == 4 and n["summary"]["observed_hours"] == 2 and n["summary"]["observed_pct"] == 50.0
    assert n["missing_ranges"] == [{"start": "2025-07-01T05:00:00Z", "end": "2025-07-01T06:00:00Z", "hours": 2}]
    assert n["request"] == {"start": "2025-07-01T04:00:00Z", "end": "2025-07-01T07:00:00Z"}


def test_national_csv_matches_json(live):
    q = "start=2025-07-01&end=2025-07-01"
    _, n, _ = live.json("/api/v1/national?" + q)
    sc, hdr, body = live.get("/api/v1/national.csv?" + q)
    assert sc == 200 and hdr["Content-Type"].startswith("text/csv")
    head, cols, rows = parse_csv(body.decode("utf-8"))
    assert cols == ["time_utc", "theoretical_mw", "observed_fingrid_mw", "observed_available"] and len(rows) == 24
    d = n["data"]
    for row, t, a, b, k in zip(rows, d["time_utc"], d["theoretical_mw"], d["observed_mw"], d["observed_available"]):
        assert row[0] == t and float(row[1]) == a and (row[2] == "" if b is None else float(row[2]) == b) and int(row[3]) == k
    text = "\n".join(head)
    assert "DEMO - SYNTHETIC" in head[0] and "demo: true" in text and "UNCALIBRATED" in text and "may be curtailed" in text and "empty = missing hour" in text
    assert "observed_hours: 22 of 24" in text


# --------------------------------------------------------------------------------- static files and server hygiene
@pytest.mark.parametrize("path", ["/../pyproject.toml", "/..%2fpyproject.toml", "/%2e%2e/pyproject.toml", "/js/../../../pyproject.toml", "/js/..%5c..%5cpyproject.toml",
                                  "/%2e%2e%5cpyproject.toml", "/....//pyproject.toml", "//etc/passwd", "/C:/Windows/win.ini", "/js/%00.js", "/.git/config", "/js/", "/js",
                                  "/missing.html", "/__init__.py", "/api/v1/../../pyproject.toml"])
def test_path_traversal_and_non_files_are_404(live, path):
    s, headers, body = live.get(path)
    assert s == 404 and headers["Content-Type"].startswith("application/json") and b"root:" not in body and b"[tool." not in body


def test_raw_traversal_request_line(live):
    host, port = live.server.server_address[:2]
    with socket.create_connection((host, port), timeout=10) as sock:
        sock.sendall(b"GET /../../pyproject.toml HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        data = sock.recv(65536)
    assert data.startswith(b"HTTP/1.") and b" 404 " in data.split(b"\r\n")[0] and b"[build-system]" not in data


def test_static_index_and_assets_with_security_headers(live):
    for path, ctype in (("/", "text/html"), ("/index.html", "text/html"), ("/style.css", "text/css"), ("/js/main.js", "text/javascript"), ("/js/chart.js", "text/javascript")):
        s, h, body = live.get(path)
        assert s == 200 and h["Content-Type"].startswith(ctype) and len(body) > 100, path
        assert h["X-Content-Type-Options"] == "nosniff" and "default-src 'none'" in h["Content-Security-Policy"] and "script-src 'self'" in h["Content-Security-Policy"]
        assert h["Referrer-Policy"] == "no-referrer" and "connect-src 'self'" in h["Content-Security-Policy"]


def test_methods_head_and_unknown_api(live):
    s, h, body = live.get("/api/v1/meta", method="POST", data=b"x=1")
    assert s == 405 and json.loads(body)["error"]["code"] == "method_not_allowed" and h["Allow"] == "GET, HEAD"
    for m in ("PUT", "DELETE"):
        assert live.get("/api/v1/meta", method=m)[0] == 405
    s, h, body = live.get("/api/v1/meta", method="HEAD")
    assert s == 200 and body == b"" and int(h["Content-Length"]) > 100
    s, body, _ = live.json("/api/v1/nothing")
    assert s == 404 and body["error"]["code"] == "not_found"
    s, body, _ = live.json("/api/v2/meta")
    assert s == 404
    s, _, body = live.get("/api/v1/simulate?" + "x" * 3000)
    assert s == 414 and json.loads(body)["error"]["code"] == "uri_too_long"


def test_server_binds_loopback_by_default_and_port_is_exclusive(hand_dir):
    site = D.SiteData.load(hand_dir)
    srv = W.make_server(site, port=0, quiet=True)
    try:
        assert srv.server_address[0] == "127.0.0.1"
        port = srv.server_address[1]
        with pytest.raises(OSError):
            W.make_server(site, port=port, quiet=True)
    finally:
        srv.server_close()


def test_tests_cannot_reach_the_internet():
    with pytest.raises(RuntimeError, match="network access is disabled"):
        socket.create_connection(("93.184.216.34", 80), timeout=1)


def test_resolve_static_unit(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_text("a")
    (tmp_path / "index.html").write_text("i")
    (tmp_path / ".hidden").write_text("h")
    assert W.resolve_static(tmp_path, "/") == (tmp_path / "index.html").resolve()
    assert W.resolve_static(tmp_path, "/sub/a.txt") is not None
    for bad in ("/sub", "/.hidden", "/sub/../../x", "/a\\b", "/sub/a.txt%00", "/nope"):
        assert W.resolve_static(tmp_path, bad) is None


# --------------------------------------------------------------------------------- static content hygiene
class _Audit(HTMLParser):
    def __init__(self):
        super().__init__()
        self.labels, self.fields, self.ids, self.inline_style, self.inline_script, self.attrs = set(), [], set(), 0, 0, {}
        self._script = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        if tag == "label" and "for" in a:
            self.labels.add(a["for"])
        if tag in ("input", "select") and a.get("type") != "hidden":
            self.fields.append(a)
        if "style" in a:
            self.inline_style += 1
        if tag == "script" and "src" not in a:
            self.inline_script += 1
        if tag in ("html", "meta", "a", "main", "dialog", "svg", "output"):
            self.attrs.setdefault(tag, []).append(a)


def test_index_html_structure_and_accessibility():
    html = (W.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    p = _Audit()
    p.feed(html)
    assert p.attrs["html"][0]["lang"] == "en"
    assert any(m.get("name") == "viewport" for m in p.attrs["meta"]) and any(m.get("name") == "color-scheme" for m in p.attrs["meta"])
    assert p.attrs["a"][0]["class"] == "skip-link" and p.attrs["a"][0]["href"] == "#main" and "main" in p.ids
    assert p.fields and all(f["id"] in p.labels for f in p.fields), [f for f in p.fields if f["id"] not in p.labels]
    assert p.inline_style == 0 and p.inline_script == 0                                  # compatible with the strict CSP
    assert "<dialog" in html and "<details" in html and 'aria-live="polite"' in html and 'role="status"' in html
    assert 'id="data-banner"' in html and 'role="img"' in html and "<caption>" in html and 'scope="col"' in html
    assert html.count("<h1") == 1


def test_static_files_make_no_external_requests():
    for f in W.STATIC_DIR.rglob("*"):
        if f.suffix in (".html", ".css", ".js"):
            text = f.read_text(encoding="utf-8").replace("http://www.w3.org/2000/svg", "")
            assert not re.search(r"https?://|//cdn|@import|url\(\s*['\"]?https?", text), f.name
            assert "eval(" not in text and "innerHTML" not in text and "document.write" not in text, f.name


def test_css_has_required_accessibility_features():
    css = (W.STATIC_DIR / "style.css").read_text(encoding="utf-8")
    for needle in ("prefers-color-scheme: dark", "prefers-reduced-motion", ":focus-visible", "container-type", "clamp(", "forced-colors"):
        assert needle in css, needle


# --------------------------------------------------------------------------------- region mask, parsing, loading
def test_finland_region_mask():
    assert D.in_finland(61.5, 23.0) and D.in_finland(65.0, 25.4) and D.in_finland(60.2, 20.0)       # inland, Oulu region, Aland
    assert not D.in_finland(62.0, 21.0) and not D.in_finland(50.0, 10.0) and not D.in_finland(70.0, 22.0)
    assert D.in_finland(60.45, 21.8)                                                                 # coast, within the 0.25 deg tolerance
    from shapely.geometry import Polygon
    assert Polygon(D.MAINLAND).is_valid and Polygon(D.ALAND).is_valid


def test_cell_size_matches_documented_values():
    ns, ew = D.cell_size_km(60.0)
    assert ns == pytest.approx(27.8, abs=0.1) and ew == pytest.approx(13.9, abs=0.1)


def test_parse_helpers():
    assert D.parse_float("1.5", "x") == 1.5 and D.parse_float(None, "x", required=False, default=3.0) == 3.0
    for bad in ("", "1,5", "0x10", "1 2", "--1"):
        with pytest.raises(D.ApiError):
            D.parse_float(bad, "x")
    assert D.parse_utc("2025-07-01", "d", is_end=True)[0].hour == 23 and D.parse_utc("2025-07-01", "d", is_end=False)[0].hour == 0
    assert D.parse_utc("2025-07-01T05z", "d", is_end=False) == (D.parse_utc("2025-07-01T05:00:00Z", "d", is_end=False)[0], False)


def test_loader_rejects_broken_artifacts(hand_dir, tmp_path):
    with pytest.raises(D.SiteDataError, match="missing"):
        D.SiteData.load(tmp_path / "nothing")
    bad = tmp_path / "trunc"
    shutil.copytree(hand_dir, bad)
    (bad / "wind_effective.bin").write_bytes(b"\x00" * 10)
    with pytest.raises(D.SiteDataError, match="bytes"):
        D.SiteData.load(bad)
    schema = tmp_path / "schema"
    shutil.copytree(hand_dir, schema)
    g = json.loads((schema / "grid.json").read_text())
    g["artifact_schema_version"] = "9.9"
    (schema / "grid.json").write_text(json.dumps(g))
    with pytest.raises(D.SiteDataError, match="unsupported artifact_schema_version"):
        D.SiteData.load(schema)
    notjson = tmp_path / "nj"
    shutil.copytree(hand_dir, notjson)
    (notjson / "manifest.json").write_text("{oops")
    with pytest.raises(D.SiteDataError, match="not valid JSON"):
        D.SiteData.load(notjson)


# --------------------------------------------------------------------------------- real (smoke) artifact end to end
def test_real_artifact_cell_matches_independent_numpy(smoke_live):
    site = smoke_live.site
    s, r, _ = smoke_live.json("/api/v1/simulate?lat=61.5&lon=23.0&capacity_mw=7.5&archetype=" + site.default_archetype)
    assert s == 200
    i, j = r["cell"]["lat_index"], r["cell"]["lon_index"]
    arch = site.arch_by_name[r["request"]["archetype"]]
    v = site.wind[:, i, j].astype(float)
    z = max(float(site.roughness[i, j]), 1e-4)
    hub = v * np.log(arch["mean_hub_height_m"] / z) / np.log(100.0 / z)
    pu = np.interp(np.maximum(hub, 0), arch["power_curve"]["v_ms"], arch["power_curve"]["power_per_unit"], right=0.0)
    assert np.allclose(r["data"]["power_mw"], np.round(np.clip(pu, 0, 1) * 7.5, 6), atol=1e-9)
    assert r["summary"]["hours"] == 504 and all(0 <= c <= 1 for c in r["data"]["capacity_factor"])
    assert r["summary"]["total_mwh"] == pytest.approx(sum(r["data"]["power_mw"]), abs=1e-6)


def test_real_artifact_national_matches_manifest(smoke_live):
    _, n, _ = smoke_live.json("/api/v1/national")
    cov = smoke_live.site.manifest["data_coverage"]
    assert n["summary"]["hours"] == cov["era5_hours"] == 504 and n["summary"]["observed_hours"] == cov["observed_hours"] == 501
    assert n["summary"]["missing_hours"] == 3 and n["synthetic"] is True and n["demo"] is False
    _, m, _ = smoke_live.json("/api/v1/meta")
    assert m["synthetic"] is True and "SYNTHETIC" in m["notice"] and m["coverage"] == {"era5_hours": 504, "observed_hours": 501}


# --------------------------------------------------------------------------------- CLI serve
def test_serve_missing_artifact_prints_generation_command(tmp_path, capsys):
    code = cli.main(["serve", "--data-dir", str(tmp_path), "--port", "0"])
    err = capsys.readouterr().err
    assert code == cli.EXIT_UNAVAILABLE == 2
    assert "SITE ARTIFACT NOT FOUND" in err and "python -m windpower demo" in err and "no data is synthesised" in err
    code = cli.main(["serve", "--mode", "full", "--data-dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert code == 2 and "python -m windpower all --mode full" in err and "python -m windpower demo" not in err
    assert not (tmp_path / "outputs").exists()                                          # nothing was generated


def test_serve_picks_real_data_before_demo_and_passes_options(hand_dir, tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(W, "run", lambda site_dir, host, port, say=print: calls.append((Path(site_dir), host, port)) or 0)
    shutil.copytree(hand_dir, tmp_path / "outputs" / "demo" / "site" / "v1")
    assert cli.main(["serve", "--data-dir", str(tmp_path), "--port", "8765"]) == 0
    assert calls[-1] == (tmp_path / "outputs" / "demo" / "site" / "v1", "127.0.0.1", 8765)
    shutil.copytree(hand_dir, tmp_path / "outputs" / "full" / "site" / "v1")
    assert cli.main(["serve", "--data-dir", str(tmp_path)]) == 0
    assert calls[-1][0] == tmp_path / "outputs" / "full" / "site" / "v1" and calls[-1][2] == 8000
    assert cli.main(["serve", "--data-dir", str(tmp_path), "--mode", "demo", "--host", "127.0.0.2"]) == 0
    assert calls[-1][0] == tmp_path / "outputs" / "demo" / "site" / "v1" and calls[-1][1] == "127.0.0.2"
    assert cli.main(["serve", "--site-dir", str(hand_dir)]) == 0 and calls[-1][0] == hand_dir
    assert cli.main(["serve", "--port", "70000"]) == cli.EXIT_USAGE


def test_serve_reports_invalid_artifact(hand_dir, tmp_path, capsys):
    bad = tmp_path / "outputs" / "demo" / "site" / "v1"
    shutil.copytree(hand_dir, bad)
    (bad / "wind_effective.bin").write_bytes(b"")
    code = cli.main(["serve", "--data-dir", str(tmp_path), "--mode", "demo"])
    err = capsys.readouterr().err
    assert code == 2 and "SITE ARTIFACT INVALID" in err and "python -m windpower demo" in err and "Traceback" not in err


def test_run_prints_url_and_label(hand_dir, capsys, monkeypatch):
    lines = []
    monkeypatch.setattr(W.ThreadingHTTPServer, "serve_forever", lambda self, *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert W.run(hand_dir, "127.0.0.1", 0, say=lines.append) == 0
    assert any("DEMO / SYNTHETIC" in ln for ln in lines) and any(re.search(r"Open http://127\.0\.0\.1:\d+/", ln) for ln in lines)


def stop_tree(proc):
    """The venv launcher on Windows spawns the real interpreter as a child: kill the whole tree of this one PID."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=30)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=15)


def test_serve_via_subprocess_serves_200(smoke_run):
    """The real command line: start `python -m windpower serve`, fetch / and /api/v1/meta over loopback, stop it."""
    _, _, data = smoke_run
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    proc = subprocess.Popen([sys.executable, "-u", "-m", "windpower", "serve", "--mode", "smoke", "--data-dir", str(data), "--port", str(port)],
                            cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        first = proc.stdout.readline() + proc.stdout.readline()
        assert f"http://127.0.0.1:{port}/" in first and "SYNTHETIC" in first
        for path in ("/", "/api/v1/meta"):
            with NO_PROXY.open(f"http://127.0.0.1:{port}{path}", timeout=20) as r:
                assert r.status == 200
    finally:
        stop_tree(proc)
    assert proc.poll() is not None


# --------------------------------------------------------------------------------- PowerShell launcher
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")


@pytest.mark.skipif(POWERSHELL is None, reason="PowerShell not available")
def test_run_site_script_parses_and_reports_missing_artifact(tmp_path):
    parse = (f"$e=$null;$t=$null;[void][System.Management.Automation.Language.Parser]::ParseFile('{REPO / 'scripts' / 'run_site.ps1'}',[ref]$t,[ref]$e);"
             "if($e.Count){$e|%{$_.Message};exit 1}")
    r = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", parse], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    env = {k: v for k, v in __import__("os").environ.items()}
    env["VIRTUAL_ENV"] = str(Path(sys.executable).parents[1])
    env["PATH"] = str(Path(sys.executable).parent) + __import__("os").pathsep + env.get("PATH", "")
    r = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(REPO / "scripts" / "run_site.ps1"), "-DataDir", str(tmp_path / "empty"), "-Port", "0"],
                       cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180)
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "SITE ARTIFACT NOT FOUND" in out and "python -m windpower demo" in out
