"""Offline, loopback catalog discovery, safe detail projection, and legacy access."""

import json
import shutil
from contextlib import closing

import pytest

from test_web import write_hand_site
from test_web_solar import Live, demo_data, demo_dir  # noqa: F401 - shared offline fixtures
from windpower import dataset_record as contract

DISCOVER = "/api/v1/datasets"
PRIVATE = "C:\\private\\cache?token=do-not-publish"
DIGEST = "a4" * 32


def sidecar(directory, dataset_id):
    obj = {
        "schema_version": "1.0", "dataset_id": dataset_id, "synthetic": True,
        "status": "partial", "partial": True, "run_mode": "demo" if dataset_id == "finland-wind" else "solar-demo",
        "coverage": {"first_hour_utc": "2025-07-01T00:00:00Z", "step": "1h", "end": PRIVATE},
        "inputs": {"sources": {"era5": {"files": [{"path": PRIVATE, "sha256": DIGEST, "bytes": 200,
                                                    "format": "NetCDF", "token": PRIVATE}],
                                          "unavailable": [PRIVATE], "attribution": PRIVATE},
                              PRIVATE: {"files": [], "unavailable": []}}},
        "pipeline": {"source_sha256": DIGEST, "command": PRIVATE, "versions": {"numpy": PRIVATE, PRIVATE: "1.0"}},
        "settings": {"site_calibration": False, "home": PRIVATE, "selected_time": {"era5_stamp_rule": PRIVATE},
                     "hybrid_irradiance": {"fallback_reason_priority": [PRIVATE]}},
        "quality": {"observed_hours": 45, "missing_observed_hours": PRIVATE, "sarah_record_status": {"0": PRIVATE}},
        "grid": {"n_lat": 6, "lon_range": [21, 24.5], "lat_range": [PRIVATE, 64]},
        "units": {"power": "MW", "time": PRIVATE},
        "provenance": {"status": "partial", "credentials": PRIVATE},
        "access": {"meta": PRIVATE}, "href": PRIVATE,
    }
    names = (contract.WIND_FILES if dataset_id == "finland-wind" else
             contract.SOLAR_BASE | {n for n in contract.SOLAR_BINS if (directory / n).is_file()})
    obj.update(metadata_identity={}, artifacts=[contract.file_record(directory, n) for n in sorted(names)])
    obj["version_id"] = contract.version_id(obj)
    (directory / "dataset.json").write_text(json.dumps(obj), encoding="utf-8")
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({**json.loads(manifest.read_bytes()),
                                    "files": obj["artifacts"] + [contract.file_record(directory, "dataset.json")]}),
                        encoding="utf-8")
    return obj["version_id"]


def test_discover_then_follow_selected_version_and_access(tmp_path, demo_dir):
    wind = write_hand_site(tmp_path / "site" / "v1")
    solar = tmp_path / "site" / "solar" / "v1"
    shutil.copytree(demo_dir, solar)
    wind_id = sidecar(wind, "finland-wind")
    solar_id = sidecar(solar, "finland-solar-pv")
    manifest = wind / "manifest.json"
    manifest.write_text(json.dumps({**json.loads(manifest.read_bytes()), "label": PRIVATE}), encoding="utf-8")
    with closing(Live(solar, wind)) as live:
        # Snapshot, not a disk-backed endpoint.
        (wind / "dataset.json").unlink()
        (solar / "dataset.json").write_text("invalid", encoding="utf-8")
        status, body, headers = live.json(DISCOVER)
        assert status == 200 and headers.get_content_type() == "application/json"
        assert "do-not-publish" not in json.dumps(body)
        assert [r["dataset_id"] for r in body["datasets"]] == ["finland-wind", "finland-solar-pv"]
        for rec in body["datasets"]:
            assert rec["coverage"]["first_hour_utc"] == "2025-07-01T00:00:00Z"
            assert rec["synthetic"] and rec["provenance_status"] == "available"
            assert rec["href"] == f"{DISCOVER}/{rec['dataset_id']}/versions/{rec['version_id']}"
            status, detail, _ = live.json(rec["href"])
            assert status == 200 and "do-not-publish" not in json.dumps(detail)
            assert detail["dataset_id"] == rec["dataset_id"]
            assert detail["version_id"] == rec["version_id"]
            assert detail["inputs"]["sources"]["era5"]["files"] == [
                {"locator": "era5/input-1", "sha256": DIGEST, "bytes": 200, "format": "NetCDF"}]
            assert detail["pipeline"] == {"source_sha256": DIGEST, "versions": {}}
            assert detail["settings"]["site_calibration"] is False
            assert detail["quality"] == {"observed_hours": 45, "sarah_record_status": {}}
            assert detail["grid"] == {"n_lat": 6, "lon_range": [21, 24.5]}
            assert detail["units"] == {"power": "MW"}
            assert detail["status"] == "partial" and detail["partial"] is True
            assert live.json(rec["access"]["meta"])[0] == 200
        wind_href = body["datasets"][0]["href"]
        for path in (wind_href.replace(wind_id, solar_id), wind_href.replace("finland-wind", "other"),
                     wind_href.replace(wind_id, "%2e%2e%2fmanifest.json")):
            status, error, _ = live.json(path)
            assert status == 404 and "do-not-publish" not in json.dumps(error)
        status, error, _ = live.json(DISCOVER + "/" + "a" * 2050)
        assert status == 414 and "do-not-publish" not in json.dumps(error)
        assert live.json(DISCOVER + "?unknown=1")[0] == 400
        assert live.get(DISCOVER, method="POST")[0] == 405


@pytest.mark.parametrize("raw,reason", [
    (None, "missing"), ("{", "invalid"), ("[]", "invalid"),
    (json.dumps({"schema_version": "1.0", "synthetic": True, "dataset_id": "finland-solar-pv",
                 "version_id": DIGEST}), "invalid"),
    (json.dumps({"schema_version": "1.0", "synthetic": True, "dataset_id": "finland-wind",
                 "version_id": "../" + PRIVATE}), "invalid"),
])
def test_unversioned_wind_stays_available(tmp_path, raw, reason):
    wind = write_hand_site(tmp_path / "wind")
    if raw is not None:
        (wind / "dataset.json").write_text(raw, encoding="utf-8")
    with closing(Live(wind_dir=wind)) as live:
        status, body, _ = live.json(DISCOVER)
        assert status == 200 and "do-not-publish" not in json.dumps(body)
        rec, = body["datasets"]
        assert rec["kind"] == "wind" and rec["provenance_status"] == reason
        assert rec["dataset_id"] is rec["version_id"] is rec["href"] is None
        assert rec["coverage"]["hours"] == 48
        assert live.json(rec["access"]["national"] + "?start=2025-07-01&end=2025-07-02")[0] == 200
        assert live.json(f"{DISCOVER}/finland-wind/versions/{DIGEST}")[0] == 404


def test_legacy_solar_without_wind(tmp_path, demo_dir):
    solar = tmp_path / "solar"
    shutil.copytree(demo_dir, solar, ignore=shutil.ignore_patterns("dataset.json"))
    with closing(Live(solar_dir=solar)) as live:
        status, body, _ = live.json(DISCOVER)
        rec, = body["datasets"]
        assert status == 200 and rec["kind"] == "solar"
        assert rec["provenance_status"] == "missing" and rec["href"] is None
        assert live.json(rec["access"]["grid"])[0] == 200
        assert live.json("/api/v1/meta")[0] == 404


def test_exported_solar_sidecar_alongside_legacy_wind(tmp_path, demo_dir):
    wind = write_hand_site(tmp_path / "wind")
    with closing(Live(demo_dir, wind)) as live:
        status, body, _ = live.json(DISCOVER)
        legacy, solar = body["datasets"]
        assert status == 200 and legacy["href"] is None
        status, detail, _ = live.json(solar["href"])
        assert status == 200 and detail["run_mode"] == "solar-demo"
        assert detail["settings"]["panel"] == "CSi" and detail["units"]["cf_hybrid"] == "1"
        assert detail["pipeline"]["versions"]["windpower"]
        assert detail["quality"]["input_evidence"] == "captured"
        assert detail["quality"]["national_series"]["hybrid_mw"] is True
        source = detail["inputs"]["sources"]["sarah-sis"]
        assert source["attribution"] == "SYNTHETIC offline fixture" and source["files"][0]["format"] == "NetCDF"
        assert source["files"][0]["locator"].startswith("sarah-sis/SISin")
        raw = json.loads((demo_dir / "dataset.json").read_bytes())
        assert detail["artifacts"] == raw["artifacts"]
        for k in ("selected_time", "sarah_hourly", "hybrid_irradiance", "national_allocation"):
            assert detail["settings"][k] == raw["settings"][k]
        for k in ("sarah_record_status", "sarah_pixel_hour_codes", "hybrid_source_codes"):
            assert detail["quality"][k] == raw["quality"][k]


@pytest.mark.parametrize("mismatch", ["sidecar", "payload", "swapped"])
def test_inconsistent_version_not_served(tmp_path, mismatch):
    wind = write_hand_site(tmp_path / "wind")
    version = sidecar(wind, "finland-wind")
    if mismatch == "swapped":
        other = write_hand_site(tmp_path / "other-wind")
        # Same dataset, different bytes; the other sidecar is internally valid.
        with (other / "grid.json").open("ab") as f:
            f.write(b"\n")
        version = sidecar(other, "finland-wind")
        shutil.copyfile(other / "dataset.json", wind / "dataset.json")
    elif mismatch == "sidecar":
        path = wind / "dataset.json"
        path.write_text(json.dumps({**json.loads(path.read_bytes()), "version_id": DIGEST}), encoding="utf-8")
        version = DIGEST
    else:
        with (wind / "grid.json").open("ab") as f:
            f.write(b"\n")
    with closing(Live(wind_dir=wind)) as live:
        rec, = live.json(DISCOVER)[1]["datasets"]
        assert rec["provenance_status"] == "invalid" and rec["href"] is None
        assert live.json(f"{DISCOVER}/finland-wind/versions/{version}")[0] == 404
        assert live.json(rec["access"]["meta"])[0] == 200
