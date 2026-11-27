"""Follow the links used by both Solar views, including the downloaded attachment."""

import shutil
from contextlib import closing

import pytest

from test_web_solar import Live, NORTH, WIN, demo_dir, demo_data, parse_csv  # noqa: F401


@pytest.mark.parametrize("recorded", [True, False], ids=["verified", "legacy"])
def test_solar_trace_through_download(tmp_path, demo_dir, recorded):
    site = tmp_path / "solar"
    shutil.copytree(demo_dir, site)
    sidecar = site / "dataset.json"
    if not recorded:
        sidecar.unlink()
    with closing(Live(solar_dir=site)) as live:
        # All data and verification are startup snapshots. No sidecar access on GET or CSV.
        sidecar.unlink(missing_ok=True)
        entry, = live.json("/api/v1/datasets")[1]["datasets"]
        meta = live.json(entry["access"]["meta"])[1]
        if recorded:
            assert live.json(meta["href"])[1]["version_id"] == entry["version_id"]
        else:
            assert entry["href"] is None and entry["provenance_status"] == "missing"
        for route, query in (("simulate", f"{NORTH}&{WIN}&capacity_mwp=2.5&mode=hybrid"), ("national", WIN)):
            status, result, _ = live.json(entry["access"][route] + "?" + query)
            assert status == 200
            for k in ("dataset_id", "version_id", "href", "provenance_status"):
                assert result[k] == entry[k] == meta[k]
            status, headers, csv = live.get(result["csv_href"])
            assert status == 200 and headers["Content-Disposition"].startswith("attachment;")
            head, _, rows = parse_csv(csv.decode("utf-8"))
            for k in ("dataset_id", "version_id", "href", "provenance_status"):
                assert f"{k}: {entry[k] or 'not recorded'}" in head
            assert [r["time_utc"] for r in rows] == result["data"]["time_utc"]
            if route == "simulate":
                assert {r["irradiance_source"] for r in rows} == {"era5_fallback"}
                assert all(r["fallback_reason"] for r in rows)
                for k in ("irradiance_source", "fallback_reason"):
                    assert [r[k] for r in rows] == result["data"][k]
                assert [float(r["power_mw"]) for r in rows] == result["data"]["power_mw"]
            else:
                assert [float(r["hybrid_mw"]) for r in rows] == result["series"]["hybrid_mw"]["values"]
            stale = result["csv_href"].replace("version_id=" + (entry["version_id"] or "not-recorded"),
                                               "version_id=" + "0" * 64)
            assert live.json(stale)[0] == 409
