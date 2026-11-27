"""Private-byte replay: these tests only write into pytest's isolated roots."""

from pathlib import Path

import pytest

from scripts import dataset_replay as replay
from windpower import config, fixtures, fixtures_solar
from windpower.dataset_record import verify_dataset


def test_offline_reexecution_and_fail_closed(solar_smoke_run, tmp_path, monkeypatch):
    code, _, data = solar_smoke_run
    assert code == 0
    cache = config.cache_dir("solar-smoke", data)

    def forbidden(*args, **kwargs):
        pytest.fail("replay attempted fixture generation")

    monkeypatch.setattr(fixtures, "write_sarah_cache", forbidden)
    monkeypatch.setattr(fixtures, "synthetic_era5_solar", forbidden)
    monkeypatch.setattr(fixtures_solar, "synthetic_public", forbidden)
    monkeypatch.setattr(fixtures_solar, "synthetic_solar_fingrid", forbidden)
    original = tmp_path / "original"
    replay.main(["build", "--mode", "solar-smoke", "--data-dir", str(original),
                 "--start", "2025-07-31T02:00", "--end", "2025-07-31",
                 "--sis-dir", str(cache / "sarah/SYNTHETIC_SIS"), "--sid-dir", str(cache / "sarah/SYNTHETIC_SID"),
                 "--public-dir", str(cache / "public"), "--fingrid-dir", str(cache / "fingrid"),
                 "--era5-path", str(next((cache / "era5").glob("*.nc"))), "--convention", "end"])
    site = Path("outputs/solar-smoke") / replay.SOLAR_SITE_DIR
    args = ["replay", "--site", str(original / site), "--inputs", str(original / "inputs"),
            "--reference-hourly", str(original / "outputs/solar-smoke" / replay.HOURLY_FILE)]
    rebuilt = tmp_path / "rebuilt"
    # Unrecorded files must never become adapter inputs.
    (original / "inputs/era5-solar/extra.nc").write_bytes(b"not a NetCDF")
    replay.main([*args, "--data-dir", str(rebuilt)])
    record = verify_dataset(rebuilt / site)
    assert record["coverage"]["hours"] == 22 and record["settings"]["convention_override"] == "end"
    assert not record["partial"] and record["synthetic"]
    with pytest.raises(FileExistsError):
        replay.main([*args, "--data-dir", str(rebuilt)])
    lookahead = original / "inputs/sarah-sis/SISin20250801.nc"
    with lookahead.open("r+b") as f:
        f.write(b"!")
    for expected in ("changed", "missing"):
        with pytest.raises(ValueError, match=expected):
            replay.main([*args, "--data-dir", str(tmp_path / "never-created")])
        lookahead.unlink(missing_ok=True)
    assert not (tmp_path / "never-created").exists()


def test_cached_only_is_not_raw_evidence(tmp_path):
    record = {"dataset_id": "finland-solar-pv", "inputs": {"schema_version": 1, "sources": {
        key: {"files": [], "unavailable": gaps} for key, gaps in
        [("sarah-products", []), ("sarah-sid", []), ("sarah-sis", ["20250801: no daily file used"])]}}}
    with pytest.raises(ValueError, match="cached SARAH"):
        replay.check_inputs(record, tmp_path)


def test_hourly_comparison_checks_missing_positions_and_coordinates(tmp_path):
    import xarray as xr

    before, after = tmp_path / "reference.nc", tmp_path / "rebuilt.nc"
    ds = xr.Dataset({"ghi": ("t", [float("nan"), 12.0], {"units": "W m-2"})}, coords={"t": [0, 1]})
    ds.to_netcdf(before)
    ds.to_netcdf(after, encoding={"ghi": {"zlib": True}})
    assert replay.compare_hourly(before, after)["variables"]["ghi"] == {"values": 2, "missing": 1}
    for changed in (ds.assign_coords(t=[1, 2]), ds.isel(t=[1, 0]).assign_coords(t=[0, 1]),
                    ds.assign_attrs(source="different")):
        changed.to_netcdf(after)
        with pytest.raises(AssertionError):
            replay.compare_hourly(before, after)
