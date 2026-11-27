"""Read-only provenance capture and replay, using only temporary fixtures."""

import json
import shutil

import pytest

from windpower import config as C
from windpower import fixtures as F
from windpower import fingrid, fingrid_solar
from windpower.pipeline import Run
from windpower.solar_pipeline import SolarRun
from windpower.source_inventory import capture_inventory, validate_inventory


def test_wind_used_inputs_and_portable_replay(tmp_path):
    run = Run("smoke", tmp_path / "private", say=lambda _: None)
    run.ingest_era5()
    # A matching but unusable file appears in manifest.files, not loaded_files.
    junk = run.era5_cache / "era5_private.nc"
    junk.write_bytes(b"not weather" * 10000)
    run = Run("smoke", tmp_path / "private", say=lambda _: None)
    run.fingrid_data()  # also generates synthetic fleet, using the current ERA5
    inv, bindings = capture_inventory(run)
    assert inv["synthetic"]
    assert set(inv["sources"]) == {"era5", "osm", "arcgis", *(f"fingrid-{i}" for i in fingrid.DATASETS)}
    assert all(len(s["files"]) == 2 and not s["unavailable"] for s in inv["sources"].values())
    assert junk not in bindings.values()
    assert {f["format"] for f in inv["sources"]["era5"]["files"]} == {"NetCDF", "ZIP (NetCDF members)"}
    text = json.dumps(inv)
    assert all(secret not in text for secret in ("private", "not weather", F.FAKE_API_KEY, "source_url"))
    root = tmp_path / "replay"
    for key, original in bindings.items():
        dst = root / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, dst)
    assert validate_inventory(json.loads(text), root) == {}
    keys = list(bindings)[:2]
    (root / keys[0]).unlink()
    with (root / keys[1]).open("r+b") as fh:
        fh.write(b"BROKEN")
    assert validate_inventory(inv, root) == {keys[0]: "missing", keys[1]: "changed"}


def test_sarah_inclusive_month_and_recorded_directories(tmp_path):
    # A lightweight recorded load for July: a private SIS flag and SID env path.
    sis, sid = tmp_path / "flag-secret", tmp_path / "env-secret"
    run = SolarRun("solar", tmp_path / "data", sarah_sis_dir=sis, env={C.SARAH_SID_ENV: str(sid)})
    sis.mkdir()
    sid.mkdir()
    dirs = run.sarah_dirs()
    days = [f"202507{i:02}" for i in range(1, 32)] + ["20250801"]
    read = {}
    for kind, directory in dirs.items():
        read[kind] = []
        for day in [*days, "20250802"]:
            file = directory / f"{kind}in{day}_private-order.nc"
            file.write_bytes(b"private sample")
            if day in days:
                read[kind].append(file.name)
    run.info["sarah"] = {"inputs": dirs, "files_read": read}
    run._memo["sarah_ongrid"] = object()
    inv, bindings = capture_inventory(run, sarah_ingested=True)
    assert len(bindings) == 64
    for kind, directory in dirs.items():
        src = inv["sources"]["sarah-" + kind.lower()]
        assert src["coverage"]["last_step"] == "2025-08-01T00:00:00Z"
        assert bindings[src["files"][-1]["path"]].parent == directory
        assert "20250801" in src["files"][-1]["path"]
    assert all(s not in json.dumps(inv) for s in ("secret", "private", "20250802"))
    read["SID"].pop()
    run.info["sarah"]["sha256"] = {"SIS": {read["SIS"][0]: "0" * 64}}
    inv, bindings = capture_inventory(run, sarah_ingested=True)
    assert "20250801: no daily file used" in inv["sources"]["sarah-sid"]["unavailable"]
    assert "bytes differ" in inv["sources"]["sarah-sis"]["unavailable"][0]
    assert len(bindings) == 62


def test_cached_sarah_without_ingest_hash_does_not_attest_replaced_raw(tmp_path):
    original = SolarRun("solar-smoke", tmp_path, hash_inputs=False, say=lambda _: None)
    original.ingest_sarah()
    _, bindings = capture_inventory(original, sarah_ingested=True)
    bindings["sarah-sis/SISin20250801.nc"].write_bytes(b"replacement after ingest")
    run = SolarRun("solar-smoke", tmp_path, say=lambda _: None)
    before = run.info["sarah"]
    run.sarah_ongrid(required=True)
    assert run.info["sarah"] is before
    inv, bindings = capture_inventory(run, sarah_ingested=False)
    assert len(bindings) == 3 and all(key.startswith("sarah-products/") for key in bindings)
    for key in ("sarah-sis", "sarah-sid"):
        assert not inv["sources"][key]["files"] and len(inv["sources"][key]["unavailable"]) == 3


@pytest.mark.parametrize("single", [False, True])
def test_solar_custom_era_public_and_partial_fingrid(tmp_path, single):
    run = SolarRun("solar-smoke", tmp_path / "data", say=lambda _: None)
    run.ingest_era5_solar()
    directory = tmp_path / "custom"
    directory.mkdir()
    custom = directory / "key-private.nc"
    only = next(run.era5_cache.glob("*.nc"))
    shutil.copyfile(only, custom)
    nested = directory / "nested"
    nested.mkdir()
    (nested / "unrelated.nc").write_text("secret")
    (directory / "token.txt").write_text("secret")

    run.ingest_solar_public()
    run.ingest_fingrid_solar()
    stale, sidecar = fingrid._cache_paths(run.fg_cache, 248, fingrid_solar.RAW_COLUMN[248])
    sidecar.unlink()  # raw CSV still exists, cannot be used by the adapter
    run = SolarRun("solar-smoke", tmp_path / "data", era5_solar_path=custom if single else directory)
    empty, bindings = capture_inventory(run)
    assert not bindings and all(s["unavailable"] for s in empty["sources"].values())
    run.era5_hourly()
    run.public_data()
    run.fingrid_solar()
    (directory / "new-stale.nc").write_bytes(b"new")
    inv, bindings = capture_inventory(run)
    assert bindings["era5-solar/input-0.nc"] == custom
    assert only not in bindings.values() and stale not in bindings.values()
    assert len(inv["sources"]["era5-solar"]["files"]) == 1
    assert inv["sources"]["era5-solar"]["coverage"]["last_stamp"] == "2025-08-01T00:00:00Z"
    assert all(len(inv["sources"][s]["files"]) == 2 for s in ("fingrid-267", "fingrid-362", "plants", "population", "municipalities"))
    assert inv["sources"]["fingrid-248"]["unavailable"]
    assert all(s not in json.dumps(inv) for s in ("private", "secret", "token", "unrelated", "stale"))
    if not single:
        custom.rename(nested / custom.name)
        inv, bindings = capture_inventory(run)
        assert not inv["sources"]["era5-solar"]["files"]
        assert inv["sources"]["era5-solar"]["unavailable"] == ["input-0.nc: file missing at capture"]


@pytest.mark.parametrize("locator", ["a/../outside", "a/C:private", "//server/share", "a/..\\escape", "a/NUL"])
def test_replay_rejects_nonportable_paths(tmp_path, locator):
    with pytest.raises(ValueError, match="unsafe inventory path"):
        validate_inventory({"schema_version": 1, "sources": {"a": {"files": [{"path": locator}]}}}, tmp_path)
