import copy
import json
import shutil

import pytest

from windpower import dataset_record as D
from windpower import sarah, sarah_regrid
from windpower import solar_public as P
from windpower.export_site import SITE_DIR
from windpower.export_site_solar import SOLAR_SITE_DIR
from windpower.solar_pipeline import SolarRun
from windpower.source_inventory import validate_inventory


def test_wind_dataset_and_payload_verification(tmp_path, smoke_run):
    code, _, data = smoke_run
    assert code == 0
    site = tmp_path / "site"
    shutil.copytree(data / "outputs/smoke" / SITE_DIR, site)
    r = D.verify_dataset(site)
    assert r["dataset_id"] == "finland-wind" and len(r["version_id"]) == 64
    assert r["synthetic"] and not r["partial"] and r["quality"]["observed_hours"] == 501
    assert r["coverage"]["hours"] == 504 and r["access"]["raw_inputs_included"] is False
    for section, field in (("settings", "split_seed"), ("pipeline", "source_sha256")):
        changed = copy.deepcopy(r)
        changed[section][field] = "different"
        assert D.version_id(changed) != r["version_id"]
    changed = copy.deepcopy(r)
    changed["inputs"]["sources"]["era5"]["files"][0]["sha256"] = "0" * 64
    assert D.version_id(changed) != r["version_id"]
    with (site / "wind_effective.bin").open("r+b") as f:
        f.write(b"bad")
    with pytest.raises(ValueError, match="artifact mismatch"):
        D.verify_dataset(site)


def test_solar_replay_and_sidecar_byte_identity(tmp_path):
    run = SolarRun("solar-smoke", tmp_path / "private", say=lambda _: None)
    run.export_site_solar()
    site = run.out / SOLAR_SITE_DIR
    first = D.verify_dataset(site)
    assert first["dataset_id"] == "finland-solar-pv" and first["synthetic"] and not first["partial"]
    assert "private" not in json.dumps(first) and first["settings"]["era5_convention"] == "end"
    assert first["coverage"]["hours"] == 48 and first["settings"]["panel"] == "CSi"
    settings, quality = first["settings"], first["quality"]
    assert settings["selected_time"]["coverage_end_exclusive_utc"] == "2025-08-01T00:00:00Z"
    assert settings["selected_time"]["era5_stamp_rule"] == "radiation at h+1h; mean of temperatures at h and h+1h"
    assert settings["sarah_hourly"]["sample_offsets_minutes"] == [0, 30, 60]
    assert settings["sarah_hourly"]["weights"] == list(sarah.TRAPEZOID)
    assert quality["sarah_record_status"] == {str(k): v for k, v in sarah.STATUS_NAMES.items()}
    assert quality["sarah_pixel_hour_codes"] == {str(k): v for k, v in sarah.CAUSE_NAMES.items()}
    assert quality["hybrid_source_codes"] == {str(k): v for k, v in sarah_regrid.SOURCE_CODES.items()}
    assert settings["hybrid_irradiance"]["satellite_code"] == sarah_regrid.SRC_SATELLITE
    assert settings["national_allocation"]["cell_mw"] == "(capacity - total_large_plant_mw) * population_share + cell_large_plant_mw"
    # A new cache receipt time changes exact byte evidence, not scientific identity.
    meta = run.public_cache / P.MUNI_META
    content = json.loads(meta.read_bytes())
    content["retrieved_utc"] = "2099-01-01T00:00:00Z"
    meta.write_text(json.dumps(content), encoding="utf-8")
    (site / "stale-secret.bin").write_bytes(b"not produced")
    reused = SolarRun("solar-smoke", tmp_path / "private", say=lambda _: None)
    reused.export_site_solar()
    second = D.verify_dataset(site)
    assert "sarah-products" in second["inputs"]["sources"] and not second["quality"]["input_gaps"]
    assert first["inputs"]["sources"]["municipalities"] != second["inputs"]["sources"]["municipalities"]
    assert second["version_id"] == first["version_id"]
    assert "stale-secret" not in json.dumps(second)
    root = tmp_path / "relocated"
    for key, src in reused.source_bindings.items():
        dst = root / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    assert validate_inventory(second["inputs"], root) == {}
    # Replacing even a parseable cache sidecar after load must not silently
    # rebind the already-memoised model to the replacement.
    meta.write_text("{}", encoding="utf-8")
    reused.export_site_solar()
    stale = D.verify_dataset(site)
    assert stale["partial"] and "municipalities" in stale["quality"]["input_gaps"]
    assert stale["version_id"] != second["version_id"]
    assert meta not in reused.source_bindings.values()


def test_incomplete_prior_sarah_record_is_not_reused(tmp_path):
    r = SolarRun("solar-smoke", tmp_path, say=lambda _: None)
    r.ingest_sarah()
    dirs = r._synthetic_sarah_paths()
    assert r._sarah_products_current(dirs)
    del r.info["sarah"]["files_read"]
    assert not r._sarah_products_current(dirs)


def test_source_digest_normalizes_only_crlf(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "__file__", str(tmp_path / "module.py"))
    module = tmp_path / "pipeline.py"
    source = b"x = 1\ny = x\n"
    module.write_bytes(source)
    before = D.code_digest("pipeline")
    for count in (1, 2):  # mixed and all-CRLF checkouts
        module.write_bytes(source.replace(b"\n", b"\r\n", count))
        assert D.code_digest("pipeline") == before
    for edited in (source.replace(b"1", b"2"), source.replace(b"\n", b"\r"), source + b" ", source[:-1]):
        module.write_bytes(edited)
        assert D.code_digest("pipeline") != before
