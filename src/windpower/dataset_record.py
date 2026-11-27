"""Public dataset contract v1. See docs/dataset-record.md for identity and replay."""

from __future__ import annotations

import json
import platform
from pathlib import Path

from . import config as C
from .io_utils import atomic_write_json, sha256_bytes, sha256_file
from .manifest import package_versions
from .source_inventory import capture_inventory

NAME = "dataset.json"
WIND_FILES = {"wind_effective.bin", "roughness_mean.bin", "wind_effective.meta.json", "grid.json",
              "archetypes.json", "national_hourly.json"}
SOLAR_BASE = {"meta.json", "grid.json", "coverage.json", "national_hourly.json"}
SOLAR_BINS = {"cf_era5.bin", "cf_hybrid.bin", "source_hybrid.bin"}


def digest(value):
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                   allow_nan=False).encode("utf-8"))


def code_digest(modules):
    """Hash local module bytes with only checkout CRLF pairs normalized to LF."""
    return digest({name: sha256_bytes(Path(__file__).with_name(name + ".py").read_bytes().replace(b"\r\n", b"\n"))
                   for name in modules.split()})


def remember_files(run, paths):
    """At adapter load, bind bytes for which there is no native ingest digest."""
    consumed = run._memo.setdefault("input_digests", {})
    for path in paths:
        if path.is_file():
            consumed[path] = sha256_file(path)
        else:
            consumed.pop(path, None)


def export_inputs(run, *, sarah_ingested=None):
    """Collect live evidence, reject changed bytes, retain bindings *only* on the run."""
    inventory, bindings = capture_inventory(run, sarah_ingested=sarah_ingested)
    observed = run._memo.get("input_digests", {})
    metadata = {}
    for src in inventory["sources"].values():
        for f in src["files"][:]:
            key = f["path"]
            path = bindings[key]
            if f["hash_basis"] == "at capture" and observed.get(path) != f["sha256"]:
                src["unavailable"].append(f"{key}: no matching digest from load")
                src["files"].remove(f)
                del bindings[key]
                continue
            if key.endswith((".meta.json", ".snapshot.json")):
                raw = path.read_bytes()
                if sha256_bytes(raw) != f["sha256"]:
                    raise ValueError("input metadata changed during export")
                obj = json.loads(raw)
                for field in ("fetched_at_utc", "retrieved_utc"):
                    obj.pop(field, None)
                metadata[key] = digest(obj)
    run.source_bindings = bindings
    return inventory, metadata


def version_id(record):
    """Canonical identity projection. Preserve byte evidence in the public record."""
    body = dict(record)
    body.pop("version_id", None)
    sources = body.pop("inputs")["sources"]
    normalized = body.pop("metadata_identity")
    inputs = {}
    for key, src in sources.items():
        # A verified intermediate cache is an alternative encoding of the same
        # raw lineage. If that lineage is incomplete, its bytes ARE the input.
        if key == "sarah-products" and all(not sources[k]["unavailable"] for k in ("sarah-sis", "sarah-sid")):
            continue
        inputs[key] = {"files": {f["path"]: normalized.get(f["path"], f["sha256"]) for f in src["files"]},
                       "unavailable": sorted(src["unavailable"])}
    return digest(body | {"inputs": inputs})


def file_record(site, name):
    p = site / name
    if not p.is_file() or p.resolve().parent != site.resolve():
        raise ValueError("missing or non-local artifact")
    return {"path": name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}


def write_dataset(site, *, solar, mode, inputs, files, coverage, grid, units, settings, quality):
    """Write after payloads, before legacy manifest. Never scan an output directory."""
    synthetic = C.is_synthetic_mode(mode)
    inv, metadata = inputs or ({"schema_version": 1, "mode": mode, "synthetic": synthetic,
                                 "window": coverage, "sources": {}}, {})
    gaps = [k for k, src in inv["sources"].items() if src["unavailable"] or not src["files"]]
    partial = bool(gaps or not inv["sources"] or quality.get("unavailable_modes")
                   or not all(quality.get("national_series", {}).values()))
    common = "__init__ config io_utils manifest dataset_record source_inventory era5 fingrid"
    modules = ("solar_pipeline export_site export_site_solar sarah sarah_regrid era5_solar solar_geometry solar_national "
               "solar_public solar_pv solar_validate fingrid_solar" if solar else
               "pipeline export_site fleet simulate calibrate validate")
    if synthetic:
        modules += " fixtures" + (" fixtures_solar" if solar else "")
    record = {
        "schema_version": "1.0", "dataset_id": "finland-solar-pv" if solar else "finland-wind",
        "run_mode": mode, "synthetic": synthetic, "partial": partial, "status": "partial" if partial else "complete",
        "coverage": coverage, "inputs": inv, "metadata_identity": metadata, "settings": settings,
        "quality": quality | {"input_gaps": sorted(gaps), "input_evidence": "captured" if inv["sources"] else "not supplied"},
        "grid": grid, "units": units,
        "pipeline": {"source_sha256": code_digest(common + " " + modules), "python": platform.python_version(),
                     "versions": package_versions()},
        "access": {"type": "local-site", "manifest": "manifest.json", "raw_inputs_included": False},
        "artifacts": [file_record(site, n) for n in sorted(files)],
    }
    record["version_id"] = version_id(record)
    atomic_write_json(site / NAME, record)
    return [*record["artifacts"], file_record(site, NAME)]


def verify_dataset(site):
    """Verify version, payload bytes and the legacy manifest's binding to the record."""
    site = Path(site)
    record = json.loads((site / NAME).read_bytes())
    if record["schema_version"] != "1.0" or record["version_id"] != version_id(record):
        raise ValueError("dataset version mismatch")
    allowed = {"finland-wind": WIND_FILES, "finland-solar-pv": SOLAR_BASE | SOLAR_BINS}[record["dataset_id"]]
    required = WIND_FILES if record["dataset_id"] == "finland-wind" else SOLAR_BASE
    names = {f["path"] for f in record["artifacts"]}
    if not required <= names <= allowed or len(names) != len(record["artifacts"]):
        raise ValueError("invalid dataset artifact list")
    manifest = json.loads((site / "manifest.json").read_bytes())
    evidence = {f["path"]: (f["sha256"], f["bytes"]) for f in manifest["files"]}
    if evidence.keys() != names | {NAME}:
        raise ValueError("manifest file list mismatch")
    for f in [*record["artifacts"], file_record(site, NAME)]:
        if f != file_record(site, f["path"]):
            raise ValueError(f"dataset artifact mismatch: {f['path']}")
        if evidence[f["path"]] != (f["sha256"], f["bytes"]):
            raise ValueError(f"manifest evidence mismatch: {f['path']}")
    return record
