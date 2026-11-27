"""Local input provenance for exporters (no fetching, fixture generation or writes).

After resolving exporter inputs on the *same* Run/SolarRun, call
``inventory, bindings = capture_inventory(run)``. Only ``inventory`` is public
JSON. ``bindings`` maps its portable file keys to private local Paths. To check
an independently staged copy, place each bound file at ``root / key`` and call
``validate_inventory(inventory, root)``. Neither operation copies raw data.

The inventory describes adapter inputs (ERA5 files opened before time clipping),
not a scan of output artifacts. Coverage is the
requested run window/grid unless a source gives a more specific range or vintage;
it is not a claim of gap-free measurements. Absent stages are not inferred from
old run_info or cache contents. Capture immediately after loading, with stable
input directories: not every adapter records an ingest-time digest.

For solar, save ``before = run.info.get("sarah")`` on construction, before any
stages. Pass ``sarah_ingested=(run.info.get("sarah") is not before)`` at
capture: fresh ingest replaces this section; product reuse does not. Without
that signal, raw SARAH lineage is attested only if the ingest digest matches.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING

import pandas as pd

from . import config as C
from . import fingrid, fingrid_solar, solar_public as public
from .io_utils import iso_utc, sha256_file
from .manifest import LICENCES_STATIC

if TYPE_CHECKING:
    from .pipeline import Run
    from .solar_pipeline import SolarRun


def _credit(key):
    return LICENCES_STATIC[key]["attribution"]


def _basename(name):
    if not name or PureWindowsPath(name).name != name or name in (".", "..") or ":" in name:
        raise ValueError("invalid recorded input filename")
    return name


class _Capture:
    def __init__(self, run):
        self.run = run
        self.sources = {}
        self.bindings = {}

    def source(self, key, attribution, coverage, active):
        self.sources[key] = {
            "attribution": attribution if not self.run.synthetic else "SYNTHETIC offline fixture",
            "coverage": coverage, "files": [], "unavailable": [] if active else ["not used by this run"],
        }
        return self.sources[key]

    def file(self, key, alias, path, fmt, expected=None):
        source = self.sources[key]
        locator = f"{key}/{alias}"
        if locator in self.bindings:
            raise ValueError("duplicate inventory file key")
        if not path.is_file():
            source["unavailable"].append(f"{alias}: file missing at capture")
            return
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            source["unavailable"].append(f"{alias}: bytes differ from recorded digest")
            return
        if fmt == "NetCDF" and zipfile.is_zipfile(path):
            fmt = "ZIP (NetCDF members)"
        self.bindings[locator] = path
        source["files"].append({"path": locator, "format": fmt, "sha256": digest,
                                "bytes": path.stat().st_size, "hash_basis": "matched record" if expected else "at capture"})

    def wind(self):
        r = self.run
        active = "era5" in r._memo
        self.source("era5", _credit("era5"), "run window", active)
        if active:
            era = r.info["era5"]
            hashes = {f["name"]: f["sha256"] for f in era["files"]}
            for i, name in enumerate(era["loaded_files"]):
                alias = name if re.fullmatch(r"era5_[0-9]{4}_[0-9]{2}\.nc", name) else f"era5_input-{i}.nc"
                self.file("era5", alias, r.era5_cache / _basename(name), "NetCDF", hashes[name])
        active = "fleet" in r._memo
        for key, name, sidecar, fmt, attr, timestamp in (
            ("osm", C.OSM_CACHE_FILE, C.OSM_META_FILE, "JSON", _credit("osm"), "base_timestamp_utc"),
            ("arcgis", C.ARCGIS_CACHE_FILE, C.ARCGIS_META_FILE, "GeoJSON",
             "Finnish wind turbine layer (ArcGIS); see layer metadata for attribution", "last_edit_utc"),
        ):
            meta = r.info["fleet"][key + "_meta"] if active else {}
            stamp = meta.get(timestamp) or meta.get("retrieved_utc")
            self.source(key, attr, iso_utc(stamp) if stamp else "snapshot date unavailable", active)
            if active:
                self.file(key, name, r.fleet_cache / name, fmt)
                meta_path = r.fleet_cache / sidecar
                if meta_path.is_file() or meta_path in r._memo.get("input_digests", {}):
                    self.file(key, sidecar, meta_path, "JSON")

    def sarah(self, ingested):
        r = self.run
        active = "sarah_ongrid" in r._memo
        days = pd.date_range(r.hours[0].normalize(), r.last_step.normalize(), freq="D")
        for kind in ("SIS", "SID"):
            key = "sarah-" + kind.lower()
            source = self.source(key, "Pfeifroth et al. (2023), SARAH-3, DOI 10.5676/EUM_SAF_CM/SARAH/V003",
                                 {"first_step": iso_utc(r.hours[0]), "last_step": iso_utc(r.last_step)}, active)
            if not active:
                continue
            record = r.info["sarah"]
            directory = Path(record["inputs"][kind])
            hashes = record.get("sha256", {}).get(kind) or {}
            read = {}
            for name in record["files_read"][kind]:
                match = re.fullmatch(kind + r"in([0-9]{8}).*\.nc", _basename(name))
                if not match:
                    raise ValueError("invalid recorded SARAH file")
                read[match[1]] = name
            for day in days:
                date = day.strftime("%Y%m%d")
                name = read.get(date)
                if name is None:
                    source["unavailable"].append(f"{date}: no daily file used")
                elif not ingested and not hashes.get(name):
                    source["unavailable"].append(f"{date}: no ingest digest to verify daily input")
                else:
                    self.file(key, f"{kind}in{date}.nc", directory / name, "NetCDF", hashes.get(name))
        if active and ingested is False:
            from .solar_pipeline import CELLS_FILE, HOURLY_FILE, PIXELS_FILE

            key = "sarah-products"
            self.source(key, "Locally regridded CM SAF SARAH-3 SIS/SID", "run window", True)
            for name, fmt in ((HOURLY_FILE, "NetCDF"), (PIXELS_FILE, "NetCDF"), (CELLS_FILE, "CSV")):
                self.file(key, name, r.out / name, fmt)

    def solar(self, sarah_ingested):
        r = self.run
        self.sarah(sarah_ingested)
        era = r._memo.get("era5_hourly")
        window = ({"first_stamp": iso_utc(era.window["first_required"]), "last_stamp": iso_utc(era.window["last_required"])}
                  if era is not None else "run window")
        source = self.source("era5-solar", _credit("era5"), window, era is not None)
        if era is not None:
            src = Path(era.source["path"])
            source["kind"] = "prepared" if era.source["kind"] == "prepared" else "raw"
            records = r.info["era5_solar"]["files"]
            # The live loader now records just its selected top-level files.
            # Never rediscover files (including nested namesakes) at export.
            if src.is_dir():
                paths = [src / _basename(f["name"]) for f in records]
            elif not src.is_file():
                source["unavailable"].append("recorded input location missing at capture")
                paths = []
            else:
                paths = [src]
            for i, p in enumerate(paths):
                expected = {f["sha256"] for f in records if f["name"] == p.name}
                if len(expected) != 1:
                    source["unavailable"].append(f"input-{i}: ambiguous recorded digest")
                    continue
                self.file("era5-solar", f"input-{i}.nc", p, "NetCDF", expected.pop())
        pub = r._memo.get("public")
        for key, name, meta, fmt, attr in (
            ("plants", public.PLANTS_FILE, public.PLANTS_META, "XLSX", "Suomen Uusiutuvat ry (2025)"),
            ("population", public.POP_FILE, public.POP_META, "CSV", "Statistics Finland"),
            ("municipalities", public.MUNI_FILE, public.MUNI_META, "GeoJSON", "Statistics Finland"),
        ):
            self.source(key, attr, "Finland, 2025", pub is not None)
            if pub is not None:
                self.file(key, name, r.public_cache / name, fmt, pub.meta[key].get("sha256"))
                self.file(key, meta, r.public_cache / meta, "JSON")

    def fingrid(self, solar):
        r = self.run
        data = r._memo.get("fingrid")
        for dataset, column in (fingrid_solar.RAW_COLUMN if solar else fingrid.DATASETS).items():
            key = f"fingrid-{dataset}"
            meta = None if data is None else data.cache_meta.get(dataset if solar else column)
            coverage = ({"first_start": iso_utc(meta["first_start_utc"]), "last_start": iso_utc(meta["last_start_utc"])}
                        if meta and meta.get("first_start_utc") and meta.get("last_start_utc") else "run window")
            source = self.source(key, _credit("fingrid"), coverage, meta is not None)
            if meta is not None:
                csv, sidecar = fingrid._cache_paths(r.fg_cache, dataset, column)
                self.file(key, csv.name, csv, "CSV")
                self.file(key, sidecar.name, sidecar, "JSON")
            elif solar and data is not None and dataset in data.unavailable:
                source["unavailable"] = ["series unavailable for requested window"]


def capture_inventory(run: Run | SolarRun, *, sarah_ingested: bool | None = None) -> tuple[dict, dict[str, Path]]:
    """Return public inventory and private replay-key -> original path bindings."""
    capture = _Capture(run)
    solar = run.mode in C.SOLAR_RUN_MODES
    if solar:
        capture.solar(sarah_ingested)
    else:
        capture.wind()
    capture.fingrid(solar)
    return ({"schema_version": 1, "mode": run.mode, "synthetic": run.synthetic,
             "window": {"first_hour": iso_utc(run.hours[0]), "last_hour": iso_utc(run.hours[-1]), "grid": run.grid.name},
             "sources": capture.sources}, capture.bindings)


def validate_inventory(inventory: dict, root: str | Path) -> dict[str, str]:
    """Check every attested file below root; return only missing/changed keys.

    Source-level `unavailable` entries remain gaps even if this returns {}.
    Unsafe locators (including links escaping root) are invalid, not missing.
    """
    if inventory["schema_version"] != 1:
        raise ValueError("unsupported source inventory version")
    root = Path(root).resolve()
    issues = {}
    for key, source in inventory["sources"].items():
        for f in source["files"]:
            locator = f["path"]
            parts = locator.split("/")
            if (len(parts) != 2 or parts[0] != key or
                    not all(re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]*", p) and not p.endswith(".")
                            and not re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", p.split(".")[0], re.I) for p in parts)):
                raise ValueError("unsafe inventory path")
            p = root.joinpath(*parts).resolve()
            if not p.is_relative_to(root):
                raise ValueError("inventory path outside replay root")
            if not p.is_file():
                issues[locator] = "missing"
            elif p.stat().st_size != f["bytes"] or sha256_file(p) != f["sha256"]:
                issues[locator] = "changed"
    return issues
