"""Public, startup-only projection of the selected sites' dataset sidecars."""

from __future__ import annotations

import math
import re
from pathlib import Path

from .. import dataset_record, sarah
from ..sarah_regrid import SOURCE_CODES


def _obj(value):
    return value if isinstance(value, dict) else {}


def _matches(value, pattern):
    return isinstance(value, str) and re.fullmatch(pattern, value, re.ASCII) is not None


_DIGEST = r"[0-9a-f]{64}"
_TIME = r"\d{4}-\d\d-\d\d(?:T\d\d:\d\d:\d\d(?:Z|\+00:00))?"
_SOURCE_KEYS = ("era5", "osm", "arcgis", "sarah-sis", "sarah-sid", "sarah-products", "era5-solar", "plants",
                "population", "municipalities", "fingrid-181", "fingrid-248", "fingrid-124", "fingrid-319",
                "fingrid-267", "fingrid-362")
_STATUS = ("complete", "partial")
_MODES = ("full", "demo", "smoke", "solar", "solar-smoke", "solar-demo")
_CREDITS = ("SYNTHETIC offline fixture", "Contains modified Copernicus Climate Change Service information",
            "(c) OpenStreetMap contributors", "Fingrid Oyj open data", "Statistics Finland", "Suomen Uusiutuvat ry (2025)",
            "Finnish wind turbine layer (ArcGIS); see layer metadata for attribution", "Locally regridded CM SAF SARAH-3 SIS/SID",
            "Pfeifroth et al. (2023), SARAH-3, DOI 10.5676/EUM_SAF_CM/SARAH/V003")
_VERSION = r"\d{1,4}(?:\.\d{1,4}){1,3}(?:(?:rc|a|b)\d{1,3})?"
_MAX_BYTES = 4 * 1024 * 1024
_UNIT_KEYS = """power energy time capacity capacity_factor wind_speed roughness
    era5_only_mw hybrid_mw era5_only_popw_mw hybrid_popw_mw capacity_267_mw forecast_248_mw surplus_362_mw
    surplus_362_kwh surplus_362_count era5_only_gwh hybrid_gwh cf_era5 cf_hybrid capacity_mwp energy_mwh
    ghi_wm2 valid_area_fraction inhabitants large_capacity_mw population_share satellite_coverage_fraction
    mean_cf_era5 mean_cf_hybrid forecast_248_gwh surplus_362_gwh""".split()

# Identity verification binds bytes; it does not make manifest prose safe to serve.
_RULE_TEXT = {
    "selected_time": {
        "selection": "inclusive UTC hour starts; a date-only end includes the last available hour of that day",
        "represented_interval": "[h, h+1h)",
        "sarah_last_sample": "last selected hour + 1h (possibly next day); required for the mean, not another output hour",
    },
    "sarah_hourly": {
        "method": "trapezoid",
        "validity": "Hybrid requires all three SIS/SID pairs valid; fill/nonfinite is missing. Clip SIS-SID to zero within negative tolerance; reject below.",
    },
    "hybrid_irradiance": {
        "selection": "valid overlap area / full cell area >= satellite_min_area_fraction and finite direct and diffuse",
        "satellite": "area-weighted SID and SIS-SID over valid pixels",
        "fallback": "ERA5 direct and diffuse for all other cell-hours",
        "unchanged": "ERA5 TOA, temperature, albedo and solar position",
        "fallback_reason_priority": ["outside domain", "missing or flagged step", "partial valid area", "dominant cause by area if none valid"],
    },
    "national_allocation": {
        "capacity_series": "Fingrid 267 hourly MW; missing stays missing",
        "large_plant_mw": "operational plant capacity assigned by municipality centroid, in-grid plants only",
        "population_share": "inhabitants assigned by 1 km cell centre / total in-grid inhabitants",
        "cell_mw": "(capacity - total_large_plant_mw) * population_share + cell_large_plant_mw",
        "population_only_cell_mw": "capacity * population_share",
        "national_mw": "sum(cell_mw * cell_capacity_factor)",
    },
}


def _rules(settings):
    rules = {}
    for section, allowed in _RULE_TEXT.items():
        raw = settings.get(section)
        if not isinstance(raw, dict):
            continue
        out = {k: v for k, v in allowed.items() if raw.get(k) == v}
        if section == "selected_time":
            if _matches(raw.get("coverage_end_exclusive_utc"), _TIME):
                out["coverage_end_exclusive_utc"] = raw["coverage_end_exclusive_utc"]
            if "era5_stamp_rule" in raw and raw["era5_stamp_rule"] in (
                    None, "radiation at h+1h; mean of temperatures at h and h+1h", "radiation and temperature at h"):
                out["era5_stamp_rule"] = raw["era5_stamp_rule"]
        elif section == "sarah_hourly":
            out.update(_numbers(raw, "required_record_status diffuse_negative_tolerance_wm2"))
            for k, n in (("weights", 3), ("sample_offsets_minutes", 3), ("valid_sample_wm2_inclusive", 2)):
                v = raw.get(k)
                if isinstance(v, list) and len(v) == n and all(type(x) in (float, int) and -1e6 <= x <= 1e6 for x in v):
                    out[k] = v
        elif section == "hybrid_irradiance":
            out.update(_numbers(raw, "satellite_code"))
        rules[section] = out
    return rules


def _numbers(obj, keys):
    return {key: v for key in keys.split() if type(v := obj.get(key)) in (int, float)
            and (isinstance(v, int) or math.isfinite(v))}


def _dates(obj):
    return {key: obj[key] for key in ("start", "end", "first_hour", "last_hour", "first_utc", "last_utc",
                                      "first_hour_utc", "last_hour_utc", "first_step", "last_step",
                                      "first_stamp", "last_stamp", "first_start", "last_start")
            if _matches(obj.get(key), _TIME)}


def _inventory(raw):
    """Publish logical identifiers, never a path supplied by an input adapter."""
    sources = {}
    for key, value in _obj(raw.get("sources")).items():
        if key not in _SOURCE_KEYS:
            continue
        src = _obj(value)
        files = []
        for f in src["files"] if isinstance(src.get("files"), list) else ():
            f = _obj(f)
            if _matches(f.get("sha256"), _DIGEST):
                item = {"locator": f"{key}/input-{len(files) + 1}", "sha256": f["sha256"], **_numbers(f, "bytes")}
                if key in ("sarah-sis", "sarah-sid"):
                    kind = key[-3:].upper()
                    if _matches(f.get("path"), rf"{key}/{kind}in[0-9]{{8}}\.nc"):
                        item["locator"] = f["path"]
                if f.get("hash_basis") in ("matched record", "at capture"):
                    item["hash_basis"] = f["hash_basis"]
                if f.get("format") in ("NetCDF", "ZIP (NetCDF members)", "CSV", "JSON", "GeoJSON", "XLSX"):
                    item["format"] = f["format"]
                files.append(item)
        coverage = src.get("coverage")
        safe_coverage = (coverage if coverage in ("run window", "Finland, 2025", "snapshot date unavailable")
                         or _matches(coverage, _TIME) else _dates(_obj(coverage)))
        sources[key] = {"files": files, "coverage": safe_coverage}
        if src.get("kind") in ("raw", "prepared"):
            sources[key]["kind"] = src["kind"]
        if src.get("attribution") in _CREDITS:
            sources[key]["attribution"] = src["attribution"]
        if isinstance(src.get("unavailable"), list):
            sources[key]["unavailable_count"] = len(src["unavailable"])
    return {"sources": sources}


def _detail(raw):
    grid = _obj(raw.get("grid"))
    pipeline = _obj(raw.get("pipeline"))
    settings = _obj(raw.get("settings"))
    quality = _obj(raw.get("quality"))
    units = _obj(raw.get("units"))
    out = {
        "schema_version": raw["schema_version"],
        # verify_dataset checked each fixed payload name, size and digest against the site.
        "artifacts": [{k: f[k] for k in ("path", "bytes", "sha256")} for f in raw["artifacts"]],
        "inputs": _inventory(_obj(raw.get("inputs"))),
        "grid": _numbers(grid, "n_lat n_lon cell_size_deg res_deg resolution_deg"),
        "pipeline": {k: pipeline[k] for k, pat in (("source_sha256", _DIGEST), ("python", _VERSION))
                     if _matches(pipeline.get(k), pat)},
        "settings": _numbers(settings, "block_hours split_seed slope_deg azimuth_deg inverter_efficiency "
                                       "satellite_min_area_fraction cf_scale"),
        "quality": _numbers(quality, "observed_hours missing_observed_hours theoretical_hours fully_covered_cells"),
        "units": {k: units[k] for k in _UNIT_KEYS
                  if units.get(k) in ("MW", "MWp", "MWh", "GWh", "kWh", "UTC", "1", "m s-1", "m", "W m-2", "h")},
    }
    versions = _obj(pipeline.get("versions"))
    out["settings"].update(_rules(settings))
    for k, codes in (("sarah_record_status", sarah.STATUS_NAMES), ("sarah_pixel_hour_codes", sarah.CAUSE_NAMES),
                     ("hybrid_source_codes", SOURCE_CODES)):
        if isinstance(quality.get(k), dict):
            out["quality"][k] = {str(i): v for i, v in codes.items() if quality[k].get(str(i)) == v}
    out["pipeline"]["versions"] = {k: versions[k] for k in ("numpy", "pandas", "xarray", "netCDF4", "h5netcdf",
                                   "scipy", "scikit-learn", "geopandas", "shapely", "pyproj", "pyarrow", "requests",
                                   "atlite", "cdsapi", "windpower") if _matches(versions.get(k), _VERSION)}
    for k in ("site_calibration", "inference_series"):
        if type(settings.get(k)) is bool:
            out["settings"][k] = settings[k]
    for k, choices in (("engine", ("atlite",)), ("panel", ("CSi",)), ("era5_convention", ("end", "start")),
                       ("convention_override", ("end", "start"))):
        if settings.get(k) in choices:
            out["settings"][k] = settings[k]
    for k, choices in (("input_gaps", _SOURCE_KEYS), ("unavailable_modes", ("hybrid", "era5"))):
        if isinstance(quality.get(k), list):
            out["quality"][k] = [v for v in choices if v in quality[k]]
    if quality.get("input_evidence") in ("captured", "not supplied"):
        out["quality"]["input_evidence"] = quality["input_evidence"]
    if isinstance(quality.get("national_series"), dict):
        series = quality["national_series"]
        keys = ("era5_only_mw", "hybrid_mw", "era5_only_popw_mw", "hybrid_popw_mw", "fingrid_248_forecast_mw",
                "fingrid_362_av06_small_scale_surplus_mw", "fingrid_267_capacity_mw")
        out["quality"]["national_series"] = {k: series[k] for k in keys if type(series.get(k)) is bool}
    if grid.get("name") in ("finland", "smoke", "solar_smoke", "finland_solar"):
        out["grid"]["name"] = grid["name"]
    for k in ("lat_range", "lon_range"):
        v = grid.get(k)
        if isinstance(v, list) and len(v) == 2 and all(type(x) in (int, float) and -180 <= x <= 180 for x in v):
            out["grid"][k] = v
    return out


def _read(directory: Path):
    path = directory / "dataset.json"
    try:
        if path.is_symlink():
            return {}, "invalid"
        if not path.exists():
            return {}, "missing"
        if not path.is_file() or path.stat().st_size > _MAX_BYTES:
            return {}, "invalid"
        return dataset_record.verify_dataset(directory), "invalid"
    except (OSError, ValueError, TypeError, AttributeError, KeyError, RecursionError):
        # IO, malformed JSON/record structure, or mismatched digest/manifest. No error text is public.
        return {}, "invalid"


def build_catalog(site, solar, prefix):
    """Return discovery and the finite map of version routes. Never resolve request IDs on disk."""
    datasets, versions = [], {}
    for artifact, kind, expected, base, extra in (
        (site, "wind", "finland-wind", prefix, ("archetypes",)),
        (solar, "solar", "finland-solar-pv", prefix + "/solar", ("grid", "coverage")),
    ):
        if artifact is None:
            continue
        raw, fallback = _read(artifact.site_dir)
        record = {"kind": kind, "dataset_id": None, "version_id": None, "href": None,
                  "synthetic": artifact.synthetic, "provenance_status": fallback,
                  "coverage": {"first_hour_utc": artifact.hour_iso[0], "last_hour_utc": artifact.hour_iso[-1],
                               "hours": len(artifact.hour_iso)},
                  "access": {k: f"{base}/{k}" for k in ("meta", "simulate", "national", *extra)}}
        datasets.append(record)
        version = raw.get("version_id")
        if (raw.get("dataset_id") != expected or raw.get("synthetic") is not artifact.synthetic
                or raw.get("schema_version") != "1.0" or not _matches(version, _DIGEST)):
            continue
        href = f"{prefix}/datasets/{expected}/versions/{version}"
        cov = _obj(raw.get("coverage"))
        record.update(dataset_id=expected, version_id=version, href=href, provenance_status="available")
        for k, choices in (("status", _STATUS), ("run_mode", _MODES)):
            if raw.get(k) in choices:
                record[k] = raw[k]
        if type(raw.get("partial")) is bool:
            record["partial"] = raw["partial"]
        record["coverage"].update(_dates(cov), **_numbers(cov, "hours"))
        if cov.get("step") == "1h":
            record["coverage"]["step"] = "1h"
        if cov.get("time_label") == "start of the hour, UTC":
            record["coverage"]["time_label"] = cov["time_label"]
        versions[href] = {**record, **_detail(raw)}
    return {"datasets": datasets}, versions
