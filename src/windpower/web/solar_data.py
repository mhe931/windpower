"""Solar site artifact loader and the pure computations behind /api/v1/solar/* (no HTTP here).

Consumes only the preprocessed artifact `site/solar/v1` written by `export_site_solar`; it never reads raw data, credentials or the
wind artifact. Arithmetic: MW = capacity factor x capacity_mwp, MWh = MW x 1 h, capacity factor is a fraction. Night hours are 0
because the stored capacity factor is exactly 0 there. No national calibration and no curtailment correction is applied.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..io_utils import redact_paths
from .data import (ISO_Z, OUTLINES, REGION_TOLERANCE_DEG, ApiError, SiteDataError, _csv_header, _single, cell_size_km, in_finland, iso, parse_float,
                   parse_utc)

SUPPORTED_SCHEMA = "1.0"
CAPACITY_MIN_MWP, CAPACITY_MAX_MWP, CAPACITY_DEFAULT_MWP = 0.001, 1000.0, 1.0
MODES = ("era5", "hybrid")
MODE_TITLES = {"era5": "ERA5 only", "hybrid": "Hybrid (SARAH-3 satellite where covered, otherwise ERA5)"}
FALLBACK_PREFIX = "era5_fallback:"
GENERATE_HINT = ("Build it with `python -m windpower solar-demo` (SYNTHETIC, offline) or `python -m windpower solar-all` (real inputs; partial "
                 "without ERA5 / Fingrid), then restart the server.")
EXPLANATIONS = {
    "outside_satellite_domain": "this cell lies (almost) entirely outside the area covered by the SARAH-3 order (north of 65 N)",
    "no_coverage": "the satellite product has no daytime data over this cell (northern / north-eastern Finland is seen at a very low angle)",
    "partial_coverage": "less than 99.9 % of the cell area has valid satellite data",
    "missing_or_flagged_step": "a satellite time step was missing or flagged (record_status not ok) in that hour",
    "sis_missing": "the satellite global irradiance was missing in that hour", "sid_missing": "the satellite direct irradiance was missing in that hour",
    "negative_diffuse": "satellite diffuse irradiance (SIS - SID) was below -1 W m-2 in that hour",
}


def cause_of(reason: str) -> "tuple[str, str]":
    """(cause id, short text) of an unavailability reason; unknown reasons are their own cause (first sentence, shortened)."""
    low = reason.lower()
    if low.startswith("sarah-3") or "sarah" in low and "era5" not in low:
        return "sarah", "SARAH-3 satellite data not ingested"
    if "era5" in low:
        return "era5", "ERA5 solar data not ingested"
    if "fingrid_api_key" in low:
        return "fingrid_key", "FINGRID_API_KEY not set"
    short = reason.split(";")[0].split(". ")[0].strip()
    return "other:" + short, (short if len(short) <= 70 else short[:67] + "...")


@dataclass
class SolarSiteData:
    site_dir: Path
    manifest: dict[str, Any]
    grid: dict[str, Any]
    coverage: dict[str, Any]
    bins: dict[str, Any]
    national: dict[str, Any]
    lats: np.ndarray
    lons: np.ndarray
    hours: list[datetime]
    hour_iso: list[str]
    cf: dict[str, np.ndarray] = field(default_factory=dict)       # uint16 [time, lat, lon], capacity factor x scale
    source: "np.ndarray | None" = None                            # uint8 [time, lat, lon]
    covered: "np.ndarray | None" = None                           # bool [lat, lon]
    coverage_fraction: "np.ndarray | None" = None                 # float [lat, lon], NaN = no daytime hours

    # --- loading ------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, site_dir: "str | Path") -> "SolarSiteData":
        d = Path(site_dir)

        def read(name: str) -> Any:
            p = d / name
            if not p.is_file():
                raise SiteDataError(f"solar site artifact file missing: {p}")
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise SiteDataError(f"{p} is not valid JSON: {exc}") from None

        manifest, grid, cov, meta, nat = (read(n) for n in ("manifest.json", "grid.json", "coverage.json", "meta.json", "national_hourly.json"))
        for name, obj in (("manifest.json", manifest), ("grid.json", grid), ("coverage.json", cov), ("meta.json", meta), ("national_hourly.json", nat)):
            if obj.get("artifact_schema_version") != SUPPORTED_SCHEMA:
                raise SiteDataError(f"solar {name}: unsupported artifact_schema_version {obj.get('artifact_schema_version')!r} (need {SUPPORTED_SCHEMA})")
        if manifest.get("kind") != "solar-site":
            raise SiteDataError("manifest.json is not a solar site artifact (kind != 'solar-site')")
        g = grid["grid"]
        lats, lons = np.asarray(g["lat"], float), np.asarray(g["lon"], float)
        if (lats.size, lons.size) != (g["n_lat"], g["n_lon"]) or not (np.all(np.diff(lats) > 0) and np.all(np.diff(lons) > 0)):
            raise SiteDataError("solar grid.json axes are inconsistent or not ascending")
        ta = meta["time_axis"]
        first = datetime.strptime(ta["first"], ISO_Z)
        hours = [first + timedelta(hours=k) for k in range(int(ta["n"]))]
        hour_iso = [iso(h) for h in hours]
        if hour_iso[-1] != ta["last"] or nat["hours"] != hour_iso:
            raise SiteDataError("solar time axes of meta.json / national_hourly.json disagree")
        shape = (len(hours), lats.size, lons.size)
        site = cls(d, manifest, grid, cov, meta, nat, lats, lons, hours, hour_iso)
        for mode in MODES:
            info = manifest["modes"].get(mode, {})
            if not info.get("available"):
                continue
            spec = meta["files"][f"cf_{mode}.bin"]
            raw = (d / f"cf_{mode}.bin").read_bytes() if (d / f"cf_{mode}.bin").is_file() else b""
            if spec["dtype"] != "uint16" or tuple(spec["shape"]) != shape or len(raw) != int(np.prod(shape)) * 2:
                raise SiteDataError(f"cf_{mode}.bin does not match its metadata (shape {shape}, {len(raw)} bytes)")
            site.cf[mode] = np.frombuffer(raw, dtype="<u2").reshape(shape)
        if "hybrid" in site.cf:
            raw = (d / "source_hybrid.bin").read_bytes() if (d / "source_hybrid.bin").is_file() else b""
            if len(raw) != int(np.prod(shape)):
                raise SiteDataError("source_hybrid.bin does not match the grid and time axis")
            site.source = np.frombuffer(raw, dtype="u1").reshape(shape)
        cells = grid.get("cells", {})
        if "satellite_covered" in cells:
            site.covered = np.asarray(cells["satellite_covered"], dtype=bool).reshape(lats.size, lons.size)
            site.coverage_fraction = np.array([np.nan if v is None else v for v in cells["satellite_coverage_fraction"]], dtype=float).reshape(lats.size, lons.size)
        return site

    # --- labels -----------------------------------------------------------------------------------------------
    @property
    def synthetic(self) -> bool:
        return bool(self.manifest["synthetic"])

    @property
    def demo(self) -> bool:
        return bool(self.manifest["demo"])

    @property
    def unavailable_modes(self) -> dict[str, str]:
        return {m: redact_paths(str((self.manifest["modes"].get(m) or {}).get("reason") or "not available")) for m in MODES if m not in self.cf}

    @property
    def unavailable_series(self) -> dict[str, str]:
        return {k: redact_paths(str(v["reason"] or "not available")) for k, v in self.national["series"].items() if not v["available"]}

    def causes(self) -> "list[dict[str, Any]]":
        """Unavailable modes and series grouped by distinct cause (one entry per cause, in first-seen order)."""
        groups: dict[str, dict[str, Any]] = {}
        items = [("mode", m, MODE_TITLES[m].split(" (")[0] + " mode", r) for m, r in self.unavailable_modes.items()]
        items += [("series", k, self.national["series"][k]["title"], r) for k, r in self.unavailable_series.items()]
        for kind, key, title, reason in items:
            cid, short = cause_of(reason)
            g = groups.setdefault(cid, {"id": cid, "short": short, "detail": reason, "affects": []})
            g["affects"].append({"kind": kind, "key": key, "title": title})
        return list(groups.values())

    @property
    def partial(self) -> bool:
        return bool(self.unavailable_modes or self.unavailable_series)

    @property
    def notice(self) -> str:
        if self.demo:
            base = "DEMO - SYNTHETIC DATA. These numbers are generated test data, not real irradiance, plants or measurements."
        elif self.synthetic:
            base = "SYNTHETIC DATA (smoke-test fixture). These numbers are not real irradiance, plants or measurements."
        else:
            base = "Retrospective SARAH-3 / ERA5 PV simulation. Uncalibrated: no national calibration and no curtailment correction."
        if self.partial and not self.synthetic:
            base += " PARTIAL RUN: some inputs were unavailable (see the unavailable items below)."
        return base

    def label_block(self) -> dict[str, Any]:
        m = self.manifest
        return {"model_version": m["model_version"], "run_mode": m["run_mode"], "synthetic": self.synthetic, "demo": self.demo, "label": m["label"],
                "notice": self.notice, "artifact_schema_version": m["artifact_schema_version"], "partial": self.partial}

    @property
    def default_mode(self) -> "str | None":
        return "hybrid" if "hybrid" in self.cf else ("era5" if "era5" in self.cf else None)

    def _licences(self) -> dict[str, Any]:
        return {k: ({kk: v.get(kk) for kk in ("name", "attribution", "status")} if isinstance(v, dict) else v) for k, v in (self.manifest.get("licences") or {}).items()}

    # --- /solar/meta ----------------------------------------------------------------------------------------
    def meta(self) -> dict[str, Any]:
        g = self.grid["grid"]
        mid = float(self.lats[len(self.lats) // 2])
        ns, ew = cell_size_km(mid, g["res_deg"])
        um, us = self.unavailable_modes, self.unavailable_series
        modes = {m: {"title": MODE_TITLES[m], "available": m in self.cf, "reason": None if m in self.cf else um[m],
                     "reason_short": None if m in self.cf else cause_of(um[m])[1]} for m in MODES}
        series = {}
        for k, v in self.national["series"].items():
            rec = {kk: v[kk] for kk in ("title", "unit", "native_unit", "scope", "available")}
            series[k] = {**rec, "reason": None if v["available"] else us[k], "reason_short": None if v["available"] else cause_of(us[k])[1]}
        conv = self.manifest.get("convention") or {}
        return {
            **self.label_block(), "api_version": "v1", "kind": "solar",
            "time_range": {"first_utc": self.hour_iso[0], "last_utc": self.hour_iso[-1], "hours": len(self.hours), "step": "1h", "label": "start of the hour, UTC"},
            "grid": {"name": g["name"], "n_lat": g["n_lat"], "n_lon": g["n_lon"], "resolution_deg": g["res_deg"], "lat_range": [float(self.lats[0]), float(self.lats[-1])],
                     "lon_range": [float(self.lons[0]), float(self.lons[-1])], "index_order": "[lat ascending][lon ascending]",
                     "cell_km_at_mid_latitude": {"north_south": round(ns, 1), "east_west": round(ew, 1), "latitude": mid}},
            "region": {"outlines_lonlat": {k: [list(p) for p in v] for k, v in OUTLINES.items()}, "tolerance_deg": REGION_TOLERANCE_DEG,
                       "note": "Coarse outline used for map drawing and to reject locations more than 0.25 degrees outside it."},
            "limits": {"capacity_mwp": {"min": CAPACITY_MIN_MWP, "max": CAPACITY_MAX_MWP, "default": CAPACITY_DEFAULT_MWP}},
            "defaults": {"mode": self.default_mode}, "modes": modes,
            "panel": {k: self.manifest["panel"].get(k) for k in ("panel", "model", "slope_deg", "azimuth_deg", "inverter_efficiency", "source", "tracking")},
            "units": {"power": "MW", "energy": "MWh", "capacity": "MWp", "capacity_factor": "1 (fraction of nominal power)", "time": "UTC, hour start, ISO 8601 Z"},
            "series": series, "unavailable": {"modes": um, "series": us, "causes": self.causes()},
            "coverage": {"available": self.covered is not None, "fully_covered_cells": int(self.covered.sum()) if self.covered is not None else None,
                         "cells": int(self.lats.size * self.lons.size), "threshold": (self.coverage.get("threshold"))},
            "time_convention": {"used": conv.get("convention_used"), "detected": conv.get("detected")} if conv else None,
            "statements": {
                "calibration": "No national calibration and no curtailment correction is applied. Site capacity is nominal MWp; MW = capacity factor x MWp.",
                "resolution": f"Cells are {g['res_deg']} degrees: about {round(ns)} km north-south by {round(ew)} km east-west at {mid:.0f} N. The result is the cell-average "
                              "irradiance, not a micro-sited estimate.",
                "panel": "Fixed south-facing panel (azimuth 180), 40 degree tilt, atlite crystalline-silicon (CSi) model with inverter efficiency 0.90.",
                "hybrid": "Hybrid uses SARAH-3 direct and diffuse irradiance where at least 99.9 % of the cell area has valid satellite data in that hour, otherwise ERA5.",
                "northern_fallback": "SARAH-3 does not cover the north / north-east of Finland (no data north of 65 N, low viewing angle): those cells always use ERA5.",
                "night": "Night hours are exactly 0 MW.",
                "scope": "Fingrid 248 is a forecast, Fingrid 362 AV06 is small-scale surplus only; neither is measured total generation.",
            },
            "licences": self._licences(),
        }

    def grid_response(self) -> dict[str, Any]:
        return {**self.label_block(), "grid": self.grid["grid"], "cells": self.grid["cells"]}

    def coverage_response(self) -> dict[str, Any]:
        out: dict[str, Any] = {**self.label_block(), "available": self.covered is not None}
        if self.covered is None:
            out["reason"] = self.unavailable_modes.get("hybrid") or "SARAH-3 coverage not available"
            return out
        cells = self.grid["cells"]
        out |= {"threshold": self.coverage.get("threshold"), "summary": self.coverage.get("summary"), "pixel_map": self.coverage.get("pixel_map"),
                "cells": {"n_lat": int(self.lats.size), "n_lon": int(self.lons.size), "order": "row-major, latitude ascending then longitude ascending",
                          "satellite_covered": cells["satellite_covered"], "satellite_coverage_fraction": cells["satellite_coverage_fraction"],
                          "domain_fraction": cells.get("domain_fraction")},
                "legend": {"satellite_covered": "static: >= 99.9 % of the cell area has satellite data on at least one valid daytime hour per source pixel",
                           "satellite_coverage_fraction": "share of daytime hours in the window that used the satellite"}}
        return out

    # --- helpers -----------------------------------------------------------------------------------------------
    def locate(self, lat: float, lon: float) -> "tuple[int, int]":
        res = float(self.grid["grid"]["res_deg"])
        half = res / 2
        lo_lat, hi_lat, lo_lon, hi_lon = self.lats[0] - half, self.lats[-1] + half, self.lons[0] - half, self.lons[-1] + half
        if not (lo_lat <= lat <= hi_lat and lo_lon <= lon <= hi_lon):
            raise ApiError(422, "outside_grid", f"Location ({lat:g}, {lon:g}) is outside the solar grid (latitude {lo_lat:g}..{hi_lat:g}, longitude {lo_lon:g}..{hi_lon:g}).", "lat")
        if not in_finland(lat, lon):
            raise ApiError(422, "outside_finland", f"Location ({lat:g}, {lon:g}) is not in Finland (more than {REGION_TOLERANCE_DEG} degrees outside the coarse outline).", "lat")
        i = min(int(math.floor((lat - lo_lat) / res + 1e-9)), self.lats.size - 1)
        j = min(int(math.floor((lon - lo_lon) / res + 1e-9)), self.lons.size - 1)
        return i, j

    def _range(self, start: "str | None", end: "str | None") -> "tuple[int, int]":
        n = len(self.hours)
        first, last = self.hours[0], self.hours[-1]
        s_dt, e_dt = first, last
        if start not in (None, ""):
            s_dt, s_date = parse_utc(start, "start", is_end=False)
            if s_date and s_dt.date() == first.date():
                s_dt = max(s_dt, first)
        if end not in (None, ""):
            e_dt, e_date = parse_utc(end, "end", is_end=True)
            if e_date and e_dt.date() == last.date():
                e_dt = min(e_dt, last)
        for name, dt in (("start", s_dt), ("end", e_dt)):
            if dt < first or dt > last:
                raise ApiError(422, "date_out_of_range", f"'{name}' {iso(dt)} is outside the available range {self.hour_iso[0]} .. {self.hour_iso[-1]}.", name)
        if e_dt < s_dt:
            raise ApiError(422, "end_before_start", "'end' must not be before 'start'.", "end")
        return int((s_dt - first).total_seconds() // 3600), min(int((e_dt - first).total_seconds() // 3600), n - 1)

    def _mode(self, raw: "str | None") -> str:
        mode = raw if raw not in (None, "") else self.default_mode
        if mode is None:
            raise ApiError(422, "mode_unavailable", "No irradiance mode is available: " + "; ".join(dict.fromkeys(self.unavailable_modes.values())), "mode")
        if mode not in MODES:
            raise ApiError(422, "unknown_mode", f"Unknown mode {str(mode)[:40]!r}. Use one of: {', '.join(MODES)}.", "mode")
        if mode not in self.cf:
            raise ApiError(422, "mode_unavailable", f"Mode '{mode}' is not available: {self.unavailable_modes[mode]}", "mode")
        return mode

    # --- /solar/simulate ---------------------------------------------------------------------------------------
    def simulate(self, query: Mapping[str, list[str]]) -> dict[str, Any]:
        q = _single(query, {"lat", "lon", "start", "end", "capacity_mwp", "mode"})
        lat, lon = parse_float(q.get("lat"), "lat"), parse_float(q.get("lon"), "lon")
        cap = parse_float(q.get("capacity_mwp"), "capacity_mwp", required=False, default=CAPACITY_DEFAULT_MWP)
        if not CAPACITY_MIN_MWP <= cap <= CAPACITY_MAX_MWP:
            raise ApiError(422, "capacity_out_of_range", f"'capacity_mwp' must be between {CAPACITY_MIN_MWP:g} and {CAPACITY_MAX_MWP:g} MWp.", "capacity_mwp")
        mode = self._mode(q.get("mode"))
        i, j = self.locate(lat, lon)
        t0, t1 = self._range(q.get("start"), q.get("end"))
        cf = np.round(self.cf[mode][t0:t1 + 1, i, j].astype(np.float64) / float(self.bins["files"][f"cf_{mode}.bin"]["scale"]), 6)
        cf = np.clip(cf, 0.0, 1.0)
        mw = np.round(cf * cap, 6)                                   # MW = CF x MWp (1 W resolution)
        mwh = np.round(mw * 1.0, 6)                                  # MWh = MW x 1 h
        total = round(float(mwh.sum()), 6)
        hours = int(mw.size)
        codes_map = self.bins["files"].get("source_hybrid.bin", {}).get("codes", {})
        if mode == "hybrid" and self.source is not None:
            codes = self.source[t0:t1 + 1, i, j]
            names = [codes_map.get(str(int(c)), f"code_{int(c)}") for c in codes]
            src = ["satellite" if n == "satellite" else "era5_fallback" for n in names]
            reasons = [None if n == "satellite" else (n[len(FALLBACK_PREFIX):] if n.startswith(FALLBACK_PREFIX) else n) for n in names]
        else:
            src, reasons = ["era5"] * hours, [None] * hours
        counts: dict[str, int] = {}
        for r in reasons:
            if r:
                counts[r] = counts.get(r, 0) + 1
        sat_h, fb_h = src.count("satellite"), src.count("era5_fallback")
        res = float(self.grid["grid"]["res_deg"])
        ns, ew = cell_size_km(float(self.lats[i]), res)
        covered = None if self.covered is None else bool(self.covered[i, j])
        cov_frac = None if self.coverage_fraction is None or not np.isfinite(self.coverage_fraction[i, j]) else round(float(self.coverage_fraction[i, j]), 4)
        if mode == "era5":
            expl = "ERA5-only mode: irradiance comes from ERA5 for every hour."
        elif covered:
            expl = "This cell is fully covered by SARAH-3: the hybrid mode uses satellite direct and diffuse irradiance except in hours listed as ERA5 fallback."
        else:
            main = max(counts, key=counts.get) if counts else "no_coverage"
            expl = ("Northern / low-coverage fallback: " + EXPLANATIONS.get(main, main) + ". The hybrid mode therefore uses ERA5 irradiance in these hours "
                    "(irradiance_source = era5_fallback).")
        half = res / 2
        return {
            **self.label_block(),
            "request": {"lat": lat, "lon": lon, "start": self.hour_iso[t0], "end": self.hour_iso[t1], "capacity_mwp": cap, "mode": mode},
            "cell": {"lat_index": i, "lon_index": j, "center_lat": float(self.lats[i]), "center_lon": float(self.lons[j]), "resolution_deg": res,
                     "bounds": {"lat": [float(self.lats[i]) - half, float(self.lats[i]) + half], "lon": [float(self.lons[j]) - half, float(self.lons[j]) + half]},
                     "extent_km": {"north_south": round(ns, 1), "east_west": round(ew, 1)}, "satellite_covered": covered, "satellite_coverage_fraction": cov_frac,
                     "source": "SYNTHETIC fixture" if self.synthetic else "ERA5 reanalysis" + (" + CM SAF SARAH-3" if mode == "hybrid" else "") + " (cell mean)"},
            "mode": {"name": mode, "title": MODE_TITLES[mode], "explanation": expl},
            "panel": {k: self.manifest["panel"].get(k) for k in ("panel", "slope_deg", "azimuth_deg", "inverter_efficiency")},
            "units": {"time": "UTC", "power_mw": "MW", "capacity_factor": "1", "energy_mwh": "MWh (= MW x 1 h)", "capacity": "MWp"},
            "calibration": "none: no national calibration or curtailment correction applied",
            "licences": self._licences(),
            "summary": {"hours": hours, "total_mwh": total, "mean_mw": round(float(mw.mean()), 6), "peak_mw": float(mw.max()),
                        "mean_capacity_factor": round(total / (cap * hours), 6), "full_load_hours": round(total / cap, 3), "zero_output_hours": int((mw == 0).sum()),
                        "satellite_hours": sat_h, "fallback_hours": fb_h, "era5_hours": src.count("era5"), "fallback_reasons": counts,
                        "fallback_explanations": {r: EXPLANATIONS.get(r, r) for r in counts}},
            "data": {"time_utc": self.hour_iso[t0:t1 + 1], "power_mw": mw.tolist(), "capacity_factor": cf.tolist(), "energy_mwh": mwh.tolist(),
                     "irradiance_source": src, "fallback_reason": reasons},
        }

    # --- /solar/national -----------------------------------------------------------------------------------------
    def national_series(self, query: Mapping[str, list[str]]) -> dict[str, Any]:
        q = _single(query, {"start", "end"})
        t0, t1 = self._range(q.get("start"), q.get("end"))
        times = self.hour_iso[t0:t1 + 1]
        hours = len(times)
        out: dict[str, Any] = {}
        us = self.unavailable_series
        for key, s in self.national["series"].items():
            rec = {kk: s[kk] for kk in ("title", "unit", "native_unit", "scope", "available")}
            rec["reason"] = None if s["available"] else us[key]
            rec["reason_short"] = None if s["available"] else cause_of(us[key])[1]
            if s["available"]:
                vals = s["values"][t0:t1 + 1]
                have = [v for v in vals if v is not None]
                rec["values"] = vals
                stats: dict[str, Any] = {"hours": hours, "hours_with_value": len(have), "hours_missing": hours - len(have), "coverage_pct": round(100.0 * len(have) / hours, 2),
                                         "mean": round(sum(have) / len(have), 4) if have else None, "peak": max(have) if have else None, "min": min(have) if have else None}
                if "capacity" not in key:
                    stats["energy_gwh"] = round(sum(have) / 1000.0, 6) if have else None
                rec["summary"] = stats
                if "values_native_kwh" in s:
                    rec["values_native_kwh"] = s["values_native_kwh"][t0:t1 + 1]
            out[key] = rec
        return {**self.label_block(), "request": {"start": times[0], "end": times[-1]}, "units": {"time": "UTC", "power": "MW", "energy": "GWh = sum(MW x 1 h) / 1000"},
                "calibration": "none: model series are uncalibrated; Fingrid series are reference data with the scope stated per series",
                "unavailable": us, "causes": self.causes(), "series": out, "data": {"time_utc": times}, "licences": self._licences()}


# --- CSV ----------------------------------------------------------------------------------------------------------------------
def _licence_lines(res: Mapping[str, Any]) -> "list[str]":
    lic = res.get("licences") or {}
    out = []
    for key in ("cm_saf_sarah3", "era5", "fingrid", "statistics_finland", "suomen_uusiutuvat"):
        v = lic.get(key)
        if isinstance(v, dict) and v.get("attribution"):
            out.append(f"source_{key}: {v['attribution']} [{v.get('name')}; {v.get('status')}]")
    return out


def _flag_lines(res: Mapping[str, Any]) -> "list[str]":
    out = []
    if res["demo"] or res["synthetic"]:
        out.append(f"*** {res['notice']} ***")
    out += [f"label: {res['label']}", f"model_version: {res['model_version']}", f"run_mode: {res['run_mode']}",
            f"synthetic: {str(res['synthetic']).lower()}", f"demo: {str(res['demo']).lower()}", f"partial_run: {str(res['partial']).lower()}"]
    out.extend(f"{k}: {res.get(k) or 'not recorded'}" for k in ("dataset_id", "version_id", "href", "provenance_status"))
    return out


def solar_simulate_csv(res: Mapping[str, Any]) -> str:
    r, c, d, s, p = res["request"], res["cell"], res["data"], res["summary"], res["panel"]
    lines = _flag_lines(res) + [
        "dataset: single-cell solar PV simulation (hourly)", f"mode: {r['mode']} ({res['mode']['title']})", f"mode_note: {res['mode']['explanation']}",
        f"panel: {p['panel']} (atlite), tilt {p['slope_deg']} deg, azimuth {p['azimuth_deg']} deg (south), inverter efficiency {p['inverter_efficiency']}",
        f"capacity_mwp: {r['capacity_mwp']}", f"cell_index_lat_lon: {c['lat_index']},{c['lon_index']}", f"cell_center_lat_deg_n: {c['center_lat']}",
        f"cell_center_lon_deg_e: {c['center_lon']}", f"grid_resolution_deg: {c['resolution_deg']}",
        f"cell_extent_km_ns_ew: {c['extent_km']['north_south']} x {c['extent_km']['east_west']}", f"cell_satellite_covered: {str(c['satellite_covered']).lower()}",
        f"cell_satellite_coverage_fraction_daytime: {c['satellite_coverage_fraction']}", f"cell_source: {c['source']}",
        f"requested_lat_lon: {r['lat']},{r['lon']}", f"period_utc: {r['start']} .. {r['end']}", f"total_energy_mwh: {s['total_mwh']}",
        f"satellite_hours: {s['satellite_hours']}", f"fallback_hours: {s['fallback_hours']}", f"fallback_reasons: {json.dumps(s['fallback_reasons'], sort_keys=True)}",
        f"calibration: {res['calibration']}",
        "units: time_utc=ISO 8601 UTC hour start; power_mw=MW = capacity_factor x capacity_mwp; capacity_factor=fraction (1); energy_mwh=MW x 1 h",
        "caveat: cell-average irradiance at 0.25 degree resolution, fixed 40 deg south-facing panel, no national calibration, no curtailment",
    ] + _licence_lines(res)
    rows = ["time_utc,power_mw,capacity_factor,energy_mwh,irradiance_source,fallback_reason"]
    rows += [f"{t},{pw},{cf},{e},{src},{'' if why is None else why}" for t, pw, cf, e, src, why in zip(
        d["time_utc"], d["power_mw"], d["capacity_factor"], d["energy_mwh"], d["irradiance_source"], d["fallback_reason"])]
    return _csv_header(lines) + "\n".join(rows) + "\n"


CSV_COLUMNS = {"era5_only_mw": "era5_only_mw", "hybrid_mw": "hybrid_mw", "era5_only_popw_mw": "era5_only_population_weighted_mw",
               "hybrid_popw_mw": "hybrid_population_weighted_mw", "fingrid_248_forecast_mw": "fingrid_248_forecast_mw",
               "fingrid_362_av06_small_scale_surplus_mw": "fingrid_362_av06_surplus_mw", "fingrid_267_capacity_mw": "fingrid_267_capacity_mw"}


def solar_national_csv(res: Mapping[str, Any]) -> str:
    ser = res["series"]
    avail = [k for k, v in ser.items() if v["available"]]
    lines = _flag_lines(res) + ["dataset: national solar PV (hourly)", f"period_utc: {res['request']['start']} .. {res['request']['end']}", f"calibration: {res['calibration']}",
                               "units: time_utc=ISO 8601 UTC hour start; *_mw=MW; fingrid_362_av06_surplus_kwh=kWh per hour (native unit); empty = missing hour, never zero-filled"]
    for k, v in ser.items():
        lines.append(f"series {CSV_COLUMNS.get(k, k)}: {'available' if v['available'] else 'UNAVAILABLE: ' + str(v['reason'])}; native unit {v['native_unit']}; scope: {v['scope']}")
    lines += _licence_lines(res)
    cols = ["time_utc"] + [CSV_COLUMNS.get(k, k) for k in avail]
    data = [res["data"]["time_utc"]] + [ser[k]["values"] for k in avail]
    if "fingrid_362_av06_small_scale_surplus_mw" in avail and "values_native_kwh" in ser["fingrid_362_av06_small_scale_surplus_mw"]:
        cols.append("fingrid_362_av06_surplus_kwh")
        data.append(ser["fingrid_362_av06_small_scale_surplus_mw"]["values_native_kwh"])
    rows = [",".join(cols)]
    for k in range(len(data[0])):
        rows.append(",".join("" if col[k] is None else str(col[k]) for col in data))
    return _csv_header(lines) + "\n".join(rows) + "\n"
