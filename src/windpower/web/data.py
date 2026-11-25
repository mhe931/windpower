"""Site artifact loader and the pure computations behind the API (no HTTP here).

Consumes only the versioned artifact `site/v1` written by `export_site`. Wind-to-power conversion reuses
the shared helpers of `simulate` (log extrapolation to hub height, power-curve interpolation). No national
calibration and no curtailment correction is applied.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

from .. import config as C
from .. import simulate as S

API_VERSION = "v1"
SUPPORTED_SCHEMA = "1.0"
CAPACITY_MIN_MW, CAPACITY_MAX_MW, CAPACITY_DEFAULT_MW = 0.01, 1000.0, 3.0
REGION_TOLERANCE_DEG = 0.25
MAX_LISTED_MISSING_HOURS = 500
ISO_Z = "%Y-%m-%dT%H:%M:%SZ"

# Coarse mainland outline: the 37-vertex polygon hard-coded in the pipeline notebook's map cell (lon, lat).
MAINLAND = [(21.4, 60.7), (22.3, 60.4), (22.9, 59.8), (24.6, 60.1), (26.2, 60.4), (28.1, 60.5), (31.1, 62.3), (31.5, 62.9),
            (30.0, 63.5), (30.4, 64.2), (29.5, 65.0), (30.2, 65.8), (29.0, 67.0), (29.9, 67.7), (28.4, 68.3), (28.5, 69.0),
            (29.0, 69.8), (27.7, 70.1), (26.2, 69.8), (25.7, 69.0), (24.7, 68.6), (23.7, 68.9), (22.3, 68.8), (21.3, 69.3),
            (20.7, 69.1), (23.6, 67.9), (23.6, 66.3), (24.0, 66.0), (25.3, 65.5), (25.4, 65.1), (24.7, 64.9), (23.4, 64.3),
            (22.4, 63.8), (21.5, 63.1), (21.1, 62.6), (21.5, 61.7), (21.4, 60.7)]
# Hand-built, very rough outline of the Aland archipelago's main islands (not in the notebook).
ALAND = [(19.5, 60.1), (20.0, 59.9), (21.0, 60.0), (21.2, 60.3), (20.6, 60.5), (19.8, 60.4), (19.5, 60.1)]
OUTLINES = {"mainland": MAINLAND, "aland": ALAND}
_REGION = prep(Polygon(MAINLAND).union(Polygon(ALAND)).buffer(REGION_TOLERANCE_DEG))


class SiteDataError(RuntimeError):
    """The artifact is missing or inconsistent."""


class ApiError(Exception):
    """Request-level error mapped to a 4xx JSON response: {"error": {"code", "message", "param"}}."""

    def __init__(self, status: int, code: str, message: str, param: "str | None" = None):
        super().__init__(message)
        self.status, self.code, self.message, self.param = status, code, message, param

    def as_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "param": self.param}}


def in_finland(lat: float, lon: float) -> bool:
    """Inside the coarse outline or within 0.25 deg of it (so coastal cells are accepted)."""
    return bool(_REGION.covers(Point(lon, lat)))


def cell_size_km(lat: float, res_deg: float = C.GRID_RES_DEG) -> "tuple[float, float]":
    """(north-south, east-west) cell extent in km at `lat`."""
    return res_deg * 111.2, res_deg * 111.32 * math.cos(math.radians(lat))


def iso(dt: datetime) -> str:
    return dt.strftime(ISO_Z)


# --- parameter parsing ----------------------------------------------------------------------------------
def parse_float(raw: "str | None", name: str, *, required: bool = True, default: "float | None" = None) -> float:
    if raw is None or raw.strip() == "":
        if required:
            raise ApiError(400, "missing_parameter", f"Parameter '{name}' is required.", name)
        return float(default)  # type: ignore[arg-type]
    try:
        value = float(raw)
    except ValueError:
        raise ApiError(400, "invalid_number", f"Parameter '{name}' must be a number, got {raw[:40]!r}.", name) from None
    if not math.isfinite(value):
        raise ApiError(400, "invalid_number", f"Parameter '{name}' must be a finite number.", name)
    return value


def parse_utc(raw: str, name: str, *, is_end: bool) -> "tuple[datetime, bool]":
    """'YYYY-MM-DD' or 'YYYY-MM-DDTHH[:MM[:SS]]' with optional trailing Z (UTC only). Returns (datetime, date_only)."""
    text = raw.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1]
    for fmt, date_only in (("%Y-%m-%d", True), ("%Y-%m-%dT%H", False), ("%Y-%m-%dT%H:%M", False), ("%Y-%m-%dT%H:%M:%S", False)):
        try:
            dt = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if not date_only and (dt.minute or dt.second):
            raise ApiError(400, "bad_date", f"'{name}' must be on a whole UTC hour (hourly resolution), got {raw!r}.", name)
        if date_only and is_end:
            dt += timedelta(hours=23)
        return dt, date_only
    raise ApiError(400, "bad_date", f"'{name}' is not a valid UTC date: use YYYY-MM-DD or YYYY-MM-DDTHH:00Z, got {raw[:40]!r}.", name)


def _single(query: Mapping[str, list[str]], allowed: "set[str]") -> dict[str, "str | None"]:
    for k, v in query.items():
        if k not in allowed:
            raise ApiError(400, "unknown_parameter", f"Unknown parameter '{k}'. Allowed: {', '.join(sorted(allowed))}.", k)
        if len(v) > 1:
            raise ApiError(400, "duplicate_parameter", f"Parameter '{k}' was given more than once.", k)
    return {k: (v[0] if v else None) for k, v in query.items()}


@dataclass
class SiteData:
    site_dir: Path
    manifest: dict[str, Any]
    grid: dict[str, Any]
    archetypes: dict[str, Any]
    wind_meta: dict[str, Any]
    national: dict[str, Any]
    wind: np.ndarray                      # float16 [time, lat, lon]
    roughness: np.ndarray                 # float16 [lat, lon]
    lats: np.ndarray
    lons: np.ndarray
    hours: list[datetime]
    hour_iso: list[str]
    arch_by_name: dict[str, dict[str, Any]] = field(default_factory=dict)
    curves: dict[str, dict[str, Any]] = field(default_factory=dict)

    # --- loading --------------------------------------------------------------------------------------
    @classmethod
    def load(cls, site_dir: "str | Path") -> "SiteData":
        d = Path(site_dir)

        def read(name: str) -> Any:
            p = d / name
            if not p.is_file():
                raise SiteDataError(f"site artifact file missing: {p}")
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise SiteDataError(f"{p} is not valid JSON: {exc}") from None

        manifest, grid, arch, wmeta, nat = (read(n) for n in ("manifest.json", "grid.json", "archetypes.json", "wind_effective.meta.json", "national_hourly.json"))
        for name, obj in (("manifest.json", manifest), ("grid.json", grid), ("archetypes.json", arch), ("wind_effective.meta.json", wmeta), ("national_hourly.json", nat)):
            if obj.get("artifact_schema_version") != SUPPORTED_SCHEMA:
                raise SiteDataError(f"{name}: unsupported artifact_schema_version {obj.get('artifact_schema_version')!r} (need {SUPPORTED_SCHEMA})")
        spec = wmeta["arrays"]["wind_effective"]
        shape = tuple(spec["shape"])
        wpath, zpath = d / spec["file"], d / wmeta["arrays"]["roughness_mean"]["file"]
        if not (wpath.is_file() and zpath.is_file()):
            raise SiteDataError("wind_effective.bin or roughness_mean.bin missing")
        raw, zraw = wpath.read_bytes(), zpath.read_bytes()
        if spec["dtype"] != "float16" or spec["byte_order"] != "little" or spec["order"] != "C" or spec["axes"] != ["time", "lat", "lon"]:
            raise SiteDataError(f"unsupported wind array layout: {spec}")
        if len(raw) != int(np.prod(shape)) * 2:
            raise SiteDataError(f"wind_effective.bin has {len(raw)} bytes, expected {int(np.prod(shape)) * 2} for shape {shape}")
        wind = np.frombuffer(raw, dtype="<f2").reshape(shape)
        rshape = tuple(wmeta["arrays"]["roughness_mean"]["shape"])
        if len(zraw) != int(np.prod(rshape)) * 2 or rshape != shape[1:]:
            raise SiteDataError("roughness_mean.bin does not match the wind grid")
        rough = np.frombuffer(zraw, dtype="<f2").reshape(rshape)
        lats, lons = np.asarray(grid["lats"], float), np.asarray(grid["lons"], float)
        if (lats.size, lons.size) != shape[1:] or not (np.all(np.diff(lats) > 0) and np.all(np.diff(lons) > 0)):
            raise SiteDataError("grid.json axes are inconsistent with the wind array or not ascending")
        first = datetime.strptime(wmeta["time"]["first_utc"], ISO_Z)
        hours = [first + timedelta(hours=k) for k in range(shape[0])]
        hour_iso = [iso(h) for h in hours]
        if hour_iso[-1] != wmeta["time"]["last_utc"] or nat["utc_hours"] != hour_iso:
            raise SiteDataError("time axis of national_hourly.json / wind_effective.meta.json disagree")
        site = cls(d, manifest, grid, arch, wmeta, nat, wind, rough, lats, lons, hours, hour_iso)
        for a in arch["archetypes"]:
            pc = a["power_curve"]
            if len(pc["v_ms"]) != 61 or len(pc["power_per_unit"]) != 61:
                raise SiteDataError(f"archetype {a['name']} lacks a 61-point curve")
            site.arch_by_name[a["name"]] = a
            site.curves[a["name"]] = {"V": np.asarray(pc["v_ms"], float), "POW": np.asarray(pc["power_per_unit"], float), "P": 1.0}
        if not site.arch_by_name:
            raise SiteDataError("archetypes.json is empty")
        return site

    # --- labels ----------------------------------------------------------------------------------------
    @property
    def synthetic(self) -> bool:
        return bool(self.manifest["synthetic"])

    @property
    def demo(self) -> bool:
        return bool(self.manifest["demo"])

    @property
    def notice(self) -> str:
        if self.demo:
            return "DEMO - SYNTHETIC DATA. These numbers are generated test data, not real weather, turbines or measurements."
        if self.synthetic:
            return "SYNTHETIC DATA (smoke-test fixture). These numbers are not real weather, turbines or measurements."
        return "Retrospective ERA5 simulation. Uncalibrated: no national bias or curtailment correction is applied."

    def label_block(self) -> dict[str, Any]:
        m = self.manifest
        return {"model_version": m["model_version"], "run_mode": m["run_mode"], "synthetic": self.synthetic, "demo": self.demo,
                "label": m["label"], "notice": self.notice, "artifact_schema_version": m["artifact_schema_version"]}

    @property
    def default_archetype(self) -> str:
        return max(self.arch_by_name.values(), key=lambda a: a["installed_mw"])["name"]

    # --- /meta -------------------------------------------------------------------------------------------
    def meta(self) -> dict[str, Any]:
        n_lat, n_lon = len(self.lats), len(self.lons)
        mid = float(self.lats[n_lat // 2])
        ns, ew = cell_size_km(mid)
        cov = self.manifest["data_coverage"]
        return {
            **self.label_block(), "api_version": API_VERSION,
            "time_range": {"first_utc": self.hour_iso[0], "last_utc": self.hour_iso[-1], "hours": len(self.hours), "step": "1h"},
            "grid": {"n_lat": n_lat, "n_lon": n_lon, "resolution_deg": self.grid["cell_size_deg"],
                     "lat_range": [float(self.lats[0]), float(self.lats[-1])], "lon_range": [float(self.lons[0]), float(self.lons[-1])],
                     "index_order": self.grid["index_order"], "cell_km_at_mid_latitude": {"north_south": round(ns, 1), "east_west": round(ew, 1), "latitude": mid}},
            "region": {"outlines_lonlat": {k: [list(p) for p in v] for k, v in OUTLINES.items()}, "tolerance_deg": REGION_TOLERANCE_DEG,
                       "note": "Coarse outline (mainland from the pipeline notebook, Aland hand-built) used for map drawing and to reject locations more than 0.25 degrees outside it."},
            "limits": {"capacity_mw": {"min": CAPACITY_MIN_MW, "max": CAPACITY_MAX_MW, "default": CAPACITY_DEFAULT_MW}},
            "defaults": {"archetype": self.default_archetype},
            "units": {"power": "MW", "energy": "MWh", "capacity_factor": "1 (MW / installed MW)", "time": "UTC, hour start, ISO 8601 Z"},
            "coverage": {"era5_hours": cov["era5_hours"], "observed_hours": cov["observed_hours"]},
            "statements": {
                "calibration": "No national calibration and no curtailment correction is applied to site simulations.",
                "resolution": "ERA5 cells are 0.25 degrees: about 28 km north-south by 14 km east-west at 60 N. The result is the cell-average wind, not a micro-sited estimate.",
                "wind": "Hourly ERA5 100 m wind speed adjusted to standard air density, extrapolated to hub height with the logarithmic profile (time-mean cell roughness) and converted with the archetype power curve.",
                "observed": "Observed Fingrid output (national view only) may be curtailed; the theoretical series is uncalibrated.",
                "real_data": ("This server shows SYNTHETIC data; real ERA5/fleet/Fingrid data have not been integrated." if self.synthetic else
                              "Built from the real-data artifact; see the manifest for provenance and licence status."),
            },
            "attribution": self.manifest.get("attribution"),
            "licences": {k: v.get("name") for k, v in (self.manifest.get("licences") or {}).items() if isinstance(v, dict)},
        }

    def archetype_list(self) -> dict[str, Any]:
        out = []
        for a in self.arch_by_name.values():
            c = a["power_curve"]
            out.append({"name": a["name"], "turbines": a["turbines"], "fleet_installed_mw": a["installed_mw"], "mean_rated_mw": a["mean_rated_mw"],
                        "hub_height_m": round(a["mean_hub_height_m"], 1), "specific_power_wm2": round(a["mean_specific_power_wm2"], 1),
                        "power_curve": {"v_ms": c["v_ms"], "power_per_unit": c["power_per_unit"]}})
        return {**self.label_block(), "default": self.default_archetype, "curve_points": 61, "units": {"v_ms": "m s-1", "power_per_unit": "1 (fraction of rated power)"},
                "archetypes": out}

    # --- helpers ---------------------------------------------------------------------------------------------
    def locate(self, lat: float, lon: float) -> "tuple[int, int]":
        res = float(self.grid["cell_size_deg"])
        half = res / 2 + 1e-9
        i = int(np.abs(self.lats - lat).argmin())
        j = int(np.abs(self.lons - lon).argmin())
        if abs(self.lats[i] - lat) > half or abs(self.lons[j] - lon) > half:
            raise ApiError(422, "outside_grid", f"Location ({lat:g}, {lon:g}) is outside the data grid (latitude {self.lats[0]:g}..{self.lats[-1]:g}, "
                           f"longitude {self.lons[0]:g}..{self.lons[-1]:g}).", "lat")
        if not in_finland(lat, lon):
            raise ApiError(422, "outside_finland", f"Location ({lat:g}, {lon:g}) is not in Finland (more than {REGION_TOLERANCE_DEG} degrees outside the coarse outline).", "lat")
        if not self.grid["supported"][i][j]:
            raise ApiError(422, "unsupported_cell", "This grid cell is not supported by the data artifact.", "lat")
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
        t0 = int((s_dt - first).total_seconds() // 3600)
        t1 = int((e_dt - first).total_seconds() // 3600)
        return t0, min(t1, n - 1)

    # --- /simulate ------------------------------------------------------------------------------------------------
    def simulate(self, query: Mapping[str, list[str]]) -> dict[str, Any]:
        q = _single(query, {"lat", "lon", "start", "end", "archetype", "capacity_mw"})
        lat, lon = parse_float(q.get("lat"), "lat"), parse_float(q.get("lon"), "lon")
        cap = parse_float(q.get("capacity_mw"), "capacity_mw", required=False, default=CAPACITY_DEFAULT_MW)
        if not CAPACITY_MIN_MW <= cap <= CAPACITY_MAX_MW:
            raise ApiError(422, "capacity_out_of_range", f"'capacity_mw' must be between {CAPACITY_MIN_MW:g} and {CAPACITY_MAX_MW:g} MW.", "capacity_mw")
        name = q.get("archetype") or self.default_archetype
        if name not in self.arch_by_name:
            raise ApiError(422, "unknown_archetype", f"Unknown archetype {name[:60]!r}. Available: {', '.join(self.arch_by_name)}.", "archetype")
        i, j = self.locate(lat, lon)
        t0, t1 = self._range(q.get("start"), q.get("end"))
        arch = self.arch_by_name[name]
        v = self.wind[t0:t1 + 1, i, j].astype(np.float64)
        z0 = float(self.roughness[i, j])
        v_hub = S.hub_height_speed(v, z0, arch["mean_hub_height_m"])
        pu = np.clip(S.power_from_speed(self.curves[name], v_hub), 0.0, 1.0)
        mw = np.round(pu * cap, 6)                           # 1 W resolution: fine even for the 0.01 MW minimum capacity
        cf = np.clip(np.round(pu, 6), 0.0, 1.0)              # = MW / installed MW, independent of the MW rounding
        hours = int(mw.size)
        mwh = np.round(S.energy_mwh(mw), 6)                  # MW x 1 h
        total = round(float(mwh.sum()), 6)
        ns, ew = cell_size_km(float(self.lats[i]))
        res = float(self.grid["cell_size_deg"])
        return {
            **self.label_block(),
            "request": {"lat": lat, "lon": lon, "start": self.hour_iso[t0], "end": self.hour_iso[t1], "archetype": name, "capacity_mw": cap},
            "cell": {"lat_index": i, "lon_index": j, "center_lat": float(self.lats[i]), "center_lon": float(self.lons[j]), "resolution_deg": res,
                     "extent_km": {"north_south": round(ns, 1), "east_west": round(ew, 1)},
                     "source": "SYNTHETIC ERA5-like fixture" if self.synthetic else "ERA5 hourly reanalysis (cell-average)",
                     "mean_roughness_m": round(z0, 5)},
            "archetype": {"name": name, "hub_height_m": round(arch["mean_hub_height_m"], 1), "specific_power_wm2": round(arch["mean_specific_power_wm2"], 1),
                          "mean_rated_mw": arch["mean_rated_mw"]},
            "units": {"time": "UTC", "power_mw": "MW", "capacity_factor": "1", "energy_mwh": "MWh (= MW x 1 h)"},
            "calibration": "none: no national calibration or curtailment correction applied",
            "summary": {"hours": hours, "total_mwh": total, "mean_mw": round(float(mw.mean()), 3), "peak_mw": float(mw.max()),
                        "mean_capacity_factor": round(total / (cap * hours), 6), "full_load_hours": round(total / cap, 2),
                        "zero_output_hours": int((mw == 0).sum())},
            "data": {"time_utc": self.hour_iso[t0:t1 + 1], "power_mw": mw.tolist(), "capacity_factor": cf.tolist(), "energy_mwh": mwh.tolist()},
        }

    # --- /national --------------------------------------------------------------------------------------------------
    def national_series(self, query: Mapping[str, list[str]]) -> dict[str, Any]:
        q = _single(query, {"start", "end"})
        t0, t1 = self._range(q.get("start"), q.get("end"))
        n = self.national
        theo = n["theoretical_mw"][t0:t1 + 1]
        obs = n["observed_mw"][t0:t1 + 1]
        avail = [v is not None for v in obs]
        hours = len(theo)
        have = [v for v in obs if v is not None]
        missing_idx = [k for k, a in enumerate(avail) if not a]
        ranges: list[list[int]] = []
        run: "list[int] | None" = None
        for k in missing_idx:
            if run and k == run[1] + 1:
                run[1] = k
            else:
                run = [k, k]
                ranges.append(run)
        times = self.hour_iso[t0:t1 + 1]
        both = [(t, o) for t, o in zip(theo, obs) if o is not None]
        return {
            **self.label_block(),
            "request": {"start": times[0], "end": times[-1]},
            "units": {"time": "UTC", "theoretical_mw": "MW", "observed_mw": "MW"},
            "series": {"theoretical": "simulation of the whole fleet, uncalibrated (no bias or curtailment correction)",
                       "observed": "Fingrid-reported wind output; may be curtailed"},
            "installed_mw": n["installed_mw"],
            "summary": {
                "hours": hours, "observed_hours": len(have), "observed_pct": round(100.0 * len(have) / hours, 2), "missing_hours": len(missing_idx),
                "theoretical_mean_mw": round(sum(theo) / hours, 3), "theoretical_peak_mw": max(theo), "theoretical_total_mwh": round(sum(theo), 3),
                "observed_mean_mw": round(sum(have) / len(have), 3) if have else None, "observed_peak_mw": max(have) if have else None,
                "observed_total_mwh": round(sum(have), 3) if have else None,
                "theoretical_mean_mw_on_observed_hours": round(sum(t for t, _ in both) / len(both), 3) if both else None,
            },
            "missing_ranges": [{"start": times[a], "end": times[b], "hours": b - a + 1} for a, b in ranges],
            "missing_hours_utc": [times[k] for k in missing_idx[:MAX_LISTED_MISSING_HOURS]],
            "missing_hours_truncated": len(missing_idx) > MAX_LISTED_MISSING_HOURS,
            "data": {"time_utc": times, "theoretical_mw": theo, "observed_mw": obs, "observed_available": [int(a) for a in avail]},
        }


# --- CSV ------------------------------------------------------------------------------------------------------------------
def _csv_header(lines: "list[str]") -> str:
    return "".join(f"# {ln}\n" for ln in lines)


def _common_header(res: Mapping[str, Any]) -> "list[str]":
    out = []
    if res["demo"] or res["synthetic"]:
        out.append(f"*** {res['notice']} ***")
    out += [f"label: {res['label']}", f"model_version: {res['model_version']}", f"run_mode: {res['run_mode']}",
            f"synthetic: {str(res['synthetic']).lower()}", f"demo: {str(res['demo']).lower()}"]
    return out


def simulate_csv(res: Mapping[str, Any]) -> str:
    r, c, a, d = res["request"], res["cell"], res["archetype"], res["data"]
    lines = _common_header(res) + [
        "dataset: single-cell wind power simulation (hourly)",
        f"cell_index_lat_lon: {c['lat_index']},{c['lon_index']}", f"cell_center_lat_deg_n: {c['center_lat']}", f"cell_center_lon_deg_e: {c['center_lon']}",
        f"grid_resolution_deg: {c['resolution_deg']}", f"cell_source: {c['source']}",
        f"requested_lat_lon: {r['lat']},{r['lon']}", f"archetype: {a['name']}", f"hub_height_m: {a['hub_height_m']}",
        f"installed_capacity_mw: {r['capacity_mw']}", f"period_utc: {r['start']} .. {r['end']}",
        f"total_energy_mwh: {res['summary']['total_mwh']}", f"calibration: {res['calibration']}",
        "units: time_utc=ISO 8601 UTC hour start; power_mw=MW; capacity_factor=MW/installed MW (1); energy_mwh=MW x 1 h",
        "caveat: cell-average ERA5 wind at 0.25 degree resolution, not a micro-sited estimate",
    ]
    rows = ["time_utc,power_mw,capacity_factor,energy_mwh"]
    rows += [f"{t},{p},{f},{e}" for t, p, f, e in zip(d["time_utc"], d["power_mw"], d["capacity_factor"], d["energy_mwh"])]
    return _csv_header(lines) + "\n".join(rows) + "\n"


def national_csv(res: Mapping[str, Any]) -> str:
    d, s = res["data"], res["summary"]
    lines = _common_header(res) + [
        "dataset: national theoretical vs observed wind power (hourly)",
        f"period_utc: {res['request']['start']} .. {res['request']['end']}", f"installed_capacity_mw: {res['installed_mw']}",
        f"observed_hours: {s['observed_hours']} of {s['hours']} ({s['observed_pct']} %)",
        "theoretical_mw: whole-fleet simulation, UNCALIBRATED (no national bias or curtailment correction)",
        "observed_fingrid_mw: Fingrid-reported wind output, may be curtailed; empty = missing hour",
        "units: time_utc=ISO 8601 UTC hour start; theoretical_mw=MW; observed_fingrid_mw=MW",
    ]
    rows = ["time_utc,theoretical_mw,observed_fingrid_mw,observed_available"]
    rows += [f"{t},{a},{'' if b is None else b},{k}" for t, a, b, k in zip(d["time_utc"], d["theoretical_mw"], d["observed_mw"], d["observed_available"])]
    return _csv_header(lines) + "\n".join(rows) + "\n"
