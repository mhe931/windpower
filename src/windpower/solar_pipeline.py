"""Solar stage orchestration. Mirrors `pipeline.Run` conventions for the solar run modes.

Stages (each rebuilds what it needs from the raw caches under `<data>/cache/...`, memoised within a process):
  ingest-sarah         CM SAF SARAH-3 daily files -> hourly means, regrid onto the 44 x 53 ERA5 solar grid, coverage map
  ingest-era5-solar    ERA5 radiation + 2 m temperature (CDS download needs credentials; --era5-solar-path uses files in place)
  ingest-solar-public  plant list (Suomen Uusiutuvat) + Statistics Finland population grid and municipalities (explicit network)
  ingest-fingrid-solar Fingrid 267 / 248 / 362 (needs FINGRID_API_KEY unless a complete cache exists)
  solar-hourly         ERA5 hour-start fields and the SARAH-3 vs ERA5 irradiance comparison
  solar-pv             atlite CSi capacity-factor cubes (ERA5-only and hybrid)
  solar-national       capacity allocation, national MW / GWh, comparison with Fingrid
  solar-validate       hard checks, reconciliation table, manifest
  export-site-solar    site/solar/v1 artifact

Availability convention: real mode never falls back to synthetic data. A stage that cannot do ANY of its supportable work
because its own required input is missing raises `Unavailable` (CLI exit 2). A stage that can do part of its work completes
that part and records the rest as unavailable (exit 0, manifest status `partial_inputs_unavailable`). `solar-all` runs every
stage, records every gap, and uses the network only with `--refresh`. Synthetic modes (`solar-smoke`, `solar-demo`) generate
every input offline and must pass all hard checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from . import config as C
from . import era5 as E5
from . import era5_solar as ES
from . import fingrid as FG
from . import fingrid_solar as FS
from . import fixtures as F
from . import fixtures_solar as FP
from . import sarah as S
from . import sarah_regrid as R
from . import solar_national as N
from . import solar_public as SPUB
from . import solar_pv as P
from . import solar_validate as SV
from .export_site_solar import export_site_solar
from .io_utils import SYNTHETIC_STAMP, CredentialUnavailableError, atomic_write_json, atomic_write_with, display_path, iso_utc, sha256_file, utc_hour_range
from .pipeline import Unavailable
from .validate import CheckResult, failed_checks, results_frame

SOLAR_STAGES = ("ingest-sarah", "ingest-era5-solar", "ingest-solar-public", "ingest-fingrid-solar", "solar-hourly", "solar-pv", "solar-national",
                "solar-validate", "export-site-solar")
SOLAR_RUNS = ("solar-smoke", "solar-demo", "solar-all")
HOURLY_FILE = "sarah_hourly_era5grid.nc"
CELLS_FILE = "sarah_coverage_cells.csv"
PIXELS_FILE = "sarah_coverage_pixels.nc"
INVENTORY_FILE = "sarah_inventory.json"
ISO = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class SolarModeSpec:
    grid: C.GridSpec
    first_date: str
    n_days: int
    start: str
    end: str
    fixture: F.SarahFixtureSpec
    era5_zip: bool = False
    seed: int = 2026


SOLAR_MODE_SPECS: dict[str, SolarModeSpec] = {
    # crosses the July/August boundary: the last hour needs the 00:00 step of the next month's first file
    "solar-smoke": SolarModeSpec(C.SOLAR_SMOKE_GRID, "2025-07-30", 3, "2025-07-30", "2025-07-31",
                                 F.SarahFixtureSpec(void_steps=("2025-07-31T12:00",), bad_steps=("2025-07-30T10:30",))),
    "solar-demo": SolarModeSpec(C.FINLAND_SOLAR, "2025-07-28", 5, "2025-07-28", "2025-07-31",
                                F.SarahFixtureSpec(lat_min=59.4, lat_max=65.0, lon_min=19.0, lon_max=32.0, north_cutoff_lat=63.5,
                                                   patch=(61.0, 61.4, 24.0, 24.6), void_steps=("2025-07-29T09:00",), bad_steps=("2025-07-30T14:30",)),
                                era5_zip=True),
}


def parse_window(start: "str | None", end: "str | None") -> pd.DatetimeIndex:
    """Hour starts of the window: a date end is an inclusive whole UTC day; a datetime end is taken as given."""
    end_text = end or C.SOLAR_WINDOW_END
    s, e = pd.Timestamp(start or C.SOLAR_WINDOW_START), pd.Timestamp(end_text)
    if "T" not in end_text and " " not in end_text:
        e = e.normalize() + pd.Timedelta(hours=23)
    if e < s:
        raise ValueError(f"window end {e} is before start {s}")
    return utc_hour_range(s, e)


def _hash_files(path: "str | Path") -> list[dict[str, Any]]:
    p = Path(path)
    files = sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else [p]
    return [{"name": q.name, "bytes": q.stat().st_size, "sha256": sha256_file(q)} for q in files]


class SolarRun:
    def __init__(self, mode: str = "solar", data_dir: "str | Path | None" = None, *, sarah_sis_dir: "str | Path | None" = None,
                 sarah_sid_dir: "str | Path | None" = None, era5_solar_path: "str | Path | None" = None, era5_convention: "str | None" = None,
                 window_start: "str | None" = None, window_end: "str | None" = None, hash_inputs: bool = True, offline: bool = False,
                 refresh: bool = False, say: Callable[[str], None] = print, env: "Mapping[str, str] | None" = None):
        if mode not in C.SOLAR_RUN_MODES:
            raise ValueError(f"mode must be one of {C.SOLAR_RUN_MODES}")
        self.mode, self.say, self.hash_inputs, self.env = mode, say, hash_inputs, env
        self.data_dir = data_dir
        self.synthetic = C.is_synthetic_mode(mode)
        self.offline, self.refresh = True if self.synthetic else offline, refresh
        self.spec = SOLAR_MODE_SPECS.get(mode)
        self.grid = self.spec.grid if self.spec else C.FINLAND_SOLAR
        self.hours = parse_window(self.spec.start, self.spec.end) if self.spec else parse_window(window_start, window_end)
        self.cache, self.interim, self.out = C.cache_dir(mode, data_dir), C.interim_dir(mode, data_dir), C.output_dir(mode, data_dir)
        self.era5_cache, self.public_cache, self.fg_cache = self.cache / "era5", self.cache / "public", self.cache / "fingrid"
        self.explicit = {"SIS": sarah_sis_dir, "SID": sarah_sid_dir}
        self.era5_solar_path, self.era5_convention = era5_solar_path, era5_convention
        self._memo: dict[str, Any] = {}
        self.gaps: dict[str, str] = {}
        info_path = self.interim / "run_info.json"
        self.info: dict[str, Any] = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
        self.info["grid"] = {"name": self.grid.name, "n_lat": self.grid.n_lat, "n_lon": self.grid.n_lon, "res_deg": self.grid.res}

    # --- helpers ---------------------------------------------------------------------------------------------------------
    def _save_info(self, **sections: Any) -> None:
        self.info.update(sections)
        atomic_write_json(self.interim / "run_info.json", self.info)

    def _csv(self, df: pd.DataFrame, name: str, index: bool = False, label: "str | None" = None) -> None:
        atomic_write_with(self.out / name, lambda tmp: df.to_csv(tmp, index=index, index_label=label, date_format=ISO))

    def _json(self, obj: Any, name: str) -> None:
        atomic_write_json(self.out / name, obj)

    def _gap(self, key: str, reason: str) -> None:
        self.gaps[key] = reason

    def _network_ok(self, explicit: bool) -> bool:
        """Real-mode network use: explicit stages unless --offline; `solar-all` only with --refresh."""
        return (not self.offline) and (explicit or self.refresh)

    @property
    def last_step(self) -> pd.Timestamp:
        return self.hours[-1] + pd.Timedelta(hours=1)

    # --- SARAH-3 -----------------------------------------------------------------------------------------------------------
    def _synthetic_sarah_paths(self) -> "dict[str, Path]":
        return {"SIS": self.cache / "sarah" / "SYNTHETIC_SIS", "SID": self.cache / "sarah" / "SYNTHETIC_SID"}

    def sarah_dirs(self) -> "dict[str, Path]":
        """Input folders per kind. Synthetic modes (re)generate fixtures; real mode resolves flag > env > default or raises Unavailable."""
        if self.synthetic:
            assert self.spec is not None
            F.write_sarah_cache(self.cache / "sarah", self.spec.first_date, self.spec.n_days, self.spec.fixture)
            return self._synthetic_sarah_paths()
        out: dict[str, Path] = {}
        missing: list[dict[str, str]] = []
        for kind in ("SIS", "SID"):
            try:
                out[kind] = S.resolve_sarah_dir(kind, self.explicit[kind], self.env, self.data_dir)
            except S.SarahError as exc:
                missing.append({"input": f"SARAH-3 {kind} folder", "reason": exc.detail, "key": "sarah"})
        if missing:
            raise Unavailable(missing)
        return out

    def ingest_sarah(self) -> dict[str, Any]:
        dirs = self.sarah_dirs()
        try:
            return self._ingest_sarah(dirs)
        except S.SarahError as exc:
            if exc.check == "input_missing":
                raise Unavailable([{"input": "SARAH-3 files", "reason": exc.detail, "key": "sarah"}]) from exc
            raise

    def _ingest_sarah(self, dirs: "dict[str, Path]") -> dict[str, Any]:
        first_step, last_step = self.hours[0], self.last_step
        inv = {k: S.inventory(dirs[k], k, hash_files=self.hash_inputs and not self.synthetic) for k in ("SIS", "SID")}
        for k in ("SIS", "SID"):
            inv[k]["window_flagged_steps"] = S.flagged_steps_in_window(inv[k], first_step, last_step)
        sis = S.load_series(dirs["SIS"], "SIS", first_step, last_step)
        sid = S.load_series(dirs["SID"], "SID", first_step, last_step)
        missing_steps = int((sis.record_status == S.STATUS_NO_STEP).sum() + (sid.record_status == S.STATUS_NO_STEP).sum())
        hourly = S.hourly_means(sis, sid, self.hours)
        self.say(f"[ingest-sarah] {len(self.hours)} hourly steps from {len(sis.times)} half-hour steps "
                 f"({sis.times[0]:%Y-%m-%dT%H:%M} .. {sis.times[-1]:%Y-%m-%dT%H:%M} UTC), source grid {sis.lat.size}x{sis.lon.size}")
        grid = R.regrid_hourly(hourly, self.grid, C.SATELLITE_MIN_AREA_FRACTION)
        summary = R.coverage_summary(grid)
        summary |= {"first_hour_utc": iso_utc(self.hours[0]), "last_hour_utc": iso_utc(self.hours[-1]),
                    "expected_hours": len(self.hours), "missing_half_hour_steps": missing_steps,
                    "hours_with_flagged_or_missing_step": int((~hourly.hour_step_ok).sum()),
                    "source_valid_hourly_share": float(hourly.valid.mean()), "synthetic": self.synthetic}
        self._write_sarah_products(grid, inv, summary)
        self._memo["sarah_ongrid"] = grid
        self._save_info(sarah={"inventory": {k: {kk: vv for kk, vv in v.items() if kk != "sha256"} for k, v in inv.items()},
                               "sha256": {k: v.get("sha256") for k, v in inv.items()}, "coverage": summary,
                               "inputs": {k: str(v) for k, v in dirs.items()}, "files_read": {"SIS": sis.files, "SID": sid.files},
                               "source": "SYNTHETIC fixture" if self.synthetic else "CM SAF SARAH-3 ICDR (SIS, SID), read in place"})
        for k in ("SIS", "SID"):
            flagged = inv[k]["window_flagged_steps"]
            self.say(f"[ingest-sarah] {k}: {inv[k]['n_files']} files, {inv[k]['time']['first_utc']} .. {inv[k]['time']['last_utc']}, "
                     f"record_status {inv[k]['record_status']['counts']}, flagged steps in window: {[s['time_utc'] for s in flagged] or 'none'}")
        v = summary["valid_area_fraction"]
        self.say(f"[ingest-sarah] fully covered cells: {summary['fully_covered_cells']} of {summary['grid']['n_cells']} "
                 f"(report target {summary['target_fully_covered_cells']} of {summary['target_grid_cells']}); "
                 f"valid area fraction mean {v['mean']:.4f}, mean in covered cells {v['mean_in_fully_covered_cells']}, "
                 f">= threshold in {100 * v['share_cell_hours_ge_threshold']:.1f}% of cell-hours")
        self.say(f"[ingest-sarah] source codes: {summary['source_counts']}")
        self.say(f"[ingest-sarah] products: {self.out}")
        return summary

    def _write_sarah_products(self, g: R.SarahOnGrid, inv: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
        stamp = f"{SYNTHETIC_STAMP} fixture - not real data" if self.synthetic else "CM SAF SARAH-3 ICDR"
        dims = ("time", "latitude", "longitude")
        ds = xr.Dataset(
            {"ghi_wm2": (dims, g.ghi, {"units": "W m-2", "long_name": "SIS hourly mean, area-weighted over valid sources"}),
             "direct_wm2": (dims, g.direct, {"units": "W m-2", "long_name": "SID hourly mean, area-weighted over valid sources"}),
             "diffuse_wm2": (dims, g.diffuse, {"units": "W m-2", "long_name": "SIS - SID hourly mean, area-weighted over valid sources"}),
             "valid_area_fraction": (dims, g.valid_area_fraction, {"units": "1", "long_name": "valid source area / ERA5 cell area (not renormalised)"}),
             "source_code": (dims, g.source_code, {"long_name": "satellite or ERA5 fallback reason",
                                                   "codes": json.dumps({str(k): v for k, v in R.SOURCE_CODES.items()})})},
            coords={"time": g.hours.values.astype("datetime64[ns]"), "latitude": g.lat, "longitude": g.lon},
            attrs={"title": f"SARAH-3 hourly SIS/SID on the {g.grid.name} grid", "time_label": "start of the hour, UTC",
                   "hourly_rule": "0.25*G(h:00) + 0.50*G(h:30) + 0.25*G(h+1:00); NaN unless all three samples are valid",
                   "satellite_min_area_fraction": g.threshold, "source": stamp, "synthetic": str(self.synthetic)})
        enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
        atomic_write_with(self.out / HOURLY_FILE, lambda tmp: ds.to_netcdf(tmp, engine="netcdf4", encoding=enc))
        lat2, lon2 = np.meshgrid(g.lat, g.lon, indexing="ij")
        cells = pd.DataFrame({"lat": lat2.ravel(), "lon": lon2.ravel(), "domain_fraction": g.domain_fraction.ravel(),
                              "static_covered_fraction": g.static_covered_fraction.ravel(), "fully_covered": g.fully_covered.ravel(),
                              "satellite_coverage_fraction": g.satellite_coverage_fraction.ravel(), "daytime_hours": g.daytime_hours_per_cell.ravel()})
        atomic_write_with(self.out / CELLS_FILE, lambda tmp: cells.to_csv(tmp, index=False))
        px = xr.Dataset({"daytime_valid_share": (("lat", "lon"), g.pixel_daytime_valid_share, {"long_name": "share of daytime hours with a fully valid hourly value"}),
                         "ever_covered": (("lat", "lon"), g.pixel_covered.astype(np.uint8))},
                        coords={"lat": g.src_lat, "lon": g.src_lon}, attrs={"source": stamp, "daytime": "solar elevation > 5 deg at h+30 min"})
        atomic_write_with(self.out / PIXELS_FILE, lambda tmp: px.to_netcdf(tmp, engine="netcdf4"))
        atomic_write_json(self.out / INVENTORY_FILE, {"synthetic": self.synthetic, "inventory": inv, "coverage": summary,
                                                      "notes": ["fully covered = static: at least one fully valid daytime hour per source pixel, area share >= threshold of the ERA5 cell"]})

    def _sarah_products_current(self, dirs: "dict[str, Path]") -> bool:
        s = self.info.get("sarah")
        if self.refresh or not s or not all((self.out / f).is_file() for f in (HOURLY_FILE, CELLS_FILE, PIXELS_FILE)):
            return False
        cov = s.get("coverage", {})
        return (s.get("inputs") == {k: str(v) for k, v in dirs.items()} and cov.get("first_hour_utc") == iso_utc(self.hours[0])
                and cov.get("last_hour_utc") == iso_utc(self.hours[-1]) and cov.get("grid", {}).get("name") == self.grid.name)

    def _load_sarah_products(self) -> R.SarahOnGrid:
        with xr.open_dataset(self.out / HOURLY_FILE) as d:
            d = d.load()
        with xr.open_dataset(self.out / PIXELS_FILE) as p:
            p = p.load()
        cells = pd.read_csv(self.out / CELLS_FILE)
        shape = (self.grid.n_lat, self.grid.n_lon)

        def col(name: str) -> np.ndarray:
            return cells[name].to_numpy().reshape(shape)

        return R.SarahOnGrid(self.grid, pd.DatetimeIndex(d["time"].values), d["latitude"].values, d["longitude"].values, d["ghi_wm2"].values,
                             d["direct_wm2"].values, d["diffuse_wm2"].values, d["valid_area_fraction"].values, d["source_code"].values.astype(np.uint8),
                             float(d.attrs["satellite_min_area_fraction"]), col("domain_fraction"), col("static_covered_fraction"),
                             col("fully_covered").astype(bool), col("satellite_coverage_fraction"), col("daytime_hours").astype(int), p["lat"].values,
                             p["lon"].values, p["daytime_valid_share"].values, p["ever_covered"].values.astype(bool))

    def sarah_ongrid(self, required: bool = False) -> "R.SarahOnGrid | None":
        """SARAH-3 on the solar grid from memory, from the products of an earlier ingest, or by ingesting; None (gap recorded) if unavailable."""
        if "sarah_ongrid" in self._memo:
            return self._memo["sarah_ongrid"]
        try:
            if self.synthetic:
                paths = self._synthetic_sarah_paths()
                ready = all(p.is_dir() and any(p.glob("*.nc")) for p in paths.values())
                dirs = paths if ready else self.sarah_dirs()
            else:
                dirs = self.sarah_dirs()
            if self._sarah_products_current(dirs):
                self._memo["sarah_ongrid"] = self._load_sarah_products()
            else:
                self._ingest_sarah(self.sarah_dirs() if self.synthetic else dirs)
        except Unavailable as exc:
            self._gap("sarah", "; ".join(i["reason"] for i in exc.items))
            if required:
                raise
            return None
        except S.SarahError as exc:
            if exc.check != "input_missing":
                raise
            self._gap("sarah", exc.detail)
            if required:
                raise Unavailable([{"input": "SARAH-3 files", "reason": exc.detail, "key": "sarah"}]) from exc
            return None
        return self._memo["sarah_ongrid"]

    # --- ERA5 solar ----------------------------------------------------------------------------------------------------------
    def _era5_source(self) -> "Path | None":
        if self.era5_solar_path:
            return Path(self.era5_solar_path)
        if self.era5_cache.is_dir() and (any(self.era5_cache.glob("*.nc")) or any(self.era5_cache.glob("*.zip"))):
            return self.era5_cache
        return None

    def ingest_era5_solar(self, explicit: bool = True) -> None:
        if self.synthetic:
            assert self.spec is not None
            self.era5_cache.mkdir(parents=True, exist_ok=True)
            for old in list(self.era5_cache.glob("era5_solar_*")):
                old.unlink()
            fx = F.synthetic_era5_solar(self.grid, self.hours[0].strftime("%Y-%m-%dT%H:%M:%S"), len(self.hours), seed=self.spec.seed)
            name = f"era5_solar_{self.hours[0]:%Y_%m}.nc"
            (F.write_era5_solar_zip if self.spec.era5_zip else F.write_netcdf)(fx.raw, self.era5_cache / name)
        elif self.era5_solar_path:
            if not Path(self.era5_solar_path).exists():
                raise Unavailable([{"input": "ERA5 solar data", "key": "era5",
                                    "reason": f"--era5-solar-path / {C.ERA5_SOLAR_ENV} does not exist: {self.era5_solar_path}"}])
        elif self.refresh or self._era5_source() is None:
            if not self._network_ok(explicit):
                raise Unavailable([{"input": "ERA5 solar data", "key": "era5",
                                    "reason": "no files: pass --era5-solar-path (or set WINDPOWER_ERA5_SOLAR) or run `ingest-era5-solar` with CDS credentials "
                                              "(`~/.cdsapirc`); downloads happen only in the explicit stage or with --refresh, and never with --offline"}])
            if not E5.cds_credentials_available():
                raise Unavailable([{"input": "ERA5 (Copernicus CDS)", "key": "era5",
                                    "reason": "credentials unavailable: set CDSAPI_URL and CDSAPI_KEY or create ~/.cdsapirc (or pass --era5-solar-path)"}])
            ES.download_era5_solar(self.era5_cache, self.hours[0], self.hours[-1], self.grid)
        self._memo.pop("era5_hourly", None)
        res = self.era5_hourly(required=True)
        assert res is not None
        self.say(f"[ingest-era5-solar] {res.ds.sizes['time']} hours, grid {res.ds.sizes['y']}x{res.ds.sizes['x']}, convention {res.convention['convention_used']} "
                 f"(detected {res.convention['detected']})")

    def era5_hourly(self, required: bool = False) -> "ES.Era5SolarResult | None":
        if "era5_hourly" in self._memo:
            return self._memo["era5_hourly"]
        src = self._era5_source()
        if src is None and self.synthetic:
            self.ingest_era5_solar()
            return self._memo["era5_hourly"]
        if src is None or not src.exists():
            reason = (f"no ERA5 solar files in {display_path(self.era5_cache)} and no --era5-solar-path / {C.ERA5_SOLAR_ENV}; run `ingest-era5-solar` (needs CDS credentials)")
            self._gap("era5", reason)
            if required:
                raise Unavailable([{"input": "ERA5 solar data", "reason": reason, "key": "era5"}])
            return None
        res = ES.load_solar_era5(src, self.hours[0], self.hours[-1], convention=self.era5_convention)
        if (tuple(np.round(res.ds["y"].values, 5)) != tuple(np.round(self.grid.lats(), 5))
                or tuple(np.round(res.ds["x"].values, 5)) != tuple(np.round(self.grid.lons(), 5))):
            raise ES.SolarEra5Error("grid", f"ERA5 solar grid {res.ds.sizes['y']}x{res.ds.sizes['x']} does not match {self.grid.name} "
                                            f"{self.grid.n_lat}x{self.grid.n_lon}")
        self._memo["era5_hourly"] = res
        self._save_info(era5_solar={
            "source": "SYNTHETIC fixture" if self.synthetic else "Copernicus CDS reanalysis-era5-single-levels (ssrd, ssr, fdir, tisr, t2m) or a prepared atlite-style dataset",
            "input": str(src), "files": _hash_files(src), "convention": res.convention, "window": res.window, "source_detail": res.source})
        return res

    # --- public inputs -----------------------------------------------------------------------------------------------------------
    def ingest_solar_public(self, explicit: bool = True) -> None:
        if self.synthetic:
            fx = FP.synthetic_public(self.grid, seed=self.spec.seed if self.spec else 2026)
            SPUB.fetch_public(self.public_cache, FP.FakePublicSession(fx), refresh=True, sleep=lambda s: None, say=self.say)
        elif self._network_ok(explicit):
            SPUB.fetch_public(self.public_cache, refresh=self.refresh, say=self.say)
        elif SPUB.public_cache_missing(self.public_cache):
            raise Unavailable([{"input": "public inputs", "key": "public",
                                "reason": "cache incomplete and the network is not allowed here (--offline, or solar-all without --refresh): run `ingest-solar-public`"}])
        self._memo.pop("public", None)
        self.public_data(required=True)
        rep = self.info["public"]["plants"]["report"]
        self.say(f"[ingest-solar-public] {rep['operational_plants']} operational plants, {rep['operational_mw']:.2f} MW; "
                 f"population grid {self.info['public']['population']['features']} cells; municipalities {self.info['public']['municipalities']['features']}")

    def public_data(self, required: bool = False) -> "SPUB.PublicData | None":
        if "public" in self._memo:
            return self._memo["public"]
        if SPUB.public_cache_missing(self.public_cache) and self.synthetic:
            self.ingest_solar_public()
            return self._memo["public"]
        missing = SPUB.public_cache_missing(self.public_cache)
        if missing:
            reason = f"public-data cache missing ({', '.join(missing)}) in {self.public_cache}; run `ingest-solar-public` (network, no credentials)"
            self._gap("public", reason)
            if required:
                raise Unavailable([{"input": "public inputs", "reason": reason, "key": "public"}])
            return None
        pub = SPUB.load_public(self.public_cache)
        self._memo["public"] = pub
        compact = {k: {kk: vv for kk, vv in m.items() if kk != "report"} for k, m in pub.meta.items()}
        compact["plants"]["report"] = pub.plants_report
        self._save_info(public=compact)
        return pub

    def allocation(self, required: bool = False) -> "N.Allocation | None":
        if "allocation" in self._memo:
            return self._memo["allocation"]
        pub = self.public_data(required=required)
        if pub is None:
            return None
        alloc = N.allocate(SPUB.place_plants(pub.plants, pub.municipalities), pub.population, self.grid)
        self._memo["allocation"] = alloc
        self._save_info(allocation=alloc.report)
        return alloc

    # --- Fingrid solar -------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _fingrid_key_available() -> bool:
        try:
            FG.get_api_key()
            return True
        except CredentialUnavailableError:
            return False

    def _fingrid_reason(self) -> str:
        return "--offline given and the cache is incomplete" if self.offline else FS.key_missing_message()

    def ingest_fingrid_solar(self, explicit: bool = True) -> None:
        if self.synthetic:
            fx = FP.synthetic_solar_fingrid(self.hours, seed=self.spec.seed if self.spec else 2026)
            client = FG.FingridClient(F.FAKE_API_KEY, F.FakeFingridSession(fx, fail_429_once=True), request_delay=0, sleep=lambda s: None, now=lambda: F.FIXED_ISO)
            data = FS.load_or_fetch_solar(self.fg_cache, self.hours[0], self.hours[-1], client=client, refresh=True)
            verification = FS.verify_solar_datasets(client)
            verification["note"] = "SYNTHETIC: metadata served by the offline fake transport"
        else:
            client = FG.FingridClient() if self._network_ok(explicit) and self._fingrid_key_available() else None
            data = FS.load_or_fetch_solar(self.fg_cache, self.hours[0], self.hours[-1], client=client, refresh=self.refresh and client is not None,
                                          key_missing_reason=self._fingrid_reason())
            verification = FS.verify_solar_datasets(client)       # DatasetMismatchError propagates (hard failure)
        self._memo["fingrid"] = data
        self._record_fingrid(data, verification)
        if len(data.unavailable) == len(FS.SERIES):
            raise Unavailable([{"input": f"Fingrid {i}", "reason": r, "key": f"fingrid_{i}"} for i, r in data.unavailable.items()])
        self.say(f"[ingest-fingrid-solar] series {sorted(data.reports)}; unavailable {sorted(data.unavailable) or 'none'}; verification {verification['overall']}")

    def _record_fingrid(self, data: FS.SolarFingridData, verification: Mapping[str, Any]) -> None:
        for i, r in data.unavailable.items():
            self._gap(f"fingrid_{i}", r)
        self._save_info(fingrid_solar={"verification": verification, "reports": {str(k): v for k, v in data.reports.items()},
                                       "cache_meta": {str(k): v for k, v in data.cache_meta.items()}, "series": [FS.series_record(i) for i in FS.SERIES],
                                       "unavailable": {str(k): v for k, v in data.unavailable.items()}, "hourly_rule": FG.HOURLY_RULE,
                                       "from_cache_only": data.from_cache_only})

    def fingrid_solar(self) -> FS.SolarFingridData:
        """Hourly 267/248/362 from the cache only (never the network); missing series are reported in `.unavailable` and as gaps."""
        if "fingrid" not in self._memo:
            if self.synthetic and not (self.fg_cache.is_dir() and any(self.fg_cache.glob("fingrid_raw_*"))):
                self.ingest_fingrid_solar()
            else:
                data = FS.load_or_fetch_solar(self.fg_cache, self.hours[0], self.hours[-1], client=None, key_missing_reason=self._fingrid_reason())
                self._memo["fingrid"] = data
                self._record_fingrid(data, (self.info.get("fingrid_solar") or {}).get("verification") or FS.verify_solar_datasets(None))
        data = self._memo["fingrid"]
        for i, r in data.unavailable.items():
            self._gap(f"fingrid_{i}", r)
        return data

    # --- hourly / PV -----------------------------------------------------------------------------------------------------------------
    def solar_hourly(self) -> None:
        g, e = self.sarah_ongrid(), self.era5_hourly()
        if g is None and e is None:
            raise Unavailable([{"input": "SARAH-3 or ERA5 solar", "reason": f"sarah: {self.gaps.get('sarah')}; era5: {self.gaps.get('era5')}", "key": "sarah"}])
        sections: dict[str, Any] = {"sarah": g is not None, "era5": e is not None}
        if e is not None:
            ds = e.ds.astype("float32")
            ds.attrs["source"] = "SYNTHETIC fixture" if self.synthetic else "ERA5"
            enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
            atomic_write_with(self.out / "era5_solar_hourly.nc", lambda tmp: ds.to_netcdf(tmp, engine="netcdf4", encoding=enc))
        if e is not None and g is not None:
            table = N.ghi_table(e.ds, g)
            sections["ghi_table"] = table
            self._json({"title": "SARAH-3 vs ERA5 irradiance on the fully covered cells (hourly means incl. night, W m-2)", **table}, "irradiance_comparison.json")
        self._save_info(hourly=sections)
        extra = ""
        if sections.get("ghi_table", {}).get("cell_hours"):
            t = sections["ghi_table"]
            extra = f"; covered-cell GHI SARAH {t['sarah_ghi_wm2']:.1f} vs ERA5 {t['era5_ghi_wm2']:.1f} W m-2"
        self.say(f"[solar-hourly] SARAH-3 {'yes' if g is not None else 'unavailable'}; ERA5 {'yes' if e is not None else 'unavailable'}{extra}")

    def cubes(self, required: bool = False) -> "P.CfCubes | None":
        if "cubes" in self._memo:
            return self._memo["cubes"]
        e = self.era5_hourly(required=required)
        if e is None:
            return None
        g = self.sarah_ongrid()
        if g is None:
            self._gap("hybrid", f"SARAH-3 unavailable: {self.gaps.get('sarah')}")
        cubes = P.compute_cubes(e.ds, g)
        self._memo["cubes"] = cubes
        return cubes

    def solar_pv(self) -> None:
        cubes = self.cubes(required=True)
        assert cubes is not None
        g = self.sarah_ongrid()
        checks = {}
        for m in ("era5", "hybrid"):
            if m == "era5" or cubes.hybrid is not None:
                alt = cubes.irradiance[m]["solar_altitude"].transpose("time", "y", "x").rename({"y": "latitude", "x": "longitude"})
                checks[m] = P.check_cf(cubes.get(m), alt)
        dims = ("time", "latitude", "longitude")
        data: dict[str, Any] = {"cf_era5": (dims, cubes.era5.values.astype("float32"), {"units": "1"})}
        if cubes.hybrid is not None:
            data["cf_hybrid"] = (dims, cubes.hybrid.values.astype("float32"), {"units": "1"})
            data["source_code"] = (dims, cubes.irradiance["hybrid"]["source_code"].transpose("time", "y", "x").values.astype(np.uint8),
                                   {"codes": json.dumps({str(k): v for k, v in R.SOURCE_CODES.items()})})
        ds = xr.Dataset(data, coords={"time": cubes.era5["time"].values, "latitude": cubes.era5["latitude"].values, "longitude": cubes.era5["longitude"].values},
                        attrs={"title": "PV capacity factor (fraction), atlite CSi, slope 40, azimuth 180", "time_label": "start of the hour, UTC",
                               "source": "SYNTHETIC fixture" if self.synthetic else "ERA5 (+ SARAH-3 for hybrid)", "synthetic": str(self.synthetic)})
        enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
        atomic_write_with(self.out / "cf_cubes.nc", lambda tmp: ds.to_netcdf(tmp, engine="netcdf4", encoding=enc))
        cell = pd.DataFrame({"lat": np.repeat(cubes.era5["latitude"].values, self.grid.n_lon), "lon": np.tile(cubes.era5["longitude"].values, self.grid.n_lat),
                             "mean_cf_era5": cubes.era5.mean("time").values.ravel()})
        covered = None if g is None else g.fully_covered
        if cubes.hybrid is not None and g is not None:
            cell["mean_cf_hybrid"] = cubes.hybrid.mean("time").values.ravel()
            cell["fully_covered"] = g.fully_covered.ravel()
            cell["satellite_coverage_fraction"] = g.satellite_coverage_fraction.ravel()
        self._csv(cell, "cf_cell_summary.csv")
        pv: dict[str, Any] = {"panel": P.panel_record(), "cf_checks": checks, "mean_cf_era5": N.mean_cf(cubes.era5),
                              "atlite_notes": ["cutout.pv reads only the variables it needs; the in-memory cutout has no wind variable and none is requested",
                                               "temperature is passed in Kelvin (CSi r_tmod = 298 K)"]}
        if cubes.hybrid is not None and covered is not None:
            pv |= {"mean_cf_hybrid": N.mean_cf(cubes.hybrid), "hybrid": P.hybrid_diagnostics(cubes.irradiance["hybrid"]),
                   "mean_cf_era5_covered": N.mean_cf(cubes.era5, covered) if covered.any() else None,
                   "mean_cf_hybrid_covered": N.mean_cf(cubes.hybrid, covered) if covered.any() else None}
        self._save_info(pv=pv)
        self._memo["pv_done"] = True
        self.say(f"[solar-pv] mean CF ERA5-only {pv['mean_cf_era5']:.4f}" + (f", hybrid {pv['mean_cf_hybrid']:.4f}" if "mean_cf_hybrid" in pv else ", hybrid unavailable"))

    # --- national ------------------------------------------------------------------------------------------------------------------------
    def national_table(self) -> "tuple[pd.DataFrame | None, dict[str, Any] | None]":
        """(hourly table with model MW and Fingrid series, capacity-reconciliation results); columns only for what is available."""
        if "national" in self._memo:
            return self._memo["national"]
        alloc, fg, cubes = self.allocation(), self.fingrid_solar(), self.cubes()
        cols: dict[str, pd.Series] = {}
        recon: "dict[str, Any] | None" = None
        cap = fg.hourly["capacity_267_mw"].reindex(self.hours) if "capacity_267_mw" in fg.hourly else None
        if cap is None:
            self._gap("national", "national MW needs the Fingrid 267 hourly capacity: " + self.gaps.get("fingrid_267", FS.key_missing_message()))
        elif cubes is not None and alloc is not None:
            cube_main, cube_pop = N.capacity_cube(cap, alloc), N.capacity_cube(cap, alloc, population_only=True)
            recon = {"large_plus_population": N.check_reconciliation(cube_main, cap), "population_only": N.check_reconciliation(cube_pop, cap)}
            cols["era5_only_mw"] = pd.Series(N.national_power(cubes.era5, cube_main), index=self.hours)
            cols["era5_only_popw_mw"] = pd.Series(N.national_power(cubes.era5, cube_pop), index=self.hours)
            if cubes.hybrid is not None:
                cols["hybrid_mw"] = pd.Series(N.national_power(cubes.hybrid, cube_main), index=self.hours)
                cols["hybrid_popw_mw"] = pd.Series(N.national_power(cubes.hybrid, cube_pop), index=self.hours)
        for c in fg.hourly.columns:
            cols[c] = fg.hourly[c]
        table = pd.DataFrame(cols).reindex(self.hours) if cols else None
        if table is not None:
            table.index.name = "time_utc"
        self._memo["national"] = (table, recon)
        return self._memo["national"]

    def solar_national(self) -> None:
        alloc = self.allocation(required=True)
        assert alloc is not None
        lat2, lon2 = np.meshgrid(self.grid.lats(), self.grid.lons(), indexing="ij")
        cells = pd.DataFrame({"lat": lat2.ravel(), "lon": lon2.ravel(), "inhabitants": alloc.pop_by_cell.ravel(), "population_share": alloc.pop_share.ravel(),
                              "large_capacity_mw": alloc.large_by_cell.ravel()})
        self._csv(cells[(cells["inhabitants"] > 0) | (cells["large_capacity_mw"] > 0)], "allocation_cells.csv")
        self._csv(alloc.plants.drop(columns=[c for c in ("installed_flag", "demolish_year", "operational") if c in alloc.plants]), "allocation_plants.csv")
        self._json(alloc.report, "allocation_report.json")
        table, recon = self.national_table()
        sec: dict[str, Any] = {"allocation": alloc.report}
        if table is not None:
            self._csv(table, "national_hourly.csv", index=True, label="time_utc")
            metrics = N.comparison_metrics(table)
            sec |= {"metrics": metrics, "reconciliation": recon}
            if "capacity_267_mw" in table:
                for m in ("era5_only_mw", "hybrid_mw"):
                    if m in table:
                        sec[f"mean_cf_vs_267_{m.split('_')[0]}"] = N.mean_cf_vs_capacity(table[m], table["capacity_267_mw"])
            daily = pd.DataFrame({c.replace("_mw", "_gwh"): table[c].resample("1D").sum(min_count=1) / 1000.0 for c in table.columns if c.endswith("_mw")})
            daily.insert(0, "hours", table.iloc[:, 0].resample("1D").count())
            self._csv(daily, "daily.csv", index=True, label="date_utc")
            self._json(metrics, "metrics.json")
            if metrics["models"]:
                self.say("[solar-national] " + ", ".join(f"{k} {v['energy']['gwh']:.2f} GWh" for k, v in metrics["models"].items()))
        else:
            self.say("[solar-national] national series unavailable: " + self.gaps.get("national", "no model or Fingrid series"))
        self._save_info(national=sec)
        self.say(f"[solar-national] L = {alloc.large_mw:.2f} MW in {alloc.report['cells_with_large_plants']} cells; population {alloc.report['inhabitants_total']:.0f} "
                 f"in {alloc.report['cells_with_population']} cells")

    # --- validation / manifest ------------------------------------------------------------------------------------------------------------
    def _observed(self, alloc: "N.Allocation | None", g: "R.SarahOnGrid | None", cubes: "P.CfCubes | None", table: "pd.DataFrame | None") -> dict[str, "float | None"]:
        obs: dict[str, "float | None"] = {"hourly_values": float(len(self.hours)), "grid_cells": float(self.grid.n_lat * self.grid.n_lon)}
        if g is not None:
            cov = R.coverage_summary(g)
            obs["fully_covered_cells"] = float(cov["fully_covered_cells"])
            obs["never_covered_source_pixels_pct"] = 100.0 * cov["source_pixels"]["never_covered_share"]
        if alloc is not None:
            r = alloc.report
            obs |= {"large_plants": r["plants"], "large_plants_mw": r["large_total_mw"], "largest_plant_mw": (r["largest_plant"] or {}).get("nominal_mw"),
                    "largest_plant_share_pct": 100 * r["largest_plant_share_of_large_capacity"], "inhabited_cells": r["population_cells"], "inhabitants": r["inhabitants_total"]}
        e = self.era5_hourly()
        if e is not None and g is not None:
            t2 = N.ghi_table(e.ds, g)
            if t2.get("cell_hours"):
                obs |= {"ghi_sarah_covered_wm2": t2["sarah_ghi_wm2"], "ghi_era5_covered_wm2": t2["era5_ghi_wm2"]}
        if cubes is not None:
            obs["mean_cf_era5_all_cells"] = N.mean_cf(cubes.era5)
            if cubes.hybrid is not None and g is not None:
                obs["mean_cf_hybrid_all_cells"] = N.mean_cf(cubes.hybrid)
                if g.fully_covered.any():
                    obs["mean_cf_era5_covered_cells"] = N.mean_cf(cubes.era5, g.fully_covered)
                    obs["mean_cf_hybrid_covered_cells"] = N.mean_cf(cubes.hybrid, g.fully_covered)
        if table is not None:
            m = N.comparison_metrics(table)

            def en(c: str) -> "float | None":
                return (m["models"].get(c) or m["references"].get(c) or {}).get("energy", {}).get("gwh")

            obs |= {"energy_era5_gwh": en("era5_only_mw"), "energy_hybrid_gwh": en("hybrid_mw"), "energy_era5_popw_gwh": en("era5_only_popw_mw"),
                    "energy_hybrid_popw_gwh": en("hybrid_popw_mw"), "energy_fingrid_248_gwh": en("fingrid_248")}
            for model, tag in (("era5_only_mw", "era5"), ("hybrid_mw", "hybrid")):
                mm = m["models"].get(model)
                if not mm:
                    continue
                for ref, rt in (("fingrid_248", "248"), ("fingrid_362_av06", "362")):
                    v = mm.get(f"vs_{ref}")
                    if v:
                        obs[f"pearson_{tag}_vs_{rt}"] = v["pearson_r"]
                        if rt == "248":
                            obs[f"energy_ratio_{tag}_to_248"] = v["energy_ratio_model_over_reference"]
                obs[f"cycle_centre_{tag}_min"] = mm["cycle_centre"]["minutes_utc"]
                if "capacity_267_mw" in table:
                    obs[f"mean_cf_vs_267_{tag}"] = N.mean_cf_vs_capacity(table[model], table["capacity_267_mw"])
            for ref, key in (("fingrid_248", "248"), ("fingrid_362_av06", "362")):
                if ref in m["references"]:
                    obs[f"cycle_centre_{key}_min"] = m["references"][ref]["cycle_centre"]["minutes_utc"]
        return obs

    def solar_validate(self) -> list[CheckResult]:
        g, e, alloc, cubes = self.sarah_ongrid(), self.era5_hourly(), self.allocation(), self.cubes()
        table, recon = self.national_table()
        if g is None and e is None and alloc is None:
            raise Unavailable([{"input": "any solar input", "key": "sarah",
                                "reason": "; ".join(f"{k}: {v}" for k, v in self.gaps.items()) or "nothing to validate"}])
        if cubes is not None and not self._memo.get("pv_done"):
            self.solar_pv()
        sources = {"sarah": self.info.get("sarah", {}).get("source", "") if g is not None else None,
                   "era5": self.info["era5_solar"]["source"] if e is not None else None,
                   "public": ("SYNTHETIC fixture" if self.synthetic else "public WFS/XLSX") if alloc is not None else None}
        ev = SV.SolarEvidence(mode=self.mode, synthetic=self.synthetic, hours=self.hours, sarah_info=self.info.get("sarah") if g is not None else None, ongrid=g,
                              era5_info=self.info.get("era5_solar") if e is not None else None, pv_info=self.info.get("pv") if cubes is not None else None,
                              allocation=alloc, plants_report=(self.info.get("public", {}).get("plants", {}).get("report")),
                              reconciliation=None if recon is None else recon["large_plus_population"], national=table,
                              fingrid_info=self.info.get("fingrid_solar"), sources={k: v for k, v in sources.items() if v})
        results = SV.run_checks(ev)
        avail = {k: self.gaps.get(k, "") for k in ("sarah", "era5", "public", "fingrid_267", "fingrid_248", "fingrid_362")}
        rec = SV.reconciliation(self._observed(alloc, g, cubes, table), self.synthetic, avail)
        self._csv(results_frame(results), "validation.csv")
        self._json({"summary": {"passed": sum(r.status == "pass" for r in results), "failed": len(failed_checks(results)),
                                "skipped": sum(r.status == "skipped" for r in results)}, "synthetic": self.synthetic, "results": [r.as_dict() for r in results]},
                   "validation.json")
        self._csv(rec, "reconciliation.csv")
        self._json(rec.to_dict(orient="records"), "reconciliation.json")
        self._refresh_manifest()
        for r in results:
            self.say(f"[solar-validate] {r.status.upper():7s} {r.name}: {r.detail}")
        self.say(f"[solar-validate] reconciliation rows: {rec['status'].value_counts().to_dict()}")
        return results

    def _read_out(self, name: str) -> Any:
        p = self.out / name
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    def _refresh_manifest(self) -> dict[str, Any]:
        """(Re)write manifest.json from run info and whatever result files exist now (so it lists the site artifact once exported)."""
        unavailable = [{"input": k, "reason": v} for k, v in self.gaps.items()]
        status = "partial_inputs_unavailable" if self.gaps else "complete"
        self._save_info(unavailable=unavailable, status=status)
        return SV.write_solar_manifest(self.out, mode=self.mode, synthetic=self.synthetic, info=dict(self.info), status=status, hours=self.hours,
                                       validation=self._read_out("validation.json"), recon=self._read_out("reconciliation.json"), unavailable=unavailable,
                                       metrics=self._read_out("metrics.json"))

    # --- artifact ----------------------------------------------------------------------------------------------------------------------------
    def export_site_solar(self) -> dict[str, Any]:
        g, alloc, cubes = self.sarah_ongrid(), self.allocation(), self.cubes()
        table, _ = self.national_table()
        if g is None and cubes is None:
            raise Unavailable([{"input": "SARAH-3 or ERA5 solar", "key": "sarah",
                                "reason": f"nothing to export: sarah: {self.gaps.get('sarah')}; era5: {self.gaps.get('era5')}"}])
        cov = self.info.get("sarah", {}).get("coverage") if g is not None else None
        site = export_site_solar(self.out, mode=self.mode, synthetic=self.synthetic, hours=self.hours, grid=self.grid, ongrid=g, cubes=cubes, allocation=alloc,
                                 national=table, info=self.info, unavailable=dict(self.gaps), coverage_summary=cov)
        self._refresh_manifest()
        modes = ", ".join(f"{k}: {'yes' if v['available'] else 'unavailable'}" for k, v in site["modes"].items())
        self.say(f"[export-site-solar] {self.out / 'site' / 'solar' / 'v1'}: {site['total_bytes'] / 1e6:.2f} MB, {len(site['files'])} files ({site['label']}); modes {modes}")
        return site

    # --- everything ----------------------------------------------------------------------------------------------------------------------------
    def run_all(self) -> "tuple[list[CheckResult], str]":
        """Every stage in order. Real mode: stages without their inputs are recorded and skipped (partial run); synthetic modes must complete.

        Returns (validation results, status). Raises Unavailable only if no stage could do any work.
        """
        completed: list[str] = []
        steps: list[tuple[str, Callable[[], Any]]] = [
            ("ingest-sarah", self._ingest_or_reuse_sarah), ("ingest-era5-solar", lambda: self.ingest_era5_solar(explicit=False)),
            ("ingest-solar-public", lambda: self.ingest_solar_public(explicit=False)), ("ingest-fingrid-solar", lambda: self.ingest_fingrid_solar(explicit=False)),
            ("solar-hourly", self.solar_hourly), ("solar-pv", self.solar_pv), ("solar-national", self.solar_national),
            ("export-site-solar", self.export_site_solar)]
        for name, fn in steps:
            try:
                fn()
                completed.append(name)
            except Unavailable as exc:
                if self.synthetic:
                    raise
                for i in exc.items:
                    self._gap(i.get("key") or name, i["reason"])
                    self.say(f"[{name}] UNAVAILABLE - {i['input']}: {i['reason']}")
        results: list[CheckResult] = []
        try:
            results = self.solar_validate()
            completed.append("solar-validate")
        except Unavailable as exc:
            if self.synthetic:
                raise
            self.say(f"[solar-validate] UNAVAILABLE - {exc}")
        if not any(c not in ("solar-validate", "export-site-solar") for c in completed):
            raise Unavailable([{"input": k, "reason": v} for k, v in self.gaps.items()] or [{"input": "all stages", "reason": "no stage could run"}])
        return results, ("partial_inputs_unavailable" if self.gaps else "complete")

    def _ingest_or_reuse_sarah(self) -> None:
        if not self.synthetic and not self.refresh and self.sarah_ongrid() is not None:
            self.say("[ingest-sarah] reusing current products (same inputs, window and grid); --refresh forces a new ingest")
        else:
            self.ingest_sarah()

    def run_stage(self, stage: str) -> "list[CheckResult] | None":
        table: dict[str, Callable[[], Any]] = {
            "ingest-sarah": self.ingest_sarah, "ingest-era5-solar": self.ingest_era5_solar, "ingest-solar-public": self.ingest_solar_public,
            "ingest-fingrid-solar": self.ingest_fingrid_solar, "solar-hourly": self.solar_hourly, "solar-pv": self.solar_pv,
            "solar-national": self.solar_national, "solar-validate": self.solar_validate, "export-site-solar": self.export_site_solar}
        out = table[stage]()
        return out if stage == "solar-validate" else None

    def write_unavailable(self, items: "list[dict[str, str]]") -> None:
        """Minimal manifest for a run that stopped on missing inputs - never over an existing (fuller) manifest."""
        from . import manifest

        self.info["unavailable_runtime"] = items
        if not (self.out / "manifest.json").is_file():
            manifest.unavailable_manifest(self.mode, items, self.out)
