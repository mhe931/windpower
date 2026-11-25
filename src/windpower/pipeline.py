"""Stage orchestration shared by the CLI and the tests.

Stages are stateless across processes: each one rebuilds what it needs from the raw caches under
`<data>/cache/...` (memoised within a process), writes its outputs under `<data>/outputs/<mode>/`
and records provenance in `<data>/interim/<mode>/run_info.json`, which the manifest is built from.

Synthetic modes (smoke, demo) generate their own fixtures and never touch the network or real data.
Full mode never falls back to synthetic data: missing inputs raise `Unavailable`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from . import calibrate as K
from . import config as C
from . import era5, export_site, fingrid, fleet, manifest, validate
from . import fixtures as F
from . import simulate as S
from .io_utils import (
    CredentialUnavailableError,
    atomic_write_json,
    atomic_write_with,
    iso_utc,
    sha256_file,
    utc_hour_range,
)

ISO = "%Y-%m-%dT%H:%M:%SZ"
STAGES = ("ingest-era5", "ingest-fleet", "ingest-fingrid", "fleet", "simulate", "calibrate", "validate", "export-site")


class Unavailable(RuntimeError):
    """Required inputs are not available (credentials, caches, offline flag); `items` lists them."""

    def __init__(self, items: "list[dict[str, str]]"):
        self.items = items
        super().__init__("; ".join(f"{i['input']}: {i['reason']}" for i in items))


@dataclass(frozen=True)
class ModeSpec:
    grid: C.GridSpec
    start: str
    hours: int
    n_farms: int
    seed: int = 2026


MODE_SPECS = {
    "smoke": ModeSpec(C.SMOKE_GRID, "2025-07-22T00:00:00", 24 * 21, 8),      # crosses July/August: plain NetCDF + CDS ZIP
    "demo": ModeSpec(C.FINLAND_GRID, "2025-07-01T00:00:00", 24 * 28, 40),    # full 45 x 52 Finland grid
}


class Run:
    def __init__(self, mode: str, data_dir: "str | Path | None" = None, offline: "bool | None" = None, refresh: bool = False,
                 start: "str | None" = None, end: "str | None" = None, say: Callable[[str], None] = print):
        if mode not in C.RUN_MODES:
            raise ValueError(f"mode must be one of {C.RUN_MODES}")
        self.mode, self.say, self.refresh = mode, say, refresh
        self.synthetic = mode in C.SYNTHETIC_MODES
        self.offline = True if self.synthetic else bool(offline)
        self.spec = MODE_SPECS.get(mode)
        self.grid = self.spec.grid if self.spec else C.FINLAND_GRID
        if self.spec:
            self.start = self.spec.start
            self.hours = utc_hour_range(self.start, pd.Timestamp(self.start) + pd.Timedelta(hours=self.spec.hours - 1))
        else:
            self.start = start or C.WINDOW_START
            self.hours = utc_hour_range(self.start, end or C.WINDOW_END)
        self.end = self.hours[-1].strftime("%Y-%m-%dT%H:%M:%S")
        self.cache = C.cache_dir(mode, data_dir)
        self.interim = C.interim_dir(mode, data_dir)
        self.out = C.output_dir(mode, data_dir)
        self.era5_cache, self.fleet_cache, self.fg_cache = self.cache / "era5", self.cache / "fleet", self.cache / "fingrid"
        self._memo: dict[str, Any] = {}
        info_path = self.interim / "run_info.json"
        self.info: dict[str, Any] = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}

    # --- helpers -------------------------------------------------------------------------
    def _save_info(self, **sections: Any) -> None:
        self.info.update(sections)
        atomic_write_json(self.interim / "run_info.json", self.info)

    def _csv(self, df: pd.DataFrame, name: str, index: bool = False) -> None:
        atomic_write_with(self.out / name, lambda tmp: df.to_csv(tmp, index=index, date_format=ISO))

    def _json(self, obj: Any, name: str) -> None:
        atomic_write_json(self.out / name, obj)

    # --- availability --------------------------------------------------------------------
    def _era5_missing(self) -> list[str]:
        names = [n for n, _ in era5.build_cds_requests(self.start, self.end, self.grid)]
        return [n for n in names if not (self.era5_cache / n).is_file() or (self.era5_cache / n).stat().st_size <= era5.MIN_FILE_BYTES]

    def _fleet_cached(self) -> bool:
        return (self.fleet_cache / C.OSM_CACHE_FILE).is_file() and (self.fleet_cache / C.ARCGIS_CACHE_FILE).is_file()

    def _fingrid_window(self) -> "tuple[str, str]":
        return fingrid.window_for_hours(self.hours[0], self.hours[-1])

    def _fingrid_cache_complete(self) -> bool:
        start, end = self._fingrid_window()
        for ds_id, col in fingrid.DATASETS.items():
            raw, meta = fingrid.read_raw_cache(self.fg_cache, ds_id, col)
            if raw is None or not fingrid._covers_start(meta, start) or not fingrid.is_cache_complete(raw, end):
                return False
        return True

    def _fingrid_key_available(self) -> bool:
        try:
            fingrid.get_api_key()
            return True
        except CredentialUnavailableError:
            return False

    def preflight(self) -> list[dict[str, str]]:
        """Everything a full run needs but does not have (empty for synthetic modes)."""
        if self.synthetic:
            return []
        items: list[dict[str, str]] = []
        if self.offline and self.refresh:
            items.append({"input": "flags", "reason": "--refresh needs the network but --offline was given"})
        if self.refresh or self._era5_missing():
            if self.offline:
                items.append({"input": "ERA5 monthly files", "reason": f"cache incomplete ({len(self._era5_missing())} month(s) missing) and --offline"})
            elif not era5.cds_credentials_available():
                items.append({"input": "ERA5 (Copernicus CDS)", "reason": "credentials unavailable: set CDSAPI_URL and CDSAPI_KEY or create ~/.cdsapirc"})
        if not self._fleet_cached() and not (self.refresh and not self.offline):
            items.append({"input": "fleet inventories (OSM + ArcGIS)", "reason": "raw cache missing; run `ingest-fleet --refresh` (network, no credentials needed)"})
        if self.refresh or not self._fingrid_cache_complete():
            if self.offline:
                items.append({"input": "Fingrid series 181/248/124/319", "reason": "cache incomplete and --offline"})
            elif not self._fingrid_key_available():
                items.append({"input": "Fingrid API", "reason": "credential unavailable: set FINGRID_API_KEY in the environment or an untracked .env"})
        return items

    def _unavailable_report(self) -> list[dict[str, str]]:
        if self.synthetic:
            return [{"input": "real ERA5 data", "reason": "synthetic run: fixtures were generated offline"},
                    {"input": "real fleet snapshot (OSM, ArcGIS)", "reason": "synthetic run"},
                    {"input": "real Fingrid series and dataset metadata", "reason": "synthetic run (metadata comes from the fake transport)"},
                    {"input": "documented-target reconciliation", "reason": "not reproducible without real data"}]
        out = list(self.info.get("unavailable_runtime", []))
        if (self.info.get("fingrid", {}).get("verification", {}).get("overall") == "unverified"):
            out.append({"input": "Fingrid dataset metadata", "reason": "unverified (no key or --offline)"})
        if not (self.info.get("fleet", {}).get("arcgis_meta") or {}).get("copyright_text"):
            out.append({"input": "ArcGIS layer licence text", "reason": "copyrightText/licenseInfo not present in the cached layer metadata"})
        return out

    # --- ERA5 ----------------------------------------------------------------------------------
    def ingest_era5(self) -> None:
        if self.synthetic:
            self.era5_cache.mkdir(parents=True, exist_ok=True)
            for old in self.era5_cache.glob("era5_*.nc"):
                old.unlink()
            F.write_era5_cache(self.era5_cache, self.grid, self.start, len(self.hours), seed=self.spec.seed)  # type: ignore[union-attr]
        elif self.refresh or self._era5_missing():
            items = [i for i in self.preflight() if i["input"].startswith("ERA5") or i["input"] == "flags"]
            if items:
                raise Unavailable(items)
            era5.download_era5_months(self.era5_cache, self.start, self.end, self.grid)
        self._memo.pop("era5", None)
        self.era5()
        self.say(f"[ingest-era5] {self.info['era5']['window']['present_hours']} hours, grid {self.info['era5']['grid']['n_lat']}x{self.info['era5']['grid']['n_lon']}")

    def era5(self):
        if "era5" not in self._memo:
            if self._era5_missing():
                if self.synthetic:
                    self.ingest_era5()
                    return self._memo["era5"]
                raise Unavailable([{"input": "ERA5 monthly files", "reason": f"missing in {self.era5_cache}; run `ingest-era5` (needs CDS credentials)"}])
            loaded = era5.load_era5_window(self.era5_cache, self.start, self.end)
            val = era5.validate_era5(loaded.ds, self.start, self.end, grid=self.grid)
            ds = era5.engineer_features(loaded.ds)
            files = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(self.era5_cache.glob("era5_*.nc"))]
            combined = manifest_hash([f["sha256"] for f in files])
            self._memo["era5"] = ds
            self._save_info(era5={
                "source": "SYNTHETIC fixture" if self.synthetic else "Copernicus CDS reanalysis-era5-single-levels (hourly)",
                "retrieval": "generated offline" if self.synthetic else "see file hashes; retrieval date not recorded by CDS client",
                "window": {"start_utc": iso_utc(self.hours[0]), "end_utc": iso_utc(self.hours[-1]), "expected_hours": len(self.hours),
                           "present_hours": val["n_times"]},
                "grid": {"name": self.grid.name, "n_lat": val["n_lat"], "n_lon": val["n_lon"], "res_deg": C.GRID_RES_DEG,
                         "lat_range": [float(ds.latitude.min()), float(ds.latitude.max())], "lon_range": [float(ds.longitude.min()), float(ds.longitude.max())],
                         "source_lat_order": val["source_lat_order"]},
                "files": files, "combined_sha256": combined, "loaded_files": loaded.loaded_files, "skipped_files": loaded.skipped_files,
                "summary_stats": era5.summary_stats(ds)})
        return self._memo["era5"]

    # --- fleet --------------------------------------------------------------------------------------
    def ingest_fleet(self) -> None:
        if self.synthetic:
            F.write_fleet_cache(self.fleet_cache, grid=self.grid, seed=self.spec.seed, n_farms=self.spec.n_farms)  # type: ignore[union-attr]
        elif self.refresh:
            if self.offline:
                raise Unavailable([{"input": "flags", "reason": "--refresh needs the network but --offline was given"}])
            fleet.refresh_fleet_cache(self.fleet_cache)
        elif not self._fleet_cached():
            raise Unavailable([{"input": "fleet inventories (OSM + ArcGIS)", "reason": "raw cache missing; run `ingest-fleet --refresh`"}])
        self._memo.pop("fleet", None)
        r = self.fleet_result()
        self.say(f"[ingest-fleet] {r.stats['total_turbines']} turbines, {r.totals['turbine_table_mw']:.2f} MW, {r.totals['active_cells']} active cells")

    def fleet_result(self) -> fleet.FleetResult:
        if "fleet" not in self._memo:
            if not self._fleet_cached():
                if self.synthetic:
                    self.ingest_fleet()
                    return self._memo["fleet"]
                raise Unavailable([{"input": "fleet inventories (OSM + ArcGIS)", "reason": f"raw cache missing in {self.fleet_cache}; run `ingest-fleet --refresh`"}])
            ds = self.era5()
            osm, osm_meta = fleet.load_osm_cache(self.fleet_cache)
            arc, arc_meta = fleet.load_arcgis_cache(self.fleet_cache)
            res = fleet.build_fleet(osm, arc, ds.latitude.values, ds.longitude.values, osm_meta, arc_meta)
            self._memo["fleet"] = res
            t = res.turbines
            imputation = {p: {"imputed": int(t[f"{p}_imputed"].sum()), "methods": {k: int(v) for k, v in t.loc[t[f"{p}_imputed"], f"{p}_impute_method"].value_counts().items()}}
                          for p in ("capacity", "hub", "rotor", "year")}
            self._save_info(fleet={
                "snapshot": res.snapshot, "stats": res.stats, "totals": res.totals, "column_mapping": res.column_mapping,
                "imputation_flag_counts": imputation, "source_counts": {k: int(v) for k, v in t["source"].value_counts().items()},
                "arcgis_meta": {k: arc_meta.get(k) for k in ("copyright_text", "license_info", "layer_name", "retrieved_utc", "last_edit_utc", "query_format")},
                "osm_meta": {k: osm_meta.get(k) for k in ("mirror", "retrieved_utc", "base_timestamp_utc", "element_count")}})
        return self._memo["fleet"]

    def fleet(self) -> None:
        r = self.fleet_result()
        self._csv(r.turbines, "fleet_turbines.csv")
        self._csv(r.pixel_agg, "fleet_pixel_aggregation.csv")
        px = r.pixels.drop(columns="geometry")
        self._csv(px, "grid.csv")
        atomic_write_with(self.out / "grid.parquet", lambda tmp: px.to_parquet(tmp, index=False))
        ds = self.era5()
        mat = pd.DataFrame(r.capacity_matrix, index=pd.Index(ds.latitude.values, name="lat"), columns=[f"{v:g}" for v in ds.longitude.values])
        self._csv(mat, "capacity_matrix.csv", index=True)
        self._csv(r.specs.drop(columns=[c for c in ("geometry",) if c in r.specs]), "fleet_specs.csv")
        self._json({"totals": r.totals, "stats": r.stats, "snapshot": r.snapshot, "column_mapping": r.column_mapping,
                    "capacity_matrix_axes": "rows = latitude ascending, columns = longitude ascending"}, "fleet_summary.json")
        self.say(f"[fleet] outputs written: {r.totals['turbine_table_mw']:.2f} MW, {r.totals['unassigned_turbines']} unassigned turbine(s)")

    # --- simulation ------------------------------------------------------------------------------------
    def sim(self) -> S.SimulationResult:
        if "sim" not in self._memo:
            self._memo["sim"] = S.simulate_archetypes(self.era5(), self.fleet_result().specs, engine="atlite")
        return self._memo["sim"]

    def simulate(self) -> None:
        sim = self.sim()
        self._csv(sim.archetypes, "archetypes.csv")
        self._csv(sim.hourly, "theoretical_hourly.csv", index=True)
        self._json({n: {"V": c["V"], "POW": c["POW"], "hub_height_m": c["hub_height"], "rated_mw": c["P"],
                        "specific_power_wm2": c["specific_power_wm2"], "installed_mw": c["installed_mw"]} for n, c in sim.curves.items()},
                   "power_curves.json")
        self._json({n: vars(r) for n, r in sim.layout.items()} | {"_notes": sim.notes}, "layout_report.json")
        self.say(f"[simulate] {len(sim.archetypes)} archetypes, installed {sim.installed_mw:.2f} MW, mean theoretical {sim.theoretical_mw.mean():.2f} MW")

    # --- Fingrid ---------------------------------------------------------------------------------------
    def _write_synthetic_fingrid(self) -> None:
        sim = self.sim()
        obs = F.synthetic_observed(sim.theoretical_mw, sim.installed_mw, seed=self.spec.seed)  # type: ignore[union-attr]
        n = len(obs)
        fx = F.synthetic_fingrid(obs, [obs.index[10], obs.index[11], obs.index[n // 2]])
        self.fg_cache.mkdir(parents=True, exist_ok=True)
        for old in self.fg_cache.glob("fingrid_raw_*"):
            old.unlink()
        start, end = self._fingrid_window()
        for ds_id, col in fingrid.DATASETS.items():
            raw = pd.DataFrame(fx.rows[ds_id])
            meta = fingrid._build_meta(ds_id, col, raw, start, end, F.FIXED_ISO, 0, 0) | {"synthetic": "SYNTHETIC"}
            fingrid.write_raw_cache(self.fg_cache, ds_id, col, raw, meta)

    def _verification(self, client: "fingrid.FingridClient | None") -> dict[str, Any]:
        if self.synthetic:
            fake = F.FakeFingridSession(F.FingridFixture({}, F.synthetic_fingrid_metadata(), pd.DataFrame(), []), fail_429_once=False)
            c = fingrid.FingridClient(F.FAKE_API_KEY, fake, request_delay=0, sleep=lambda s: None, now=lambda: F.FIXED_ISO)
            block = fingrid.verify_datasets(c)
            block["note"] = "SYNTHETIC: metadata served by the offline fake transport"
            return block
        return fingrid.verify_datasets(client)  # DatasetMismatchError propagates (hard failure)

    def ingest_fingrid(self) -> None:
        if self.synthetic:
            self._write_synthetic_fingrid()
        elif self.refresh or not self._fingrid_cache_complete():
            items = [i for i in self.preflight() if i["input"].startswith("Fingrid") or i["input"] == "flags"]
            if items:
                raise Unavailable(items)
        self._memo.pop("fingrid", None)
        self.fingrid_data()
        self.say(f"[ingest-fingrid] verification: {self.info['fingrid']['verification']['overall']}; "
                 f"missing wind hours: {len(self.info['fingrid']['reports']['wind_power_mw']['missing_hours'])}")

    def fingrid_data(self) -> fingrid.FingridData:
        if "fingrid" not in self._memo:
            if not self._fingrid_cache_complete():
                if self.synthetic:
                    self.ingest_fingrid()
                    return self._memo["fingrid"]
                if self.offline or not self._fingrid_key_available():
                    raise Unavailable([i for i in self.preflight() if i["input"].startswith("Fingrid")] or
                                      [{"input": "Fingrid API", "reason": "cache incomplete"}])
            client = None
            if not self.synthetic and not self.offline and self._fingrid_key_available():
                client = fingrid.FingridClient()
            need_client = client is not None and (self.refresh or not self._fingrid_cache_complete())
            data = fingrid.load_or_fetch_hourly(self.fg_cache, self.hours[0], self.hours[-1], client=client if need_client else None,
                                                refresh=self.refresh and client is not None)
            self._memo["fingrid"] = data
            verification = self._verification(client)
            self._save_info(fingrid={"verification": verification, "reports": data.reports, "cache_meta": data.cache_meta,
                                     "series": {c: fingrid.EXPECTED_UNITS[i] for i, c in fingrid.DATASETS.items()},
                                     "hourly_rule": fingrid.HOURLY_RULE, "from_cache_only": data.from_cache_only})
        return self._memo["fingrid"]

    # --- joined table + calibration ---------------------------------------------------------------------------
    def joined(self) -> "tuple[pd.DataFrame, dict[str, Any]]":
        if "joined" not in self._memo:
            sim = self.sim()
            self._memo["joined"] = fingrid.join_hourly(sim.theoretical_mw, self.fingrid_data().hourly, sim.installed_mw, era5_hours=self.hours)
        return self._memo["joined"]

    def calibration(self) -> K.CalibrationResult:
        if "calib" not in self._memo:
            joined, _ = self.joined()
            self._memo["calib"] = K.run_calibration(joined, self.sim().installed_mw, synthetic=self.synthetic)
        return self._memo["calib"]

    def calibrate(self) -> None:
        joined, counts = self.joined()
        calib = self.calibration()
        daily = S.daily_summary(joined)
        self._csv(joined, "hourly_joined.csv", index=True)
        self._csv(daily, "daily.csv", index=True)
        self._csv(calib.table, "calibration_hourly.csv", index=True)
        self._csv(calib.blocks, "split_blocks.csv")
        self._csv(validate.metrics_frame(calib.metrics), "metrics.csv")
        self._json(calib.metrics, "metrics.json")
        self._json({"frontier": calib.models.frontier.as_dict(), "anchor_train_hours": calib.models.frontier.n_anchors, "bounds_ok": calib.models.frontier.within_bounds,
                    "forward_check": {k: v for k, v in calib.forward.items() if k != "metrics"}, "counts": calib.counts, "notes": calib.notes}, "calibration_summary.json")
        self._save_info(calibration={"counts": calib.counts, "notes": calib.notes, "forward_status": calib.forward["status"],
                                     "forward_reason": calib.forward.get("reason"), "join_counts": counts})
        self.say(f"[calibrate] {counts['comparable_hours']}/{counts['era5_hours']} comparable hours; train {calib.counts['train_hours']} / test {calib.counts['test_hours']}; "
                 f"forward check {calib.forward['status']}")

    # --- validation + manifest ------------------------------------------------------------------------------------
    def validate(self) -> list[validate.CheckResult]:
        ds, r, sim = self.era5(), self.fleet_result(), self.sim()
        joined, counts = self.joined()
        calib = self.calibration()
        daily = S.daily_summary(joined)
        ev = validate.Evidence(
            era5_index=pd.DatetimeIndex(ds["valid_time"].values), expected_hours=len(self.hours), sim=sim, joined=joined, join_counts=counts,
            capacity_matrix=r.capacity_matrix, turbines=r.turbines, lats=ds.latitude.values, lons=ds.longitude.values, totals=r.totals,
            grid_summary_mw=float(r.pixels["installed_capacity_mw"].sum()), specs_mw=float(r.specs["cap_mw"].sum()), calib=calib, daily=daily)
        results = validate.run_all_checks(ev)
        failed = validate.failed_checks(results)
        stats = era5.summary_stats(ds)
        observed = {"era5_hours": len(ds["valid_time"]), "turbines": r.stats["total_turbines"], "osm_kept": r.stats["osm"]["kept"],
                    "arcgis_kept": r.stats["arcgis"]["kept"], "capacity_mw": r.totals["turbine_table_mw"], "active_cells": r.totals["active_cells"],
                    "year_coverage_pct": r.stats["year_coverage_pct"], **stats}
        recon = validate.full_reconciliation(observed, calib, self.synthetic)
        self._csv(validate.metrics_frame(calib.metrics), "metrics.csv")
        self._json(calib.metrics, "metrics.json")
        self._csv(recon, "reconciliation.csv")
        self._json(recon.to_dict(orient="records"), "reconciliation.json")
        self._csv(validate.results_frame(results), "validation.csv")
        self._json({"summary": {"passed": sum(x.status == validate.PASS for x in results), "failed": len(failed),
                                "skipped": sum(x.status == validate.SKIPPED for x in results)},
                    "synthetic": self.synthetic, "results": [x.as_dict() for x in results]}, "validation.json")
        self._save_info(time_coverage={"first_utc": iso_utc(self.hours[0]), "last_utc": iso_utc(self.hours[-1]), "expected_hours": len(self.hours),
                                       "present_hours": len(ds["valid_time"]), "comparable_hours": counts["comparable_hours"],
                                       "missing_hours": {k: v["missing_hours"] for k, v in self.fingrid_data().reports.items()}},
                        unavailable=self._unavailable_report())
        self.write_manifest()
        for x in results:
            self.say(f"[validate] {x.status.upper():7s} {x.name}: {x.detail}")
        return results

    def write_manifest(self) -> dict[str, Any]:
        return manifest.write_manifest(self.mode, self.synthetic, self.info, self.out)

    def write_unavailable(self, items: "list[dict[str, str]]") -> None:
        self.info["unavailable_runtime"] = items
        manifest.unavailable_manifest(self.mode, items, self.out)

    # --- site artifact -----------------------------------------------------------------------------------------------
    def export_site(self) -> dict[str, Any]:
        fl = self.info.get("fleet", {})
        site = export_site.export_site(
            self.out, ds=self.era5(), sim=self.sim(), joined=self.joined()[0], fleet=self.fleet_result(), calib=self.calibration(),
            run_mode=self.mode, synthetic=self.synthetic,
            provenance={"era5": self.info.get("era5", {}).get("source"), "fleet_snapshot": fl.get("snapshot"),
                        "fingrid_verification": self.info.get("fingrid", {}).get("verification", {}).get("overall")},
            arcgis_meta=fl.get("arcgis_meta"))
        self.write_manifest()
        self.say(f"[export-site] {self.out / 'site' / 'v1'}: {site['total_bytes'] / 1e6:.2f} MB, {len(site['files'])} files ({site['label']})")
        return site

    # --- everything ----------------------------------------------------------------------------------------------------
    def run_all(self) -> list[validate.CheckResult]:
        items = self.preflight()
        if items:
            raise Unavailable(items)
        self.ingest_era5()
        self.ingest_fleet()
        self.fleet()
        self.simulate()
        self.ingest_fingrid()
        self.calibrate()
        results = self.validate()
        self.export_site()
        return results

    def run_stage(self, stage: str) -> "list[validate.CheckResult] | None":
        table: Mapping[str, Callable[[], Any]] = {
            "ingest-era5": self.ingest_era5, "ingest-fleet": self.ingest_fleet, "ingest-fingrid": self.ingest_fingrid,
            "fleet": self.fleet, "simulate": self.simulate, "calibrate": self.calibrate, "validate": self.validate,
            "export-site": self.export_site}
        out = table[stage]()
        return out if stage == "validate" else None


def manifest_hash(hashes: "list[str]") -> str:
    import hashlib

    return hashlib.sha256("".join(sorted(hashes)).encode()).hexdigest()
