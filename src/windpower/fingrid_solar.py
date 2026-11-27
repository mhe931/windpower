"""Fingrid solar series for the solar report: 267 capacity, 248 forecast, 362 small-scale surplus (AV06 = solar).

Reuses the wind client (`fingrid.FingridClient`: key from FINGRID_API_KEY / untracked .env only, paging, 429 handling) and
its raw-cache helpers; the wind module and its datasets are untouched. Definitions (lead-verified 2026-11-27):

* 267  Total production capacity used in the solar power forecast - MW, 1 h, not official capacity (manual estimates)
* 248  Solar power generation forecast, updated every 15 minutes - MWh/h, 15 min, a FORECAST (72 h ahead), not measured
* 362  Small-scale electricity surplus production by production type (Datahub) - kWh, 1 h, one row per production type per
       hour; `additionalJson.ProductionType == "AV06"` is solar; surplus only (after own consumption), not total generation

Native units stay in the raw cache and the tables; MW derivations are explicit (kWh / 1000 per 1 h; MWh/h = MW).
Missing hours stay NaN and are listed - nothing is filled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import fingrid as FG
from .io_utils import utc_hour_range

SOLAR_PRODUCTION_TYPE = "AV06"


@dataclass(frozen=True)
class SeriesDef:
    dataset_id: int
    key: str
    name: str                 # expected dataset name (for the definition check)
    native_unit: str
    derived_unit: str
    period: str
    period_seconds: int
    time_aggregation: str
    scope: str


SERIES: dict[int, SeriesDef] = {
    267: SeriesDef(267, "capacity_267_mw", "Total production capacity used in the solar power forecast", "MW", "MW", "1 h", 3600,
                   "mean of the samples in the hour (hourly data: the value itself)",
                   "capacity used in Fingrid's solar forecast (grid-scale + small-scale); updated manually with estimates, not official capacity"),
    248: SeriesDef(248, "forecast_248_mw", "Solar power generation forecast - updated every 15 minutes", "MWh/h", "MW", "15 min", 900,
                   "hourly mean of the four 15-min MWh/h values = MW (negative night values kept)",
                   "forecast, not measured generation (based on weather forecasts and rough capacity-location estimates)"),
    362: SeriesDef(362, "surplus_362", "Small-scale electricity surplus production by production type at accounting points in Finnish distribution networks",
                   "kWh", "MW", "1 h", 3600, f"rows with additionalJson.ProductionType == {SOLAR_PRODUCTION_TYPE}; kWh per 1 h hour; MW = kWh / 1000",
                   "small-scale surplus only (after own consumption), not total generation; may include partly estimated metering; latest data lags ~4 days"),
}
EXPECTED_UNITS = {i: s.native_unit for i, s in SERIES.items()}
DATASET_COLUMNS = {i: s.key for i, s in SERIES.items()}
RAW_COLUMN = {267: "capacity_267", 248: "forecast_248", 362: "surplus_362_all_types"}


# --- raw rows ---------------------------------------------------------------------------------------------------------------
def _additional(row: Mapping[str, Any]) -> dict[str, Any]:
    a = row.get("additionalJson")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except json.JSONDecodeError:
            a = None
    return a if isinstance(a, dict) else {}


def _number(v: Any) -> float:
    if v is None:
        return float("nan")
    try:
        return float(str(v).replace(",", ".").strip())
    except ValueError:
        return float("nan")


def flatten_rows(rows: "list[dict[str, Any]]") -> pd.DataFrame:
    """API rows -> flat frame; for 362-style rows the additionalJson members become production_type/uom/res/count columns.

    `value` is the row's top-level value; when additionalJson also carries `Value` and it disagrees the row is flagged
    (`value_conflict`) so that the hourly step can fail loudly instead of guessing.
    """
    out = []
    for r in rows:
        a = _additional(r)
        v, av = _number(r.get("value")), _number(a.get("Value"))
        rec = {"datasetId": r.get("datasetId"), "startTime": r.get("startTime"), "endTime": r.get("endTime"), "value": v if np.isfinite(v) else av}
        if a:
            rec.update(production_type=a.get("ProductionType"), uom=a.get("Uom"), res=a.get("Res"), count=_number(a.get("Count")),
                       value_conflict=bool(np.isfinite(v) and np.isfinite(av) and abs(v - av) > 1e-9))
        out.append(rec)
    return pd.DataFrame(out)


def solar_rows_362(raw: pd.DataFrame) -> "tuple[pd.DataFrame, dict[str, Any]]":
    """The AV06 rows of dataset 362 (filtered on production type, never on value) plus a report of what was seen."""
    if "production_type" not in raw.columns:
        raise FG.FingridError("dataset 362 rows carry no additionalJson.ProductionType; cannot select solar (AV06)")
    types = raw["production_type"].astype(str).value_counts().to_dict()
    sol = raw[raw["production_type"].astype(str) == SOLAR_PRODUCTION_TYPE].copy()
    if sol.empty:
        raise FG.FingridError(f"dataset 362 has no {SOLAR_PRODUCTION_TYPE} rows; production types seen: {sorted(types)}")
    if "value_conflict" in sol and sol["value_conflict"].astype(bool).any():
        raise FG.FingridError(f"dataset 362 {SOLAR_PRODUCTION_TYPE}: top-level value and additionalJson.Value disagree in "
                              f"{int(sol['value_conflict'].astype(bool).sum())} row(s); which one is kWh per hour must be clarified")
    uoms = sorted({str(u).upper() for u in sol["uom"].dropna()})
    res = sorted({str(u).upper() for u in sol["res"].dropna()})
    if uoms and uoms != ["KWH"]:
        raise FG.FingridError(f"dataset 362 {SOLAR_PRODUCTION_TYPE}: unit(s) {uoms} != KWH")
    if res and res != ["PT1H"]:
        raise FG.FingridError(f"dataset 362 {SOLAR_PRODUCTION_TYPE}: resolution(s) {res} != PT1H")
    t = FG._start_times(sol)
    if t.duplicated().any():
        raise FG.FingridError(f"dataset 362 {SOLAR_PRODUCTION_TYPE}: {int(t.duplicated().sum())} duplicate hour(s); several rows per hour are ambiguous")
    return sol, {"production_types_seen": types, "solar_rows": int(len(sol)), "uom": uoms, "res": res}


# --- hourly tables -----------------------------------------------------------------------------------------------------------------
def hourly_series(ds_id: int, raw: pd.DataFrame, first_hour: Any, last_hour: Any) -> "tuple[pd.DataFrame, dict[str, Any]]":
    """Hourly columns for one dataset (native + derived units) and its report (missing hours, minimum, counts)."""
    d = SERIES[ds_id]
    d_id = ds_id
    if d_id == 362:
        sol, seen = solar_rows_362(raw)
        kwh, rep = FG.to_hourly(sol[["startTime", "value"]], "surplus_362_kwh", first_hour, last_hour, 362)
        cnt, _ = FG.to_hourly(sol[["startTime"]].assign(value=sol["count"].to_numpy()), "surplus_362_count", first_hour, last_hour, 362)
        df = pd.DataFrame({"surplus_362_kwh": kwh, "surplus_362_mw": kwh / 1000.0, "surplus_362_count": cnt})
        rep.update(seen, derived="MW = kWh / 1000 for each 1 h hour")
    else:
        col = d.key
        s, rep = FG.to_hourly(raw, col, first_hour, last_hour, d_id)
        df = pd.DataFrame({col: s})
    vals = df.iloc[:, 0]
    rep.update(native_unit=d.native_unit, derived_unit=d.derived_unit, min=None if vals.dropna().empty else float(vals.min()),
               max=None if vals.dropna().empty else float(vals.max()), n_negative=int((vals < 0).sum()), scope=d.scope)
    return df, rep


def series_record(ds_id: int) -> dict[str, Any]:
    d = SERIES[ds_id]
    return {"id": d.dataset_id, "name": d.name, "native_unit": d.native_unit, "derived_unit": d.derived_unit, "period": d.period,
            "time_aggregation": d.time_aggregation, "scope": d.scope, "key": d.key}


# --- cache / fetch -------------------------------------------------------------------------------------------------------------------
def _cache_complete(raw: "pd.DataFrame | None", meta: "Mapping[str, Any] | None", start: str, end: str, step_s: int) -> bool:
    if raw is None or meta is None or not FG._covers_start(meta, start):
        return False
    return bool(FG._start_times(raw).max() >= pd.Timestamp(end) - pd.Timedelta(seconds=step_s))


def fetch_raw(client: FG.FingridClient, ds_id: int, start: str, end: str, cache_dir: "str | Path", refresh: bool = False
              ) -> "tuple[pd.DataFrame, dict[str, Any]]":
    """Fetch one dataset window (paged by the wind client) into the generic raw cache; reuses a complete cache."""
    col = RAW_COLUMN[ds_id]
    raw, meta = FG.read_raw_cache(cache_dir, ds_id, col)
    if not refresh and _cache_complete(raw, meta, start, end, SERIES[ds_id].period_seconds):
        return raw, meta  # type: ignore[return-value]
    rows, pages = client.fetch_rows(ds_id, start, end)
    if not rows:
        raise FG.FingridError(f"dataset {ds_id} returned 0 rows for {start}..{end}")
    frame = flatten_rows(rows)
    frame = frame.drop_duplicates(subset=[c for c in ("startTime", "production_type") if c in frame.columns]).reset_index(drop=True)
    meta = FG._build_meta(ds_id, col, frame, start, end, client.now(), pages, pages) | {"series_key": SERIES[ds_id].key, "native_unit": SERIES[ds_id].native_unit}
    FG.write_raw_cache(cache_dir, ds_id, col, frame, meta)
    return frame, meta


@dataclass
class SolarFingridData:
    hourly: pd.DataFrame                         # index valid_time (UTC naive); only the available series' columns
    reports: dict[int, dict[str, Any]]
    cache_meta: dict[int, dict[str, Any]]
    unavailable: dict[int, str]                  # dataset id -> exact reason (e.g. missing FINGRID_API_KEY)
    from_cache_only: bool = True


def load_or_fetch_solar(cache_dir: "str | Path", first_hour: Any, last_hour: Any, *, client: "FG.FingridClient | None" = None,
                        refresh: bool = False, key_missing_reason: "str | None" = None) -> SolarFingridData:
    """Hourly 267/248/362(AV06) for the window. A series that cannot be had (no cache, no client) is reported in `unavailable`,
    naming the missing credential; the others are still returned. Network only with a client."""
    hours = utc_hour_range(first_hour, last_hour)
    start, end = FG.window_for_hours(first_hour, last_hour)
    frames: list[pd.DataFrame] = []
    reports: dict[int, dict[str, Any]] = {}
    metas: dict[int, dict[str, Any]] = {}
    unavailable: dict[int, str] = {}
    used_network = False
    for ds_id, d in SERIES.items():
        raw, meta = FG.read_raw_cache(cache_dir, ds_id, RAW_COLUMN[ds_id])
        usable = (not refresh) and _cache_complete(raw, meta, start, end, d.period_seconds)
        if not usable:
            if client is None:
                unavailable[ds_id] = key_missing_reason or (f"cache for dataset {ds_id} missing or incomplete for {start}..{end} and no "
                                                           f"{FG.ENV_KEY} available to fetch it (set {FG.ENV_KEY} in the environment or an untracked .env)")
                continue
            raw, meta = fetch_raw(client, ds_id, start, end, cache_dir, refresh=refresh)
            used_network = True
        df, rep = hourly_series(ds_id, raw, first_hour, last_hour)  # type: ignore[arg-type]
        frames.append(df)
        reports[ds_id] = rep
        metas[ds_id] = dict(meta or {})
    hourly = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=hours)
    hourly.index.name = "valid_time"
    return SolarFingridData(hourly.reindex(hours), reports, metas, unavailable, from_cache_only=not used_network)


def verify_solar_datasets(client: "FG.FingridClient | None") -> dict[str, Any]:
    """Definition check via dataset metadata: unit mismatch raises DatasetMismatchError (as in wind); name/period are recorded and flagged."""
    block = FG.verify_datasets(client, datasets=DATASET_COLUMNS, expected=EXPECTED_UNITS)
    for rec in block["datasets"]:
        d = SERIES[rec["dataset_id"]]
        rec["expected_period"], rec["expected_name"] = d.period, d.name
        if rec["status"] in ("verified", "unit_not_reported", "mismatch"):
            freq = FG.normalise_unit(rec.get("frequency")) if rec.get("frequency") else None
            want = FG.normalise_unit(d.period)
            rec["period_status"] = "not_reported" if freq is None else ("verified" if want in freq or freq in want else "mismatch")
            nm = (rec.get("name") or "").lower()
            rec["name_status"] = "not_reported" if not nm else ("verified" if d.name.lower()[:30] in nm or nm in d.name.lower() else "differs")
    return block


def key_missing_message() -> str:
    return f"Fingrid credential unavailable: {FG.ENV_KEY} is not set in the environment or an untracked .env"
