"""Fingrid open-data client: paging, rate limits, atomic cache, hourly resampling, joins.

The API key is read only from the FINGRID_API_KEY environment variable or an untracked `.env`;
it is never logged, cached or written to the manifest. Everything downstream of the raw cache
runs offline.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import REPO_ROOT
from .io_utils import (
    CredentialUnavailableError,
    atomic_write_json,
    atomic_write_with,
    iso_utc,
    to_utc_naive_index,
    utc_hour_range,
    utc_now_iso,
)

ENV_KEY = "FINGRID_API_KEY"
BASE_URL = "https://data.fingrid.fi/api/datasets/{dataset_id}/data"
META_URL = "https://data.fingrid.fi/api/datasets/{dataset_id}"
DATASETS: dict[int, str] = {
    181: "wind_power_mw",
    248: "solar_power_mw",
    124: "total_consumption_mw",
    319: "imbalance_price_eur_mwh",
}
EXPECTED_UNITS: dict[int, str] = {181: "MW", 248: "MW", 124: "MW", 319: "EUR/MWh"}
# Fingrid's real definition of 248 is "MWh/h": a per-hour energy rate, numerically the hourly mean MW. It is accepted next to "MW".
UNIT_ALIASES: dict[int, tuple[str, ...]] = {248: ("MWh/h",)}
PAGE_SIZE = 10_000
REQUEST_DELAY_SEC = 6.5
MAX_ATTEMPTS = 5
DEFAULT_RETRY_AFTER_SEC = 60
COMPLETE_TOLERANCE = pd.Timedelta(minutes=15)
HOURLY_RULE = "mean of all samples whose startTime lies in [h, h+1h), labelled h (UTC); empty hours stay NaN (no imputation)"
CACHE_SCHEMA = 1

Sleep = Callable[[float], None]


class FingridError(RuntimeError):
    """Transport/API failure that is not a missing credential."""


class DatasetMismatchError(FingridError):
    """Dataset metadata disagrees with the expected unit; `records` holds the manifest block."""

    def __init__(self, message: str, records: "list[dict[str, Any]]"):
        super().__init__(message)
        self.records = records


# --- credentials -------------------------------------------------------------
def _read_dotenv_value(path: Path, name: str) -> "str | None":
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{name}="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            return value or None
    return None


def get_api_key(env: "Mapping[str, str] | None" = None, dotenv: "Path | str | None" = None) -> str:
    """Key from the environment, else from an untracked `.env`; raises if neither provides one."""
    env = os.environ if env is None else env
    key = (env.get(ENV_KEY) or "").strip()
    if not key:
        key = _read_dotenv_value(Path(dotenv) if dotenv else REPO_ROOT / ".env", ENV_KEY) or ""
    if not key:
        raise CredentialUnavailableError(
            f"Fingrid credential unavailable: set {ENV_KEY} in the environment or an untracked .env "
            "(or run from a complete cache)"
        )
    return key


def window_for_hours(first_hour: Any, last_hour: Any) -> "tuple[str, str]":
    """Request window [first hour start, last hour end] as ISO-8601 Z strings."""
    start = to_utc_naive_index([first_hour])[0]
    end = to_utc_naive_index([last_hour])[0] + pd.Timedelta(minutes=59, seconds=59)
    return iso_utc(start), iso_utc(end)


# --- client ------------------------------------------------------------------
class FingridClient:
    """Thin client; `session` must offer requests.Session-like `get(url, headers=, params=, timeout=)`."""

    def __init__(
        self,
        api_key: "str | None" = None,
        session: Any = None,
        *,
        env: "Mapping[str, str] | None" = None,
        dotenv: "Path | str | None" = None,
        request_delay: float = REQUEST_DELAY_SEC,
        max_attempts: int = MAX_ATTEMPTS,
        page_size: int = PAGE_SIZE,
        backoff: float = 10.0,
        timeout: float = 60.0,
        sleep: Sleep = time.sleep,
        now: Callable[[], str] = utc_now_iso,
    ):
        self._key = api_key or get_api_key(env, dotenv)
        if session is None:
            import requests  # lazy so offline code paths never need it

            session = requests.Session()
        self.session = session
        self.request_delay, self.max_attempts, self.page_size = request_delay, max_attempts, page_size
        self.backoff, self.timeout, self.sleep, self.now = backoff, timeout, sleep, now
        self.request_count = 0

    def __repr__(self) -> str:  # never expose the key
        return "FingridClient(api_key=<hidden>)"

    # one HTTP GET with 429/Retry-After handling and bounded retries
    def _get(self, url: str, params: "Mapping[str, Any] | None", what: str) -> Any:
        headers = {"x-api-key": self._key, "Accept": "application/json"}
        last: "Exception | None" = None
        for attempt in range(1, self.max_attempts + 1):
            self.request_count += 1
            try:
                resp = self.session.get(url, headers=headers, params=params, timeout=self.timeout)
            except Exception as exc:
                last = exc
            else:
                status = getattr(resp, "status_code", 200)
                if status == 200:
                    return resp.json()
                if status in (401, 403):
                    raise FingridError(f"{what}: HTTP {status} - API key rejected (check {ENV_KEY})")
                if status == 429:
                    wait = _retry_after(getattr(resp, "headers", {}) or {})
                    last = FingridError(f"HTTP 429 (retry after {wait:g}s)")
                    if attempt < self.max_attempts:
                        self.sleep(wait)
                    continue
                last = FingridError(f"HTTP {status}")
            if attempt < self.max_attempts:
                self.sleep(self.backoff * attempt)
        raise FingridError(f"{what} failed after {self.max_attempts} attempts: {last}")

    def fetch_metadata(self, dataset_id: int) -> dict[str, Any]:
        return self._get(META_URL.format(dataset_id=dataset_id), None, f"metadata {dataset_id}")

    def fetch_rows(self, dataset_id: int, start: str, end: str) -> "tuple[list[dict[str, Any]], int]":
        """Page through [start, end]; start a new window after the last row when the server stops paging.

        Returns (rows, pages_fetched).
        """
        url = BASE_URL.format(dataset_id=dataset_id)
        end_ts = pd.Timestamp(end)
        fetch_start = start
        rows: list[dict[str, Any]] = []
        pages = 0
        while pd.Timestamp(fetch_start) < end_ts:
            window_rows: list[dict[str, Any]] = []
            page = 1
            while True:
                params = {"startTime": fetch_start, "endTime": end, "format": "json", "page": page,
                          "pageSize": self.page_size, "sortBy": "startTime", "sortOrder": "asc"}
                payload = self._get(url, params, f"dataset {dataset_id} page {page}")
                pages += 1
                data = payload.get("data", [])
                pagination = payload.get("pagination") or {}
                last_page, next_page = pagination.get("lastPage"), pagination.get("nextPage")
                if not data:
                    break
                window_rows.extend(data)
                if next_page is None or (last_page is not None and page >= last_page):
                    break
                page += 1
                self.sleep(self.request_delay)
            if not window_rows:
                break
            rows.extend(window_rows)
            last_ts = _utc(window_rows[-1]["startTime"])
            next_start = last_ts + pd.Timedelta(seconds=1)
            if next_start <= _utc(fetch_start):
                break
            fetch_start = next_start.strftime("%Y-%m-%dT%H:%M:%SZ")
            self.sleep(self.request_delay)
        return rows, pages

    def fetch_dataset(self, dataset_id: int, start: str, end: str, cache_dir: "str | Path", *, refresh: bool = False,
                      column: "str | None" = None) -> "tuple[pd.DataFrame, dict[str, Any]]":
        """Fetch one series into the cache, resuming from a partial cache; returns (raw rows, metadata)."""
        col = column or DATASETS.get(dataset_id, f"dataset_{dataset_id}")
        cached, meta = read_raw_cache(cache_dir, dataset_id, col)
        if refresh or cached is None or not _covers_start(meta, start):
            cached, meta = None, None
        fetch_start, pages, requests_before = start, 0, self.request_count
        if cached is not None:
            if is_cache_complete(cached, end):
                return cached, meta  # type: ignore[return-value]
            fetch_start = (_start_times(cached).max() + pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        new_rows, pages = self.fetch_rows(dataset_id, fetch_start, end)
        frames = [f for f in (cached, pd.DataFrame(new_rows)) if f is not None and not f.empty]
        if not frames:
            raise FingridError(f"dataset {dataset_id} ({col}) returned 0 rows for {start}..{end}")
        raw = pd.concat(frames, ignore_index=True)
        dedup = [c for c in ("startTime", "endTime") if c in raw.columns]
        raw = raw.drop_duplicates(subset=dedup).assign(_t=lambda d: _start_times(d)).sort_values("_t").drop(columns="_t")
        raw = raw.reset_index(drop=True)
        meta = _build_meta(dataset_id, col, raw, start, end, self.now(), (meta or {}).get("pages_fetched", 0) + pages,
                           (meta or {}).get("requests", 0) + self.request_count - requests_before)
        write_raw_cache(cache_dir, dataset_id, col, raw, meta)
        return raw, meta


def _utc(value: Any) -> pd.Timestamp:
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _retry_after(headers: Mapping[str, Any]) -> float:
    for k, v in headers.items():
        if str(k).lower() == "retry-after":
            try:
                return max(0.0, float(v))
            except (TypeError, ValueError):
                break
    return float(DEFAULT_RETRY_AFTER_SEC)


# --- cache -------------------------------------------------------------------
def _cache_paths(cache_dir: "str | Path", dataset_id: int, column: str) -> "tuple[Path, Path]":
    base = Path(cache_dir) / f"fingrid_raw_{dataset_id}_{column}"
    return base.with_suffix(".csv"), base.with_suffix(".meta.json")


def _start_times(raw: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(raw["startTime"], utc=True, format="ISO8601")


def _build_meta(dataset_id: int, column: str, raw: pd.DataFrame, start: str, end: str, fetched_at: str,
                pages: int, requests: int) -> dict[str, Any]:
    t = _start_times(raw)
    return {
        "schema_version": CACHE_SCHEMA, "dataset_id": dataset_id, "column": column,
        "source_url": BASE_URL.format(dataset_id=dataset_id), "fetched_at_utc": fetched_at,
        "window_start": start, "window_end": end, "row_count": int(len(raw)),
        "first_start_utc": t.min().strftime("%Y-%m-%dT%H:%M:%SZ"), "last_start_utc": t.max().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pages_fetched": pages, "requests": requests, "complete": bool(t.max() >= pd.Timestamp(end) - COMPLETE_TOLERANCE),
    }


def write_raw_cache(cache_dir: "str | Path", dataset_id: int, column: str, raw: pd.DataFrame, meta: Mapping[str, Any]) -> None:
    csv_path, meta_path = _cache_paths(cache_dir, dataset_id, column)
    atomic_write_with(csv_path, lambda tmp: raw.to_csv(tmp, index=False))
    atomic_write_json(meta_path, dict(meta))  # metadata last: a metadata file implies a complete CSV


def read_raw_cache(cache_dir: "str | Path", dataset_id: int, column: "str | None" = None) -> "tuple[pd.DataFrame | None, dict[str, Any] | None]":
    col = column or DATASETS.get(dataset_id, f"dataset_{dataset_id}")
    csv_path, meta_path = _cache_paths(cache_dir, dataset_id, col)
    if not (csv_path.is_file() and meta_path.is_file()):
        return None, None
    raw = pd.read_csv(csv_path)
    if raw.empty or "startTime" not in raw.columns:
        return None, None
    return raw, json.loads(meta_path.read_text(encoding="utf-8"))


def _covers_start(meta: "Mapping[str, Any] | None", start: str) -> bool:
    return meta is not None and pd.Timestamp(meta["window_start"]) <= pd.Timestamp(start)


def is_cache_complete(raw: pd.DataFrame, end: str) -> bool:
    return bool(_start_times(raw).max() >= pd.Timestamp(end) - COMPLETE_TOLERANCE)


# --- hourly resampling and missing-hour report ---------------------------------
def to_hourly(raw: pd.DataFrame, column: str, first_hour: Any, last_hour: Any, dataset_id: "int | None" = None
              ) -> "tuple[pd.Series, dict[str, Any]]":
    """Hourly UTC series over [first_hour, last_hour] plus a per-series report.

    Samples are grouped by the hour containing their startTime and averaged (HOURLY_RULE). Hours
    without samples stay NaN and are listed in the report; hours with fewer samples than the
    series' typical count are listed as `partial_hours` but still averaged.
    """
    hours = utc_hour_range(first_hour, last_hour)
    t = _start_times(raw).dt.tz_convert("UTC").dt.tz_localize(None)
    vals = pd.to_numeric(raw["value"], errors="coerce")
    s = pd.Series(vals.to_numpy(), index=pd.DatetimeIndex(t.to_numpy())).dropna().sort_index()
    s = s[~s.index.duplicated(keep="first")]
    in_window = s[(s.index >= hours[0]) & (s.index < hours[-1] + pd.Timedelta(hours=1))]
    grouped = in_window.groupby(in_window.index.floor("h"))
    hourly = grouped.mean().reindex(hours).rename(column)
    counts = grouped.size().reindex(hours).fillna(0).astype(int)
    hourly.index.name = "valid_time"
    steps = np.diff(in_window.index.values).astype("timedelta64[s]").astype(int) if len(in_window) > 1 else np.array([])
    res = int(np.median(steps)) if steps.size else None
    typical = int(counts[counts > 0].median()) if (counts > 0).any() else 0
    missing = hours[counts.to_numpy() == 0]
    partial = hours[(counts.to_numpy() > 0) & (counts.to_numpy() < typical)]
    report = {
        "series": column, "dataset_id": dataset_id, "aggregation": HOURLY_RULE, "resolution_seconds": res,
        "samples_per_hour_typical": typical, "expected_hours": len(hours), "present_hours": int((counts > 0).sum()),
        "missing_hours": [iso_utc(h) for h in missing], "partial_hours": [iso_utc(h) for h in partial],
    }
    return hourly, report


@dataclass
class FingridData:
    hourly: pd.DataFrame                      # index valid_time (UTC, tz-naive), columns DATASETS values + net_load_mw
    reports: dict[str, dict[str, Any]]        # per-series missing-hour reports
    cache_meta: dict[str, dict[str, Any]]     # per-series cache metadata (fetched_at, window, row counts, dataset id)
    from_cache_only: bool = True


def build_hourly_table(series: Mapping[str, pd.Series]) -> pd.DataFrame:
    df = pd.concat(list(series.values()), axis=1)
    df["net_load_mw"] = df["total_consumption_mw"] - df["wind_power_mw"] - df["solar_power_mw"]
    return df


def load_or_fetch_hourly(
    cache_dir: "str | Path", first_hour: Any, last_hour: Any, *, client: "FingridClient | None" = None,
    refresh: bool = False, datasets: Mapping[int, str] = DATASETS,
) -> FingridData:
    """Hourly table for the ERA5 window from the cache; fetches only what is missing and only with a client.

    Without a client and with an incomplete cache this raises CredentialUnavailableError naming the series.
    """
    start, end = window_for_hours(first_hour, last_hour)
    series: dict[str, pd.Series] = {}
    reports: dict[str, dict[str, Any]] = {}
    metas: dict[str, dict[str, Any]] = {}
    used_network = False
    for ds_id, col in datasets.items():
        raw, meta = read_raw_cache(cache_dir, ds_id, col)
        usable = raw is not None and not refresh and _covers_start(meta, start) and is_cache_complete(raw, end)
        if not usable:
            if client is None:
                raise CredentialUnavailableError(
                    f"Fingrid cache for dataset {ds_id} ({col}) is missing or incomplete for {start}..{end} and no "
                    f"{ENV_KEY} is available to fetch it"
                )
            raw, meta = client.fetch_dataset(ds_id, start, end, cache_dir, refresh=refresh, column=col)
            used_network = True
        series[col], reports[col] = to_hourly(raw, col, first_hour, last_hour, ds_id)  # type: ignore[arg-type]
        metas[col] = dict(meta or {})
    return FingridData(build_hourly_table(series), reports, metas, from_cache_only=not used_network)


# --- dataset definition / units verification ------------------------------------
def _first_text(meta: Mapping[str, Any], keys: Sequence[str]) -> "str | None":
    for k in keys:
        v = meta.get(k)
        if isinstance(v, dict):
            v = v.get("en") or next(iter(v.values()), None)
        if v not in (None, ""):
            return str(v)
    return None


def normalise_unit(unit: "str | None") -> "str | None":
    if unit is None:
        return None
    u = unit.strip().lower().replace("€", "eur").replace("euro", "eur")
    return re.sub(r"[\s_]+", "", u)


def verify_datasets(
    client: "FingridClient | None", datasets: Mapping[int, str] = DATASETS,
    expected: Mapping[int, str] = EXPECTED_UNITS, *, strict: bool = True,
) -> dict[str, Any]:
    """Fetch /api/datasets/{id} and compare unit with expectations; returns the manifest block.

    Statuses per dataset: `verified`, `mismatch` (raises DatasetMismatchError when strict),
    `unit_not_reported` and `unverified` (no client/credential). Tolerant of field-name variants.
    """
    records: list[dict[str, Any]] = []
    for ds_id, col in datasets.items():
        rec: dict[str, Any] = {"dataset_id": ds_id, "series": col, "expected_unit": expected.get(ds_id)}
        if client is None:
            rec.update(status="unverified", reason=f"no {ENV_KEY}; dataset metadata not fetched")
        else:
            meta = client.fetch_metadata(ds_id)
            unit = _first_text(meta, ("unitEn", "unit", "unitFi"))
            rec.update(
                name=_first_text(meta, ("nameEn", "name", "title", "nameFi")),
                reported_unit=unit,
                frequency=_first_text(meta, ("dataPeriodEn", "dataPeriod", "frequency", "resolution", "updateCadenceEn")),
                license=_first_text(meta, ("license", "licenseEn", "licence")),
                retrieved_utc=client.now(),
            )
            if unit is None:
                rec.update(status="unit_not_reported", reason="metadata has no unit field")
            elif normalise_unit(unit) == normalise_unit(expected.get(ds_id)):
                rec["status"] = "verified"
            elif normalise_unit(unit) in {normalise_unit(a) for a in UNIT_ALIASES.get(ds_id, ())}:
                rec.update(status="verified", note=f"{unit!r} is accepted as equivalent to {expected.get(ds_id)!r} for dataset {ds_id} "
                                                  "(energy per hour = mean power)")
            else:
                rec.update(status="mismatch", reason=f"reported unit {unit!r} != expected {expected.get(ds_id)!r}")
        records.append(rec)
    bad = [r for r in records if r["status"] == "mismatch"]
    if bad and strict:
        raise DatasetMismatchError("; ".join(f"dataset {r['dataset_id']}: {r['reason']}" for r in bad), records)
    statuses = {r["status"] for r in records}
    overall = "verified" if statuses == {"verified"} else ("unverified" if statuses == {"unverified"} else "partial")
    return {"overall": overall, "datasets": records}


# --- joined hourly table ---------------------------------------------------------
def join_hourly(theoretical: pd.Series, fingrid: pd.DataFrame, installed_mw: float,
                era5_hours: "pd.DatetimeIndex | None" = None) -> "tuple[pd.DataFrame, dict[str, Any]]":
    """Left join on the ERA5 hours: every ERA5 hour is kept, with NaN where Fingrid has no value.

    Columns follow notebook 1: theoretical_wind_mw, actual_wind_mw, consumption/solar/price,
    theoretical_cf, actual_cf, residual_mw (theoretical - actual), residual_cf.
    """
    theo = pd.Series(theoretical.to_numpy(), index=to_utc_naive_index(theoretical.index), name="theoretical_wind_mw")
    theo.index.name = "valid_time"
    hours = theo.index if era5_hours is None else to_utc_naive_index(era5_hours)
    fg = fingrid.copy()
    fg.index = to_utc_naive_index(fg.index)
    keep = [c for c in ("wind_power_mw", "total_consumption_mw", "solar_power_mw", "imbalance_price_eur_mwh", "net_load_mw") if c in fg]
    df = pd.DataFrame({"theoretical_wind_mw": theo.reindex(hours)}).join(fg[keep], how="left")
    df = df.rename(columns={"wind_power_mw": "actual_wind_mw"})
    df["theoretical_cf"] = df["theoretical_wind_mw"] / installed_mw
    df["actual_cf"] = df["actual_wind_mw"] / installed_mw
    df["residual_mw"] = df["theoretical_wind_mw"] - df["actual_wind_mw"]
    df["residual_cf"] = df["theoretical_cf"] - df["actual_cf"]
    n_comparable = int(df["actual_wind_mw"].notna().sum())
    counts = {
        "era5_hours": int(len(hours)), "joined_rows": int(len(df)), "comparable_hours": n_comparable,
        "dropped_missing_actual": int(len(df) - n_comparable),
        "join_coverage": n_comparable / len(hours) if len(hours) else 0.0,
    }
    return df, counts


def comparable_rows(df: pd.DataFrame, required: Sequence[str] = ("theoretical_wind_mw", "actual_wind_mw")) -> pd.DataFrame:
    """Rows usable for metrics: every `required` column present (missing hours are dropped, not imputed)."""
    return df.dropna(subset=list(required))
