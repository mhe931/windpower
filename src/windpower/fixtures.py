"""Deterministic SYNTHETIC fixtures: ERA5-like files, ArcGIS-like GeoJSON, Overpass-like JSON.

Nothing here resembles real measurements; every artifact carries the SYNTHETIC stamp.
Generated files must be written under git-ignored data/ or a pytest tmp dir.
"""

from __future__ import annotations

import math
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from . import config as C
from .io_utils import SYNTHETIC_STAMP, atomic_write_json, atomic_write_with

STAMP_TEXT = f"{SYNTHETIC_STAMP} fixture - generated offline, not real data"
FIXED_ISO = "2025-10-01T00:00:00Z"  # fixed so fixture snapshots are reproducible
_ZIP_DATE = (2025, 1, 1, 0, 0, 0)
_UNITS = {
    "u10": "m s**-1", "v10": "m s**-1", "u100": "m s**-1", "v100": "m s**-1",
    "t2m": "K", "d2m": "K", "sp": "Pa", "fsr": "m", "lcc": "(0 - 1)", "tp": "m",
}
_INSTANT = ("u10", "v10", "u100", "v100", "t2m", "d2m", "sp", "lcc")
_ACCUM = ("tp", "fsr")


# --- ERA5 ------------------------------------------------------------------
def _ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    out = np.zeros(n)
    shocks = rng.standard_normal(n) * math.sqrt(1 - phi**2)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + shocks[i]
    return out


def synthetic_era5(
    grid: C.GridSpec = C.SMOKE_GRID,
    start: str = "2025-07-01T00:00:00",
    hours: int = 48,
    seed: int = 2026,
    base_temp_c: float = 14.0,
    lat_descending: bool = True,
) -> xr.Dataset:
    """ERA5-like hourly dataset with CDS conventions (valid_time dim, descending latitude, float32).

    `base_temp_c` lets tests produce cold/humid hours that trigger the icing flag.
    """
    rng = np.random.default_rng(seed)
    lats = np.array(grid.lats())
    lons = np.array(grid.lons())
    nt, ny, nx = hours, lats.size, lons.size
    times = pd.date_range(pd.Timestamp(start), periods=nt, freq="h")
    hod = np.asarray(times.hour)[:, None, None]

    def national(phi: float) -> np.ndarray:
        return _ar1(rng, nt, phi)[:, None, None]

    gx = np.linspace(-1, 1, nx)[None, None, :]
    gy = np.linspace(-1, 1, ny)[None, :, None]
    shape = (nt, ny, nx)
    u100 = 5.0 + 2.5 * national(0.97) + 0.8 * gx + 0.8 * rng.standard_normal(shape)
    v100 = 1.0 + 2.0 * national(0.97) + 0.5 * gy + 0.8 * rng.standard_normal(shape)
    ratio = np.clip(0.62 + 0.06 * rng.standard_normal(shape), 0.45, 0.85)
    u10, v10 = u100 * ratio, v100 * ratio
    t_c = base_temp_c + 4.0 * np.sin(2 * np.pi * (hod - 9) / 24) + 2.0 * national(0.98) - 1.5 * gy + 0.3 * rng.standard_normal(shape)
    spread = 0.2 + rng.gamma(2.0, 0.9, shape)
    sp = 101000.0 + 700.0 * national(0.99) - 300.0 * gy + 20.0 * rng.standard_normal(shape)
    fsr_cell = np.exp(rng.normal(np.log(0.05), 1.0, (1, ny, nx)))
    fsr = np.clip(fsr_cell * (1.0 + 0.05 * rng.standard_normal(shape)), 1e-4, 1.5)
    lcc = np.clip(rng.beta(0.6, 0.8, shape), 0.0, 1.0)
    tp = np.where(rng.random(shape) < 0.15, rng.gamma(1.5, 0.0004, shape), 0.0)

    data = {
        "u10": u10, "v10": v10, "u100": u100, "v100": v100,
        "t2m": t_c + 273.15, "d2m": t_c + 273.15 - spread, "sp": sp,
        "fsr": fsr, "lcc": lcc, "tp": tp,
    }
    ds = xr.Dataset(
        {k: (("valid_time", "latitude", "longitude"), v.astype(np.float32), {"units": _UNITS[k]}) for k, v in data.items()},
        coords={"valid_time": times.values.astype("datetime64[ns]"), "latitude": lats, "longitude": lons},
        attrs={"title": f"{SYNTHETIC_STAMP} ERA5-like fixture", "synthetic": SYNTHETIC_STAMP, "seed": seed,
               "grid": grid.name, "note": STAMP_TEXT},
    )
    if lat_descending:
        ds = ds.isel(latitude=slice(None, None, -1))
    return ds


def write_netcdf(ds: xr.Dataset, path: "str | Path") -> Path:
    """Plain NetCDF4 file (a month of the fixture cache)."""
    return atomic_write_with(path, lambda tmp: ds.to_netcdf(tmp, engine="netcdf4"))


def write_cds_zip(ds: xr.Dataset, path: "str | Path") -> Path:
    """CDS-style bundle: a ZIP holding an `instant` and an `accum` NetCDF with scalar number/expver coords."""
    with tempfile.TemporaryDirectory(prefix="fixture_zip_") as tmp:
        members: list[tuple[str, bytes]] = []
        for stream, names in (("instant", _INSTANT), ("accum", _ACCUM)):
            part = ds[list(names)].assign_coords(number=0, expver="0001")
            p = Path(tmp) / f"{stream}.nc"
            part.to_netcdf(p, engine="netcdf4")
            members.append((f"data_stream-oper_stepType-{stream}.nc", p.read_bytes()))

        def build(out: Path) -> None:
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, blob in members:
                    zf.writestr(zipfile.ZipInfo(name, _ZIP_DATE), blob, zipfile.ZIP_DEFLATED)

        return atomic_write_with(path, build)


def write_era5_cache(
    cache_dir: "str | Path",
    grid: C.GridSpec = C.SMOKE_GRID,
    start: str = "2025-07-01T00:00:00",
    hours: int = 48,
    seed: int = 2026,
    zip_months: tuple[int, ...] = (1,),
    **kwargs: Any,
) -> list[Path]:
    """Write era5_YYYY_MM.nc per month; months at positions in `zip_months` become CDS ZIP bundles."""
    ds = synthetic_era5(grid, start, hours, seed, **kwargs)
    t = pd.DatetimeIndex(ds["valid_time"].values)
    paths: list[Path] = []
    for pos, ((year, month), _) in enumerate(pd.Series(t, index=t).groupby([t.year, t.month])):
        sel = np.flatnonzero((t.year == year) & (t.month == month))
        part = ds.isel(valid_time=sel)
        target = Path(cache_dir) / f"era5_{year}_{month:02d}.nc"
        (write_cds_zip if pos in zip_months else write_netcdf)(part, target)
        paths.append(target)
    return paths


# --- fleet -----------------------------------------------------------------
_M_PER_DEG = 111_320.0
# (commissioning year, capacity MW, rotor m, hub m, manufacturer, model)
_ERAS = [
    (2005, 1.8, 80.0, 75.0, "SYNTH-A", "SA-1800"),
    (2014, 3.2, 112.0, 115.0, "SYNTH-B", "SB-3200"),
    (2019, 4.3, 138.0, 138.0, "SYNTH-C", "SC-4300"),
    (2023, 5.6, 160.0, 152.0, "SYNTH-D", "SD-5600"),
]


def offset_deg(lat: float, lon: float, dx_m: float, dy_m: float) -> tuple[float, float]:
    """Move a point by metres east/north (spherical approximation, fine for fixture layout)."""
    return lat + dy_m / _M_PER_DEG, lon + dx_m / (_M_PER_DEG * math.cos(math.radians(lat)))


@dataclass
class FleetFixture:
    arcgis: dict[str, Any]
    osm: dict[str, Any]
    arcgis_meta: dict[str, Any]
    osm_meta: dict[str, Any]
    expected: dict[str, int] = field(default_factory=dict)


def synthetic_fleet(grid: C.GridSpec = C.SMOKE_GRID, seed: int = 2026, n_farms: int = 8) -> FleetFixture:
    """Synthetic ArcGIS-like and Overpass-like inventories with deliberate edge cases.

    Included: inactive/future-year/out-of-capacity-range ArcGIS rows, a duplicate-location pair,
    OSM elements that match ArcGIS within 150 m, unmatched OSM-only turbines, inactive OSM
    elements, one turbine on a cell edge (exact in the OSM copy) and one outside the grid.
    """
    rng = np.random.default_rng(seed)
    lat_span, lon_span = grid.lat_max - grid.lat_min, grid.lon_max - grid.lon_min
    arc_rows: list[dict[str, Any]] = []   # {props, lat, lon}
    osm_rows: list[dict[str, Any]] = []   # {tags, lat, lon, kind}
    exp = {"osm_matched_unique": 0}

    def arc(props: dict[str, Any], lat: float, lon: float) -> None:
        arc_rows.append({"props": props, "lat": lat, "lon": lon})

    for k in range(n_farms):
        year, cap, rotor, hub, manu, model = _ERAS[k % len(_ERAS)]
        c_lat = grid.lat_min + 0.4 + rng.random() * (lat_span - 0.8)
        c_lon = grid.lon_min + 0.4 + rng.random() * (lon_span - 0.8)
        n_t = int(rng.integers(3, 8))
        for t in range(n_t):
            lat, lon = offset_deg(c_lat, c_lon, 500.0 * t + rng.normal(0, 15), rng.normal(0, 15))
            props = {
                "Nimi": f"SYNTHETIC Park {k}", "Tila": "Toiminnassa",
                "Teho_MW": f"{cap:.1f}".replace(".", ",") + " MW" if rng.random() < 0.3 else f"{cap:.1f} MW",
                "Napakorkeus_m": hub if rng.random() > 0.15 else None,
                "Roottorin_halkaisija_m": rotor if rng.random() > 0.15 else None,
                "Valmistaja": manu, "Turbiinimalli": model,
                "Kayttoonottovuosi": year + int(rng.integers(0, 2)) if rng.random() > 0.3 else None,
            }
            if rng.random() < 0.1:
                props["Teho_MW"] = None
            arc(props, lat, lon)
            if rng.random() < 0.6:  # OSM twin 0-60 m away -> matched
                o_lat, o_lon = offset_deg(lat, lon, rng.uniform(-40, 40), rng.uniform(-40, 40))
                tags = {"power": "generator", "generator:source": "wind", "name": f"SYNTHETIC osm {k}-{t}"}
                if rng.random() < 0.5:
                    tags["generator:output:electricity"] = f"{cap:.1f} MW"
                if rng.random() < 0.3:
                    tags["height:hub"] = f"{hub:.0f}"
                osm_rows.append({"tags": tags, "lat": o_lat, "lon": o_lon, "kind": "node"})
                exp["osm_matched_unique"] += 1
        # OSM-only turbine 400 m off the line: farther than 150 m from every ArcGIS turbine
        u_lat, u_lon = offset_deg(c_lat, c_lon, 0.0, 400.0)
        osm_rows.append({"tags": {"power": "generator", "generator:source": "wind", "name": f"SYNTHETIC osm-only {k}",
                                  "generator:output:electricity": "3000 kW"}, "lat": u_lat, "lon": u_lon, "kind": "way"})

    # duplicate-location pair
    d_lat, d_lon = grid.lat_min + 0.9, grid.lon_min + 0.7
    for suffix in ("", " (dup)"):
        arc({"Nimi": f"SYNTHETIC Duplicate{suffix}", "Tila": "Toiminnassa", "Teho_MW": "3.0 MW", "Napakorkeus_m": 110.0,
             "Roottorin_halkaisija_m": 110.0, "Valmistaja": "SYNTH-B", "Turbiinimalli": "SB-3000", "Kayttoonottovuosi": 2013},
            d_lat, d_lon)

    # exactly on the boundary between two cells (0.125 deg offsets are exact in binary)
    e_lat, e_lon = grid.lat_min + 0.125 + 0.25 * 4, grid.lon_min + 0.125 + 0.25 * 4
    a_lat, a_lon = offset_deg(e_lat, e_lon, 3.0, 0.0)
    arc({"Nimi": "SYNTHETIC Edge", "Tila": "Toiminnassa", "Teho_MW": "4.0 MW", "Napakorkeus_m": 125.0,
         "Roottorin_halkaisija_m": 130.0, "Valmistaja": "SYNTH-C", "Turbiinimalli": "SC-4000", "Kayttoonottovuosi": 2018}, a_lat, a_lon)
    osm_rows.append({"tags": {"power": "generator", "generator:source": "wind", "name": "SYNTHETIC osm edge"},
                     "lat": e_lat, "lon": e_lon, "kind": "node"})
    exp["osm_matched_unique"] += 1

    # outside the ERA5 grid
    arc({"Nimi": "SYNTHETIC Outside", "Tila": "Toiminnassa", "Teho_MW": "3.0 MW", "Napakorkeus_m": 100.0,
         "Roottorin_halkaisija_m": 100.0, "Valmistaja": "SYNTH-B", "Turbiinimalli": "SB-3000", "Kayttoonottovuosi": 2012},
        grid.lat_max + 1.0, grid.lon_min + 1.0)
    n_kept_arc = len(arc_rows)

    # out-of-range capacities stay in (capacity -> NaN); inactive/future-year rows are dropped
    for cap_bad in (0.1, 25.0):
        arc({"Nimi": "SYNTHETIC BadCap", "Tila": "Toiminnassa", "Teho_MW": f"{cap_bad:g} MW", "Napakorkeus_m": 120.0,
             "Roottorin_halkaisija_m": 120.0, "Valmistaja": "SYNTH-B", "Turbiinimalli": "SB-x", "Kayttoonottovuosi": 2016},
            grid.lat_min + 2.2 + cap_bad / 100, grid.lon_min + 3.1)
    n_kept_arc += 2
    for i, status in enumerate(("Suunnitteilla", "Rakenteilla", "Purettu")):
        arc({"Nimi": f"SYNTHETIC {status}", "Tila": status, "Teho_MW": "3.0 MW", "Napakorkeus_m": 120.0,
             "Roottorin_halkaisija_m": 120.0, "Valmistaja": "SYNTH-B", "Turbiinimalli": "SB-3000", "Kayttoonottovuosi": 2020},
            grid.lat_min + 1.5 + i * 0.3, grid.lon_min + 3.4)
    arc({"Nimi": "SYNTHETIC FutureYear", "Tila": "Toiminnassa", "Teho_MW": "3.0 MW", "Napakorkeus_m": 120.0,
         "Roottorin_halkaisija_m": 120.0, "Valmistaja": "SYNTH-B", "Turbiinimalli": "SB-3000", "Kayttoonottovuosi": 2031},
        grid.lat_min + 2.9, grid.lon_min + 3.6)
    n_arc_filtered = 4

    # inactive OSM elements (lifecycle key, status value, future start_date)
    inactive = [
        {"power": "generator", "generator:source": "wind", "construction:power": "generator"},
        {"power": "generator", "generator:source": "wind", "status": "planned"},
        {"power": "generator", "generator:source": "wind", "start_date": "2030-01-01"},
    ]
    for i, tags in enumerate(inactive):
        osm_rows.append({"tags": dict(tags, name="SYNTHETIC inactive"), "lat": grid.lat_min + 2.0 + 0.2 * i,
                         "lon": grid.lon_max - 0.6, "kind": "node"})

    arcgis = {
        "type": "FeatureCollection",
        "synthetic": SYNTHETIC_STAMP,
        "note": STAMP_TEXT,
        "features": [
            {"type": "Feature", "id": i + 1,
             "properties": {"OBJECTID": i + 1, **r["props"]},
             "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]}}
            for i, r in enumerate(arc_rows)
        ],
    }
    elements = []
    for i, r in enumerate(osm_rows):
        el: dict[str, Any] = {"type": r["kind"], "id": 9_000_000_000 + i, "tags": r["tags"]}
        if r["kind"] == "node":
            el.update(lat=r["lat"], lon=r["lon"])
        else:
            el["center"] = {"lat": r["lat"], "lon": r["lon"]}
        elements.append(el)
    osm = {
        "version": 0.6, "generator": f"{SYNTHETIC_STAMP} Overpass-style fixture", "synthetic": SYNTHETIC_STAMP,
        "note": STAMP_TEXT,
        "osm3s": {"timestamp_osm_base": FIXED_ISO, "copyright": f"{SYNTHETIC_STAMP} - not OpenStreetMap data"},
        "elements": elements,
    }
    osm_kept = len(osm_rows) - len(inactive)
    expected = {
        "arcgis_kept": n_kept_arc, "arcgis_filtered": n_arc_filtered,
        "osm_kept": osm_kept, "osm_skipped_inactive": len(inactive),
        "osm_matched_unique": exp["osm_matched_unique"],
        "total_turbines": n_kept_arc + osm_kept - exp["osm_matched_unique"],
        "arcgis_capacity_out_of_range": 2,
    }
    return FleetFixture(
        arcgis=arcgis, osm=osm,
        arcgis_meta={"synthetic": SYNTHETIC_STAMP, "retrieved_utc": FIXED_ISO, "last_edit_utc": FIXED_ISO,
                     "layer_name": "SYNTHETIC layer", "copyright_text": STAMP_TEXT, "license_info": STAMP_TEXT},
        osm_meta={"synthetic": SYNTHETIC_STAMP, "retrieved_utc": FIXED_ISO, "base_timestamp_utc": FIXED_ISO},
        expected=expected,
    )


def write_fleet_cache(cache_dir: "str | Path", fixture: "FleetFixture | None" = None, **kwargs: Any) -> FleetFixture:
    """Write the fixture in the exact cache layout fleet.py reads offline."""
    fx = fixture or synthetic_fleet(**kwargs)
    cache = Path(cache_dir)
    atomic_write_json(cache / C.ARCGIS_CACHE_FILE, fx.arcgis)
    atomic_write_json(cache / C.ARCGIS_META_FILE, fx.arcgis_meta)
    atomic_write_json(cache / C.OSM_CACHE_FILE, fx.osm)
    atomic_write_json(cache / C.OSM_META_FILE, fx.osm_meta)
    return fx


# --- Fingrid-like series -----------------------------------------------------------
FAKE_API_KEY = "synthetic-test-key"  # placeholder for fake transports only; not a credential
_FG_RESOLUTION_S = {181: 180, 248: 900, 124: 180, 319: 900}   # wind/consumption 3 min, solar/price 15 min
_FG_COLUMNS = {181: "wind_power_mw", 248: "solar_power_mw", 124: "total_consumption_mw", 319: "imbalance_price_eur_mwh"}
_FG_NAMES = {181: "Wind power production - real time data", 248: "Solar power production - real time data",
             124: "Electricity consumption in Finland - real time data", 319: "Imbalance price"}
_FG_UNITS = {181: "MW", 248: "MW", 124: "MW", 319: "EUR/MWh"}


def synthetic_observed(theoretical: pd.Series, installed_mw: float, seed: int = 2026) -> pd.DataFrame:
    """Hourly 'Fingrid-like' series derived from a simulated theoretical series (SYNTHETIC).

    Actual wind = 0.88 x theoretical with noise, cut back by a curtailment pattern when the
    synthetic price is low or wind is large relative to consumption. Consumption scales with the
    installed capacity so that the wind-to-consumption gates are exercised.
    """
    rng = np.random.default_rng(seed)
    idx = theoretical.index
    n = len(idx)
    theo = theoretical.to_numpy(float)
    cf = theo / installed_mw
    hour = np.asarray(idx.hour)
    consumption = installed_mw * 1.6 * (1.0 + 0.08 * np.sin(2 * np.pi * (hour - 8) / 24)) * (1 + 0.02 * rng.standard_normal(n))
    price = 45.0 - 70.0 * (cf - 0.35) + 8.0 * rng.standard_normal(n)
    ratio = theo / np.maximum(consumption, 1.0)
    actual = 0.88 * theo - 0.02 * installed_mw + 0.03 * installed_mw * rng.standard_normal(n)
    curtailed = (price < 15.0) | (ratio > 0.40)
    actual = np.where(curtailed, actual * rng.uniform(0.55, 0.9, n), actual)
    solar = np.maximum(0.0, installed_mw * 0.5 * np.sin(np.pi * (hour - 5) / 14)) * (hour >= 5) * (hour <= 19)
    return pd.DataFrame({"wind_power_mw": np.maximum(actual, 0.0), "solar_power_mw": solar,
                         "total_consumption_mw": consumption, "imbalance_price_eur_mwh": price}, index=idx)


def fingrid_rows(hourly: pd.Series, resolution_s: int, dataset_id: int, missing_hours: "Any" = ()) -> list[dict[str, Any]]:
    """Sub-hourly API rows whose in-hour mean equals the hourly value (zero-sum cosine offsets).

    Hours listed in `missing_hours` get no rows at all.
    """
    k = 3600 // resolution_s
    offs = np.cos(2 * np.pi * np.arange(k) / k) * 0.5
    skip = {pd.Timestamp(h) for h in missing_hours}
    rows: list[dict[str, Any]] = []
    for ts, val in hourly.items():
        if pd.Timestamp(ts) in skip:
            continue
        for j in range(k):
            t0 = pd.Timestamp(ts) + pd.Timedelta(seconds=j * resolution_s)
            rows.append({"datasetId": dataset_id, "startTime": t0.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                         "endTime": (t0 + pd.Timedelta(seconds=resolution_s)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                         "value": round(float(val) + float(offs[j]), 6)})
    return rows


@dataclass
class FingridFixture:
    rows: dict[int, list[dict[str, Any]]]
    metadata: dict[int, dict[str, Any]]
    hourly: pd.DataFrame            # the exact hourly truth the rows average to (before dropping missing hours)
    missing_wind_hours: list[pd.Timestamp]


def synthetic_fingrid_metadata() -> dict[int, dict[str, Any]]:
    """Dataset-definition metadata of the fake API (SYNTHETIC names; units as expected by the verifier)."""
    return {i: {"id": i, "nameEn": f"{SYNTHETIC_STAMP} {_FG_NAMES[i]}", "unitEn": _FG_UNITS[i],
                "dataPeriodEn": f"{_FG_RESOLUTION_S[i] // 60} min", "license": STAMP_TEXT, "synthetic": SYNTHETIC_STAMP}
            for i in _FG_COLUMNS}


def synthetic_fingrid(observed: pd.DataFrame, missing_wind_hours: "Any" = ()) -> FingridFixture:
    rows = {}
    for ds_id, col in _FG_COLUMNS.items():
        rows[ds_id] = fingrid_rows(observed[col], _FG_RESOLUTION_S[ds_id], ds_id, missing_wind_hours if ds_id == 181 else ())
    return FingridFixture(rows, synthetic_fingrid_metadata(), observed, [pd.Timestamp(h) for h in missing_wind_hours])


class FakeResponse:
    def __init__(self, status_code: int, payload: Any = None, headers: "dict[str, str] | None" = None):
        self.status_code, self._payload, self.headers = status_code, payload, headers or {}

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeFingridSession:
    """Offline stand-in for requests.Session: paging, key check, one 429, optional per-window row cap.

    `calls` records (url, params) without headers, so the key is never stored.
    """

    def __init__(self, fixture: FingridFixture, expected_key: str = FAKE_API_KEY, fail_429_once: bool = True,
                 max_rows_per_window: "int | None" = None, unit_overrides: "dict[int, str] | None" = None):
        self.fixture, self.expected_key = fixture, expected_key
        self.pending_429 = fail_429_once
        self.max_rows_per_window = max_rows_per_window
        self.unit_overrides = unit_overrides or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, headers: "dict[str, str] | None" = None, params: "dict[str, Any] | None" = None,
            timeout: Any = None) -> FakeResponse:
        import re

        self.calls.append((url, dict(params or {})))
        if (headers or {}).get("x-api-key") != self.expected_key:
            return FakeResponse(403, {"message": "forbidden"})
        m = re.search(r"/datasets/(\d+)(/data)?$", url)
        if not m:
            return FakeResponse(404, {})
        ds_id = int(m.group(1))
        if not m.group(2):
            meta = dict(self.fixture.metadata[ds_id])
            if ds_id in self.unit_overrides:
                meta["unitEn"] = self.unit_overrides[ds_id]
            return FakeResponse(200, meta)
        if self.pending_429:
            self.pending_429 = False
            return FakeResponse(429, {"message": "slow down"}, {"Retry-After": "2"})
        p = params or {}
        lo, hi = pd.Timestamp(p["startTime"]), pd.Timestamp(p["endTime"])
        rows = [r for r in self.fixture.rows[ds_id] if lo <= pd.Timestamp(r["startTime"]) <= hi]
        if self.max_rows_per_window is not None:
            rows = rows[: self.max_rows_per_window]
        size, page = int(p["pageSize"]), int(p["page"])
        last = max(1, -(-len(rows) // size))
        data = rows[(page - 1) * size: page * size]
        return FakeResponse(200, {"data": data, "pagination": {"currentPage": page, "lastPage": last,
                                                               "nextPage": page + 1 if page < last else None,
                                                               "pageSize": size, "total": len(rows)}})

