"""Fleet inventory: OSM + ArcGIS fetch/cache, cleaning, 150 m merge, imputation, grid aggregation."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from . import config as C
from .io_utils import atomic_write_json, epoch_ms_to_iso, utc_now_iso

# --- parse helpers (notebook 1) -------------------------------------------------


def is_inactive_text(val: Any) -> bool:
    if val is None or (not isinstance(val, (list, tuple, np.ndarray)) and pd.isna(val)):
        return False
    s = str(val).lower().strip()
    return any(kw in s for kw in C.INACTIVE_KEYWORDS)


def parse_mw(val: Any, default_unit: str = "mw") -> float:
    """'3,6 MW' -> 3.6, '3600 kW' -> 3.6, '0.003 GW' -> 3.0; unit-less numbers use `default_unit`."""
    if val is None or pd.isna(val):
        return np.nan
    m = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*(gw|mw|kw)?", str(val).lower())
    if not m:
        return np.nan
    num = float(m.group(1).replace(",", "."))
    unit = m.group(2) or default_unit
    if unit == "kw":
        return num / 1000.0
    if unit == "gw":
        return num * 1000.0
    return num


def parse_meters(val: Any) -> float:
    if val is None or pd.isna(val):
        return np.nan
    m = re.search(r"([0-9]+(?:[.,][0-9]+)?)", str(val))
    return float(m.group(1).replace(",", ".")) if m else np.nan


def parse_year(val: Any) -> float:
    if val is None or pd.isna(val):
        return np.nan
    m = re.search(r"(19[89][0-9]|20[0-3][0-9])", str(val))
    return int(m.group(1)) if m else np.nan


def find_col(columns: Iterable[str], keywords: Sequence[str]) -> "str | None":
    """First column (in keyword priority order) whose lower-cased name contains a keyword."""
    cols = list(columns)
    for kw in keywords:
        for c in cols:
            if kw in c.lower():
                return c
    return None


# --- network fetchers (explicit; never used by offline paths or tests) -------------
Sleep = Callable[[float], None]


def _retry(call: Callable[[], Any], what: str, attempts: int, sleep: Sleep, backoff_s: float = 5.0) -> Any:
    last: "Exception | None" = None
    for k in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:  # transport or HTTP error: retry with linear backoff
            last = exc
            if k < attempts:
                sleep(backoff_s * k)
    raise RuntimeError(f"{what} failed after {attempts} attempts: {last}")


def _default_session() -> Any:
    import requests  # lazy: keeps offline imports light

    s = requests.Session()
    s.headers.update({"User-Agent": C.HTTP_USER_AGENT, "Accept": "application/json"})
    return s


def fetch_osm(
    session: Any = None,
    mirrors: Sequence[str] = C.OSM_MIRRORS,
    query: str = C.OSM_QUERY,
    timeout: float = 180.0,
    attempts: int = 3,
    sleep: Sleep = time.sleep,
) -> "tuple[dict[str, Any], dict[str, Any]]":
    """POST the Overpass query to each mirror in turn; return (full response, snapshot metadata)."""
    session = session or _default_session()
    errors: list[str] = []
    for url in mirrors:
        def call(url: str = url) -> dict[str, Any]:
            resp = session.post(url, data={"data": query}, timeout=timeout)
            resp.raise_for_status()
            body = resp.json()
            if not isinstance(body.get("elements"), list):
                raise ValueError("response has no 'elements' list")
            if "error" in str(body.get("remark", "")).lower():  # Overpass reports runtime errors in-band
                raise ValueError(f"overpass remark: {body['remark']}")
            return body

        try:
            body = _retry(call, f"Overpass {url}", attempts, sleep)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        meta = {
            "source": "OpenStreetMap via Overpass API", "mirror": url, "retrieved_utc": utc_now_iso(),
            "base_timestamp_utc": (body.get("osm3s") or {}).get("timestamp_osm_base"),
            "element_count": len(body["elements"]),
            "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
            "licence": "ODbL 1.0 (c) OpenStreetMap contributors",
        }
        return body, meta
    raise RuntimeError("All Overpass mirrors failed: " + " | ".join(errors))


def _esri_to_feature(feat: Mapping[str, Any]) -> dict[str, Any]:
    geom = feat.get("geometry") or {}
    if "x" in geom and "y" in geom:
        g: "dict[str, Any] | None" = {"type": "Point", "coordinates": [geom["x"], geom["y"]]}
    elif geom.get("rings"):
        g = {"type": "Polygon", "coordinates": [geom["rings"][0]]}
    else:
        g = None
    return {"type": "Feature", "properties": dict(feat.get("attributes") or {}), "geometry": g}


def fetch_arcgis(
    session: Any = None,
    service_url: str = C.ARCGIS_SERVICE_URL,
    page_size: int = C.ARCGIS_PAGE_SIZE,
    attempts: int = 3,
    sleep: Sleep = time.sleep,
) -> "tuple[dict[str, Any], dict[str, Any]]":
    """Page the FeatureServer layer by ObjectID; return (GeoJSON FeatureCollection, snapshot metadata)."""
    session = session or _default_session()

    def get_json(url: str, **params: Any) -> dict[str, Any]:
        def call() -> dict[str, Any]:
            r = session.get(url, params={**params, "f": "json"}, timeout=30)
            r.raise_for_status()
            return r.json()
        return _retry(call, f"GET {url}", attempts, sleep)

    def post_json(url: str, data: dict[str, Any]) -> dict[str, Any]:
        def call() -> dict[str, Any]:
            r = session.post(url, data=data, timeout=60)
            r.raise_for_status()
            return r.json()
        return _retry(call, f"POST {url}", attempts, sleep)

    service = get_json(service_url)
    layers = service.get("layers", [])
    layer_id = layers[0]["id"] if layers else 1
    layer = get_json(f"{service_url}/{layer_id}")
    use_geojson = "geoJSON" in str(layer.get("supportedQueryFormats", ""))
    query_url = f"{service_url}/{layer_id}/query"
    ids = sorted(post_json(query_url, {"where": "1=1", "returnIdsOnly": "true", "f": "json"}).get("objectIds") or [])
    features: list[dict[str, Any]] = []
    for start in range(0, len(ids), page_size):
        page = post_json(query_url, {
            "objectIds": ",".join(map(str, ids[start:start + page_size])), "outFields": "*", "outSR": "4326",
            "returnGeometry": "true", "f": "geojson" if use_geojson else "json",
        })
        got = page.get("features", [])
        features.extend(got if use_geojson else (_esri_to_feature(f) for f in got))
    editing = layer.get("editingInfo") or {}
    meta = {
        "source": "Finnish wind turbine layer (ArcGIS FeatureServer)", "service_url": service_url,
        "layer_id": layer_id, "layer_name": layer.get("name"), "query_format": "geojson" if use_geojson else "esri-json",
        "retrieved_utc": utc_now_iso(), "object_count": len(ids), "feature_count": len(features),
        "editing_info": editing, "last_edit_utc": epoch_ms_to_iso(editing.get("dataLastEditDate")),
        "copyright_text": layer.get("copyrightText") or service.get("copyrightText"),
        "license_info": layer.get("licenseInfo") or service.get("licenseInfo"),
    }
    return {"type": "FeatureCollection", "features": features}, meta


def refresh_fleet_cache(cache_dir: "str | Path", session: Any = None, sleep: Sleep = time.sleep) -> None:
    """Explicit network refresh of both raw inventories (full responses + snapshot sidecars)."""
    cache = Path(cache_dir)
    osm, osm_meta = fetch_osm(session, sleep=sleep)
    atomic_write_json(cache / C.OSM_CACHE_FILE, osm)
    atomic_write_json(cache / C.OSM_META_FILE, osm_meta)
    arc, arc_meta = fetch_arcgis(session, sleep=sleep)
    atomic_write_json(cache / C.ARCGIS_CACHE_FILE, arc)
    atomic_write_json(cache / C.ARCGIS_META_FILE, arc_meta)


# --- offline cache readers -----------------------------------------------------
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_osm_cache(cache_dir: "str | Path") -> "tuple[dict[str, Any], dict[str, Any]]":
    cache = Path(cache_dir)
    path = cache / C.OSM_CACHE_FILE
    if not path.is_file():
        raise FileNotFoundError(f"OSM cache {path} missing; run the fleet fetch with explicit refresh (network)")
    body = _read_json(path)
    if isinstance(body, list):  # legacy notebook cache stored only the element list
        body = {"elements": body}
    meta_path = cache / C.OSM_META_FILE
    meta = _read_json(meta_path) if meta_path.is_file() else {"retrieved_utc": None}
    meta.setdefault("base_timestamp_utc", (body.get("osm3s") or {}).get("timestamp_osm_base"))
    return body, meta


def load_arcgis_cache(cache_dir: "str | Path") -> "tuple[gpd.GeoDataFrame, dict[str, Any]]":
    cache = Path(cache_dir)
    path = cache / C.ARCGIS_CACHE_FILE
    if not path.is_file():
        raise FileNotFoundError(f"ArcGIS cache {path} missing; run the fleet fetch with explicit refresh (network)")
    gdf = gpd.read_file(path)
    gdf = gdf.set_crs(C.CRS_GEO) if gdf.crs is None else gdf.to_crs(C.CRS_GEO)
    meta_path = cache / C.ARCGIS_META_FILE
    meta = _read_json(meta_path) if meta_path.is_file() else {"retrieved_utc": None, "last_edit_utc": None}
    return gdf, meta


# --- cleaning ------------------------------------------------------------------
OSM_COLUMNS = ["osm_id", "name", "lat", "lon", "osm_capacity_mw", "osm_hub_height_m", "osm_rotor_diam_m",
               "osm_manufacturer", "osm_model", "osm_year"]


def _first(tags: Mapping[str, Any], *keys: str) -> Any:
    for k in keys:
        if tags.get(k):
            return tags[k]
    return None


def clean_osm(osm: "Mapping[str, Any] | Sequence[Mapping[str, Any]]") -> "tuple[gpd.GeoDataFrame, dict[str, int]]":
    """Active OSM wind generators as points (EPSG:4326); lifecycle-tagged elements are dropped."""
    elements = osm.get("elements", []) if isinstance(osm, Mapping) else list(osm)
    rows: list[dict[str, Any]] = []
    skipped = 0
    for el in elements:
        tags = el.get("tags", {}) or {}
        bad_key = any(kw in k.lower() for k in tags for kw in C.INACTIVE_KEYWORDS)
        bad_val = any(is_inactive_text(tags.get(k)) for k in ("power", "generator:source", "status", "operational_status"))
        if bad_key or bad_val:
            skipped += 1
            continue
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue
        yr = parse_year(_first(tags, "start_date", "commissioning_date", "year_of_construction", "construction:date", "date"))
        if pd.notna(yr) and yr > C.CURRENT_MAX_YEAR:
            skipped += 1
            continue
        cap = parse_mw(tags.get("generator:output:electricity"))
        if pd.notna(cap) and not (C.MIN_TURBINE_MW <= cap <= C.MAX_TURBINE_MW):
            cap = np.nan
        rows.append({
            "osm_id": str(el["id"]), "name": tags.get("name") or tags.get("operator"),
            "lat": float(lat), "lon": float(lon), "osm_capacity_mw": cap,
            "osm_hub_height_m": parse_meters(_first(tags, "height:hub", "hub_height")),
            "osm_rotor_diam_m": parse_meters(tags.get("rotor:diameter")),
            "osm_manufacturer": _first(tags, "manufacturer", "generator:manufacturer"),
            "osm_model": _first(tags, "model", "generator:model"),
            "osm_year": yr,
        })
    df = pd.DataFrame(rows, columns=OSM_COLUMNS)
    for c in ("lat", "lon", "osm_capacity_mw", "osm_hub_height_m", "osm_rotor_diam_m", "osm_year"):
        df[c] = df[c].astype(float)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=C.CRS_GEO)
    return gdf, {"elements": len(elements), "kept": len(df), "skipped_inactive": skipped}


_ARC_KEYWORDS: dict[str, list[str]] = {
    "name": ["name", "nimi", "hanke", "puisto", "project"],
    "status": ["status", "tila", "state", "vaihe", "rakennusvaihe"],
    "capacity": ["capacity", "teho", "mw", "kapasiteetti", "power"],
    "hub": ["hub", "napakorkeus", "height", "korkeus"],
    "rotor": ["rotor", "roottori", "diameter", "halkaisija"],
    "manufacturer": ["manufacturer", "valmistaja", "brand", "toimittaja"],
    "model": ["model", "malli", "tyyppi", "type"],
    "year": ["year", "vuosi", "valmistum", "kayttoonot", "rakennus"],
}


def arcgis_column_mapping(columns: Iterable[str]) -> dict[str, "str | None"]:
    """Fuzzy ArcGIS field mapping; the result must be logged in the run manifest."""
    cols = [c for c in columns if c != "geometry"]
    return {k: find_col(cols, kws) for k, kws in _ARC_KEYWORDS.items()}


def clean_arcgis(gdf_raw: gpd.GeoDataFrame) -> "tuple[gpd.GeoDataFrame, dict[str, Any], dict[str, Any]]":
    """Active, plausibly dated ArcGIS turbines -> (gdf, column mapping, stats)."""
    m = arcgis_column_mapping(gdf_raw.columns)
    n = len(gdf_raw)
    cap_default_unit = "kw" if m["capacity"] and "kw" in m["capacity"].lower() else "mw"

    def col(key: str, fn: "Callable[[Any], Any] | None" = None, default: Any = None) -> Any:
        c = m[key]
        if c is None:
            return default
        return gdf_raw[c].apply(fn) if fn else gdf_raw[c]

    off = pd.DataFrame({
        "off_name": col("name"),
        "off_status": col("status", default="Operational"),
        "off_capacity_mw": col("capacity", lambda v: parse_mw(v, cap_default_unit), np.nan),
        "off_hub_height_m": col("hub", parse_meters, np.nan),
        "off_rotor_diam_m": col("rotor", parse_meters, np.nan),
        "off_manufacturer": col("manufacturer"),
        "off_model": col("model"),
        "off_year": col("year", parse_year, np.nan),
    }, index=gdf_raw.index)
    gdf = gpd.GeoDataFrame(off, geometry=gdf_raw.geometry, crs=gdf_raw.crs)
    status_counts = gdf_raw[m["status"]].value_counts(dropna=False).to_dict() if m["status"] else {}
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    n_no_geom = n - len(gdf)

    active = ~gdf["off_status"].apply(is_inactive_text)
    year_ok = gdf["off_year"].isna() | (gdf["off_year"] <= C.CURRENT_MAX_YEAR)
    skipped = int((~(active & year_ok)).sum())
    gdf = gdf[active & year_ok].copy()

    bad_cap = gdf["off_capacity_mw"].notna() & ((gdf["off_capacity_mw"] < C.MIN_TURBINE_MW) | (gdf["off_capacity_mw"] > C.MAX_TURBINE_MW))
    gdf.loc[bad_cap, "off_capacity_mw"] = np.nan
    pts = gdf.to_crs(C.CRS_METRIC).geometry.centroid.to_crs(C.CRS_GEO)
    gdf["off_lat"], gdf["off_lon"] = pts.y, pts.x
    stats = {"raw_rows": n, "kept": len(gdf), "skipped_inactive_or_future": skipped, "dropped_no_geometry": n_no_geom,
             "capacity_out_of_range": int(bad_cap.sum()), "status_counts": {str(k): int(v) for k, v in status_counts.items()}}
    return gdf.reset_index(drop=True), m, stats


# --- merge ---------------------------------------------------------------------
def match_and_merge(
    gdf_off: gpd.GeoDataFrame, gdf_osm: gpd.GeoDataFrame, radius_m: float = C.MATCH_RADIUS_M
) -> "tuple[pd.DataFrame, dict[str, int]]":
    """Nearest-neighbour match (<= radius_m in EPSG:3067), ArcGIS fields preferred via combine_first.

    Matched turbines take OSM coordinates (as in the notebook); unmatched OSM points are appended.
    Several ArcGIS rows may share one OSM point; ArcGIS rows at identical coordinates stay separate.
    """
    off = gdf_off.reset_index(drop=True)
    osm = gdf_osm.reset_index(drop=True)
    if len(off) and len(osm):
        joined = gpd.sjoin_nearest(off.to_crs(C.CRS_METRIC), osm.to_crs(C.CRS_METRIC), how="left",
                                   max_distance=radius_m, distance_col="match_dist_m").to_crs(C.CRS_GEO)
        n_rows = len(joined)
        joined = joined.sort_values(["match_dist_m", "index_right"], kind="stable")
        joined = joined[~joined.index.duplicated(keep="first")].sort_index()
        n_ties = n_rows - len(joined)
        matched_idx = joined["index_right"].dropna().unique()
    else:
        joined = off.copy()
        for c in OSM_COLUMNS:
            joined[c] = np.nan
        joined["match_dist_m"] = np.nan
        n_ties, matched_idx = 0, np.array([])
    unmatched = osm.loc[~osm.index.isin(matched_idx)].copy()

    arc = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore")).reset_index(drop=True)
    arc["_origin"] = "arcgis"
    osm_only = pd.DataFrame(unmatched.drop(columns=["geometry"])).reset_index(drop=True)
    osm_only["_origin"] = "osm"
    j = pd.concat([arc, osm_only], ignore_index=True)

    fallback_ids = pd.Series("OFF_" + j.index.astype(str), index=j.index)
    source = np.where(j["_origin"] == "osm", "osm", np.where(j["osm_id"].notna(), "arcgis+osm", "arcgis"))
    out = pd.DataFrame({
        "turbine_id": j["osm_id"].fillna(fallback_ids),
        "farm_name": j["off_name"].combine_first(j["name"]),
        "status": j["off_status"].fillna("Operational (OSM)"),
        "lat": j["lat"].combine_first(j["off_lat"]),
        "lon": j["lon"].combine_first(j["off_lon"]),
        "capacity_mw": j["off_capacity_mw"].combine_first(j["osm_capacity_mw"]),
        "osm_capacity_mw": j["osm_capacity_mw"],
        "official_capacity_mw": j["off_capacity_mw"],
        "hub_height_m": j["off_hub_height_m"].combine_first(j["osm_hub_height_m"]),
        "rotor_diameter_m": j["off_rotor_diam_m"].combine_first(j["osm_rotor_diam_m"]),
        "manufacturer": j["off_manufacturer"].combine_first(j["osm_manufacturer"]),
        "model": j["off_model"].combine_first(j["osm_model"]),
        "commissioning_year": j["off_year"].combine_first(j["osm_year"]),
        "source": source,
        "match_dist_m": j["match_dist_m"],
    })
    stats = {
        "arcgis_rows": len(off), "osm_rows": len(osm), "matched_arcgis_rows": int(arc["osm_id"].notna().sum()),
        "matched_osm_unique": int(len(matched_idx)), "osm_only_appended": len(unmatched),
        "tie_rows_dropped": int(n_ties), "total": len(out),
    }
    return out, stats


# --- imputation ----------------------------------------------------------------
IMPUTE_FIELDS = {
    # column -> (provenance prefix, radius m, neighbour statistic, median fallback)
    "capacity_mw": ("capacity", C.CAP_IMPUTE_RADIUS_M, "mean", True),
    "hub_height_m": ("hub", C.HUB_IMPUTE_RADIUS_M, "mean", True),
    "rotor_diameter_m": ("rotor", C.ROTOR_IMPUTE_RADIUS_M, "mean", True),
    "commissioning_year": ("year", C.YEAR_IMPUTE_RADIUS_M, "median", False),
}


def impute_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing capacity/hub/rotor/year from the nearest known turbine within a radius.

    Capacity/hub/rotor fall back to the fleet median; year does not. Each field gets
    `<prefix>_imputed` (bool) and `<prefix>_impute_method` (text, "" when observed).
    """
    out = df.copy()
    metric = gpd.GeoDataFrame(out[["lat", "lon"]].copy(), geometry=gpd.points_from_xy(out["lon"], out["lat"]),
                              crs=C.CRS_GEO).to_crs(C.CRS_METRIC)
    for col, (prefix, radius, stat, fallback) in IMPUTE_FIELDS.items():
        out[f"{prefix}_imputed"] = False
        out[f"{prefix}_impute_method"] = ""
        known = metric.assign(**{col: out[col]})
        known = known[known[col].notna()][[col, "geometry"]]
        unknown = metric[out[col].isna()][["geometry"]]
        if len(unknown) and len(known):
            near = gpd.sjoin_nearest(unknown, known, how="left", max_distance=radius)
            vals = near.groupby(near.index)[col].agg(stat).dropna()
            out.loc[vals.index, col] = vals
            out.loc[vals.index, f"{prefix}_imputed"] = True
            out.loc[vals.index, f"{prefix}_impute_method"] = f"nearest_known_{stat}_within_{int(radius)}m"
        if fallback:
            still = out[col].isna()
            if still.any() and out[col].notna().any():
                out.loc[still, col] = out[col].median()
                out.loc[still, f"{prefix}_imputed"] = True
                out.loc[still, f"{prefix}_impute_method"] = "fleet_median_fallback"
    return out


# --- grid ----------------------------------------------------------------------
def grid_cell_polygons(lats: Sequence[float], lons: Sequence[float], res: float = C.GRID_RES_DEG) -> gpd.GeoDataFrame:
    """One res x res box per ERA5 cell centre; pixel_i indexes lats, pixel_j indexes lons."""
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    ii, jj = np.meshgrid(np.arange(lats.size), np.arange(lons.size), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    clat, clon = lats[ii], lons[jj]
    return gpd.GeoDataFrame({
        "pixel_i": ii, "pixel_j": jj, "era5_lat": clat, "era5_lon": clon,
        "geometry": shapely.box(clon - res / 2, clat - res / 2, clon + res / 2, clat + res / 2),
    }, crs=C.CRS_GEO)


def assign_cells(df: pd.DataFrame, grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Point-in-polygon (`within`) join to grid cells, then notebook rounding.

    Points on a cell edge or outside the grid get NaN pixel indices (unassigned).
    """
    pts = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=C.CRS_GEO)
    out = (gpd.sjoin(pts, grid_gdf[["pixel_i", "pixel_j", "era5_lat", "era5_lon", "geometry"]], how="left", predicate="within")
           .drop(columns=["geometry", "index_right"]).reset_index(drop=True))
    out["pixel_i"] = out["pixel_i"].astype("Int64")
    out["pixel_j"] = out["pixel_j"].astype("Int64")
    out["lat"] = out["lat"].round(6)
    out["lon"] = out["lon"].round(6)
    for c in ("capacity_mw", "hub_height_m", "rotor_diameter_m"):
        out[c] = out[c].round(2)
    out["commissioning_year"] = out["commissioning_year"].round()
    return out


def aggregate_pixels(df: pd.DataFrame, grid_gdf: gpd.GeoDataFrame, shape: "tuple[int, int]") -> "tuple[pd.DataFrame, pd.DataFrame, np.ndarray]":
    """(pixel_agg, df_pixels, capacity_matrix[lat_i, lon_j]) from assigned turbines."""
    agg = (df.dropna(subset=["pixel_i", "pixel_j"]).astype({"pixel_i": int, "pixel_j": int})
           .groupby(["pixel_i", "pixel_j"])
           .agg(turbine_count=("turbine_id", "count"), installed_capacity_mw=("capacity_mw", "sum"),
                mean_hub_height_m=("hub_height_m", "mean"), mean_rotor_diam_m=("rotor_diameter_m", "mean"),
                mean_commissioning_year=("commissioning_year", "mean"))
           .reset_index())
    pixels = grid_gdf.merge(agg, on=["pixel_i", "pixel_j"], how="left")
    pixels["turbine_count"] = pixels["turbine_count"].fillna(0).astype(int)
    pixels["installed_capacity_mw"] = pixels["installed_capacity_mw"].fillna(0.0)
    matrix = np.zeros(shape, dtype=np.float64)  # notebook used float32; float64 keeps totals exact
    matrix[agg["pixel_i"].to_numpy(), agg["pixel_j"].to_numpy()] = agg["installed_capacity_mw"].to_numpy()
    return agg, pixels, matrix


def capacity_totals(df: pd.DataFrame, pixel_agg: pd.DataFrame, matrix: np.ndarray) -> dict[str, float]:
    """Capacity bookkeeping: table total == pixel aggregation + unassigned == matrix + unassigned."""
    assigned = df["pixel_i"].notna() & df["pixel_j"].notna()
    return {
        "turbine_table_mw": float(df["capacity_mw"].sum()),
        "unassigned_mw": float(df.loc[~assigned, "capacity_mw"].sum()),
        "unassigned_turbines": int((~assigned).sum()),
        "pixel_agg_mw": float(pixel_agg["installed_capacity_mw"].sum()),
        "matrix_mw": float(matrix.sum()),
        "active_cells": int((matrix > 0).sum()),
    }


# --- per-turbine spec table (notebook 1, simulation section) ------------------------
def _find_spec_col(df: pd.DataFrame, candidates: Sequence[str]) -> "str | None":
    for cand in candidates:
        for c in df.columns:
            if c.lower() == cand:
                return c
    for cand in candidates:
        for c in df.columns:
            if cand in c.lower():
                return c
    return None


def _era_pick(year: pd.Series, values: Sequence[float]) -> pd.Series:
    """values[k] for year < ERA_YEAR_EDGES[k]; values[-1] otherwise (also for NaN year)."""
    conds = [year < e for e in C.ERA_YEAR_EDGES]
    return pd.Series(np.select(conds, list(values[:-1]), default=values[-1]), index=year.index)


def build_turbine_spec_table(turbines: pd.DataFrame) -> pd.DataFrame:
    """Cleaned per-turbine specs (`df_t`): cap_mw, rotor_m, hub_m, specific_power_wm2 and archetype bins.

    Era defaults only fill values still missing after imputation; `*_era_default` flags record it.
    `hub_raw_m` keeps the hub column before the tip-height correction.
    """
    col_lat = _find_spec_col(turbines, ["latitude", "lat"])
    col_lon = _find_spec_col(turbines, ["longitude", "lon"])
    col_cap = _find_spec_col(turbines, ["capacity_mw", "rated_power_mw", "power_mw", "cap_mw", "capacity"])
    col_hub = _find_spec_col(turbines, ["hub_height_m", "hub_height", "height_m", "hub_m", "hub"])
    col_rot = _find_spec_col(turbines, ["rotor_diameter_m", "rotor_diameter", "diameter_m", "rotor_m", "rotor"])
    col_yr = _find_spec_col(turbines, ["commissioning_year", "start_year", "year"])
    if col_lat is None or col_lon is None:
        raise ValueError("turbine table needs latitude/longitude columns")

    df = turbines.dropna(subset=[col_lat, col_lon]).copy()
    df["lat"] = pd.to_numeric(df[col_lat], errors="coerce")
    df["lon"] = pd.to_numeric(df[col_lon], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    year = pd.to_numeric(df[col_yr], errors="coerce").clip(*C.YEAR_CLIP) if col_yr else pd.Series(np.nan, index=df.index)
    median_year = float(year.median()) if year.notna().any() else np.nan
    df["year_clean"] = year.fillna(median_year)  # all-NaN stays NaN -> newest-era defaults

    cap = pd.Series(np.nan, index=df.index)
    if col_cap:
        cap = pd.to_numeric(df[col_cap], errors="coerce")
        if "kw" in col_cap.lower() or cap.median() > 100.0:
            cap = cap / 1000.0
    df["cap_era_default"] = cap.isna()
    df["cap_mw"] = cap.fillna(_era_pick(df["year_clean"], C.ERA_CAPACITY_MW))

    rotor = pd.to_numeric(df[col_rot], errors="coerce") if col_rot else pd.Series(np.nan, index=df.index)
    era_sp = _era_pick(df["year_clean"], C.ERA_SPECIFIC_POWER)
    inferred = 2.0 * np.sqrt(df["cap_mw"] * 1e6 / (np.pi * era_sp))
    df["rotor_inferred"] = rotor.isna()
    df["rotor_m"] = rotor.fillna(inferred)

    hub = pd.to_numeric(df[col_hub], errors="coerce") if col_hub else pd.Series(np.nan, index=df.index)
    df["hub_raw_m"] = hub
    tip = hub > C.TIP_HEIGHT_THRESHOLD_M  # reported value is tip height: hub = tip - rotor/2
    hub = hub.where(~tip, hub - 0.5 * df["rotor_m"])
    df["hub_tip_corrected"] = tip
    df["hub_era_default"] = hub.isna()
    df["hub_m"] = hub.fillna(_era_pick(df["year_clean"], C.ERA_HUB_M))

    swept = np.pi * (df["rotor_m"] / 2.0) ** 2
    df["specific_power_wm2"] = (df["cap_mw"] * 1e6 / swept).clip(*C.SP_CLIP)
    df["sp_bin"] = pd.cut(df["specific_power_wm2"], bins=list(C.SP_BIN_EDGES), labels=list(C.SP_BIN_LABELS))
    df["hub_bin"] = pd.cut(df["hub_m"], bins=list(C.HUB_BIN_EDGES), labels=list(C.HUB_BIN_LABELS))
    return df


# --- snapshot metadata and orchestration -----------------------------------------
def fleet_snapshot(arcgis_meta: Mapping[str, Any], osm_meta: Mapping[str, Any]) -> dict[str, "str | None"]:
    return {
        "arcgis_retrieved_utc": arcgis_meta.get("retrieved_utc"),
        "arcgis_last_edit_utc": arcgis_meta.get("last_edit_utc"),
        "osm_base_timestamp_utc": osm_meta.get("base_timestamp_utc"),
        "osm_retrieved_utc": osm_meta.get("retrieved_utc"),
        "note": C.FLEET_SNAPSHOT_NOTE,
    }


@dataclass
class FleetResult:
    turbines: pd.DataFrame          # merged, imputed, cell-assigned (df_turbines)
    specs: pd.DataFrame             # cleaned per-turbine specs (df_t)
    grid: gpd.GeoDataFrame
    pixel_agg: pd.DataFrame
    pixels: pd.DataFrame
    capacity_matrix: np.ndarray
    totals: dict[str, float]
    stats: dict[str, Any]
    snapshot: dict[str, "str | None"]
    column_mapping: dict[str, "str | None"] = field(default_factory=dict)


def build_fleet(
    osm: "Mapping[str, Any]", arcgis_raw: gpd.GeoDataFrame, lats: Sequence[float], lons: Sequence[float],
    osm_meta: "Mapping[str, Any] | None" = None, arcgis_meta: "Mapping[str, Any] | None" = None,
) -> FleetResult:
    """Offline fleet build from raw responses and the ERA5 axes."""
    gdf_osm, osm_stats = clean_osm(osm)
    gdf_off, mapping, arc_stats = clean_arcgis(arcgis_raw)
    merged, merge_stats = match_and_merge(gdf_off, gdf_osm)
    imputed = impute_attributes(merged)
    grid = grid_cell_polygons(lats, lons)
    turbines = assign_cells(imputed, grid)
    agg, pixels, matrix = aggregate_pixels(turbines, grid, (len(lats), len(lons)))
    totals = capacity_totals(turbines, agg, matrix)
    stats = {
        "osm": osm_stats, "arcgis": arc_stats, "merge": merge_stats,
        "year_coverage_pct": float(turbines["commissioning_year"].notna().mean() * 100.0) if len(turbines) else 0.0,
        "total_turbines": len(turbines),
    }
    return FleetResult(turbines, build_turbine_spec_table(turbines), grid, agg, pixels, matrix, totals, stats,
                       fleet_snapshot(arcgis_meta or {}, osm_meta or {}), mapping)


def build_fleet_from_cache(cache_dir: "str | Path", lats: Sequence[float], lons: Sequence[float]) -> FleetResult:
    """Offline entry point: read cached raw inventories only (no network)."""
    osm, osm_meta = load_osm_cache(cache_dir)
    arc, arc_meta = load_arcgis_cache(cache_dir)
    return build_fleet(osm, arc, lats, lons, osm_meta, arc_meta)
