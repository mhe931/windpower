"""Public inputs of the solar report: Suomen Uusiutuvat plant list (XLSX) and Statistics Finland WFS layers.

* plants: https://suomenuusiutuvat.fi/media/julkinen-aurinkovoima_vuositilastot_6_2025.xlsx (status 30.6.2025)
* population grid `vaestoruutu:vaki2025_1km` (inhabitants per 1 km cell, EPSG:3067) and municipalities
  `tilastointialueet:kunta1000k_2025` from https://geo.stat.fi/geoserver/wfs

Network access happens only in `fetch_public` (explicit stage `ingest-solar-public`); everything else reads the cache
under `<data>/cache/...`. Layer and field names are discovered via GetCapabilities / DescribeFeatureType and the chosen
mapping is recorded in the cache metadata. No credentials are needed or read. The XLSX is parsed with the standard
library (no openpyxl dependency).
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import config as C
from .io_utils import atomic_write_bytes, atomic_write_json, atomic_write_text, sha256_bytes, utc_now_iso

PLANTS_URL = "https://suomenuusiutuvat.fi/media/julkinen-aurinkovoima_vuositilastot_6_2025.xlsx"
WFS_URL = "https://geo.stat.fi/geoserver/wfs"
POP_LAYER = "vaestoruutu:vaki2025_1km"
MUNI_LAYER = "tilastointialueet:kunta1000k_2025"
PLANTS_FILE, PLANTS_META = "plants_suomenuusiutuvat.xlsx", "plants_suomenuusiutuvat.meta.json"
POP_FILE, POP_META = "population_vaki2025_1km.csv", "population_vaki2025_1km.meta.json"
MUNI_FILE, MUNI_META = "municipalities_kunta1000k_2025.geojson", "municipalities_kunta1000k_2025.meta.json"
WFS_PAGE = 50_000
CRS_FINLAND = "EPSG:3067"
EXPECTED_PLANTS, EXPECTED_MW = 30, 250.66

# role -> candidate attribute names (first one present in DescribeFeatureType wins)
POP_FIELD_CANDIDATES = {"cell_id": ("grd_id",), "x": ("xkoord",), "y": ("ykoord",), "inhabitants": ("vaesto", "vaesto_yht")}
MUNI_FIELD_CANDIDATES = {"code": ("kunta",), "name_fi": ("nimi",), "name_sv": ("namn",), "name_en": ("name",)}
LICENCE_HINTS = ("cc by", "creative commons", "licen", "lisenss", "käyttöehd", "attribution", "nimeä")
# The WFS responses and the layer metadata records state no licence (the metadata URL returns HTTP 404 on 2026-11-27):
# the licence below is the one the solar report cites for Statistics Finland data and is NOT confirmed by the service.
ASSUMED_LICENCE = {"assumed_name": "CC BY 4.0", "assumed_basis": "solar report (Statistics Finland open data); not stated by the WFS service or its metadata record",
                   "verified": False}
Sleep = Callable[[float], None]


class PublicDataError(RuntimeError):
    """A public input is malformed, incomplete or unreachable."""


class MunicipalityMatchError(PublicDataError):
    """A plant's municipality does not map to exactly one municipality."""


# --- HTTP ---------------------------------------------------------------------------------------------------------------
def _get(session: Any, url: str, params: "Mapping[str, Any] | None", what: str, *, attempts: int = 4,
         sleep: Sleep = time.sleep, timeout: float = 120.0) -> Any:
    headers = {"User-Agent": C.HTTP_USER_AGENT}
    last: "Exception | str | None" = None
    for i in range(1, attempts + 1):
        try:
            resp = session.get(url, params=params, headers=headers, timeout=timeout)
        except Exception as exc:  # network error: retry, then fail loudly
            last = exc
        else:
            if resp.status_code == 200:
                return resp
            last = f"HTTP {resp.status_code}"
        if i < attempts:
            sleep(2.0 * i)
    raise PublicDataError(f"{what} failed after {attempts} attempts: {last}")


# --- XLSX (stdlib) ---------------------------------------------------------------------------------------------------------
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def _col_index(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):  # type: ignore[union-attr]
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_xlsx_sheets(path: "str | Path") -> "dict[str, list[list[Any]]]":
    """{sheet name: rows (lists of str / float / None)} for a plain .xlsx file."""
    out: dict[str, list[list[Any]]] = {}
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")) for si in root.findall("m:si", _NS)]
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
        for sh in wb.find("m:sheets", _NS):  # type: ignore[union-attr]
            target = rels[sh.get(f"{{{_NS['r']}}}id")]
            part = target.lstrip("/") if target.startswith("/") else f"xl/{target}"
            rows: list[list[Any]] = []
            for row in ET.fromstring(z.read(part)).iter(f"{{{_NS['m']}}}row"):
                cells: dict[int, Any] = {}
                for c in row.findall("m:c", _NS):
                    kind, v = c.get("t"), c.find("m:v", _NS)
                    if kind == "inlineStr":
                        val: Any = "".join(t.text or "" for t in c.iter(f"{{{_NS['m']}}}t"))
                    elif v is None or v.text is None:
                        continue
                    elif kind == "s":
                        val = shared[int(v.text)]
                    elif kind in ("str", "e"):
                        val = v.text
                    elif kind == "b":
                        val = float(v.text)
                    else:
                        val = float(v.text)
                    cells[_col_index(c.get("r"))] = val
                if cells:
                    r = [None] * (max(cells) + 1)
                    for k, val in cells.items():
                        r[k] = val
                    rows.append(r)
            out[sh.get("name")] = rows
    return out


def _num(v: Any) -> "float | None":
    if v is None:
        return None
    try:
        return float(str(v).replace(",", ".").strip())
    except ValueError:
        return None


def _excel_date(serial: float) -> str:
    return (pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(serial))).strftime("%Y-%m-%d")


def parse_plants(path: "str | Path") -> "tuple[pd.DataFrame, dict[str, Any]]":
    """Operational plants of the alphabetical sheet (the one with Name / Municipality / Nominal power columns).

    Returns (all rows incl. flags, report). A plant is operational if its Installed flag is 1 and it has no demolish year.
    The summary sheet's totals (count, cumulative MW, status date) are recorded and compared; a mismatch is reported.
    """
    sheets = read_xlsx_sheets(path)
    table, sheet_name = None, None
    for name, rows in sheets.items():
        for i, r in enumerate(rows):
            low = [str(x).lower() if x is not None else "" for x in r]
            if any(x.startswith("name") for x in low) and any("municipality" in x for x in low) and any("nominal power" in x for x in low):
                table, sheet_name, head = rows[i + 1:], name, low
                break
        if table is not None:
            break
    if table is None:
        raise PublicDataError("plant list: no sheet with 'Name', 'Municipality' and 'Nominal power' columns found")

    def col(*needles: str) -> int:
        for j, h in enumerate(head):
            if all(n in h for n in needles):
                return j
        raise PublicDataError(f"plant list: column {needles} not found in {head}")

    ci = {"name": col("name"), "municipality": col("municipality"), "nominal_mw": col("nominal power"), "year": col("installation year"),
          "region": col("region"), "installed": col("installed"), "demolish_year": col("demolish")}
    recs = []
    for r in table:
        get = lambda k: r[ci[k]] if ci[k] < len(r) else None  # noqa: E731
        if not get("name") or _num(get("nominal_mw")) is None:
            continue
        recs.append({"name": str(get("name")).strip(), "municipality": str(get("municipality") or "").strip(),
                     "nominal_mw": round(_num(get("nominal_mw")) or 0.0, 6), "year": _num(get("year")),
                     "region": str(get("region") or "").strip() or None, "installed_flag": _num(get("installed")),
                     "demolish_year": _num(get("demolish_year"))})
    df = pd.DataFrame(recs)
    df["operational"] = (df["installed_flag"] == 1) & df["demolish_year"].isna()
    summary: dict[str, Any] = {}
    for name, rows in sheets.items():
        for r in rows:
            txt = " ".join(str(x) for x in r if isinstance(x, str)).lower()
            nums = [x for x in r if isinstance(x, float)]
            if "kumulatiivinen" in txt and "mw" in txt and nums:
                summary["cumulative_mw"] = nums[0]
            elif "kumulatiivinen" in txt and ("kpl" in txt or "lukum" in txt) and nums:
                summary["cumulative_count"] = nums[0]
        for r in rows[:8]:
            nums = [x for x in r if isinstance(x, float) and 40000 < x < 60000]
            if nums and "status_date" not in summary and not any(isinstance(x, str) and len(x) > 12 for x in r):
                summary["status_date"] = _excel_date(nums[0])
    op = df[df["operational"]]
    report = {"sheet": sheet_name, "rows": int(len(df)), "operational_plants": int(len(op)), "operational_mw": float(round(op["nominal_mw"].sum(), 6)),
              "not_operational": df.loc[~df["operational"], ["name", "municipality", "nominal_mw"]].to_dict("records"),
              "summary_sheet": summary,
              "summary_consistent": bool("cumulative_mw" in summary and abs(summary["cumulative_mw"] - op["nominal_mw"].sum()) < 1e-6
                                         and summary.get("cumulative_count", len(op)) == len(op)),
              "largest": None if op.empty else {"name": op.loc[op["nominal_mw"].idxmax(), "name"], "municipality": op.loc[op["nominal_mw"].idxmax(), "municipality"],
                                                "nominal_mw": float(op["nominal_mw"].max())}}
    return df.reset_index(drop=True), report


# --- WFS -----------------------------------------------------------------------------------------------------------------------
def _xml_text(root: ET.Element, tag: str, within: "ET.Element | None" = None) -> "str | None":
    el = (within if within is not None else root).find(f".//{{*}}{tag}")
    return None if el is None or el.text is None else el.text.strip()


def discover_layer(session: Any, layer: str, sleep: Sleep = time.sleep) -> dict[str, Any]:
    """GetCapabilities entry (title, abstract, CRS, bbox, metadata URL, fees/constraints) of a layer."""
    resp = _get(session, WFS_URL, {"service": "WFS", "version": "2.0.0", "request": "GetCapabilities"}, "WFS GetCapabilities", sleep=sleep)
    root = ET.fromstring(resp.content if hasattr(resp, "content") else resp.text)
    for ft in root.findall(".//{*}FeatureType"):
        if _xml_text(ft, "Name") == layer:
            md = ft.find("{*}MetadataURL")
            return {"name": layer, "title": _xml_text(ft, "Title"), "abstract": _xml_text(ft, "Abstract"), "default_crs": _xml_text(ft, "DefaultCRS"),
                    "metadata_url": None if md is None else md.get("{http://www.w3.org/1999/xlink}href"),
                    "fees": _xml_text(root, "Fees"), "access_constraints": _xml_text(root, "AccessConstraints")}
    raise PublicDataError(f"WFS layer {layer!r} not found in GetCapabilities of {WFS_URL}")


def describe_fields(session: Any, layer: str, sleep: Sleep = time.sleep) -> list[str]:
    resp = _get(session, WFS_URL, {"service": "WFS", "version": "2.0.0", "request": "DescribeFeatureType", "typeNames": layer},
                f"WFS DescribeFeatureType {layer}", sleep=sleep)
    root = ET.fromstring(resp.content if hasattr(resp, "content") else resp.text)
    return [e.get("name") for e in root.iter("{http://www.w3.org/2001/XMLSchema}element") if e.get("type") and e.get("name")]


def choose_fields(available: Sequence[str], candidates: Mapping[str, Sequence[str]], layer: str, required: Sequence[str]) -> dict[str, "str | None"]:
    out: dict[str, "str | None"] = {}
    for role, names in candidates.items():
        out[role] = next((n for n in names if n in available), None)
    missing = [r for r in required if out.get(r) is None]
    if missing:
        raise PublicDataError(f"layer {layer}: attributes for {missing} not found; available: {sorted(available)}")
    return out


def fetch_features(session: Any, layer: str, *, sort_by: str, property_names: "Sequence[str] | None" = None, page_size: int = WFS_PAGE,
                   sleep: Sleep = time.sleep) -> list[dict[str, Any]]:
    """All features of a layer as GeoJSON dicts, paged with startIndex + sortBy (stable order)."""
    feats: list[dict[str, Any]] = []
    start = 0
    while True:
        params = {"service": "WFS", "version": "2.0.0", "request": "GetFeature", "typeNames": layer, "outputFormat": "application/json",
                  "count": page_size, "startIndex": start, "sortBy": sort_by}
        if property_names:
            params["propertyName"] = ",".join(property_names)
        payload = _get(session, WFS_URL, params, f"WFS GetFeature {layer} startIndex={start}", sleep=sleep).json()
        got = payload.get("features", [])
        feats.extend(got)
        if len(got) < page_size:
            break
        start += len(got)
    return feats


def find_licence_text(session: Any, metadata_url: "str | None", capabilities: Mapping[str, Any], sleep: Sleep = time.sleep) -> dict[str, Any]:
    """Best-effort licence / attribution text: WFS Fees/AccessConstraints, then the layer's metadata record. Records what was NOT found."""
    found: dict[str, Any] = {"wfs_fees": capabilities.get("fees"), "wfs_access_constraints": capabilities.get("access_constraints"),
                             "metadata_url": metadata_url, "snippets": [], "metadata_fetch": None}
    if metadata_url:
        try:
            resp = _get(session, metadata_url, None, "layer metadata record", attempts=2, sleep=sleep, timeout=60.0)
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text = re.sub(r"\s+", " ", text)
            for m in re.finditer("|".join(re.escape(h) for h in LICENCE_HINTS), text, flags=re.I):
                snip = text[max(0, m.start() - 80): m.end() + 160].strip()
                if snip not in found["snippets"]:
                    found["snippets"].append(snip)
                if len(found["snippets"]) >= 4:
                    break
            found["metadata_fetch"] = "ok"
        except PublicDataError as exc:
            found["metadata_fetch"] = f"failed: {exc}"
    found["licence_stated_in_service"] = bool(found["wfs_fees"] or found["wfs_access_constraints"] or found["snippets"])
    return found


# --- cache I/O -------------------------------------------------------------------------------------------------------------------
def _meta(kind: str, url: str, payload: bytes, **extra: Any) -> dict[str, Any]:
    return {"kind": kind, "source_url": url, "retrieved_utc": utc_now_iso(), "bytes": len(payload), "sha256": sha256_bytes(payload), **extra}


def system_trust_session() -> Any:
    """requests.Session that verifies TLS against the operating-system trust store (full verification stays ON).

    requests' bundled certifi list can miss roots/intermediates that the OS trusts (e.g. a managed Windows machine), which
    makes legitimate public sites fail verification. An explicit REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE is respected instead.
    """
    import os
    import ssl

    import requests
    from requests.adapters import HTTPAdapter

    session = requests.Session()
    if os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE"):
        return session

    class _SystemStoreAdapter(HTTPAdapter):
        def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
            kwargs["ssl_context"] = ssl.create_default_context()     # CERT_REQUIRED + hostname check, system CA store
            super().init_poolmanager(*args, **kwargs)

    session.mount("https://", _SystemStoreAdapter())
    return session


def fetch_public(cache_dir: "str | Path", session: Any = None, *, refresh: bool = False, sleep: Sleep = time.sleep,
                 say: Callable[[str], None] = lambda m: None, plants_url: str = PLANTS_URL) -> dict[str, Any]:
    """Download (network) the three public inputs into `cache_dir` unless cached; returns a summary of what was written/kept."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    if session is None:
        session = system_trust_session()
    done: dict[str, Any] = {}
    if refresh or not (cache / PLANTS_META).is_file():
        raw = _get(session, plants_url, None, "plant list download", sleep=sleep).content
        atomic_write_bytes(cache / PLANTS_FILE, raw)
        df, rep = parse_plants(cache / PLANTS_FILE)       # fail before the metadata makes the cache look complete
        atomic_write_json(cache / PLANTS_META, _meta(
            "plants", plants_url, raw, report=rep, licence={"name": None, "note": "no licence statement found in the workbook; owner: Suomen Uusiutuvat ry "
                                                                                "(cite as Suomen Uusiutuvat ry, 2025)"}, attribution="Suomen Uusiutuvat ry (2025)"))
        done["plants"] = "downloaded"
        say(f"[ingest-solar-public] plants: {rep['operational_plants']} operational, {rep['operational_mw']:.2f} MW")
    else:
        done["plants"] = "cached"
    if refresh or not (cache / POP_META).is_file():
        cap = discover_layer(session, POP_LAYER, sleep)
        avail = describe_fields(session, POP_LAYER, sleep)
        fields = choose_fields(avail, POP_FIELD_CANDIDATES, POP_LAYER, ("x", "y", "inhabitants"))
        props = [v for v in fields.values() if v]
        feats = fetch_features(session, POP_LAYER, sort_by="oid" if "oid" in avail else props[0], property_names=props, sleep=sleep)
        rows = [{"cell_id": (f["properties"].get(fields["cell_id"]) if fields["cell_id"] else None), "x": f["properties"][fields["x"]],
                 "y": f["properties"][fields["y"]], "inhabitants": f["properties"][fields["inhabitants"]]} for f in feats]
        df = pd.DataFrame(rows)
        csv = df.to_csv(index=False).encode("utf-8")
        atomic_write_bytes(cache / POP_FILE, csv)
        lic = find_licence_text(session, cap.get("metadata_url"), cap, sleep)
        lic.update(ASSUMED_LICENCE)
        atomic_write_json(cache / POP_META, _meta("population_grid", WFS_URL, csv, layer=POP_LAYER, field_mapping=fields, features=len(df), capabilities=cap,
                                                  crs=CRS_FINLAND, licence=lic, attribution="Statistics Finland (population grid 1 km x 1 km, 2025)",
                                                  coordinate_note="xkoord/ykoord are the south-west corner of the 1 km cell in EPSG:3067; the centroid is +500 m"))
        done["population"] = "downloaded"
        say(f"[ingest-solar-public] population grid: {len(df)} features")
    else:
        done["population"] = "cached"
    if refresh or not (cache / MUNI_META).is_file():
        cap = discover_layer(session, MUNI_LAYER, sleep)
        fields = choose_fields(describe_fields(session, MUNI_LAYER, sleep), MUNI_FIELD_CANDIDATES, MUNI_LAYER, ("code", "name_fi"))
        feats = fetch_features(session, MUNI_LAYER, sort_by="id", sleep=sleep, page_size=5000)
        blob = json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False).encode("utf-8")
        atomic_write_bytes(cache / MUNI_FILE, blob)
        lic = find_licence_text(session, cap.get("metadata_url"), cap, sleep)
        lic.update(ASSUMED_LICENCE)
        atomic_write_json(cache / MUNI_META, _meta("municipalities", WFS_URL, blob, layer=MUNI_LAYER, field_mapping=fields, features=len(feats), capabilities=cap,
                                                   crs=CRS_FINLAND, licence=lic, attribution="Statistics Finland (municipalities 2025, 1:1 000 000)"))
        done["municipalities"] = "downloaded"
        say(f"[ingest-solar-public] municipalities: {len(feats)} features")
    else:
        done["municipalities"] = "cached"
    return done


def public_cache_missing(cache_dir: "str | Path") -> list[str]:
    c = Path(cache_dir)
    return [name for name, metaf, f in (("plant list", PLANTS_META, PLANTS_FILE), ("population grid", POP_META, POP_FILE),
                                        ("municipalities", MUNI_META, MUNI_FILE)) if not ((c / metaf).is_file() and (c / f).is_file())]


@dataclass
class PublicData:
    plants: pd.DataFrame            # all rows with `operational` flag
    plants_report: dict[str, Any]
    population: pd.DataFrame        # cell_id, x, y (SW corner, EPSG:3067), inhabitants
    municipalities: pd.DataFrame    # code, name_fi, name_sv, name_en, x, y (centroid, EPSG:3067), lon, lat
    meta: dict[str, dict[str, Any]]


def municipality_table(geojson_path: "str | Path", fields: Mapping[str, "str | None"]) -> pd.DataFrame:
    """Municipality centroids (shapely centroid in EPSG:3067, then WGS84) with all name variants."""
    from pyproj import Transformer
    from shapely.geometry import shape

    feats = json.loads(Path(geojson_path).read_text(encoding="utf-8"))["features"]
    rows = []
    for f in feats:
        g = shape(f["geometry"])
        if g.is_empty:
            continue
        c = g.centroid
        p = f["properties"]
        rows.append({"code": str(p.get(fields["code"])), "name_fi": p.get(fields["name_fi"]) if fields.get("name_fi") else None,
                     "name_sv": p.get(fields["name_sv"]) if fields.get("name_sv") else None,
                     "name_en": p.get(fields["name_en"]) if fields.get("name_en") else None, "x": float(c.x), "y": float(c.y)})
    df = pd.DataFrame(rows)
    if df.empty or not (df["x"].between(0, 1.2e6).all() and df["y"].between(6.0e6, 7.9e6).all()):
        raise PublicDataError("municipality centroids are not plausible EPSG:3067 coordinates (x 0-1.2e6, y 6.0e6-7.9e6)")
    lon, lat = Transformer.from_crs(CRS_FINLAND, "EPSG:4326", always_xy=True).transform(df["x"].to_numpy(), df["y"].to_numpy())
    df["lon"], df["lat"] = lon, lat
    return df


def load_public(cache_dir: "str | Path") -> PublicData:
    c = Path(cache_dir)
    missing = public_cache_missing(c)
    if missing:
        raise FileNotFoundError(f"public-data cache incomplete in {c}: missing {missing}; run `ingest-solar-public`")
    meta = {k: json.loads((c / f).read_text(encoding="utf-8")) for k, f in (("plants", PLANTS_META), ("population", POP_META), ("municipalities", MUNI_META))}
    plants, rep = parse_plants(c / PLANTS_FILE)
    pop = pd.read_csv(c / POP_FILE)
    munis = municipality_table(c / MUNI_FILE, meta["municipalities"]["field_mapping"])
    return PublicData(plants, rep, pop, munis, meta)


# --- municipality matching ------------------------------------------------------------------------------------------------------------
_SUFFIX = re.compile(r"\b(kunta|kaupunki|stad|kommun|city|municipality)\b", re.I)


def normalise_name(name: Any, fold: bool = False) -> str:
    """Case/whitespace/punctuation-insensitive key; `fold` also maps a/o/a-ring and diacritics to ASCII."""
    s = unicodedata.normalize("NFC", str(name or "")).casefold()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = _SUFFIX.sub(" ", s)
    s = re.sub(r"[^\w]+", " ", s.replace("-", " "))
    if fold:
        s = unicodedata.normalize("NFKD", s.replace("ä", "a").replace("ö", "o").replace("å", "a"))
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).strip()


def _name_variants(*names: Any) -> list[str]:
    out: list[str] = []
    for n in names:
        if n is None or (isinstance(n, float) and np.isnan(n)):
            continue
        parts = re.split(r"\s*[/,]\s*|\s+-\s+", str(n))
        out.extend([str(n), *parts])
    return [v for v in out if v.strip()]


def match_municipality(name: str, munis: pd.DataFrame) -> pd.Series:
    """The one municipality row whose Finnish / Swedish / other name equals `name` (exact key first, then ASCII-folded)."""
    for fold in (False, True):
        index: dict[str, set[int]] = {}
        for i, r in munis.iterrows():
            for v in _name_variants(r.get("name_fi"), r.get("name_sv"), r.get("name_en")):
                index.setdefault(normalise_name(v, fold), set()).add(i)
        hits: set[int] = set()
        for v in _name_variants(name):
            hits |= index.get(normalise_name(v, fold), set())
        codes = {munis.loc[i, "code"] for i in hits}
        if len(codes) == 1:
            return munis.loc[sorted(hits)[0]]
        if len(codes) > 1:
            raise MunicipalityMatchError(f"municipality {name!r} is ambiguous: codes {sorted(codes)}")
    raise MunicipalityMatchError(f"municipality {name!r} not found among {len(munis)} municipalities")


def place_plants(plants: pd.DataFrame, munis: pd.DataFrame) -> pd.DataFrame:
    """Operational plants with the centroid of their municipality; every plant must map or MunicipalityMatchError is raised."""
    op = plants[plants["operational"]].copy()
    failures: list[str] = []
    rows = []
    for _, p in op.iterrows():
        try:
            m = match_municipality(p["municipality"], munis)
        except MunicipalityMatchError as exc:
            failures.append(f"{p['name']}: {exc}")
            continue
        rows.append({"municipality_code": m["code"], "municipality_name": m["name_fi"], "lat": float(m["lat"]), "lon": float(m["lon"])})
    if failures:
        raise MunicipalityMatchError(f"{len(failures)} plant(s) could not be placed: " + "; ".join(failures))
    return pd.concat([op.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
