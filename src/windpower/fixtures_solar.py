"""SYNTHETIC fixtures for the solar pipeline's public inputs and Fingrid solar series (offline fakes, no network).

Everything is stamped SYNTHETIC: the plant list (an .xlsx written with the standard library, same layout as the real
workbook), a fake WFS (GetCapabilities / DescribeFeatureType / paged GetFeature) serving a population grid and a
municipality lattice in EPSG:3067, and fake Fingrid 267 / 248 / 362 rows (362 with several production types per hour).
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from . import config as C
from .fixtures import FakeResponse, FingridFixture, STAMP_TEXT, _ZIP_DATE, fingrid_rows
from .io_utils import SYNTHETIC_STAMP, atomic_write_with
from . import solar_public as SPUB

PLANT_CAPACITIES_FIRST29 = [103.2, 32.0, 13.0, 13.0, 9.5, 7.0, 5.2, 5.0, 5.0, 5.0, 4.9, 5.1, 4.3, 4.0, 3.8, 3.8, 2.8, 2.8, 2.0, 2.0, 2.0, 2.0, 2.0,
                            1.98, 1.98, 1.7, 1.5, 1.1, 1.0]
LARGE_TOTAL_MW = 250.66


# --- XLSX writer (stdlib) ---------------------------------------------------------------------------------------------------------
def _cell(ref: str, v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f'<c r="{ref}"><v>{float(v)!r}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(v))}</t></is></c>'


def _col_letter(i: int) -> str:
    return chr(ord("A") + i)


def _sheet(rows: "list[list[Any]]") -> str:
    body = "".join(f'<row r="{r}">' + "".join(_cell(f"{_col_letter(c)}{r}", v) for c, v in enumerate(row)) + "</row>" for r, row in enumerate(rows, start=1))
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>{body}</sheetData></worksheet>'


def write_xlsx(path: "str | Path", sheets: "dict[str, list[list[Any]]]") -> Path:
    names = list(sheets)
    ct = ('<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
          '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
          + "".join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                    for i in range(1, len(names) + 1)) + "</Types>")
    rels = ('<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
    wb = ('<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
          + "".join(f'<sheet name="{escape(n)}" sheetId="{i}" r:id="rId{i}"/>' for i, n in enumerate(names, start=1)) + "</sheets></workbook>")
    wbrels = ('<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
              + "".join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
                        for i in range(1, len(names) + 1)) + "</Relationships>")

    def build(out: Path) -> None:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, blob in (("[Content_Types].xml", ct), ("_rels/.rels", rels), ("xl/workbook.xml", wb), ("xl/_rels/workbook.xml.rels", wbrels),
                               *((f"xl/worksheets/sheet{i}.xml", _sheet(sheets[n])) for i, n in enumerate(names, start=1))):
                zf.writestr(zipfile.ZipInfo(name, _ZIP_DATE), blob, zipfile.ZIP_DEFLATED)

    return atomic_write_with(path, build)


# --- public-data fixture -------------------------------------------------------------------------------------------------------------
@dataclass
class PublicFixture:
    grid: C.GridSpec
    plants_xlsx: bytes
    plant_rows: list[dict[str, Any]]                # all rows incl. not-in-use / demolished
    municipalities: list[dict[str, Any]]            # GeoJSON features in EPSG:3067
    population: list[dict[str, Any]]                # GeoJSON features (SW corner x/y, vaesto)
    outside_inhabitants: int                        # inhabitants placed outside the grid on purpose
    large_mw: float = LARGE_TOTAL_MW
    licence_text: "str | None" = None
    meta: dict[str, Any] = field(default_factory=dict)


def synthetic_public(grid: C.GridSpec = C.SOLAR_SMOKE_GRID, seed: int = 2026) -> PublicFixture:
    """SYNTHETIC plants (30 operational summing to 250.66 MW + 2 not operational), municipality lattice, population cells.

    Municipalities are 40 km squares tiling the grid bbox (shrunk by 0.2 deg so every centroid lies inside the grid); their
    names carry Finnish letters and a Swedish variant. Plants are listed by Finnish name, Swedish name, upper case or with
    stray blanks. A few population cells lie outside the grid on purpose.
    """
    from pyproj import Transformer

    rng = np.random.default_rng(seed)
    to3067 = Transformer.from_crs("EPSG:4326", "EPSG:3067", always_xy=True)
    x0, y0 = to3067.transform(grid.lon_min + 0.2, grid.lat_min + 0.2)
    x1, y1 = to3067.transform(grid.lon_max - 0.2, grid.lat_max - 0.2)
    size = 40_000.0
    nx, ny = max(2, int((x1 - x0) // size)), max(2, int((y1 - y0) // size))
    munis, k = [], 0
    for iy in range(ny):
        for ix in range(nx):
            xa, ya = x0 + ix * size, y0 + iy * size
            ring = [[xa, ya], [xa + size, ya], [xa + size, ya + size], [xa, ya + size], [xa, ya]]
            munis.append({"type": "Feature", "id": f"kunta1000k_2025.{k + 1}", "geometry": {"type": "Polygon", "coordinates": [ring]},
                          "properties": {"id": k + 1, "kunta": f"{900 + k:03d}", "vuosi": 2025, "nimi": f"Syntääri {k:03d}", "namn": f"Syntby {k:03d}",
                                         "name": f"Syntääri {k:03d}"}})
            k += 1
    # plants: spread over the lattice, written in different styles
    caps = PLANT_CAPACITIES_FIRST29 + [round(LARGE_TOTAL_MW - sum(PLANT_CAPACITIES_FIRST29), 2)]
    order = rng.permutation(len(munis))
    rows: list[dict[str, Any]] = []
    for n, cap in enumerate(caps):
        m = munis[int(order[n % len(munis)])]["properties"]
        style = n % 4
        name = {0: m["nimi"], 1: m["namn"], 2: m["nimi"].upper(), 3: f"  {m['nimi'].lower()} "}[style]
        rows.append({"name": f"SYNTHETIC Plant {n:02d}", "municipality": name, "nominal_mw": cap, "year": 2018 + n % 8, "region": "Synteettinen",
                     "installed": 1, "demolish_year": None, "code": m["kunta"]})
    rows.append({"name": "SYNTHETIC Not In Use", "municipality": munis[0]["properties"]["nimi"], "nominal_mw": 7.0, "year": 2015, "region": "Synteettinen",
                 "installed": 0, "demolish_year": None, "code": munis[0]["properties"]["kunta"]})
    rows.append({"name": "SYNTHETIC Demolished", "municipality": munis[1]["properties"]["nimi"], "nominal_mw": 3.0, "year": 2012, "region": "Synteettinen",
                 "installed": 1, "demolish_year": 2024, "code": munis[1]["properties"]["kunta"]})
    head = [None, "Name", "Municipality", "Owner / Owner of SPV", "SPV", "Nominal power\n(MW)", "No of panels", "Installation year", "Region",
            "Installed (1)/\nNot in use(0)", "Demolish year"]
    sheet2 = [[None, f"{SYNTHETIC_STAMP} fixture - solar PV parks (not real data)"], [], head] + [
        [None, r["name"], r["municipality"], "SYNTHETIC", None, r["nominal_mw"], None, r["year"], r["region"], r["installed"], r["demolish_year"]] for r in rows]
    n_op = sum(1 for r in rows if r["installed"] == 1 and r["demolish_year"] is None)
    sheet3 = [[None, f"{SYNTHETIC_STAMP} summary"], [], [None, "Asennettu yli 1 MW aurinkovoimakapasiteetti"], [None, 45838.0],
              [None, "Voimaloiden lukumäärä - kumulatiivinen ", float(n_op), "kpl"], [None, "Kapasiteetti - kumulatiivinen (MW)", LARGE_TOTAL_MW, "MW"]]
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="fixture_plants_")) / "plants.xlsx"
    write_xlsx(tmp, {"Aurinkovoimalat": [[f"{SYNTHETIC_STAMP} fixture"]], "Aakkosjärjestys": sheet2, "Tiivistelmä": sheet3})
    blob = tmp.read_bytes()
    tmp.unlink()

    # population: cells around municipality centroids (1 km cells, SW corner on the 1 km lattice) + a few outside the grid
    cells: dict[tuple[int, int], int] = {}
    for m in munis:
        ring = m["geometry"]["coordinates"][0]
        cx, cy = (ring[0][0] + ring[2][0]) / 2, (ring[0][1] + ring[2][1]) / 2
        for _ in range(int(rng.integers(15, 40))):
            gx, gy = int((cx + rng.normal(0, 8000)) // 1000) * 1000, int((cy + rng.normal(0, 8000)) // 1000) * 1000
            cells[(gx, gy)] = cells.get((gx, gy), 0) + int(rng.integers(1, 1500))
    ox, oy = to3067.transform(grid.lon_min - 1.5, grid.lat_min - 1.0)
    outside = {}
    for q in range(3):
        outside[(int(ox // 1000) * 1000 + 1000 * q, int(oy // 1000) * 1000)] = 100 + 50 * q
    cells.update(outside)
    pop = [{"type": "Feature", "id": f"vaki2025_1km.{i + 1}", "geometry": None,
            "properties": {"oid": i + 1, "grd_id": f"1kmN{y // 1000}E{x // 1000:04d}", "id_nro": i, "xkoord": x, "ykoord": y, "kunta": "000",
                           "vaesto": v, "miehet": -1, "naiset": -1}} for i, ((x, y), v) in enumerate(sorted(cells.items()))]
    return PublicFixture(grid, blob, rows, munis, pop, int(sum(outside.values())), LARGE_TOTAL_MW)


class FakeHttpResponse:
    def __init__(self, status: int, content: bytes = b"", payload: Any = None):
        self.status_code, self.content, self._payload = status, content, payload

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return self._payload if self._payload is not None else json.loads(self.content)


class FakePublicSession:
    """Offline stand-in for requests.Session serving the plant workbook and a WFS (capabilities, describe, paged GetFeature)."""

    def __init__(self, fx: PublicFixture, fail_first: int = 0, licence_in_metadata: bool = False):
        self.fx, self.calls, self.fail_first = fx, [], fail_first
        self.licence_in_metadata = licence_in_metadata

    def _caps(self) -> bytes:
        def ft(name: str, title: str, meta: str) -> str:
            return (f'<FeatureType><Name>{name}</Name><Title>{title}</Title><Abstract>{SYNTHETIC_STAMP} {title}</Abstract>'
                    f'<DefaultCRS>urn:ogc:def:crs:EPSG::3067</DefaultCRS><MetadataURL xlink:href="https://fake.invalid/meta/{meta}"/></FeatureType>')
        return ('<?xml version="1.0"?><WFS_Capabilities xmlns="http://www.opengis.net/wfs/2.0" xmlns:ows="http://www.opengis.net/ows/1.1" '
                'xmlns:xlink="http://www.w3.org/1999/xlink"><ows:ServiceIdentification><ows:Title>SYNTHETIC WFS</ows:Title></ows:ServiceIdentification>'
                f'<FeatureTypeList>{ft(SPUB.POP_LAYER, "Population grid", "pop")}{ft(SPUB.MUNI_LAYER, "Municipalities", "muni")}</FeatureTypeList>'
                "</WFS_Capabilities>").encode()

    @staticmethod
    def _describe(layer: str) -> bytes:
        names = (("oid", "xsd:long"), ("grd_id", "xsd:string"), ("xkoord", "xsd:long"), ("ykoord", "xsd:long"), ("vaesto", "xsd:long")) if layer == SPUB.POP_LAYER \
            else (("id", "xsd:int"), ("kunta", "xsd:string"), ("nimi", "xsd:string"), ("namn", "xsd:string"), ("name", "xsd:string"))
        els = "".join(f'<xsd:element name="{n}" nillable="true" type="{t}"/>' for n, t in names)
        return f'<?xml version="1.0"?><xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema"><xsd:element name="{layer}">{els}</xsd:element></xsd:schema>'.encode()

    def get(self, url: str, params: "dict[str, Any] | None" = None, headers: "dict[str, str] | None" = None, timeout: Any = None) -> FakeHttpResponse:
        p = dict(params or {})
        self.calls.append((url, p))
        if self.fail_first > 0:
            self.fail_first -= 1
            return FakeHttpResponse(503)
        if url == SPUB.PLANTS_URL:
            return FakeHttpResponse(200, self.fx.plants_xlsx)
        if url.startswith("https://fake.invalid/meta/"):
            if self.licence_in_metadata:
                return FakeHttpResponse(200, b"<md>Use constraints: CC BY 4.0 licence, attribution to the producer</md>")
            return FakeHttpResponse(404)
        if url != SPUB.WFS_URL:
            return FakeHttpResponse(404)
        req = p.get("request")
        if req == "GetCapabilities":
            return FakeHttpResponse(200, self._caps())
        if req == "DescribeFeatureType":
            return FakeHttpResponse(200, self._describe(p["typeNames"]))
        if req == "GetFeature":
            feats = self.fx.population if p["typeNames"] == SPUB.POP_LAYER else self.fx.municipalities
            start, count = int(p.get("startIndex", 0)), int(p["count"])
            page = feats[start: start + count]
            if p.get("propertyName"):
                keep = set(str(p["propertyName"]).split(","))
                page = [{**f, "properties": {k: v for k, v in f["properties"].items() if k in keep}} for f in page]
            return FakeHttpResponse(200, payload={"type": "FeatureCollection", "features": page})
        return FakeHttpResponse(400)


# --- Fingrid solar fixture ------------------------------------------------------------------------------------------------------------
SOLAR_FG_NAMES = {267: "Total production capacity used in the solar power forecast",
                  248: "Solar power generation forecast - updated every 15 minutes",
                  362: "Small-scale electricity surplus production by production type at accounting points in Finnish distribution networks"}
SOLAR_FG_UNITS = {267: "MW", 248: "MWh/h", 362: "kWh"}
SOLAR_FG_PERIOD = {267: "1 h", 248: "15 min", 362: "1 h"}
PRODUCTION_TYPES = ("AV01", "AV02", "AV03", "AV06", "AV07")


def synthetic_solar_fingrid(hours: pd.DatetimeIndex, seed: int = 2026, missing_248_hours: Any = (), missing_362_hours: Any = (),
                            capacity_start_mw: float = 1400.0) -> FingridFixture:
    """SYNTHETIC rows for 267 (hourly MW), 248 (15-min MWh/h, small negative at night), 362 (hourly kWh, several production types per hour).

    `hourly` holds the exact hourly truth the rows average to: capacity_267_mw, forecast_248_mw, surplus_362_kwh (AV06).
    AV06 is deliberately NOT the largest type, so a filter on value instead of on ProductionType would fail the tests.
    """
    from . import solar_geometry as G

    rng = np.random.default_rng(seed)
    n = len(hours)
    cz = np.maximum(G.cos_zenith(hours + pd.Timedelta(minutes=30), np.array([61.5]), np.array([25.0]))[:, 0, 0], 0.0)
    shape = cz ** 1.1
    cap = capacity_start_mw + 0.4 * np.arange(n)
    f248 = np.where(shape > 0, cap * 0.2 * shape * (1 + 0.03 * rng.standard_normal(n)), -0.6)
    kwh = np.round(cap * 1000.0 * 0.05 * shape * (1 + 0.05 * rng.standard_normal(n)), 1)
    truth = pd.DataFrame({"capacity_267_mw": cap, "forecast_248_mw": f248, "surplus_362_kwh": kwh}, index=hours)

    def iso(t: pd.Timestamp) -> str:
        return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    rows267 = [{"datasetId": 267, "startTime": iso(t), "endTime": iso(t + pd.Timedelta(hours=1)), "value": round(float(v), 6)} for t, v in truth["capacity_267_mw"].items()]
    rows248 = fingrid_rows(truth["forecast_248_mw"], 900, 248, missing_248_hours)
    skip362 = {pd.Timestamp(h) for h in missing_362_hours}
    rows362 = []
    for t, v in truth["surplus_362_kwh"].items():
        if t in skip362:
            continue
        for typ in PRODUCTION_TYPES:
            val = float(v) if typ == "AV06" else float(rng.uniform(2e4, 9e5))   # other types are much larger than solar
            rows362.append({"datasetId": 362, "startTime": iso(t), "endTime": iso(t + pd.Timedelta(hours=1)), "value": round(val, 1),
                            "additionalJson": {"ProductionType": typ, "Uom": "KWH", "Res": "PT1H", "Value": round(val, 1), "Count": int(rng.integers(500, 90000))}})
    meta = {i: {"id": i, "nameEn": f"{SYNTHETIC_STAMP} {SOLAR_FG_NAMES[i]}", "unitEn": SOLAR_FG_UNITS[i], "dataPeriodEn": SOLAR_FG_PERIOD[i],
                "license": STAMP_TEXT, "synthetic": SYNTHETIC_STAMP} for i in SOLAR_FG_UNITS}
    return FingridFixture({267: rows267, 248: rows248, 362: rows362}, meta, truth, [])


__all__ = ["synthetic_public", "PublicFixture", "FakePublicSession", "write_xlsx", "synthetic_solar_fingrid", "FakeResponse", "LARGE_TOTAL_MW"]
