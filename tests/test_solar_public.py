"""Public inputs: XLSX plant list, WFS fetch/cache (fake session), municipality matching, field discovery."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from windpower import config as C
from windpower import fixtures_solar as FP
from windpower import solar_public as SP


def test_plant_workbook_parses_to_30_operational_plants_summing_to_250_66(tmp_path, public_fx):
    path = tmp_path / "plants.xlsx"
    path.write_bytes(public_fx.plants_xlsx)
    df, rep = SP.parse_plants(path)
    assert rep["sheet"] == "Aakkosjärjestys" and rep["rows"] == 32
    assert rep["operational_plants"] == 30 and rep["operational_mw"] == pytest.approx(250.66, abs=1e-9)
    assert {r["name"] for r in rep["not_operational"]} == {"SYNTHETIC Not In Use", "SYNTHETIC Demolished"}      # flag 0 and demolish year are excluded
    assert rep["summary_consistent"] and rep["summary_sheet"]["cumulative_mw"] == 250.66 and rep["summary_sheet"]["status_date"] == "2025-06-30"
    assert rep["largest"] == {"name": "SYNTHETIC Plant 00", "municipality": rep["largest"]["municipality"], "nominal_mw": 103.2}
    assert df["operational"].sum() == 30 and df["nominal_mw"].dtype == float


def test_plant_workbook_without_the_expected_columns_fails_loudly(tmp_path):
    p = FP.write_xlsx(tmp_path / "bad.xlsx", {"Sheet": [["a", "b"], [1, 2]]})
    with pytest.raises(SP.PublicDataError, match="Municipality"):
        SP.parse_plants(p)


def test_summary_mismatch_is_reported_not_hidden(tmp_path, public_fx):
    rows = [[None, "Name", "Municipality", None, None, "Nominal power (MW)", None, "Installation year", "Region", "Installed (1)/ Not in use(0)", "Demolish year"],
            [None, "A", "Oulu", None, None, 10.0, None, 2020, "x", 1, None]]
    summary = [[None, "Kapasiteetti - kumulatiivinen (MW)", 99.0, "MW"]]
    p = FP.write_xlsx(tmp_path / "m.xlsx", {"list": rows, "sum": summary})
    _, rep = SP.parse_plants(p)
    assert rep["operational_mw"] == 10.0 and rep["summary_sheet"]["cumulative_mw"] == 99.0 and rep["summary_consistent"] is False


def test_fetch_public_writes_cache_with_provenance_and_is_idempotent(tmp_path, public_fx):
    sess = FP.FakePublicSession(public_fx)
    done = SP.fetch_public(tmp_path, sess, sleep=lambda s: None)
    assert done == {"plants": "downloaded", "population": "downloaded", "municipalities": "downloaded"}
    calls = len(sess.calls)
    assert SP.public_cache_missing(tmp_path) == []
    metas = {k: json.loads((tmp_path / f).read_text(encoding="utf-8")) for k, f in (("plants", SP.PLANTS_META), ("pop", SP.POP_META), ("muni", SP.MUNI_META))}
    assert metas["plants"]["source_url"] == SP.PLANTS_URL and metas["plants"]["report"]["operational_plants"] == 30 and len(metas["plants"]["sha256"]) == 64
    assert metas["pop"]["layer"] == SP.POP_LAYER and metas["pop"]["field_mapping"] == {"cell_id": "grd_id", "x": "xkoord", "y": "ykoord", "inhabitants": "vaesto"}
    assert metas["pop"]["features"] == len(public_fx.population) and metas["pop"]["crs"] == "EPSG:3067"
    assert metas["muni"]["field_mapping"]["name_sv"] == "namn" and metas["muni"]["capabilities"]["default_crs"].endswith("3067")
    for m in metas.values():
        assert m["retrieved_utc"].endswith("Z") and m["attribution"]
    assert metas["pop"]["licence"]["verified"] is False and metas["pop"]["licence"]["assumed_name"] == "CC BY 4.0"
    assert metas["pop"]["licence"]["licence_stated_in_service"] is False and "failed" in metas["pop"]["licence"]["metadata_fetch"]     # 404 is recorded
    again = SP.fetch_public(tmp_path, sess, sleep=lambda s: None)
    assert set(again.values()) == {"cached"} and len(sess.calls) == calls                       # no request when cached
    SP.fetch_public(tmp_path, sess, refresh=True, sleep=lambda s: None)
    assert len(sess.calls) > calls                                                              # --refresh fetches again


def test_licence_text_found_in_metadata_is_recorded(tmp_path, public_fx):
    SP.fetch_public(tmp_path, FP.FakePublicSession(public_fx, licence_in_metadata=True), sleep=lambda s: None)
    lic = json.loads((tmp_path / SP.POP_META).read_text(encoding="utf-8"))["licence"]
    assert lic["licence_stated_in_service"] is True and any("CC BY 4.0" in s for s in lic["snippets"])


def test_wfs_paging_uses_start_index_and_returns_every_feature(public_fx):
    sess = FP.FakePublicSession(public_fx)
    feats = SP.fetch_features(sess, SP.POP_LAYER, sort_by="oid", property_names=["xkoord", "ykoord", "vaesto"], page_size=7, sleep=lambda s: None)
    assert len(feats) == len(public_fx.population) and len({f["id"] for f in feats}) == len(feats)
    starts = [p["startIndex"] for _, p in sess.calls if p.get("request") == "GetFeature"]
    assert starts[:3] == [0, 7, 14] and all(p.get("sortBy") == "oid" for _, p in sess.calls)
    assert set(feats[0]["properties"]) == {"xkoord", "ykoord", "vaesto"}                          # propertyName restricts the payload


def test_transient_http_errors_are_retried_and_persistent_ones_fail_loudly(tmp_path, public_fx):
    SP.fetch_public(tmp_path, FP.FakePublicSession(public_fx, fail_first=2), sleep=lambda s: None)     # 2 x HTTP 503, then fine
    assert SP.public_cache_missing(tmp_path) == []
    with pytest.raises(SP.PublicDataError, match="failed after 4 attempts: HTTP 503"):
        SP.fetch_public(tmp_path / "other", FP.FakePublicSession(public_fx, fail_first=99), sleep=lambda s: None)
    assert SP.public_cache_missing(tmp_path / "other")                                              # nothing half-written looks complete


def test_field_mapping_is_discovered_and_missing_fields_fail():
    assert SP.choose_fields(["oid", "grd_id", "xkoord", "ykoord", "vaesto"], SP.POP_FIELD_CANDIDATES, "L", ("x", "y", "inhabitants"))["inhabitants"] == "vaesto"
    with pytest.raises(SP.PublicDataError, match="inhabitants"):
        SP.choose_fields(["xkoord", "ykoord"], SP.POP_FIELD_CANDIDATES, "L", ("x", "y", "inhabitants"))
    with pytest.raises(SP.PublicDataError, match="not found in GetCapabilities"):
        SP.discover_layer(FP.FakePublicSession(FP.synthetic_public(C.SOLAR_SMOKE_GRID)), "nope:nothing")


def test_load_public_gives_population_and_municipality_centroids_inside_the_grid(public_cache, public_fx):
    pub = SP.load_public(public_cache)
    assert len(pub.population) == len(public_fx.population) and pub.population["inhabitants"].sum() == sum(f["properties"]["vaesto"] for f in public_fx.population)
    m = pub.municipalities
    assert len(m) == len(public_fx.municipalities) and m["lat"].between(C.SOLAR_SMOKE_GRID.lat_min, C.SOLAR_SMOKE_GRID.lat_max).all()
    assert m["lon"].between(C.SOLAR_SMOKE_GRID.lon_min, C.SOLAR_SMOKE_GRID.lon_max).all()
    # centroid of a 40 km square in EPSG:3067 is its centre
    ring = public_fx.municipalities[0]["geometry"]["coordinates"][0]
    assert m.loc[0, "x"] == pytest.approx((ring[0][0] + ring[2][0]) / 2) and m.loc[0, "y"] == pytest.approx((ring[0][1] + ring[2][1]) / 2)
    with pytest.raises(FileNotFoundError, match="ingest-solar-public"):
        SP.load_public(public_cache / "missing")


MUNIS = pd.DataFrame({"code": ["005", "620", "322", "910", "911"],
                      "name_fi": ["Alajärvi", "Pyhäjärvi", "Kemiönsaari", "Raasepori", "Äänekoski"],
                      "name_sv": ["Alajärvi", "Pyhäjärvi", "Kimitoön", "Raseborg", "Äänekoski"],
                      "name_en": ["Alajärvi", "Pyhäjärvi", "Kimitoön", "Raasepori", "Äänekoski"],
                      "x": 1.0, "y": 1.0, "lon": [24.0, 25.9, 22.6, 23.4, 25.7], "lat": [63.0, 63.7, 60.1, 60.0, 62.6]})


@pytest.mark.parametrize("text, code", [("Pyhäjärvi", "620"), ("PYHÄJÄRVI", "620"), ("  pyhäjärvi \n", "620"), ("Pyhajarvi", "620"), ("Kimitoön", "322"),
                                        ("kemiönsaari", "322"), ("Raasepori / Raseborg", "910"), ("Raseborg", "910"), ("Äänekosken kaupunki", None),
                                        ("Äänekoski (kaupunki)", "911"), ("aanekoski", "911"), ("Alajärvi kunta", "005")])
def test_municipality_names_match_robustly(text, code):
    if code is None:
        with pytest.raises(SP.MunicipalityMatchError):        # an inflected form is not guessed
            SP.match_municipality(text, MUNIS)
    else:
        assert SP.match_municipality(text, MUNIS)["code"] == code


def test_ambiguous_and_unknown_municipalities_raise():
    dup = pd.concat([MUNIS, MUNIS.iloc[[1]].assign(code="999")], ignore_index=True)
    with pytest.raises(SP.MunicipalityMatchError, match="ambiguous"):
        SP.match_municipality("Pyhäjärvi", dup)
    with pytest.raises(SP.MunicipalityMatchError, match="not found"):
        SP.match_municipality("Atlantis", MUNIS)


def test_every_plant_must_map_or_the_placement_fails_loudly(public_cache):
    pub = SP.load_public(public_cache)
    placed = SP.place_plants(pub.plants, pub.municipalities)
    assert len(placed) == 30 and placed["lat"].notna().all() and placed["municipality_code"].notna().all()
    # all four writing styles of the fixture (Finnish, Swedish, UPPER, padded lower case) mapped to the right municipality code
    expected = pd.DataFrame(pub.plants[pub.plants["operational"]]).reset_index(drop=True)
    assert len(expected) == len(placed)
    broken = pub.plants.copy()
    broken.loc[broken.index[0], "municipality"] = "Nowhereville"
    broken.loc[broken.index[1], "municipality"] = "Elsewhere"
    with pytest.raises(SP.MunicipalityMatchError, match="2 plant"):
        SP.place_plants(broken, pub.municipalities)
    assert np.isclose(placed["nominal_mw"].sum(), 250.66)
