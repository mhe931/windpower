"""Fingrid solar series: 267 capacity, 248 forecast, 362 small-scale surplus (AV06 among several production types)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from windpower import fingrid as FG
from windpower import fixtures as F
from windpower import fixtures_solar as FP
from windpower import fingrid_solar as FS

HOURS = pd.date_range("2025-07-01", periods=72, freq="h")


def client_for(fx, **session_kw):
    sess = F.FakeFingridSession(fx, **session_kw)
    return FG.FingridClient(F.FAKE_API_KEY, sess, request_delay=0, sleep=lambda s: None, now=lambda: F.FIXED_ISO), sess


@pytest.fixture(scope="module")
def fx():
    return FP.synthetic_solar_fingrid(HOURS, seed=1, missing_248_hours=[HOURS[10]], missing_362_hours=[HOURS[20], HOURS[21]])


def test_series_definitions_carry_native_unit_derivation_and_scope():
    assert FS.EXPECTED_UNITS == {267: "MW", 248: "MWh/h", 362: "kWh"}
    s = {i: FS.series_record(i) for i in FS.SERIES}
    assert s[267]["scope"].startswith("capacity used in Fingrid's solar forecast") and "not official" in s[267]["scope"]
    assert "forecast, not measured generation" in s[248]["scope"] and s[248]["native_unit"] == "MWh/h" and s[248]["derived_unit"] == "MW"
    assert "surplus only (after own consumption), not total generation" in s[362]["scope"] and s[362]["native_unit"] == "kWh"
    assert "AV06" in s[362]["time_aggregation"] and "/ 1000" in s[362]["time_aggregation"]
    assert all(set(r) >= {"id", "name", "native_unit", "derived_unit", "time_aggregation", "scope", "period"} for r in s.values())
    assert set(FG.DATASETS) == {181, 248, 124, 319} and FG.EXPECTED_UNITS[248] == "MW"      # the wind client's tables are untouched


def test_fetch_hourly_units_and_aggregation(tmp_path, fx):
    client, sess = client_for(fx)
    data = FS.load_or_fetch_solar(tmp_path, HOURS[0], HOURS[-1], client=client)
    h, truth = data.hourly, fx.hourly
    assert not data.unavailable and set(data.reports) == {267, 248, 362} and not data.from_cache_only
    np.testing.assert_allclose(h["capacity_267_mw"], truth["capacity_267_mw"], atol=1e-5)              # MW stays MW
    ok = h["forecast_248_mw"].notna()
    np.testing.assert_allclose(h.loc[ok, "forecast_248_mw"], truth.loc[ok, "forecast_248_mw"], atol=1e-5)  # MWh/h of 4 quarter hours -> mean MW
    np.testing.assert_allclose(h["surplus_362_kwh"].dropna(), truth.loc[h["surplus_362_kwh"].notna(), "surplus_362_kwh"], atol=1e-5)
    np.testing.assert_allclose(h["surplus_362_mw"], h["surplus_362_kwh"] / 1000.0)                      # kWh per 1 h -> MW = /1000
    assert h["surplus_362_count"].dropna().between(500, 90000).all()
    assert any(p.get("pageSize") for _, p in sess.calls)


def test_248_keeps_negative_night_values_and_reports_the_minimum(tmp_path, fx):
    data = FS.load_or_fetch_solar(tmp_path, HOURS[0], HOURS[-1], client=client_for(fx)[0])
    rep = data.reports[248]
    assert rep["min"] == pytest.approx(-0.6, abs=1e-6) and rep["n_negative"] > 10 and rep["native_unit"] == "MWh/h" and rep["derived_unit"] == "MW"
    assert (data.hourly["forecast_248_mw"].dropna() < 0).any()                                           # not clipped to zero
    assert rep["resolution_seconds"] == 900 and rep["samples_per_hour_typical"] == 4


def test_missing_hours_stay_nan_and_are_listed(tmp_path, fx):
    data = FS.load_or_fetch_solar(tmp_path, HOURS[0], HOURS[-1], client=client_for(fx)[0])
    assert np.isnan(data.hourly.loc[HOURS[10], "forecast_248_mw"]) and data.reports[248]["missing_hours"] == ["2025-07-01T10:00:00Z"]
    assert data.hourly.loc[[HOURS[20], HOURS[21]], ["surplus_362_kwh", "surplus_362_mw"]].isna().all().all()
    assert data.reports[362]["missing_hours"] == ["2025-07-01T20:00:00Z", "2025-07-01T21:00:00Z"]
    assert data.hourly["capacity_267_mw"].notna().all() and len(data.hourly) == 72


def test_362_is_filtered_on_production_type_not_on_value(tmp_path, fx):
    raw = pd.DataFrame(FS.flatten_rows(fx.rows[362]))
    assert set(raw["production_type"]) == set(FP.PRODUCTION_TYPES) and len(raw) == 70 * 5            # two hours have no rows at all
    sol, seen = FS.solar_rows_362(raw)
    assert set(sol["production_type"]) == {"AV06"} and len(sol) == 70 and seen["production_types_seen"]["AV06"] == 70
    biggest = raw.groupby("production_type")["value"].sum().idxmax()
    assert biggest != "AV06"                                                    # a value-based pick would take the wrong type
    data = FS.load_or_fetch_solar(tmp_path, HOURS[0], HOURS[-1], client=client_for(fx)[0])
    assert data.hourly["surplus_362_kwh"].max() < 1e5 < raw["value"].max()
    assert data.reports[362]["solar_rows"] == 70 and data.reports[362]["uom"] == ["KWH"] and data.reports[362]["res"] == ["PT1H"]


def test_362_contract_violations_fail_loudly():
    base = FS.flatten_rows(FP.synthetic_solar_fingrid(HOURS[:4], seed=2).rows[362])
    with pytest.raises(FG.FingridError, match="no AV06"):
        FS.solar_rows_362(base[base["production_type"] != "AV06"])
    dup = pd.concat([base, base[base["production_type"] == "AV06"].iloc[[0]]])
    with pytest.raises(FG.FingridError, match="duplicate"):
        FS.solar_rows_362(dup)
    with pytest.raises(FG.FingridError, match="unit"):
        FS.solar_rows_362(base.assign(uom=np.where(base["production_type"] == "AV06", "MWH", base["uom"])))
    with pytest.raises(FG.FingridError, match="PT1H"):
        FS.solar_rows_362(base.assign(res="PT15M"))
    conflict = base.copy()
    conflict.loc[conflict["production_type"] == "AV06", "value_conflict"] = True
    with pytest.raises(FG.FingridError, match="disagree"):
        FS.solar_rows_362(conflict)
    with pytest.raises(FG.FingridError, match="ProductionType"):
        FS.solar_rows_362(base.drop(columns="production_type"))


def test_flatten_rows_reads_additional_json_as_dict_or_string():
    row = {"datasetId": 362, "startTime": "2025-07-01T00:00:00.000Z", "endTime": "2025-07-01T01:00:00.000Z", "value": 12.5}
    a = FS.flatten_rows([{**row, "additionalJson": {"ProductionType": "AV06", "Uom": "KWH", "Res": "PT1H", "Value": "12,5", "Count": 7}},
                         {**row, "value": None, "additionalJson": json.dumps({"ProductionType": "AV01", "Uom": "KWH", "Res": "PT1H", "Value": 3, "Count": 1})}])
    assert list(a["production_type"]) == ["AV06", "AV01"] and list(a["value"]) == [12.5, 3.0] and not a["value_conflict"].any() and list(a["count"]) == [7.0, 1.0]


def test_paging_rate_limit_and_cache_reuse(tmp_path, fx):
    client, sess = client_for(fx, max_rows_per_window=None, fail_429_once=True)
    client.page_size = 100                                                     # force several pages for 362 (360 rows) and 248 (288 rows)
    data = FS.load_or_fetch_solar(tmp_path, HOURS[0], HOURS[-1], client=client)
    assert client.request_count > 6 and data.hourly["surplus_362_kwh"].notna().sum() == 70
    n = client.request_count
    again = FS.load_or_fetch_solar(tmp_path, HOURS[0], HOURS[-1], client=client)          # complete cache: no new request
    assert client.request_count == n and again.hourly.equals(data.hourly)
    cached = FS.load_or_fetch_solar(tmp_path, HOURS[0], HOURS[-1], client=None)            # and it works without any client
    assert not cached.unavailable and cached.from_cache_only and cached.hourly.equals(data.hourly)
    for p in tmp_path.rglob("*"):
        if p.is_file():
            assert F.FAKE_API_KEY not in p.read_text(encoding="utf-8", errors="ignore")  # the key is never cached
    meta = json.loads((tmp_path / "fingrid_raw_362_surplus_362_all_types.meta.json").read_text())
    assert meta["source_url"].endswith("/datasets/362/data") and meta["native_unit"] == "kWh" and meta["series_key"] == "surplus_362"


def test_without_a_key_the_series_are_unavailable_with_the_exact_credential(tmp_path, fx):
    data = FS.load_or_fetch_solar(tmp_path, HOURS[0], HOURS[-1], client=None)
    assert set(data.unavailable) == {267, 248, 362} and data.hourly.empty or data.hourly.shape[1] == 0
    assert all("FINGRID_API_KEY" in r for r in data.unavailable.values()) and len(data.hourly) == 72
    assert "FINGRID_API_KEY" in FS.key_missing_message()
    # a partial cache: only 267 fetched earlier -> 267 available, the other two named as unavailable
    client, _ = client_for(fx)
    FS.fetch_raw(client, 267, *FG.window_for_hours(HOURS[0], HOURS[-1]), tmp_path)
    part = FS.load_or_fetch_solar(tmp_path, HOURS[0], HOURS[-1], client=None, key_missing_reason="no key (test)")
    assert list(part.hourly.columns) == ["capacity_267_mw"] and set(part.unavailable) == {248, 362} and part.unavailable[248] == "no key (test)"


def test_incomplete_cache_triggers_a_refetch_but_362_hourly_tail_counts_as_complete(tmp_path, fx):
    client, _ = client_for(fx)
    FS.load_or_fetch_solar(tmp_path, HOURS[0], HOURS[-1], client=client)
    start, end = FG.window_for_hours(HOURS[0], HOURS[-1])
    raw, meta = FG.read_raw_cache(tmp_path, 362, FS.RAW_COLUMN[362])
    assert FS._cache_complete(raw, meta, start, end, 3600)                      # last start 23:00 of an hourly series is complete
    assert not FS._cache_complete(raw, meta, start, "2025-07-05T00:00:00Z", 3600)


def test_definition_check_records_name_unit_period_and_flags_mismatches(fx):
    client, _ = client_for(fx)
    block = FS.verify_solar_datasets(client)
    assert block["overall"] == "verified"
    for rec in block["datasets"]:
        assert rec["status"] == "verified" and rec["reported_unit"] == FS.EXPECTED_UNITS[rec["dataset_id"]]
        assert rec["period_status"] == "verified" and rec["name_status"] == "verified" and rec["expected_period"] in ("1 h", "15 min")
    for ds_id, wrong in ((248, "MW"), (362, "MWh"), (267, "kW")):
        bad_client, _ = client_for(fx, unit_overrides={ds_id: wrong})
        with pytest.raises(FG.DatasetMismatchError) as e:
            FS.verify_solar_datasets(bad_client)
        assert str(ds_id) in str(e.value) and any(r["status"] == "mismatch" for r in e.value.records)
    assert FS.verify_solar_datasets(None)["overall"] == "unverified"
    meta = {**fx.metadata, 248: {**fx.metadata[248], "dataPeriodEn": "1 h"}}
    odd = FS.verify_solar_datasets(client_for(F.FingridFixture(fx.rows, meta, fx.hourly, []))[0])
    assert [r for r in odd["datasets"] if r["dataset_id"] == 248][0]["period_status"] == "mismatch"
