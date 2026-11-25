from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from windpower import fingrid as G
from windpower import fixtures as F
from windpower.io_utils import CredentialUnavailableError


@pytest.fixture()
def two_days(pipeline):
    obs = pipeline.obs.iloc[:48]
    miss = [obs.index[5], obs.index[6]]
    return obs, miss, F.synthetic_fingrid(obs, miss)


def make_client(fx, sleeps=None, *, page_size=10_000, **session_kw):
    sess = F.FakeFingridSession(fx, **session_kw)
    sleeps = [] if sleeps is None else sleeps
    client = G.FingridClient(F.FAKE_API_KEY, sess, request_delay=0, page_size=page_size, sleep=sleeps.append,
                             now=lambda: "2025-10-01T12:00:00Z")
    return client, sess, sleeps


# --- credentials -------------------------------------------------------------------
def test_api_key_from_env_dotenv_and_absent(tmp_path):
    assert G.get_api_key({"FINGRID_API_KEY": " abc "}) == "abc"
    env_file = tmp_path / ".env"
    env_file.write_text('# c\nOTHER=1\nFINGRID_API_KEY="from-file"\n')
    assert G.get_api_key({}, dotenv=env_file) == "from-file"
    assert G.get_api_key({"FINGRID_API_KEY": "env-wins"}, dotenv=env_file) == "env-wins"
    with pytest.raises(CredentialUnavailableError, match="Fingrid credential unavailable"):
        G.get_api_key({}, dotenv=tmp_path / "missing.env")
    (tmp_path / "empty.env").write_text("FINGRID_API_KEY=\n")
    with pytest.raises(CredentialUnavailableError):
        G.get_api_key({}, dotenv=tmp_path / "empty.env")


def test_client_requires_key_and_hides_it(tmp_path):
    with pytest.raises(CredentialUnavailableError):
        G.FingridClient(env={}, dotenv=tmp_path / "none")
    c = G.FingridClient("secret-value", session=object())
    assert "secret-value" not in repr(c) and "secret-value" not in str(vars(c).get("session", ""))


def test_window_for_hours():
    assert G.window_for_hours("2025-07-01T00:00:00", "2025-09-30T23:00:00") == ("2025-07-01T00:00:00Z", "2025-09-30T23:59:59Z")


# --- paging, rate limit, retries ----------------------------------------------------
def test_pagination_collects_every_row(two_days, tmp_path):
    obs, miss, fx = two_days
    client, sess, sleeps = make_client(fx, page_size=100, fail_429_once=False)
    start, end = G.window_for_hours(obs.index[0], obs.index[-1])
    raw, meta = client.fetch_dataset(181, start, end, tmp_path)
    assert len(raw) == len(fx.rows[181]) == meta["row_count"]
    pages = [p["page"] for u, p in sess.calls if u.endswith("/181/data")]
    n_pages = -(-len(raw) // 100)
    assert pages[:n_pages] == list(range(1, n_pages + 1)) and n_pages == 10   # 46 h x 20 samples; then an empty follow-up query
    assert all(p["pageSize"] == 100 and p["sortBy"] == "startTime" for u, p in sess.calls)
    assert raw["startTime"].is_unique and raw["startTime"].is_monotonic_increasing


def test_429_honours_retry_after_then_succeeds(two_days, tmp_path):
    obs, _, fx = two_days
    client, sess, sleeps = make_client(fx)                 # first data call answers 429, Retry-After: 2
    start, end = G.window_for_hours(obs.index[0], obs.index[-1])
    raw, meta = client.fetch_dataset(248, start, end, tmp_path)
    assert [s for s in sleeps if s] == [2.0] and len(raw) == len(fx.rows[248])   # zero sleeps are the configurable request delay
    assert meta["requests"] == 3 and client.request_count == 3     # 429, data page, empty follow-up window


def test_persistent_errors_fail_after_max_attempts_with_backoff(two_days):
    class Down:
        def get(self, *a, **k):
            return F.FakeResponse(503, {})

    sleeps: list[float] = []
    c = G.FingridClient("k", Down(), request_delay=0, sleep=sleeps.append, backoff=10)
    with pytest.raises(G.FingridError, match="failed after 5 attempts: HTTP 503"):
        c.fetch_metadata(181)
    assert sleeps == [10, 20, 30, 40]


def test_transport_exception_is_retried():
    calls = []

    class Flaky:
        def get(self, *a, **k):
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("reset")
            return F.FakeResponse(200, {"unitEn": "MW"})

    c = G.FingridClient("k", Flaky(), sleep=lambda s: None)
    assert c.fetch_metadata(181) == {"unitEn": "MW"} and len(calls) == 3


def test_rejected_key_fails_immediately_without_leaking_key(two_days):
    _, _, fx = two_days
    sess = F.FakeFingridSession(fx, expected_key="other")
    sleeps: list[float] = []
    c = G.FingridClient("my-wrong-key", sess, sleep=sleeps.append)
    with pytest.raises(G.FingridError, match="HTTP 403") as ei:
        c.fetch_metadata(181)
    assert "my-wrong-key" not in str(ei.value) and sleeps == [] and len(sess.calls) == 1
    assert "my-wrong-key" not in repr(sess.calls)


def test_windowing_when_server_caps_rows(two_days, tmp_path):
    obs, _, fx = two_days
    client, sess, _ = make_client(fx, fail_429_once=False, max_rows_per_window=300)
    start, end = G.window_for_hours(obs.index[0], obs.index[-1])
    raw, meta = client.fetch_dataset(181, start, end, tmp_path)
    assert len(raw) == len(fx.rows[181]) and raw["startTime"].is_unique
    starts = {p["startTime"] for u, p in sess.calls if u.endswith("/181/data")}
    assert len(starts) >= 2 and start in starts          # follow-up windows start after the last row


def test_empty_dataset_is_an_error(two_days, tmp_path):
    obs, _, fx = two_days
    fx.rows[181] = []
    client, _, _ = make_client(fx, fail_429_once=False)
    with pytest.raises(G.FingridError, match="returned 0 rows"):
        client.fetch_dataset(181, *G.window_for_hours(obs.index[0], obs.index[-1]), tmp_path)


# --- cache ------------------------------------------------------------------------------
def test_cache_files_metadata_and_atomicity(two_days, tmp_path):
    obs, _, fx = two_days
    client, _, _ = make_client(fx)
    start, end = G.window_for_hours(obs.index[0], obs.index[-1])
    raw, meta = client.fetch_dataset(181, start, end, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fingrid_raw_181_wind_power_mw.csv", "fingrid_raw_181_wind_power_mw.meta.json"]
    cached, cmeta = G.read_raw_cache(tmp_path, 181)
    assert len(cached) == len(raw)
    assert cmeta["dataset_id"] == 181 and cmeta["fetched_at_utc"] == "2025-10-01T12:00:00Z"
    assert (cmeta["window_start"], cmeta["window_end"]) == (start, end)
    assert cmeta["row_count"] == len(raw) and cmeta["complete"] is True and cmeta["first_start_utc"] == start
    assert set(cached.columns) >= {"datasetId", "startTime", "endTime", "value"}


def test_complete_cache_runs_without_client_or_network(two_days, tmp_path):
    obs, miss, fx = two_days
    client, sess, _ = make_client(fx)
    first, last = obs.index[0], obs.index[-1]
    data = G.load_or_fetch_hourly(tmp_path, first, last, client=client)
    assert data.from_cache_only is False
    n_calls = len(sess.calls)
    again = G.load_or_fetch_hourly(tmp_path, first, last, client=None)            # no key, no session
    assert again.from_cache_only is True and len(sess.calls) == n_calls
    pd.testing.assert_frame_equal(again.hourly, data.hourly)
    assert set(again.cache_meta) == set(G.DATASETS.values())


def test_incomplete_cache_without_credentials_fails_clearly(two_days, tmp_path):
    obs, _, fx = two_days
    with pytest.raises(CredentialUnavailableError, match=r"dataset 181 \(wind_power_mw\).*missing or incomplete"):
        G.load_or_fetch_hourly(tmp_path, obs.index[0], obs.index[-1], client=None)
    # partially cached -> still names the first incomplete series
    client, _, _ = make_client(fx)
    client.fetch_dataset(181, *G.window_for_hours(obs.index[0], obs.index[-1]), tmp_path)
    with pytest.raises(CredentialUnavailableError, match="dataset 248"):
        G.load_or_fetch_hourly(tmp_path, obs.index[0], obs.index[-1], client=None)


def test_resume_fetches_only_the_missing_tail(two_days, tmp_path):
    obs, _, fx = two_days
    client, sess, _ = make_client(fx, fail_429_once=False)
    start, _ = G.window_for_hours(obs.index[0], obs.index[-1])
    mid_end = G.window_for_hours(obs.index[0], obs.index[23])[1]
    full_end = G.window_for_hours(obs.index[0], obs.index[-1])[1]
    part, pmeta = client.fetch_dataset(181, start, mid_end, tmp_path)
    assert pmeta["window_end"] == mid_end and len(part) == 22 * 20       # two wind hours are missing on day 1
    sess.calls.clear()
    full, fmeta = client.fetch_dataset(181, start, full_end, tmp_path)
    assert len(full) == len(fx.rows[181]) and full["startTime"].is_unique
    first_params = [p for u, p in sess.calls if u.endswith("/data")][0]
    assert first_params["startTime"] == "2025-07-01T23:57:01Z"          # last cached sample 23:57:00 + 1 s
    assert fmeta["pages_fetched"] >= pmeta["pages_fetched"] + 1 and fmeta["window_end"] == full_end


def test_refresh_refetches(two_days, tmp_path):
    obs, _, fx = two_days
    client, sess, _ = make_client(fx, fail_429_once=False)
    first, last = obs.index[0], obs.index[-1]
    G.load_or_fetch_hourly(tmp_path, first, last, client=client)
    n = len(sess.calls)
    G.load_or_fetch_hourly(tmp_path, first, last, client=client)
    assert len(sess.calls) == n
    G.load_or_fetch_hourly(tmp_path, first, last, client=client, refresh=True)
    assert len(sess.calls) > n


def test_cache_with_later_window_start_is_not_reused(two_days, tmp_path):
    obs, _, fx = two_days
    client, sess, _ = make_client(fx, fail_429_once=False)
    late_start = G.window_for_hours(obs.index[12], obs.index[-1])
    client.fetch_dataset(181, *late_start, tmp_path)
    full = G.window_for_hours(obs.index[0], obs.index[-1])
    raw, meta = client.fetch_dataset(181, *full, tmp_path)
    assert meta["window_start"] == full[0] and raw["startTime"].min() < late_start[0]


# --- hourly rule and missing hours ----------------------------------------------------------
def test_hourly_mean_rule_recovers_truth_for_3min_and_15min(two_days, tmp_path):
    obs, miss, fx = two_days
    client, _, _ = make_client(fx)
    data = G.load_or_fetch_hourly(tmp_path, obs.index[0], obs.index[-1], client=client)
    h = data.hourly
    ok = ~h.index.isin(miss)
    for col in ("solar_power_mw", "total_consumption_mw", "imbalance_price_eur_mwh"):
        np.testing.assert_allclose(h[col], obs[col], atol=1e-5)
    np.testing.assert_allclose(h.loc[ok, "wind_power_mw"], obs.loc[ok, "wind_power_mw"], atol=1e-5)
    assert data.reports["wind_power_mw"]["resolution_seconds"] == 180
    assert data.reports["solar_power_mw"]["resolution_seconds"] == 900
    assert data.reports["wind_power_mw"]["samples_per_hour_typical"] == 20
    assert "mean of all samples" in data.reports["wind_power_mw"]["aggregation"]
    assert h.index.name == "valid_time" and h.index.tz is None and h.index.is_monotonic_increasing and len(h) == 48


def test_missing_hours_are_reported_not_imputed(two_days, tmp_path):
    obs, miss, fx = two_days
    client, _, _ = make_client(fx)
    data = G.load_or_fetch_hourly(tmp_path, obs.index[0], obs.index[-1], client=client)
    rep = data.reports["wind_power_mw"]
    assert rep["missing_hours"] == ["2025-07-01T05:00:00Z", "2025-07-01T06:00:00Z"]
    assert rep["expected_hours"] == 48 and rep["present_hours"] == 46
    assert data.hourly["wind_power_mw"].isna().sum() == 2
    assert data.reports["solar_power_mw"]["missing_hours"] == []
    nl = data.hourly["net_load_mw"]
    assert nl.isna().sum() == 2                                   # net load inherits the gap
    ok = ~data.hourly["wind_power_mw"].isna()
    np.testing.assert_allclose(nl[ok], (data.hourly["total_consumption_mw"] - data.hourly["wind_power_mw"] - data.hourly["solar_power_mw"])[ok])


def test_partial_hours_are_flagged_but_averaged():
    ts = pd.date_range("2025-07-01", periods=3 * 4, freq="15min")
    raw = pd.DataFrame({"startTime": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "value": [1, 1, 1, 1, 2, 2, 2, 2, 5, 7, 0, 0]})
    raw = raw.drop(index=[10, 11])                                  # hour 02:00 keeps only 2 of 4 samples
    s, rep = G.to_hourly(raw, "x", "2025-07-01T00:00:00", "2025-07-01T03:00:00")
    assert s.tolist()[:3] == [1.0, 2.0, 6.0] and np.isnan(s.iloc[3])
    assert rep["partial_hours"] == ["2025-07-01T02:00:00Z"] and rep["missing_hours"] == ["2025-07-01T03:00:00Z"]


def test_hourly_labels_hour_start_and_ignores_samples_outside_window():
    raw = pd.DataFrame({"startTime": ["2025-06-30T23:57:00Z", "2025-07-01T00:00:00Z", "2025-07-01T00:30:00Z", "2025-07-01T01:00:00Z"],
                        "value": [100, 2.0, 4.0, 50]})
    s, _ = G.to_hourly(raw, "x", "2025-07-01T00:00:00", "2025-07-01T00:00:00")
    assert s.tolist() == [3.0] and str(s.index[0]) == "2025-07-01 00:00:00"


def test_non_utc_offsets_are_converted():
    raw = pd.DataFrame({"startTime": ["2025-07-01T03:00:00+03:00"], "value": [9.0]})
    s, _ = G.to_hourly(raw, "x", "2025-07-01T00:00:00", "2025-07-01T00:00:00")
    assert s.tolist() == [9.0]


# --- dataset metadata verification -------------------------------------------------------------
def test_verify_datasets_ok_and_recorded(two_days):
    _, _, fx = two_days
    client, _, _ = make_client(fx, fail_429_once=False)
    block = G.verify_datasets(client)
    assert block["overall"] == "verified"
    r181 = block["datasets"][0]
    assert (r181["dataset_id"], r181["reported_unit"], r181["expected_unit"], r181["status"]) == (181, "MW", "MW", "verified")
    assert r181["frequency"] == "3 min" and "SYNTHETIC" in r181["name"] and r181["retrieved_utc"] == "2025-10-01T12:00:00Z"
    assert [d["reported_unit"] for d in block["datasets"]] == ["MW", "MW", "MW", "EUR/MWh"]


def test_verify_datasets_unit_mismatch_fails_loudly(two_days):
    _, _, fx = two_days
    client, _, _ = make_client(fx, fail_429_once=False, unit_overrides={181: "MWh", 319: "c/kWh"})
    with pytest.raises(G.DatasetMismatchError, match=r"dataset 181: reported unit 'MWh' != expected 'MW'") as ei:
        G.verify_datasets(client)
    assert [r["status"] for r in ei.value.records] == ["mismatch", "verified", "verified", "mismatch"]
    soft = G.verify_datasets(client, strict=False)
    assert soft["overall"] == "partial" and soft["datasets"][3]["reason"].startswith("reported unit 'c/kWh'")


def test_verify_datasets_without_credentials_is_unverified():
    block = G.verify_datasets(None)
    assert block["overall"] == "unverified"
    assert all(d["status"] == "unverified" and "FINGRID_API_KEY" in d["reason"] for d in block["datasets"])


def test_verify_handles_missing_unit_and_euro_symbol(two_days):
    _, _, fx = two_days
    fx.metadata[181].pop("unitEn")
    fx.metadata[319]["unitEn"] = "€/MWh"
    client, _, _ = make_client(fx, fail_429_once=False)
    block = G.verify_datasets(client)
    assert block["datasets"][0]["status"] == "unit_not_reported" and block["datasets"][3]["status"] == "verified"
    assert G.normalise_unit("EUR / MWh") == "eur/mwh" and G.normalise_unit(None) is None


# --- joined table -----------------------------------------------------------------------------
def test_join_keeps_era5_hours_and_counts_comparable(pipeline, joined):
    n = len(pipeline.sim.hourly)
    assert len(joined) == n and joined["actual_wind_mw"].isna().sum() == 3
    _, counts = G.join_hourly(pipeline.sim.theoretical_mw, joined.rename(columns={"actual_wind_mw": "wind_power_mw"}), pipeline.sim.installed_mw)
    assert counts == {"era5_hours": n, "joined_rows": n, "comparable_hours": n - 3, "dropped_missing_actual": 3,
                      "join_coverage": (n - 3) / n}
    comp = G.comparable_rows(joined)
    assert len(comp) == n - 3 and comp["actual_wind_mw"].notna().all()
    np.testing.assert_allclose(comp["residual_mw"], comp["theoretical_wind_mw"] - comp["actual_wind_mw"])
    np.testing.assert_allclose(comp["theoretical_cf"], comp["theoretical_wind_mw"] / pipeline.sim.installed_mw)


def test_join_only_uses_era5_hours(pipeline):
    theo = pipeline.sim.theoretical_mw.iloc[:24]
    fg = pipeline.obs.rename(columns={"wind_power_mw": "wind_power_mw"})
    df, counts = G.join_hourly(theo, fg, 100.0)
    assert len(df) == 24 == counts["era5_hours"] and counts["comparable_hours"] == 24


def test_join_reports_hours_missing_in_fingrid(pipeline):
    theo = pipeline.sim.theoretical_mw.iloc[:24]
    fg = pipeline.obs.iloc[:20]                                    # Fingrid stops early: 4 ERA5 hours have no row
    df, counts = G.join_hourly(theo, fg, 100.0, era5_hours=theo.index)
    assert counts["era5_hours"] == 24 and counts["joined_rows"] == 24 and counts["comparable_hours"] == 20
    assert counts["dropped_missing_actual"] == 4
