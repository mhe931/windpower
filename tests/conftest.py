"""Shared offline fixtures. Everything is SYNTHETIC and lives in pytest tmp dirs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from windpower import config as C
from windpower import fixtures as F
from windpower import fixtures_solar as FP

START = "2025-07-01T00:00:00"
HOURS = 48
END = "2025-07-02T23:00:00"


@pytest.fixture(autouse=True)
def _offline_and_credential_free(monkeypatch):
    """Tests must never reach the network or depend on a developer's credentials."""
    import socket

    real_connect, real_create = socket.socket.connect, socket.create_connection

    def is_loopback(address) -> bool:
        host = address[0] if isinstance(address, tuple) else address
        return host in ("127.0.0.1", "::1", "localhost")

    def guarded_connect(self, address, *args, **kwargs):
        if is_loopback(address):   # in-process test servers only; everything else stays blocked
            return real_connect(self, address, *args, **kwargs)
        raise RuntimeError("network access is disabled in the test suite")

    def guarded_create(address, *args, **kwargs):
        if is_loopback(address):
            return real_create(address, *args, **kwargs)
        raise RuntimeError("network access is disabled in the test suite")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", guarded_create)
    for name in ("FINGRID_API_KEY", "CDSAPI_URL", "CDSAPI_KEY", "CDSAPI_RC"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def era5_ds() -> xr.Dataset:
    """48 h on the small grid, CDS conventions (descending latitude)."""
    return F.synthetic_era5(C.SMOKE_GRID, START, HOURS, seed=7)


@pytest.fixture(scope="session")
def fleet_fx() -> F.FleetFixture:
    return F.synthetic_fleet(C.SMOKE_GRID, seed=11, n_farms=8)


@pytest.fixture(scope="session")
def fleet_cache(tmp_path_factory, fleet_fx):
    d = tmp_path_factory.mktemp("fleet_cache")
    F.write_fleet_cache(d, fleet_fx)
    return d


@pytest.fixture(scope="session")
def smoke_axes() -> tuple[np.ndarray, np.ndarray]:
    return np.array(C.SMOKE_GRID.lats()), np.array(C.SMOKE_GRID.lons())


@pytest.fixture(scope="session")
def fleet_result(fleet_cache, smoke_axes):
    from windpower import fleet

    return fleet.build_fleet_from_cache(fleet_cache, *smoke_axes)


class Pipeline:
    """70 days of SYNTHETIC ERA5 -> simulation -> Fingrid-like series -> joined table (built once)."""


@pytest.fixture(scope="session")
def pipeline(fleet_result):
    from windpower import era5, simulate

    hours = 24 * 70
    ds = era5.engineer_features(era5.canonicalise(F.synthetic_era5(C.SMOKE_GRID, START, hours, seed=5)))
    sim = simulate.simulate_archetypes(ds, fleet_result.specs, engine="direct")
    obs = F.synthetic_observed(sim.theoretical_mw, sim.installed_mw, seed=3)
    first = sim.hourly.index[0]
    missing = [first + pd.Timedelta(hours=h) for h in (10, 11, 500)]
    p = Pipeline()
    p.ds, p.sim, p.obs, p.missing = ds, sim, obs, missing
    p.fgx = F.synthetic_fingrid(obs, missing)
    return p


@pytest.fixture(scope="session")
def joined(pipeline):
    """Joined hourly table built from the exact hourly truth with the 3 missing wind hours blanked."""
    from windpower import fingrid

    fg = pipeline.obs.copy()
    fg.loc[pipeline.missing, "wind_power_mw"] = np.nan
    fg["net_load_mw"] = fg["total_consumption_mw"] - fg["wind_power_mw"] - fg["solar_power_mw"]
    df, counts = fingrid.join_hourly(pipeline.sim.theoretical_mw, fg, pipeline.sim.installed_mw)
    return df


@pytest.fixture(scope="session")
def smoke_run(tmp_path_factory):
    """One real `smoke` run (sockets blocked) shared by manifest / export / CLI tests: (exit code, stdout, data_dir)."""
    import contextlib
    import io
    import socket

    from windpower import cli

    data = tmp_path_factory.mktemp("smoke_data")
    mp = pytest.MonkeyPatch()

    def blocked(*a, **k):
        raise RuntimeError("network access is disabled in the test suite")

    mp.setattr(socket.socket, "connect", blocked)
    mp.setattr(socket, "create_connection", blocked)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            code = cli.main(["smoke", "--data-dir", str(data)])
    finally:
        mp.undo()
    return code, buf.getvalue(), data


SOLAR_START, SOLAR_END = "2025-07-30T00:00:00", "2025-07-31T23:00:00"   # 48 h; the last hour needs the Aug 1 00:00 SARAH step


@pytest.fixture(scope="session")
def sarah_spec() -> F.SarahFixtureSpec:
    return F.SarahFixtureSpec(void_steps=("2025-07-31T12:00",), bad_steps=("2025-07-30T10:30",))


@pytest.fixture(scope="session")
def sarah_smoke(tmp_path_factory, sarah_spec):
    """SYNTHETIC CM SAF-format order (3 days over a month boundary) -> hourly means -> regridded onto the solar smoke grid."""
    from types import SimpleNamespace

    from windpower import sarah, sarah_regrid

    base = tmp_path_factory.mktemp("sarah_smoke")
    sis_dir, sid_dir = F.write_sarah_cache(base, "2025-07-30", 3, sarah_spec)
    hours = pd.date_range(SOLAR_START, SOLAR_END, freq="h")
    last = hours[-1] + pd.Timedelta(hours=1)
    sis, sid = sarah.load_series(sis_dir, "SIS", hours[0], last), sarah.load_series(sid_dir, "SID", hours[0], last)
    hourly = sarah.hourly_means(sis, sid, hours)
    return SimpleNamespace(sis_dir=sis_dir, sid_dir=sid_dir, hours=hours, sis=sis, sid=sid, hourly=hourly,
                           ongrid=sarah_regrid.regrid_hourly(hourly, C.SOLAR_SMOKE_GRID))


@pytest.fixture(scope="session")
def era5_solar_fx() -> F.Era5SolarFixture:
    return F.synthetic_era5_solar(C.SOLAR_SMOKE_GRID, SOLAR_START, 48, seed=5)


@pytest.fixture(scope="session")
def era5_solar_result(era5_solar_fx):
    from windpower import era5_solar

    return era5_solar.load_solar_era5(era5_solar_fx.raw, SOLAR_START, SOLAR_END)


@pytest.fixture(scope="session")
def public_fx() -> "FP.PublicFixture":
    return FP.synthetic_public(C.SOLAR_SMOKE_GRID, seed=3)


@pytest.fixture(scope="session")
def public_cache(tmp_path_factory, public_fx):
    """SYNTHETIC public inputs fetched through the fake session into a cache folder (the real fetch code path)."""
    from windpower import solar_public

    d = tmp_path_factory.mktemp("public_cache")
    solar_public.fetch_public(d, FP.FakePublicSession(public_fx), sleep=lambda s: None)
    return d


@pytest.fixture(scope="session")
def solar_smoke_run(tmp_path_factory):
    """One real `solar-smoke` CLI run (sockets blocked): (exit code, stdout, data_dir)."""
    import contextlib
    import io

    from windpower import cli

    data = tmp_path_factory.mktemp("solar_smoke_data")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(["solar-smoke", "--data-dir", str(data)])
    return code, buf.getvalue(), data


def tiny_ds(**overrides: float) -> xr.Dataset:
    """1 x 1 x 1 float64 dataset with plain physical values; overrides replace single variables."""
    vals = dict(u10=1.5, v10=2.0, u100=3.0, v100=4.0, t2m=273.15, d2m=272.15, sp=101325.0, fsr=0.05, lcc=0.7, tp=0.0)
    vals.update(overrides)
    return xr.Dataset(
        {k: (("valid_time", "latitude", "longitude"), np.array([[[v]]], dtype=np.float64)) for k, v in vals.items()},
        coords={"valid_time": pd.date_range(START, periods=1, freq="h"), "latitude": [60.0], "longitude": [21.0]},
    )
