"""Shared offline fixtures. Everything is SYNTHETIC and lives in pytest tmp dirs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from windpower import config as C
from windpower import fixtures as F

START = "2025-07-01T00:00:00"
HOURS = 48
END = "2025-07-02T23:00:00"


@pytest.fixture(autouse=True)
def _offline_and_credential_free(monkeypatch):
    """Tests must never reach the network or depend on a developer's credentials."""
    import socket

    def blocked(*args, **kwargs):
        raise RuntimeError("network access is disabled in the test suite")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
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


def tiny_ds(**overrides: float) -> xr.Dataset:
    """1 x 1 x 1 float64 dataset with plain physical values; overrides replace single variables."""
    vals = dict(u10=1.5, v10=2.0, u100=3.0, v100=4.0, t2m=273.15, d2m=272.15, sp=101325.0, fsr=0.05, lcc=0.7, tp=0.0)
    vals.update(overrides)
    return xr.Dataset(
        {k: (("valid_time", "latitude", "longitude"), np.array([[[v]]], dtype=np.float64)) for k, v in vals.items()},
        coords={"valid_time": pd.date_range(START, periods=1, freq="h"), "latitude": [60.0], "longitude": [21.0]},
    )
