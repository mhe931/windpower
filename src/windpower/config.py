"""Constants shared by all stages (values taken from the reference notebooks)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- run modes and paths ---------------------------------------------------
RUN_MODES = ("smoke", "full", "demo")
SYNTHETIC_MODES = ("smoke", "demo")   # modes that never touch real data or the network
# Solar pipeline modes (separate tuples: the wind constants above are part of the wind contract).
SOLAR_RUN_MODES = ("solar-smoke", "solar-demo", "solar")
SOLAR_SYNTHETIC_MODES = ("solar-smoke", "solar-demo")
ALL_RUN_MODES = (*RUN_MODES, *SOLAR_RUN_MODES)
REPO_ROOT = Path(__file__).resolve().parents[2]


def is_synthetic_mode(run_mode: str) -> bool:
    return run_mode in SYNTHETIC_MODES or run_mode in SOLAR_SYNTHETIC_MODES


def data_dir(base: "str | Path | None" = None) -> Path:
    """Root of git-ignored generated data: explicit `base`, else WINDPOWER_DATA_DIR, else <repo>/data."""
    if base:
        return Path(base)
    override = os.environ.get("WINDPOWER_DATA_DIR")
    return Path(override) if override else REPO_ROOT / "data"


def cache_dir(run_mode: str = "full", base: "str | Path | None" = None) -> Path:
    """Raw downloads. Synthetic modes get their own cache so fixture data never mixes with real data."""
    _check_mode(run_mode)
    sub = f"synthetic_{run_mode}" if is_synthetic_mode(run_mode) else ("solar" if run_mode == "solar" else "real")
    return data_dir(base) / "cache" / sub


def interim_dir(run_mode: str = "full", base: "str | Path | None" = None) -> Path:
    _check_mode(run_mode)
    return data_dir(base) / "interim" / run_mode


def output_dir(run_mode: str = "full", base: "str | Path | None" = None) -> Path:
    _check_mode(run_mode)
    return data_dir(base) / "outputs" / run_mode


def _check_mode(run_mode: str) -> None:
    if run_mode not in ALL_RUN_MODES:
        raise ValueError(f"run_mode must be one of {ALL_RUN_MODES}, got {run_mode!r}")


# --- study window ----------------------------------------------------------
WINDOW_START = "2025-07-01T00:00:00"  # UTC, first ERA5 hour
WINDOW_END = "2025-09-30T23:00:00"  # UTC, last ERA5 hour (2,208 hours)
EXPECTED_WINDOW_HOURS = 2208

# --- ERA5 grid -------------------------------------------------------------
GRID_RES_DEG = 0.25
GRID_TOL = 1e-6


@dataclass(frozen=True)
class GridSpec:
    """Regular lat/lon grid of cell centres at GRID_RES_DEG spacing."""

    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    res: float = GRID_RES_DEG

    @property
    def n_lat(self) -> int:
        return int(round((self.lat_max - self.lat_min) / self.res)) + 1

    @property
    def n_lon(self) -> int:
        return int(round((self.lon_max - self.lon_min) / self.res)) + 1

    def lats(self) -> "list[float]":
        return [self.lat_min + i * self.res for i in range(self.n_lat)]

    def lons(self) -> "list[float]":
        return [self.lon_min + j * self.res for j in range(self.n_lon)]

    @property
    def cds_area(self) -> "list[float]":
        """CDS `area` order: north, west, south, east."""
        return [self.lat_max, self.lon_min, self.lat_min, self.lon_max]


# 45 x 52 grid inferred from the report figures (verify against real files).
FINLAND_GRID = GridSpec("finland", 59.5, 70.5, 19.0, 31.75)
SMOKE_GRID = GridSpec("smoke", 60.0, 63.0, 21.0, 25.0)
GRIDS = {g.name: g for g in (FINLAND_GRID, SMOKE_GRID)}

# --- solar grids -------------------------------------------------------------
# The solar report's ERA5 cutout (atlite, area 59.5-70.25 N, 19.0-32.0 E) is 44 x 53 = 2,332 cells. It is NOT the
# wind grid above (45 x 52, 59.5-70.5 N, 19.0-31.75 E): one row more to the north, one column less to the east in
# the wind grid. Both grids stay as they are; solar code must only use FINLAND_SOLAR.
FINLAND_SOLAR = GridSpec("finland_solar", 59.5, 70.25, 19.0, 32.0)
SOLAR_SMOKE_GRID = GridSpec("solar_smoke", 60.0, 61.5, 21.0, 23.0)   # SYNTHETIC fixtures only
SOLAR_GRIDS = {g.name: g for g in (FINLAND_SOLAR, SOLAR_SMOKE_GRID)}

# --- solar (CM SAF SARAH-3 + ERA5) ---------------------------------------------
SARAH_RES_DEG = 0.05
SARAH_FILL = -999
SARAH_MAX_WM2 = 1500.0               # upper plausibility bound of a valid irradiance sample
SARAH_DIFFUSE_TOL_WM2 = 1.0          # SIS - SID in [-1, 0) is clipped to 0, below -1 the sample is invalid
SATELLITE_MIN_AREA_FRACTION = 0.999  # a cell-hour uses SARAH only if >= 99.9 % of the ERA5 cell area is valid
SARAH_DAYTIME_MIN_ELEVATION_DEG = 5.0
SOLAR_WINDOW_START = "2025-07-01"    # inclusive dates (UTC); the default window is July 2025 (744 hours)
SOLAR_WINDOW_END = "2025-07-31"
SOLAR_REPORT_TARGETS = {"covered_cells": 777, "grid_cells": 2332, "hours": 744}   # compared, never forced
SARAH_SIS_ENV, SARAH_SID_ENV, ERA5_SOLAR_ENV = "WINDPOWER_SARAH_SIS_DIR", "WINDPOWER_SARAH_SID_DIR", "WINDPOWER_ERA5_SOLAR"
SARAH_DEFAULT_DIRS = {"SIS": "ORD73128", "SID": "ORD73129"}   # below <data>/interim/smoke/ (a misleading name; real SARAH-3 files)

# PV baseline of the solar report
PV_PANEL = "CSi"
PV_SLOPE_DEG = 40.0
PV_AZIMUTH_DEG = 180.0
PV_INVERTER_EFFICIENCY = 0.90

# --- ERA5 variables --------------------------------------------------------
CDS_DATASET = "reanalysis-era5-single-levels"
CDS_VARIABLES = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "forecast_surface_roughness",
    "low_cloud_cover",
    "total_precipitation",
)
REQUIRED_VARS = ("u10", "v10", "u100", "v100", "t2m", "d2m", "sp", "fsr", "lcc", "tp")

# Plausibility ranges (inclusive lower, exclusive upper unless noted in era5.py).
RANGE_T_K = (200.0, 330.0)
RANGE_SP_PA = (5.0e4, 1.1e5)
RANGE_LCC = (0.0, 1.0)
TP_MIN_M = -1e-6
TP_MAX_M = 0.5
WIND_ABS_MAX = 80.0

# --- physics constants and thresholds (notebook 1) --------------------------
R_SPECIFIC_AIR = 287.05  # notebook 2 fallback used 287.058; see docs/assumptions.md
RHO_0 = 1.225
ICING_T_MIN_C = -15.0
ICING_T_MAX_C = 1.0
ICING_SPREAD_MAX_C = 1.5
ICING_LCC_MIN = 0.6
ICING_TP_MIN_M = 0.0001
SHEAR_ALPHA_MAX = 0.6

# --- fleet -----------------------------------------------------------------
CRS_GEO = "EPSG:4326"
CRS_METRIC = "EPSG:3067"
MIN_TURBINE_MW = 0.5
MAX_TURBINE_MW = 15.0
MATCH_RADIUS_M = 150.0
CAP_IMPUTE_RADIUS_M = 5000.0
HUB_IMPUTE_RADIUS_M = 3000.0
ROTOR_IMPUTE_RADIUS_M = 3000.0
YEAR_IMPUTE_RADIUS_M = 2000.0
CURRENT_MAX_YEAR = 2026
YEAR_CLIP = (1990, 2026)
TIP_HEIGHT_THRESHOLD_M = 180.0
SP_CLIP = (170.0, 450.0)
SP_BIN_EDGES = (0.0, 265.0, 340.0, float("inf"))
SP_BIN_LABELS = ("LowSP_Modern", "MidSP", "HighSP_Legacy")
HUB_BIN_EDGES = (0.0, 100.0, 140.0, float("inf"))
HUB_BIN_LABELS = ("LowHub", "MidHub", "HighHub")
# (upper-exclusive year, value) eras; anything newer takes the last value.
ERA_YEAR_EDGES = (2010, 2016, 2021)
ERA_CAPACITY_MW = (1.8, 3.2, 4.3, 5.6)
ERA_SPECIFIC_POWER = (390.0, 320.0, 275.0, 245.0)
ERA_HUB_M = (75.0, 115.0, 138.0, 152.0)

INACTIVE_KEYWORDS = (
    "proposed", "planned", "construction", "disused", "abandoned",
    "decommissioned", "demolished", "inactive", "removed", "cancelled",
    "suunnit", "rakenteilla", "luvitus", "purettu", "poistettu",
    "hylätty", "keskeytetty", "ei käytössä",
)

OSM_MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
OSM_QUERY = """
[out:json][timeout:90];
area["ISO3166-1"="FI"][admin_level=2]->.finland;
(
  node["power"="generator"]["generator:source"="wind"](area.finland);
  way["power"="generator"]["generator:source"="wind"](area.finland);
);
out center;
"""
ARCGIS_SERVICE_URL = (
    "https://services3.arcgis.com/NqGokVvI0NQ6O9Jf/arcgis/rest/services/"
    "Tuulivoimalat_WindTurbines/FeatureServer"
)
ARCGIS_PAGE_SIZE = 500
HTTP_USER_AGENT = "FinlandWindEnergyResearch/1.0 (Academic Spatial Modeling)"
FLEET_SNAPSHOT_NOTE = "current inventory may differ from the 2025 fleet"
OSM_CACHE_FILE = "finland_turbines_osm.json"  # full Overpass response
OSM_META_FILE = "finland_turbines_osm.snapshot.json"
ARCGIS_CACHE_FILE = "finland_wind_turbines_official.geojson"
ARCGIS_META_FILE = "finland_wind_turbines_official.snapshot.json"

# --- documented reconciliation targets (compare, never force) ---------------
TARGETS = {
    "era5_hours": 2208,
    "turbines": 2228,
    "osm_kept": 1859,
    "arcgis_kept": 2080,
    "capacity_mw": 9790.18,
    "active_cells": 208,
    "mean_speed_100": 5.46,
    "mean_air_density": 1.202,
    "icing_share_pct": 0.07,
    "year_coverage_pct": 28.5,
}
