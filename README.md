# windpower

Reproducible baseline for Finland's wind power: ERA5 weather + an OSM/ArcGIS turbine inventory → nine-archetype
[atlite](https://github.com/PyPSA/atlite) simulation → comparison with Fingrid's reported wind output → curtailment
reconstruction and de-biasing → a small versioned artifact for a website.

It reconstructs two notebooks (`finland_wind_data_pipeline`, `model_learning`) and a report whose originals are
**local-only** (`references/`, git-ignored, never committed). Every number in those documents is a *reconciliation
target*: it is compared with what this code produces and reported, never forced.

## Status (read this first)

| Area | State |
|---|---|
| Code for all stages, offline test suite, `smoke` and `demo` end-to-end runs | **Verified** on SYNTHETIC fixtures (see "Tests") |
| Real ERA5 (needs Copernicus CDS credentials) | **Blocked** - not available; never run |
| Real Fingrid series 181/248/124/319 (needs `FINGRID_API_KEY`) | **Blocked** - not available; the response schema of `/api/datasets/{id}` is a guess until a first authenticated run |
| Real fleet snapshot (OSM Overpass + ArcGIS FeatureServer) | **Not fetched** - code and caching are tested against fakes only; public, needs network only (`ingest-fleet --refresh`) |
| Reconciliation with the documented values (2,208 h, 2,228 turbines, 9,790.18 MW, Pearson r 0.9296, test RMSE 323.2 MW ...) | **Not run** - status `not_reproducible_no_real_data` in synthetic runs; see `docs/reconciliation.md` |
| `scripts/run_full.ps1` (Conda) | Parse-checked and its credential/check path (`-CheckOnly`) tested; **not executed end to end** - conda is not installed on the development machine, and `environment.yml` is untested |
| Local website (`serve`) | API, CSV, static hygiene **verified** by pytest; JavaScript only checked by hand in headless Edge (no Node); only ever shown with SYNTHETIC data - real-data integration untested (`docs/site.md`) |
| Licences | ERA5 / OSM / Fingrid statements are **assumed**, to be verified (`docs/data-sources.md`); atlite MIT is read from package metadata |

Synthetic outputs say nothing about Finland: they exist to exercise code paths and are labelled `SYNTHETIC`
(and `DEMO` for the demo mode) in every artifact.

## Two different models - do not mix them up

| | Retrospective benchmark | Website-facing candidate |
|---|---|---|
| Target | `reconstructed_clean_actual` = Fingrid actual + estimated curtailment | `observed_actual` (what Fingrid reported) |
| Inputs | theoretical wind **plus** imbalance price and consumption (for the curtailment gate) | theoretical wind MW + hour-of-day only |
| Models | frontier → linear / quasi-linear spline (4 knots, Ridge 10) | spline + hour harmonics (`inference_only`) |
| Meaning | explains structural bias; the documented 323.2 MW is measured against the *reconstructed* target, not a forecast error | out-of-sample error on held-out 72 h blocks; uses no price/consumption/Fingrid values at inference |

Metric records always carry `{metric, value, unit, target, split, n, model}`. `pearson_r2` is the squared correlation
(what the report calls "R²"); `regression_r2 = 1 − SSE/SST` is reported separately. The site simulation applies **no**
national calibration.

## Quick start (Windows PowerShell)

```powershell
# 1. environment: venv ...
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
# ... or Conda
conda env create -f environment.yml; conda activate windpower

# 2. offline end-to-end smoke run on a synthetic fixture (about 20 s; exit code 0 = all hard checks passed)
powershell -File scripts\run_smoke.ps1          # or: python -m windpower smoke

# 3. synthetic DEMO on the full 45 x 52 Finland grid + website artifact
python -m windpower demo                        # -> data\outputs\demo\site\v1\

# 4. tests (offline)
python -m pytest -q
```

`pip install -e .` builds in an isolated environment and therefore needs `setuptools` from the package index
(add `--no-build-isolation` if it is already installed). Full real-data run: see `docs/reproduce.md`.

## Local website (Milestone 2)

```powershell
python -m windpower demo                            # SYNTHETIC demo artifact -> data\outputs\demo\site\v1
python -m windpower serve --mode demo --port 8765   # prints http://127.0.0.1:8765/ ; Ctrl+C to stop
powershell -File scripts\run_site.ps1 -Mode demo -Port 8765
```

Site simulation (location on a map or lat/lon, date range, archetype, capacity → hourly MW, capacity factor, MWh, CSV)
and Finland overview (national theoretical vs observed series, coverage). Stdlib server + vanilla JS, no network at
runtime; demo/synthetic data is flagged by a persistent banner, in every API response and in every CSV header. No
national calibration is applied to site simulations. If the artifact is missing, `serve` prints the command that
creates it. Details, API, accessibility notes and gaps: `docs/site.md`.

**Solar in the website.** The same server can serve a Wind | Solar switch (hash routes `#simulate`, `#overview`,
`#solar-simulate`, `#solar-overview`) from two artifacts:

```powershell
python -m windpower demo                                  # wind demo artifact (SYNTHETIC)
python -m windpower solar-demo                            # solar demo artifact (SYNTHETIC) -> data\outputs\solar-demo\site\solar\v1
python -m windpower serve --mode demo --solar-mode solar-demo --port 8765   # wind AND solar
python -m windpower serve --mode solar-demo --port 8765   # solar (and whatever wind artifact exists)
python -m windpower solar-all                             # REAL SARAH-3 + public data; ERA5/Fingrid parts stay "unavailable" without their inputs
python -m windpower serve                                 # no flags: auto-discovers real (full, solar) before demo before smoke
```

The solar page offers a 0.25 degree cell simulation (lat/lon, dates inside the artifact window, capacity in MWp,
ERA5-only or hybrid irradiance, CSV) and a national overview (modelled MW vs Fingrid 248 forecast, 362 AV06
small-scale surplus and 267 capacity). Modes whose inputs are missing are shown as unavailable with the exact reason.

## Stages

`python -m windpower <stage> [--mode smoke|full|demo] [--data-dir DIR] [--offline] [--refresh] [--start ..] [--end ..]`

| Stage | What it does | Main outputs (`data/outputs/<mode>/`) |
|---|---|---|
| `ingest-era5` | monthly ERA5 files (CDS download in full mode) → validate window/grid/units | `data/cache/.../era5/` |
| `ingest-fleet` | OSM + ArcGIS raw responses (`--refresh` = network) | `data/cache/.../fleet/` |
| `ingest-fingrid` | 4 Fingrid series, paged/rate-limited, cached; dataset metadata verification | `data/cache/.../fingrid/` |
| `fleet` | active filter, 150 m merge, imputation, grid join, capacity matrix | `fleet_turbines.csv`, `fleet_pixel_aggregation.csv`, `grid.csv/.parquet`, `capacity_matrix.csv`, `fleet_specs.csv` |
| `simulate` | nine archetypes with atlite → hourly national MW | `archetypes.csv`, `theoretical_hourly.csv`, `power_curves.json` |
| `calibrate` | join with Fingrid, 72 h block split, frontier, curtailment, de-biasing, `inference_only`, forward check | `hourly_joined.csv`, `daily.csv`, `calibration_hourly.csv`, `metrics.csv/json`, `split_blocks.csv` |
| `validate` | hard checks + reconciliation + manifest | `validation.csv/json`, `reconciliation.csv/json`, `manifest.json` |
| `export-site` | compact website artifact | `site/v1/` |
| `smoke` | all of the above offline on the small synthetic fixture | |
| `demo` | same on the synthetic Finland grid, labelled DEMO | |
| `all` | all of the above for `--mode` (full needs credentials) | |
| `serve` | local website (map, hourly simulation, national overview; wind and/or solar) from `site/v1` and `site/solar/v1`; never generates data; `--solar-mode`, `--solar-site-dir` | see `docs/site.md` |
| `ingest-sarah` | **solar (S1)**: CM SAF SARAH-3 SIS/SID daily files → hourly trapezoid means, conservative regrid onto the 44 x 53 ERA5 solar grid, coverage map; `--mode solar` (real, default), `solar-smoke`, `solar-demo` (SYNTHETIC). Inputs: `--sarah-sis-dir/--sarah-sid-dir` (env `WINDPOWER_SARAH_SIS_DIR/_SID_DIR`; default `data/interim/smoke/ORD73128` / `ORD73129` if present), `--window-start/--window-end` (default 2025-07-01..2025-07-31) | `sarah_hourly_era5grid.nc`, `sarah_coverage_cells.csv`, `sarah_coverage_pixels.nc`, `sarah_inventory.json`; run info in `data/interim/<mode>/run_info.json`. Method and assumptions: `docs/assumptions.md` S1-S17 |
| `ingest-era5-solar`, `ingest-solar-public`, `ingest-fingrid-solar` | **solar (S2)**: ERA5 radiation (CDS or `--era5-solar-path`), public plant list + Statistics Finland WFS (explicit network), Fingrid 267/248/362 (`FINGRID_API_KEY`) | caches under `data/cache/solar/` |
| `solar-hourly`, `solar-pv`, `solar-national`, `solar-validate`, `export-site-solar` | ERA5 hourly + irradiance comparison; atlite CF cubes; allocation + national MW/GWh; hard checks + reconciliation + manifest; `site/solar/v1` | see `docs/solar.md` |
| `solar-smoke`, `solar-demo`, `solar-all` | the whole solar chain: offline SYNTHETIC (7 x 9 / 44 x 53 cells) or real with partial-availability handling | `data/outputs/<solar mode>/` |

Stages rebuild what they need from the raw caches, so they can be run separately; in synthetic modes missing
fixtures are generated, in `full` mode missing inputs are an error - **never a silent fallback to synthetic data**.

**Exit codes:** `0` ok · `1` a hard validation check failed · `2` required inputs unavailable (credentials, caches,
`--offline`) · `3` unexpected error · `64` usage error. Unavailable inputs are also written to `manifest.json`.

## Layout

```
src/windpower/   config io_utils fixtures era5 fleet fingrid simulate calibrate validate manifest export_site pipeline cli web/ (server, data, static/)
                 solar: sarah sarah_regrid solar_geometry era5_solar solar_pv solar_pipeline solar_public fingrid_solar solar_national solar_validate export_site_solar fixtures_solar
tests/           offline pytest suite (sockets blocked, credentials stripped)
scripts/         run_smoke.ps1  run_full.ps1  run_site.ps1
docs/            assumptions  data-sources  licenses  reconciliation  reproduce  site
data/            (git-ignored) cache/ interim/ outputs/<mode>/
references/      (git-ignored, local-only originals and text exports - never commit)
```

## Hard checks (`validate`)

Unique strictly hourly UTC index and expected hour count · join coverage (comparable / ERA5 hours) · grid orientation
(`capacity_matrix[i, j]` ↔ `lat[i], lon[j]`, ascending axes) · capacity totals (turbine table = pixel aggregation +
unassigned = matrix + unassigned = spec table = archetype table) · split isolation (disjoint, whole contiguous blocks,
anchors from train only, fits unchanged when test labels are perturbed) · units (MW/MWh/CF, no negative power, daily GWh)
· frontier bounds · metric labels. The forward-in-time check is *skipped with a reason* below 60 days of coverage.

## Tests

`python -m pytest -q` runs fully offline (sockets blocked, credential variables removed, fake transports for
Fingrid/ArcGIS/Overpass/CDS). It includes a real `smoke` run, the CLI exit codes, and the PowerShell scripts.

## Provenance and licences

`manifest.json` records versions, git commit, source dates, snapshot dates, file hashes, schemas with units, licences
and an explicit list of what was unavailable. Attribution: *Contains modified Copernicus Climate Change Service
information*; © OpenStreetMap contributors (ODbL); Fingrid open data (CC BY 4.0, to be verified); atlite (MIT). The
fleet inventory is a *current* snapshot and may differ from the 2025 fleet.
