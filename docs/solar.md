# Solar PV (Finland, July 2025): method, inputs, commands, reconciliation

This is the solar part of the project: hourly PV capacity factors from SARAH-3 (CM SAF satellite) and ERA5, with ERA5 used
where the satellite does not cover a cell, scaled to a national series with Fingrid's capacity (dataset 267), a list of
the 30 largest plants and a 1 km population grid. Method source: the local solar report (`references/`, git-ignored). Every
assumption and every thing that could not be verified is in `docs/assumptions.md` (S1-S33). Wind code and artifacts are
unchanged (`site/v1`, `/api/v1/*`).

## Method in one page

1. **SARAH-3 -> hourly.** 30-min instantaneous SIS (GHI) and SID (direct) per 0.05 deg cell. Hour h (start-labelled, UTC) =
   0.25 G(h:00) + 0.50 G(h:30) + 0.25 G(h+1:00), the last sample from the next day / month file. A sample is valid if
   `record_status` = 0, not fill (-999), 0 <= value <= 1500 W m-2. Any invalid sample -> NaN (never 0). SID needs a valid SIS
   in the same sample; diffuse = SIS - SID (below -1 W m-2 invalid, [-1, 0) clipped to 0).
2. **Conservative regrid** onto the 44 x 53 ERA5 solar grid (`FINLAND_SOLAR`; the wind grid is 45 x 52 and untouched) with
   exact spherical-area weights. `valid_area_fraction` = valid source area / full ERA5 cell area. A cell-hour uses the
   satellite only if >= 99.9 %; every cell-hour carries a source / fallback reason code. "Fully covered" cells (report: 777 of
   2,332) = static map: every source pixel with >= 1 fully valid daytime hour, area share >= 99.9 %.
3. **ERA5.** `ssrd ssr fdir tisr` (J m-2, accumulated over the hour ending at the stamp) -> W m-2; stamp h+1 -> hour h;
   temperature = mean of stamps h and h+1; solar position at h+30 min; albedo = (ssrd - ssr) / ssrd; diffuse = ssrd - fdir.
   The time convention is **detected** from the data (TOA vs NOAA solar geometry), recorded with its evidence, and applied once;
   an ambiguous result is an error unless `--era5-convention end|start` is given.
4. **PV.** atlite `cutout.pv` (CSi panel, inverter efficiency 0.90 verified inside atlite, slope 40, azimuth 180) on an
   in-memory cutout. ERA5-only and hybrid (satellite direct + diffuse where covered, ERA5 elsewhere; TOA, temperature, albedo
   always ERA5). Night CF is exactly 0.
5. **National.** P(t) = sum_c CF(t,c) [ (C(t) - L) w_c + L_c ], C(t) = Fingrid 267, L = 250.66 MW of large plants placed in the
   cell of their municipality centroid, w_c = population share. Sum over cells == C(t) is a hard check. A population-only
   variant is a sensitivity case.
6. **Comparison.** Energy, ratios, Pearson r (labelled as Pearson, not regression R2), daily-cycle centre, against Fingrid 248
   (a FORECAST) and 362 AV06 (small-scale SURPLUS only): neither is measured total generation.

## Inputs inventory

| Input | Where | Status on the development machine |
|---|---|---|
| SARAH-3 SIS / SID, 94 daily files each, 2025-06-30 .. 2025-10-01 | `data\interim\smoke\ORD73128` (SIS), `ORD73129` (SID). **The folder name `smoke` is misleading: these are the REAL CM SAF files** (title "CM SAF ... Heliosat Edition 3 (SARAH-3)", DOI 10.5676/EUM_SAF_CM/SARAH/V003). The `.tar` archives next to them are untouched; nothing is ever copied or modified. | present, processed |
| Plant list, population grid, municipalities | public; `ingest-solar-public` -> `data\cache\solar\public\` | fetched 2026-11-27 |
| ERA5 radiation + temperature | CDS (`~/.cdsapirc`) or `--era5-solar-path` / `WINDPOWER_ERA5_SOLAR` | **missing** (no credentials) |
| Fingrid 267 / 248 / 362 | `FINGRID_API_KEY` (environment or untracked `.env`) | **missing** (no key) |

## Commands (Windows PowerShell, repository root)

```powershell
.\.venv\Scripts\python.exe -m pip install -e . -q

# offline, SYNTHETIC end to end (every stage, all hard checks)
.\.venv\Scripts\python.exe -m windpower solar-smoke          # 7 x 9 cells, 48 h, about 20 s
.\.venv\Scripts\python.exe -m windpower solar-demo           # full 44 x 53 grid, 96 h, about 35 s

# real inputs, stage by stage
.\.venv\Scripts\python.exe -m windpower ingest-sarah --window-start 2025-07-01 --window-end 2025-07-31
.\.venv\Scripts\python.exe -m windpower ingest-solar-public            # network, no credentials; --refresh re-downloads
.\.venv\Scripts\python.exe -m windpower ingest-era5-solar              # needs ~/.cdsapirc  (or: --era5-solar-path <file|dir|zip>)
.\.venv\Scripts\python.exe -m windpower ingest-fingrid-solar           # needs $env:FINGRID_API_KEY (never printed or cached)
.\.venv\Scripts\python.exe -m windpower solar-hourly
.\.venv\Scripts\python.exe -m windpower solar-pv
.\.venv\Scripts\python.exe -m windpower solar-national
.\.venv\Scripts\python.exe -m windpower solar-validate
.\.venv\Scripts\python.exe -m windpower export-site-solar

# everything the available inputs allow (caches only; add --refresh to use the network)
.\.venv\Scripts\python.exe -m windpower solar-all
```

Other input locations: `--sarah-sis-dir`, `--sarah-sid-dir` (env `WINDPOWER_SARAH_SIS_DIR`, `WINDPOWER_SARAH_SID_DIR`),
`--era5-solar-path` (env `WINDPOWER_ERA5_SOLAR`), `--era5-convention end|start`, `--window-start/--window-end` (default
2025-07-01 .. 2025-07-31, an end date is inclusive), `--data-dir`, `--offline`, `--refresh`.

**Exit codes:** 0 ok (including a partial real run, see the manifest) | 1 hard check / input-contract failure |
2 a stage's own required inputs are unavailable | 3 unexpected / data-access error | 64 usage error.
**Partial runs:** `solar-all` records every gap in `manifest.json` (`status: partial_inputs_unavailable`, `unavailable`)
and still writes `site/solar/v1` with the unavailable modes and series marked `available: false` and a reason; nothing is
zero-filled and real mode never uses synthetic data.

## Outputs (`data\outputs\<mode>\`, git-ignored)

`sarah_hourly_era5grid.nc`, `sarah_coverage_cells.csv`, `sarah_coverage_pixels.nc`, `sarah_inventory.json` (ingest-sarah) -
`era5_solar_hourly.nc`, `irradiance_comparison.json` (solar-hourly) - `cf_cubes.nc`, `cf_cell_summary.csv` (solar-pv) -
`allocation_cells.csv`, `allocation_plants.csv`, `allocation_report.json`, `national_hourly.csv`, `daily.csv`,
`metrics.json` (solar-national) - `validation.csv/json`, `reconciliation.csv/json`, `manifest.json` (solar-validate) -
`site/solar/v1/` (export-site-solar). Run provenance: `data\interim\<mode>\run_info.json`. Fixtures and raw caches:
`data\cache\solar\` (real) or `data\cache\synthetic_<mode>\`.

## Reconciliation, real run of 2026-11-27 (July 2025, SARAH-3 + public data only)

`solar-all` with default paths: status `partial_inputs_unavailable`, 5 checks pass, 5 skipped (each names its missing input),
0 fail. 33 reconciliation rows: 10 `match`, 23 `unavailable`. Differences are reported, nothing is tuned.

| Metric | Report | Observed | Status |
|---|---|---|---|
| hourly values | 744 | 744 | match |
| fully covered ERA5 cells | 777 | 777 | match |
| ERA5 grid cells | 2,332 | 2,332 | match |
| source pixels that never had daytime data | 25.4 % | 25.39 % | match (abs 0.05) |
| large plants (operational 30.6.2025) | 30 | 30 | match |
| large-plant capacity | 250.66 MW | 250.66 MW | match |
| largest plant (Isosuo, Utajarvi) | 103.2 MW (41 %) | 103.2 MW (41.17 %) | match |
| inhabited 1 km cells | 96,904 | 96,904 | match |
| inhabitants | 5,573,552 | 5,573,552 | match |
| ERA5-only / hybrid energy, 248 energy, ratios, Pearson r (248 and 362), cycle centres, mean CF, GHI table 2, population-only energy | 214.0 / 216.2 / 248.2 GWh, 0.862 / 0.871, 0.981 / 0.983 / 0.978 / 0.985, 10:21 / 10:17 / 10:24 / 10:32, 0.195 ..., 242.5 / 241.9 W m-2 | - | **unavailable**: ERA5 solar data (no CDS credentials) and/or Fingrid 267 / 248 / 362 (no `FINGRID_API_KEY`) |

What is blocked and what unblocks it: put CDS credentials in `~/.cdsapirc` (or pass `--era5-solar-path`) and set
`FINGRID_API_KEY`, then run `ingest-era5-solar`, `ingest-fingrid-solar`, `solar-all`. The hybrid-vs-ERA5 and GHI rows need ERA5;
the energy / correlation / cycle rows need ERA5 and Fingrid. Synthetic runs show their numbers but every row is
`not_reproducible_synthetic`.

## Labels and caveats to keep when showing results

* 248 = "forecast, not measured generation"; 362 AV06 = "small-scale surplus only (after own consumption), not total generation";
  267 = "capacity used in the forecast, not official". The national model has no calibration.
* SYNTHETIC / DEMO banners are in `manifest.json` and the artifact (`label`, `synthetic`, `demo`).
* Licences: CM SAF / EUMETSAT (file header text), Copernicus (ERA5), Fingrid CC BY 4.0, Statistics Finland (CC BY 4.0 assumed,
  not stated by the service), Suomen Uusiutuvat (none stated) - see `docs/licenses.md`.
