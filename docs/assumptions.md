# Assumptions and deviations

Original notebooks and data caches were not available; everything is reconstructed from the PDF text
exports (`notebook1_pipeline`, `report_fa`). Many long lines are clipped at the right edge. Where a
line was clipped the simplest reading consistent with the report was chosen and is listed here.
This file covers the ERA5, fleet, Fingrid, simulation and calibration stages (handoffs 1 and 2a); later stages append their own sections.

Cell references are to `finland_wind_data_pipeline` (notebook 1).

## ERA5 (cell "ERA5 load + features")

| # | Item | Choice | Why |
|---|------|--------|-----|
| E1 | Relative humidity | Magnus-Tetens: `es = 6.112·exp(17.67·T/(T+243.5))`, `e` likewise with the dewpoint, `RH = clip(100·e/es, 0, 100)` | Denominator clipped in the export; the standard constants match the visible numerators. |
| E2 | `icing_flag` | `(-15 <= T_c <= +1) & (T - Td <= 1.5) & ((lcc >= 0.6) \| (tp > 0.0001))`, stored as float32 0/1 | Right-hand bounds were clipped; the comment above the cell states the −15 °C, +1 °C, 1.5 °C and "low clouds or precipitation" criteria. |
| E3 | `icing_severity` | `temp_factor·humid_factor·cloud_factor`; `temp_factor = clip(1-\|T_c+4\|/11, 0, 1)`, `humid_factor = clip(1-spread/2.5, 0, 1)`, `cloud_factor = clip(max(lcc, (tp > 1e-4)), 0, 1)` | Visible in the export except the last clip bounds. |
| E4 | Air density | `sp / (287.05·t2m)` | Pipeline notebook. Notebook 2's fallback used 287.058; the difference is 2·10⁻⁵ relative and is ignored. |
| E5 | Shear exponent | `clip(log(max(s100,0.1)/max(s10,0.1))/log(10), 0, 0.6)` | As in the notebook. |
| E6 | File discovery | `era5_*.nc` larger than 1 KiB; unreadable files are reported in `LoadedEra5.skipped_files`, not silently dropped | The notebook printed a skip message; we keep the reason. |
| E7 | ZIP members | Members are read from the archive into a private temp dir (archive paths are never trusted), opened with engine fallback `h5netcdf → netcdf4 → scipy → default`, merged with `compat="override"`; members < 1 KiB are skipped | Notebook extracted next to the download; this avoids leftover `*_extracted/` directories and zip-slip. |
| E8 | Canonical form | dims `valid_time, latitude, longitude`, all ascending, duplicate times dropped (first kept), scalar `number`/`expver` coords dropped; source latitude order kept in `attrs["source_lat_order"]` | Spec. Datasets are loaded into memory so file handles are released (required on Windows). |
| E9 | `expver` as a *dimension* (mixed ERA5/ERA5T months) | Not handled; validation fails with a dims error | Not seen in the notebook; handle when real files are available. |
| E10 | Units check | The `units` attribute (when present) must be one of the CDS spellings; ranges are checked on values; NaNs in required variables fail | Spec ranges: t2m 200–330 K, sp 5·10⁴–1.1·10⁵ Pa, lcc 0–1, tp ≥ −1e-6 and < 0.5 m h⁻¹, \|u\| < 80 m s⁻¹. d2m uses the t2m range. |
| E11 | Grid extent | The 45 × 52 grid (lat 59.5–70.5, lon 19.0–31.75) is **inferred from the report figures**; `validate_era5(grid=…)` checks it only when a grid spec is passed. Verify against real files. | |
| E12 | CDS requests | One request per calendar month; only days inside the requested window; `data_format=netcdf`, `download_format=unarchived`; the response may still be a ZIP, which the reader handles | The exact payload used for the original download is unknown. |

## Fleet (cells "Source 1/2", "Cross-verify and merge", "Spatial imputation", "Map to ERA5 pixel")

| # | Item | Choice | Why |
|---|------|--------|-----|
| F1 | OSM tag fallbacks | hub: `height:hub` or `hub_height`; manufacturer: `manufacturer` or `generator:manufacturer`; model: `model` or `generator:model` | The `or tags.get(...)` tails were clipped. |
| F2 | ArcGIS capacity unit | Unit-less numbers are MW, **unless** the chosen column name contains "kw" (then kW). Explicit units in the text always win. | Small extension of the notebook (which would turn 3600 kW into 3600 MW and then drop it as out of range). |
| F3 | ArcGIS field mapping | Fuzzy `find_col` with the notebook keyword lists; `rakennusvaihe`, `kayttoonot` etc. kept as written. The mapping is returned (`FleetResult.column_mapping`) and must be logged in the manifest | Spec. |
| F4 | Coordinates of matched turbines | OSM coordinates win (`lat.combine_first(off_lat)`); ArcGIS-only turbines use the centroid in EPSG:3067 converted back to EPSG:4326 | Literal notebook behaviour. |
| F5 | Attribute preference | ArcGIS first, OSM fills gaps (`combine_first`) | Spec. `source` is `arcgis`, `osm` or `arcgis+osm`. |
| F6 | Tie handling in the 150 m join | `sjoin_nearest` may return several rows per ArcGIS point; rows are ordered by distance then OSM index and the first is kept (deterministic). The notebook kept the first row in arbitrary order. | Determinism. |
| F7 | Duplicate-location ArcGIS entries | Not de-duplicated (notebook behaviour). Two ArcGIS rows at one coordinate stay two turbines, and each OSM point is consumed at most once. The fixture contains such a pair. | Whether the real layer contains duplicates is unknown; reconcile against 2,228 turbines when the real layer is cached. |
| F8 | Imputation | Nearest known turbine within the radius (cap 5 km, hub 3 km, rotor 3 km, year 2 km); mean of equidistant ties (median for year); cap/hub/rotor then fall back to the fleet median (computed after neighbour imputation); year has no fallback. The notebook used the hub radius for rotor; the spec's 3 km coincides. | Notebook + spec. Provenance: `<field>_imputed` (bool) and `<field>_impute_method` (`nearest_known_mean_within_<r>m`, `nearest_known_median_within_2000m`, `fleet_median_fallback`). |
| F9 | Year rounding | `commissioning_year.round()` (zero decimals), left as float with NaN | The rounding argument was clipped. |
| F10 | Grid join | Point-in-polygon `within` of the 0.25° box: a point exactly on a cell edge or outside the grid is **unassigned** (NaN pixel). Unassigned turbines/capacity are counted in `capacity_totals`. | Notebook behaviour. The reconciliation target (208 active cells) is computed from this join. |
| F11 | Capacity matrix dtype | float64 (notebook: float32) so totals agree to 1e-9 | Accuracy; cast when exporting. |
| F12 | Pixel indices | `Int64` (nullable) in the turbine table | Needed for unassigned points. |
| F13 | Layout for simulation (later handoff) | Capacity per cell by nearest ERA5 centre (`argmin`) in the simulator vs `within` here; both are kept on purpose | Spec assumption 6. |

## Per-turbine specification table (`df_t`, simulation cell)

| # | Item | Choice | Why |
|---|------|--------|-----|
| S1 | Column detection | Exact lower-case match over the candidate list first, then substring match | Notebook `_find_col`; candidate lists beyond the first entries were clipped. |
| S2 | Year | `clip(1990, 2026)`, NaN → median of the clipped years; with no year information the newest-era defaults apply | Lower bound assumed; all-NaN fallback assumed. |
| S3 | Capacity | `/1000` if the column name contains "kw" or the median exceeds 100; NaN → era default 1.8/3.2/4.3/5.6 MW (year < 2010 / < 2016 / < 2021 / else) | Spec. |
| S4 | Rotor | NaN → `2·sqrt(cap·10⁶/(π·SP_era))`, SP_era = 390/320/275/245 W m⁻² | Spec. |
| S5 | Hub | Values > 180 m are tip heights: `hub = raw − 0.5·rotor`; NaN → era default 75/115/138/152 m. `hub_raw_m` keeps the uncorrected value. | Spec. This explains the report's ">200 m" (raw hub map) versus 166 m maximum (corrected `df_t`); to be confirmed on real data. |
| S6 | Specific power | `cap·10⁶/(π(rotor/2)²)`, clipped to [170, 450] W m⁻² | Upper bound assumed (HighSP archetype means reach 430.6). |
| S7 | Bins | SP edges [0, 265, 340, ∞] → `LowSP_Modern / MidSP / HighSP_Legacy`; hub edges [0, 100, 140, ∞] → `LowHub / MidHub / HighHub`; right-closed (`pd.cut` default), so a 100 m hub is `LowHub` | Spec. |
| S8 | Flags | `cap_era_default`, `rotor_inferred`, `hub_era_default`, `hub_tip_corrected` added | Provenance. |

## Fingrid (cell "Fingrid fetch", handoff 2a)

| # | Item | Choice | Why |
|---|------|--------|-----|
| G1 | Key | Only `FINGRID_API_KEY` from the environment, else an untracked `.env` in the repo root; never logged, cached, or printed (`repr` hides it). The notebook's hard-coded default is deliberately not reproduced. | Security. |
| G2 | Request shape | `GET /api/datasets/{id}/data` with `startTime`, `endTime`, `format=json`, `page`, `pageSize=10000`, `sortBy=startTime`, `sortOrder=asc`, header `x-api-key`; `pagination.lastPage/nextPage` drive paging. | Notebook. |
| G3 | Windowing | When a window ends (no `nextPage`), the next window starts at last row `startTime` + 1 s until `endTime` is reached or no rows return. This costs one empty follow-up query per series. | Notebook (the "20,000-row" cap comment). The cap value itself was clipped; the fake transport models it as a configurable per-window row cap. |
| G4 | Retries | 5 attempts; `429` waits `Retry-After` seconds (default 60 s if the header is missing/unparseable); other HTTP errors/exceptions back off `10·attempt` s; `401/403` fail immediately. Delay between pages 6.5 s (0 in tests). | Notebook + spec; the 429 default was clipped. |
| G5 | Cache | Per dataset: `fingrid_raw_<id>_<column>.csv` (all rows as returned: `datasetId, startTime, endTime, value`) plus `.meta.json` (dataset id, column, fetched_at, requested window, row count, first/last sample, pages, requests, complete flag). Both are written atomically, metadata last. | Spec: full response + metadata. |
| G6 | Reuse/resume | A cache is reused when its requested window starts at or before the new start and its last sample is within 15 min of the end (notebook rule). A partial cache resumes from last sample + 1 s. A cache that does not reach the end (e.g. data genuinely missing at the tail) triggers a re-query on every run that has a key; without a key it fails with `CredentialUnavailableError`. | Notebook. |
| G7 | Hourly rule | `resample("1h").mean()` on UTC `startTime`: each hour is the mean of all samples whose start lies in `[h, h+1h)`, labelled `h`. 3-min and 15-min series are therefore averaged by sample count, not by duration. Hours with no sample stay NaN; hours with fewer samples than the series' typical count are listed as `partial_hours` but still averaged. The rule text and observed resolution are stored in every per-series report. | Spec + notebook. |
| G8 | Missing hours | Reported per series as ISO UTC lists; never imputed. `net_load_mw = consumption − wind − solar` is NaN wherever any input is NaN. | Spec. |
| G9 | Joined table | Left join on the ERA5 hours (all ERA5 hours kept; columns follow notebook 1). "Comparable" hours are those with a non-missing actual (2,202 of 2,208 in the reference run); counts and `join_coverage` are returned. | Spec. |
| G10 | Metadata verification | `GET /api/datasets/{id}`; the unit is read from the first present of `unitEn, unit, unitFi`, name from `nameEn, name, title, nameFi`, frequency from `dataPeriodEn, dataPeriod, frequency, resolution, updateCadenceEn`, licence from `license, licenseEn, licence`. **These field names are guesses** (the real response schema was not available). Units compare case/space-insensitively (`€` → `eur`). Mismatch ⇒ `DatasetMismatchError` (records attached); no unit field ⇒ `unit_not_reported`; no key ⇒ `unverified`. | The real schema must be checked on the first authenticated run. |

## Simulation (cells "Prepare ERA5 for atlite", "Multi-archetype simulation", handoff 2a)

| # | Item | Choice | Why |
|---|------|--------|-----|
| M1 | atlite | `atlite.Cutout(path=…, data=<in-memory dataset>)` works offline with atlite 0.7.0 (no download, no file written). The dataset needs `wnd100m`, `roughness`, attrs `module/prepared_features/dx/dy`. `Cutout.__repr__` fails for such datasets (variables lack `module` attrs); it is never called. | Verified by running it. |
| M2 | Direct engine | A NumPy re-implementation of `convert_wind` (log extrapolation to hub height, `np.interp` on the curve, capacity-weighted sum) is provided and used for speed in tests; both engines agree to < 1e-4 MW on the fixture (atlite computes partly in float32). | Spec fallback, kept as a cross-check. |
| M3 | Power curve interpolation | Linear between the 0.5 m/s points; because the curve is 0 at 2.5 m/s and positive at 3.0 m/s, speeds between 2.5 and 3.0 m/s produce a small linear-ramp output (atlite/notebook behaviour). Above 30 m/s: 0. | Notebook. |
| M4 | Layout | `np.add.at` of turbine capacity on the nearest ERA5 centre (argmin per axis), per archetype. Turbines not strictly inside that cell's box (cell edge) or outside the grid are still placed; they are counted in `LayoutReport`, logged at WARNING and noted in `SimulationResult.notes`. | Spec assumption 6. |
| M5 | Hub-height speed | `v_hub = v100·ln(h/z0)/ln(100/z0)`, `z0` clipped to ≥ 1e-4 m, result floored at 0. | atlite's logarithmic method. |
| M6 | Daily table | UTC days; `hours` counts hours with an actual value when an actual column exists; GWh = Σ MW·1 h / 1000. | Notebook. |

## Calibration (notebook 2, handoff 2a)

| # | Item | Choice | Why |
|---|------|--------|-----|
| K1 | Split | `block_id = (t − t.min()) // 72 h`; per-block mean theoretical wind; `n_bins = min(4, max(1, n_blocks//2))`; `pd.qcut(labels=False, duplicates="drop")`; `default_rng(2026)`; per stratum in ascending order: shuffle, `n_test = max(1, int(round(0.2·len)))` (0 if < 2 blocks); fallback last block. Reproduces 31 blocks → 24 train / 7 test. Python's banker's rounding applies (`round(0.4)=0`, so a 2-block stratum gives 1 test block). The split depends on theoretical wind only (a model input), never on labels. | Spec assumption 9, notebook cell. The table is split *after* dropping rows with missing inputs, so block boundaries are relative to the first usable hour. |
| K2 | Anchors | train rows with price > 20 EUR/MWh, wind/consumption < 0.40, theoretical < 3,800 MW (strict inequalities). | Notebook. Fewer than 10 anchors ⇒ `CalibrationError`. |
| K3 | Frontier | `scipy.optimize.least_squares`, `soft_l1`, `f_scale=150`, x0 `[0, 0.92, 10, 0, 0]`, bounds `[-500,0.5,0,-500,-500]..[500,1.3,80,500,500]`, design `[1, P, −(P/1000)², sin h, cos h]` on **train anchors only**. β₂ is stored as the fitted positive parameter (37.72); the notebook printed its negation. | Notebook. |
| K4 | Gate | `score_price = clip((15 − price)/15, 0, 1)`; `score_grid = clip((ratio − 0.4)/0.3, 0, 1)`; `w = 1 − (1 − s_price)(1 − s_grid)`; `curtailment = w·max(0, p_base − actual)`; `p_base = clip(Xβ, 0, 1.15·theoretical)`; `clean = actual + curtailment`. | Notebook 2 (this soft gate is what the notebook uses; the hard thresholds of spec assumption 7 are report text and are not used here). |
| K5 | De-biasing | `LinearRegression` and `SplineTransformer(4, 2, include_bias=False)+Ridge(10)` of `residual_clean` on theoretical wind, fit on train rows only. | Notebook. |
| K6 | `inference_only` model | `ColumnTransformer(spline(theoretical), passthrough(hour_sin, hour_cos)) → Ridge(10)` trained on **observed** actual on train blocks; `predict(theoretical_mw, hour)` has no price/consumption/Fingrid argument; output clipped to [0, installed MW]. Hyper-parameters are those of K5 (no tuning). | Spec (b). |
| K7 | Forward check | Needs ≥ 60 days of span (`last − first + 1 h`, inclusive). Whole 72 h blocks in time order; first `round(0.7·n_blocks)` (at least 1, at most n−1) train, rest test; everything (frontier, linear, spline, inference model) is refit on the forward-train part; metrics use split `forward_test`. Skipped, with an explicit reason, below 60 days or when too few anchors exist. | Spec (c); the block rounding is our choice. |
| K8 | Metric records | `{metric, value, unit, target, split, n, model}`; residual = estimate − reference (positive = over-estimate). `pearson_r2` is the squared correlation; `regression_r2 = 1 − SSE/SST` uses theoretical (or the inference prediction) as-is as the predictor of observed actual, so it can be negative and never exceeds `pearson_r2`. | Spec (a). |
| K9 | Reconciliation | Documented values from the notebooks/report are compared, never forced. Status is `not_reproducible_no_real_data` for synthetic runs; otherwise `match` within 1 % relative (min 0.01 absolute), `differs`, or `unavailable`. | Spec + handoff. |
| K10 | Not implemented in this handoff | Capacity-weighted ERA5 features (spec assumption 10), regime labels (assumption 7) and the ML dataset plots of notebook 2: none feed the frontier, the reconstruction or the inference model. | Scope. |

## Pipeline, validation, manifest and site artifact (handoff 2b)

| # | Item | Choice | Why |
|---|------|--------|-----|
| P1 | `pipeline.py` | Stage logic lives in `pipeline.py` (not listed in the spec's layout); `cli.py` only parses arguments and maps exceptions to exit codes. | Keeps `cli.py` small and testable. |
| P2 | Exit codes | 0 ok, 1 hard check/contract failure (also `DatasetMismatchError`, ERA5 `ValidationError`), 2 inputs unavailable, 3 unexpected error/Fingrid transport error, 64 usage error. argparse's own exit code 2 is remapped to 64. | Spec: unavailable inputs must be distinguishable from validation failure. |
| P3 | Run modes | `smoke` (13 × 17 grid, 21 days from 2025-07-22 so the data cross a month boundary: plain NetCDF + CDS ZIP), `demo` (45 × 52 Finland grid, 28 days from 2025-07-01, labelled DEMO), `full` (2,208 h window). Synthetic modes use their own cache (`cache/synthetic_<mode>`) and are always offline. | Spec/handoff; sizes chosen for run time (smoke about 20 s). |
| P4 | Synthetic Fingrid | In synthetic modes the raw Fingrid cache is derived from the simulation (3 missing wind hours: index 10, 11, n/2), written through the same cache format and read through the same `load_or_fetch_hourly`. Dataset metadata comes from the fake transport and is flagged SYNTHETIC. | Exercises the real code path offline. |
| P5 | Full mode | Missing ERA5/Fingrid credentials, missing caches or `--offline` ⇒ `Unavailable` (listed, written to a minimal `manifest.json` with status `incomplete_inputs_unavailable`); never a synthetic fallback. `--refresh` re-downloads and therefore needs credentials; the fleet needs `--refresh` only for the first fetch. | Spec. |
| P6 | Stage statelessness | Each stage rebuilds its inputs from the raw caches (memoised per process); provenance accumulates in `interim/<mode>/run_info.json`; the manifest is built from that file plus a scan of `outputs/<mode>/`. The simulation is recomputed by every stage that needs it. | Simplicity; separate invocations stay consistent. |
| P7 | Hard checks | See `validate.py`. Documented values (9,790.18 MW etc.) never decide pass/fail. Join coverage threshold 0.95 (configurable). `units` tolerates floating-point negatives down to −1e-6 MW. The forward check is `skipped` (not failed) below 60 days. | Spec. |
| P8 | Reconciliation statuses | `match` within 1 % relative (min 0.01); counts exact; synthetic ⇒ `not_reproducible_no_real_data` (the spec's earlier "not_applicable (synthetic)" wording is superseded by the handoff). | Handoff. |
| P9 | Manifest | One `manifest.json` per run (schema 1.0) with environment/git/library versions, ERA5 window/grid/file hashes, fleet snapshot and imputation counts, Fingrid definitions/coverage/missing hours, per-file rows/columns/dtypes/units/sha256 (CSV schemas are read back from the files), licences, unavailable list, validation, reconciliation and the target/split/model of every metric. CSV dtypes are those pandas infers on re-reading. | Spec. |
| P10 | Site artifact `site/v1` | `wind_effective.bin` is little-endian float16 `[time, lat, lon]` (about 10.3 MB for 2,208 h on 45 × 52); roughness is the time-mean per cell (hourly roughness is not exported); `grid.json` marks every ERA5 cell as supported (no land/sea or border mask exists in the repository) and adds the fleet capacity matrix; `national_hourly.json` also carries the `inference_only` series. Export fails above 15 MB. No national calibration is applied to the site simulation. | Spec; size budget. |
| P11 | Scripts | `run_smoke.ps1` picks the active venv/conda Python, else `.venv`, else PATH. `run_full.ps1` accepts `-CheckOnly`/`-DryRun`, `-UseActivePython`, `-SkipEnv`, `-Refresh`, `-EnvFile`; it prints credential *presence* only and exits 2 when anything is unavailable. Conda is not installed here, so the conda branch (env create/update, `conda run`) is untested. | Handoff. |

## Fixtures and tests

* All fixtures are labelled `SYNTHETIC` (dataset attrs, JSON members, feature names) and are generated
  into pytest tmp dirs or git-ignored `data/`; nothing generated is tracked.
* The test suite blocks sockets and removes credential variables (`FINGRID_API_KEY`, `CDSAPI_*`).
  Fetchers are exercised only through injected fake sessions (`FakeFingridSession`, fake ArcGIS/Overpass
  sessions); the CDS client is replaced by a fake. The fake API key `synthetic-test-key` is a placeholder.
* Synthetic "observed" Fingrid data (`synthetic_observed`) = 0.88 × theoretical + noise, cut back when the synthetic
  price is low or wind/consumption is high, so the calibration code has curtailment to find. Its numbers
  say nothing about the real system.
* Fixture CDS ZIPs mimic the instant/accum split (`tp`, `fsr` in the accum file) with fixed ZIP
  timestamps; NetCDF bytes themselves are not guaranteed bit-identical, values are.
* Synthetic ERA5 values are plausible-looking random fields, not meteorology; they exist to exercise
  code paths (including cold/humid hours for icing via `base_temp_c`).

## Packaging

* `pip install -e .` uses an isolated build and therefore needs `setuptools>=68` from the package index
  (the venv itself ships without it). Use `--no-build-isolation` if setuptools is already installed.
* `requirements.txt` pins the dependency closure of the libraries used, resolved from the development
  venv (Python 3.13). `environment.yml` (conda-forge, Python 3.12) leaves versions unpinned.
