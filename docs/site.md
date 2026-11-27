# Local Finland wind simulator website

A small local website (Python standard-library `http.server` + vanilla HTML/CSS/JS, no CDN, no build step, no
network access at runtime) that reads the versioned artifact `site/v1` produced by `python -m windpower
export-site|demo|all` and offers two views:

1. **Site simulation** - pick a location on a Finland map (click, or arrow keys) or type latitude/longitude, a UTC
   date range, a turbine archetype and an installed capacity; get hourly MW, capacity factor and MWh as a chart,
   a summary, an accessible data table and a CSV download.
2. **Finland overview** - national theoretical (simulated) MW against observed Fingrid MW, data coverage and missing
   hours, date-range zoom, CSV download of the plotted series.

The functional idea follows renewables.ninja (select a place, a period and a turbine, get a time series and a CSV);
layout, code, text and styling here are original.

## Run it

```powershell
python -m windpower demo                         # once: writes the SYNTHETIC demo artifact (offline, ~20 s)
python -m windpower serve --mode demo --port 8765   # prints the URL; Ctrl+C stops it
powershell -File scripts\run_site.ps1 -Mode demo -Port 8765   # same, picks the right Python; -Generate creates the demo artifact first
```

`serve` options: `--mode demo|full|smoke` (default: the first existing artifact of full, demo, smoke), `--site-dir DIR`
(serve any `site/v1` directory), `--data-dir`, `--host` (default `127.0.0.1`, local only), `--port` (default 8000).
If no artifact exists the command prints where it looked and the exact command that creates it, and exits with code 2;
it never generates or synthesises data on its own. The port is bound exclusively (a second server on the same port
fails instead of silently sharing it).

## Data and labelling

The server consumes only `site/v1` (`manifest.json`, `grid.json`, `archetypes.json`, `wind_effective.bin` +
`.meta.json`, `roughness_mean.bin`, `national_hourly.json`) and validates it on start (schema version, array sizes,
ascending axes, matching time axes, 61-point curves). Acquisition, simulation and calibration never run in a request.

**Synthetic data is unmistakable.** The manifest flags (`synthetic`, `demo`, `label`) drive a sticky banner on every
page ("DEMO - SYNTHETIC DATA ..."), a `[DEMO]`/`[SYNTHETIC]` prefix in the page title, the chart captions and result
text, the footer, every JSON response (`model_version`, `run_mode`, `synthetic`, `demo`, `label`, `notice`), and the
first lines of every CSV (`# *** DEMO - SYNTHETIC DATA ... ***`, `# demo: true`) and the CSV file name
(`..._DEMO-SYNTHETIC.csv`).

**Real-data integration gap.** No real ERA5, fleet or Fingrid data exists in this repository yet, so the site has only
ever shown synthetic data. A real run (`python -m windpower all --mode full`, needs credentials) writes
`data/outputs/full/site/v1`, which `serve` prefers automatically; the page then shows a neutral banner ("Retrospective
ERA5 simulation. Uncalibrated ..."). That path is untested beyond the synthetic artifact's identical schema.

## API (read-only, GET/HEAD, JSON unless noted)

| Endpoint | Purpose |
|---|---|
| `/api/v1/meta` | labels, time range, grid description and cell size in km, coarse Finland outline, capacity limits, units, statements, attribution |
| `/api/v1/archetypes` | archetypes with hub height, specific power, fleet share and the 61-point power curve |
| `/api/v1/simulate?lat=&lon=&start=&end=&archetype=&capacity_mw=` | hourly `time_utc`, `power_mw`, `capacity_factor`, `energy_mwh` + summary and cell metadata |
| `/api/v1/simulate.csv?...` | same parameters; CSV with `#` header lines and the columns above |
| `/api/v1/national?start=&end=` | hourly `theoretical_mw`, `observed_mw` (null = missing), `observed_available`, coverage summary, missing ranges |
| `/api/v1/national.csv?...` | CSV of the same series |
| `/` and static files | the single-page UI |

Parameters: `lat`, `lon` required (decimal degrees); `start`/`end` optional UTC, `YYYY-MM-DD` (an `end` date means
through 23:00) or `YYYY-MM-DDTHH[:00[:00]]Z`, inclusive hours, default the whole artifact range; `archetype` default
= the archetype with the most fleet MW; `capacity_mw` default 3, allowed 0.01 - 1000.

**Computation** (server side, shared helpers of `simulate.py`): hourly density-adjusted 100 m wind of the cell →
`hub_height_speed` (logarithmic profile, time-mean cell roughness, archetype mean hub height) → `power_from_speed`
(linear interpolation of the archetype's 61-point curve, per unit) → × installed capacity. MW has 1 W resolution,
`energy_mwh` = MW × 1 h, `capacity_factor` = per-unit output in [0, 1]; the summary total is the sum of the returned
hourly values. **No national calibration and no curtailment correction is applied**, and the page says so.

**Resolution.** Cells are 0.25°: about 27.8 km north-south by 13.9 km east-west at 60°N (11.8 km at 65°N). The result
is the cell-average ERA5 wind, not a micro-sited estimate; the response includes `cell.extent_km` and the page repeats this.

**Errors** are JSON with the right 4xx status, no stack traces: `{"error": {"code", "message", "param"}, "data_label": {...}}`.

| Status | Codes |
|---|---|
| 400 | `missing_parameter`, `invalid_number`, `bad_date`, `unknown_parameter`, `duplicate_parameter`, `bad_request` |
| 422 | `outside_grid`, `outside_finland`, `unsupported_cell`, `capacity_out_of_range`, `unknown_archetype`, `date_out_of_range`, `end_before_start` |
| 404 / 405 / 414 | `not_found`, `method_not_allowed`, `uri_too_long` |

`outside_finland` uses a coarse outline: the 37-vertex mainland polygon hard-coded in the pipeline notebook's map cell,
plus a hand-built rough Åland polygon (not from the notebook), with a 0.25° tolerance so coastal cells are accepted.
Offshore sites further out are rejected. The outline is also what the map draws.

Chart values, table and CSV come from the same computation: the page draws from the JSON response, and its download
link requests the CSV with exactly the parameters echoed in that response (`request.*`); tests check that CSV rows equal
the JSON values.

## Solar (API v1, `/api/v1/solar/*`)

The server can additionally serve the solar artifact `data/outputs/<solar mode>/site/solar/v1` (written by
`python -m windpower solar-demo|solar-all|export-site-solar`, see `docs/solar.md`). It is read only from there, is
validated on start (axes, array sizes, `kind`), and is independent of the wind artifact: either may be missing
(wind routes then answer `404 wind_artifact_not_built`, solar routes `404 solar_artifact_not_built`, each with the exact
generation commands; the page shows an explicit "artifact not built" panel for that source).

```powershell
python -m windpower demo; python -m windpower solar-demo           # SYNTHETIC artifacts, offline
python -m windpower serve --mode demo --solar-mode solar-demo --port 8765   # wind + solar
python -m windpower serve --mode solar-demo --port 8765            # solar demo (+ a wind artifact if one is found)
python -m windpower serve --solar-site-dir D:\path\site\solar\v1   # a custom solar directory, solar only
python -m windpower solar-all; python -m windpower serve           # real partial run, auto-discovered
```

`serve` options added: `--solar-mode solar|solar-demo|solar-smoke`, `--solar-site-dir DIR`. `--mode` also accepts the
solar modes. Selection rules are in `docs/assumptions.md` S35 (an explicit `--site-dir` never pulls in solar and an
explicit `--solar-site-dir` never pulls in wind).

| Endpoint | Purpose |
|---|---|
| `/api/v1/solar/meta` | run mode, SYNTHETIC/DEMO/partial flags, window, 44 x 53 grid (0.25 deg), panel (CSi, 40 deg, azimuth 180, inverter 0.90), units, per-mode availability with exact reasons, series scope texts, coverage summary, statements, licences, Finland outline |
| `/api/v1/solar/grid` | grid axes and per-cell static fields (satellite-covered flag, coverage fractions, population share, large capacity) |
| `/api/v1/solar/coverage` | satellite-covered cells, per-cell coverage fraction, 0.05 deg pixel map, threshold (99.9 %), counts |
| `/api/v1/solar/simulate?lat=&lon=&start=&end=&capacity_mwp=&mode=era5\|hybrid` | hourly `time_utc`, `power_mw`, `capacity_factor`, `energy_mwh`, `irradiance_source`, `fallback_reason` + summary (total MWh, mean CF, peak MW, `satellite_hours`, `fallback_hours`, `fallback_reasons`) and cell metadata |
| `/api/v1/solar/simulate.csv?...` | same parameters; the CSV of exactly that response |
| `/api/v1/solar/national?start=&end=` and `national.csv` | every national series with `available`, `reason`, `unit`, `native_unit`, `scope`, values (null = missing, never zero-filled), energy/coverage summary |

Arithmetic: `power_mw = capacity_factor x capacity_mwp`, `energy_mwh = power_mw x 1 h` (hourly values rounded to 1e-6),
CF is a fraction in [0, 1], night hours are exactly 0. The cell is the 0.25 degree cell containing the point. In
`hybrid` mode each hour is `satellite` (SARAH-3 direct and diffuse, only where the cell has at least 99.9 % valid area in
that hour) or `era5_fallback` with a reason (`outside_satellite_domain`: the north/north-east of Finland is not covered
by the satellite product; `no_coverage`, `partial_coverage`, `missing_or_flagged_step`, `sis_missing`, `sid_missing`,
`negative_diffuse`). In `era5` mode the source is `era5`. Default `mode` = `hybrid` if available; `capacity_mwp`
default 1, range 0.001 - 1000. No national calibration and no curtailment is applied.

Errors use the same JSON shape as the wind API. Solar-specific: 400 `missing_parameter`, `invalid_number`, `bad_date`;
422 `outside_grid`, `outside_finland`, `date_out_of_range` (outside the artifact window), `end_before_start`,
`capacity_out_of_range`, `unknown_mode`, `mode_unavailable` (message = the precise gap, e.g. "ERA5 solar data not ...").

CSV: `#` header lines (flags `synthetic`/`demo`/`partial_run`, model version, run mode, mode, panel, capacity MWp, cell
centre, 0.25 degree resolution, cell extent, coverage fraction, "no national calibration, no curtailment", series scopes
and `source_*` licence/attribution lines for CM SAF/EUMETSAT, Copernicus, Fingrid CC BY 4.0, ...) then the columns
`time_utc,power_mw,capacity_factor,energy_mwh,irradiance_source,fallback_reason`. The chart, table and CSV link all
come from one API response; the tests assert CSV rows equal JSON values.

### Solar UI

* A top-level **Wind | Solar** switch (a `nav` with `aria-current`, plain links to `#simulate`/`#overview` and
  `#solar-simulate`/`#solar-overview`, so it is keyboard operable and works as deep links). A solar-only server opens on Solar.
* **Solar simulation**: map with the satellite-covered cells overlaid as hatching (a pattern, so it is distinguishable
  without colour; a legend and a "show coverage" checkbox), latitude/longitude fields, dates limited to the artifact
  window, capacity in MWp, a radio group for the irradiance mode (unavailable modes are disabled and show their reason),
  summary (total MWh, mean CF, peak MW, satellite vs ERA5-fallback hours and reasons, a plain-language "northern fallback"
  explanation), data-table toggle, CSV download, and a text on resolution (0.25 degrees, about 14 x 28 km at 60 N,
  cell-mean irradiance, fixed 40 degree south-facing panel, no national calibration). Fallback hours are hatched in the chart.
* **Solar overview**: ERA5-only (solid) and hybrid (dashed) modelled national MW with direct labels; Fingrid 248
  (dotted) labelled "Fingrid forecast (not measured generation)"; Fingrid 362 AV06 labelled "Small-scale surplus only
  (kWh/h as MW) - different scope, not total generation" in a **separate panel with its own axis**; 267 capacity as a
  tile (scope "capacity used in the forecast, not official") and a table/CSV column; coverage tiles per series; explicit unavailable states with the reason text; date-range zoom; CSV
  of the plotted series.
* A persistent DEMO/SYNTHETIC banner when flagged, and a notice on partial real runs listing the unavailable inputs
  **once per cause** (short cause, detail, "Affects: ..."), not once per item; tiles and radio buttons carry only the short text
  (`reason_short`, e.g. "ERA5 solar data not ingested", "FINGRID_API_KEY not set"). No absolute local paths are ever served
  (S42). With no usable mode "Run simulation" is disabled and says why; with no available series the CSV button is disabled
  ("Download CSV: none available") and a chart panel without data shows one short state line. Same hygiene as the wind page: no `innerHTML`, strict CSP, no external requests, light/dark,
  reduced motion, forced-colors, 320 px to wide screens.

**Solar verification status.** API, CSV, partial artifact, CLI and HTML/CSS structure: pytest
(`tests/test_web_solar.py`). The solar JavaScript is syntax- and behaviour-**unverified** in a browser by the author
(Node is not installed and no browser session was run); a manual browser check is still required. The real SARAH-3 +
public-data run is partial: with no ERA5 or Fingrid input the real artifact serves coverage, grid and the unavailable
states only. Guidance gap: `npx` is missing here as well, so `modern-web-guidance` was again followed manually.

## Security notes

Loopback by default; GET/HEAD only; path traversal is blocked (decoded `..`, hidden names, backslashes, NUL, drive
letters, directories, symlink escapes are all 404); a strict Content-Security-Policy (`default-src 'none'`, scripts,
styles, images, connections only from the same origin; no inline script or style), `nosniff`, `no-referrer`,
`X-Frame-Options: DENY`; the UI never uses `innerHTML`; request logs omit query strings (coordinates).

## Accessibility and design notes

Principles from the modern-web-guidance guides (read directly) were applied by hand:

* semantic landmarks (banner region, `nav`, `main`, `footer`), one `h1`, a skip link, native `<details>` for data
  tables and `<dialog>` for map help, `<output>` for chart read-outs, every control has a visible `<label>`;
* forms: native types (`number`, `date`, `select`), min/max, errors shown inline *and* in a summary box that takes
  focus after a failed submit; `aria-invalid` is `false` until a field is actually flagged and `aria-describedby`
  lists the error only while it is shown; one polite `role="status"` live region announces results;
* keyboard: the map is a focusable group - arrow keys move one cell, Enter runs the simulation, the latitude/longitude
  fields are the equivalent alternative; visible `:focus-visible` outlines; targets ≥ 44 px; the page works without a mouse;
* charts: inline SVG with a text caption stating the headline numbers, `role="img"`, a data-table alternative, series
  told apart by **line style (solid vs dashed) and direct labels**, not only colour, plus a shaded band for missing data;
  hover/pointer read-out; drag across the national chart to zoom (the date fields are the keyboard route);
* responsive from 320 px (single column, tables scroll in a labelled region) to wide screens: CSS grid, `clamp()` type
  and spacing, container queries for the form grids; `prefers-color-scheme` light/dark palettes (contrast checked:
  text ≥ 13:1, secondary text ≥ 7:1, series colours ≥ 5.9:1 on their backgrounds, borders ≥ 4.5:1),
  `prefers-reduced-motion` (only optional transitions), `forced-colors` fallback, `color-scheme` meta;
* times are shown with `Intl.DateTimeFormat` in UTC and labelled "UTC"; API times are ISO 8601 `Z`.

**Verification status.** Server, API, CSV, static hygiene and the structure of `index.html` are covered by pytest
(`tests/test_web.py`). The JavaScript has no automated tests (Node is not installed); it was exercised by loading the
page in headless Microsoft Edge with the demo artifact (simulation result, validation error, overview, dark and
320 px layouts; screenshots inspected by hand). Not done: screen-reader testing, keyboard-only walkthrough in a real
browser, Firefox/Safari checks, an automated contrast or axe audit.

**Guidance gap.** `npx` is not available on this machine (`where npx` finds nothing), so the `modern-web-guidance`
tool could not be run; its guides under `.copilot\skills\modern-web-guidance\guides\` (accessibility, css, forms,
html) were read instead and followed manually as listed above.

## Known limitations

* Roughness is the time-mean per cell (the artifact does not carry hourly roughness), so the hub-height profile is
  slightly smoother than the pipeline's hourly one.
* Wind is stored as float16 (about 3 significant digits), adding up to ~0.1 % noise in the wind speed.
* No per-turbine siting, wake, availability or loss model; one archetype with the capacity you enter.
* One artifact is loaded at start-up; restart the server after regenerating it.
