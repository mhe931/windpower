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
