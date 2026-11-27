# Replay a recorded solar site on Windows

There are **two different experiments**: a 96-hour `solar-demo` with synthetic
SARAH, ERA5, public and Fingrid inputs, and a 744-hour real `solar` export with
July SARAH-3 and public allocation data but **no** ERA5 or Fingrid series.
Only the first can answer the site PV query. Do not fill the second with
fixture data. Neither experiment measures national PV production.

## From a fresh checkout (PowerShell)

Prerequisites: the **same pipeline source and Python/package versions** as
the original; a trusted Python environment with the project requirements already
installed (see README Quick start for provisioning); and a private, authorized
local copy of the inputs. Git does **not** contain the kit. For an existing
kit, use the producer revision that created it rather than a newer branch tip.

```powershell
git clone --branch feature/eo-provenance-catalog https://github.com/mhe931/windpower.git
Set-Location windpower
$repo = $PWD.Path
$py = (Get-Command python -CommandType Application | Select-Object -First 1).Source # activated project env
$env:PYTHONPATH = Join-Path $repo src
$tool = Join-Path $repo scripts\dataset_replay.py
$known = Read-Host 'Absolute path to authorized existing windpower data root (read only)'
$fx = Join-Path $known cache\synthetic_solar-demo
$lab = Join-Path $env:TEMP ('wp-t5-july-' + [guid]::NewGuid())
New-Item -ItemType Directory $lab -ErrorAction Stop | Out-Null
Set-Location $lab
```

**Build the reference and capture its inputs.** Point `--sis-dir` and `--sid-dir`
at the actual daily `.nc` folders, not their parent. `--public-dir` contains
plants XLSX, population CSV, municipalities GeoJSON **and** their cache metadata;
`--fingrid-dir` contains the three cached CSV/meta pairs. `--era5-path` can be a
single file or the loader's directory of NetCDF/ZIP inputs. The examples use
the layout of the known local caches, not a request to generate/download them.
If any synthetic input is unavailable, obtain the original private kit rather
than running `solar-demo` to make a replacement.

```powershell
& $py $tool build --mode solar-demo --sis-dir "$fx\sarah\SYNTHETIC_SIS" --sid-dir "$fx\sarah\SYNTHETIC_SID" --era5-path "$fx\era5" --public-dir "$fx\public" --fingrid-dir "$fx\fingrid" --data-dir "$lab\fx-original"
if ($LASTEXITCODE) { throw 'fixture capture failed' }
& $py $tool build --mode solar --start 2025-07-01 --end 2025-07-31 --sis-dir "$known\interim\smoke\ORD73128" --sid-dir "$known\interim\smoke\ORD73129" --public-dir "$known\cache\solar\public" --data-dir "$lab\real-original"
if ($LASTEXITCODE) { throw 'partial capture failed' }
```

`--start`/`--end` select UTC hour starts; a date-only end includes 23:00.
Supply `--convention end` or `start` only if the original needs that ERA5
override; otherwise the normal detection applies. The source inventory includes
**August 1** for July 31 23:00: the trapezoidal SARAH mean uses the following
00:00 observation. End-stamped ERA5 also needs the next stamp. Don't strip
the extra day or re-encode the files. Supply a stable input directory during
capture. Real gaps present *at build time* remain gaps, not zeros.

**Rebuild, don't copy the site.** For each build, only `inputs/` (private)
and `outputs/<mode>/site/solar/v1/` (reference for verification) are required.
They can be moved independently; neither `run_info.json` nor the original
cache paths are needed. If already given these two directories, **start here**
and substitute their locations. Use a new, nonexistent `--data-dir` each time.

```powershell
New-Item -ItemType Directory "$lab\replay-work" -ErrorAction Stop | Out-Null
Set-Location "$lab\replay-work"
$f = "$lab\fx-original\outputs\solar-demo"
$r = "$lab\real-original\outputs\solar"
& $py $tool replay --site "$f\site\solar\v1" --inputs "$lab\fx-original\inputs" --reference-hourly "$f\sarah_hourly_era5grid.nc" --data-dir "$lab\fx-rebuilt"
if ($LASTEXITCODE) { throw 'fixture replay failed' }
& $py $tool replay --site "$r\site\solar\v1" --inputs "$lab\real-original\inputs" --reference-hourly "$r\sarah_hourly_era5grid.nc" --data-dir "$lab\real-rebuilt"
if ($LASTEXITCODE) { throw 'partial replay failed' }
```

The tool verifies the reference record, manifest linkage and artifact hashes;
checks exact raw file sizes and SHA-256s (including timestamp-bearing
sidecars); stages **only** recorded files; then runs SARAH hourly processing
and regridding, allocation, and, where possible, ERA5 conversion, atlite PV,
national processing and export. No HTTP input client, fixture generator or
original site payload supplies model inputs. The rebuild takes the mode,
inclusive UTC window and convention override from the record. The remaining
grid, panel, coverage and encoding settings are code-defined; incompatible
settings/code/environment fail the version check, not a tolerance test.
An unrecorded file in the kit is ignored. Missing/changed recorded files
are errors, even for a partial dataset. A cached SARAH intermediate is not
proof of its raw daily lineage: with verified complete lineage we check the
recorded intermediates but rebuild from daily inputs; with cached-only or
unverified lineage this **raw** replay stops rather than claiming success.

Each replay starts two short-lived loopback HTTP servers in turn, one per
artifact. It checks dataset/version equality and compares exact response bodies
for `/api/v1/solar/{simulate,simulate.csv,national,national.csv}`. The site
comparison covers **every** timestamp, MW, CF, MWh, irradiance source and
fallback reason, including null/blank reasons; the default mode is hybrid if
available. No clipping of comparison dates or rounding beyond the published API.
The expected no-mode result is HTTP 422 `mode_unavailable`, not a row of zeroes.

`--reference-hourly` is comparison-only: it is opened **after** processing,
not staged as an input. Both examples compare the independently reconstructed
hourly SARAH NetCDF to the reference with xarray `assert_identical` after CF
decoding: all variables (including GHI, direct, diffuse, valid area fraction
and source code), coordinates, NaN positions and dataset/variable attributes.
Compression, chunking and container timestamps need not match. The report lists
shape, values and missing count for every field. If the private kit excludes
that reference, omit the option; the report then explicitly says `not compared`.
Without this check the real partial site verifies coverage and availability,
not all of the underlying hourly irradiance values.

**Repeat the query interactively.** No frontend build service is required;
`serve` uses the static assets shipped with the package. In the shell above:

```powershell
& $py -m windpower serve --data-dir "$lab\fx-rebuilt" --mode solar-demo --port 8765
```

In a second PowerShell window:

```powershell
$url = 'http://127.0.0.1:8765/api/v1'
$q = 'lat=60.17&lon=24.94&capacity_mwp=2.5'
Invoke-RestMethod "$url/datasets" | ConvertTo-Json -Depth 8
(Invoke-RestMethod "$url/solar/simulate?$q").summary
(Invoke-WebRequest -UseBasicParsing "$url/solar/simulate.csv?$q").Content
```

Ctrl+C the first server; repeat with `fx-original` for the same query, or
`--data-dir "$lab\real-rebuilt" --mode solar` for the real partial case.
For the latter use `Invoke-RestMethod "$url/solar/national"` to inspect times
and unavailable series. `http://127.0.0.1:8765/#solar-simulate` is the UI.
Keep the private roots outside Git and shared/cloud storage; don't publish
the kit, credentials or private paths. Delete only your own scratch roots
when no longer needed. A checksum binds bytes, not authenticity or permission.

## Execution evidence (Windows, 2026-11-27)

Both builds used the checkout's `.venv\Scripts\python.exe` (Python 3.13.17,
atlite 0.7.0), local caches, and new private output directories. Final
verification ran both replays from a separate checkout of the staged source
with LF line endings, with `PYTHONPATH` pointing to that checkout's `src`.
The original Windows checkout had CRLF and mixed endings. Only Python source
line endings are normalized for identity; raw inputs are verified byte for byte.
The recorded package versions are in
`dataset.json:pipeline.versions`; the version hashes below bind this particular
code/environment and these inputs, not all runs of the same model.

**Synthetic fixture (July 28–31).** Both sides verified `finland-solar-pv` /
`f237398d3b187a8fda91522ca6e3b127934f872b8699f378a134d48db3d08fc9`.
Producer SHA-256: `3ccb974787e52029c22451a88b58cc8b6a5a1d13e1c9437841351b36d6678584`.
23 input files (10,843,733 bytes): 5 SIS + 5 SID daily files, 1 synthetic
ERA5 archive, 6 public fixture files and 6 Fingrid fixture files. All seven site
payload hashes matched; 756 of 2,332 cells fully covered. Each of the five
SARAH variables had 223,872 matching values (96 × 44 × 53); 140,172 missing
in **each** irradiance field and none missing in validity or source code.
At 60.17 N, 24.94 E, 2.5 MWp, all 96 rows matched: 43.92825 MWh, peak
1.933 MW, 93 satellite hours and 3 `missing_or_flagged_step` fallback hours.
Identical HTTP simulation JSON / CSV SHA-256s:
`4f9ca25d4e56e932e2026257828b36fa2c7956a34b9bed77f0057b1477159e59` /
`abea622bfa33882d121e12909cbf05f69b5bc9d0609d967d9bcffef7bc81a284`.

**Real partial July 1–31.** Both sides verified `finland-solar-pv` /
`abcfcb824d2bb78db0eb6c2202ce567297863aa97e1b5319b61d3d93e9cfb030`.
Producer SHA-256: `9de982a808eba4bd643f666a23c10658580b5261c4e652d33e892bba92ec9b60`.
70 files (46,936,194 bytes): 32 SIS + 32 SID daily files and the three
public inputs with their sidecars. Each replay used 1,489 half-hour stamps
through Aug 1 00:00 UTC for 744 hourly means. All four site payloads matched;
777 of 2,332 grid cells were fully covered. Decoded equality passed for all
five fields, 1,735,008 cell-hours **per field** (744 × 44 × 53):
1,046,808 identical NaN positions in each of GHI, direct and diffuse;
zero missing in `valid_area_fraction` and `source_code`. Of the latter,
578,088 cell-hours were satellite, 828,072 outside domain, 218,736 no
coverage and 110,112 partial coverage. These are SARAH selection codes,
**not** actual ERA5 replacements. ERA5 solar and Fingrid 248/267/362 were
absent in both runs. Both returned HTTP 422 `mode_unavailable` for site PV,
and 744 national timestamps without values for the seven unavailable series.
National CSV SHA-256: `e9d9c51cd9d037a548c5f78c5c1c2db3bed966430d15e0a64feead91e2604247`;
coverage payload SHA-256: `7d06d23d6d8470102e5bc8e8c8cfe8d09eb8266ee06b8b3f57010fb70dc08a24`.

**Failure checks.** From the checkout:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_dataset_replay.py --basetemp "$env:TEMP\wp-t5-test-a3"
```

Use a new temporary test path on subsequent runs (pytest clears its basetemp).
3 passed, 123 existing pandas deprecation warnings. The custom-window case
uses a path-supplied ERA5 file and explicit convention, ignores an unrecorded
extra file, refuses to overwrite the destination, and rejects a same-size byte
change or deletion of the SARAH lookahead before creating a new run. Tests
also reject cached-only lineage and altered science data/coordinates/attributes
while accepting different compression. A pre-final replay across an exporter
revision failed the version check as expected; its hashes are not evidence for
the final producer.

## Independent browser checks

**Real July.** Both Solar views linked to the same real partial version and
showed its 744-hour UTC interval. Site simulation was disabled rather than
substituting fixture power. Keyboard navigation back to Wind and running its
labeled smoke simulation still produced a chart.

**Final rebuilt `solar-demo`.** The project browser operator verified the
96-hour version above independently at the Helsinki site and 2.5 MWp:
43.92825 MWh, 93 satellite hours and three flagged ERA5 fallback hours. All
96 rows × six fields of the actual browser-downloaded CSV agreed with the
API; every SVG power point was within 0.0498 pixels of its API value. The
dataset detail link led to nine input sources and seven payloads. A northern
site used ERA5 for all 96 hours; an out-of-window date was rejected.

Supplementary check on the **synthetic `solar-smoke`** UI at
60.45 N, 22.27 E, 2 MWp, hybrid, July 30–31. All 48 downloaded CSV rows
matched the API on the six hourly fields; the total was 18.996 MWh, with
45 satellite hours and three flagged-step fallbacks. All 48 plotted power
points agreed within 0.048 SVG pixels (path coordinates round to tenths).
The link matched the response, and keyboard navigation to Solar overview and
the Wind site run passed. This check preceded the final producer metadata;
it is not a browser check of the final `solar-demo` version or the real site.

## Architecture crosswalk

Read-only comparison against local thesis project `kvarken`, revision `8367041`.
No thesis source, text or data
is bundled, and the runtime has no dependency on that checkout.

| Reference at that revision | What the windpower implementation does |
|---|---|
| `src/kvarken_eo/provenance.py`: SHA-256 of a canonical payload, append one ingestion record to JSONL | **Reused principle, adapted implementation.** Canonical dataset identity plus exact raw and artifact byte hashes. A local `dataset.json`/manifest pair describes one export; it is not an append-only ingestion log. |
| `src/kvarken_eo/catalog.py`: SQLite/WAL catalog with spatial prefilter and scene query | **Adapted discovery concept.** Windpower discovers the loaded local dataset/version records, not scenes in a persistent SQL index. A historical SQLite index would be a proposed extension. |
| `src/kvarken_eo/api.py`: `/api/health` and `/api/scenes` with bbox/time filters | **Adapted interface concept.** Windpower exposes `/api/v1/datasets` and version detail links, plus grid-cell and UTC-window solar query/CSV routes. These are not compatible scene or health endpoints. |
| Object storage | **Proposed only.** No compatible S3 adapter identified in the reference. Local file-key bindings suffice here; neither an object store nor an S3 service is deployed or tested. |
