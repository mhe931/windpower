# Data sources, licences and caveats

Statuses: **verified** = checked from something in this repository's environment; **assumed** = stated from
general knowledge and *not* checked against the live licence text; **to be verified** = must be checked before
publishing anything derived from real data.

| Source | Used for | Access | Licence / attribution | Status |
|---|---|---|---|---|
| ERA5 hourly single levels (Copernicus Climate Change Service), dataset `reanalysis-era5-single-levels` | `u10 v10 u100 v100 t2m d2m sp fsr lcc tp` on 59.5-70.5 N, 19.0-31.75 E, 0.25° | `cdsapi`, credentials from `CDSAPI_URL`/`CDSAPI_KEY` or `~/.cdsapirc`; one request per month | Copernicus licence (CC BY 4.0 compatible). *Contains modified Copernicus Climate Change Service information.* | assumed - **blocked, no credentials, never downloaded** |
| OpenStreetMap wind generators (`power=generator`, `generator:source=wind`) via Overpass API | turbine positions and tags | public, mirrors `overpass-api.de`, `overpass.kumi.systems`; full response + `osm3s.timestamp_osm_base` cached | ODbL 1.0, © OpenStreetMap contributors | assumed - not fetched in this repo |
| Finnish wind turbine layer (ArcGIS FeatureServer `Tuulivoimalat_WindTurbines`) | official turbine attributes | public; paged by 500 ObjectIDs; layer `editingInfo`/`dataLastEditDate` and `copyrightText`/`licenseInfo` stored in the snapshot sidecar | taken from the layer metadata **at fetch time** (`manifest.licences.arcgis_layer`) | to be verified - text not captured yet |
| Fingrid open data, datasets 181 (wind), 248 (solar), 124 (consumption), 319 (imbalance price) | observed wind output and market/grid signals | `data.fingrid.fi` API, header `x-api-key` from `FINGRID_API_KEY`; ~6.5 s between requests | CC BY 4.0 (Fingrid Oyj) | to be verified at https://data.fingrid.fi - **blocked, no key** |
| atlite 0.7.0 | wind conversion | PyPI | MIT | verified from installed package metadata |

## Caveats that affect results

* **Snapshot dates.** The fleet inventory is whatever OSM and the Finnish layer contain on the day of the fetch
  (`manifest.fleet.snapshot`: ArcGIS retrieval and last-edit date, OSM base timestamp and retrieval time). It "may
  differ from the 2025 fleet" that produced the documented 9,790.18 MW; differences in `docs/reconciliation.md`
  must not be read as code errors without checking the snapshot dates.
* **ERA5T vs final ERA5.** Recent months can be preliminary (ERA5T). Files with an `expver` dimension are not
  handled and fail validation.
* **Fingrid resolution.** Wind and consumption are sub-hourly (3-15 min). The hourly value is the mean of all
  samples starting in the hour (`fingrid.HOURLY_RULE`); missing hours are reported, never imputed. The 6 missing
  wind hours of the reference run are a *documented target*, not an assumption.
* **Fingrid dataset metadata.** The unit check (`MW` for 181/248/124, `EUR/MWh` for 319) reads field names that
  are guesses; `unit_not_reported` and `unverified` statuses are recorded in the manifest instead of passing silently.
* **No key material is stored.** Keys come from the environment or an untracked `.env`; they are never logged,
  cached, written to the manifest or used in tests (tests use a placeholder with a fake transport).
* **Local-only references.** The PDFs, the DOCX report and their text exports live in `references/`, which is
  git-ignored. They are not redistributed; nothing in this repository quotes them beyond the numeric targets.

## Synthetic fixtures

`windpower.fixtures` generates ERA5-like NetCDF/ZIP files, an ArcGIS-like GeoJSON with Finnish field names, an
Overpass-like JSON and fake Fingrid pages. Everything is stamped `SYNTHETIC`; no third-party data is contained.
