# Reproducing the results

## 0. What you can and cannot reproduce today

Without credentials only the synthetic `smoke` and `demo` runs work (offline). The documented numbers can only be
reproduced by a **full** run, which needs ERA5 (Copernicus CDS account) and a Fingrid API key.

## 1. Offline (synthetic)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps      # needs setuptools from the index (build isolation)
.\.venv\Scripts\python.exe -m pytest -q                        # offline suite
powershell -File scripts\run_smoke.ps1                         # ~20 s, prints SMOKE PASS, exit 0
.\.venv\Scripts\python.exe -m windpower demo                   # synthetic Finland grid -> data\outputs\demo\site\v1\
```

`run_smoke.ps1` uses the active virtualenv/conda Python, else `.venv`, else `python` on PATH. Generated data goes to
`data\` (git-ignored); pass `-DataDir <dir>` to write elsewhere.

## 2. Full run (real data)

Credentials (never commit them; `.env` and `.cdsapirc` are git-ignored):

| Need | Provide |
|---|---|
| Fingrid | `FINGRID_API_KEY` in the environment, or `FINGRID_API_KEY=...` in an untracked `.env` (copy `.env.example`) |
| ERA5 | `CDSAPI_URL` + `CDSAPI_KEY`, or `%USERPROFILE%\.cdsapirc` (accept the ERA5 licence on the CDS site) |
| Fleet | none (public); needs network |

```powershell
# Conda route (creates/updates env "windpower" from environment.yml, checks credentials, runs every stage)
powershell -File scripts\run_full.ps1 -CheckOnly        # report what is missing; exit 2 if anything is
powershell -File scripts\run_full.ps1                   # then the real run

# Existing venv instead of conda
powershell -File scripts\run_full.ps1 -UseActivePython

# Stage by stage (same result)
python -m windpower ingest-era5   --mode full           # CDS download, one request per month, validated
python -m windpower ingest-fleet  --mode full --refresh # OSM + ArcGIS raw responses, snapshot sidecars
python -m windpower ingest-fingrid --mode full          # paged, rate-limited, cached; dataset metadata checked
python -m windpower all --mode full --offline           # later: everything from the caches, no network
```

Flags: `--refresh` re-downloads (needs network and credentials), `--offline` forbids the network,
`--start/--end` change the UTC window (default 2025-07-01T00 .. 2025-09-30T23, 2,208 hours), `--data-dir` moves the
cache/interim/outputs root.

Exit codes: `0` ok · `1` hard check failed · `2` inputs unavailable (listed on stderr and in `manifest.json`;
no synthetic data is ever substituted) · `3` unexpected error · `64` usage.

Fingrid fetches honour `Retry-After` on HTTP 429, retry 5 times, and resume from a partial cache; the first run
takes a while because of the 6.5 s request delay.

## 3. After a full run

1. `data\outputs\full\validation.json` - every hard check must pass (`python -m windpower validate --mode full`).
2. `data\outputs\full\reconciliation.csv` - compare with `docs/reconciliation.md`; record the observed values and
   the snapshot dates (`manifest.json → fleet.snapshot`) in the "Real run" section there.
3. `data\outputs\full\manifest.json` - licences marked "to be verified" must be checked before publishing; confirm the
   Fingrid metadata units (`fingrid.verification`) and the ArcGIS layer licence text.
4. `data\outputs\full\site\v1\` - the website artifact (< 15 MB for the full Finland Q3 grid; about 10.3 MB for the wind
   field).

## 4. Determinism

Fixtures are seeded (`seed=2026`); the block split uses `default_rng(2026)`. File hashes in `manifest.json` let
two runs be compared. NetCDF bytes of synthetic files are not guaranteed identical, but their values are.
