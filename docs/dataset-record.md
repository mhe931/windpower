# `dataset.json` (site dataset contract 1.0)

Both production exporters publish this record beside the existing `manifest.json`.
Read `dataset_id` (`finland-wind` or `finland-solar-pv`) and opaque `version_id` together.
`synthetic` is independent of `partial`: a synthetic run can have complete evidence, and
a real SARAH-only site is not national PV generation. `quality` describes current payload
availability, **not** the last validation report found on disk. `coverage` is inclusive
UTC hour starts; `grid` describes cell centres. `artifacts` lists only this export's
payloads. `access` refers to the local site; it does not grant access to any raw data.

## Identity

`windpower.dataset_record.version_id(record)` computes SHA-256 over UTF-8 JSON with
sorted keys, no optional whitespace and no NaNs. The projection includes settings,
coverage, grid, units, status, quality, environment versions, the digest of the local
pipeline module bytes (including uncommitted source), actual payload names/sizes/SHA-256s,
and each selected input's portable key and digest or explicit gaps. For cache JSON
sidecars the identity digest is over canonical JSON without top-level `retrieved_utc`
and `fetched_at_utc`; the **exact** sidecar SHA-256 remains in `inputs` for byte replay.
When the cached SARAH products and their complete daily lineage are verified, reuse
does not add the intermediate encoding to the identity. Otherwise those cached bytes
are independent input evidence and gaps remain explicit.

No Git dirty flag, generation clock, working directory or credential is an identifier.
`dataset.json` and the timestamp-bearing legacy manifest are not their own payloads:
the legacy manifest separately hashes `dataset.json`. `verify_dataset(site_dir)` checks
the version, local payload bytes and that manifest linkage. These are consistency
checks, not signatures. Re-encoding a source NetCDF/ZIP or changing any payload bytes
can legitimately change the version even when the hourly values compare equal.

`pipeline.source_sha256` covers the mode-specific list of local producer modules in
`dataset_record.write_dataset`, not a whole checkout or a runtime execution trace.
Only for those Python modules, each CRLF byte pair is hashed as LF so checkout line endings
do not affect identity. Lone CRs, other whitespace and the presence of a final newline remain
significant. Input and exported payload byte hashes are unchanged by this rule.
External packages are identified by reported version, not by hashing their installed
code or resources (including atlite's CSi panel template). Local edits to those resources
are not detected by the code digest; changed exported bytes are detected by payload hashes.
The reader, web-side site calculation, routes and UI are outside the producer digest.

## Private replay (Task 5)

Use the **same run instance** that produced the site, e.g. `run = SolarRun(...)` then
`run.export_site_solar()` (wind: `Run(...).export_site()`). `run.source_bindings` is
a local-only mapping of portable keys to the original files at export. It is never
serialized. Do not publish or stage raw inputs unless authorized. To check an authorized
private copy, starting with a new empty local directory:

```python
from pathlib import Path
from shutil import copyfile
from windpower.dataset_record import verify_dataset
from windpower.source_inventory import validate_inventory

record = verify_dataset(site_dir)
root = Path(private_replay_dir)
root.mkdir(exist_ok=False)
for key, original in run.source_bindings.items():
    target = root / key
    target.parent.mkdir(parents=True, exist_ok=True)
    copyfile(original, target)
assert not validate_inventory(record["inputs"], root)
```

This checks the recorded bytes, not scientific completeness. Inspect
`quality.input_gaps` and `inputs.sources[*].unavailable` separately. A standalone
array-level export without a live run has no raw lineage and is explicitly partial.
