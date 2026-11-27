"""Build a private solar replay kit and independently process its recorded inputs."""

import argparse
import hashlib
import http.client
import json
import shutil
from pathlib import Path

import xarray as xr

from windpower.dataset_record import verify_dataset
from windpower.export_site_solar import SOLAR_SITE_DIR
from windpower.io_utils import iso_utc, sha256_file
from windpower.solar_pipeline import HOURLY_FILE, SolarRun, parse_window
from windpower.source_inventory import validate_inventory
from windpower.web.server import make_server, start_in_thread
from windpower.web.solar_data import SolarSiteData


class BytesOnlyRun(SolarRun):
    """Keep the real mode/grid/labels, but never invoke synthetic ingest on cache miss."""

    def sarah_dirs(self):
        return self.explicit

    def _synthetic_sarah_paths(self):
        return self.explicit

    def blocked(self, *args, **kwargs):
        raise ValueError("required cached input missing; fetching/generation disabled")

    ingest_solar_public = ingest_fingrid_solar = ingest_era5_solar = blocked


def fresh_run(root, mode, start, end, convention, sis, sid, era5):
    if mode not in ("solar", "solar-demo", "solar-smoke"):
        raise ValueError("not a solar run")
    root.mkdir(parents=True, exist_ok=False)
    run = BytesOnlyRun(mode, root, offline=True, env={}, sarah_sis_dir=sis, sarah_sid_dir=sid,
                       era5_solar_path=era5, era5_convention=convention)
    if start or end:
        run.hours = parse_window(start or iso_utc(run.hours[0]), end or iso_utc(run.hours[-1]))
    return run


def check_inputs(record, root):
    sources = record["inputs"]["sources"]
    if record["dataset_id"] != "finland-solar-pv" or not sources:
        raise ValueError("a solar export with captured raw evidence is required")
    issues = validate_inventory(record["inputs"], root)
    if issues:
        raise ValueError(f"recorded input validation failed: {issues}")
    for key, source in sources.items():
        for gap in source["unavailable"]:
            if gap in ("not used by this run", "series unavailable for requested window"):
                continue
            if key in ("sarah-sis", "sarah-sid") and gap.endswith(": no daily file used"):
                continue
            raise ValueError(f"incomplete input evidence: {key}: {gap}")
    if "sarah-products" in sources and any(sources[k]["unavailable"] for k in ("sarah-sis", "sarah-sid")):
        raise ValueError("cached SARAH products are not evidence of replayable daily inputs")


def copy(src, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as inp, target.open("xb") as out:
        shutil.copyfileobj(inp, out)


def raw_hashes(source):
    return {f["path"]: f["sha256"] for f in source["files"]}


def compare_hourly(reference, rebuilt):
    """Scientific equality after CF decoding, not equality of NetCDF compression/encoding."""
    with xr.open_dataset(reference) as before, xr.open_dataset(rebuilt) as after:
        xr.testing.assert_identical(before, after)
        return {"dimensions": dict(after.sizes), "variables": {
            name: {"values": var.size, "missing": int(var.isnull().sum())} for name, var in after.data_vars.items()}}


def export(run):
    run.export_site_solar()
    site = run.out / SOLAR_SITE_DIR
    return site, verify_dataset(site)


def build(args):
    for path in (args.sis_dir, args.sid_dir, args.public_dir, args.fingrid_dir, args.era5_path):
        if path is not None and not path.exists():
            raise ValueError(f"specified input does not exist: {path}")
    run = fresh_run(args.data_dir, args.mode, args.start, args.end, args.convention,
                    args.sis_dir, args.sid_dir, args.era5_path)
    # These adapters read only in the offline export path. Do not call run_all/ingest.
    if args.public_dir:
        run.public_cache = args.public_dir
    if args.fingrid_dir:
        run.fg_cache = args.fingrid_dir
    site, record = export(run)
    if run.synthetic and record["partial"]:
        raise ValueError("fixture build is incomplete; supply the complete recorded fixture caches")
    inputs = args.data_dir / "inputs"
    for key, path in run.source_bindings.items():
        copy(path, inputs / key)
    check_inputs(record, inputs)
    print(json.dumps({"site": str(site), "inputs": len(run.source_bindings), "version_id": record["version_id"]}))


QUERY = "lat=60.17&lon=24.94&capacity_mwp=2.5"


def responses(site):
    """Run the same request against each site's actual loopback HTTP handler."""
    solar = SolarSiteData.load(site)
    query = QUERY if solar.manifest["run_mode"] != "solar-smoke" else "lat=60.75&lon=22&capacity_mwp=2.5"
    with make_server(None, port=0, quiet=True, solar=solar) as server:
        thread = start_in_thread(server)
        conn = http.client.HTTPConnection(*server.server_address[:2], timeout=30)
        try:
            result = {}
            for endpoint in ("national", "national.csv", "simulate", "simulate.csv"):
                path = endpoint + ("?" + query if endpoint.startswith("simulate") else "")
                conn.request("GET", "/api/v1/solar/" + path)
                res = conn.getresponse()
                expected = 422 if endpoint.startswith("simulate") and not solar.default_mode else 200
                body = res.read()
                if res.status != expected or expected == 422 and json.loads(body)["error"]["code"] != "mode_unavailable":
                    raise ValueError(f"unexpected {endpoint} response: {res.status}")
                result[endpoint] = body
            return result
        finally:
            conn.close()
            server.shutdown()
            thread.join()


def replay(args):
    original = verify_dataset(args.site)
    check_inputs(original, args.inputs)
    sources = original["inputs"]["sources"]
    cov = original["coverage"]
    stage = args.data_dir / "raw"
    run = fresh_run(args.data_dir, original["run_mode"], cov["first_hour_utc"], cov["last_hour_utc"],
                    original["settings"]["convention_override"], stage / "sarah-sis", stage / "sarah-sid", None)
    # Only attested files enter adapter-visible directories. Never glob the supplied kit.
    for key, source in sources.items():
        for i, f in enumerate(source["files"]):
            src = args.inputs / f["path"]
            if key in ("sarah-sis", "sarah-sid"):
                target = stage / key / src.name
            elif key == "era5-solar":
                target = run.era5_cache / f"input-{i:06}.nc"
            elif key in ("plants", "population", "municipalities"):
                target = run.public_cache / src.name
            elif key.startswith("fingrid-"):
                target = run.fg_cache / src.name
            elif key == "sarah-products":
                continue  # verified above; recompute from attested daily lineage
            else:
                raise ValueError(f"unsupported source: {key}")
            copy(src, target)
            if sha256_file(target) != f["sha256"]:
                raise ValueError(f"changed during staging: {f['path']}")
    site, rebuilt = export(run)
    if (original["dataset_id"], original["version_id"]) != (rebuilt["dataset_id"], rebuilt["version_id"]):
        raise ValueError(f"rebuilt dataset/version differs: {rebuilt['version_id']}")
    # Version identity can ignore sidecar retrieval timestamps. Byte replay cannot.
    for key, source in sources.items():
        if key != "sarah-products" and raw_hashes(source) != raw_hashes(rebuilt["inputs"]["sources"][key]):
            raise ValueError(f"rebuilt input evidence differs: {key}")
    before, after = responses(args.site), responses(site)
    for name in before:
        if before[name] != after[name]:
            raise ValueError(f"HTTP response differs: {name}")
    sim = json.loads(after["simulate"])
    if "error" not in sim:
        for field in ("time_utc", "power_mw", "capacity_factor", "energy_mwh", "irradiance_source", "fallback_reason"):
            if len(sim["data"][field]) != cov["hours"]:
                raise ValueError(f"incomplete hourly simulation response: {field}")
    hourly = compare_hourly(args.reference_hourly, run.out / HOURLY_FILE) if args.reference_hourly else "not compared"
    print(json.dumps({"dataset_id": rebuilt["dataset_id"], "version_id": rebuilt["version_id"], "hours": cov["hours"],
                      "partial": rebuilt["partial"], "gaps": rebuilt["quality"]["input_gaps"],
                      "sarah_hourly": hourly, "simulate": sim.get("summary", sim.get("error")),
                      "response_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in after.items()}}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="export once and retain the live run's bound bytes")
    b.add_argument("--mode", required=True, choices=["solar", "solar-demo", "solar-smoke"])
    for flag in ("sis-dir", "sid-dir", "public-dir", "fingrid-dir", "era5-path"):
        b.add_argument("--" + flag, type=Path, required=flag in ("sis-dir", "sid-dir"))
    b.add_argument("--start")
    b.add_argument("--end")
    b.add_argument("--convention", choices=["start", "end"])
    r = sub.add_parser("replay", help="verify original and re-export from only the recorded files")
    r.add_argument("--site", type=Path, required=True)
    r.add_argument("--inputs", type=Path, required=True)
    r.add_argument("--reference-hourly", type=Path, help="comparison only: original sarah_hourly_era5grid.nc")
    for p in (b, r):
        p.add_argument("--data-dir", type=Path, required=True, help="new, nonexistent private directory")
    args = parser.parse_args(argv)
    (build if args.command == "build" else replay)(args)


if __name__ == "__main__":
    main()
