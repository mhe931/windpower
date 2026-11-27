"""Command line: `python -m windpower <stage> [options]`.

Exit codes: 0 ok | 1 hard validation/contract failure | 2 required inputs unavailable (credentials,
cache, --offline) | 3 unexpected error | 64 usage error.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Any, Callable, Sequence

from . import config as C
from .era5 import ValidationError
from .era5_solar import ConventionAmbiguousError, SolarEra5Error
from .fingrid import DatasetMismatchError, FingridError
from .pipeline import STAGES, Run, Unavailable
from .sarah import SarahError
from .solar_national import AllocationError
from .solar_pipeline import SOLAR_RUNS, SOLAR_STAGES, SolarRun
from .solar_public import MunicipalityMatchError, PublicDataError
from .solar_pv import PvError
from .validate import CheckResult, failed_checks

EXIT_OK, EXIT_VALIDATION, EXIT_UNAVAILABLE, EXIT_ERROR, EXIT_USAGE = 0, 1, 2, 3, 64
STAGE_CHOICES = (*STAGES, *SOLAR_STAGES, *SOLAR_RUNS, "smoke", "demo", "all", "serve")
GENERATE_COMMAND = {"full": "python -m windpower all --mode full   (needs ERA5 + Fingrid credentials; see docs/reproduce.md)",
                    "demo": "python -m windpower demo", "smoke": "python -m windpower smoke",
                    "solar-demo": "python -m windpower solar-demo", "solar-smoke": "python -m windpower solar-smoke",
                    "solar": "python -m windpower solar-all   (real inputs; partial without ERA5 / Fingrid, see docs/solar.md)"}
WIND_AUTO, SOLAR_AUTO = ("full", "demo", "smoke"), ("solar", "solar-demo", "solar-smoke")


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # argparse's default exit code 2 is reserved for "unavailable inputs"
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="python -m windpower", description="Finland wind-power baseline pipeline.")
    p.add_argument("stage", choices=STAGE_CHOICES, help="stage to run; 'smoke' and 'demo' are fully offline synthetic end-to-end runs")
    p.add_argument("--mode", choices=C.ALL_RUN_MODES, default=None,
                   help="run mode for stages and 'all' (default: full; smoke/demo stages fix their own); for 'serve' the default is the first "
                        "existing site artifact of full, demo, smoke; solar stages use solar (real), solar-smoke or solar-demo (SYNTHETIC; default: solar)")
    p.add_argument("--site-dir", default=None, help="serve: directory of the WIND site artifact (…/site/v1); overrides --mode/--data-dir for wind")
    p.add_argument("--solar-site-dir", default=None, help="serve: directory of the SOLAR site artifact (…/site/solar/v1)")
    p.add_argument("--solar-mode", choices=C.SOLAR_RUN_MODES, default=None,
                   help="serve: solar artifact of this solar mode (default: --mode if it is a solar mode, else the first existing of solar, solar-demo, solar-smoke)")
    p.add_argument("--host", default="127.0.0.1", help="serve: bind address (default 127.0.0.1, local only)")
    p.add_argument("--port", type=int, default=8000, help="serve: TCP port (default 8000)")
    p.add_argument("--data-dir", default=None, help="root for cache/interim/outputs (default: <repo>/data or $WINDPOWER_DATA_DIR)")
    p.add_argument("--offline", action="store_true", help="never use the network (always on for smoke/demo)")
    p.add_argument("--refresh", action="store_true", help="re-download raw inputs (needs network and credentials where applicable)")
    p.add_argument("--start", default=None, help="first UTC hour, full mode only (default 2025-07-01T00:00:00)")
    p.add_argument("--end", default=None, help="last UTC hour, full mode only (default 2025-09-30T23:00:00)")
    p.add_argument("--quiet", action="store_true", help="only print the final summary")
    solar = p.add_argument_group("solar options (stages ingest-sarah ... export-site-solar, solar-smoke, solar-demo, solar-all)")
    solar.add_argument("--sarah-sis-dir", default=None, help=f"folder of CM SAF SIS daily files (env {C.SARAH_SIS_ENV}; default <data>/interim/smoke/ORD73128 if it exists)")
    solar.add_argument("--sarah-sid-dir", default=None, help=f"folder of CM SAF SID daily files (env {C.SARAH_SID_ENV}; default <data>/interim/smoke/ORD73129 if it exists)")
    solar.add_argument("--era5-solar-path", default=None, help=f"ERA5 solar file, directory or ZIP (env {C.ERA5_SOLAR_ENV})")
    solar.add_argument("--era5-convention", choices=("end", "start"), default=None,
                       help="force the ERA5 time-stamp convention (end = accumulations stamped at the end of the hour); default: detect from the data")
    solar.add_argument("--window-start", default=None, help=f"first UTC day or hour (default {C.SOLAR_WINDOW_START})")
    solar.add_argument("--window-end", default=None, help=f"last UTC day (inclusive) or hour (default {C.SOLAR_WINDOW_END})")
    return p


def _print_unavailable(items: Sequence[dict[str, str]], mode: str) -> None:
    print(f"INPUTS UNAVAILABLE ({mode} mode) - nothing was run and no synthetic data was substituted:", file=sys.stderr)
    for i in items:
        print(f"  - {i['input']}: {i['reason']}", file=sys.stderr)


def summarise(results: Sequence[CheckResult], label: str, synthetic: bool) -> int:
    failed = failed_checks(results)
    p = sum(r.status == "pass" for r in results)
    s = sum(r.status == "skipped" for r in results)
    tag = " (SYNTHETIC data)" if synthetic else ""
    print(f"CHECKS: {p} pass, {len(failed)} fail, {s} skipped{tag}")
    for r in failed:
        print(f"  FAIL {r.name}: {r.detail}")
    for r in results:
        if r.status == "skipped":
            print(f"  skipped {r.name}: {r.detail}")
    print(f"{label} {'FAIL' if failed else 'PASS'}")
    return EXIT_VALIDATION if failed else EXIT_OK


def _first_artifact(candidates: "Sequence[tuple[str, Any]]") -> "tuple[str, Any] | None":
    return next(((m, d) for m, d in candidates if (d / "manifest.json").is_file()), None)


def serve(args: argparse.Namespace) -> int:
    """Serve the website from existing site artifacts (wind `site/v1` and/or solar `site/solar/v1`); never generates or synthesises data.

    Wind: --site-dir, else --mode (wind mode), else the first of full, demo, smoke. Solar: --solar-site-dir, else --solar-mode, else
    --mode if it is a solar mode, else the first existing of solar, solar-demo, solar-smoke. `--mode` naming one source only selects that
    source's artifact explicitly; the other is then picked up automatically if one exists, otherwise its views say "not built".
    """
    from pathlib import Path

    from .web import server
    from .web.data import SiteDataError

    if not (0 <= args.port <= 65535):
        print("--port must be between 0 and 65535", file=sys.stderr)
        return EXIT_USAGE
    wind_explicit = bool(args.site_dir) or args.mode in C.RUN_MODES
    solar_mode = args.solar_mode or (args.mode if args.mode in C.SOLAR_RUN_MODES else None)
    solar_explicit = bool(args.solar_site_dir) or solar_mode is not None
    if args.site_dir:
        wind_c = [("custom", Path(args.site_dir))]
    elif args.mode in C.SOLAR_RUN_MODES:
        wind_c = [(m, C.output_dir(m, args.data_dir) / "site" / "v1") for m in WIND_AUTO]
    else:
        wind_c = [(m, C.output_dir(m, args.data_dir) / "site" / "v1") for m in ([args.mode] if args.mode else WIND_AUTO)]
    if args.solar_site_dir:
        solar_c = [("custom", Path(args.solar_site_dir))]
    elif args.site_dir and not solar_explicit:
        solar_c = []                                # a custom wind directory means: serve exactly that (no automatic solar discovery)
    else:
        solar_c = [(m, C.output_dir(m, args.data_dir) / "site" / "solar" / "v1") for m in ([solar_mode] if solar_mode else SOLAR_AUTO)]
    if args.solar_site_dir and not wind_explicit and not args.site_dir:
        wind_c = []
    wind, solar = _first_artifact(wind_c), _first_artifact(solar_c)
    problems = []
    if wind is None and wind_explicit:
        problems.append(("wind", wind_c))
    if solar is None and solar_explicit:
        problems.append(("solar", solar_c))
    if not problems and wind is None and solar is None:
        problems = [("wind", wind_c), ("solar", solar_c)]
    if problems:
        print("SITE ARTIFACT NOT FOUND - nothing is served and no data is synthesised. Looked in:", file=sys.stderr)
        for kind, cands in problems:
            for _, d in cands:
                print(f"  {d}", file=sys.stderr)
        print("Generate it with:", file=sys.stderr)
        for kind, cands in problems:
            for m, _ in cands:
                if m in GENERATE_COMMAND:
                    print(f"  {GENERATE_COMMAND[m]}", file=sys.stderr)
        if not wind_explicit and not solar_explicit:
            print("(for a quick local look use the demo and solar-demo commands; they create clearly labelled SYNTHETIC data)", file=sys.stderr)
        return EXIT_UNAVAILABLE
    try:
        kw = {"solar_dir": solar[1]} if solar else {}
        return server.run(wind[1] if wind else None, args.host, args.port, **kw)
    except SiteDataError as exc:
        which = "solar" if "solar" in str(exc).lower() or (wind is None and solar is not None) else "wind"
        mode = (solar if which == "solar" else wind)[0]  # type: ignore[index]
        print(f"SITE ARTIFACT INVALID: {exc}\nRegenerate it with: {GENERATE_COMMAND.get(mode, 'python -m windpower demo')}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except OSError as exc:
        print(f"cannot listen on {args.host}:{args.port}: {exc}", file=sys.stderr)
        return EXIT_ERROR


def run_solar(args: argparse.Namespace) -> int:
    """Solar stages and runs. Exit 2: a stage's own required inputs are unavailable (nothing synthetic is substituted); 1: hard check or
    input-contract failure; 0: done, including a PARTIAL real run whose gaps are recorded in manifest.json."""
    stage = args.stage
    mode = {"solar-smoke": "solar-smoke", "solar-demo": "solar-demo"}.get(stage, args.mode or "solar")
    if mode not in C.SOLAR_RUN_MODES or (stage in ("solar-smoke", "solar-demo") and args.mode not in (None, mode)):
        print(f"stage {stage} needs a solar mode ({', '.join(C.SOLAR_RUN_MODES)}), got --mode {args.mode}", file=sys.stderr)
        return EXIT_USAGE
    say: Callable[[str], None] = (lambda m: None) if args.quiet else print
    try:
        run = SolarRun(mode, args.data_dir, sarah_sis_dir=args.sarah_sis_dir, sarah_sid_dir=args.sarah_sid_dir,
                       era5_solar_path=args.era5_solar_path or os.environ.get(C.ERA5_SOLAR_ENV), era5_convention=args.era5_convention,
                       window_start=args.window_start, window_end=args.window_end, offline=args.offline, refresh=args.refresh, say=say)
    except ValueError as exc:
        print(f"invalid options: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        if run.synthetic:
            print(f"== windpower {stage}: offline synthetic run - every number below is SYNTHETIC ==")
        if stage in SOLAR_RUNS:
            results, status = run.run_all()
            label = {"solar-smoke": "SOLAR-SMOKE", "solar-demo": "SOLAR-DEMO", "solar-all": f"SOLAR-ALL[{mode}]"}[stage]
            if status != "complete":
                print(f"STATUS {status}: " + "; ".join(f"{k}: {v}" for k, v in run.gaps.items()))
            code = summarise(results, label, run.synthetic)
            if code == EXIT_OK:
                print(f"outputs: {run.out}")
            return code
        if stage == "ingest-sarah":
            summary = run.ingest_sarah()
            print(f"INGEST-SARAH[{mode}] OK: {summary['hours']} hours, {summary['fully_covered_cells']} of {summary['grid']['n_cells']} cells fully covered"
                  f"{' (SYNTHETIC data)' if run.synthetic else ''}")
            return EXIT_OK
        results = run.run_stage(stage)
        if results is not None:
            return summarise(results, f"SOLAR-VALIDATE[{mode}]", run.synthetic)
        if run.gaps:
            print(f"{stage.upper()}[{mode}] OK (partial): " + "; ".join(f"{k} unavailable: {v}" for k, v in run.gaps.items()))
        else:
            print(f"{stage.upper()}[{mode}] OK{' (SYNTHETIC data)' if run.synthetic else ''}")
        return EXIT_OK
    except Unavailable as exc:
        _print_unavailable(exc.items, mode)
        try:
            run.write_unavailable(exc.items)
        except OSError:
            pass
        return EXIT_UNAVAILABLE
    except (SarahError, SolarEra5Error, MunicipalityMatchError, AllocationError, PvError, DatasetMismatchError) as exc:
        kind = "AMBIGUOUS CONVENTION" if isinstance(exc, ConventionAmbiguousError) else "INPUT CONTRACT FAILURE"
        print(f"{kind}: {exc}", file=sys.stderr)
        return EXIT_VALIDATION
    except (PublicDataError, FingridError) as exc:
        print(f"DATA ACCESS ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception:  # noqa: BLE001 - last resort: distinct exit code, full traceback for the log
        traceback.print_exc()
        return EXIT_ERROR


def main(argv: "Sequence[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    stage = args.stage
    if stage == "serve":
        return serve(args)
    if stage in SOLAR_STAGES or stage in SOLAR_RUNS:
        return run_solar(args)
    if args.mode in C.SOLAR_RUN_MODES:
        print(f"--mode {args.mode} is a solar mode; wind stages use {', '.join(C.RUN_MODES)}", file=sys.stderr)
        return EXIT_USAGE
    mode = {"smoke": "smoke", "demo": "demo"}.get(stage, args.mode or "full")
    say: Callable[[str], None] = (lambda m: None) if args.quiet else print
    run = Run(mode, args.data_dir, offline=args.offline, refresh=args.refresh, start=args.start, end=args.end, say=say)
    try:
        if stage in ("smoke", "demo", "all"):
            if stage != "all":
                print(f"== windpower {stage}: offline synthetic run - every number below is SYNTHETIC ==")
            results = run.run_all()
            label = {"smoke": "SMOKE", "demo": "DEMO", "all": f"ALL[{mode}]"}[stage]
            code = summarise(results, label, run.synthetic)
            if code == EXIT_OK and run.synthetic:
                print(f"outputs: {run.out}")
            return code
        results = run.run_stage(stage)
        if results is not None:
            return summarise(results, f"VALIDATE[{mode}]", run.synthetic)
        return EXIT_OK
    except Unavailable as exc:
        _print_unavailable(exc.items, mode)
        try:
            run.write_unavailable(exc.items)
        except OSError:
            pass
        return EXIT_UNAVAILABLE
    except (ValidationError, DatasetMismatchError) as exc:
        print(f"VALIDATION FAILURE: {exc}", file=sys.stderr)
        return EXIT_VALIDATION
    except FingridError as exc:
        print(f"FINGRID ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception:  # noqa: BLE001 - last resort: distinct exit code, full traceback for the log
        traceback.print_exc()
        return EXIT_ERROR
