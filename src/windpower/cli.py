"""Command line: `python -m windpower <stage> [options]`.

Exit codes: 0 ok | 1 hard validation/contract failure | 2 required inputs unavailable (credentials,
cache, --offline) | 3 unexpected error | 64 usage error.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Callable, Sequence

from . import config as C
from .era5 import ValidationError
from .fingrid import DatasetMismatchError, FingridError
from .pipeline import STAGES, Run, Unavailable
from .validate import CheckResult, failed_checks

EXIT_OK, EXIT_VALIDATION, EXIT_UNAVAILABLE, EXIT_ERROR, EXIT_USAGE = 0, 1, 2, 3, 64
STAGE_CHOICES = (*STAGES, "smoke", "demo", "all", "serve")
GENERATE_COMMAND = {"full": "python -m windpower all --mode full   (needs ERA5 + Fingrid credentials; see docs/reproduce.md)",
                    "demo": "python -m windpower demo", "smoke": "python -m windpower smoke"}


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # argparse's default exit code 2 is reserved for "unavailable inputs"
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="python -m windpower", description="Finland wind-power baseline pipeline.")
    p.add_argument("stage", choices=STAGE_CHOICES, help="stage to run; 'smoke' and 'demo' are fully offline synthetic end-to-end runs")
    p.add_argument("--mode", choices=C.RUN_MODES, default=None,
                   help="run mode for stages and 'all' (default: full; smoke/demo stages fix their own); for 'serve' the default is the first "
                        "existing site artifact of full, demo, smoke")
    p.add_argument("--site-dir", default=None, help="serve: directory of the site artifact (…/site/v1); overrides --mode/--data-dir")
    p.add_argument("--host", default="127.0.0.1", help="serve: bind address (default 127.0.0.1, local only)")
    p.add_argument("--port", type=int, default=8000, help="serve: TCP port (default 8000)")
    p.add_argument("--data-dir", default=None, help="root for cache/interim/outputs (default: <repo>/data or $WINDPOWER_DATA_DIR)")
    p.add_argument("--offline", action="store_true", help="never use the network (always on for smoke/demo)")
    p.add_argument("--refresh", action="store_true", help="re-download raw inputs (needs network and credentials where applicable)")
    p.add_argument("--start", default=None, help="first UTC hour, full mode only (default 2025-07-01T00:00:00)")
    p.add_argument("--end", default=None, help="last UTC hour, full mode only (default 2025-09-30T23:00:00)")
    p.add_argument("--quiet", action="store_true", help="only print the final summary")
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


def serve(args: argparse.Namespace) -> int:
    """Serve the website from an existing site artifact; never generates or synthesises data."""
    from pathlib import Path

    from .web import server
    from .web.data import SiteDataError

    if not (0 <= args.port <= 65535):
        print("--port must be between 0 and 65535", file=sys.stderr)
        return EXIT_USAGE
    if args.site_dir:
        candidates = [("custom", Path(args.site_dir))]
    else:
        modes = [args.mode] if args.mode else ["full", "demo", "smoke"]
        candidates = [(m, C.output_dir(m, args.data_dir) / "site" / "v1") for m in modes]
    found = next(((m, d) for m, d in candidates if (d / "manifest.json").is_file()), None)
    if found is None:
        print("SITE ARTIFACT NOT FOUND - nothing is served and no data is synthesised. Looked in:", file=sys.stderr)
        for _, d in candidates:
            print(f"  {d}", file=sys.stderr)
        print("Generate it with:", file=sys.stderr)
        for m, _ in candidates:
            if m in GENERATE_COMMAND:
                print(f"  {GENERATE_COMMAND[m]}", file=sys.stderr)
        if not args.mode and not args.site_dir:
            print("(for a quick local look use the demo command; it creates clearly labelled SYNTHETIC data)", file=sys.stderr)
        return EXIT_UNAVAILABLE
    try:
        return server.run(found[1], args.host, args.port)
    except SiteDataError as exc:
        print(f"SITE ARTIFACT INVALID: {exc}\nRegenerate it with: {GENERATE_COMMAND.get(found[0], 'python -m windpower demo')}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except OSError as exc:
        print(f"cannot listen on {args.host}:{args.port}: {exc}", file=sys.stderr)
        return EXIT_ERROR


def main(argv: "Sequence[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    stage = args.stage
    if stage == "serve":
        return serve(args)
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
