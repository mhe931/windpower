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
STAGE_CHOICES = (*STAGES, "smoke", "demo", "all")


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # argparse's default exit code 2 is reserved for "unavailable inputs"
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="python -m windpower", description="Finland wind-power baseline pipeline.")
    p.add_argument("stage", choices=STAGE_CHOICES, help="stage to run; 'smoke' and 'demo' are fully offline synthetic end-to-end runs")
    p.add_argument("--mode", choices=C.RUN_MODES, default="full", help="run mode for stages and 'all' (default: full; smoke/demo stages fix their own)")
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


def main(argv: "Sequence[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    stage = args.stage
    mode = {"smoke": "smoke", "demo": "demo"}.get(stage, args.mode)
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
