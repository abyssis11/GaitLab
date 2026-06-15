#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monocap_v2.core.backend_convention_audit import (
    DEFAULT_OUT_DIR,
    generate_axis_candidates,
    generate_physical_axis_candidates,
    parse_benchmark_dirs,
    run_backend_convention_audit,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a cached Cam/backend/trial Level A convention audit.")
    ap.add_argument(
        "--benchmark-dirs",
        required=True,
        help="Comma-separated NAME=PATH entries, e.g. Cam1=monocap_v2/benchmarks/subject7_walking_level_a,Cam0=...",
    )
    ap.add_argument("--trials", default="walking1,walking2,walking3")
    ap.add_argument("--backends", default="wham,metrabs,rtmw3d")
    ap.add_argument("--evaluation-hz", default="60,100")
    ap.add_argument("--time-offset-min", type=float, default=-0.30)
    ap.add_argument("--time-offset-max", type=float, default=0.30)
    ap.add_argument("--time-offset-step", type=float, default=0.02)
    ap.add_argument(
        "--axis-set",
        choices=["physical", "full"],
        default="full",
        help="Use all 48 signed permutations, or a smaller no-permutation physical sign set.",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    axis_candidates = generate_axis_candidates() if args.axis_set == "full" else generate_physical_axis_candidates()
    report = run_backend_convention_audit(
        benchmark_dirs=parse_benchmark_dirs(args.benchmark_dirs),
        trials=_csv_arg(args.trials),
        backends=_csv_arg(args.backends),
        evaluation_hz_values=[float(item) for item in _csv_arg(args.evaluation_hz)],
        out_dir=args.out,
        time_offset_min=args.time_offset_min,
        time_offset_max=args.time_offset_max,
        time_offset_step=args.time_offset_step,
        axis_candidates=axis_candidates,
    )
    outputs = report.get("outputs") or {}
    print(
        f"[backend-convention-audit] {report.get('status')} axis_set={args.axis_set} "
        f"candidates={report.get('candidate_count')}",
        flush=True,
    )
    print(f"[backend-convention-audit] rows -> {outputs.get('convention_audit_rows')}", flush=True)
    print(f"[backend-convention-audit] summary -> {outputs.get('stability_summary_md')}", flush=True)
    return 0 if report.get("status") in {"ok", "warning"} else 1


def _csv_arg(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
