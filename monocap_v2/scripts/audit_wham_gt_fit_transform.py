#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monocap_v2.core.gt_fit_transform import run_wham_gt_fit_transform_audit


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fit diagnostic WHAM-to-OpenSim-FK transforms for one WHAM trial.")
    ap.add_argument("--benchmark-dir", required=True, type=Path)
    ap.add_argument("--trial", default="walking2")
    ap.add_argument("--evaluation-hz", type=float, default=60.0)
    ap.add_argument("--convention-profile", default=None)
    ap.add_argument("--time-offset-s", type=float, default=None, help="Diagnostic-only WHAM time offset in seconds.")
    ap.add_argument("--fit-direction", choices=["wham_to_gt", "gt_to_wham"], default="wham_to_gt")
    ap.add_argument("--out", type=Path, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    report = run_wham_gt_fit_transform_audit(
        benchmark_dir=args.benchmark_dir,
        trial=args.trial,
        evaluation_hz=args.evaluation_hz,
        convention_profile=args.convention_profile,
        time_offset_s=args.time_offset_s,
        fit_direction=args.fit_direction,
        out_dir=args.out,
    )
    outputs = report.get("outputs") or {}
    print(f"[wham-gt-fit] {report.get('status')} -> {outputs.get('json')}", flush=True)
    print(f"[wham-gt-fit] csv -> {outputs.get('csv')}", flush=True)
    print(f"[wham-gt-fit] plot -> {outputs.get('plot')}", flush=True)
    print(f"[wham-gt-fit] video -> {outputs.get('video')}", flush=True)
    return 0 if report.get("status") in {"ok", "warning"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
