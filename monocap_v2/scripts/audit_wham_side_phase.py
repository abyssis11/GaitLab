#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monocap_v2.core.side_phase_audit import run_wham_side_phase_audit


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit WHAM 2D/3D and OpenSim FK left-right side phase.")
    ap.add_argument("--benchmark-dir", required=True, type=Path)
    ap.add_argument("--trial", default="walking2")
    ap.add_argument("--evaluation-hz", type=float, default=60.0)
    ap.add_argument("--convention-profile", default=None)
    ap.add_argument("--out", type=Path, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    report = run_wham_side_phase_audit(
        benchmark_dir=args.benchmark_dir,
        trial=args.trial,
        evaluation_hz=args.evaluation_hz,
        convention_profile=args.convention_profile,
        out_dir=args.out,
    )
    outputs = report.get("outputs") or {}
    print(f"[wham-side-phase] {report.get('status')} -> {outputs.get('json')}", flush=True)
    print(f"[wham-side-phase] csv -> {outputs.get('csv')}", flush=True)
    print(f"[wham-side-phase] plot -> {outputs.get('plot')}", flush=True)
    if outputs.get("projection_overlay"):
        print(f"[wham-side-phase] overlay -> {outputs.get('projection_overlay')}", flush=True)
    return 0 if report.get("status") in {"ok", "warning"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
