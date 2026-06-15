#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monocap_v2.core.side_label_audit import (
    DEFAULT_BENCHMARK_DIRS,
    DEFAULT_OUT_DIR,
    DEFAULT_SUMMARY_CSV,
    run_side_label_audit,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a direction-aware Level A side-label audit.")
    ap.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    ap.add_argument(
        "--benchmark-dirs",
        default=DEFAULT_BENCHMARK_DIRS,
        help="Comma-separated NAME=PATH entries, e.g. Cam1=...,Cam0=...",
    )
    ap.add_argument("--camera", default=None)
    ap.add_argument("--trial", default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--evaluation-hz", type=float, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true", help="Skip per-case phase PNGs.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    report = run_side_label_audit(
        summary_csv=args.summary_csv,
        benchmark_dirs=args.benchmark_dirs,
        out_dir=args.out,
        camera=args.camera,
        trial=args.trial,
        backend=args.backend,
        evaluation_hz=args.evaluation_hz,
        write_plots=not args.no_plots,
    )
    outputs = report.get("outputs") or {}
    print(f"[side-label-audit] {report.get('status')} cases={report.get('case_count')}", flush=True)
    print(f"[side-label-audit] rows -> {outputs.get('rows_csv')}", flush=True)
    print(f"[side-label-audit] summary -> {outputs.get('summary_md')}", flush=True)
    return 0 if report.get("status") in {"ok", "warning"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
