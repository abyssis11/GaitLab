#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monocap_v2.core.wham_convention_matrix import run_wham_convention_matrix_audit


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a configurable WHAM convention matrix audit.")
    ap.add_argument("--benchmark-dir", required=True, type=Path)
    ap.add_argument("--trials", default="walking1,walking2,walking3")
    ap.add_argument("--profile-set", choices=["physical", "full"], default="physical")
    ap.add_argument("--evaluation-hz", default="native,100", help="Comma-separated values such as native,100.")
    ap.add_argument("--time-offset-min", type=float, default=-0.30)
    ap.add_argument("--time-offset-max", type=float, default=0.30)
    ap.add_argument("--time-offset-step", type=float, default=0.02)
    ap.add_argument("--out", type=Path, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    report = run_wham_convention_matrix_audit(
        benchmark_dir=args.benchmark_dir,
        trials=_csv_arg(args.trials),
        profile_set=args.profile_set,
        evaluation_hz_values=_evaluation_hz_arg(args.evaluation_hz),
        time_offset_min=args.time_offset_min,
        time_offset_max=args.time_offset_max,
        time_offset_step=args.time_offset_step,
        out_dir=args.out,
    )
    outputs = report.get("outputs") or {}
    print(f"[wham-convention-matrix] {report.get('status')} -> {outputs.get('summary_md')}", flush=True)
    print(f"[wham-convention-matrix] csv -> {outputs.get('csv')}", flush=True)
    return 0 if report.get("status") in {"ok", "warning"} else 1


def _csv_arg(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _evaluation_hz_arg(value: str) -> list[float | None]:
    out: list[float | None] = []
    for item in _csv_arg(value):
        if item.lower() in {"native", "none", "null"}:
            out.append(None)
        else:
            out.append(float(item))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
