#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monocap_v2.core.level_a_audit import run_level_a_audit


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit cached Level A backend artifacts for video, convention, label, and timing issues.")
    ap.add_argument("--benchmark-dir", required=True, type=Path)
    ap.add_argument("--trials", default=None, help="Optional comma-separated trial IDs.")
    ap.add_argument("--backends", default=None, help="Optional comma-separated backend IDs.")
    ap.add_argument("--time-offsets", default=None, help="Optional comma-separated diagnostic offsets in seconds.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    report = run_level_a_audit(
        args.benchmark_dir,
        trials=_csv(args.trials),
        backends=_csv(args.backends),
        time_offsets=[float(item) for item in _csv(args.time_offsets)] if args.time_offsets else None,
    )
    print(f"[level-a-audit] {report['status']} -> {report['outputs']['summary_md']}", flush=True)
    print(f"[level-a-audit] json -> {report['outputs']['json']}", flush=True)
    return 0 if report.get("status") != "failed" else 1


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
