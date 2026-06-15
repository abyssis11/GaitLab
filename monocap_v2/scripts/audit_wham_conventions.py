#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monocap_v2.core.wham_convention_audit import AUDIT_PROFILES, run_wham_convention_audit


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit WHAM Level A evaluation convention profiles.")
    ap.add_argument("--benchmark-dir", required=True, type=Path)
    ap.add_argument("--trials", default="walking1,walking2,walking3")
    ap.add_argument("--profiles", default=",".join(AUDIT_PROFILES))
    ap.add_argument("--evaluation-hz", type=float, default=None, help="Optional uniform evaluation rate, e.g. 100.")
    ap.add_argument("--out", type=Path, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    report = run_wham_convention_audit(
        benchmark_dir=args.benchmark_dir,
        trials=_csv_arg(args.trials),
        profiles=_csv_arg(args.profiles),
        out_dir=args.out,
        evaluation_hz=args.evaluation_hz,
    )
    outputs = report.get("outputs") or {}
    print(f"[wham-convention-audit] {report.get('status')} -> {outputs.get('summary_md')}", flush=True)
    print(f"[wham-convention-audit] csv -> {outputs.get('csv')}", flush=True)
    return 0 if report.get("status") in {"ok", "warning"} else 1


def _csv_arg(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
