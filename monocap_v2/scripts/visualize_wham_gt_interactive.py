#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monocap_v2.core.wham_gt_interactive import render_wham_gt_interactive


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create an interactive WHAM-vs-OpenSim-FK GT diagnostic viewer.")
    ap.add_argument("--benchmark-dir", required=True, type=Path)
    ap.add_argument("--trial", default="walking1")
    ap.add_argument("--out-html", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--speed", type=float, default=0.25)
    ap.add_argument("--evaluation-hz", type=float, default=None, help="Optional uniform evaluation rate, e.g. 100.")
    ap.add_argument("--matrix-audit", type=Path, default=None, help="Optional wham_convention_matrix_audit.json to load top candidates into the viewer.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.benchmark_dir / "visualizations"
    out_html = args.out_html or out_dir / f"wham_gt_{args.trial}_interactive.html"
    out_json = args.out_json or out_dir / f"wham_gt_{args.trial}_interactive_qc.json"
    report = render_wham_gt_interactive(
        benchmark_dir=args.benchmark_dir,
        trial=args.trial,
        out_html=out_html,
        out_json=out_json,
        default_speed=args.speed,
        evaluation_hz=args.evaluation_hz,
        matrix_audit_path=args.matrix_audit,
    )
    print(f"[wham-gt-interactive] {report['status']} -> {out_html}", flush=True)
    print(f"[wham-gt-interactive] qc -> {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
