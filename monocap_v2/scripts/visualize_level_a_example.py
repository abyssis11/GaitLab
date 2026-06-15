#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monocap_v2.core.level_a_visualization import render_level_a_overlay
from monocap_v2.core.logging_utils import write_json


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize Level A backend predictions and OpenSim FK GT on one trial.")
    ap.add_argument("--benchmark-dir", required=True, type=Path)
    ap.add_argument("--trial", default="walking1")
    ap.add_argument("--backends", default="metrabs,rtmw3d,wham")
    ap.add_argument("--out-mp4", type=Path, default=None)
    ap.add_argument("--out-png", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=2100)
    ap.add_argument("--height", type=int, default=820)
    ap.add_argument("--evaluation-hz", type=float, default=None)
    ap.add_argument("--wham-convention-profile", default=None)
    ap.add_argument("--swap-reference-lr", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.benchmark_dir / "visualizations"
    out_mp4 = args.out_mp4 or out_dir / f"level_a_{args.trial}_all_backends_vs_gt.mp4"
    out_png = args.out_png or out_dir / f"level_a_{args.trial}_all_backends_vs_gt_frame.png"
    out_json = args.out_json or out_dir / f"level_a_{args.trial}_all_backends_vs_gt_qc.json"
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    report = render_level_a_overlay(
        benchmark_dir=args.benchmark_dir,
        trial=args.trial,
        backends=backends,
        out_mp4=out_mp4,
        out_png=out_png,
        out_json=out_json,
        preview_fps=args.fps,
        width=args.width,
        height=args.height,
        evaluation_hz=args.evaluation_hz,
        wham_convention_profile=args.wham_convention_profile,
        swap_reference_lr=args.swap_reference_lr,
    )
    write_json(out_json, report)
    print(f"[level-a-vis] {report['status']} -> {out_mp4}", flush=True)
    print(f"[level-a-vis] frame -> {out_png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
