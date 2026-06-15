#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monocap_v2.core.logging_utils import write_json
from monocap_v2.core.opensim_fk_visualization import load_fk_reference, render_opensim_fk_preview


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize cached OpenSim FK joint-center reference.")
    ap.add_argument("--reference-npz", required=True, type=Path)
    ap.add_argument("--out-mp4", type=Path, default=None)
    ap.add_argument("--out-png", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    stem = args.reference_npz.with_suffix("")
    out_mp4 = args.out_mp4 or stem.with_name(f"{stem.name}_preview.mp4")
    out_png = args.out_png or stem.with_name(f"{stem.name}_representative_frame.png")
    out_json = args.out_json or stem.with_name(f"{stem.name}_visualization_qc.json")
    report = render_opensim_fk_preview(
        load_fk_reference(args.reference_npz),
        out_mp4=out_mp4,
        out_png=out_png,
        fps=args.fps,
        width=args.width,
        height=args.height,
        max_frames=args.max_frames,
    )
    write_json(out_json, report)
    print(f"[opensim-fk-vis] {report['status']} -> {out_mp4}", flush=True)
    print(f"[opensim-fk-vis] frame -> {out_png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
