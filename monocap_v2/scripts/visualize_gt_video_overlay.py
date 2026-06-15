#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monocap_v2.core.gt_video_overlay import render_gt_labeled_video_overlay
from monocap_v2.core.logging_utils import read_yaml


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Project mocap-derived OpenSim FK joint centers onto walking video with labels.")
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--reference-npz", type=Path, default=None)
    ap.add_argument("--out-mp4", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--video-space", choices=["raw", "sync"], default="raw")
    ap.add_argument("--preview-fps", type=float, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run
    cfg = read_yaml(run_dir / "run_config.yaml")
    trial = str(cfg.get("trial_id") or "trial")
    repo_root = Path(str(cfg.get("repo_root") or Path.cwd()))
    reference_npz = args.reference_npz or repo_root / "monocap_v2" / "benchmarks" / "subject7_walking_level_a" / "reference" / f"opensim_fk_{trial}.npz"
    out_mp4 = args.out_mp4 or run_dir / "reports" / "gt_labeled_overlay.mp4"
    out_json = args.out_json or run_dir / "reports" / "gt_labeled_overlay_qc.json"
    report = render_gt_labeled_video_overlay(
        run_dir=run_dir,
        reference_npz=reference_npz,
        out_mp4=out_mp4,
        out_json=out_json,
        video_space=args.video_space,
        preview_fps=args.preview_fps,
    )
    print(f"[gt-video-overlay] {report['status']} -> {out_mp4}", flush=True)
    print(f"[gt-video-overlay] qc -> {out_json}", flush=True)
    return 0 if report.get("status") in {"ok", "warning"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
