#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run a small WHAM API smoke test and summarize returned artifacts.")
    parser.add_argument("--video", type=Path, required=True, help="Input video for the smoke run.")
    parser.add_argument("--wham-repo", type=Path, default=repo_root / "external" / "WHAM")
    parser.add_argument("--out", type=Path, default=repo_root / "monocap_v2" / "runs" / "wham_smoke")
    parser.add_argument("--start-frame", type=int, default=0, help="First input frame to include when trimming.")
    parser.add_argument("--max-frames", type=int, default=30, help="Trim to this many frames before running WHAM. Use 0 for full video.")
    parser.add_argument("--run-global", action="store_true", help="Attempt DPVO/global WHAM. Default is local-only.")
    parser.add_argument("--visualize", action="store_true", help="Ask WHAM to render its visualization.")
    parser.add_argument("--json", action="store_true", help="Print the JSON summary to stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "smoke_summary.json"

    try:
        video = args.video.resolve()
        if args.max_frames > 0:
            video = trim_video(
                video,
                (args.out / f"{video.stem}_{args.start_frame:06d}_{args.max_frames:04d}.mp4").resolve(),
                args.start_frame,
                args.max_frames,
            )
        summary = run_wham_smoke(args, video)
        summary["status"] = "ok"
        returncode = 0
    except Exception as exc:
        summary = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        returncode = 1

    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(f"WHAM smoke status: {summary['status']}")
        print(f"Summary: {summary_path}")
        if summary["status"] != "ok":
            print(summary["error"])
    return returncode


def trim_video(source: Path, target: Path, start_frame: int, max_frames: int) -> Path:
    import cv2

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create smoke video: {target}")
    frames = 0
    while frames < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frames += 1
    cap.release()
    writer.release()
    if frames == 0:
        raise RuntimeError(f"No frames were read from video: {source} starting at frame {start_frame}")
    return target.resolve()


def run_wham_smoke(args: argparse.Namespace, video: Path) -> dict[str, Any]:
    wham_repo = args.wham_repo.resolve()
    if not (wham_repo / "wham_api.py").exists():
        raise RuntimeError(f"WHAM API not found under {wham_repo}")

    old_cwd = Path.cwd()
    sys.path.insert(0, str(wham_repo))
    os.chdir(wham_repo)
    try:
        from wham_api import WHAM_API
        from lib.data.datasets import CustomDataset

        # WHAM_API iterates CustomDataset directly, while demo.py calls load_data(),
        # which initializes this attribute. Keep the shim local to our smoke harness.
        if not hasattr(CustomDataset, "prefix"):
            CustomDataset.prefix = ""

        model = WHAM_API()
        results, tracking_results, slam_results = model(
            str(video),
            output_dir=str(args.out.resolve()),
            calib=None,
            run_global=bool(args.run_global),
            visualize=bool(args.visualize),
        )
    finally:
        os.chdir(old_cwd)

    return {
        "video": str(video),
        "output_dir": str(args.out.resolve()),
        "run_global_requested": bool(args.run_global),
        "subjects": summarize_subject_results(results),
        "tracking": summarize_mapping(tracking_results),
        "slam": summarize_value(slam_results),
    }


def summarize_subject_results(results: Any) -> dict[str, Any]:
    output = {}
    for subject_id, values in dict(results).items():
        output[str(subject_id)] = summarize_mapping(values)
    return output


def summarize_mapping(values: Any) -> dict[str, Any]:
    if not isinstance(values, dict):
        return {"value": summarize_value(values)}
    return {str(key): summarize_value(value) for key, value in values.items()}


def summarize_value(value: Any) -> dict[str, Any]:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return {"type": type(value).__name__, "shape": list(shape), "dtype": str(dtype)}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "length": len(value)}
    if isinstance(value, dict):
        return {"type": "dict", "keys": sorted(str(key) for key in value.keys())}
    return {"type": type(value).__name__, "value": value if isinstance(value, (str, int, float, bool, type(None))) else str(value)}


if __name__ == "__main__":
    raise SystemExit(main())
