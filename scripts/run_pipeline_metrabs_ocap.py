#!/usr/bin/env python3
"""
run_pipeline.py

Runs the requested CLI steps sequentially, stopping on the first failure.

By default, it will auto-fix a common typo in your commands: if a --manifest/-m
path ends with ".yam" and that file does not exist, but a ".yaml" file at the
same path DOES exist, it will substitute ".yaml" automatically.

Usage:
    python run_pipeline.py
    python run_pipeline.py --no-auto-fix

Tip: Run this script from the project repo root so all relative paths resolve.
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from typing import List

def run_step(cmd: str, *, cwd: str = None) -> None:
    """Run a single command, print timing, and stop on failure."""
    tokens = shlex.split(cmd)
    printable = " ".join(tokens)
    print(f"\n=== Running: {printable}")
    start = time.time()
    proc = subprocess.run(tokens, cwd=cwd)
    dur = time.time() - start
    print(f"=== Finished (exit={proc.returncode}) in {dur:.1f}s")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

def main():
    MANIFEST = "./manifests/OpenCapDataset/subject3-2.yaml"
    CONFIG = "./config/paths.yaml"
    TRIAL = "walking1"
    SUBJECT_H = 1690

    # Commands as provided (kept verbatim on purpose).
    steps = [
        #f"python src/pose/metrabs_pose_estimation.py  -m {MANIFEST} -p {CONFIG} --trial {TRIAL} --video-field video_sync --calib-pickle",

        #f"python src/validation/validate_rtmw3d_vs_mocap_gpjatk.py -m {MANIFEST} -p {CONFIG} --trial {TRIAL} --root-center pelvis --summary-per-joint --procrustes similarity --resample-to mocap --out-json --metrabs-ocap --hnorm --subject-height-mm {SUBJECT_H}",

        f"python src/pose/metrabs_pose_estimation.py  -m {MANIFEST} -p {CONFIG} --trial {TRIAL} --video-field video_sync --calib-pickle --for-opensim",

        f"python src/marker_enhancer/marker_enhancer2.py     -m {MANIFEST}   -p {CONFIG}     --trial {TRIAL} --models-path     ./models/marker_enhancer/     --version 0.3     --metrabs --trc-type metrabs",

        f"python src/validation/validate_rtmw3d_vs_mocap_gpjatk.py  -m {MANIFEST} -p {CONFIG} --trial {TRIAL} --root-center pelvis --summary-per-joint --procrustes similarity --resample-to mocap --out-json --enhanc --metrabs --subject-height-mm {SUBJECT_H}"

    ]

    print("Pipeline starting...")
    for i, s in enumerate(steps, 1):
        print(f"\n--- Step {i}/{len(steps)} ---")
        run_step(s)
    print("\n✅ Pipeline finished successfully.")

if __name__ == "__main__":
    main()
