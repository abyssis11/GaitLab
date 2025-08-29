#!/usr/bin/env python3
# rtmw3d_scale_from_height.py
"""
Scale RTMW3D predictions to METRIC (mm) using subject height from meta/manifest.

What it does
------------
- Loads manifest via your IO.load_manifest.load_manifest.
- Finds rtmw3d/preds.jsonl and trial-level meta.json.
- Computes model-space "height" per frame = max distance(head_joints, foot_joints).
- Uses a robust percentile (default 95th) across frames to estimate standing height.
- Global scale s = height_mm / height_units.
- Writes preds_metric.jsonl with 'keypoints_xyz_mm' per person, plus 'scale_mm_per_unit' field.

Defaults expect the same output structure as your rtmw3d.py:
  <outputs_root or manifest.output_dir>/<subject>/<session>/<camera>/<trial>/rtmw3d/preds.jsonl
  and meta at:
  <...>/<trial>/meta.json

Usage
-----
python rtmw3d_scale_from_height.py \
  -m manifests/OpenCapDataset/subject2.yaml \
  -p config/paths.yaml \
  --trial walking1

Optional overrides:
  --preds /path/to/preds.jsonl
  --meta  /path/to/meta.json
  --out   /path/to/preds_metric.jsonl
  --height-mm 1960
  --head-joints nose
  --foot-joints left_heel,right_heel,left_big_toe,right_big_toe,left_ankle,right_ankle
  --person-index -1        # auto: highest mean_score
  --min-mean-score 0.0
  --height-percentile 95.0
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np

# Use YOUR manifest loader
from IO.load_manifest import load_manifest


# ----------------------------
# Path helpers (mirror rtmw3d.py)
# ----------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def decide_output_dir(manifest: dict, trial: dict, outputs_root_cli: str | None):
    base = manifest.get('output_dir')
    if not base:
        outputs_root = outputs_root_cli or Path.cwd() / "outputs"
        subj = manifest.get('subject_id', 'subject')
        sess = manifest.get('session', 'Session')
        cam  = manifest.get('camera', 'Cam')
        base = Path(outputs_root) / subj / sess / cam
    return Path(base) / trial['id']  # NOTE: trial root (meta.json lives here); rtmw3d is a subdir

def find_trial(manifest: dict, trial_id: str):
    for subset, trials in manifest.get('trials', {}).items():
        for t in trials:
            if t.get('id') == trial_id:
                return subset, t
    return None, None


# ----------------------------
# Core height->scale calculation
# ----------------------------

def _idxs_from_names(names, requested):
    name2idx = {n: i for i, n in enumerate(names or [])}
    idxs = [name2idx[n] for n in requested if n in name2idx]
    missing = [n for n in requested if n not in name2idx]
    return idxs, missing

def _frame_max_head_foot_distance(kps, head_idxs, foot_idxs):
    """kps: (K,3) model units. Return max ||head - foot|| over sets."""
    heads = kps[head_idxs]
    feet  = kps[foot_idxs]
    diffs = heads[:, None, :] - feet[None, :, :]    # (H,F,3)
    d = np.linalg.norm(diffs, axis=-1)              # (H,F)
    return float(np.nanmax(d)) if d.size else np.nan

def compute_global_scale_from_preds(
    preds_path: Path,
    kp_names: list[str],
    height_mm: float,
    head_names: list[str],
    foot_names: list[str],
    person_index: int = -1,
    min_mean_score: float = 0.0,
    height_percentile: float = 95.0,
):
    head_idxs, miss_h = _idxs_from_names(kp_names, head_names)
    foot_idxs, miss_f = _idxs_from_names(kp_names, foot_names)
    if not head_idxs or not foot_idxs:
        raise SystemExit(
            f"Requested joints missing in keypoint_names.\n"
            f"  Head missing: {miss_h}\n"
            f"  Foot missing: {miss_f}\n"
            f"Available: {kp_names[:10]}... (K={len(kp_names)})"
        )

    dists = []
    with open(preds_path, 'r', encoding='utf-8') as f:
        for ln in f:
            o = json.loads(ln)
            persons = o.get('persons') or []
            if not persons:
                continue

            # choose person
            if person_index == -1:
                ms = [p.get('mean_score', 0.0) for p in persons]
                pi = int(np.argmax(ms)) if ms else 0
            else:
                pi = min(max(0, person_index), len(persons) - 1)

            p = persons[pi]
            if p.get('mean_score', 1.0) < min_mean_score:
                continue

            kps = np.asarray(p.get('keypoints_xyz'), dtype=float)  # (K,3)
            if kps.ndim != 2 or kps.shape[1] != 3:
                continue

            d = _frame_max_head_foot_distance(kps, head_idxs, foot_idxs)
            if np.isfinite(d):
                dists.append(d)

    if not dists:
        raise SystemExit("No valid frames found in preds.jsonl to compute scale.")

    dists = np.asarray(dists, dtype=float)
    height_units = np.percentile(dists, float(height_percentile))
    if not np.isfinite(height_units) or height_units <= 1e-8:
        raise SystemExit("Computed model-space height is invalid (<=0). Check joints/data.")

    scale = float(height_mm) / float(height_units)
    return scale, height_units, {
        "percentile": float(height_percentile),
        "median_units": float(np.nanmedian(dists)),
        "mean_units": float(np.nanmean(dists)),
    }

def write_metric_preds(preds_path: Path, out_path: Path, scale_mm_per_unit: float):
    n_in = n_out = 0
    with open(preds_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
        for ln in fin:
            n_in += 1
            o = json.loads(ln)
            new_persons = []
            for p in (o.get('persons') or []):
                kps = np.asarray(p.get('keypoints_xyz'), dtype=float)
                if kps.ndim != 2 or kps.shape[1] != 3:
                    continue
                p2 = dict(p)
                p2["keypoints_xyz_mm"] = (scale_mm_per_unit * kps).tolist()
                new_persons.append(p2)

            o["persons"] = new_persons
            o["scale_mm_per_unit"] = float(scale_mm_per_unit)
            fout.write(json.dumps(o) + "\n")
            n_out += 1
    return n_in, n_out


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Scale RTMW3D preds to metric (mm) using subject height.")
    ap.add_argument("-m", "--manifest", required=True, help="Path to manifest.yaml")
    ap.add_argument("-p", "--paths",    required=True, help="Path to paths.yaml")
    ap.add_argument("--trial", required=True, help="Trial ID (e.g., walking1)")
    ap.add_argument("--outputs-root", default=None, help="Override outputs root if manifest.output_dir absent")
    # Optional path overrides
    ap.add_argument("--preds", default=None, help="Override path to preds.jsonl")
    ap.add_argument("--meta",  default=None, help="Override path to meta.json")
    ap.add_argument("--out",   default=None, help="Override output preds_metric.jsonl path")
    # Height + selection
    ap.add_argument("--height-mm", type=float, default=None, help="Subject height in mm (defaults to meta['subject']['height_m']*1000)")
    ap.add_argument("--head-joints", type=str, default="nose", help="Comma list of head joints (names in meta.keypoint_names)")
    ap.add_argument("--foot-joints", type=str,
                    default="left_heel,right_heel,left_big_toe,right_big_toe,left_ankle,right_ankle",
                    help="Comma list of foot joints")
    ap.add_argument("--person-index", type=int, default=-1, help="-1: auto (highest mean_score), else fixed index")
    ap.add_argument("--min-mean-score", type=float, default=0.0, help="Ignore persons below this mean score")
    ap.add_argument("--height-percentile", type=float, default=95.0, help="Percentile of head-foot distance to use")
    args = ap.parse_args()

    # Load and resolve manifest
    manifest = load_manifest(args.manifest, args.paths)

    # Find the trial entry
    subset, trial = find_trial(manifest, args.trial)
    if trial is None:
        raise SystemExit(f"Trial '{args.trial}' not found in manifest.")

    # Resolve directories/paths (mirror rtmw3d.py layout)
    trial_root = decide_output_dir(manifest, trial, args.outputs_root)               # .../<trial>/
    rtmw3d_dir = trial_root / "rtmw3d"                                              # .../<trial>/rtmw3d/
    preds_path = Path(args.preds) if args.preds else (rtmw3d_dir / "preds.jsonl")
    meta_path  = Path(args.meta)  if args.meta  else (trial_root / "meta.json")
    out_path   = Path(args.out)   if args.out   else (rtmw3d_dir / "preds_metric.jsonl")

    if not preds_path.exists():
        raise SystemExit(f"preds.jsonl not found: {preds_path}")
    if not meta_path.exists():
        raise SystemExit(f"meta.json not found: {meta_path}")
    ensure_dir(out_path.parent)

    # Load meta (get keypoint_names and height if needed)
    meta = json.load(open(meta_path, 'r', encoding='utf-8'))
    kp_names = meta.get("keypoint_names") or []
    if not kp_names:
        raise SystemExit("meta.json has empty keypoint_names; cannot map joints.")

    # Resolve subject height (mm)
    height_mm = args.height_mm
    if height_mm is None:
        subj = meta.get("subject") or {}
        hm = subj.get("height_m")
        if isinstance(hm, (int, float)) and hm > 0:
            height_mm = float(hm) * 1000.0
    if height_mm is None:
        raise SystemExit("--height-mm not provided and subject.height_m missing in meta.json.")

    head_names = [s.strip() for s in args.head_joints.split(",") if s.strip()]
    foot_names = [s.strip() for s in args.foot_joints.split(",") if s.strip()]

    # Compute global scale
    scale, height_units, stats = compute_global_scale_from_preds(
        preds_path=preds_path,
        kp_names=kp_names,
        height_mm=float(height_mm),
        head_names=head_names,
        foot_names=foot_names,
        person_index=args.person_index,
        min_mean_score=args.min_mean_score,
        height_percentile=args.height_percentile,
    )

    print(f"[INFO] subject_height_mm={height_mm:.1f} | "
          f"model_height_P{args.height_percentile:.1f}={height_units:.3f} | "
          f"scale={scale:.6f} mm/unit")

    # Write scaled preds
    n_in, n_out = write_metric_preds(preds_path, out_path, scale)
    print(f"[DONE] {out_path} | frames_in={n_in}, frames_out={n_out}")

if __name__ == "__main__":
    main()
