#!/usr/bin/env python3
# rtmw3d_scale_from_height.py
"""
Scale RTMW3D predictions to METRIC (mm) using subject height from meta/manifest,
and export a TRC file with keypoints_xyz_mm (+ synthetic 'neck' and 'midHip).
If --trc-rate exceeds source FPS, resample to the target rate and low-pass filter.

Usage
-----
python src/pose/rtmw3d_scale_from_height.py \
  -m manifests/OpenCapDataset/subject2.yaml \
  -p config/paths.yaml \
  --trial walking1 \
  --trc --trc-rate 100 --trc-lpf-hz 6.0
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
    return Path(base) / trial['id']

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
# TRC export + resampling + LPF
# ----------------------------

def _canon(s: str) -> str:
    return s.lower().replace(" ", "_").replace("-", "_").replace(".", "")

def _infer_shoulder_indices(kp_names: list[str]) -> tuple[int | None, int | None]:
    canon_map = {_canon(n): i for i, n in enumerate(kp_names)}
    l_idx = next((canon_map[k] for k in ("left_shoulder","l_shoulder") if k in canon_map), None)
    r_idx = next((canon_map[k] for k in ("right_shoulder","r_shoulder") if k in canon_map), None)
    if l_idx is None:
        for c,i in canon_map.items():
            if "shoulder" in c and (c.startswith("l_") or "left" in c):
                l_idx = i; break
    if r_idx is None:
        for c,i in canon_map.items():
            if "shoulder" in c and (c.startswith("r_") or "right" in c):
                r_idx = i; break
    return l_idx, r_idx

def _infer_hip_indices(kp_names: list[str]) -> tuple[int | None, int | None]:
    """Return (left_idx, right_idx) for hips using common name patterns, else (None, None)."""
    canon_map = {_canon(n): i for i, n in enumerate(kp_names)}
    # Direct hits first
    l_idx = next((canon_map[k] for k in ("left_hip","l_hip") if k in canon_map), None)
    r_idx = next((canon_map[k] for k in ("right_hip","r_hip") if k in canon_map), None)
    # Fallback: any name containing "hip" + left/right hint
    if l_idx is None:
        for c, i in canon_map.items():
            if "hip" in c and (c.startswith("l_") or "left" in c):
                l_idx = i; break
    if r_idx is None:
        for c, i in canon_map.items():
            if "hip" in c and (c.startswith("r_") or "right" in c):
                r_idx = i; break
    return l_idx, r_idx


def _choose_person(persons: list[dict], person_index: int) -> dict | None:
    if not persons:
        return None
    if person_index == -1:
        ms = [p.get('mean_score', 0.0) for p in persons]
        pi = int(np.argmax(ms)) if ms else 0
    else:
        pi = min(max(0, person_index), len(persons) - 1)
    return persons[pi]

def _fmt_num(x: float) -> str:
    if x is None or not np.isfinite(x):
        return ""
    return f"{x:.5f}"

def _interp1d_nan_safe(y: np.ndarray, t_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    """Linear interpolate handling NaNs; extrapolation -> NaN."""
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return np.full_like(t_tgt, np.nan, dtype=float)
    yv = y[valid]
    tv = t_src[valid]
    out = np.interp(t_tgt, tv, yv)
    out[t_tgt < tv[0]] = np.nan
    out[t_tgt > tv[-1]] = np.nan
    return out

def _moving_average_zero_phase(y: np.ndarray, win_len: int) -> np.ndarray:
    """Centered MA with reflect padding; preserves length; NaN-tolerant."""
    if win_len < 3 or not np.isfinite(y).any():
        return y.copy()
    if win_len % 2 == 0:
        win_len += 1
    half = win_len // 2

    # Fill NaNs at ends to avoid propagation, then restore mask afterward
    y_in = y.copy()
    mask_nan = ~np.isfinite(y_in)
    if mask_nan.all():
        return y_in
    # forward/backward fill
    idx = np.where(~mask_nan)[0]
    first, last = idx[0], idx[-1]
    y_in[:first] = y_in[first]
    y_in[last+1:] = y_in[last]
    # simple linear fill internal NaNs
    runs = np.where(np.diff(np.hstack(([0], ~mask_nan, [0]))))[0].reshape(-1,2)
    for a,b in runs:
        if b-a > 1 and mask_nan[a:b].all():
            y_in[a:b] = np.linspace(y_in[a-1], y_in[b], b-a+1)[1:-1]

    pad = np.pad(y_in, (half,), mode='reflect')
    kernel = np.ones(win_len, dtype=float) / float(win_len)
    filt = np.convolve(pad, kernel, mode='valid')
    # restore NaNs where original was NaN
    filt[mask_nan] = np.nan
    return filt

def _auto_ma_window_len(fs: float, cutoff_hz: float) -> int:
    """For moving average, f_c(-3dB) ~ 0.443 * fs / L  =>  L ~ 0.443*fs/fc"""
    fc = max(0.1, float(cutoff_hz))
    L = int(round(0.443 * float(fs) / fc))
    L = max(3, min(L, int(2*fs)+1))
    if L % 2 == 0:
        L += 1
    return L

def _resample_xyz(frames_xyz: np.ndarray, src_rate: float, tgt_rate: float,
                  lpf_hz: float | None = 6.0, apply_filter: bool = True) -> np.ndarray:
    """
    frames_xyz: (F, K, 3), src_rate -> tgt_rate using linear interpolation,
    then zero-phase moving-average LPF (optional).
    """
    F, K, _ = frames_xyz.shape
    if F == 0:
        return frames_xyz

    t_src = np.arange(F, dtype=float) / float(src_rate)
    duration = t_src[-1]
    N_tgt = int(round(duration * float(tgt_rate))) + 1
    t_tgt = np.arange(N_tgt, dtype=float) / float(tgt_rate)

    out = np.full((N_tgt, K, 3), np.nan, dtype=float)

    for k in range(K):
        for c in range(3):
            y = frames_xyz[:, k, c]
            yi = _interp1d_nan_safe(y, t_src, t_tgt)
            if apply_filter and (lpf_hz is not None) and np.isfinite(yi).any():
                L = _auto_ma_window_len(tgt_rate, lpf_hz)
                yi = _moving_average_zero_phase(yi, L)
            out[:, k, c] = yi
    return out

def _trim_trailing_all_nan(frames_xyz: np.ndarray) -> np.ndarray:
    """Remove trailing frames where *all* markers/channels are NaN."""
    if frames_xyz.size == 0:
        return frames_xyz
    keep = frames_xyz.shape[0]
    while keep > 0 and not np.isfinite(frames_xyz[keep-1]).any():
        keep -= 1
    return frames_xyz[:keep]

def write_trc_from_metric(
    preds_metric_path: Path,
    trc_out_path: Path,
    kp_names: list[str],
    data_rate_hz: float,
    source_rate_hz: float,
    person_index: int = -1,
    lpf_hz: float | None = 6.0,
    filter_on_resample: bool = True,
):
    """
    Write a TRC file using keypoints_xyz_mm.
    If data_rate_hz > source_rate_hz: resample to data_rate_hz and low-pass filter (unless disabled).
    Adds a synthetic 'neck' = midpoint(L/R shoulder) (appended as last marker) when shoulders exist.
    """
    ensure_dir(trc_out_path.parent)

    # Read frames into (F, K, 3)
    K = len(kp_names)
    frames = []
    with open(preds_metric_path, "r", encoding="utf-8") as f:
        for ln in f:
            o = json.loads(ln)
            p = _choose_person(o.get("persons") or [], person_index)
            if p is None:
                frames.append(None); continue
            kps_mm = np.asarray(p.get("keypoints_xyz_mm"), dtype=float)
            if kps_mm.ndim != 2 or kps_mm.shape[1] != 3:
                frames.append(None); continue
            frames.append(kps_mm)

    F_src = len(frames)
    frames_xyz = np.full((F_src, K, 3), np.nan, dtype=float)
    for i, kps in enumerate(frames):
        if kps is None: continue
        frames_xyz[i, :min(K, kps.shape[0]), :] = kps[:K]

    # Resample if needed
    target_rate = float(data_rate_hz)
    src_rate = float(source_rate_hz)
    if target_rate > src_rate and F_src > 1:
        frames_xyz = _resample_xyz(
            frames_xyz, src_rate, target_rate,
            lpf_hz=lpf_hz, apply_filter=filter_on_resample
        )
        used_rate = target_rate
    else:
        used_rate = src_rate  # keep original timing if not upsampling

    # Compute 'neck' AFTER resampling for better continuity
    l_idx, r_idx = _infer_shoulder_indices(kp_names)
    add_neck = (l_idx is not None and r_idx is not None)

    if add_neck:
        L = frames_xyz[:, l_idx, :]
        R = frames_xyz[:, r_idx, :]
        neck = 0.5 * (L + R)
        # Ensure NaNs if either side is NaN
        nanmask = ~np.isfinite(L).all(axis=1) | ~np.isfinite(R).all(axis=1)
        neck[nanmask] = np.nan
        frames_xyz = np.concatenate([frames_xyz, neck[:, None, :]], axis=1)

    # Compute 'midHip' AFTER resampling (midpoint of left/right hip)
    hl_idx, hr_idx = _infer_hip_indices(kp_names)
    add_midhip = (hl_idx is not None and hr_idx is not None)

    if add_midhip:
        HL = frames_xyz[:, hl_idx, :]
        HR = frames_xyz[:, hr_idx, :]
        midhip = 0.5 * (HL + HR)
        nanmask_h = ~np.isfinite(HL).all(axis=1) | ~np.isfinite(HR).all(axis=1)
        midhip[nanmask_h] = np.nan
        frames_xyz = np.concatenate([frames_xyz, midhip[:, None, :]], axis=1)

    frames_xyz = _trim_trailing_all_nan(frames_xyz)
    num_frames = frames_xyz.shape[0]
    marker_names = list(kp_names) + (["neck"] if add_neck else []) + (["midHip"] if add_midhip else [])
    num_markers = len(marker_names)

    # Header
    with open(trc_out_path, "w", encoding="utf-8") as fout:
        fout.write(f"PathFileType\t4\t(X/Y/Z)\t{trc_out_path}\n")
        fout.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        fout.write(f"{target_rate:.1f}\t{target_rate:.1f}\t{num_frames}\t{num_markers}\tmm\t{used_rate:.1f}\t1\t{num_frames}\n")

        labels = "\t\t\t".join(marker_names)
        fout.write(f"Frame#\tTime\t{labels}\t\t\t\n")

        xyz_cols = []
        for i in range(1, num_markers + 1):
            xyz_cols.extend([f"X{i}", f"Y{i}", f"Z{i}"])
        fout.write("\t" + "\t".join(xyz_cols) + "\n\n")

        dt = 1.0 / float(target_rate) if target_rate > 0 else 0.0
        for fi in range(num_frames):
            t = fi * dt
            row_vals = [str(fi + 1), f"{t:.8f}"]
            for m in range(num_markers):
                x, y, z = frames_xyz[fi, m, :]
                row_vals.extend([_fmt_num(float(x)), _fmt_num(float(y)), _fmt_num(float(z))])
            fout.write("\t".join(row_vals) + "\n")


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Scale RTMW3D preds to metric (mm) using subject height (and optionally export TRC with resampling/LPF).")
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
    # TRC export options
    ap.add_argument("--trc", action="store_true", help="Also write a TRC file with keypoints_xyz_mm (+ synthetic neck).")
    ap.add_argument("--trc-out", default=None, help="Output TRC path (default: <trial>/rtmw3d/rtmw3d.trc)")
    ap.add_argument("--trc-rate", type=float, default=None, help="Sampling rate (Hz) for TRC; defaults to meta['fps'] or 100.0")
    ap.add_argument("--trc-lpf-hz", type=float, default=6.0, help="Low-pass cutoff (Hz) applied when upsampling.")
    ap.add_argument("--trc-no-filter", action="store_true", help="Disable low-pass filtering during resample.")

    args = ap.parse_args()

    manifest = load_manifest(args.manifest, args.paths)
    subset, trial = find_trial(manifest, args.trial)
    if trial is None:
        raise SystemExit(f"Trial '{args.trial}' not found in manifest.")

    trial_root = decide_output_dir(manifest, trial, args.outputs_root)
    rtmw3d_dir = trial_root / "rtmw3d"
    preds_path = Path(args.preds) if args.preds else (rtmw3d_dir / "preds.jsonl")
    meta_path  = Path(args.meta)  if args.meta  else (trial_root / "meta.json")
    out_path   = Path(args.out)   if args.out   else (rtmw3d_dir / "preds_metric.jsonl")

    if not preds_path.exists():
        raise SystemExit(f"preds.jsonl not found: {preds_path}")
    if not meta_path.exists():
        raise SystemExit(f"meta.json not found: {meta_path}")
    ensure_dir(out_path.parent)

    meta = json.load(open(meta_path, 'r', encoding='utf-8'))
    kp_names = meta.get("keypoint_names") or []
    if not kp_names:
        raise SystemExit("meta.json has empty keypoint_names; cannot map joints.")

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

    n_in, n_out = write_metric_preds(preds_path, out_path, scale)
    print(f"[DONE] {out_path} | frames_in={n_in}, frames_out={n_out}")

    upsampling = False
    if args.trc:
        src_rate = float(meta.get("fps") or meta.get("frame_rate") or 60.0)
        target_rate = float(args.trc_rate) if args.trc_rate is not None else src_rate

        if target_rate > src_rate:
            upsampling = True
            print(f"[INFO] Upsampling TRC: {src_rate:.3f} Hz → {target_rate:.3f} Hz "
                  f"({'LPF off' if args.trc_no_filter else f'LPF {args.trc_lpf_hz:.2f} Hz'})")
            
        trc_out = Path(args.trc_out) if args.trc_out else (rtmw3d_dir / f"rtmw3d{'_upsampled' if upsampling else ''}.trc")


        write_trc_from_metric(
            preds_metric_path=out_path,
            trc_out_path=trc_out,
            kp_names=kp_names,
            data_rate_hz=target_rate,
            source_rate_hz=src_rate,
            person_index=args.person_index,
            lpf_hz=args.trc_lpf_hz,
            filter_on_resample=(not args.trc_no_filter),
        )
        # Count markers with/without neck/midHip for message
        l_idx, r_idx = _infer_shoulder_indices(kp_names)
        add_neck = (l_idx is not None and r_idx is not None)
        hl_idx, hr_idx = _infer_hip_indices(kp_names)
        add_midhip = (hl_idx is not None and hr_idx is not None)
        print(f"[DONE] {trc_out} | rate={target_rate:.1f} Hz | markers={len(kp_names) + (1 if add_neck else 0) + (1 if add_midhip else 0)}")


if __name__ == "__main__":
    main()
