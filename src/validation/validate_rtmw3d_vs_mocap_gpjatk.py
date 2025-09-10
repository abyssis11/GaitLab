#!/usr/bin/env python3
"""
mpjpe_from_trc.py

Compute MPJPE (Mean Per Joint Position Error) between two TRC files:
- A "prediction" TRC (e.g., RTMW3D canonical, usually ~25 Hz)
- A "reference" TRC (e.g., optical mocap, often ~100 Hz)

Key features
------------
- Robust TRC parser (handles standard Vicon-style TRC headers).
- Joint mapping using basic lower-limb markers (LASI, RASI, LKNE, RKNE, LANK, RANK, LHEE, RHEE, LTOE, RTOE).
- Optional pelvis root-centering using the midpoint of LASI & RASI (per frame).
- Optional per-frame rigid / similarity Procrustes alignment (Kabsch).
- Resampling by time with linear interpolation to a common timeline.
- Unit normalization (mm -> m) when the TRC header declares "Units".
- Axis canonicalization presets or explicit axis mapping (e.g., "-y,x,z" for Vicon -> canonical).
- Per-joint and overall MPJPE summaries, plus optional CSV dump of per-frame errors.

Usage
-----
Example with your two files:

    python src/validation/validate_rtmw3d_vs_mocap_gpjatk.py \
        -m ./manifests/GPJATK/subject1.yaml \
        -p ./config/paths.yaml \
        --trial walking1 \
        --root-center pelvis \
        --summary-per-joint \
        --procrustes similarity \
        --resample-to mocap \
        --out-json

Interpretation:
- "prediction" (first path) is assumed already canonical (RTMW3D).
- "reference" (second path) is mocap; we map axes to canonical via --mocap-preset.
- We resample both onto the RTMW3D timeline.
- We root-center both sequences by pelvis midpoint before MPJPE.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from IO.load_manifest import load_manifest
import os


# ---------- Logging ----------
def log_step(msg): print(f"[STEP] {msg}")
def log_info(msg): print(f"[INFO] {msg}")
def log_warn(msg): print(f"[WARN] {msg}")
def log_err (msg): print(f"[ERROR] {msg}")
def log_done(msg): print(f"[DONE] {msg}")

# ---------- IO helpers ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


import numpy as np


# -------------------------------
# Joint mapping (marker -> joint)
# -------------------------------

BASIC_JOINTS_GPJATK = [
    ("left_hip",      ("left_hip","LASI")),
    ("right_hip",     ("right_hip","RASI")),
    ("left_knee",     ("left_knee","LKNE")),
    ("right_knee",    ("right_knee","RKNE")),
    ("left_ankle",    ("left_ankle","LANK")),
    ("right_ankle",   ("right_ankle","RANK")),
    ("left_heel",     ("left_heel","LHEE")),
    ("right_heel",    ("right_heel","RHEE")),
    ("left_big_toe",  ("left_big_toe","LTOE")),
    ("right_big_toe", ("right_big_toe","RTOE")),
]

# Target → tuple of source names (predictions first: *_study, then reference without suffix)
BASIC_ENHANCER_STUDY = [
    # ----- Right leg: knee/ankle/foot -----
    ("r_knee",          ("r_knee_study", "r_knee")),
    ("r_ankle",         ("r_ankle_study", "r_ankle" )),
    ("r_calc",          ("r_calc_study", "r_calc")),
    ("r_5meta",         ("r_5meta_study", "r_5meta")),
    ("r_toe",           ("r_toe_study", "r_toe")),

    # ----- Left leg: knee/ankle/foot -----
    ("L_knee",          ("L_knee_study", "L_knee")),
    ("L_shank_antsup",  ("L_sh1_study", "L_shank_antsup")),
    ("L_ankle",         ("L_ankle_study","L_ankle")),
    ("L_calc",          ("L_calc_study", "L_calc")),
    ("L_5meta",         ("L_5meta_study", "L_5meta")),
    ("L_toe",           ("L_toe_study", "L_toe")),

    # ----- Shank extra points -----
    ("r_sh2",           ("r_sh2_study", "r_sh2")),
    ("r_sh3",           ("r_sh3_study", "r_sh3")),
    ("L_sh2",           ("L_sh2_study", "L_sh2")),
    ("L_sh3",           ("L_sh3_study", "L_sh3")),

    # ----- Thigh points (right) -----
    ("r_thigh1",        ("r_thigh1_study", "r_thigh1")),
    ("r_thigh2",        ("r_thigh2_study", "r_thigh2")),
    ("r_thigh3",        ("r_thigh3_study", "r_thigh3")),

    # ----- Thigh points (left) -----
    ("L_thigh1",        ("L_thigh1_study", "L_thigh1")),
    ("L_thigh2",        ("L_thigh2_study", "L_thigh2")),
    ("L_thigh3",        ("L_thigh3_study", "L_thigh3")),

    # ----- Pelvis landmarks -----
    ("r.ASIS",          ("r.ASIS_study", "r.ASIS")),
    ("r.PSIS",          ("r.PSIS_study", "r.PSIS")),
    ("L.ASIS",          ("L.ASIS_study", "L.ASIS")),
    ("L.PSIS",          ("L.PSIS_study", "L.PSIS")),
    ("C7",              ("C7_study", "C7")),

    # ----- Shoulders / chest -----
    ("R_Shoulder",      ("r_shoulder_study", "R_Shoulder")),
    ("L_Shoulder",      ("L_shoulder_study", "L_Shoulder")),

    # ----- Right upper limb -----
    ("R_elbow_med",     ("r_melbow_study", "R_elbow_med")),
    ("R_elbow_lat",     ("r_lelbow_study", "R_elbow_lat")),
    ("R_wrist_radius",  ("r_lwrist_study", "R_wrist_radius")),
    ("R_wrist_ulna",    ("r_mwrist_study", "R_wrist_ulna")),

    # ----- Left upper limb -----
    ("L_elbow_med",     ("L_melbow_study", "L_elbow_med")),
    ("L_elbow_lat",     ("L_lelbow_study", "L_elbow_lat")),
    ("L_wrist_radius",  ("L_lwrist_study", "L_wrist_radius")),
    ("L_wrist_ulna",    ("L_mwrist_study", "L_wrist_ulna")),

    # ----- Hip joint centers -----
    ("R_HJC",           ("RHJC_study", "R_HJC")),
    ("L_HJC",           ("LHJC_study", "L_HJC")),
]


# -------------------------------
# Utilities
# -------------------------------

def _clean_name(s: str) -> str:
    s = s.strip().lower()
    # allow dots in names
    return re.sub(r'[^a-z0-9_.]+', '', s)


@dataclass
class TRCData:
    time: np.ndarray                  # shape (T,)
    markers: Dict[str, np.ndarray]    # name -> (T, 3)
    units: str                        # 'm' or 'mm' (best-effort)
    data_rate: Optional[float] = None # Hz (if parsed)
    raw_header: List[str] = None      # original header lines (informational)



def parse_trc(path: str, missing_as_nan: bool = False) -> TRCData:
    """
    Parse a Vicon-style TRC file (robust to tabs/commas). If the standard header is present,
    it looks for a line containing both "Frame" and "Time". Returns coordinates in meters.
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.read().splitlines()

    header_lines = []
    frame_header_idx = None
    xzy_header_idx = None
    # Scan the first 300 lines for a header containing both "Frame" and "Time"
    for i, line in enumerate(lines[:300]):
        header_lines.append(line)
        if (re.search(r'\bFrame\b', line, re.I) and re.search(r'\bTime\b', line, re.I)):
            frame_header_idx = i
            xzy_header_idx = i + 1
            break

    if frame_header_idx is None:
        raise ValueError(f"Could not find 'Frame/Time' header in TRC: {path}")

    # Try to find "Units" and "DataRate" meta line above
    units = 'm'
    data_rate = None
    for j in range(max(0, frame_header_idx - 10), frame_header_idx + 1):
        line = header_lines[j]
        if re.search(r'\bUnits\b', line, re.I):
            cols = re.split(r'[\s,\t]+', line.strip())
            # Look at next line for values
            if j + 1 < len(lines):
                vals = re.split(r'[\s,\t]+', lines[j + 1].strip())
                if len(vals) >= len(cols):
                    try:
                        units_col = [c.lower() for c in cols].index('units')
                        units_value = vals[units_col]
                        units = units_value.lower()
                    except Exception:
                        pass
                    try:
                        dr_col = [c.lower() for c in cols].index('datarate')
                        data_rate = float(vals[dr_col])
                    except Exception:
                        pass
            break

    # Parse marker names from the "Frame/Time/..." header line
    marker_names_line = lines[frame_header_idx]
    cols = re.split(r'[\s,\t]+', marker_names_line.strip())
    if len(cols) < 3:
        raise ValueError("Malformed TRC marker header.")
    marker_names = cols[2:]

    data_start = xzy_header_idx + 1  # skip the X1 Y1 Z1 ... row

    # Map markers to column indices: [Frame, Time, X1, Y1, Z1, X2, Y2, Z2, ...]
    marker_indices = {}
    for i, m in enumerate(marker_names):
        marker_indices[_clean_name(m)] = (2 + 3*i, 2 + 3*i + 1, 2 + 3*i + 2)

    times = []
    data_buffers = {name: [] for name in marker_indices.keys()}

    for line in lines[data_start:]:
        if not line.strip():
            continue
        parts = re.split(r'[\s,\t]+', line.strip())
        if len(parts) < 2:
            continue
        try:
            _ = float(parts[0]); _ = float(parts[1])
        except Exception:
            break

        times.append(float(parts[1]))
        for mname, (ix, iy, iz) in marker_indices.items():
            def parse_cell(idx):
                if idx >= len(parts):
                    return float('nan')
                s = parts[idx].strip()
                if s == '' or s.lower() == 'nan':
                    return float('nan')
                try:
                    v = float(s)
                except Exception:
                    v = float('nan')
                if missing_as_nan and v == 0.0:
                    return float('nan')
                return v
            x = parse_cell(ix); y = parse_cell(iy); z = parse_cell(iz)
            data_buffers[mname].append((x, y, z))

    time = np.asarray(times, dtype=float)
    markers = {name: np.asarray(vals, dtype=float) for name, vals in data_buffers.items()}

    if isinstance(units, str) and 'mm' in units:
        for k in markers:
            markers[k] *= 1.0/1000.0
        units = 'm'
    else:
        units = 'm'

    return TRCData(time=time, markers=markers, units=units, data_rate=data_rate, raw_header=header_lines)

def apply_axis_expr(arr: np.ndarray, expr: str) -> np.ndarray:
    """
    Apply an axis mapping expression like "x,y,z" or "-y,x,z" to an array of shape (..., 3).
    The expression can reference x,y,z each exactly once (with optional '-' sign).
    """
    expr = expr.strip().lower()
    parts = [p.strip() for p in expr.split(',')]
    if len(parts) != 3:
        raise ValueError(f"Axis expr must have 3 comma-separated components, got: {expr!r}")
    # Build a (3,3) transform matrix
    # e.g., "-y,x,z" -> rows: [-e_y, e_x, e_z]
    basis = {'x': np.array([1.0, 0.0, 0.0]),
             'y': np.array([0.0, 1.0, 0.0]),
             'z': np.array([0.0, 0.0, 1.0])}
    M = np.zeros((3,3), dtype=float)
    used = set()
    for row, token in enumerate(parts):
        sign = -1.0 if token.startswith('-') else 1.0
        axis = token.lstrip('+-')
        if axis not in basis:
            raise ValueError(f"Invalid axis token {token!r} in expr {expr!r}")
        if axis in used:
            raise ValueError(f"Axis {axis!r} used more than once in expr {expr!r}")
        used.add(axis)
        M[row, :] = sign * basis[axis]
    # Apply: out = arr @ M.T
    shp = arr.shape
    out = arr.reshape(-1, 3) @ M.T
    return out.reshape(shp)


def axes_preset(name: str) -> str:
    """
    Named axis remaps for convenience.
    - 'none'               : 'x,y,z'
    - 'vicon_to_rtmw3d'    : '-y,x,z'    # Vicon (X forward, Y left, Z up) -> canonical (X right, Y forward, Z up)
    """
    name = (name or 'none').lower()
    if name == 'none':
        return 'x,y,z'
    if name in ('vicon_to_rtmw3d', 'vicon2rtmw3d'):
        return '-y,x,z'
    raise ValueError(f"Unknown axes preset: {name}")


def resample_timeseries(t_src: np.ndarray, X: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """
    Linear resample X (T, J, 3) defined at times t_src to new times t_dst.
    NaNs are supported: each coordinate channel is interpolated only where valid.
    """
    T_src, J, C = X.shape
    assert C == 3
    Y = np.full((len(t_dst), J, 3), np.nan, dtype=float)
    for j in range(J):
        for c in range(3):
            x = X[:, j, c]
            mask = ~np.isnan(x)
            if mask.sum() < 2:
                # Not enough points to interpolate
                continue
            # Restrict to valid segment
            xs = x[mask]; ts = t_src[mask]
            # For extrapolation beyond range, we clamp to ends
            Y[:, j, c] = np.interp(t_dst, ts, xs, left=np.nan, right=np.nan)
            # np.interp does not allow NaN for left/right; emulate by setting NaN outside the valid interval
            valid_range = (t_dst >= ts.min()) & (t_dst <= ts.max())
            Y[~valid_range, j, c] = np.nan
    return Y


def build_joint_array(trc: TRCData, mapping: List[Tuple[str, Tuple[str, ...]]]) -> Tuple[np.ndarray, List[str]]:
    """
    Create an array of shape (T, J, 3) according to mapping.
    Each joint can average multiple markers; missing markers yield NaN.
    Returns (array, joint_names_in_order).
    """
    T = trc.time.shape[0]
    joint_names = []
    joint_data = []
    # Build name map for ease (lowercased)
    mk = { _clean_name(k): v for k,v in trc.markers.items() }

    for jname, mtuple in mapping:
        jkey = _clean_name(jname)
        acc = np.zeros((T, 3), dtype=float)
        count = np.zeros((T, 1), dtype=float)
        for m in mtuple:
            mkey = _clean_name(m)
            if mkey not in mk:
                continue
            V = mk[mkey].astype(float)
            # Mark rows with any NaN as invalid for this marker
            valid = ~np.isnan(V).any(axis=1)
            acc[valid] += V[valid]
            count[valid, 0] += 1.0
        Jarr = np.full((T,3), np.nan, dtype=float)
        good = count[:,0] > 0
        if np.any(good):
            Jarr[good] = acc[good] / count[good]
        joint_data.append(Jarr)
        joint_names.append(jkey)
    X = np.stack(joint_data, axis=1)  # (T, J, 3)
    return X, joint_names


def compute_pelvis_mid(X: np.ndarray, joint_names: List[str]) -> np.ndarray:
    """Compute pelvis midpoint from left_hip and right_hip in X (T, J, 3)."""
    try:
        #print(joint_names)
        if "left_hip" and "right_hip" in joint_names:
            li = joint_names.index(_clean_name("left_hip"))
            ri = joint_names.index(_clean_name("right_hip"))
        else:
            li = joint_names.index(_clean_name("l.asis"))
            ri = joint_names.index(_clean_name("r.asis"))
    except ValueError:
        # Fall back: try LASI/RASI from markers if they were mapped with same names
        raise ValueError("left_hip/right_hip not present in joint array; cannot compute pelvis midpoint.")
    L = X[:, li, :]  # (T,3)
    R = X[:, ri, :]  # (T,3)
    pelv = 0.5 * (L + R)
    # If either side is NaN per-frame, pelvis becomes NaN
    pelv[np.isnan(L).any(axis=1) | np.isnan(R).any(axis=1)] = np.nan
    return pelv


def kabsch_align(P: np.ndarray, Q: np.ndarray, mode: str = 'rigid') -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Align P to Q with Kabsch/Procrustes.
    Inputs: P and Q as (n,3) each, corresponding points (with no NaNs).
    mode: 'rigid' (R,t) or 'similarity' (s*R,t)
    Returns: (R, t, P_aligned, scale)
    """
    assert mode in ('rigid', 'similarity')
    if P.shape[0] < 3:
        # Not enough points — return identity
        R = np.eye(3)
        t = np.zeros(3)
        return R, t, P.copy(), 1.0

    muP = P.mean(axis=0)
    muQ = Q.mean(axis=0)
    Pc = P - muP
    Qc = Q - muQ
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    if mode == 'similarity':
        varP = np.sum(Pc**2)
        scale = np.sum(S) / (varP + 1e-12)
    else:
        scale = 1.0
    t = muQ - scale * (R @ muP)
    P_aligned = (scale * (R @ P.T)).T + t
    return R, t, P_aligned, scale


def compute_mpjpe(
    X_pred: np.ndarray, X_ref: np.ndarray, 
    root_center: bool = False,
    procrustes: str = 'none',
    joint_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """
    Compute per-frame MPJPE between X_pred and X_ref.
    X_* shape: (T, J, 3), possibly with NaNs for missing joints.
    root_center: subtract pelvis midpoint (left/right hip) per frame from both before error.
    procrustes: 'none' | 'rigid' | 'similarity' (per-frame); applied after optional root-centering.
    Returns: (per_frame_mpjpe[T], overall_mean, per_joint_mean[dict])
    """
    assert X_pred.shape == X_ref.shape
    T, J, C = X_pred.shape
    assert C == 3

    Xp = X_pred.copy()
    Xr = X_ref.copy()

    if root_center:
        if joint_names is None:
            raise ValueError("joint_names required for root centering.")
        pelv_p = compute_pelvis_mid(Xp, joint_names)
        pelv_r = compute_pelvis_mid(Xr, joint_names)
        for t in range(T):
            if not np.isnan(pelv_p[t]).any():
                Xp[t] -= pelv_p[t]
            else:
                Xp[t] = np.nan
            if not np.isnan(pelv_r[t]).any():
                Xr[t] -= pelv_r[t]
            else:
                Xr[t] = np.nan

    per_frame = np.full(T, np.nan, dtype=float)

    # Per-joint aggregates
    joint_sums = np.zeros(J, dtype=float)
    joint_counts = np.zeros(J, dtype=float)

    for t in range(T):
        P = Xp[t]  # (J,3)
        Q = Xr[t]
        mask = ~np.isnan(P).any(axis=1) & ~np.isnan(Q).any(axis=1)
        if mask.sum() < 1:
            continue
        Pm = P[mask]
        Qm = Q[mask]

        if procrustes in ('rigid', 'similarity'):
            _, _, Pm_aligned, _ = kabsch_align(Pm, Qm, mode=procrustes)
            diffs = Pm_aligned - Qm
        else:
            diffs = Pm - Qm

        dists = np.linalg.norm(diffs, axis=1)  # per-joint distances at frame t (only masked joints)
        per_frame[t] = np.mean(dists)

        # Accumulate per-joint
        # Need to scatter back into joint indices
        idxs = np.where(mask)[0]
        for local_k, j_idx in enumerate(idxs):
            joint_sums[j_idx] += dists[local_k]
            joint_counts[j_idx] += 1.0

    overall_mean = np.nanmean(per_frame)

    per_joint_mean = {}
    if joint_names is not None:
        for j, name in enumerate(joint_names):
            if joint_counts[j] > 0:
                per_joint_mean[name] = joint_sums[j] / joint_counts[j]
            else:
                per_joint_mean[name] = np.nan

    return per_frame, overall_mean, per_joint_mean


def main():
    ap = argparse.ArgumentParser(description="Compute MPJPE between two TRC files with basic lower-limb mapping.")
    ap.add_argument("-m", "--manifest", required=True)
    ap.add_argument("-p", "--paths", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--resample-to", choices=["rtmw3d", "prediction", "mocap", "reference", "common", "hz"],
                    default="prediction",
                    help="Timeline to resample onto. 'prediction'/'rtmw3d' = first file; 'reference'/'mocap' = second file; "
                         "'common' = overlapping time grid at slower rate; 'hz' requires --target-hz.")
    ap.add_argument("--target-hz", type=float, default=None, help="Target Hz if --resample-to hz.")
    ap.add_argument("--time-shift-mocap", type=float, default=0.0,
                    help="Shift reference/mocap times by this many seconds (positive delays mocap).")
    ap.add_argument("--mocap-preset", type=str, default="none",
                    help="Axis preset for reference/mocap (e.g., 'vicon_to_rtmw3d').")
    ap.add_argument("--mocap-axis-expr", type=str, default=None,
                    help="Explicit axis map for reference/mocap, e.g., '-y,x,z'. Overrides --mocap-preset if given.")
    ap.add_argument("--prediction-preset", type=str, default="none",
                    help="Axis preset for prediction/rtmw3d if needed (defaults to 'none').")
    ap.add_argument("--prediction-axis-expr", type=str, default=None,
                    help="Explicit axis map for prediction, e.g., 'x,y,z'. Overrides --prediction-preset if given.")
    ap.add_argument("--root-center", choices=["none", "pelvis"], default="pelvis",
                    help="Root-centering strategy; 'pelvis' uses midpoint of LASI&RASI. Default: pelvis.")
    ap.add_argument("--procrustes", choices=["none", "rigid", "similarity"], default="none",
                    help="Optional per-frame Kabsch alignment after root-centering.")
    ap.add_argument("--missing-as-nan", action="store_true",
                    help="Treat (0,0,0) triplets as missing values (NaN).")
    ap.add_argument("--summary-per-joint", action="store_true",
                    help="Print a per-joint MPJPE table.")
    ap.add_argument("--out-json", action="store_true", 
                    help="Optional JSON to save per-frame MPJPE and per-joint distances.")
    ap.add_argument("--enhanc", action="store_true", 
                    help="Validate enhancer")
    args = ap.parse_args()

    log_step("Loading and resolving manifest")
    manifest = load_manifest(args.manifest, args.paths)

    # Find trial
    log_step(f"Finding trial '{args.trial}'")
    trial = None
    for subset, trials in manifest.get("trials", {}).items():
        for t in trials:
            if t.get("id") == args.trial:
                trial = t
                break
        if trial: break
    if trial is None:
        raise SystemExit(f"[ERROR] Trial '{args.trial}' not found in manifest.")

    # Paths
    base = manifest.get('output_dir')
    subj = manifest.get('subject_id', 'subject')
    if not base:
        sess = manifest.get('session', 'Session')
        cam  = manifest.get('camera', 'Cam')
        base = Path(manifest.get('outputs_root', Path.cwd() / "outputs")) / subj / sess / cam
    trial_root = Path(base) / trial['id']
    rtmw3d_dir = trial_root / "rtmw3d"
    eval_dir = trial_root / "rtmw3d_eval"
    enh_dir  = trial_root / "enhancer"
    enh_eval_dir  = trial_root / "enhancer_eval"
    ensure_dir(eval_dir)
    ensure_dir(enh_dir)
    ensure_dir(enh_eval_dir)

    meta_path  = trial_root / "meta.json"
    prediction_trc = rtmw3d_dir / "rtmw3d_cannonical.trc"
    reference_trc = trial["mocap_trc"]
    mpjpe_out = eval_dir / f"{subj}_mpjpe.json"
    joints_mapping = BASIC_JOINTS_GPJATK
    if args.enhanc:
        prediction_trc = enh_dir / f"enhancer_{args.trial}_cannonical.trc"
        mpjpe_out = enh_eval_dir / f"{subj}_enhancer_mpjpe.json"
        joints_mapping = BASIC_ENHANCER_STUDY

    #print(joints_mapping)
    log_info(f"reference_trc : {reference_trc}")
    log_info(f"prediction_trc : {prediction_trc}")
    log_info(f"meta.json  : {meta_path}")


    # Load TRCs
    trc_pred = parse_trc(prediction_trc, missing_as_nan=args.missing_as_nan)
    trc_ref  = parse_trc(reference_trc,  missing_as_nan=args.missing_as_nan)

    # Axis canonicalization
    pred_expr = args.prediction_axis_expr or axes_preset(args.prediction_preset)
    ref_expr  = args.mocap_axis_expr or axes_preset(args.mocap_preset)

    # Build joint arrays
    X_pred, joints = build_joint_array(trc_pred, joints_mapping)
    X_ref, _       = build_joint_array(trc_ref,  joints_mapping)

    # Apply axis transforms
    X_pred = apply_axis_expr(X_pred, pred_expr)
    X_ref  = apply_axis_expr(X_ref,  ref_expr)

    # Time handling
    t_pred = trc_pred.time.copy()
    t_ref  = trc_ref.time.copy() + args.time_shift_mocap

    # Decide target timeline
    if args.resample_to in ('prediction', 'rtmw3d'):
        t_dst = t_pred
    elif args.resample_to in ('reference', 'mocap'):
        t_dst = t_ref
    elif args.resample_to == 'hz':
        if not args.target_hz or args.target_hz <= 0:
            ap.error("--resample-to hz requires --target-hz > 0")
        t0 = max(t_pred.min(), t_ref.min())
        t1 = min(t_pred.max(), t_ref.max())
        if t1 <= t0:
            ap.error("No overlapping time range between sequences.")
        dt = 1.0 / float(args.target_hz)
        n = int(math.floor((t1 - t0) / dt)) + 1
        t_dst = t0 + np.arange(n, dtype=float) * dt
    elif args.resample_to == 'common':
        # Use the slower of the two (coarser) effective rates and the overlapping time window
        def est_rate(t):
            if len(t) < 2:
                return None
            dt = np.median(np.diff(t))
            return 1.0 / dt if dt > 0 else None
        r_pred = est_rate(t_pred) or 25.0
        r_ref  = est_rate(t_ref)  or 100.0
        target_hz = min(r_pred, r_ref)
        t0 = max(t_pred.min(), t_ref.min())
        t1 = min(t_pred.max(), t_ref.max())
        if t1 <= t0:
            ap.error("No overlapping time range between sequences.")
        dt = 1.0 / float(target_hz)
        n = int(math.floor((t1 - t0) / dt)) + 1
        t_dst = t0 + np.arange(n, dtype=float) * dt
    else:
        ap.error(f"Unknown --resample-to: {args.resample_to}")

    # Resample both sequences to t_dst
    Xp_rs = resample_timeseries(t_pred, X_pred, t_dst)
    Xr_rs = resample_timeseries(t_ref,  X_ref,  t_dst)

    # Compute MPJPE
    root_center = (args.root_center == 'pelvis')
    per_frame, overall, per_joint = compute_mpjpe(
        Xp_rs, Xr_rs, 
        root_center=root_center, 
        procrustes=args.procrustes,
        joint_names=joints
    )

    # Report
    print("MPJPE evaluation")
    print("-----------------")
    print(f"Prediction TRC: {prediction_trc}")
    print(f"Reference  TRC: {reference_trc}")
    print(f"Units normalized to meters (if declared in TRC headers).")
    print(f"Axis map (prediction): {pred_expr}")
    print(f"Axis map (reference):  {ref_expr}")
    print(f"Root-centering: {'pelvis' if root_center else 'none'}")
    print(f"Procrustes: {args.procrustes}")
    print(f"Resampled onto {len(t_dst)} frames (t in [{t_dst[0]:.3f}, {t_dst[-1]:.3f}] s).")
    print(f"Overall MPJPE: {overall*1000.0:.3f} mm")  # show in mm by default

    if args.summary_per_joint:
        print("\nPer-joint MPJPE (mm):")
        # Sort by joint name for readability
        for jname in sorted(per_joint.keys()):
            val = per_joint[jname]
            out = f"{val*1000.0:.3f}" if not np.isnan(val) else "NaN"
            print(f"  {jname:>14s}: {out}")

    # JSON report (optional)
    if args.out_json:
        def _to_num_or_none(x):
            try:
                if x is None:
                    return None
                if isinstance(x, float) and (x != x):
                    return None
                return float(x)
            except Exception:
                return None
        report = {
            "prediction_trc": str(prediction_trc),
            "reference_trc": str(reference_trc),
            "axis_map": {"prediction": pred_expr, "reference": ref_expr},
            "root_centering": "pelvis" if root_center else "none",
            "procrustes": args.procrustes,
            "resampled": {
                "num_frames": int(len(t_dst)),
                "time_start_s": _to_num_or_none(t_dst[0] if len(t_dst) > 0 else None),
                "time_end_s": _to_num_or_none(t_dst[-1] if len(t_dst) > 0 else None),
            },
            "overall_mpjpe_mm": _to_num_or_none(overall*1000.0),
            "per_joint_mpjpe_mm": {k: _to_num_or_none(v*1000.0) for k, v in per_joint.items()} if per_joint else {},
        }
        import json
        with open(mpjpe_out, "w", encoding="utf-8") as jf:
            json.dump(report, jf, indent=2)
        print(f"Saved JSON report to: {mpjpe_out}")

if __name__ == "__main__":
    main()
