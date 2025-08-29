#!/usr/bin/env python3
# validate_rtmw3d_vs_mocap.py
"""
Validate RTMW3D vs MoCap (TRC @ 100 Hz, mm) WITHOUT auto-scaling.

Now supports two joint sets:
  --joint-set basic   (your original mapping; default)
  --joint-set ocap20  (OpenCap video 20: neck, mid_hip, shoulders, elbows, wrists,
                       hips, knees, ankles, heels, small toes, big toes)

Keeps your behavior:
- Uses manifest loader
- Requires metric predictions (keypoints_xyz_mm); no auto-scale
- Resamples to 100 Hz & optional time-offset estimation
- Pelvis-centers + single Kabsch rotation
- Reports per-joint MPJPE and overall MPJPE (mm), plus RMSE
- Verbose logging

python src/validation/validate_rtmw3d_vs_mocap.py \
        -m manifests/OpenCapDataset/subject2.yaml \
        -p config/paths.yaml \
        --trial walking1 \
        --joint-set ocap20 \
        --estimate-offset
"""

import argparse
import json
from pathlib import Path
import numpy as np

from IO.load_manifest import load_manifest

# ---------------------------
# Logging helpers
# ---------------------------

def log_step(msg): print(f"[STEP] {msg}")
def log_info(msg): print(f"[INFO] {msg}")
def log_warn(msg): print(f"[WARN] {msg}")
def log_done(msg): print(f"[DONE] {msg}")

# ---------------------------
# Basic utils
# ---------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def jsonable(x):
    import numpy as _np
    from pathlib import Path as _Path
    if isinstance(x, dict): return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if isinstance(x, _np.ndarray): return x.tolist()
    if isinstance(x, (_np.floating, _np.integer)): return x.item()
    if isinstance(x, _Path): return str(x)
    return x

# ---------------------------
# TRC loader (Vicon-like)
# ---------------------------

def load_trc(path: Path):
    """
    Read TRC into (time_s, markers_dict{name: (T,3) in mm}, data_rate).
    Matches the header layout you pasted earlier.
    """
    txt = Path(path).read_text(encoding='utf-8', errors='ignore').strip().splitlines()
    if len(txt) < 6:
        raise RuntimeError(f"TRC file too short: {path}")

    vals = txt[2].split()
    if len(vals) < 4:
        raise RuntimeError("TRC header missing numeric line (#2).")
    data_rate = float(vals[0])  # expected 100.0

    header_names_row = txt[3].split()
    marker_names = header_names_row[2:]  # after "Frame# Time"

    data = []
    for line in txt[5:]:
        if not line.strip(): continue
        data.append(line.split())
    data = np.array(data, dtype=float)

    expected_cols = 2 + 3 * len(marker_names)
    if data.shape[1] != expected_cols:
        raise RuntimeError(f"Unexpected TRC columns: got {data.shape[1]} expected {expected_cols}")

    time_s = data[:, 1].astype(float)
    markers = {}
    for i, name in enumerate(marker_names):
        x = data[:, 2 + 3*i + 0]
        y = data[:, 2 + 3*i + 1]
        z = data[:, 2 + 3*i + 2]
        markers[name] = np.stack([x, y, z], axis=-1)
    return time_s, markers, data_rate

# ---------------------------
# RTMW3D reader (jsonl, metric only)
# ---------------------------

def read_rtmw3d_preds_mm(preds_path: Path, expect_k=None, person_index=-1, min_mean_score=0.0):
    """
    Return: times (F,), preds_mm [F,K,3], and a string describing the source of mm data.
    NO AUTO-SCALE:
      - Uses 'keypoints_xyz_mm' only
      - Else tries ('keypoints_xyz' + 'keypoints_xyz_units') and treats it as mm *only if you wrote it like that)
      - Else -> None (caller exits)
    """
    times = []
    P_mm = []
    used_source = None
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

            if 'keypoints_xyz_mm' in p:
                kps = np.asarray(p['keypoints_xyz_mm'], dtype=float)
                used_source = 'keypoints_xyz_mm'
            elif ('keypoints_xyz' in p) and ('keypoints_xyz_units' in p):
                # only accept if you know these are already mm
                kps = np.asarray(p['keypoints_xyz'], dtype=float)
                used_source = 'keypoints_xyz (assumed mm)'
            else:
                continue

            if kps.ndim != 2 or kps.shape[1] != 3:
                continue
            if expect_k is not None and kps.shape[0] != expect_k:
                continue

            times.append(float(o.get('time_sec', 0.0)))
            P_mm.append(kps)

    if not times:
        return None, None, None
    return np.asarray(times, dtype=float), np.asarray(P_mm, dtype=float), used_source

# ---------------------------
# Resampling & alignment
# ---------------------------

def resample_to_grid(t_src, X_src, t_grid):
    """
    X_src: (T_src,K,3) -> (T_grid,K,3) via per-dim linear interp.
    Outside range -> NaN.
    """
    T_src, K, _ = X_src.shape
    T = len(t_grid)
    Y = np.full((T, K, 3), np.nan, dtype=float)
    for k in range(K):
        for d in range(3):
            Y[:, k, d] = np.interp(t_grid, t_src, X_src[:, k, d], left=np.nan, right=np.nan)
    return Y

def estimate_time_offset(sig_a_t, sig_a, sig_b_t, sig_b, max_offset=1.0, fs=100.0):
    """
    Cross-correlate scalar signals resampled to fs, constrain lag to ±max_offset.
    Returns offset (sec) to ADD to A so it best aligns to B.
    """
    t0 = max(sig_a_t.min(), sig_b_t.min())
    t1 = min(sig_a_t.max(), sig_b_t.max())
    if t1 - t0 < 0.5:
        return 0.0
    t_grid = np.arange(t0, t1, 1.0/fs)
    a = np.interp(t_grid, sig_a_t, sig_a)
    b = np.interp(t_grid, sig_b_t, sig_b)
    a = (a - np.nanmean(a)) / (np.nanstd(a) + 1e-8)
    b = (b - np.nanmean(b)) / (np.nanstd(b) + 1e-8)
    max_lag = int(round(max_offset * fs))
    corr = np.correlate(a, b, mode='full')
    lags = np.arange(-len(a)+1, len(a))
    sel = (lags >= -max_lag) & (lags <= max_lag)
    best = int(lags[sel][np.argmax(corr[sel])])
    return -best / fs

def kabsch_rotation(X, Y):
    """Find rotation R minimizing ||R X - Y||_F. X,Y: (N,3) centered."""
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R

def center_by_pelvis(X, name2idx, Lhip_name, Rhip_name):
    """X: (...,K,3). Subtract pelvis center per-frame."""
    L = name2idx.get(Lhip_name, None)
    R = name2idx.get(Rhip_name, None)
    if L is None or R is None:
        return X
    Xc = X.copy()
    pelv = 0.5 * (X[..., L, :] + X[..., R, :])
    Xc -= pelv[..., None, :]
    return Xc

# ---------------------------
# Joint mapping sets
# ---------------------------

# Your original (kept as-is)
BASIC_JOINTS = [
    ("left_hip",      ("L_HJC","L_HJC_reg")),
    ("right_hip",     ("R_HJC","R_HJC_reg")),
    ("left_knee",     ("L_knee",)),
    ("right_knee",    ("r_knee",)),
    ("left_ankle",    ("L_ankle",)),
    ("right_ankle",   ("r_ankle",)),
    ("left_heel",     ("L_calc",)),
    ("right_heel",    ("r_calc",)),
    ("left_big_toe",  ("L_toe",)),
    ("right_big_toe", ("r_toe",)),
]

# OpenCap video 20
OPENCAP20 = [
    'neck','mid_hip',
    'left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist',
    'left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle',
    'left_heel','right_heel','left_small_toe','right_small_toe','left_big_toe','right_big_toe',
]

def build_mocap_selector(markers_dict):
    lower_map = {k.lower(): k for k in markers_dict.keys()}
    def pick(*candidates):
        for c in candidates:
            k = lower_map.get(c.lower())
            if k is not None:
                return k
        return None
    return pick

def extract_mocap_stack_basic(t_trc, markers, joint_map):
    pick = build_mocap_selector(markers)
    used = []
    for h3wb_name, candidates in joint_map:
        mk = pick(*candidates)
        if mk is not None:
            used.append((h3wb_name, mk))
    if not used:
        raise RuntimeError("None of the requested MoCap markers found (basic set).")

    K = len(used)
    T = len(t_trc)
    M = np.full((T, K, 3), np.nan, dtype=float)
    joints = []
    used_markers = []
    for j, (hname, mk) in enumerate(used):
        M[:, j, :] = markers[mk]
        joints.append(hname)
        used_markers.append(mk)
    return joints, used_markers, M

def ocap20_from_trc(t_trc, markers):
    """
    Assemble OpenCap-20 from TRC markers (T,20,3).
    - Neck = mid(L/R shoulder)
    - Mid-hip = mid(L/R HJC) if available, else NaN
    - Elbow/Wrist = mid of medial/lat if available
    """
    T = len(t_trc)
    pick = build_mocap_selector(markers)

    def mid(A, B):
        return 0.5*(A + B)

    def get(name):
        return markers[name] if name in markers else None

    # Shoulders
    r_sh = get(pick('R_Shoulder','R_Shoulder_Acromion','r_shoulder'))
    l_sh = get(pick('L_Shoulder','L_Shoulder_Acromion','l_shoulder'))

    # HJCs
    r_hjc = get(pick('R_HJC','R_HJC_reg','R_HJCR','r_hjc'))
    l_hjc = get(pick('L_HJC','L_HJC_reg','L_HJCL','l_hjc'))

    # Knees/ankles
    r_knee = get(pick('r_knee','R_knee','R_Knee'))
    l_knee = get(pick('L_knee','l_knee','L_Knee'))
    r_ank  = get(pick('r_ankle','R_ankle','R_Ankle'))
    l_ank  = get(pick('L_ankle','l_ankle','L_Ankle'))

    # Heels / toes
    r_heel = get(pick('r_calc','R_calc','R_heel','R_Calc'))
    l_heel = get(pick('L_calc','l_calc','L_heel','L_Calc'))
    r_small = get(pick('r_5meta','R_5meta','R_small_toe','R_5th_met'))
    l_small = get(pick('L_5meta','l_5meta','L_small_toe','L_5th_met'))
    r_big   = get(pick('r_toe','R_toe','R_big_toe','R_1st_met'))
    l_big   = get(pick('L_toe','l_toe','L_big_toe','L_1st_met'))

    # Elbows: mid of med/lat if available
    r_el_med = get(pick('R_elbow_med','r_elbow_med'))
    r_el_lat = get(pick('R_elbow_lat','r_elbow_lat'))
    l_el_med = get(pick('L_elbow_med','l_elbow_med'))
    l_el_lat = get(pick('L_elbow_lat','l_elbow_lat'))

    if r_el_med is not None and r_el_lat is not None:
        r_el = mid(r_el_med, r_el_lat)
    else:
        r_el = get(pick('R_elbow','r_elbow','R_elbow_lat','R_elbow_med'))

    if l_el_med is not None and l_el_lat is not None:
        l_el = mid(l_el_med, l_el_lat)
    else:
        l_el = get(pick('L_elbow','l_elbow','L_elbow_lat','L_elbow_med'))

    # Wrists: mid of radius/ulna if available
    r_wr_rad = get(pick('R_wrist_radius','r_wrist_radius'))
    r_wr_uln = get(pick('R_wrist_ulna','r_wrist_ulna'))
    l_wr_rad = get(pick('L_wrist_radius','l_wrist_radius'))
    l_wr_uln = get(pick('L_wrist_ulna','l_wrist_ulna'))

    if r_wr_rad is not None and r_wr_uln is not None:
        r_wr = mid(r_wr_rad, r_wr_uln)
    else:
        r_wr = get(pick('R_wrist','r_wrist'))
    if l_wr_rad is not None and l_wr_uln is not None:
        l_wr = mid(l_wr_rad, l_wr_uln)
    else:
        l_wr = get(pick('L_wrist','l_wrist'))

    # Neck / mid_hip
    neck   = mid(l_sh, r_sh) if (l_sh is not None and r_sh is not None) else None
    midhip = mid(l_hjc, r_hjc) if (l_hjc is not None and r_hjc is not None) else None

    parts = [
        neck, midhip,
        l_sh, r_sh, l_el, r_el, l_wr, r_wr,
        l_hjc, r_hjc, l_knee, r_knee, l_ank, r_ank,
        l_heel, r_heel, l_small, r_small, l_big, r_big
    ]

    if any(p is None for p in parts):
        missing = [OPENCAP20[i] for i,p in enumerate(parts) if p is None]
        log_warn(f"Missing TRC markers for: {missing}")
        return None, None

    M = np.stack(parts, axis=1)  # (T,20,3)
    return M, list(OPENCAP20)

def ocap20_from_preds_full(P_full, kp_names):
    """
    Build (F,20,3) from full RTMW3D (F,K,3) with H3WB names.
    neck=mid(LS,RS), mid_hip=mid(LH,RH)
    """
    name2idx = {n:i for i,n in enumerate(kp_names)}
    def g(n): 
        i = name2idx.get(n, -1); 
        return None if i < 0 else P_full[:, i, :]

    Lsh = g('left_shoulder');  Rsh = g('right_shoulder')
    Lhp = g('left_hip');       Rhp = g('right_hip')
    if Lsh is None or Rsh is None or Lhp is None or Rhp is None:
        return None, None

    neck   = 0.5*(Lsh + Rsh)
    midhip = 0.5*(Lhp + Rhp)

    parts = [
        neck, midhip,
        g('left_shoulder'),  g('right_shoulder'),
        g('left_elbow'),     g('right_elbow'),
        g('left_wrist'),     g('right_wrist'),
        g('left_hip'),       g('right_hip'),
        g('left_knee'),      g('right_knee'),
        g('left_ankle'),     g('right_ankle'),
        g('left_heel'),      g('right_heel'),
        g('left_small_toe'), g('right_small_toe'),
        g('left_big_toe'),   g('right_big_toe'),
    ]
    if any(p is None for p in parts):
        missing = [OPENCAP20[i] for i,p in enumerate(parts) if p is None]
        log_warn(f"Missing RTMW3D joints for: {missing}")
        return None, None
    P = np.stack(parts, axis=1)  # (F,20,3)
    return P, list(OPENCAP20)

# ---------------------------
# Main eval
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Validate RTMW3D (metric only) vs MoCap (TRC @ 100 Hz, mm)")
    ap.add_argument("-m", "--manifest", required=True)
    ap.add_argument("-p", "--paths", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--outputs-root", default=None)

    # Options
    ap.add_argument("--person-index", type=int, default=-1)
    ap.add_argument("--min-mean-score", type=float, default=0.0)
    ap.add_argument("--estimate-offset", action="store_true")
    ap.add_argument("--max-offset", type=float, default=1.0, help="max |offset| (s) for xcorr")
    ap.add_argument("--save-series", action="store_true", help="save aligned 100Hz arrays to npz")
    ap.add_argument("--joint-set", choices=["basic","ocap20"], default="basic",
                    help="Which joint set to compare: your original 'basic' or OpenCap-20")

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

    # Resolve paths
    def decide_trial_root(manifest: dict, trial: dict, outputs_root_cli: str | None):
        base = manifest.get('output_dir')
        if not base:
            outputs_root = outputs_root_cli or Path.cwd() / "outputs"
            subj = manifest.get('subject_id', 'subject')
            sess = manifest.get('session', 'Session')
            cam  = manifest.get('camera', 'Cam')
            base = Path(outputs_root) / subj / sess / cam
        return Path(base) / trial['id']

    trial_root = decide_trial_root(manifest, trial, args.outputs_root)
    rtmw3d_dir = trial_root / "rtmw3d"
    eval_dir = trial_root / "rtmw3d_eval"
    ensure_dir(eval_dir)

    preds_metric = rtmw3d_dir / "preds_metric.jsonl"
    preds_plain  = rtmw3d_dir / "preds.jsonl"
    meta_path    = trial_root / "meta.json"
    trc_path     = Path(trial.get("mocap_trc", ""))

    log_info(f"Trial root          : {trial_root}")
    log_info(f"RTMW3D dir          : {rtmw3d_dir}")
    log_info(f"Preds (metric) path : {preds_metric}")
    log_info(f"Preds (plain)  path : {preds_plain}")
    log_info(f"meta.json path      : {meta_path}")
    log_info(f"TRC path            : {trc_path}")
    log_info(f"Joint set           : {args.joint_set}")

    # Existence checks
    preds_path = preds_metric if preds_metric.exists() else preds_plain
    if not preds_path.exists():
        raise SystemExit(f"[ERROR] Predictions not found: {preds_path}")
    if not meta_path.exists():
        raise SystemExit(f"[ERROR] meta.json not found: {meta_path}")
    if not trc_path.exists():
        raise SystemExit(f"[ERROR] TRC not found: {trc_path}")

    # Load meta for names
    log_step("Reading meta.json for keypoint names")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    kp_names = meta.get("keypoint_names") or []
    if not kp_names:
        raise SystemExit("[ERROR] meta.keypoint_names missing.")
    log_info(f"Keypoints K         : {len(kp_names)}")

    # Load predictions (MM ONLY, no auto-scale)
    log_step(f"Reading RTMW3D predictions from {preds_path}")
    t_pred, P_mm_full, used_source = read_rtmw3d_preds_mm(
        preds_path, expect_k=len(kp_names),
        person_index=args.person_index,
        min_mean_score=args.min_mean_score
    )
    if t_pred is None or P_mm_full is None:
        log_warn("No metric predictions found in file.")
        raise SystemExit("[EXIT] Aborting evaluation (no metric keypoints).")

    log_info(f"Frames read (preds) : {len(t_pred)}")
    log_info(f"Metric source       : {used_source}")
    log_info(f"Pred time range     : {t_pred.min():.3f}..{t_pred.max():.3f} s")

    # Load TRC
    log_step(f"Reading TRC from {trc_path}")
    t_trc, markers, trc_rate = load_trc(trc_path)
    log_info(f"TRC rate            : {trc_rate:.3f} Hz")
    log_info(f"TRC frames          : {len(t_trc)}")
    log_info(f"TRC time range      : {t_trc.min():.3f}..{t_trc.max():.3f} s")
    log_info(f"TRC markers         : {len(markers)} available")

    # Build joint stacks
    if args.joint_set == "basic":
        log_step("Mapping joints (BASIC set)")
        joints, used_markers, M_mm = extract_mocap_stack_basic(t_trc, markers, BASIC_JOINTS)
        name2idx = {n:i for i,n in enumerate(kp_names)}
        pred_idx = [name2idx[jn] for jn in joints]
        P_mm_sel = P_mm_full[:, pred_idx, :]  # (F,K,3)
    else:
        log_step("Building OpenCap-20 joints (neck/mid_hip composed)")
        M_mm, joints = ocap20_from_trc(t_trc, markers)
        if M_mm is None:
            raise SystemExit("[ERROR] Could not assemble OpenCap-20 from TRC (see missing markers warnings).")
        P_mm_sel, _ = ocap20_from_preds_full(P_mm_full, kp_names)
        if P_mm_sel is None:
            raise SystemExit("[ERROR] Could not assemble OpenCap-20 from RTMW3D (missing keypoints).")
        used_markers = ["composed_or_multi"] * len(joints)  # informational

    K = len(joints)
    log_info(f"Matched joints K    : {K}")
    for jn, mk in zip(joints, used_markers):
        log_info(f"  {jn:>15s}  <-  {mk}")

    # Optional time offset estimation (use shank lengths as alignment signal)
    def seg_len(X, a, b):
        return np.linalg.norm(X[:, a, :] - X[:, b, :], axis=-1)

    try:
        jLhip = joints.index("left_hip"); jRhip = joints.index("right_hip")
        jLank = joints.index("left_ankle"); jRank = joints.index("right_ankle")
        sig_pred = seg_len(P_mm_sel, jLhip, jLank) + seg_len(P_mm_sel, jRhip, jRank)
        sig_trc  = seg_len(M_mm,      jLhip, jLank) + seg_len(M_mm,      jRhip, jRank)
    except ValueError:
        # If a joint is missing (shouldn't in ocap20), fall back to zeros to skip offset estimation
        sig_pred = np.zeros(P_mm_sel.shape[0])
        sig_trc  = np.zeros(M_mm.shape[0])

    t_offset = 0.0
    if args.estimate_offset:
        log_step("Estimating time offset via cross-correlation")
        t_offset = estimate_time_offset(t_pred, sig_pred, t_trc, sig_trc, max_offset=args.max_offset, fs=100.0)
        log_info(f"Estimated offset    : {t_offset:+.3f} s (to add to predictions)")
    else:
        log_info("Offset estimation   : disabled (using 0.000 s)")

    # Resample preds to 100 Hz grid (after applying offset)
    log_step("Resampling both modalities to 100 Hz common grid")
    t0 = max(t_trc.min(), t_pred.min() + t_offset)
    t1 = min(t_trc.max(), t_pred.max() + t_offset)
    if t1 - t0 < 0.5:
        raise SystemExit("[ERROR] Insufficient temporal overlap after offset.")
    t_grid = np.arange(t0, t1, 0.01)  # 100 Hz
    log_info(f"Common time range   : {t0:.3f}..{t1:.3f} s")
    log_info(f"Grid frames (100Hz) : {len(t_grid)}")

    P_100 = resample_to_grid(t_pred + t_offset, P_mm_sel, t_grid)
    M_100 = resample_to_grid(t_trc, M_mm, t_grid)

    # Center by pelvis (midpoint of left/right hip)
    log_step("Centering both series by pelvis midpoint (left/right hip)")
    name2idx_joint = {jn: i for i, jn in enumerate(joints)}
    P_c = center_by_pelvis(P_100, name2idx_joint, "left_hip", "right_hip")
    M_c = center_by_pelvis(M_100, name2idx_joint, "left_hip", "right_hip")

    # Kabsch rotation across valid frames
    log_step("Estimating single best-fit rotation (Kabsch)")
    mask = ~np.isnan(P_c).any(axis=(1, 2)) & ~np.isnan(M_c).any(axis=(1, 2))
    if mask.sum() < 10:
        raise SystemExit("[ERROR] Too few valid frames after resampling and centering.")
    X = P_c[mask].reshape((-1, 3))
    Y = M_c[mask].reshape((-1, 3))
    X -= X.mean(axis=0, keepdims=True)
    Y -= Y.mean(axis=0, keepdims=True)
    R = kabsch_rotation(X, Y)
    log_info(f"Rotation det(R)     : {np.linalg.det(R):+.6f}")

    # Errors
    log_step("Computing errors (per-joint MPJPE & RMSE, overall MPJPE & RMSE)")
    P_rot = (P_c @ R.T)
    diff = P_rot - M_c                # (T,K,3)
    dists = np.linalg.norm(diff, axis=-1)  # (T,K)
    valid = ~np.isnan(dists)

    mpjpe_per_joint = {}
    rmse_per_joint = {}
    for k, jn in enumerate(joints):
        dk = dists[:, k]
        dk = dk[~np.isnan(dk)]
        if dk.size:
            mpjpe_per_joint[jn] = float(np.mean(dk))
            rmse_per_joint[jn]  = float(np.sqrt(np.mean(dk**2)))
        else:
            mpjpe_per_joint[jn] = float("nan")
            rmse_per_joint[jn]  = float("nan")

    overall_mpjpe = float(np.nanmean(dists[valid]))
    overall_rmse  = float(np.sqrt(np.nanmean((dists[valid]**2))))

    # Log summary
    log_info("Per-joint metrics (mm):")
    for jn in joints:
        log_info(f"  {jn:>15s}  MPJPE={mpjpe_per_joint[jn]:7.2f}  RMSE={rmse_per_joint[jn]:7.2f}")
    log_info(f"OVERALL MPJPE (mm)  : {overall_mpjpe:.2f}")
    log_info(f"OVERALL RMSE  (mm)  : {overall_rmse:.2f}")
    log_info(f"Frames used (valid) : {int(valid.sum())} of {valid.size}")

    # Save report
    log_step("Writing report.json")
    report = {
        "trial_id": args.trial,
        "preds_path": str(preds_path),
        "metric_source": used_source,
        "time_offset_sec": float(t_offset),
        "rotation_3x3": R,
        "fps_trc": float(trc_rate),
        "frames_used": int(valid.sum()),
        "joints_order": joints,
        "mocap_markers_used": used_markers,
        "mpjpe_per_joint_mm": mpjpe_per_joint,
        "rmse_per_joint_mm": rmse_per_joint,
        "overall_mpjpe_mm": overall_mpjpe,
        "overall_rmse_mm": overall_rmse,
        "joint_set": args.joint_set,
    }
    report_path = eval_dir / "report.json"
    report_path.write_text(json.dumps(jsonable(report), indent=2), encoding="utf-8")
    log_done(f"Report saved        : {report_path}")

    # Optional aligned series
    if args.save_series:
        out_npz = eval_dir / "aligned_100hz.npz"
        np.savez_compressed(
            out_npz,
            t_100=t_grid,
            joints_order=np.array(joints, dtype=object),
            mocap_mm=M_c,
            pred_aligned_mm=P_rot
        )
        log_info(f"Aligned series saved: {out_npz}")

    log_done(f"OVERALL MPJPE={overall_mpjpe:.2f} mm | OVERALL RMSE={overall_rmse:.2f} mm")

if __name__ == "__main__":
    main()
