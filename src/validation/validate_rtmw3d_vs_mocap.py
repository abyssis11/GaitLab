#!/usr/bin/env python3
# validate_rtmw3d_vs_mocap.py
"""
Validate RTMW3D vs MoCap (TRC @ 100 Hz, mm).

Includes:
- Metrics: MPJPE (rotation-only), N-MPJPE (scale-only), PA-MPJPE (scale+rotation), and T-only (translation only).
- --eval-on {grid,pred,trc}: evaluate on a 100 Hz grid, prediction timestamps, or TRC timestamps.
- --mocap-joint-centers: compute knee/ankle joint centers from medial/lateral markers when available.
- Optional evaluation from raw, root-relative predictions: --use-raw-preds and --force-raw-preds.
- YAML/JSON/NPY/NPZ parsing for a mocap->video rotation **and translation**; rotation is always used, translation is applied only when centering is disabled.
- Pelvis centering per-frame (translation removed) by default; can be disabled with --center none.
- Identity control (--person-index) and time-offset estimation (off by default; enable with --estimate-offset).
- Optional export of the sequence-specific Kabsch rotation: --export-rotation <path> (YAML).

Usage examples:

Pelvis-centered (translation-invariant, default):
python src/validation/validate_rtmw3d_vs_mocap.py \
  -m manifests/OpenCapDataset/subject2.yaml \
  -p config/paths.yaml \
  --trial walking1 \
  --joint-set ocap20 \
  --save-series

Absolute/global evaluation (uses R and, if available, t from extrinsics; no pelvis-centering):
python src/validation/validate_rtmw3d_vs_mocap.py \
  -m manifests/OpenCapDataset/subject2.yaml \
  -p config/paths.yaml \
  --trial walking1 \
  --joint-set basic \
  --center none
"""

import argparse
import json
from pathlib import Path
import numpy as np

# Adjust import path to your project as needed
from IO.load_manifest import load_manifest  # noqa: E402


# ---------------------------
# Logging helpers
# ---------------------------

def log_step(msg): print(f"[STEP] {msg}")
def log_info(msg): print(f"[INFO] {msg}")
def log_warn(msg): print(f"[WARN] {msg}")
def log_done(msg): print(f"[DONE] {msg}")


# ---------------------------
# Utils
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
# RTMW3D reader
# ---------------------------

def read_rtmw3d_preds(preds_path: Path, expect_k=None, person_index=0, min_mean_score=0.0, allow_raw=False, prefer_raw=False):
    """
    Return:
        times (F,),
        P (F,K,3) in:
            - millimetres if metric present (or XYZ units=mm),
            - raw (unitless) if allow_raw/prefer_raw forces plain xyz,
        used_source: str,
        is_metric_mm: bool

    Priority:
      1) If prefer_raw and 'keypoints_xyz' present -> choose raw (unitless unless units=mm).
      2) Else use 'keypoints_xyz_mm' if present.
      3) Else if ('keypoints_xyz' + 'keypoints_xyz_units' == 'mm') -> treat as mm.
      4) Else if allow_raw -> use 'keypoints_xyz' (unitless), is_metric_mm=False.
      5) Else -> (None, None, None, None)
    """
    times = []
    P = []
    used_source = None
    is_metric_mm = None

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

            kps = None
            # Prefer raw if asked
            if prefer_raw and ('keypoints_xyz' in p):
                kps = np.asarray(p['keypoints_xyz'], dtype=float)
                units = p.get('keypoints_xyz_units', '').lower()
                if units == 'mm':
                    used_source = 'keypoints_xyz (units=mm)'
                    is_metric_mm = True
                else:
                    used_source = 'keypoints_xyz (RAW units)'
                    is_metric_mm = False
            elif 'keypoints_xyz_mm' in p:
                kps = np.asarray(p['keypoints_xyz_mm'], dtype=float)
                used_source = 'keypoints_xyz_mm'
                is_metric_mm = True
            elif 'keypoints_xyz' in p:
                kps = np.asarray(p['keypoints_xyz'], dtype=float)
                units = p.get('keypoints_xyz_units', '').lower()
                if units == 'mm':
                    used_source = 'keypoints_xyz (units=mm)'
                    is_metric_mm = True
                elif allow_raw:
                    used_source = 'keypoints_xyz (RAW units)'
                    is_metric_mm = False
                else:
                    kps = None

            if kps is None:
                continue

            if kps.ndim != 2 or kps.shape[1] != 3:
                continue
            if expect_k is not None and kps.shape[0] != expect_k:
                continue

            times.append(float(o.get('time_sec', 0.0)))
            P.append(kps)

    if not times:
        return None, None, None, None
    return np.asarray(times, dtype=float), np.asarray(P, dtype=float), used_source, bool(is_metric_mm)


# ---------------------------
# Resampling & alignment helpers
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

def umeyama_similarity(X, Y):
    """Return (s, R) minimizing || s * R @ X - Y ||_F. X, Y are (N,3) centered."""
    H = (X.T @ Y) / X.shape[0]
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    varX = (X**2).sum() / X.shape[0]
    s = float(S.sum() / (varX + 1e-12))
    return s, R

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

def extract_mocap_stack_basic(t_trc, markers, joint_map, use_joint_centers=False):
    pick = build_mocap_selector(markers)
    used_markers = []
    joints = []
    arrays = []  # list of (T,3)

    def mid_by_names(names_a, names_b):
        a = None; b = None
        for na in names_a:
            n = pick(na)
            if n and n in markers:
                a = markers[n]; break
        for nb in names_b:
            n = pick(nb)
            if n and n in markers:
                b = markers[n]; break
        if a is not None and b is not None:
            return 0.5*(a + b), (n, n)
        return None, (None, None)

    kneeL_med = ['L_knee_med','L_KneeMed','L_knee_m','L_MFC','L_FemCon_Med']
    kneeL_lat = ['L_knee_lat','L_KneeLat','L_knee_l','L_LFC','L_FemCon_Lat']
    kneeR_med = ['R_knee_med','r_knee_med','R_KneeMed','R_MFC','R_FemCon_Med']
    kneeR_lat = ['R_knee_lat','r_knee_lat','R_KneeLat','R_LFC','R_FemCon_Lat']

    ankL_med  = ['L_ankle_med','L_AnkleMed','L_MedMalleolus','L_med_malleolus','L_med_mall','L_MedMall']
    ankL_lat  = ['L_ankle_lat','L_AnkleLat','L_LatMalleolus','L_lat_malleolus','L_lat_mall','L_LatMall']
    ankR_med  = ['R_ankle_med','r_ankle_med','R_AnkleMed','R_MedMalleolus','R_med_malleolus','R_med_mall','R_MedMall']
    ankR_lat  = ['R_ankle_lat','r_ankle_lat','R_AnkleLat','R_LatMalleolus','R_lat_malleolus','R_lat_mall','R_LatMall']

    for h3wb_name, candidates in joint_map:
        arr = None
        mark_info = None

        if use_joint_centers and h3wb_name == 'left_knee':
            arr, _ = mid_by_names(kneeL_med, kneeL_lat)
            if arr is not None: mark_info = 'mid(L_knee_med,L_knee_lat)'
        if arr is None and use_joint_centers and h3wb_name == 'right_knee':
            arr, _ = mid_by_names(kneeR_med, kneeR_lat)
            if arr is not None: mark_info = 'mid(R_knee_med,R_knee_lat)'
        if arr is None and use_joint_centers and h3wb_name == 'left_ankle':
            arr, _ = mid_by_names(ankL_med, ankL_lat)
            if arr is not None: mark_info = 'mid(L_ankle_med,L_ankle_lat)'
        if arr is None and use_joint_centers and h3wb_name == 'right_ankle':
            arr, _ = mid_by_names(ankR_med, ankR_lat)
            if arr is not None: mark_info = 'mid(R_ankle_med,R_ankle_lat)'

        if arr is None:
            mk = None
            for c in candidates:
                cand = pick(c)
                if cand is not None and cand in markers:
                    mk = cand; break
            if mk is not None:
                arr = markers[mk]
                mark_info = mk

        if arr is not None:
            joints.append(h3wb_name)
            used_markers.append(mark_info if mark_info else "unknown")
            arrays.append(arr)

    if not arrays:
        raise RuntimeError("None of the requested MoCap markers found (basic set).")

    T = len(t_trc)
    K = len(arrays)
    M = np.full((T, K, 3), np.nan, dtype=float)
    for j, A in enumerate(arrays):
        M[:, j, :] = A
    return joints, used_markers, M

def ocap20_from_trc(t_trc, markers, use_joint_centers=False):
    pick = build_mocap_selector(markers)

    def mid(A, B):
        return 0.5*(A + B)

    def get(name_or_none):
        if name_or_none is None:
            return None
        return markers[name_or_none] if name_or_none in markers else None

    # Shoulder / HJC names
    r_sh_name = pick('R_Shoulder','R_Shoulder_Acromion','r_shoulder')
    l_sh_name = pick('L_Shoulder','L_Shoulder_Acromion','l_shoulder')
    r_hjc_name = pick('R_HJC','R_HJC_reg','R_HJCR','r_hjc')
    l_hjc_name = pick('L_HJC','L_HJC_reg','L_HJCL','l_hjc')

    # Knees and ankles (single markers)
    r_knee_name = pick('r_knee','R_knee','R_Knee')
    l_knee_name = pick('L_knee','l_knee','L_Knee')
    r_ank_name  = pick('r_ankle','R_ankle','R_Ankle')
    l_ank_name  = pick('L_ankle','l_ankle','L_Ankle')

    # Heels / toes
    r_heel_name = pick('r_calc','R_calc','R_heel','R_Calc')
    l_heel_name = pick('L_calc','l_calc','L_heel','L_Calc')
    r_small_name = pick('r_5meta','R_5meta','R_small_toe','R_5th_met')
    l_small_name = pick('L_5meta','l_5meta','L_small_toe','L_5th_met')
    r_big_name   = pick('r_toe','R_toe','R_big_toe','R_1st_met')
    l_big_name   = pick('L_toe','l_toe','L_big_toe','L_1st_met')

    # Elbows and wrists via medial/lat when available
    r_el_med = get(pick('R_elbow_med','r_elbow_med')); r_el_lat = get(pick('R_elbow_lat','r_elbow_lat'))
    l_el_med = get(pick('L_elbow_med','l_elbow_med')); l_el_lat = get(pick('L_elbow_lat','l_elbow_lat'))
    if r_el_med is not None and r_el_lat is not None: r_el = mid(r_el_med, r_el_lat)
    else: r_el = get(pick('R_elbow','r_elbow','R_elbow_lat','R_elbow_med'))
    if l_el_med is not None and l_el_lat is not None: l_el = mid(l_el_med, l_el_lat)
    else: l_el = get(pick('L_elbow','l_elbow','L_elbow_lat','L_elbow_med'))

    r_wr_rad = get(pick('R_wrist_radius','r_wrist_radius')); r_wr_uln = get(pick('R_wrist_ulna','r_wrist_ulna'))
    l_wr_rad = get(pick('L_wrist_radius','l_wrist_radius')); l_wr_uln = get(pick('L_wrist_ulna','l_wrist_ulna'))
    if r_wr_rad is not None and r_wr_uln is not None: r_wr = mid(r_wr_rad, r_wr_uln)
    else: r_wr = get(pick('R_wrist','r_wrist'))
    if l_wr_rad is not None and l_wr_uln is not None: l_wr = mid(l_wr_rad, l_wr_uln)
    else: l_wr = get(pick('L_wrist','l_wrist'))

    # Knee/ankle joint centers when requested
    def mid_by_names(names_a, names_b):
        A = None; B = None
        for na in names_a:
            nm = pick(na)
            if nm and nm in markers: A = markers[nm]; break
        for nb in names_b:
            nm = pick(nb)
            if nm and nm in markers: B = markers[nm]; break
        if A is not None and B is not None: return 0.5*(A + B)
        return None

    if use_joint_centers:
        kneeL_med = ['L_knee_med','L_KneeMed','L_knee_m','L_MFC','L_FemCon_Med']
        kneeL_lat = ['L_knee_lat','L_KneeLat','L_knee_l','L_LFC','L_FemCon_Lat']
        kneeR_med = ['R_knee_med','r_knee_med','R_KneeMed','R_MFC','R_FemCon_Med']
        kneeR_lat = ['R_knee_lat','r_knee_lat','R_KneeLat','R_LFC','R_FemCon_Lat']

        ankL_med  = ['L_ankle_med','L_AnkleMed','L_MedMalleolus','L_med_malleolus','L_med_mall','L_MedMall']
        ankL_lat  = ['L_ankle_lat','L_AnkleLat','L_LatMalleolus','L_lat_malleolus','L_lat_mall','L_LatMall']
        ankR_med  = ['R_ankle_med','r_ankle_med','R_AnkleMed','R_MedMalleolus','R_med_malleolus','R_med_mall','R_MedMall']
        ankR_lat  = ['R_ankle_lat','r_ankle_lat','R_AnkleLat','R_LatMalleolus','R_lat_malleolus','R_lat_mall','R_LatMall']

        Lk = mid_by_names(kneeL_med, kneeL_lat); Rk = mid_by_names(kneeR_med, kneeR_lat)
        La = mid_by_names(ankL_med, ankL_lat);   Ra = mid_by_names(ankR_med, ankR_lat)
        if Lk is not None: l_knee = Lk
        else: l_knee = get(l_knee_name)
        if Rk is not None: r_knee = Rk
        else: r_knee = get(r_knee_name)
        if La is not None: l_ank = La
        else: l_ank = get(l_ank_name)
        if Ra is not None: r_ank = Ra
        else: r_ank = get(r_ank_name)
    else:
        l_knee = get(l_knee_name); r_knee = get(r_knee_name)
        l_ank  = get(l_ank_name);  r_ank  = get(r_ank_name)

    l_sh = get(l_sh_name); r_sh = get(r_sh_name)
    l_hjc = get(l_hjc_name); r_hjc = get(r_hjc_name)
    neck   = 0.5*(l_sh + r_sh) if (l_sh is not None and r_sh is not None) else None
    midhip = 0.5*(l_hjc + r_hjc) if (l_hjc is not None and r_hjc is not None) else None

    parts = [
        neck, midhip,
        get(l_sh_name), get(r_sh_name),
        l_el, r_el, l_wr, r_wr,
        get(l_hjc_name), get(r_hjc_name),
        l_knee, r_knee,
        l_ank,  r_ank,
        get(l_heel_name), get(r_heel_name),
        get(l_small_name), get(r_small_name),
        get(l_big_name),   get(r_big_name),
    ]

    if any(p is None for p in parts):
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
        i = name2idx.get(n, -1)
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
        return None, None
    P = np.stack(parts, axis=1)  # (F,20,3)
    return P, list(OPENCAP20)


# ---------------------------
# Extrinsics loader (rotation + optional translation)
# ---------------------------

def try_load_extrinsics(path: Path):
    """
    Try to load a 3x3 rotation and optional translation vector from various file types.
    Returns (R, t) where t can be None.
    Supported keys for rotation: R, rotation, R_fromMocap_toVideo, matrix
    Supported keys for translation: position_fromMocapOrigin_toVideoOrigin, t_fromMocap_toVideo, t, translation
    """
    import json as _json, numpy as _np, re
    pth = Path(path)
    if not pth.exists():
        return None, None

    def _validate_R(R):
        if R is None: return None
        R = _np.asarray(R, dtype=float)
        if R.shape != (3,3):
            return None
        I = _np.eye(3)
        if not _np.allclose(R.T @ R, I, atol=1e-3):
            return None
        det = float(_np.linalg.det(R))
        if abs(det) < 1e-3:
            return None
        if det < 0:
            # try to fix reflection by flipping last column
            R[:, -1] *= -1.0
            if not _np.allclose(R.T @ R, I, atol=1e-3) or abs(float(_np.linalg.det(R)) - 1.0) > 1e-2:
                return None
        return R

    def _maybe_vec3(x):
        try:
            arr = _np.array(x, dtype=float).reshape(-1)
            if arr.size >= 3:
                return arr[:3]
        except Exception:
            pass
        return None

    # PICKLE / PKL
    try:
        if pth.suffix.lower() in ('.pickle', '.pkl'):
            import pickle as _pickle
            with open(pth, 'rb') as fh:
                obj = _pickle.load(fh)
            if isinstance(obj, dict):
                R = None; t = None
                for key in ('rotation','R','R_fromMocap_toVideo','matrix','rotation_matrix'):
                    if key in obj:
                        M = _np.array(obj[key], dtype=float)
                        if M.size == 16: M = M.reshape(4,4)
                        if M.size == 9:  M = M.reshape(3,3)
                        if M.shape == (4,4): M = M[:3,:3]
                        R = _validate_R(M)
                        if R is not None: break
                for tk in ('translation','t','t_fromMocap_toVideo','position_fromMocapOrigin_toVideoOrigin','camera_center','C_w'):
                    if tk in obj:
                        t = _maybe_vec3(obj[tk])
                        break
                # simple unit guard: if |t| < 5 and not ~0, assume meters -> convert to mm
                if t is not None:
                    nrm = float(_np.linalg.norm(t))
                    if nrm > 1e-6 and nrm < 5.0:
                        t = t * 1000.0
                if R is not None:
                    return R, t
    except Exception:
        pass

    # YAML (with or without PyYAML)
    try:
        if pth.suffix.lower() in ('.yml', '.yaml'):
            try:
                import yaml  # type: ignore
                obj = yaml.safe_load(pth.read_text(encoding='utf-8'))
            except Exception:
                obj = None

            if isinstance(obj, dict):
                R = None; t = None
                for key in ('R','rotation','R_fromMocap_toVideo','matrix'):
                    if key in obj:
                        M = np.array(obj[key], dtype=float)
                        if M.size == 16: M = M.reshape(4,4)
                        if M.size == 9:  M = M.reshape(3,3)
                        if M.shape == (4,4): M = M[:3,:3]
                        R = _validate_R(M)
                        if R is not None: break
                for tk in ('position_fromMocapOrigin_toVideoOrigin','t_fromMocap_toVideo','t','translation'):
                    if tk in obj:
                        t = _maybe_vec3(obj[tk])
                        break
                if R is not None:
                    return R, t

            # minimal fallback parser
            txt = pth.read_text(encoding='utf-8')
            def _grab_block(name):
                pat = rf"^{name}\s*:\s*\n((?:[ \t].*\n)+)"
                m = re.search(pat, txt, re.MULTILINE)
                return m.group(1) if m else None
            blk = _grab_block('R_fromMocap_toVideo') or _grab_block('rotation') or _grab_block('R')
            if blk:
                rows = []
                for line in blk.splitlines():
                    line = line.strip()
                    if not line.startswith('-'): 
                        continue
                    nums = [float(s) for s in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]
                    if len(nums) == 3:
                        rows.append(nums)
                if len(rows) == 3 and all(len(r)==3 for r in rows):
                    R = _validate_R(rows)
                    if R is not None:
                        # try to parse translation list right under any known key
                        t = None
                        t_blk = _grab_block('position_fromMocapOrigin_toVideoOrigin') or _grab_block('t_fromMocap_toVideo') or _grab_block('t') or _grab_block('translation')
                        if t_blk:
                            nums = [float(s) for s in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", t_blk)]
                            if len(nums) >= 3:
                                t = np.array(nums[:3], dtype=float)
                        return R, t
    except Exception:
        pass

    return None, None


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Validate RTMW3D vs MoCap (TRC @ 100 Hz, mm)")
    ap.add_argument("-m", "--manifest", required=True)
    ap.add_argument("-p", "--paths", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--outputs-root", default=None)

    # Options
    ap.add_argument("--person-index", type=int, default=0,
                    help="Which person stream to evaluate (default=0). Use -1 for max mean_score per frame (may cause identity switches).")
    ap.add_argument("--min-mean-score", type=float, default=0.0)
    ap.add_argument("--estimate-offset", action="store_true", default=False,
                    help="Estimate small time offset between streams (OFF by default).")
    ap.add_argument("--max-offset", type=float, default=1.0, help="max |offset| (s) for xcorr")
    ap.add_argument("--save-series", action="store_true", help="save aligned arrays to npz")
    ap.add_argument("--joint-set", choices=["basic","ocap20"], default="basic",
                    help="Which joint set to compare: 'basic' or 'ocap20' (OpenCap-20)")
    ap.add_argument("--eval-on", choices=["grid","pred","trc"], default="grid",
                    help="Evaluation timebase: 'grid'=100Hz (default), 'pred'=use prediction timestamps, 'trc'=use MoCap timestamps.")
    ap.add_argument("--mocap-joint-centers", action="store_true",
                    help="Use joint centers for knees/ankles when medial+lat markers exist (midpoint).")
    ap.add_argument("--center", choices=["pelvis","none"], default="pelvis",
                    help="Pelvis-centering mode. 'pelvis' subtracts pelvis per frame (translation-invariant). 'none' keeps absolute positions and applies translation if available.")

    # Predictions options
    ap.add_argument("--extrinsics", type=str, default=None,
                    help="Path to extrinsics file (yaml/json/npy/npz/pickle). Overrides meta calibration if provided.")
    ap.add_argument("--extrinsics-from-camera-center", action="store_true", default=False,
                    help="Interpret extrinsics file's 'translation' as camera center C_w and convert to OpenCV t_cw = -R*C_w.")
    ap.add_argument("--use-raw-preds", action="store_true",
                    help="Allow evaluating from raw 'keypoints_xyz' if metric mm not available (metrics rely on similarity alignment).")
    ap.add_argument("--force-raw-preds", action="store_true",
                    help="Prefer raw 'keypoints_xyz' even if metric exists (for paper-style root-relative checks).")

    # Export
    ap.add_argument("--export-rotation", type=str, default=None,
                    help="Path to save the estimated Kabsch rotation as YAML (R_fromMocap_toVideo).")

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

    # Load meta for names & possible extrinsics path
    log_step("Reading meta.json for keypoint names")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    kp_names = meta.get("keypoint_names") or []
    if not kp_names:
        raise SystemExit("[ERROR] meta.keypoint_names missing.")
    log_info(f"Keypoints K         : {len(kp_names)}")

    R_mocap_to_video = None
    # Attempt to load extrinsics from CLI first
    if args.extrinsics:
        cand = Path(args.extrinsics)
        R_try, t_try = try_load_extrinsics(cand)
        if R_try is not None:
            R_mocap_to_video = R_try
            t_mocap_to_video = t_try
            log_info(f"Loaded mocap->video extrinsics from: {cand}")
        else:
            log_warn(f"Failed to parse extrinsics from: {cand}")

    # If still not set, attempt to load mocap->video extrinsics path from meta
    if R_mocap_to_video is None:
        calib = meta.get('calibration', {}) if isinstance(meta, dict) else {}
        rot_candidates = []
        if isinstance(calib, dict):
            for key in ('mocap_to_video', 'mocap2video', 'world_to_camera', 'mocap_to_cam', 'mocapToVideoTransform'):
                val = calib.get(key)
                if isinstance(val, str) and val:
                    rot_candidates.append(val)
        if rot_candidates:
            for rel in rot_candidates:
                for base in (trial_root, meta_path.parent, Path('.')):
                    cand = (base / rel) if not Path(rel).is_absolute() else Path(rel)
                    R_try, t_try = try_load_extrinsics(cand)
                    if R_try is not None:
                        R_mocap_to_video = R_try
                        t_mocap_to_video = t_try
                        log_info(f"Loaded mocap->video rotation from: {cand}")
                        if t_try is not None:
                            log_info(f"Also found translation (video frame): [{t_try[0]:.3f}, {t_try[1]:.3f}, {t_try[2]:.3f}]")
                        break
                if R_mocap_to_video is not None:
                    break
        if R_mocap_to_video is None:
            log_warn(f"Could not load mocap->video rotation from any of: {rot_candidates}")

    # Load predictions
    log_step(f"Reading RTMW3D predictions from {preds_path}")
    t_pred, P_full, used_source, is_metric_mm = read_rtmw3d_preds(
        preds_path, expect_k=len(kp_names),
        person_index=args.person_index,
        min_mean_score=args.min_mean_score,
        allow_raw=getattr(args, 'use_raw_preds', False),
        prefer_raw=getattr(args, 'force_raw_preds', False)
    )
    if t_pred is None or P_full is None:
        if args.use_raw_preds or args.force_raw_preds:
            raise SystemExit("[EXIT] Aborting: no accepted predictions found (even with raw enabled).")
        else:
            raise SystemExit("[EXIT] Aborting: no metric predictions found. Try --use-raw-preds.")
    log_info(f"Frames read (preds) : {len(t_pred)}")
    log_info(f"Pred source         : {used_source}")
    log_info(f"Pred units metric?  : {is_metric_mm}")
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
        joints, used_markers, M_mm = extract_mocap_stack_basic(t_trc, markers, BASIC_JOINTS, use_joint_centers=getattr(args, 'mocap_joint_centers', False))
        name2idx = {n:i for i,n in enumerate(kp_names)}
        pred_idx = [name2idx[jn] for jn in joints]
        P_sel = P_full[:, pred_idx, :]  # (F,K,3)
    else:
        log_step("Building OpenCap-20 joints (neck/mid_hip composed)")
        M_mm, joints = ocap20_from_trc(t_trc, markers, use_joint_centers=getattr(args, 'mocap_joint_centers', False))
        if M_mm is None:
            raise SystemExit("[ERROR] Could not assemble OpenCap-20 from TRC (missing markers).")
        P_sel, _ = ocap20_from_preds_full(P_full, kp_names)
        if P_sel is None:
            raise SystemExit("[ERROR] Could not assemble OpenCap-20 from RTMW3D (missing keypoints).")
        used_markers = ["composed_or_multi"] * len(joints)

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
        sig_pred = seg_len(P_sel, jLhip, jLank) + seg_len(P_sel, jRhip, jRank)
        sig_trc  = seg_len(M_mm,  jLhip, jLank) + seg_len(M_mm,  jRhip, jRank)
    except ValueError:
        sig_pred = np.zeros(P_sel.shape[0])
        sig_trc  = np.zeros(M_mm.shape[0])

    t_offset = 0.0
    if args.estimate_offset:
        log_step("Estimating time offset via cross-correlation")
        t_offset = estimate_time_offset(t_pred, sig_pred, t_trc, sig_trc, max_offset=args.max_offset, fs=100.0)
        log_info(f"Estimated offset    : {t_offset:+.3f} s (to add to predictions)")
    else:
        log_info("Offset estimation   : disabled (using 0.000 s)")

    # ---------- Choose evaluation timebase ----------
    if args.eval_on == "grid":
        log_step("Resampling both modalities to 100 Hz common grid")
        t0 = max(t_trc.min(), t_pred.min() + t_offset)
        t1 = min(t_trc.max(), t_pred.max() + t_offset)
        if t1 - t0 < 0.5:
            raise SystemExit("[ERROR] Insufficient temporal overlap after offset.")
        t_eval = np.arange(t0, t1, 0.01)  # 100 Hz
        log_info(f"Common time range   : {t0:.3f}..{t1:.3f} s")
        log_info(f"Frames (grid@100Hz) : {len(t_eval)}")
        P_eval = resample_to_grid(t_pred + t_offset, P_sel, t_eval)
        M_eval = resample_to_grid(t_trc, M_mm, t_eval)
    elif args.eval_on == "pred":
        log_step("Evaluating on prediction timestamps (resample MoCap to predictions)")
        t_eval_full = t_pred + t_offset
        mask = (t_eval_full >= t_trc.min()) & (t_eval_full <= t_trc.max())
        if mask.sum() < 5:
            raise SystemExit("[ERROR] Too few overlapping prediction timestamps.")
        t_eval = t_eval_full[mask]
        P_eval = P_sel[mask]
        M_eval = resample_to_grid(t_trc, M_mm, t_eval)
        log_info(f"Frames (pred times) : {len(t_eval)}")
    else:  # args.eval_on == "trc"
        log_step("Evaluating on MoCap timestamps (resample predictions to MoCap)")
        t_eval = t_trc.copy()
        P_eval = resample_to_grid(t_pred + t_offset, P_sel, t_eval)
        M_eval = M_mm
        log_info(f"Frames (trc times)  : {len(t_eval)}")

    # Apply mocap->video extrinsics if available (before centering)
    if R_mocap_to_video is not None:
        M_eval = M_eval @ R_mocap_to_video.T
        if args.center == "none":
            if t_mocap_to_video is not None:
                M_eval = M_eval + t_mocap_to_video.reshape(1,1,3)
            else:
                log_warn("No translation found in extrinsics; absolute evaluation will only use rotation.")

    # Center by pelvis (midpoint of left/right hip) unless disabled
    if args.center == "pelvis":
        log_step("Centering both series by pelvis midpoint (left/right hip)")
        name2idx_joint = {jn: i for i, jn in enumerate(joints)}
        P_c = center_by_pelvis(P_eval, name2idx_joint, "left_hip", "right_hip")
        M_c = center_by_pelvis(M_eval, name2idx_joint, "left_hip", "right_hip")
    else:
        log_step("Centering disabled (absolute evaluation in global coordinates)")
        P_c = P_eval.copy()
        M_c = M_eval.copy()

    # Build flattened stacks (drop NaNs)
    mask_valid = ~np.isnan(P_c).any(axis=(1, 2)) & ~np.isnan(M_c).any(axis=(1, 2))
    if mask_valid.sum() < 10:
        raise SystemExit("[ERROR] Too few valid frames after resampling and centering.")
    X = P_c[mask_valid].reshape((-1, 3))
    Y = M_c[mask_valid].reshape((-1, 3))
    X -= X.mean(axis=0, keepdims=True)
    Y -= Y.mean(axis=0, keepdims=True)

    # --- Translation-only (as-is w.r.t centering mode) ---
    P_tonly = P_c.copy()

    # --- Rotation only (Kabsch) ---
    log_step("Estimating best-fit rotation (Kabsch) and similarity (Umeyama)")
    R = kabsch_rotation(X, Y)

    # --- Similarity (scale + rotation) ---
    s_sim, R_sim = umeyama_similarity(X, Y)
    detR = float(np.linalg.det(R))
    detR_sim = float(np.linalg.det(R_sim))
    log_info(f"Rotation det(R)     : {detR:+.6f}")
    log_info(f"Similarity det(Rsim): {detR_sim:+.6f}")
    log_info(f"Isotropic scale     : {s_sim:.6f}")

    # Build aligned prediction variants
    P_rot_only   = (P_c @ R.T)                  # rotation only
    P_scale_only = (s_sim * P_c)               # scale only
    P_pa         = (s_sim * (P_c @ R_sim.T))   # similarity (scale + rotation)

    # Error helpers
    def mpjpe_mm(P, M):
        d = np.linalg.norm(P - M, axis=-1)  # (T,K)
        return float(np.nanmean(d))

    def per_joint_stats(P, M, joints):
        d = np.linalg.norm(P - M, axis=-1)  # (T,K)
        mpjpe = {}
        rmse  = {}
        for k, jn in enumerate(joints):
            dk = d[:, k]
            dk = dk[~np.isnan(dk)]
            if dk.size:
                mpjpe[jn] = float(np.mean(dk))
                rmse[jn]  = float(np.sqrt(np.mean(dk**2)))
            else:
                mpjpe[jn] = float("nan")
                rmse[jn]  = float("nan")
        return mpjpe, rmse, d

    # Per-joint (keep rotation-only for continuity) and overall metrics
    log_step("Computing metrics")
    mpjpe_per_joint_rot, rmse_per_joint_rot, d_rot = per_joint_stats(P_rot_only, M_c, joints)
    overall_mpjpe_rot = float(np.nanmean(d_rot))
    overall_rmse_rot  = float(np.sqrt(np.nanmean(d_rot**2)))

    overall_tonly    = mpjpe_mm(P_tonly,    M_c)
    overall_nmpjpe   = mpjpe_mm(P_scale_only, M_c)
    overall_pampjpe  = mpjpe_mm(P_pa,        M_c)

    # Log summary
    log_info("Per-joint metrics (rotation-only) [mm]:")
    for jn in joints:
        log_info(f"  {jn:>15s}  MPJPE={mpjpe_per_joint_rot[jn]:7.2f}  RMSE={rmse_per_joint_rot[jn]:7.2f}")
    log_info(f"OVERALL MPJPE (rot-only) [mm] : {overall_mpjpe_rot:.2f}")
    log_info(f"OVERALL RMSE  (rot-only) [mm] : {overall_rmse_rot:.2f}")
    log_info(f"OVERALL T-only MPJPE    [mm]  : {overall_tonly:.2f}")
    log_info(f"OVERALL N-MPJPE         [mm]  : {overall_nmpjpe:.2f}")
    log_info(f"OVERALL PA-MPJPE        [mm]  : {overall_pampjpe:.2f}")

    # Optional export of the estimated Kabsch rotation (sequence-specific)
    if args.export_rotation:
        try:
            outp = Path(args.export_rotation)
            outp.parent.mkdir(parents=True, exist_ok=True)
            data = {"R_fromMocap_toVideo": R.tolist()}
            try:
                import yaml  # type: ignore
                outp.write_text(yaml.safe_dump(data), encoding="utf-8")
            except Exception:
                def _row(r): return f"- [{r[0]:.9f}, {r[1]:.9f}, {r[2]:.9f}]"
                txt = "R_fromMocap_toVideo:\n" + "\n".join("  "+_row(r) for r in R.tolist()) + "\n"
                outp.write_text(txt, encoding="utf-8")
            log_done(f"Exported estimated Kabsch rotation to: {outp}")
        except Exception as e:
            log_warn(f"Could not export rotation: {e}")

    # Save report
    log_step("Writing report.json")
    report = {
        "trial_id": args.trial,
        "preds_path": str(preds_path),
        "pred_source": used_source,
        "pred_units_metric_mm": bool(is_metric_mm),
        "time_offset_sec": float(t_offset),
        "rotation_3x3": R,
        "similarity_rotation_3x3": R_sim,
        "similarity_isotropic_scale": float(s_sim),
        "fps_trc": float(trc_rate),
        "joints_order": joints,
        "mocap_markers_used": used_markers,
        "per_joint_rot_only_mpjpe_mm": mpjpe_per_joint_rot,
        "per_joint_rot_only_rmse_mm": rmse_per_joint_rot,
        "overall_mpjpe_rot_only_mm": overall_mpjpe_rot,
        "overall_rmse_rot_only_mm": overall_rmse_rot,
        "overall_t_only_mpjpe_mm": overall_tonly,
        "overall_n_mpjpe_mm": overall_nmpjpe,
        "overall_pa_mpjpe_mm": overall_pampjpe,
        "joint_set": args.joint_set,
        "eval_on": args.eval_on,
        "mocap_joint_centers": bool(args.mocap_joint_centers),
        "center_mode": args.center
    }
    report_path = eval_dir / "report.json"
    report_path.write_text(json.dumps(jsonable(report), indent=2), encoding="utf-8")
    log_done(f"Report saved        : {report_path}")

    # Optional aligned series
    if args.save_series:
        out_npz = eval_dir / "aligned_100hz.npz"
        np.savez_compressed(
            out_npz,
            t_eval=t_eval,
            joints_order=np.array(joints, dtype=object),
            mocap_mm=M_c,
            pred_aligned_rot_only_mm=P_rot_only,
            pred_aligned_pa_mm=P_pa,
            pred_tonly_mm=P_tonly
        )
        log_info(f"Aligned series saved: {out_npz}")

    log_done(
        "SUMMARY | center={} | MPJPE(rot-only)={:.2f} mm | T-only={:.2f} mm | N-MPJPE={:.2f} mm | PA-MPJPE={:.2f} mm | scale={:.6f}".format(
            args.center, overall_mpjpe_rot, overall_tonly, overall_nmpjpe, overall_pampjpe, s_sim
        )
    )


if __name__ == "__main__":
    main()
