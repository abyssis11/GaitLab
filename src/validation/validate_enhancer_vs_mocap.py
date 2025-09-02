#!/usr/bin/env python3
# validate_enhancer_vs_mocap.py
"""
Validate Marker-Enhancer predictions (mm) against MoCap TRC (mm) using the same
procedure as the RTMW3D validator (no auto-scaling).

What it does
------------
1) Loads manifest (IO.load_manifest.load_manifest)
2) Locates:
   - <trial>/enhancer/body_pred_mm_Tx35x3.npz  (from our enhancer script)
   - <trial>/enhancer/arms_pred_mm_Tx8x3.npz   (optional but recommended)
   - <trial>/enhancer/enh_input_60hz.npz       (to get the t_60 timeline)
   - <trial>/meta.json
   - trial['mocap_trc']                        (TRC @ 100 Hz, mm)
3) Compares either:
   A) --compare-set ocap20         → Build OpenCap-20 joints on both sides
   B) --compare-set enhancer_native→ Intersect native enhancer marker names with TRC names
4) (Optional) estimates time offset (±max_offset s), resamples both to 100 Hz
5) Pelvis-centres (or centroid if HJCs missing), single Kabsch rotation
   (optionally similarity/Kabsch with --allow-scale)
6) Reports per-point MPJPE & RMSE (mm) and overall MPJPE/RMSE
7) Saves:
   - <trial>/enhancer/report_enhancer.json
   - <trial>/enhancer/aligned_100hz_enh.npz (if --save-series)

Usage
-----
python src/validation/validate_enhancer_vs_mocap.py \
  -m manifests/OpenCapDataset/subject2.yaml \
  -p config/paths.yaml \
  --trial walking1 \
  --estimate-offset \
  --compare-set ocap20 \
  --save-series
"""

import argparse
import json
from pathlib import Path
import numpy as np
import re

from IO.load_manifest import load_manifest

# ---------------------------
# Logging
# ---------------------------
def log_step(msg): print(f"[STEP] {msg}")
def log_info(msg): print(f"[INFO] {msg}")
def log_warn(msg): print(f"[WARN] {msg}")
def log_done(msg): print(f"[DONE] {msg}")
def log_err(msg):  print(f"[ERROR] {msg}")

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
    txt = Path(path).read_text(encoding='utf-8', errors='ignore').strip().splitlines()
    if len(txt) < 6:
        raise RuntimeError(f"TRC file too short: {path}")
    vals = txt[2].split()
    if len(vals) < 4:
        raise RuntimeError("TRC header missing numeric line (#2).")
    data_rate = float(vals[0])  # expected ~100.0
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
# Resample & align
# ---------------------------
def resample_to_grid(t_src, X_src, t_grid):
    if X_src.ndim != 3:
        raise RuntimeError(f"Expected (T,K,3), got {X_src.shape}")
    T_src, K, _ = X_src.shape
    Y = np.full((len(t_grid), K, 3), np.nan, dtype=float)
    for k in range(K):
        for d in range(3):
            Y[:, k, d] = np.interp(t_grid, t_src, X_src[:, k, d], left=np.nan, right=np.nan)
    return Y

def estimate_time_offset(sig_a_t, sig_a, sig_b_t, sig_b, max_offset=1.0, fs=100.0):
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
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R

def kabsch_similarity(X, Y, allow_scale=False):
    """Return (scale, R). X,Y zero-mean (N,3)."""
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    if allow_scale:
        num = np.sum(S)
        den = np.sum(X**2)
        s = (num / (den + 1e-12))
    else:
        s = 1.0
    return s, R

def center_by_pelvis(X, name2idx, Lhip_name, Rhip_name):
    L = name2idx.get(Lhip_name, None)
    R = name2idx.get(Rhip_name, None)
    if L is None or R is None:
        return X
    Xc = X.copy()
    pelv = 0.5 * (X[..., L, :] + X[..., R, :])
    Xc -= pelv[..., None, :]
    return Xc

# ---------------------------
# OpenCap-20 set
# ---------------------------
OCAP20 = [
    'neck','mid_hip',
    'left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist',
    'left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle',
    'left_heel','right_heel','left_small_toe','right_small_toe','left_big_toe','right_big_toe',
]

def build_ci_lookup(names):
    return {str(n).lower(): i for i, n in enumerate(names)}

def pick_ci(ci_map, *cands):
    for c in cands:
        i = ci_map.get(str(c).lower())
        if i is not None:
            return i
    return None

def ocap20_from_trc(t_trc, markers):
    import numpy as _np
    lower_map = {k.lower(): k for k in markers.keys()}

    def get(*cands):
        for name in cands:
            k = lower_map.get(str(name).lower())
            if k is not None:
                return markers[k]  # (T,3)
        return None

    def mid(A, B):
        if A is None or B is None:
            return None
        return 0.5 * (A + B)

    # Shoulders → neck
    L_sh = get('L_Shoulder','L_Shoulder_Acromion','l_shoulder')
    R_sh = get('R_Shoulder','R_Shoulder_Acromion','r_shoulder')
    neck = mid(L_sh, R_sh)

    # HJCs → mid_hip
    L_hj = get('L_HJC','L_HJC_reg','L_HJCL','l_hjc')
    R_hj = get('R_HJC','R_HJC_reg','R_HJCR','r_hjc')
    mid_hip = mid(L_hj, R_hj)

    # Knees / Ankles
    L_kn = get('L_knee','l_knee','L_Knee')
    R_kn = get('R_knee','r_knee','R_Knee')
    L_an = get('L_ankle','l_ankle','L_Ankle')
    R_an = get('R_ankle','r_ankle','R_Ankle')

    # Heels / toes
    L_he = get('L_calc','l_calc','L_heel','L_Calc')
    R_he = get('R_calc','r_calc','R_heel','R_Calc')
    L_sm = get('L_5meta','l_5meta','L_5th_met')
    R_sm = get('R_5meta','r_5meta','R_5th_met')
    L_bg = get('L_toe','l_toe','L_1st_met','L_big_toe')
    R_bg = get('R_toe','r_toe','R_1st_met','R_big_toe')

    # Elbows
    L_el_med = get('L_elbow_med','l_elbow_med','L_EPI_med')
    L_el_lat = get('L_elbow_lat','l_elbow_lat','L_EPI_lat')
    L_el = mid(L_el_med, L_el_lat) if (L_el_med is not None and L_el_lat is not None) else get('L_elbow','l_elbow')

    R_el_med = get('R_elbow_med','r_elbow_med','R_EPI_med')
    R_el_lat = get('R_elbow_lat','r_elbow_lat','R_EPI_lat')
    R_el = mid(R_el_med, R_el_lat) if (R_el_med is not None and R_el_lat is not None) else get('R_elbow','r_elbow')

    # Wrists
    L_wr_rad = get('L_wrist_radius','l_wrist_radius','L_WRA','L_WR_rad')
    L_wr_uln = get('L_wrist_ulna','l_wrist_ulna','L_WRB','L_WR_uln')
    L_wr = mid(L_wr_rad, L_wr_uln) if (L_wr_rad is not None and L_wr_uln is not None) else get('L_wrist','l_wrist')

    R_wr_rad = get('R_wrist_radius','r_wrist_radius','R_WRA','R_WR_rad')
    R_wr_uln = get('R_wrist_ulna','r_wrist_ulna','R_WRB','R_WR_uln')
    R_wr = mid(R_wr_rad, R_wr_uln) if (R_wr_rad is not None and R_wr_uln is not None) else get('R_wrist','r_wrist')

    parts = [neck, mid_hip,
             L_sh, R_sh, L_el, R_el, L_wr, R_wr,
             L_hj, R_hj, L_kn, R_kn, L_an, R_an, L_he, R_he, L_sm, R_sm, L_bg, R_bg]

    if any(p is None for p in parts):
        missing = [OCAP20[i] for i,p in enumerate(parts) if p is None]
        log_warn(f"TRC missing for: {missing}")
        return None, None

    M = _np.stack(parts, axis=1)  # (T,20,3)
    return M, list(OCAP20)

def ocap20_from_enhancer(names, X_mm):
    """
    Build (T,20,3) from enhancer markers (T,N,3) by robust name matching.
    """
    T, N, _ = X_mm.shape
    ci = build_ci_lookup(names)

    def idx(*c): return pick_ci(ci, *c)
    def get(*c):
        i = idx(*c)
        return None if i is None else X_mm[:, i, :]

    def mid(A, B):
        if A is None or B is None: return None
        return 0.5*(A + B)

    L_sh = get('L_Shoulder','l_shoulder','Lsho','L_SHO')
    R_sh = get('R_Shoulder','r_shoulder','Rsho','R_SHO')
    neck = mid(L_sh, R_sh)

    L_hj = get('L_HJC','L_HJC_reg','l_hjc') or get('Lhip','Left_Hip_JC')
    R_hj = get('R_HJC','R_HJC_reg','r_hjc') or get('Rhip','Right_Hip_JC')
    mid_hip = mid(L_hj, R_hj)

    L_el = mid(get('L_elbow_med','L_EPI_med','L_ELMed','L_EL_med'),
               get('L_elbow_lat','L_EPI_lat','L_ELLat','L_EL_lat')) or get('L_elbow','l_elbow')
    R_el = mid(get('R_elbow_med','R_EPI_med','R_ELMed','R_EL_med'),
               get('R_elbow_lat','R_EPI_lat','R_ELLat','R_EL_lat')) or get('R_elbow','r_elbow')

    L_wr = mid(get('L_wrist_radius','L_WRA','L_WR_rad'),
               get('L_wrist_ulna','L_WRB','L_WR_uln')) or get('L_wrist','l_wrist')
    R_wr = mid(get('R_wrist_radius','R_WRA','R_WR_rad'),
               get('R_wrist_ulna','R_WRB','R_WR_uln')) or get('R_wrist','r_wrist')

    L_kn = mid(get('L_knee_med','L_KNE_med','L_KNE_M'),
               get('L_knee_lat','L_KNE_lat','L_KNE_L')) or get('L_knee','l_knee')
    R_kn = mid(get('R_knee_med','R_KNE_med','R_KNE_M'),
               get('R_knee_lat','R_KNE_lat','R_KNE_L')) or get('R_knee','r_knee')

    L_an = get('L_ankle','l_ankle','L_ANK')
    R_an = get('R_ankle','r_ankle','R_ANK')

    L_he = get('L_calc','l_calc','L_HEE','L_heel')
    R_he = get('R_calc','r_calc','R_HEE','R_heel')
    L_sm = get('L_5meta','l_5meta','L_5th_met')
    R_sm = get('R_5meta','r_5meta','R_5th_met')
    L_bg = get('L_toe','l_toe','L_1st_met','L_big_toe')
    R_bg = get('R_toe','r_toe','R_1st_met','R_big_toe')

    parts = [neck, mid_hip, L_sh, R_sh, L_el, R_el, L_wr, R_wr,
             L_hj, R_hj, L_kn, R_kn, L_an, R_an, L_he, R_he, L_sm, R_sm, L_bg, R_bg]

    K = len(OCAP20)
    P = np.full((T, K, 3), np.nan, dtype=float)
    missing = []
    for k, arr in enumerate(parts):
        if arr is not None:
            P[:, k, :] = arr
        else:
            missing.append(OCAP20[k])
    if missing:
        log_warn(f"Enhancer missing for: {missing}")
    return P, list(OCAP20)

# ---------------------------
# NPZ helpers (load enhancer)
# ---------------------------
def normalize_name(n: str) -> str:
    s = str(n).lower()
    for suf in ('_augmenter', 'augmenter', '_mmpose', 'mmpose', '_openpose', 'openpose'):
        if s.endswith(suf):
            s = s[: -len(suf)]
    s = re.sub(r'[^a-z0-9]', '', s)  # R_5meta → r5meta
    return s

# canonical → TRC-name candidates (all strings run through normalize_name)
_ENH_TO_TRC_SYNONYMS = {
    # hips / pelvis
    'rhjc': ['rhjc','rhjcreg','rhipjc'],
    'lhjc': ['lhjc','lhjcreg','lhipjc'],
    'rasis': ['rasis','r_asis'],
    'lasis': ['lasis','l_asis'],
    'rpsis': ['rpsis','r_psis'],
    'lpsis': ['lpsis','l_psis'],

    # shoulders
    'rshoulder': ['rshoulder','rshoulderacromion'],
    'lshoulder': ['lshoulder','lshoulderacromion'],

    # elbows (lat/med)
    'relbow':  ['relbowlat','relbow'],
    'rmelbow': ['relbowmed','rmelbow'],
    'lelbow':  ['lelbowlat','lelbow'],
    'lmelbow': ['lelbowmed','lmelbow'],

    # wrists (radius/ulna)
    'rwrist':  ['rwristradius','rwrist'],
    'rmwrist': ['rwristulna','rmwrist'],
    'lwrist':  ['lwristradius','lwrist'],
    'lmwrist': ['lwristulna','lmwrist'],

    # knees (lat/med)
    'rknee':  ['rkneelat','rknee'],
    'rmknee': ['rkneemed','rmknee'],
    'lknee':  ['lkneelat','lknee'],
    'lmknee': ['lkneemed','lmknee'],

    # ankles
    'rankle':  ['rankle','ranklelat','ranklemed'],
    'rmankle': ['ranklemed','rankle'],
    'lankle':  ['lankle','lanklelat','lanklemed'],
    'lmankle': ['lanklemed','lankle'],

    # feet
    'rtoe':   ['r1stmet','rbigtoe','rtoe'],
    'ltoe':   ['l1stmet','lbigtoe','ltoe'],
    'r5meta': ['r5meta','r5thmet'],
    'l5meta': ['l5meta','l5thmet'],
    'rcalc':  ['rcalc','rheel','rcalcaneus'],
    'lcalc':  ['lcalc','lheel','lcalcaneus'],

    # spine
    'c7': ['c7'],
}

def _find_trc_idx_for_enh(enh_norm: str, trc_norm2idx: dict[int, int]) -> int | None:
    if enh_norm in trc_norm2idx:
        return trc_norm2idx[enh_norm]
    for c in _ENH_TO_TRC_SYNONYMS.get(enh_norm, []):
        if c in trc_norm2idx:
            return trc_norm2idx[c]
    return None

def build_ci_map(names):
    return {normalize_name(n): i for i, n in enumerate(names)}

def intersect_enhancer_with_trc(names_enh, X_enh, trc_markers):
    """
    Return (P_enh, M_trc, common_enhancer_names) for markers present in BOTH enhancer and TRC.
    Matching is robust to suffixes/underscores/case and uses a small synonym table.
    """
    import numpy as _np
    if not names_enh:
        raise RuntimeError("Enhancer marker names are required for --compare-set enhancer_native.")

    # Build TRC lookup (normalized → first index)
    trc_raw_names = list(trc_markers.keys())
    trc_norm2idx = {}
    for i, nm in enumerate(trc_raw_names):
        key = normalize_name(nm)
        if key not in trc_norm2idx:
            trc_norm2idx[key] = i

    enh_idxs = []
    trc_arrays = []
    common_names = []
    used_trc = set()

    for i_enh, nm in enumerate(names_enh):
        enh_norm = normalize_name(nm)
        j_trc = _find_trc_idx_for_enh(enh_norm, trc_norm2idx)
        if j_trc is None or j_trc in used_trc:
            continue
        used_trc.add(j_trc)

        enh_idxs.append(i_enh)
        trc_arrays.append(trc_markers[trc_raw_names[j_trc]])
        common_names.append(nm)

    if not enh_idxs:
        raise RuntimeError("No common marker names between enhancer and TRC after normalization/synonyms. "
                           "Try --compare-set ocap20.")

    P = X_enh[:, enh_idxs, :]                        # (T_enh, M, 3)
    M = _np.stack(trc_arrays, axis=1).astype(float)  # (T_trc, M, 3)
    log_info(f"Matched native markers: M={len(enh_idxs)}")
    return P, M, common_names


def load_npz_with_names(npz_path: Path, names_path: Path | None = None):
    """
    Load enhancer predictions and (optionally) marker names from an .npz bundle.
    Accepts keys: 'pred_mm', 'markers_mm', 'outputs', 'Y', 'pred', 'arr_0'
    Names under: 'names', 'marker_names', 'markers_names', 'joint_names'
    """
    import json as _json
    import numpy as _np

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    d = _np.load(npz_path, allow_pickle=True)
    files = set(getattr(d, "files", []))

    data_keys = ["pred_mm", "markers_mm", "outputs", "Y", "pred", "arr_0"]
    chosen_key = None
    for k in data_keys:
        if k in files:
            chosen_key = k
            break
    if chosen_key is None:
        raise RuntimeError(f"No prediction array found in {npz_path}. Looked for keys: {data_keys}")

    X = _np.asarray(d[chosen_key])
    if X.ndim != 3 or X.shape[-1] != 3:
        raise RuntimeError(f"Expected (T,M,3) prediction array in {npz_path}[{chosen_key}], got {X.shape}")

    names = None
    for nk in ["names", "marker_names", "markers_names", "joint_names"]:
        if nk in files:
            raw = d[nk]
            try:
                names = [str(x) for x in list(raw)]
            except Exception:
                names = [str(x) for x in _np.asarray(raw).tolist()]
            break

    if names is None and names_path is not None and names_path.exists():
        suf = names_path.suffix.lower()
        try:
            if suf == ".json":
                obj = _json.loads(names_path.read_text(encoding="utf-8"))
                names = obj.get("names", obj)
                names = [str(x) for x in names]
            elif suf in {".npy", ".npz"}:
                nx = _np.load(names_path, allow_pickle=True)
                raw = nx[nx.files[0]] if hasattr(nx, "files") else nx
                names = [str(x) for x in _np.asarray(raw).tolist()]
            else:
                names = [ln.strip() for ln in names_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except Exception as e:
            log_warn(f"Failed to read names from {names_path}: {e}")

    log_info(f"Loaded {npz_path.name} → key='{chosen_key}', shape={tuple(X.shape)}, names={'yes' if names else 'no'}")
    return X, names

def load_prep_time(npz_path: Path, default_fs: float = 60.0):
    """
    Robustly load time vector from enhancer prep NPZ.
    Looks for keys in order: 't_60', 't', 'time_60', 'time'.
    If none are found, synthesizes time from length and default_fs.
    """
    import numpy as _np
    d = _np.load(npz_path, allow_pickle=True)
    keys = ['t_60', 't', 'time_60', 'time']
    for k in keys:
        if k in getattr(d, 'files', []):
            t = _np.asarray(d[k], dtype=float).reshape(-1)
            log_info(f"Prep NPZ time key   : '{k}' (len={len(t)})")
            return t
    # Fallback: infer T
    T = None
    for carrier in ('ocap20_centered_mm', 'X_body_47', 'X_arms_23'):
        if carrier in getattr(d, 'files', []):
            arr = _np.asarray(d[carrier])
            if arr.ndim >= 1:
                T = arr.shape[0]
                break
    if T is None:
        raise RuntimeError(f"No time vector in {npz_path.name} and couldn't infer length from arrays.")
    t = _np.arange(T, dtype=float) / float(default_fs)
    log_warn(f"No explicit time in {npz_path.name}; synthesized at {default_fs:.1f} Hz (len={T}).")
    return t

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Validate Marker-Enhancer (mm) vs MoCap (100 Hz, mm)")
    ap.add_argument("-m", "--manifest", required=True)
    ap.add_argument("-p", "--paths", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--outputs-root", default=None)

    ap.add_argument("--compare-set", choices=["ocap20", "enhancer_native"], default="ocap20",
                    help="Compare over OCAP-20 (composed joints) or native enhancer markers (name intersection).")

    # Optional overrides / aids
    ap.add_argument("--body-npz", default=None, help="Override path to body_pred_mm_Tx35x3.npz")
    ap.add_argument("--arms-npz", default=None, help="Override path to arms_pred_mm_Tx8x3.npz")
    ap.add_argument("--body-names", default=None, help="(Legacy) external names file for body npz")
    ap.add_argument("--arms-names", default=None, help="(Legacy) external names file for arms npz")

    # Filtering
    ap.add_argument("--include", default=None, help="Regex of names to INCLUDE (evaluated before exclude)")
    ap.add_argument("--exclude", default=None, help="Regex of names to EXCLUDE")

    # Alignment
    ap.add_argument("--estimate-offset", action="store_true")
    ap.add_argument("--max-offset", type=float, default=1.0, help="max |offset| (s) for xcorr")
    ap.add_argument("--allow-scale", action="store_true",
                    help="Estimate similarity (scale+rotation) instead of rotation-only.")
    ap.add_argument("--save-series", action="store_true", help="save aligned series to npz")

    args = ap.parse_args()

    # Load manifest
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
    if not base:
        subj = manifest.get('subject_id', 'subject')
        sess = manifest.get('session', 'Session')
        cam  = manifest.get('camera', 'Cam')
        base = Path(manifest.get('outputs_root', Path.cwd() / "outputs")) / subj / sess / cam
    trial_root = Path(base) / trial['id']
    enh_dir  = trial_root / "enhancer"
    ensure_dir(enh_dir)

    trc_path = Path(trial.get("mocap_trc", ""))
    meta_path = trial_root / "meta.json"
    body_npz = Path(args.body_npz) if args.body_npz else (enh_dir / "body_pred_mm_Tx35x3.npz")
    arms_npz = Path(args.arms_npz) if args.arms_npz else (enh_dir / "arms_pred_mm_Tx8x3.npz")
    prep_npz = enh_dir / "enh_input_60hz.npz"  # for t_60

    log_info(f"Trial root          : {trial_root}")
    log_info(f"Enhancer dir        : {enh_dir}")
    log_info(f"Body NPZ            : {body_npz}")
    log_info(f"Arms NPZ            : {arms_npz} (optional)")
    log_info(f"Prep NPZ (t_60)     : {prep_npz}")
    log_info(f"TRC path            : {trc_path}")
    log_info(f"meta.json           : {meta_path}")

    if not body_npz.exists():
        raise SystemExit(f"[ERROR] Body NPZ not found: {body_npz}")
    if not trc_path.exists():
        raise SystemExit(f"[ERROR] TRC not found: {trc_path}")
    if not meta_path.exists():
        log_warn("meta.json not found (continuing; only affects display labels).")
    if not prep_npz.exists():
        log_warn("enh_input_60hz.npz not found; will synthesize t_60 from number of frames.")

    # Load enhancer outputs
    log_step("Loading Enhancer predictions (mm)")
    Xb, names_b = load_npz_with_names(body_npz, Path(args.body_names) if args.body_names else None)
    log_info(f"Body shapes         : {Xb.shape} (T x Nb x 3)")
    if names_b: log_info(f"Body names (Nb)     : {len(names_b)}")

    Xa, names_a = (None, None)
    if arms_npz.exists():
        Xa, names_a = load_npz_with_names(arms_npz, Path(args.arms_names) if args.arms_names else None)
        log_info(f"Arms shapes         : {Xa.shape if Xa is not None else None}")
        if names_a: log_info(f"Arms names (Na)     : {len(names_a)}")

    # Combine body + arms
    if Xa is not None:
        if Xb.shape[0] != Xa.shape[0]:
            raise SystemExit("[ERROR] Body/arms T mismatch; ensure both produced from same run.")
        X_all = np.concatenate([Xb, Xa], axis=1)
        names_all = (names_b or []) + (names_a or [])
    else:
        X_all = Xb
        names_all = names_b or []
    T60, N, _ = X_all.shape

    # Time base (60 Hz)
    if prep_npz.exists():
        t_60 = load_prep_time(prep_npz, default_fs=60.0)
        t_60 = np.asarray(t_60, dtype=float)
    else:
        t_60 = np.arange(T60) / 60.0

    # Load TRC
    log_step("Reading TRC")
    t_trc, markers_trc, trc_rate = load_trc(trc_path)

    # ---- Build the comparison sets ----
    if args.compare_set == "ocap20":
        log_step("Assembling OpenCap-20 from Enhancer & TRC")
        P_raw, names_cmp = ocap20_from_enhancer(names_all, X_all)           # (T60, 20, 3)
        M_raw, names_trc = ocap20_from_trc(t_trc, markers_trc)              # (Ttrc, 20, 3)
        if M_raw is None:
            raise SystemExit("[ERROR] Could not assemble OpenCap-20 from TRC (see warnings).")
        # enforce common order
        if names_trc != names_cmp:
            order = {n:i for i,n in enumerate(names_cmp)}
            sel = [order[n] for n in names_trc]
            P_raw = P_raw[:, sel, :]
            names_cmp = names_trc

        joint_names = names_cmp  # 20 names
        P_use, M_use = P_raw, M_raw

        # Indices for pelvis/offset
        def _idx_try(nm):
            try: return joint_names.index(nm)
            except ValueError: return None
        LHIP = _idx_try("left_hip")
        RHIP = _idx_try("right_hip")
        LANK = _idx_try("left_ankle")
        RANK = _idx_try("right_ankle")

    else:  # enhancer_native
        log_step("Intersecting enhancer native markers with TRC names")
        P_use, M_use, joint_names = intersect_enhancer_with_trc(names_all, X_all, markers_trc)

        # Print a quick preview of matches
        log_info("First 20 matched markers (enhancer → TRC order):")
        for i, n in enumerate(joint_names[:20]):
            log_info(f"  [{i:02d}] {n}")
        log_info(f"Total native matches: {len(joint_names)}")

        # HJCs & ankles if present (for offset/pelvis)
        names_norm = [normalize_name(n) for n in joint_names]
        def _idx_try_norm(key):
            try: return names_norm.index(key)
            except ValueError: return None
        LHIP = _idx_try_norm('lhjc')
        RHIP = _idx_try_norm('rhjc')
        LANK = _idx_try_norm('lankle')
        RANK = _idx_try_norm('rankle')

    # ---- Optional include/exclude filtering ----
    keep = np.arange(len(joint_names)).tolist()
    if args.include:
        rx = re.compile(args.include, re.I)
        keep = [i for i,n in enumerate(joint_names) if rx.search(n)]
    if args.exclude:
        rx = re.compile(args.exclude, re.I)
        keep = [i for i in keep if not rx.search(joint_names[i])]
    if not keep:
        raise SystemExit("[ERROR] After include/exclude filtering, no joints remain.")

    if len(keep) != len(joint_names):
        log_info(f"Filtering kept {len(keep)}/{len(joint_names)} markers.")
        P_use = P_use[:, keep, :]
        M_use = M_use[:, keep, :]
        joint_names = [joint_names[i] for i in keep]
        # Recompute pelvis indices after filtering (best effort)
        def _lower_index(name):
            try: return [jn.lower() for jn in joint_names].index(name.lower())
            except ValueError: return None
        # Try both OCAP and native conventions
        LHIP = _lower_index("left_hip")  or _lower_index("lhjc")
        RHIP = _lower_index("right_hip") or _lower_index("rhjc")
        LANK = _lower_index("left_ankle") or _lower_index("lankle")
        RANK = _lower_index("right_ankle") or _lower_index("rankle")

    # ---- Estimate time offset (optional) ----
    def seg_len(X, a, b):
        if a is None or b is None: return None
        if a >= X.shape[1] or b >= X.shape[1]: return None
        d = np.linalg.norm(X[:, a, :] - X[:, b, :], axis=-1)
        return d if np.isfinite(d).any() else None

    t_offset = 0.0
    sig_enh = seg_len(P_use, LHIP, LANK)
    sig_trc = seg_len(M_use, LHIP, LANK)
    if args.estimate_offset and (sig_enh is not None) and (sig_trc is not None):
        log_step("Estimating time offset via cross-correlation")
        t_offset = estimate_time_offset(t_60, sig_enh, t_trc, sig_trc, max_offset=args.max_offset, fs=100.0)
        log_info(f"Estimated offset    : {t_offset:+.3f} s (add to Enhancer)")
    else:
        log_info("Offset estimation   : skipped (missing hip/ankle pairs or disabled). Using 0.000 s")

    # ---- Resample to common 100 Hz grid ----
    log_step("Resampling both series to 100 Hz common grid")
    t0 = max(t_trc.min(), t_60.min() + t_offset)
    t1 = min(t_trc.max(), t_60.max() + t_offset)
    if t1 - t0 < 0.5:
        raise SystemExit("[ERROR] Insufficient temporal overlap after offset.")
    t_grid = np.arange(t0, t1, 0.01)
    log_info(f"Common time range   : {t0:.3f}..{t1:.3f} s")
    log_info(f"Grid frames (100Hz) : {len(t_grid)}")
    
    log_info(f"Enhancer t_60 range : {t_60.min():.3f}..{t_60.max():.3f} s (T={len(t_60)})")
    log_info(f"TRC time range      : {t_trc.min():.3f}..{t_trc.max():.3f} s (T={len(t_trc)})")
    log_info(f"Enh after offset    : {(t_60+t_offset).min():.3f}..{(t_60+t_offset).max():.3f} s")
    log_info(f"Common window       : {t0:.3f}..{t1:.3f} s  → {len(t_grid)} frames @100Hz")

    P_100 = resample_to_grid(t_60 + t_offset, P_use, t_grid)
    M_100 = resample_to_grid(t_trc,          M_use, t_grid)

    # ---- Centering ----
    log_step("Centering series")
    if (LHIP is not None) and (RHIP is not None) and (LHIP < P_100.shape[1]) and (RHIP < P_100.shape[1]):
        pelv_P = 0.5*(P_100[:, LHIP, :] + P_100[:, RHIP, :])
        pelv_M = 0.5*(M_100[:, LHIP, :] + M_100[:, RHIP, :])
        P_c = P_100 - pelv_P[:, None, :]
        M_c = M_100 - pelv_M[:, None, :]
    else:
        log_warn("Pelvis centering fell back to centroid (missing HJCs).")
        P_c = P_100 - np.nanmean(P_100, axis=1, keepdims=True)
        M_c = M_100 - np.nanmean(M_100, axis=1, keepdims=True)

    # ---- Kabsch / Similarity Kabsch ----
    log_step("Estimating best-fit alignment")
    mask = ~np.isnan(P_c).any(axis=(1, 2)) & ~np.isnan(M_c).any(axis=(1, 2))
    if mask.sum() < 10:
        raise SystemExit("[ERROR] Too few valid frames after resampling and centering.")
    X = P_c[mask].reshape((-1, 3)); X -= X.mean(axis=0, keepdims=True)
    Y = M_c[mask].reshape((-1, 3)); Y -= Y.mean(axis=0, keepdims=True)
    s, R = kabsch_similarity(X, Y, allow_scale=args.allow_scale)
    log_info(f"Alignment           : {'similarity' if args.allow_scale else 'rotation-only'} "
             f"(det R={np.linalg.det(R):+.6f}, scale={s:.4f})")

    P_aligned = s * (P_c @ R.T)

    # ---- Errors ----
    log_step("Computing errors (per-point MPJPE & RMSE, overall MPJPE & RMSE)")
    diff = P_aligned - M_c
    dists = np.linalg.norm(diff, axis=-1)  # (Tg, K)
    valid = ~np.isnan(dists)

    mpjpe_per_joint = {}
    rmse_per_joint = {}
    for k, jn in enumerate(joint_names):
        dk = dists[:, k]
        dk = dk[~np.isnan(dk)]
        if dk.size:
            mpjpe_per_joint[jn] = float(np.mean(dk))
            rmse_per_joint[jn]  = float(np.sqrt(np.mean(dk**2)))
        else:
            mpjpe_per_joint[jn] = float("nan")
            rmse_per_joint[jn]  = float("nan")

    overall_mpjpe = float(np.nanmean(dists[valid]))
    overall_rmse  = float(np.sqrt(np.nanmean(dists[valid]**2)))

    # ---- Print summary ----
    log_info("Per-point metrics (mm):")
    for jn in joint_names:
        log_info(f"  {jn:>18s}  MPJPE={mpjpe_per_joint[jn]:7.2f}  RMSE={rmse_per_joint[jn]:7.2f}")
    log_info(f"OVERALL MPJPE (mm)  : {overall_mpjpe:.2f}")
    log_info(f"OVERALL RMSE  (mm)  : {overall_rmse:.2f}")
    log_info(f"Frames used (valid) : {int(valid.sum())} of {valid.size}")

    # ---- Save report ----
    report = {
        "trial_id": args.trial,
        "compare_set": args.compare_set,
        "time_offset_sec": float(t_offset),
        "alignment": "similarity" if args.allow_scale else "rotation-only",
        "scale": float(s),
        "rotation_3x3": R,
        "fps_trc": 100.0,
        "frames_used": int(valid.sum()),
        "names_order": joint_names,
        "mpjpe_per_point_mm": mpjpe_per_joint,
        "rmse_per_point_mm": rmse_per_joint,
        "overall_mpjpe_mm": overall_mpjpe,
        "overall_rmse_mm": overall_rmse,
        "notes": "Enhancer predictions treated as metric (mm); no auto-scale.",
    }
    out_json = enh_dir / "report_enhancer.json"
    out_json.write_text(json.dumps(jsonable(report), indent=2), encoding="utf-8")
    log_done(f"Report saved        : {out_json}")

    if args.save_series:
        out_npz = enh_dir / "aligned_100hz_enh.npz"
        np.savez_compressed(
            out_npz,
            t_100=t_grid,
            names=np.array(joint_names, dtype=object),
            mocap_mm=M_c,
            pred_aligned_mm=P_aligned
        )
        log_done(f"Aligned series saved: {out_npz}")

    log_done(f"OVERALL MPJPE={overall_mpjpe:.2f} mm | OVERALL RMSE={overall_rmse:.2f} mm")

if __name__ == "__main__":
    main()
