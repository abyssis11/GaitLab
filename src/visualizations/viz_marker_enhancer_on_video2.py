#!/usr/bin/env python3
# viz_marker_enhancer_on_video2.py
"""
Visualize **TRC markers** on a single video frame — TRC selection by **frame number** (default).

What it does
------------
- Loads your dataset manifest (IO.load_manifest) and resolves paths (same style as the RTMW3D viz).
- Opens the requested video (default: `video_sync`) and selects a frame by index or time.
- Reads a TRC file (default: `<trial>/enhancer/enhancer_<trial>.trc`).
- Filters markers by name suffix (default: `_study`).
- **Aligns TRC by frame number**: picks the TRC row whose `Frame#` is closest to the video frame index
  (plus optional offset). You can switch back to time-based alignment if needed.
- Draws the selected TRC markers on the frame:
  - **space=px**: treat TRC X/Y as pixel coordinates.
  - **space=mm**: treat TRC X/Y as metric; render centered/scaled on the image (no calibration, no cache).
  - **space=auto (default)**: heuristically decide based on value ranges vs. image size.
- Saves a PNG to `<trial>/enhancer_vis/frame_<idx>_trc.png` by default.

Usage
-----
python src/visualizations/viz_marker_enhancer_on_video2.py \
  -m manifests/OpenCapDataset/subject2.yaml \
  -p config/paths.yaml \
  --trial walking1 \
  --frame-index 24 \
  --trc-suffix _study \
  --trc-align frame \
  --trc-frame-offset 0 \
  --space auto \
  --label-markers

Switch back to time alignment (if desired): add `--trc-align time`.
"""

import argparse
from pathlib import Path
import json

import cv2
import numpy as np

from IO.load_manifest import load_manifest


# ---------------------------
# Logging helpers
# ---------------------------

def log_step(msg):
    print(f"[STEP] {msg}")

def log_info(msg):
    print(f"[INFO] {msg}")

def log_warn(msg):
    print(f"[WARN] {msg}")

def log_done(msg):
    print(f"[DONE] {msg}")

# ---------------------------
# Drawing helpers
# ---------------------------

def draw_points(img, pts_px, color=(255, 0, 255), label_prefix=None, names=None, radius=4, thickness=2):
    overlay = img.copy()
    h, w = img.shape[:2]
    for i, p in enumerate(pts_px):
        if not np.all(np.isfinite(p)):
            continue
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < w and 0 <= y < h:
            if label_prefix is not None:
                name = names[i] if (names is not None and i < len(names)) else str(i)
                txt = f"{label_prefix}:{name}"
                if name in ["L_knee_study", "r_knee_study", "r_ankle_study", "L_ankle_study", "r_shoulder_study", "r_lelbow_study","r_calc_study", "L_calc_study", "r_5meta_study", "L_5meta_study", "r_lwrist_study"]:
                    cv2.putText(overlay, txt, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (10, 10, 10), 3, cv2.LINE_AA)
                    cv2.putText(overlay, txt, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(overlay, (x, y), max(1, radius), color, -1, lineType=cv2.LINE_AA)

    img[:] = overlay


def put_header(img, text):
    cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 3, cv2.LINE_AA)
    cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def draw_legend(img, items):
    pad = 10
    x0, y0 = 12, img.shape[0] - 12
    line_h = 24
    box_w = 320
    box_h = pad * 2 + line_h * len(items)
    cv2.rectangle(img, (x0 - 8, y0 - box_h), (x0 - 8 + box_w, y0), (0, 0, 0), -1)
    cv2.rectangle(img, (x0 - 8, y0 - box_h), (x0 - 8 + box_w, y0), (255, 255, 255), 1)
    for i, (label, bgr) in enumerate(items):
        y = y0 - box_h + pad + i * line_h + 16
        cv2.circle(img, (x0, y - 6), 6, bgr, -1, lineType=cv2.LINE_AA)
        cv2.putText(img, label, (x0 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(img, label, (x0 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)



# ---------------------------
# TRC reader (robust to OpenSim format)
# ---------------------------

def load_trc(trc_path: Path):
    """Read an OpenSim-style TRC file.

    Returns
    -------
    frames : (T,) int array (Frame# column)
    times  : (T,) float array (seconds)
    names  : list[str] of marker names
    data   : (T, K, 3) float array (units as in TRC; often mm)
    """
    if not trc_path.exists():
        raise FileNotFoundError(f"TRC not found: {trc_path}")

    with open(trc_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = [ln.rstrip("\n\r") for ln in f]

    # Find header line that starts with Frame# and has Time next
    hdr_idx = None
    for i, ln in enumerate(raw_lines):
        toks = ln.strip().split()
        if toks and toks[0].lower().startswith("frame") and len(toks) >= 2 and toks[1].lower().startswith("time"):
            hdr_idx = i
            break
    if hdr_idx is None or hdr_idx + 2 >= len(raw_lines):
        raise ValueError("Unrecognized TRC header; expected 'Frame#\tTime\t<marker names> ...'")

    header_names = raw_lines[hdr_idx].strip().split()
    marker_names = header_names[2:]
    if not marker_names:
        raise ValueError("No marker names found in TRC header")

    # Parse numeric rows
    frames = []
    times = []
    rows = []
    for ln in raw_lines[hdr_idx + 2:]:
        if not ln.strip():
            continue
        toks = ln.replace(",", " ").split()
        if len(toks) < 2:
            continue
        try:
            fnum = float(toks[0])  # Frame# (may be float/int)
            t = float(toks[1])     # Time (sec)
        except Exception:
            continue
        vals = []
        for tok in toks[2:]:
            try:
                vals.append(float(tok))
            except Exception:
                vals.append(np.nan)
        frames.append(int(round(fnum)))
        times.append(t)
        rows.append(vals)

    if not rows:
        raise ValueError("No data rows parsed from TRC")

    frames = np.asarray(frames, dtype=int)
    times = np.asarray(times, dtype=float)
    K = len(marker_names)
    T = len(rows)
    data = np.full((T, K, 3), np.nan, dtype=float)
    for ti, vals in enumerate(rows):
        ntrip = min(K, len(vals) // 3)
        for k in range(ntrip):
            data[ti, k, :] = vals[3 * k: 3 * k + 3]

    return frames, times, marker_names, data

def _parse_trc(trc_path: Path):
    """Low-level TRC parser.

    Returns
    -------
    frames : (T,) int array (Frame# column)
    times  : (T,) float array (seconds)
    names  : list[str] of marker names
    data   : (T, K, 3) float array (units as in TRC; often mm)
    """
    if not trc_path.exists():
        raise FileNotFoundError(f"TRC not found: {trc_path}")

    with open(trc_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = [ln.rstrip("") for ln in f]

    # Find header line that starts with Frame# and has Time next
    hdr_idx = None
    for i, ln in enumerate(raw_lines):
        toks = ln.strip().split()
        if toks and toks[0].lower().startswith("frame") and len(toks) >= 2 and toks[1].lower().startswith("time"):
            hdr_idx = i
            break
    if hdr_idx is None or hdr_idx + 2 >= len(raw_lines):
        raise ValueError("Unrecognized TRC header; expected 'Frame#	Time	<marker names> ...'")

    header_names = raw_lines[hdr_idx].strip().split()
    marker_names = header_names[2:]
    if not marker_names:
        raise ValueError("No marker names found in TRC header")

    # Parse numeric rows
    frames = []
    times = []
    rows = []
    for ln in raw_lines[hdr_idx + 2:]:
        if not ln.strip():
            continue
        toks = ln.replace(",", " ").split()
        if len(toks) < 2:
            continue
        try:
            fnum = float(toks[0])  # Frame# (may be float/int)
            t = float(toks[1])     # Time (sec)
        except Exception:
            continue
        vals = []
        for tok in toks[2:]:
            try:
                vals.append(float(tok))
            except Exception:
                vals.append(np.nan)
        frames.append(int(round(fnum)))
        times.append(t)
        rows.append(vals)

    if not rows:
        raise ValueError("No data rows parsed from TRC")

    frames = np.asarray(frames, dtype=int)
    times = np.asarray(times, dtype=float)
    K = len(marker_names)
    T = len(rows)
    data = np.full((T, K, 3), np.nan, dtype=float)
    for ti, vals in enumerate(rows):
        ntrip = min(K, len(vals) // 3)
        for k in range(ntrip):
            data[ti, k, :] = vals[3 * k: 3 * k + 3]

    return frames, times, marker_names, data

def read_trc_find_frame(trc_path: Path, frame_index=None, time_target=None, tol_sec=0.02, suffix_filter: str | None = None):
    """
    Return dict for the chosen TRC row (similar to read_jsonl_find_frame):
    {
    'time_sec': float,
    'frame_index': int,
    'keypoints_xyz_mm': (K,3) ndarray,
    'keypoint_names': list[str]
    }


    Selection:
    - If frame_index is given, pick that exact Frame#. If no exact match, return None.
    - Else, find the nearest time to time_target within tol_sec; if none within tol_sec, return None.


    If suffix_filter is provided (e.g., "_study"), only keep columns whose names end with it.
    """
    frames, times, names, data = _parse_trc(trc_path)


    # Optional suffix filter
    if suffix_filter:
        keep = [i for i, n in enumerate(names) if str(n).endswith(suffix_filter)]
        if keep:
            names = [names[i] for i in keep]
            data = data[:, keep, :]


    # Choose row
    idx = None
    if frame_index is not None:
        hits = np.where(frames == int(frame_index))[0]
        if hits.size == 0:
            return None
        idx = int(hits[0])
    elif time_target is not None:
        diffs = np.abs(times - float(time_target))
        j = int(np.argmin(diffs))
        if diffs[j] > float(tol_sec):
            return None
        idx = j
    else:
        return None


    entry = {
        "time_sec": float(times[idx]),
        "frame_index": int(frames[idx]),
        "keypoints_xyz_mm": np.asarray(data[idx, :, :], dtype=float),
        "keypoint_names": [str(x) for x in names],
    }
    return entry

# ---------------------------
# Minimal RTMW3D JSONL helpers (used for similarity/affine)
# ---------------------------
def _extract_keypoints_xyz_mm(p: dict):
    """Return an (N,3) float numpy array of 3D keypoints in millimeters if present, else None."""
    candidates = ["keypoints_xyz_mm", "kps3d_mm", "pose3d_mm", "xyz_mm"]
    for k in candidates:
        if k not in p or p[k] is None:
            continue
        a = np.asarray(p[k], dtype=float)
        if a.ndim == 2 and a.shape[1] == 3:
            return a
        if a.ndim == 1 and a.size % 3 == 0:
            return a.reshape(-1, 3)
    return None


def _extract_person_list(rec: dict):
    """Return a list of person dicts from a jsonl record, trying common keys."""
    for key in ("persons", "people", "detections", "predictions"):
        if key in rec and isinstance(rec[key], list):
            return rec[key]
    # Fallback: sometimes there's a single person dict under a key
    for key in ("person", "prediction"):
        if key in rec and isinstance(rec[key], dict):
            return [rec[key]]
    return []


def _extract_keypoints_2d(p: dict):
    """Return an (N,2) float numpy array of 2D keypoints if present, else None.

    Tries several likely field names, and supports both list-of-lists and dict forms.
    """
    candidates = [
        "keypoints_px", "keypoints2d", "keypoints_2d",
        "kpts_px", "kps2d", "pose2d", "pose_2d", "keypoints",
    ]

    for k in candidates:
        if k not in p:
            continue
        arr = p[k]
        if arr is None:
            continue
        # dict with x/y arrays
        if isinstance(arr, dict) and "x" in arr and "y" in arr:
            x = np.asarray(arr["x"], dtype=float).reshape(-1)
            y = np.asarray(arr["y"], dtype=float).reshape(-1)
            if x.shape[0] == y.shape[0]:
                return np.stack([x, y], axis=-1)
        # list or ndarray with shape (*,2)
        a = np.asarray(arr, dtype=float)
        if a.ndim == 2 and a.shape[1] == 2:
            return a
    return None


def _extract_mean_score(p: dict):
    for k in ("mean_score", "score", "avg_score", "confidence"):
        v = p.get(k, None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    # If absent, treat as 1.0 so selection by max works
    return 1.0

def read_jsonl_find_frame(preds_path: Path, frame_index=None, time_target=None, tol_sec=0.02,
                          person_index=-1, min_mean_score=0.0):
    """
    Return dict for the chosen line:
      { 'time_sec', 'frame_index', 'keypoints_px' (K,2), 'keypoint_names' (optional) }

    Selection:
      - If frame_index is given, pick that exact frame.
      - Else, find the nearest time to time_target within tol_sec.
    Person selection: index or highest mean_score.
    """
    best = None
    best_dt = float("inf")

    with open(preds_path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue

            # time/frame matching
            fi_ok = True
            dt = 0.0
            if frame_index is not None:
                fi_ok = int(rec.get("frame_index", -999)) == int(frame_index)
                if not fi_ok:
                    continue
            elif time_target is not None:
                rec_t = float(rec.get("time_sec", 0.0))
                dt = abs(rec_t - float(time_target))
                if dt > tol_sec:
                    continue

            persons = _extract_person_list(rec)
            if not persons:
                continue

            # choose person
            if person_index == -1:
                scores = [
                    _extract_mean_score(p) if _extract_keypoints_2d(p) is not None else -1.0
                    for p in persons
                ]
                if max(scores, default=-1.0) < min_mean_score:
                    continue
                pi = int(np.argmax(scores))
            else:
                pi = min(max(0, person_index), len(persons) - 1)

            p = persons[pi]
            kpx = _extract_keypoints_2d(p)
            kmm = _extract_keypoints_xyz_mm(p)
            if kpx is None and kmm is None:
                continue

            if _extract_mean_score(p) < float(min_mean_score):
                continue

            entry = {
                "time_sec": float(rec.get("time_sec", 0.0)),
                "frame_index": int(rec.get("frame_index", -1)),
            }
            if kpx is not None:
                entry["keypoints_px"] = np.asarray(kpx, dtype=float)
            if kmm is not None:
                entry["keypoints_xyz_mm"] = np.asarray(kmm, dtype=float)

            # keypoint names if present at record or person level
            kp_names = rec.get("keypoint_names") or p.get("keypoint_names")
            if isinstance(kp_names, list) and all(isinstance(x, str) for x in kp_names):
                entry["keypoint_names"] = kp_names

            # keep first exact match; else keep nearest
            keep = False
            if frame_index is not None:
                keep = True
            else:
                keep = dt < best_dt

            if keep:
                best = entry
                best_dt = dt
                if frame_index is not None:
                    break

    return best


# ---------------------------
# Similarity fit (2D) and affine fit (3D→2D)
# ---------------------------

def fit_similarity_2d(X: np.ndarray, Y: np.ndarray):
    """Return s, R(2x2), t(2,) such that Y ≈ s * R @ X + t.
    X, Y: (N,2)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    assert X.shape == Y.shape and X.ndim == 2 and X.shape[1] == 2
    N = X.shape[0]
    if N < 2:
        raise ValueError("Need at least 2 points for similarity fit")
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    X0 = X - muX
    Y0 = Y - muY
    # 2x2 covariance
    C = (Y0.T @ X0) / N
    U, S, Vt = np.linalg.svd(C)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    varX = (X0**2).sum() / N
    s = (S @ np.ones_like(S)) / (varX + 1e-12)
    t = muY - s * (R @ muX)
    return float(s), R, t

def apply_similarity_2d(Pxy: np.ndarray, s: float, R: np.ndarray, t: np.ndarray):
    log_info(f"R.shape={R.shape}")
    log_info(f"Pxy.T.shape={Pxy.T.shape}")

    return (s * (R @ Pxy.T)).T + t.reshape(1, 2)


def _fit_affine_3d_to_2d(X_mm, Y_px):
    """Return M (4x2) s.t. Y ≈ [X|1] @ M, and a mask of used rows."""
    X = np.asarray(X_mm, dtype=float)
    Y = np.asarray(Y_px, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3 or Y.ndim != 2 or Y.shape[1] != 2:
        return None, None
    mask = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
    Xf, Yf = X[mask], Y[mask]
    if Xf.shape[0] < 4:
        return None, mask
    X_aug = np.hstack([Xf, np.ones((Xf.shape[0], 1))])
    M, *_ = np.linalg.lstsq(X_aug, Yf, rcond=None)
    return M, mask

import numpy as np
import math

def _rotate_points_2d(P, theta_rad, about=None):
    """Rotate Nx2 points by theta around 'about' (2-vector)."""
    if about is None:
        about = np.nanmean(P, axis=0)
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    R = np.array([[c, -s],
                  [s,  c]], dtype=float)
    return (P - about) @ R.T + about

def normalize_center_with_rotation(body_px, img, small_span_thresh=50,
                                   fill=0.65, target_xy_frac=(0.4, 0.6),
                                   angle_deg=None, auto_angle='pca'):
    """
    body_px: Nx2 float array
    img: HxW[xC] numpy image
    angle_deg: if not None, rotate by this angle (degrees).
    auto_angle: 'pca' or 'none' (used only if angle_deg is None).
    """
    if not (isinstance(body_px, np.ndarray) and body_px.ndim == 2 and body_px.shape[1] == 2):
        return body_px

    P = np.asarray(body_px, dtype=float).copy()
    h, w = img.shape[:2]

    # Keep only rows where both coordinates are finite
    valid = np.isfinite(P).all(axis=1)
    if valid.sum() < 2:
        return P  # not enough info to scale/rotate/center

    Q = P[valid]

    # ----- SCALE (replace nanptp with nan-safe max-min on valid rows) -----
    min_xy = Q.min(axis=0)              # (2,)
    max_xy = Q.max(axis=0)              # (2,)
    ranges = max_xy - min_xy            # (2,)
    span = float(np.max(ranges))        # scalar

    if 0 < span < small_span_thresh:
        s = fill * min(w, h) / span
        P[valid] *= s
        Q = P[valid]  # refresh after scaling

    # ----- ROTATION (manual or auto via PCA) -----
    theta = 0.0
    if angle_deg is not None:
        theta = math.radians(angle_deg)
    elif auto_angle == 'pca':
        mu = Q.mean(axis=0, keepdims=True)   # (1,2)
        Z = Q - mu
        # 2x2 covariance; stable for valid.sum() >= 2
        C = (Z.T @ Z) / max(len(Z), 1)
        evals, V = np.linalg.eigh(C)         # eigenvectors are columns of V
        v = V[:, np.argmax(evals)]           # principal axis (2,)
        # Rotate so the principal axis aligns with the image +y axis (vertical)
        theta = -math.atan2(v[0], v[1])

    # Rotate about the centroid of valid points
    mu = Q.mean(axis=0)                      # (2,)
    c, s_ = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s_],
                  [s_,  c]], dtype=float)
    P[valid] = (Q - mu) @ R.T + mu

    # ----- CENTER to target fraction of the image -----
    cx, cy = w * float(target_xy_frac[0]), h * float(target_xy_frac[1])
    m = P[valid].mean(axis=0)                # (2,)
    P += np.array([cx - m[0], cy - m[1]], dtype=float)

    return P

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Visualize TRC markers on a video frame (no cache, no preds).")
    ap.add_argument("-m", "--manifest", required=True)
    ap.add_argument("-p", "--paths", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--outputs-root", default=None)
    ap.add_argument("--video-field", choices=["video_sync", "video_raw"], default="video_sync")

    # TRC options
    ap.add_argument("--trc-path", default=None, help="Path to TRC (default: <trial>/enhancer/enhancer_<trial>.trc)")
    ap.add_argument("--trc-suffix", default="_study", help="Only draw TRC markers whose names end with this suffix")
    ap.add_argument("--trc-frame-offset", type=int, default=0,
                    help="Offset added to video frame_index when aligning by frame. Useful for off-by-one cases.")

    # Frame/time selection for the video
    ap.add_argument("--frame-index", type=int, default=None, help="Use this exact video frame index")
    ap.add_argument("--time-sec", type=float, default=None, help="Or pick nearest frame to this time (sec)")

    # Labels & output
    ap.add_argument("--label-markers", action="store_true", help="Draw text labels for marker names")
    ap.add_argument("--out", default=None, help="PNG output (default: <trial>/enhancer_vis/frame_<f>_trc.png)")

    # Person selection in preds
    ap.add_argument("--person-index", type=int, default=-1)
    ap.add_argument("--min-mean-score", type=float, default=0.0)

    args = ap.parse_args()

    # 1) Manifest
    log_step("Loading and resolving manifest")
    manifest = load_manifest(args.manifest, args.paths)

    # Locate trial
    trial = None
    for subset, trials in (manifest.get("trials", {}) or {}).items():
        for t in trials:
            if t.get("id") == args.trial:
                trial = t
                break
        if trial:
            break
    if trial is None:
        raise SystemExit(f"[ERROR] Trial '{args.trial}' not found in manifest.")

    # Resolve paths (same style as rtmw3d viz)
    def decide_trial_root(manifest: dict, trial: dict, outputs_root_cli: str | None):
        base = manifest.get("output_dir")
        if not base:
            outputs_root = outputs_root_cli or Path.cwd() / "outputs"
            subj = manifest.get("subject_id", "subject")
            sess = manifest.get("session", "Session")
            cam = manifest.get("camera", "Cam")
            base = Path(outputs_root) / subj / sess / cam
        return Path(base) / trial["id"]

    trial_root = decide_trial_root(manifest, trial, args.outputs_root)
    enh_dir = trial_root / "enhancer"
    vis_dir = trial_root / "enhancer_vis"
    rtmw3d_dir = trial_root / "rtmw3d"
    vis_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(trial.get(args.video_field))
    trc_path = Path(args.trc_path) if args.trc_path else (enh_dir / f"enhancer_{args.trial}.trc")
    meta_path = trial_root / "meta.json"
    preds_path = rtmw3d_dir / "preds_metric.jsonl"
    enh_input_path = enh_dir / "enh_input_60hz.npz"

    log_info(f"Video path          : {video_path}")
    log_info(f"TRC path            : {trc_path}")
    log_info(f"meta.json           : {meta_path}")
    log_info(f"Enhancer cache      : {enh_input_path}")

    if not video_path.exists():
        raise SystemExit(f"[ERROR] video not found: {video_path}")
    if not trc_path.exists():
        raise SystemExit(f"[ERROR] TRC not found: {trc_path}")
    if not meta_path.exists():
        log_warn("meta.json not found (continuing; only affects names/edges).")

    # 2) Open video and decode sequentially up to target frame (like rtmw3d viz)
    log_step("Opening video")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

    # 2a) Decide the target frame index (but do NOT seek randomly)
    if args.frame_index is not None:
        fi_target = int(args.frame_index)
    elif args.time_sec is not None:
        fi_target = int(round(args.time_sec * fps))
    else:
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fi_target = max(0, nframes // 2)

    # 2b) SEQUENTIAL decode up to fi_target (exact, like the estimator)
    ok, frame_bgr = True, None
    fi_actual = -1
    for fi_actual in range(fi_target + 1):
        ok, frame_bgr = cap.read()
        if not ok:
            raise SystemExit(f"[ERROR] failed to read frame {fi_actual}")

    # Use OpenCV's timestamp if available (purely informational)
    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    t_frame = (pos_msec / 1000.0) if (pos_msec and not np.isnan(pos_msec) and pos_msec > 0) else (fi_actual / fps)
    log_info(f"Decoded sequentially: frame_index={fi_actual} (t≈{t_frame:.3f}s)")

    # 3) Load TRC, filter names by suffix
    log_step("Loading TRC")
    found = read_trc_find_frame(
        trc_path,
        frame_index=args.trc_frame_offset,           # <-- single source of truth for matching
        time_target=None,                # <-- do not provide time; avoids ambiguity
        tol_sec=1.0 / max(fps, 1.0),
        suffix_filter=args.trc_suffix
    )
    if found is None:
        raise SystemExit(f"[ERROR] No TRC row found (exact match required). Try --trc-frame-offset.")
    log_info(
        f"Selected TRC by frame: Frame#={found['frame_index']}  t_trc={found['time_sec']:.3f} s"
    )


    names_trc = found["keypoint_names"]
    Xmm_full = np.asarray(found["keypoints_xyz_mm"], dtype=float)
    
    # 4) Decide space (auto/px/mm) + projection mode
    h, w = frame_bgr.shape[:2]
    # interpret TRC columns X,Y as 2D if already in pixels
    XY_trc = Xmm_full[:, :2]

    def to_pixels_from_mm_simple(XY_mm):
        # Normalize metric coords to image: center & scale based on median pairwise distance
        pts = XY_mm.copy()
        ctr = np.nanmean(pts, axis=0)
        pts0 = pts - ctr
        valid = np.isfinite(pts0).all(axis=1)
        P = pts0[valid]
        if P.shape[0] >= 2:
            D = np.linalg.norm(P[None, :, :] - P[:, None, :], axis=-1)
            med = np.median(D[np.triu_indices_from(D, 1)])
            scale = (0.35 * min(w, h)) / (med + 1e-6)  # 35% of min dimension
        else:
            scale = 1.0
        px = pts0 * scale
        px += np.array([w * 0.5, h * 0.55])  # slightly below center
        return px

    found_y = read_jsonl_find_frame(preds_path,
                                frame_index=fi_actual,
                                time_target=None,
                                tol_sec=1.0/max(fps,1.0),
                                person_index=args.person_index,
                                min_mean_score=args.min_mean_score)
    Ypx = np.asarray(found_y["keypoints_px"], dtype=float)
    X3d_pred = np.asarray(found_y["keypoints_xyz_mm"], dtype=float)
    K = min(X3d_pred.shape[0], Ypx.shape[0])
    good = np.isfinite(X3d_pred[:K]).all(axis=1) & np.isfinite(Ypx[:K]).all(axis=1)
    X_fit = X3d_pred[:K][good]
    Y_fit = Ypx[:K][good]
    if X_fit.shape[0] >= 4:
        M, _ = _fit_affine_3d_to_2d(X_fit, Y_fit)
        if M is not None:
            X_aug = np.hstack([Xmm_full, np.ones((Xmm_full.shape[0], 1))])
            log_info(f"Xmm_full: {Xmm_full.shape} M: {M.shape}")
            kp_px = X_aug @ M
            log_info("Calibrated mm→px from RTMW3D preds; projected TRC via affine.")
        else:
            log_warn("Affine fit (preds) failed — using simple mm overlay.")
            kp_px = to_pixels_from_mm_simple(XY_trc)


    # Load enhancer input cache (OCAP20 mm + times)
    log_step("Loading enhancer cache (OCAP20 & times)")
    data = np.load(enh_input_path, allow_pickle=True)
    t_60 = data["t_60"].astype(float)
    ocap20_names = [str(x) for x in data["ocap20_names"].tolist()]
    Xc_mm = data["ocap20_centered_mm"].astype(float)   # (T,20,3)
    midhip_mm = data["midhip_mm"].astype(float)         # (T,3)

        # Absolute OCAP20 in mm at each time
    Xabs_mm = Xc_mm + midhip_mm[:, None, :]

    # Pick nearest time index
    i_time = int(np.argmin(np.abs(t_60 - t_frame)))
    log_info(f"Selected enhancer time index: {i_time}  (t_60={t_60[i_time]:.3f} s)")

    ocap20_xy_mm = Xabs_mm[i_time, :, :2]  # (20,2)

    kp2d_px = None
    kp2d_names = None
    if found_y is not None:
        kp2d_px = found_y["keypoints_px"]
        kp2d_names = found_y.get("keypoint_names", None)
        if kp2d_names is None:
            # Try meta.json for names
            meta_path = decide_trial_root(manifest, trial, args.outputs_root) / "meta.json"
            try:
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if isinstance(meta.get("keypoint_names"), list):
                        kp2d_names = meta.get("keypoint_names")
            except Exception:
                pass
        if kp2d_names is None:
            log_warn("No keypoint names available in RTMW3D 2D — similarity fit may be unavailable.")

    # Gather correspondences for similarity (OCAP20 only)
    have_similarity = False
    if kp2d_px is not None and kp2d_names is not None:
        name2i_3d = {n: i for i, n in enumerate(ocap20_names)}
        name2i_2d = {n: i for i, n in enumerate(kp2d_names)}
        common = [n for n in ocap20_names if n in name2i_2d]
        if len(common) >= 3:
            X = np.stack([ocap20_xy_mm[name2i_3d[n]] for n in common], axis=0)
            Y = np.stack([kp2d_px[name2i_2d[n]] for n in common], axis=0)
            # Drop NaNs
            mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
            X = X[mask]; Y = Y[mask]
            if X.shape[0] >= 3:
                try:
                    s, R, t = fit_similarity_2d(X, Y)
                    have_similarity = True
                    log_info(f"Similarity fit OK using {X.shape[0]} correspondences.")
                except Exception as e:
                    log_warn(f"Similarity fit failed: {e}")
        else:
            log_warn("Too few OCAP20↔RTMW3D name overlaps for similarity fit; will use simple fit.")

        # 7) Projection to pixels
    def project_xy_mm_to_px(XY_mm):
        if XY_mm is None:
            return None
        if have_similarity:
            return apply_similarity_2d(XY_mm, s, R, t)

    body_px = project_xy_mm_to_px(kp_px)

    # 5) Draw
    log_step("Drawing overlay")
    img = frame_bgr.copy()

    if isinstance(body_px, np.ndarray) and body_px.ndim == 2 and body_px.shape[1] == 2:
        log_info("Center normalized mm projection (with rotation)")
        # manual angle example: angle_deg=15
        # auto PCA: angle_deg=None (default), auto_angle='pca'
        body_px = normalize_center_with_rotation(
            body_px, img,
            small_span_thresh=50,
            fill=0.5,
            target_xy_frac=(0.5, 0.5),
            angle_deg=0,          # set degrees here to force a rotation
            auto_angle=None         # or 'none' to skip auto-rotation
        )

    draw_points(img, body_px, color=(255, 0, 255), label_prefix=("T" if args.label_markers else None), names=names_trc)

    mode = f"frame {fi_actual}  t={t_frame:.3f}s;)"
    put_header(img, mode)
    draw_legend(img, [(f"TRC markers (*{args.trc_suffix})", (255, 0, 255))])

    # 6) Save
    default_out = vis_dir / f"frame_{fi_actual}_trc_mm.png"
    out_path = Path(args.out) if args.out else default_out
    cv2.imwrite(str(out_path), img)
    log_done(f"Saved visualization : {out_path}")


if __name__ == "__main__":
    main()
