#!/usr/bin/env python3
# viz_overlay_on_video.py
"""
Overlay RTMW3D + MoCap (projected) on a single video frame.

What it does
------------
- Loads manifest and resolves paths (your IO.load_manifest).
- Reads a frame from the video (default: video_sync).
- Gets the matching RTMW3D jsonl line (by frame_index or nearest time).
- Loads TRC (100 Hz) and picks the closest timestamp to the frame time,
  optionally adding a global offset (e.g., from validate report.json).
- Converts MoCap joints to the 20-pt "OpenCap-like" set (ocap20).
- Transforms MoCap 3D from mocap frame -> video world (mocap_to_video).
- Projects 3D -> pixels via intrinsics/distortion/extrinsics (OpenCV).
- Draws both sets on the video frame with simple skeleton edges.
- Saves PNG to <trial>/rtmw3d_eval/overlay_frame_<idx>.png

Usage
-----
python viz_overlay_on_video.py \
  -m manifests/OpenCapDataset/subject2.yaml \
  -p config/paths.yaml \
  --trial walking1 \
  --frame-index 240 \
  --offset 0.07 \
  --video-field video_sync

Notes
-----
- If you don’t know the offset: set --offset 0 and it will still work (the
  video_sync files are usually close). You can also read the offset we estimated
  in validate_rtmw3d_vs_mocap.py: <trial>/rtmw3d_eval/report.json
- If calibration parsing fails, you’ll still get RTMW3D overlay alone.
"""

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import yaml

from IO.load_manifest import load_manifest


# ---------------------------
# Logging
# ---------------------------

def log_step(msg): print(f"[STEP] {msg}")
def log_info(msg): print(f"[INFO] {msg}")
def log_warn(msg): print(f"[WARN] {msg}")
def log_done(msg): print(f"[DONE] {msg}")


# ---------------------------
# TRC loader (Vicon-like)
# ---------------------------

def load_trc(path: Path):
    """
    Read TRC into (time_s, markers_dict{name: (T,3) in mm}, data_rate).
    Expects the header layout you pasted earlier.
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
        if not line.strip():
            continue
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
# RTMW3D jsonl helpers
# ---------------------------

def read_jsonl_find_frame(preds_path: Path, frame_index=None, time_target=None, tol_sec=0.02,
                          person_index=-1, min_mean_score=0.0):
    """
    Return dict for the chosen line:
      { 'time_sec', 'frame_index', 'keypoints_px' (K,2), 'keypoint_names' optional }
    Selection:
      - If frame_index is given, pick that exact frame.
      - Else, find nearest time to time_target within tol_sec.
    Person selection: person_index or highest mean_score.
    """
    best = None
    best_dt = 1e9
    with open(preds_path, 'r', encoding='utf-8') as f:
        for ln in f:
            o = json.loads(ln)
            if frame_index is not None and int(o.get('frame_index', -999)) != frame_index:
                continue
            if time_target is not None and frame_index is None:
                dt = abs(float(o.get('time_sec', 0.0)) - float(time_target))
                if dt > tol_sec:
                    continue
            else:
                dt = 0.0

            persons = o.get('persons') or []
            if not persons:
                continue
            if person_index == -1:
                ms = [p.get('mean_score', 0.0) for p in persons]
                pi = int(np.argmax(ms)) if ms else 0
            else:
                pi = min(max(0, person_index), len(persons) - 1)
            p = persons[pi]
            if p.get('mean_score', 1.0) < min_mean_score:
                continue
            kpx = np.asarray(p.get('keypoints_px'), dtype=float)
            if kpx.ndim != 2 or kpx.shape[1] != 2:
                continue

            # keep first exact match; else keep nearest
            if dt < best_dt:
                best_dt = dt
                best = dict(time_sec=float(o.get('time_sec', 0.0)),
                            frame_index=int(o.get('frame_index', -1)),
                            keypoints_px=kpx)
            if frame_index is not None:
                break

    return best


# ---------------------------
# OCAP20 joints & edges
# ---------------------------

OCAP20 = [
    "neck", "mid_hip",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_small_toe", "right_small_toe",
    "left_big_toe", "right_big_toe",
]

OCAP20_EDGES = [
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_ankle", "left_big_toe"), ("left_ankle", "left_small_toe"),
    ("right_ankle", "right_heel"), ("right_ankle", "right_big_toe"), ("right_ankle", "right_small_toe"),
    ("left_hip", "left_shoulder"), ("right_hip", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("neck", "left_shoulder"), ("neck", "right_shoulder"),
    ("mid_hip", "left_hip"), ("mid_hip", "right_hip"),
]

def build_selector(markers_dict):
    lower_map = {k.lower(): k for k in markers_dict.keys()}
    def pick(*cands):
        for c in cands:
            k = lower_map.get(c.lower())
            if k is not None:
                return k
        return None
    return pick

def compose_ocap20_from_trc_at_time(t_trc, markers, t_query, joint_names=OCAP20):
    """
    Returns dict name->3D(mm) for OCAP20 at closest TRC sample to t_query.
    Uses robust marker picking and simple compositions for neck/mid_hip/toes.
    """
    ti = int(np.argmin(np.abs(t_trc - t_query)))
    sel = {k: v[ti] for k, v in markers.items()}  # k -> (3,)
    pick = build_selector(markers)

    out = {}

    # Hips
    L_HJC = pick("L_HJC", "L_HJC_reg")
    R_HJC = pick("R_HJC", "R_HJC_reg")
    if L_HJC: out["left_hip"]  = sel[L_HJC]
    if R_HJC: out["right_hip"] = sel[R_HJC]
    if L_HJC and R_HJC:
        out["mid_hip"] = 0.5 * (sel[L_HJC] + sel[R_HJC])

    # Knees / Ankles
    if pick("L_knee"):   out["left_knee"]  = sel[pick("L_knee")]
    if pick("r_knee"):   out["right_knee"] = sel[pick("r_knee")]
    if pick("L_ankle"):  out["left_ankle"] = sel[pick("L_ankle")]
    if pick("r_ankle"):  out["right_ankle"] = sel[pick("r_ankle")]

    # Feet (heels + toes)
    if pick("L_calc"):   out["left_heel"]  = sel[pick("L_calc")]
    if pick("r_calc"):   out["right_heel"] = sel[pick("r_calc")]
    if pick("L_5meta"):  out["left_small_toe"]  = sel[pick("L_5meta")]
    if pick("r_5meta"):  out["right_small_toe"] = sel[pick("r_5meta")]
    if pick("L_toe"):    out["left_big_toe"]    = sel[pick("L_toe")]
    if pick("r_toe"):    out["right_big_toe"]   = sel[pick("r_toe")]

    # Shoulders
    if pick("L_Shoulder"): out["left_shoulder"]  = sel[pick("L_Shoulder")]
    if pick("R_Shoulder"): out["right_shoulder"] = sel[pick("R_Shoulder")]

    # Neck ~ average C7 with shoulder midpoint if available
    c7 = pick("C7")
    if pick("L_Shoulder") and pick("R_Shoulder"):
        mid_sh = 0.5 * (sel[pick("L_Shoulder")] + sel[pick("R_Shoulder")])
        if c7:
            out["neck"] = 0.5 * (mid_sh + sel[c7])
        else:
            out["neck"] = mid_sh
    elif c7:
        out["neck"] = sel[c7]

    return out, ti


# ---------------------------
# Calibration IO
# ---------------------------

def load_intrinsics_extrinsics(pkl_path: Path):
    """
    Expects a pickle with keys like:
      intrinsicMat (3x3), distortion (1x5 or 5,), rotation (3x3), translation (3x1)
    Returns (K, dist, rvec, tvec) or (None, None, None, None) if not available.
    """
    try:
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)
        K = np.asarray(d.get('intrinsicMat'), dtype=float)
        dist = np.asarray(d.get('distortion'), dtype=float).reshape(-1)
        R = np.asarray(d.get('rotation'), dtype=float)
        t = np.asarray(d.get('translation'), dtype=float).reshape(3)
        rvec, _ = cv2.Rodrigues(R)
        return K, dist, rvec.reshape(3, 1), t.reshape(3, 1)
    except Exception as e:
        log_warn(f"Failed to read intrinsics/extrinsics pickle: {e}")
        return None, None, None, None

def load_mocap_to_video(yaml_path: Path):
    """
    Try to read a mocap->video transform from YAML.
    Accepts either:
      - a 4x4 matrix under keys: 'T' or 'matrix' or 'transform'
      - separate 'rotation' (3x3) and 'translation' (3,)
    Returns (R, t) where X_video = R @ X_mocap + t
    """
    try:
        d = yaml.safe_load(open(yaml_path, 'r'))
    except Exception as e:
        log_warn(f"Failed to read mocap_to_video YAML: {e}")
        return None, None

    def as_array(x): return np.asarray(x, dtype=float)

    # 4x4 variants
    for key in ('T', 'matrix', 'transform'):
        if key in d:
            M = as_array(d[key])
            if M.shape == (4, 4):
                R = M[:3, :3]
                t = M[:3, 3:4]
                return R, t
    # R/t variant
    R = d.get('rotation', None)
    t = d.get('translation', None)
    if R is not None and t is not None:
        R = as_array(R).reshape(3, 3)
        t = as_array(t).reshape(3, 1)
        return R, t

    log_warn("mocap_to_video format not recognized; expected 4x4 'T' or {rotation, translation}.")
    return None, None


# ---------------------------
# Projection
# ---------------------------

def project_points_mm(X_mm, K, dist, rvec, tvec):
    """
    X_mm: (N,3) in the same world frame used by (rvec, tvec).
    Returns pixels (N,2) using cv2.projectPoints (handles radial/tangential dist).
    """
    X = X_mm.reshape(-1, 1, 3).astype(np.float64)
    img_pts, _ = cv2.projectPoints(X, rvec, tvec, K, dist)
    return img_pts.reshape(-1, 2)


# ---------------------------
# Drawing
# ---------------------------

def draw_skeleton(img, pts_px, names, edges, color=(0, 255, 0), thickness=2, radius=3):
    name2i = {n: i for i, n in enumerate(names)}
    h, w = img.shape[:2]
    # points
    for i, p in enumerate(pts_px):
        if not np.all(np.isfinite(p)):  # skip NaNs
            continue
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    # edges
    for a, b in edges:
        ia, ib = name2i.get(a), name2i.get(b)
        if ia is None or ib is None:
            continue
        pa = pts_px[ia] if ia < len(pts_px) else None
        pb = pts_px[ib] if ib < len(pts_px) else None
        if pa is None or pb is None:
            continue
        if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
            continue
        xa, ya = int(round(pa[0])), int(round(pa[1]))
        xb, yb = int(round(pb[0])), int(round(pb[1]))
        if (0 <= xa < w and 0 <= ya < h) or (0 <= xb < w and 0 <= yb < h):
            cv2.line(img, (xa, ya), (xb, yb), color, thickness, lineType=cv2.LINE_AA)


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Overlay RTMW3D + projected MoCap on a video frame.")
    ap.add_argument("-m", "--manifest", required=True)
    ap.add_argument("-p", "--paths", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--outputs-root", default=None)
    ap.add_argument("--video-field", choices=["video_sync", "video_raw"], default="video_sync")

    # Frame/time selection
    ap.add_argument("--frame-index", type=int, default=None, help="Use this exact video frame index")
    ap.add_argument("--time-sec", type=float, default=None, help="Or pick nearest frame to this time (sec)")

    # Person selection in preds
    ap.add_argument("--person-index", type=int, default=-1)
    ap.add_argument("--min-mean-score", type=float, default=0.0)

    # Timing
    ap.add_argument("--offset", type=float, default=0.0, help="Add to TRC time to align to video (sec).")

    # Output
    ap.add_argument("--out", default=None, help="Output PNG path (default: <trial>/rtmw3d_eval/overlay_frame_<f>.png)")

    args = ap.parse_args()

    # 1) Manifest
    log_step("Loading and resolving manifest")
    manifest = load_manifest(args.manifest, args.paths)

    # Locate trial
    trial = None
    for subset, trials in manifest.get("trials", {}).items():
        for t in trials:
            if t.get("id") == args.trial:
                trial = t
                break
        if trial:
            break
    if trial is None:
        raise SystemExit(f"[ERROR] Trial '{args.trial}' not found in manifest.")

    # Resolve paths
    def decide_trial_root(manifest: dict, trial: dict, outputs_root_cli: str | None):
        base = manifest.get('output_dir')
        if not base:
            outputs_root = outputs_root_cli or Path.cwd() / "outputs"
            subj = manifest.get('subject_id', 'subject')
            sess = manifest.get('session', 'Session')
            cam = manifest.get('camera', 'Cam')
            base = Path(outputs_root) / subj / sess / cam
        return Path(base) / trial['id']

    trial_root = decide_trial_root(manifest, trial, args.outputs_root)
    rtmw3d_dir = trial_root / "rtmw3d"
    eval_dir = trial_root / "rtmw3d_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(trial.get(args.video_field))
    preds_path = rtmw3d_dir / "preds.jsonl"
    meta_path = trial_root / "meta.json"
    trc_path = Path(trial.get("mocap_trc", ""))

    log_info(f"Video path          : {video_path}")
    log_info(f"Preds jsonl         : {preds_path}")
    log_info(f"TRC path            : {trc_path}")
    log_info(f"meta.json           : {meta_path}")

    if not video_path.exists():
        raise SystemExit(f"[ERROR] video not found: {video_path}")
    if not preds_path.exists():
        raise SystemExit(f"[ERROR] preds.jsonl not found: {preds_path}")
    if not meta_path.exists():
        log_warn("meta.json not found (continuing; only affects display labels).")
    if not trc_path.exists():
        log_warn("TRC not found; will draw RTMW3D only.")

    # 2) Open video, pick frame
    log_step("Opening video")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if args.frame_index is not None:
        fi = int(args.frame_index)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    elif args.time_sec is not None:
        fi = int(round(args.time_sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    else:
        # default: middle frame
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fi = max(0, nframes // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)

    ok, frame_bgr = cap.read()
    if not ok:
        raise SystemExit("[ERROR] failed to read selected frame")
    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    t_frame = (pos_msec / 1000.0) if (pos_msec and not np.isnan(pos_msec) and pos_msec > 0) else (fi / fps)
    log_info(f"Using frame_index   : {fi}  (t={t_frame:.3f} s)")

    # 3) RTMW3D: find 2D for that frame or nearest in time
    log_step("Fetching RTMW3D 2D keypoints for the chosen frame")
    found = read_jsonl_find_frame(
        preds_path,
        frame_index=fi,
        time_target=t_frame,
        tol_sec=1.0 / max(fps, 1.0),
        person_index=args.person_index,
        min_mean_score=args.min_mean_score,
    )
    kp_px = None
    if found is None:
        log_warn("Could not find matching RTMW3D line for this frame; skipping RTMW3D overlay.")
    else:
        kp_px = found["keypoints_px"]
        log_info(f"RTMW3D line time    : {found['time_sec']:.3f} s | K={kp_px.shape[0]} 2D joints")

    # 4) MoCap → project to pixels
    proj_pts = None
    if trc_path.exists():
        log_step("Reading TRC and projecting to pixels via calibration")
        # Load TRC
        t_trc, markers, trc_rate = load_trc(trc_path)
        t_query = t_frame + float(args.offset)
        log_info(f"TRC ts picked       : {t_query:.3f} s (offset applied: {args.offset:+.3f} s)")

        # Compose ocap20 joints @ t_query
        joints_3d_dict, trc_idx = compose_ocap20_from_trc_at_time(t_trc, markers, t_query)
        names = [n for n in OCAP20 if n in joints_3d_dict]
        if not names:
            log_warn("No OCAP20 joints found in TRC at that time; cannot project MoCap.")
        else:
            X_mocap = np.stack([joints_3d_dict[n] for n in names], axis=0)  # (N,3) mm

            # Load calibration
            K = dist = rvec = tvec = None
            mocap_R = mocap_t = None
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding='utf-8'))
                except Exception:
                    pass

            cal = (meta.get("calibration") or {})
            pkl_path = Path(cal.get("intrinsics_extrinsics") or "")
            T_yaml = Path(cal.get("mocap_to_video") or "")

            if pkl_path and pkl_path.exists():
                K, dist, rvec, tvec = load_intrinsics_extrinsics(pkl_path)
            else:
                log_warn("intrinsics_extrinsics pickle missing; cannot project MoCap.")

            if T_yaml and T_yaml.exists():
                mocap_R, mocap_t = load_mocap_to_video(T_yaml)
            else:
                log_warn("mocap_to_video YAML missing; assuming MoCap is already in video world.")

            if K is not None and dist is not None and rvec is not None and tvec is not None:
                # Transform MoCap to video world if transform available
                if mocap_R is not None and mocap_t is not None:
                    X_video = (X_mocap @ mocap_R.T) + mocap_t.reshape(1, 3)
                else:
                    X_video = X_mocap

                # Project
                try:
                    proj_pts = project_points_mm(X_video, K, dist, rvec, tvec)
                    proj_names = names
                    log_info(f"Projected MoCap pts : {proj_pts.shape}")
                except Exception as e:
                    log_warn(f"cv2.projectPoints failed: {e}")
            else:
                log_warn("Missing camera intrinsics/extrinsics; skipping MoCap projection.")

    # 5) Draw overlay
    log_step("Drawing overlay")
    img = frame_bgr.copy()

    # Draw MoCap projection first (blue)
    if proj_pts is not None:
        draw_skeleton(img, proj_pts, proj_names, OCAP20_EDGES, color=(255, 170, 0), thickness=2, radius=3)

    # Draw RTMW3D 2D (green x markers + dashed edges)
    if kp_px is not None:
        # Try to subset to OCAP20 names if meta has them; else draw all pts as x
        meta = {}
        kp_names = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
                kp_names = meta.get("keypoint_names")
            except Exception:
                pass

        if kp_names and isinstance(kp_names, list):
            name2i = {n: i for i, n in enumerate(kp_names)}
            idxs = [name2i[n] for n in OCAP20 if n in name2i]
            kp_sel = kp_px[idxs] if len(idxs) else kp_px
            names_sel = [n for n in OCAP20 if n in name2i] if len(idxs) else None
        else:
            kp_sel = kp_px
            names_sel = None

        # Points: draw as 'x'
        h, w = img.shape[:2]
        for p in kp_sel:
            if not np.all(np.isfinite(p)):
                continue
            x, y = int(round(p[0])), int(round(p[1]))
            if 0 <= x < w and 0 <= y < h:
                cv2.drawMarker(img, (x, y), (0, 255, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)

        # Edges if we have names
        if names_sel is not None:
            draw_skeleton(img, kp_sel, names_sel, OCAP20_EDGES, color=(0, 255, 0), thickness=1, radius=0)

    # Labels
    cv2.putText(img, f"frame {fi}  t={t_frame:.3f}s  offset={args.offset:+.3f}s",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 3, cv2.LINE_AA)
    cv2.putText(img, f"frame {fi}  t={t_frame:.3f}s  offset={args.offset:+.3f}s",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # --- Legend box (bottom-left) ---
    legend_items = [
        ("MoCap (projected)", (255, 170, 0)),  # orange
        ("RTMW3D",            (0, 255, 0)),    # green
    ]
    pad = 10
    x0, y0 = 12, img.shape[0] - 12
    line_h = 24
    box_w = 260
    box_h = pad*2 + line_h*len(legend_items)
    cv2.rectangle(img, (x0-8, y0-box_h), (x0-8+box_w, y0), (0, 0, 0), -1)
    cv2.rectangle(img, (x0-8, y0-box_h), (x0-8+box_w, y0), (255, 255, 255), 1)

    for i, (label, bgr) in enumerate(legend_items):
        y = y0 - box_h + pad + i*line_h + 16
        # color swatch
        cv2.circle(img, (x0, y-6), 6, bgr, -1, lineType=cv2.LINE_AA)
        # text (white with dark outline)
        cv2.putText(img, label, (x0+18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(img, label, (x0+18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # 6) Save
    out_path = Path(args.out) if args.out else (eval_dir / f"overlay_frame_{fi}.png")
    cv2.imwrite(str(out_path), img)
    log_done(f"Saved overlay PNG   : {out_path}")

if __name__ == "__main__":
    main()
