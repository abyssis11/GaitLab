#!/usr/bin/env python3
# viz_rtmw3d_on_video.py
"""
Visualize RTMW3D keypoints for a single video frame (2D px or 3D-mm projected).

What it does
------------
- Loads the dataset manifest (your IO.load_manifest) and resolves paths.
- Opens the requested video (default: video_sync) and selects a frame by index or time.
- Reads RTMW3D predictions from either `preds.jsonl` or `preds_metric.jsonl`.
- Finds the matching jsonl line (exact frame_index or nearest in time).
- Selects a person (by index or highest mean_score).
- Draws the 2D keypoints;
  - **OCAP20 mode**: draws only the OCAP20 subset (and OCAP20 edges if names exist).
  - **All mode**: draws all available keypoints as markers (no edges).
- Saves a PNG to <trial>/rtmw3d_vis/frame_<idx>.png by default.

Usage
-----
python viz_rtmw3d_on_video.py \
  -m manifests/OpenCapDataset/subject2.yaml \
  -p config/paths.yaml \
  --trial walking1 \
  --frame-index 240 \
  --preds-type preds \
  --kp-set ocap20            # or: all

Notes
-----
- This script intentionally **does not** use MoCap, calibration, or projection. It's
  only for lightweight 2D visualization of RTMW3D results on the video frame.
- If keypoint names are provided (either in meta.json or in the jsonl), OCAP20 mode will
  also draw OCAP20 skeleton edges.
"""

import argparse
import json
from pathlib import Path

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
# OCAP20 joints & edges (for optional skeleton lines)
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


# ---------------------------
# JSONL readers (robust to slight schema variants)
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
# Drawing
# ---------------------------

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

def draw_skeleton(img, pts_px, names, edges, color=(0, 255, 0), thickness=1, radius=0):
    name2i = {n: i for i, n in enumerate(names)}
    h, w = img.shape[:2]
    # points
    for i, p in enumerate(pts_px):
        if not np.all(np.isfinite(p)):
            continue
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), max(1, radius), color, -1, lineType=cv2.LINE_AA)
    # edges
    for a, b in edges:
        ia, ib = name2i.get(a), name2i.get(b)
        if ia is None or ib is None:
            continue
        if ia >= len(pts_px) or ib >= len(pts_px):
            continue
        pa, pb = pts_px[ia], pts_px[ib]
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
    ap = argparse.ArgumentParser(description="Visualize RTMW3D 2D keypoints on a video frame.")
    ap.add_argument("-m", "--manifest", required=True)
    ap.add_argument("-p", "--paths", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--outputs-root", default=None)
    ap.add_argument("--video-field", choices=["video_sync", "video_raw"], default="video_sync")

    # Which predictions file to use
    ap.add_argument(
        "--preds-type",
        choices=["preds", "preds_metric"],
        default="preds",
        help="Choose which jsonl to visualize: preds.jsonl or preds_metric.jsonl",
    )
    ap.add_argument(
        "--preds-path",
        default=None,
        help="Optional explicit path to a predictions jsonl (overrides --preds-type)",
    )

    # Keypoint set selection
    ap.add_argument(
        "--kp-set",
        choices=["ocap20", "all"],
        default="ocap20",
        help="Which keypoints to draw: OCAP20 subset (with edges) or all available (markers only).",
    )
    ap.add_argument(
        "--space", choices=["auto", "px", "mm"], default="auto",
        help="Coordinate space to visualize. 'mm' uses keypoints_xyz_mm (preds_metric) projected to the image via an affine fit; 'px' uses keypoints_px; 'auto': mm for preds_metric if available, else px."
    )

    # Frame/time selection
    ap.add_argument("--frame-index", type=int, default=None, help="Use this exact video frame index")
    ap.add_argument("--time-sec", type=float, default=None, help="Or pick nearest frame to this time (sec)")

    # Person selection in preds
    ap.add_argument("--person-index", type=int, default=-1)
    ap.add_argument("--min-mean-score", type=float, default=0.0)

    # Output
    ap.add_argument("--out", default=None, help="Output PNG path (default: <trial>/rtmw3d_vis/frame_<f>.png)")

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

    # Resolve paths
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
    rtmw3d_dir = trial_root / "rtmw3d"
    vis_dir = trial_root / "rtmw3d_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(trial.get(args.video_field))
    if args.preds_path:
        preds_path = Path(args.preds_path)
    else:
        preds_path = rtmw3d_dir / ("preds.jsonl" if args.preds_type == "preds" else "preds_metric.jsonl")
    meta_path = trial_root / "meta.json"

    log_info(f"Video path          : {video_path}")
    log_info(f"Preds jsonl         : {preds_path}")
    log_info(f"meta.json           : {meta_path}")

    if not video_path.exists():
        raise SystemExit(f"[ERROR] video not found: {video_path}")
    if not preds_path.exists():
        raise SystemExit(f"[ERROR] predictions jsonl not found: {preds_path}")
    if not meta_path.exists():
        log_warn("meta.json not found (continuing; only affects names/edges).")

    # 2) Open video
    log_step("Opening video")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

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

    # 3) RTMW3D: fetch predictions for THIS exact frame index (index only; ignore time)
    log_step("Fetching RTMW3D keypoints for the chosen frame")
    found = read_jsonl_find_frame(
        preds_path,
        frame_index=fi_actual,           # <-- single source of truth for matching
        time_target=None,                # <-- do not provide time; avoids ambiguity
        tol_sec=1.0 / max(fps, 1.0),
        person_index=args.person_index,
        min_mean_score=args.min_mean_score,
    )

    # Optional sanity log
    log_info(("decoded:", fi_actual, f"{t_frame:.3f}s",
            "| preds:", found["frame_index"], f'{found["time_sec"]:.3f}s'))


    if found is None:
        raise SystemExit("[ERROR] Could not find matching RTMW3D line for this frame.")

    space = args.space
    if space == "auto":
        space = "mm" if (args.preds_type == "preds_metric" and "keypoints_xyz_mm" in found) else "px"

    if space == "mm":
        if "keypoints_xyz_mm" not in found:
            log_warn("No keypoints_xyz_mm found; falling back to px.")
            space = "px"

    if space == "mm":
        Xmm = found.get("keypoints_xyz_mm")
        Ypx = found.get("keypoints_px")
        kp_px = None
        if Xmm is not None and Ypx is not None:
            M, mask = _fit_affine_3d_to_2d(Xmm, Ypx)
            if M is not None:
                X_aug = np.hstack([Xmm, np.ones((Xmm.shape[0], 1))])
                kp_px = X_aug @ M
                log_info("Projected mm→px via affine fit using overlapping joints.")
        if kp_px is None and Xmm is not None:
            log_warn("Affine fit unavailable; using normalized orthographic placement for mm coords.")
            X = np.asarray(Xmm, dtype=float)
            Xc = X - np.nanmean(X, axis=0)
            import numpy as _np
            valid = _np.all(_np.isfinite(Xc), axis=1)
            pts = Xc[valid][:, :2]
            if pts.shape[0] >= 2:
                dists = _np.linalg.norm(pts[None, :, :] - pts[:, None, :], axis=-1)
                med = _np.median(dists[_np.triu_indices_from(dists, 1)])
                scale = 200.0 / (med + 1e-6)
            else:
                scale = 1.0
            kp_px = Xc[:, :2] * scale
    else:
        kp_px = found["keypoints_px"]

    _mode = "mm→px" if space == "mm" else "2D"
    log_info(f"RTMW3D line time    : {found['time_sec']:.3f} s | K={kp_px.shape[0]} {_mode} joints")

    # 4) Optional names
    kp_names = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta.get("keypoint_names"), list):
                kp_names = meta.get("keypoint_names")
        except Exception:
            pass
    if kp_names is None:
        kp_names = found.get("keypoint_names")

    # 5) Choose which keypoints to draw
    names_sel = None
    if args.kp_set == "ocap20":
        if isinstance(kp_names, list):
            name2i = {n: i for i, n in enumerate(kp_names)}
            idxs = [name2i[n] for n in OCAP20 if n in name2i]
            if len(idxs):
                kp_sel = kp_px[idxs]
                names_sel = [n for n in OCAP20 if n in name2i]  # for OCAP20 edges
            else:
                log_warn("OCAP20 names not present in predictions; drawing all points without edges.")
                kp_sel = kp_px
        else:
            log_warn("No keypoint names available; drawing all points.")
            kp_sel = kp_px
    else:  # args.kp_set == "all"
        kp_sel = kp_px
        # names_sel stays None -> no edges in 'all' mode

    # 6) Draw overlay
    log_step("Drawing overlay")
    img = frame_bgr.copy()

    # Center normalized mm projection (fallback) if needed
    if isinstance(kp_px, np.ndarray) and kp_px.ndim == 2 and kp_px.shape[1] == 2:
        h, w = img.shape[:2]
        span = float(max(np.ptp(kp_px[:,0]) + 1e-6, np.ptp(kp_px[:,1]) + 1e-6))
        if span > 0 and (span < 50):
            s = 0.4 * min(w, h) / span
            kp_px = kp_px * s
        cx, cy = int(w * 0.5), int(h * 0.5)
        mx, my = int(np.nanmean(kp_px[:,0])), int(np.nanmean(kp_px[:,1]))
        kp_px = kp_px + np.array([cx - mx, cy - my])

    # Points as tilted crosses
    h, w = img.shape[:2]
    for p in kp_sel:
        if not np.all(np.isfinite(p)):
            continue
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.drawMarker(img, (x, y), (0, 255, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)

    # Edges only for OCAP20 mode when names available
    if names_sel is not None:
        draw_skeleton(img, kp_sel, names_sel, OCAP20_EDGES, color=(0, 255, 0), thickness=1, radius=0)

    # Labels
    label_mode = ("mm/affine" if space == "mm" else "px") + " " + ("OCAP20" if args.kp_set == "ocap20" and names_sel is not None else ("all" if args.kp_set == "all" else "points"))
    cv2.putText(
        img,
        f"frame {args.frame_index}  t={t_frame:.3f}s  ({label_mode})",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (10, 10, 10),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"frame {args.frame_index}  t={t_frame:.3f}s  ({label_mode})",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Legend box (bottom-left)
    legend_text = f"RTMW3D ({'OCAP20' if args.kp_set == 'ocap20' else 'all'})"
    legend_items = [(legend_text, (0, 255, 0))]
    pad = 10
    x0, y0 = 12, img.shape[0] - 12
    line_h = 24
    box_w = 220
    box_h = pad * 2 + line_h * len(legend_items)
    cv2.rectangle(img, (x0 - 8, y0 - box_h), (x0 - 8 + box_w, y0), (0, 0, 0), -1)
    cv2.rectangle(img, (x0 - 8, y0 - box_h), (x0 - 8 + box_w, y0), (255, 255, 255), 1)
    for i, (label, bgr) in enumerate(legend_items):
        y = y0 - box_h + pad + i * line_h + 16
        cv2.circle(img, (x0, y - 6), 6, bgr, -1, lineType=cv2.LINE_AA)
        cv2.putText(img, label, (x0 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(img, label, (x0 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # 7) Save
    default_out = vis_dir / f"frame_{args.frame_index}_{'preds' if args.preds_type == 'preds' else 'preds_metric'}_{space}.png"
    out_path = Path(args.out) if args.out else default_out
    cv2.imwrite(str(out_path), img)
    log_done(f"Saved visualization : {out_path}")


if __name__ == "__main__":
    main()
