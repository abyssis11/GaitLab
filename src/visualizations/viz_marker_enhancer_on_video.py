#!/usr/bin/env python3
# viz_marker_enhancer_on_video.py
"""
Visualize Marker‑Enhancer 3D marker predictions on a video frame (as a 2D overlay).

Overview
--------
- Loads your dataset manifest (IO.load_manifest) and resolves paths like the RTMW3D visualizer.
- Opens the requested video (default: `video_sync`) and selects a frame by index or time.
- Reads RTMW3D 2D keypoints for that frame to estimate a simple 2D similarity projection
  (scale/rotation/translation) from OCAP20 (3D, mm) → pixels.
  *Uses `enhancer/enh_input_60hz.npz` OCAP20 (3D) and `rtmw3d/preds*.jsonl` (2D).* 
- Loads Marker‑Enhancer outputs (body 35 and/or arms 8 markers) at the nearest time index.
- Projects the 3D markers to the image plane with the estimated 2D similarity and draws them.
- Saves a PNG to `<trial>/enhancer_vis/frame_<idx>.png` by default.

Why similarity projection?
--------------------------
We avoid requiring camera intrinsics/extrinsics. RTMW3D metric predictions are in a
camera-like coordinate system; treating X/Y mm as the image plane and estimating a
2D similarity from OCAP20→2D provides a reasonable overlay without calibration.
If the 2D predictions are missing or names don’t match, we fall back to a simple
center+scale fit to the image.

Usage
-----
python src/visualizations/viz_marker_enhancer_on_video.py \
  -m manifests/OpenCapDataset/subject2.yaml \
  -p config/paths.yaml \
  --trial walking1 \
  --frame-index 240 \
  --preds-type preds \
  --set both                  # body | arms | both

Or by time:
python viz_marker_enhancer_on_video.py -m <manifest> -p <paths> --trial <trial> --time-sec 3.2

Notes
-----
- Expects you already ran the prep (and optionally the models) so that
  `<trial>/enhancer/enh_input_60hz.npz` and `body_pred_mm_Tx35x3.npz` / `arms_pred_mm_Tx8x3.npz`
  exist.
- If only one of body/arms exists, `--set both` will draw what is available.
- Labels can be toggled with `--label-markers` (off by default).
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
# JSONL readers (robust to slight schema variants) — reused/adapted
# ---------------------------

def _extract_person_list(rec: dict):
    for key in ("persons", "people", "detections", "predictions"):
        if key in rec and isinstance(rec[key], list):
            return rec[key]
    for key in ("person", "prediction"):
        if key in rec and isinstance(rec[key], dict):
            return [rec[key]]
    return []


def _extract_keypoints_2d(p: dict):
    candidates = [
        "keypoints_px", "keypoints2d", "keypoints_2d",
        "kpts_px", "kps2d", "pose2d", "pose_2d", "keypoints",
    ]
    for k in candidates:
        if k not in p or p[k] is None:
            continue
        arr = p[k]
        if isinstance(arr, dict) and "x" in arr and "y" in arr:
            x = np.asarray(arr["x"], dtype=float).reshape(-1)
            y = np.asarray(arr["y"], dtype=float).reshape(-1)
            if x.shape[0] == y.shape[0]:
                return np.stack([x, y], axis=-1)
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
    return 1.0


def read_jsonl_find_frame(preds_path: Path, frame_index=None, time_target=None, tol_sec=0.02,
                          person_index=-1, min_mean_score=0.0):
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
            if kpx is None:
                continue
            if _extract_mean_score(p) < float(min_mean_score):
                continue
            entry = {
                "time_sec": float(rec.get("time_sec", 0.0)),
                "frame_index": int(rec.get("frame_index", -1)),
                "keypoints_px": np.asarray(kpx, dtype=float),
            }
            kp_names = rec.get("keypoint_names") or p.get("keypoint_names")
            if isinstance(kp_names, list) and all(isinstance(x, str) for x in kp_names):
                entry["keypoint_names"] = kp_names
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
# 2D Similarity (Umeyama) for X→Y mapping
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
    return (s * (R @ Pxy.T)).T + t.reshape(1, 2)


# ---------------------------
# Drawing helpers
# ---------------------------

def draw_points(img, pts_px, color, label_prefix=None, names=None, radius=4, thickness=2, alpha=1.0):
    overlay = img.copy()
    h, w = img.shape[:2]
    for i, p in enumerate(pts_px):
        if not np.all(np.isfinite(p)):
            continue
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(overlay, (x, y), max(1, radius), color, -1, lineType=cv2.LINE_AA)
            if label_prefix is not None:
                name = names[i] if (names is not None and i < len(names)) else str(i)
                txt = f"{label_prefix}:{name}"
                cv2.putText(overlay, txt, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (10, 10, 10), 3, cv2.LINE_AA)
                cv2.putText(overlay, txt, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    else:
        img[:] = overlay


def put_header(img, text):
    cv2.putText(
        img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 3, cv2.LINE_AA
    )
    cv2.putText(
        img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
    )


def draw_legend(img, items):
    # items: list of (label, bgr)
    pad = 10
    x0, y0 = 12, img.shape[0] - 12
    line_h = 24
    box_w = 260
    box_h = pad * 2 + line_h * len(items)
    cv2.rectangle(img, (x0 - 8, y0 - box_h), (x0 - 8 + box_w, y0), (0, 0, 0), -1)
    cv2.rectangle(img, (x0 - 8, y0 - box_h), (x0 - 8 + box_w, y0), (255, 255, 255), 1)
    for i, (label, bgr) in enumerate(items):
        y = y0 - box_h + pad + i * line_h + 16
        cv2.circle(img, (x0, y - 6), 6, bgr, -1, lineType=cv2.LINE_AA)
        cv2.putText(img, label, (x0 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(img, label, (x0 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Visualize Marker-Enhancer markers on a video frame.")
    ap.add_argument("-m", "--manifest", required=True)
    ap.add_argument("-p", "--paths", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--outputs-root", default=None)
    ap.add_argument("--video-field", choices=["video_sync", "video_raw"], default="video_sync")

    # Which predictions file to use for 2D alignment
    ap.add_argument("--preds-type", choices=["preds", "preds_metric"], default="preds",
                    help="Choose which RTMW3D jsonl to use for 2D alignment: preds.jsonl or preds_metric.jsonl")
    ap.add_argument("--preds-path", default=None, help="Optional explicit path to RTMW3D predictions jsonl")

    # Select which marker set(s) to draw
    ap.add_argument("--set", choices=["body", "arms", "both"], default="both")

    # Frame/time selection
    ap.add_argument("--frame-index", type=int, default=None, help="Use this exact video frame index")
    ap.add_argument("--time-sec", type=float, default=None, help="Or pick nearest frame to this time (sec)")

    # Person selection and thresholds
    ap.add_argument("--person-index", type=int, default=-1)
    ap.add_argument("--min-mean-score", type=float, default=0.0)

    # Labels & output
    ap.add_argument("--label-markers", action="store_true", help="Draw text labels for marker names")
    ap.add_argument("--out", default=None, help="PNG output (default: <trial>/enhancer_vis/frame_<f>.png)")

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
    enh_dir = trial_root / "enhancer"
    vis_dir = trial_root / "enhancer_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(trial.get(args.video_field))
    if args.preds_path:
        preds_path = Path(args.preds_path)
    else:
        preds_path = rtmw3d_dir / ("preds.jsonl" if args.preds_type == "preds" else "preds_metric.jsonl")

    # Required cache for OCAP20 & times
    enh_input_path = enh_dir / "enh_input_60hz.npz"

    # Optional model outputs
    body_npz = enh_dir / "body_pred_mm_Tx35x3.npz"
    arms_npz = enh_dir / "arms_pred_mm_Tx8x3.npz"

    log_info(f"Video path          : {video_path}")
    log_info(f"RTMW3D preds jsonl  : {preds_path}")
    log_info(f"Enhancer cache      : {enh_input_path}")
    log_info(f"Body markers (npz)  : {body_npz}")
    log_info(f"Arms markers (npz)  : {arms_npz}")

    if not video_path.exists():
        raise SystemExit(f"[ERROR] video not found: {video_path}")
    if not enh_input_path.exists():
        raise SystemExit(f"[ERROR] enhancer input npz not found: {enh_input_path}")

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
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fi = max(0, nframes // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)

    ok, frame_bgr = cap.read()
    if not ok:
        raise SystemExit("[ERROR] failed to read selected frame")
    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    t_frame = (pos_msec / 1000.0) if (pos_msec and not np.isnan(pos_msec) and pos_msec > 0) else (fi / fps)
    log_info(f"Using frame_index   : {fi}  (t={t_frame:.3f} s)")

    # 3) Load enhancer input cache (OCAP20 mm + times)
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

    # 4) Try to read RTMW3D 2D for that frame (for alignment)
    found = None
    if preds_path.exists():
        log_step("Fetching RTMW3D 2D keypoints for similarity fit")
        found = read_jsonl_find_frame(
            preds_path,
            frame_index=fi,
            time_target=t_frame,
            tol_sec=1.0 / max(fps, 1.0),
            person_index=args.person_index,
            min_mean_score=args.min_mean_score,
        )
    else:
        log_warn("RTMW3D predictions jsonl not found — will use simple fit.")

    kp2d_px = None
    kp2d_names = None
    if found is not None:
        kp2d_px = found["keypoints_px"]
        kp2d_names = found.get("keypoint_names", None)
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

    # 5) Gather correspondences for similarity (OCAP20 only)
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

    # 6) Load Marker‑Enhancer outputs
    def safe_load_npz(path, key_pred="pred_mm", key_names="names"):
        if not path.exists():
            return None, None
        d = np.load(path, allow_pickle=True)
        P = d.get(key_pred, None)
        if P is None:
            # backward compatibility: sometimes array is the first unnamed entry
            if len(d.files) and P is None:
                P = d[d.files[0]]
        names = d.get(key_names, None)
        if names is not None:
            names = [str(x) for x in names.tolist()]
        return P, names

    body_TxK3, body_names = safe_load_npz(body_npz)
    arms_TxK3, arms_names = safe_load_npz(arms_npz)

    if args.set in ("body", "both") and body_TxK3 is None:
        log_warn("Body markers NPZ not found; skipping body.")
    if args.set in ("arms", "both") and arms_TxK3 is None:
        log_warn("Arms markers NPZ not found; skipping arms.")

    # Ensure time index is within bounds
    def get_frame_xy_mm(TxK3):
        if TxK3 is None:
            return None
        i = min(max(0, i_time), TxK3.shape[0] - 1)
        return TxK3[i, :, :2]

    body_xy_mm = get_frame_xy_mm(body_TxK3) if args.set in ("body", "both") else None
    arms_xy_mm = get_frame_xy_mm(arms_TxK3) if args.set in ("arms", "both") else None

    # 7) Projection to pixels
    def project_xy_mm_to_px(XY_mm):
        if XY_mm is None:
            return None
        if have_similarity:
            return apply_similarity_2d(XY_mm, s, R, t)
        # Fallback: simple fit to image (isotropic scale to 60% of min(img_w,img_h))
        h, w = frame_bgr.shape[:2]
        pts = XY_mm.copy()
        # Normalize by subtracting pelvis (use OCAP20 mid_hip if available)
        # Here we center by the XY centroid for robustness
        ctr = np.nanmean(pts, axis=0)
        pts0 = pts - ctr
        max_range = np.nanmax(np.linalg.norm(pts0, axis=1)) + 1e-6
        scale = 0.6 * min(w, h) / max_range
        px = pts0 * scale
        # center on image center
        px += np.array([w / 2.0, h * 0.55])  # a bit below center looks natural
        return px

    body_px = project_xy_mm_to_px(body_xy_mm)
    arms_px = project_xy_mm_to_px(arms_xy_mm)

    # 8) Draw overlay
    log_step("Drawing overlay")
    img = frame_bgr.copy()

    legend_items = []
    if body_px is not None:
        draw_points(img, body_px, color=(0, 255, 0), label_prefix=("B" if args.label_markers else None), names=body_names)
        legend_items.append(("Marker‑Enhancer (Body)", (0, 255, 0)))
    if arms_px is not None:
        draw_points(img, arms_px, color=(255, 255, 0), label_prefix=("A" if args.label_markers else None), names=arms_names)
        legend_items.append(("Marker‑Enhancer (Arms)", (255, 255, 0)))

    mode = "similarity" if have_similarity else "simple"
    put_header(img, f"frame {fi}  t={t_frame:.3f}s  (proj: {mode})")
    if legend_items:
        draw_legend(img, legend_items)

    # 9) Save
    default_out = vis_dir / f"frame_{fi}.png"
    out_path = Path(args.out) if args.out else default_out
    cv2.imwrite(str(out_path), img)
    log_done(f"Saved visualization : {out_path}")


if __name__ == "__main__":
    main()
