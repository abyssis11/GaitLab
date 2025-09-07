#!/usr/bin/env python3
"""
 python src/visualizations/viz_mocap_on_video.py   -m manifests/OpenCapDataset/subject2.yaml   -p config/paths.yaml   --trial walking1   --video-source sync  --mode perspective   --out f001.png --cam-units auto --debug --time-offset 0.08 --frame-index 24 --nudge-right-mm 178 --nudge-up-mm 10 --nudge-forward-mm 70 --yaw-deg 90
"""
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import argparse, sys, re, pickle
import numpy as np
import cv2

try:
    import yaml
except Exception:
    yaml = None

def log_info(m): print(f"[INFO] {m}")
def log_step(m): print(f"[STEP] {m}")
def log_warn(m): print(f"[WARN] {m}")

def load_manifest_wrapper(manifest_path: str, paths_path: str):
    try:
        from IO.load_manifest import load_manifest as lm
    except Exception:
        from load_manifest import load_manifest as lm
    return lm(manifest_path, paths_path)

# ---------- TRC ----------
def parse_trc(trc_path: Path) -> Dict[str, Any]:
    with open(trc_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    hdr_idx = None
    for i, ln in enumerate(lines):
        if ln.split()[0].lower().startswith("frame"):
            hdr_idx = i; break
    if hdr_idx is None or hdr_idx < 3:
        raise ValueError("TRC header not recognized (missing 'Frame#').")
    meta_keys = lines[1].split()
    meta_vals = lines[2].split()
    meta = dict(zip(meta_keys, meta_vals))
    def ffloat(x, d):
        try: return float(x)
        except: return d
    data_rate = ffloat(meta.get("DataRate","100"), 100.0)
    units     = meta.get("Units","mm")
    name_row = lines[hdr_idx].split()
    marker_names = name_row[2:]
    M = len(marker_names)
    data_rows = lines[hdr_idx+2:]
    frames, times, markers_xyz = [], [], []
    for ln in data_rows:
        parts = ln.split()
        if len(parts) < 2: continue
        need = 2 + 3*M
        if len(parts) < need: parts += ["nan"]*(need - len(parts))
        try:
            fid = int(float(parts[0])); t = float(parts[1])
        except:
            continue
        vals = np.array([float(x) for x in parts[2:2+3*M]], float)
        xyz = np.full((M,3), np.nan)
        xyz[:,0] = vals[0::3]; xyz[:,1] = vals[1::3]; xyz[:,2] = vals[2::3]
        frames.append(fid); times.append(t); markers_xyz.append(xyz)
    return dict(
        data_rate=data_rate,
        units=units,
        marker_names=marker_names,
        time=np.asarray(times, float),
        markers=np.asarray(markers_xyz, float),
        frames=np.asarray(frames, int),
    )

def trc_interpolate_at_time(trc: Dict[str,Any], t_target: float) -> np.ndarray:
    t = trc["time"]; X = trc["markers"]
    if t_target <= t[0]: return X[0]
    if t_target >= t[-1]: return X[-1]
    i = np.searchsorted(t, t_target, side="left")
    i0 = max(0, i-1); i1 = min(len(t)-1, i)
    t0, t1 = t[i0], t[i1]
    if t1 == t0: return X[i0]
    w = (t_target - t0)/(t1 - t0)
    return (1-w)*X[i0] + w*X[i1]

# ---------- Calibration I/O ----------
def _first(d: Dict[str,Any], keys: List[str]):
    for k in keys:
        if isinstance(d, dict) and (k in d) and (d[k] is not None):
            return d[k]
    return None

def read_cam_pickle(path: Path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        return None, None, None, None
    K    = _first(obj, ["intrinsicMat","K","camera_matrix","intrinsics"])
    dist = _first(obj, ["distortion","dist","distCoeffs"])
    R    = _first(obj, ["rotation","R","R_cam"])
    t    = _first(obj, ["translation","t","t_cam"])
    if K is not None:    K = np.asarray(K, float).reshape(3,3)
    if dist is not None: dist = np.asarray(dist, float).reshape(-1,1)
    if R is not None:    R = np.asarray(R, float).reshape(3,3)
    if t is not None:    t = np.asarray(t, float).reshape(3,)
    log_warn(f"Reading pickle; K={K} dist={dist} R={R} t={t}")
    return K, dist, R, t

def read_mocap_to_video_yaml(path: Path):
    R = t = None; scale = 1.0
    data = None
    if yaml is not None:
        try:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except Exception:
            data = None
    if isinstance(data, dict):
        R = data.get("R_fromMocap_toVideo") or data.get("R") or data.get("rotation")
        t = data.get("position_fromMocapOrigin_toVideoOrigin") or data.get("t") or data.get("translation")
        scale = data.get("scale") or data.get("units_scale") or 1.0
        if R is not None: R = np.asarray(R, float).reshape(3,3)
        if t is not None: t = np.asarray(t, float).reshape(3,)
    else:
        txt = Path(path).read_text(encoding="utf-8", errors="ignore")
        def _grab_block(name):
            m = re.search(rf"^{name}\s*:\s*\n((?:[ \t].*\n)+)", txt, re.MULTILINE)
            return m.group(1) if m else None
        blk = _grab_block("R_fromMocap_toVideo") or _grab_block("rotation") or _grab_block("R")
        if blk:
            rows = []
            for line in blk.splitlines():
                nums = [float(s) for s in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]
                if len(nums) == 3: rows.append(nums)
            if rows: R = np.asarray(rows, float).reshape(3,3)
        t_blk = _grab_block("position_fromMocapOrigin_toVideoOrigin") or _grab_block("t") or _grab_block("translation")
        if t_blk:
            nums = [float(s) for s in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", t_blk)]
            if len(nums) >= 3: t = np.asarray(nums[:3], float)
        scale = 1.0
    if R is None: R = np.eye(3)
    if t is None: t = np.zeros(3)
    # YAML t is commonly meters — convert to mm if small.
    t_mm = t*1000.0 if np.linalg.norm(t) < 10.0 else t
    return R, t_mm, float(scale)

def invert_extrinsics(R: np.ndarray, t: np.ndarray):
    Rinv = R.T
    tinv = -Rinv @ t.reshape(3,)
    return Rinv, tinv

# ---------- Projection ----------
def project_points_with_cam(Xw_mm: np.ndarray,
                            K: np.ndarray, dist: Optional[np.ndarray],
                            R_cam: np.ndarray, t_cam_mm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """World(mm) -> camera -> pixels."""
    Rc = R_cam.astype(np.float64)
    tc = t_cam_mm.reshape(3,1).astype(np.float64)
    rvec, _ = cv2.Rodrigues(Rc)
    Xc = (Rc @ Xw_mm.T + tc).T
    Zc = Xc[:,2].copy()
    proj, _ = cv2.projectPoints(Xw_mm.reshape(-1,1,3).astype(np.float64),
                                rvec, tc, K.astype(np.float64),
                                None if dist is None else dist.astype(np.float64))
    return proj.reshape(-1,2).astype(np.float64), Zc

def project_points_identity_cam(Xc_mm: np.ndarray,
                                K: np.ndarray, dist: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Camera(mm) -> pixels (R=I, t=0)."""
    rvec = np.zeros((3,1), np.float64)
    tvec = np.zeros((3,1), np.float64)
    Zc = Xc_mm[:,2].copy()
    proj, _ = cv2.projectPoints(Xc_mm.reshape(-1,1,3).astype(np.float64),
                                rvec, tvec, K.astype(np.float64),
                                None if dist is None else dist.astype(np.float64))
    return proj.reshape(-1,2).astype(np.float64), Zc

def draw_markers(img, P: np.ndarray, names: List[str], show_labels: bool):
    P = np.asarray(P)
    if P.ndim == 3 and P.shape[1:] == (1,2):
        P = P[:,0,:]
    P = P.reshape(-1,2).astype(np.float64)
    H, W = img.shape[:2]
    for i in range(P.shape[0]):
        x, y = P[i,0], P[i,1]
        if not (np.isfinite(x) and np.isfinite(y)): 
            continue
        xi = int(round(float(x))); yi = int(round(float(y)))
        if xi < -2000 or yi < -2000 or xi > W+2000 or yi > H+2000:
            continue
        cv2.drawMarker(img, (xi, yi), (255,255,255),
                       markerType=cv2.MARKER_TILTED_CROSS, markerSize=8,
                       thickness=2, line_type=cv2.LINE_AA)
        cv2.circle(img, (xi, yi), 2, (0,0,0), -1, lineType=cv2.LINE_AA)
        if show_labels:
            label = names[i] if i < len(names) else f"M{i}"
            cv2.putText(img, label, (xi+5, yi-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240,240,240), 1, cv2.LINE_AA)

def rot_y(deg):
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[ c, 0,  s],
                     [ 0, 1,  0],
                     [-s, 0,  c]], dtype=float)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser("TRC→Video overlay (single frame)")
    ap.add_argument("-m","--manifest", required=True, type=str)
    ap.add_argument("-p","--paths", required=True, type=str)
    ap.add_argument("--trial", required=True, type=str)
    ap.add_argument("--video-source", choices=["sync","raw"], default="sync")
    ap.add_argument("--frame-index", type=int, default=None)
    ap.add_argument("--time-sec", type=float, default=None)
    ap.add_argument("--time-offset", type=float, default=0.0)
    ap.add_argument("--mode", choices=["perspective","orthographic"], default="perspective")
    ap.add_argument("--swap-extrinsics", action="store_true")
    ap.add_argument("--ignore-distortion", action="store_true")
    ap.add_argument("--cam-units", choices=["auto","mm","m"], default="auto")
    ap.add_argument("--cam-translation-as-center", action="store_true")
    ap.add_argument("--assume-yaml-is-camera", action="store_true",
                    help="Treat YAML mocap->video as camera frame; ignore pickle R|t.")
    ap.add_argument("--swap-m2v", action="store_true",
                    help="Invert mocap->video YAML (if file stores video->mocap).")
    # NEW: skip both YAML and pickle, and synthetic intrinsics
    ap.add_argument("--no-calib", "--skip-calibration", dest="no_calib", action="store_true",
                    help="Ignore BOTH YAML and pickle. Use identity extrinsics and synthetic intrinsics.")
    ap.add_argument("--focal-px", type=float, default=None,
                    help="Focal length (pixels) used when --no-calib is set. Default: max(W,H).")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--show-labels", action="store_true")
    ap.add_argument("--intrinsics-pickle", type=str, default=None)
    ap.add_argument("--mocap2video-yaml", type=str, default=None)
    ap.add_argument("--out", type=str, default="./trc_overlay.png")
    ap.add_argument("--m2v-rotation-only", action="store_true",
                help="Use only rotation from mocapToVideo YAML (set translation to 0).")
    ap.add_argument("--principal-center", action="store_true",
                help="Override K[0,2],K[1,2] to image center (W/2,H/2)")
    ap.add_argument("--m2v-apply", choices=["R","RT"], default="RT",
                    help="Apply YAML rotation as R (no transpose) or RT (transpose).")
    ap.add_argument("--yaw-deg", type=float, default=0.0,
                    help="Extra yaw (degrees) about world up (video Y) pre-multiplied to YAML R.")
    ap.add_argument("--nudge-right-mm", type=float, default=0.0,
                help="Camera-frame X translation (+right) in mm")
    ap.add_argument("--nudge-up-mm", type=float, default=0.0,
                    help="Camera-frame Y translation (+down in OpenCV, use negative to move up) in mm")
    ap.add_argument("--nudge-forward-mm", type=float, default=0.0,
                    help="Camera-frame Z translation (+forward) in mm")

    args = ap.parse_args()

    manifest = load_manifest_wrapper(args.manifest, args.paths)
    trials = manifest.get("trials", {}) or {}
    trial = None
    for lst in trials.values():
        if isinstance(lst, list):
            for t in lst:
                if str(t.get("id")) == args.trial:
                    trial = t; break
        if trial: break
    if trial is None:
        sys.exit(f"[ERROR] Trial '{args.trial}' not found in manifest.")

    video_path = Path(trial.get("video_sync") if args.video_source=="sync" else trial.get("video_raw") or "")
    trc_path   = Path(trial.get("mocap_trc",""))
    if not video_path: sys.exit("[ERROR] Missing video path in trial.")
    if not trc_path:   sys.exit("[ERROR] Missing TRC path in trial.")

    # TRC & frame
    log_step("Parsing TRC")
    trc = parse_trc(trc_path)
    names = trc["marker_names"]
    t0, t1 = float(trc["time"][0]), float(trc["time"][-1])
    log_info(f"TRC: {len(trc['time'])} frames @ {trc['data_rate']:.1f} Hz, units={trc['units']}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if args.frame_index is not None:
        fi = int(args.frame_index)
    elif args.time_sec is not None:
        fi = int(round(args.time_sec * fps))
    else:
        fi = 0
    fi = max(0, min(fi, max(0, nframes-1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"[ERROR] Failed to read frame {fi}")
    H, W = frame.shape[:2]
    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    t_video = (pos_msec/1000.0) if (pos_msec and not np.isnan(pos_msec) and pos_msec>0) else (fi/fps)
    log_info(f"Video frame={fi} (t≈{t_video:.3f}s, fps≈{fps:.2f})")
    log_info(f"TRC range: [{t0:.3f}, {t1:.3f}] s")

    t_trc = t_video + float(args.time_offset)
    if t_trc < t0 or t_trc > t1:
        log_warn(f"t_trc={t_trc:.3f}s outside TRC; clamping")
        t_trc = min(max(t_trc, t0), t1)
    X_trc_mm = trc_interpolate_at_time(trc, t_trc)

    # ======== NO-CALIBRATION PATH (skip YAML & pickle) ========
    if args.no_calib:
        # Synthetic intrinsics
        focal = float(args.focal_px) if args.focal_px is not None else float(max(W, H))
        K = np.array([[focal, 0.0, W*0.5],
                      [0.0,  focal, H*0.5],
                      [0.0,  0.0,   1.0]], dtype=float)
        dist = None
        log_info(f"[no-calib] Using synthetic K with focal={focal:.1f}px, cx={W*0.5:.1f}, cy={H*0.5:.1f}")
        # Identity mocap->video world (optionally yaw)
        R_m2v = np.eye(3, dtype=float)
        t_m2v_mm = np.zeros(3, dtype=float)
        scale = 1.0
        R_eff = rot_y(args.yaw_deg) @ R_m2v
        Xw = (X_trc_mm * float(scale)) @ R_eff.T + t_m2v_mm
        # Identity camera (R=I,t=0)
        uv, Zcam = project_points_identity_cam(Xw, K, dist)
        # Draw
        img = frame.copy()
        draw_markers(img, uv, names, args.show_labels)
        if args.debug:
            inb = (uv[:,0]>=0) & (uv[:,0]<W) & (uv[:,1]>=0) & (uv[:,1]<H) & np.isfinite(uv[:,0]) & np.isfinite(uv[:,1])
            if np.isfinite(uv).any():
                log_info(f"In-frame: {int(inb.sum())}/{uv.shape[0]}")
                log_info(f"u range: [{np.nanmin(uv[:,0]):.1f}, {np.nanmax(uv[:,0]):.1f}] | "
                         f"v range: [{np.nanmin(uv[:,1]):.1f}, {np.nanmax(uv[:,1]):.1f}]")
        hud = f"TRC t={t_trc:.3f}s | video t={t_video:.3f}s | mode={args.mode} | no-calib"
        cv2.putText(img, hud, (8, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        cv2.imwrite(args.out, img); log_info(f"Saved overlay: {args.out}")
        return

    # ======== STANDARD CALIBRATION PATH ========
    # Load intrinsics and YAML
    calib = manifest.get("calibration", {}) or {}
    intrin_path = Path(args.intrinsics_pickle or calib.get("intrinsics_extrinsics",""))
    m2v_path    = Path(args.mocap2video_yaml or calib.get("mocap_to_video",""))

    K = dist = R_cam = t_cam = None
    if intrin_path and intrin_path.exists():
        K, dist, R_cam, t_cam = read_cam_pickle(intrin_path)
        # Attempt to rescale K to video size if pickle has imageSize
        imgsz = None
        try:
            with open(intrin_path, "rb") as _f:
                _obj = pickle.load(_f)
            imgsz = _obj.get("imageSize", None)
        except Exception:
            imgsz = None
        if imgsz is not None:
            arr = np.asarray(imgsz, dtype=float).reshape(-1)
            if arr.size >= 2:
                candidates = []
                for (w0, h0, tag) in [(arr[0], arr[1], "WH"), (arr[1], arr[0], "HW")]:
                    if w0 <= 0 or h0 <= 0:
                        continue
                    sx = W / w0
                    sy = H / h0
                    # Penalize anisotropy and aspect mismatch
                    aspect0 = w0 / h0
                    aspectV = W / H
                    penalty = abs(sx - sy) + abs(aspectV - aspect0)
                    candidates.append((penalty, sx, sy, tag))
                if candidates:
                    _, sx, sy, tag = min(candidates, key=lambda x: x[0])
                    if abs(sx - 1) > 1e-3 or abs(sy - 1) > 1e-3:
                        K = K.copy()
                        K[0,0] *= sx; K[0,2] *= sx
                        K[1,1] *= sy; K[1,2] *= sy
                        log_info(f"Rescaled intrinsics to video size ({tag}): sx={sx:.4f}, sy={sy:.4f}")

    if args.principal_center and K is not None:
        K = K.copy()
        K[0,2] = W * 0.5
        K[1,2] = H * 0.5
        log_info(f"Principal point set to image center: ({K[0,2]:.1f}, {K[1,2]:.1f})")

    if args.ignore_distortion:
        dist = None
    if K is None:
        sys.exit("[ERROR] Missing intrinsics in pickle (need K for projection).")

    if not m2v_path or not m2v_path.exists():
        log_warn("No mocap_to_video file; assuming identity.")
        R_m2v = np.eye(3); t_m2v_mm = np.zeros(3); scale=1.0
    else:
        R_m2v, t_m2v_mm, scale = read_mocap_to_video_yaml(m2v_path)
        if args.m2v_rotation_only:
            t_m2v_mm = np.zeros(3, dtype=float)
            log_info("YAML translation ignored (rotation-only).")
        if args.swap_m2v:
            R_m2v, t_m2v_mm = invert_extrinsics(R_m2v, t_m2v_mm)

    # MoCap(mm) -> Video world(mm)
    R_eff = rot_y(args.yaw_deg) @ R_m2v
    if args.m2v_apply == "RT":
        Xw = (X_trc_mm * float(scale)) @ R_eff.T + t_m2v_mm
    else:
        Xw = (X_trc_mm * float(scale)) @ R_eff     + t_m2v_mm

    # Choose how to project
    if args.mode != "perspective":
        # orthographic fallback (sanity)
        Xc2d = Xw - np.nanmean(Xw, axis=0)
        P = Xc2d[:, :2]
        img = frame.copy()
        draw_markers(img, P, names, args.show_labels)
        cv2.putText(img, f"TRC t={t_trc:.3f}s | video t={t_video:.3f}s | mode=orthographic",
                    (8, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        cv2.imwrite(args.out, img); log_info(f"Saved overlay: {args.out}"); return

    if args.assume_yaml_is_camera:
        # Treat YAML output as CAMERA coords; do not apply pickle R|t
        uv, Zcam = project_points_identity_cam(Xw, K, dist)
    else:
        if R_cam is None or t_cam is None:
            sys.exit("[ERROR] Camera extrinsics missing, or use --assume-yaml-is-camera.")
        # conversion options for pickle t
        if args.cam_units == "m":
            t_cam_mm = t_cam * 1000.0
        elif args.cam_units == "mm":
            t_cam_mm = t_cam.copy()
        else:
            t_cam_mm = t_cam * 1000.0 if np.linalg.norm(t_cam) < 5.0 else t_cam.copy()
        if args.cam_translation_as_center:
            t_cam_mm = - (R_cam @ t_cam_mm.reshape(3))
        if args.swap_extrinsics:
            R_cam, t_cam_mm = invert_extrinsics(R_cam, t_cam_mm)
        # camera nudges (optional)
        t_cam_mm = t_cam_mm + np.array(
            [args.nudge_right_mm, args.nudge_up_mm, args.nudge_forward_mm], dtype=float
        )
        log_info(f"Applied camera nudge (mm): right={args.nudge_right_mm:.1f}, "
                 f"up={args.nudge_up_mm:.1f}, forward={args.nudge_forward_mm:.1f}")
        uv, Zcam = project_points_with_cam(Xw, K, dist, R_cam, t_cam_mm)

    # Diagnostics
    validZ = np.isfinite(Zcam)
    if validZ.any():
        frac_behind = float((Zcam[validZ] <= 0).sum())/float(validZ.sum())
        log_info(f"Median Z_cam={np.nanmedian(Zcam):.1f} mm | behind={100*frac_behind:.1f}%")

    img = frame.copy()
    draw_markers(img, uv, names, args.show_labels)

    if args.debug:
        inb = (uv[:,0]>=0) & (uv[:,0]<W) & (uv[:,1]>=0) & (uv[:,1]<H) & np.isfinite(uv[:,0]) & np.isfinite(uv[:,1])
        if np.isfinite(uv).any():
            log_info(f"In-frame: {int(inb.sum())}/{uv.shape[0]}")
            log_info(f"u range: [{np.nanmin(uv[:,0]):.1f}, {np.nanmax(uv[:,0]):.1f}] | "
                     f"v range: [{np.nanmin(uv[:,1]):.1f}, {np.nanmax(uv[:,1]):.1f}]")

    hud = f"TRC t={t_trc:.3f}s | video t={t_video:.3f}s | mode={args.mode}"
    cv2.putText(img, hud, (8, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    # --- Decide output folder & file name under the trial root ---
    base = manifest.get("output_dir")
    trial_root = Path(base) / trial["id"]
    save_dir = trial_root / "mocap_vis"
    save_dir.mkdir(parents=True, exist_ok=True)

    # keep only the file name part of --out (default to frame-based name)
    out_name = f"mocap_overlay_frame_{fi}{'_labels' if args.show_labels else ''}.png"
    out_path = save_dir / out_name
    log_info(f"Saving overlay to: {out_path}")
    cv2.imwrite(str(out_path), img)
    log_info(f"Saved overlay: {out_path}")

if __name__ == "__main__":
    main()
