#!/usr/bin/env python3
"""
Make RTMW3D predictions absolute (camera/world coordinates) using calibration.

Inputs
------
- preds_metric.jsonl: per-frame entries with persons[0].keypoints_xyz_mm and persons[0].keypoints_px
- cameraIntrinsicsExtrinsics.pickle: dict with keys
    - intrinsicMat: (3,3) float64
    - distortion: (1,5) float64
    - rotation: (3,3) float64  # checkerboard/world -> camera
    - translation: (3,1) float64

Outputs
-------
- preds_metric_abs.jsonl: same frames, but each person gets:
    - R_cam: 3x3 rotation (camera frame)
    - t_cam_mm: 3 translation in **mm**
    - keypoints_cam_mm_abs: (K,3) absolute in camera coords (mm)
    - reproj_rmse_px: float (pixels)
- Optional TRCs:
    - --trc-cam: TRC in camera frame (mm) from keypoints_cam_mm_abs
    - --trc-world: TRC in world frame (mm) if checkerboard extrinsics provided

Use:
python src/pose/rtmw3d_make_absolute.py \
    -m ./manifests/OpenCapDataset/subject2.yaml \
    -p ./config/paths.yaml \
    --trial static2 \
    --trc-cam \
    --trc-world \
    --ema 0.2 --min-score 0.2  # optional smoothing and joint filtering

Notes
-----
- Uses a fixed scale: X_mm are treated as metric; solve only R,t via cv2.solvePnP.
- Selects a robust joint subset (COCO body: shoulders/hips/knees/ankles, heels/toes if present).
- Carries over all original fields unchanged.
"""

import argparse, json, sys
import pickle as _pkl
from pathlib import Path
import numpy as np
from IO.load_manifest import load_manifest
import os
import cv2
import xml.etree.ElementTree as ET

# ---------- Logging ----------
def log_step(msg): print(f"[STEP] {msg}")
def log_info(msg): print(f"[INFO] {msg}")
def log_warn(msg): print(f"[WARN] {msg}")
def log_err (msg): print(f"[ERROR] {msg}")
def log_done(msg): print(f"[DONE] {msg}")

# ---------- IO helpers ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _should_undistort(u_px_sample, K, dist, thresh_px=0.1):
    """Heuristic: if undistorting changes points by < thresh_px median, skip."""
    import cv2, numpy as np
    if u_px_sample.size == 0:
        return False
    u = np.asarray(u_px_sample, dtype=np.float32).reshape(-1,1,2)
    ud = cv2.undistortPoints(u, K, dist, P=K)
    d = np.linalg.norm((ud - u), axis=2).reshape(-1)
    med = float(np.median(d))
    return med >= thresh_px

def _scores_to_mean(scores, K):
    if scores is None: return None
    s = np.asarray(scores)
    if s.ndim == 0:
        return np.full((K,), float(s))
    if s.ndim == 1:
        out = np.full((K,), np.nan, float); out[:min(K, s.size)] = s[:min(K, s.size)]
        return out
    # s.ndim >= 2
    if s.shape[0] != K:
        out = np.full((K, s.shape[1]), np.nan, float); out[:min(K, s.shape[0]),:] = s[:min(K, s.shape[0]),:]
        s = out
    return np.nanmean(s, axis=1)

def _Rx(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], float)

def _Ry(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]], float)

def _Rz(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]], float)

def euler_matrix(rx, ry, rz, order="xyz"):
    """
    Return rotation matrix for intrinsic rotations about x,y,z in 'order'.
    Column-vector convention: p_cam = R * p_world.
    'xyz' means apply Rx, then Ry, then Rz -> R = Rz @ Ry @ Rx.
    'zyx' means apply Rz, then Ry, then Rx -> R = Rx @ Ry @ Rz.
    """
    if order.lower() == "xyz":
        return _Rz(rz) @ _Ry(ry) @ _Rx(rx)
    elif order.lower() == "zyx":
        return _Rx(rx) @ _Ry(ry) @ _Rz(rz)
    else:
        raise ValueError(f"Unsupported Euler order: {order}")

def parse_xml_calibration_tsai(xml_path, angles="auto"):
    """
    Parse a Tsai-style <Camera> XML.

    Geometry: width,height,[dpx,dpy] (pixel sizes; default 1).
    Intrinsic: focal (pixels), cx, cy (pixels), sx (x-scale), kappa1 (radial k1).
    Extrinsic: tx,ty,tz (mm), rx,ry,rz (either Rodrigues or Euler; controlled by 'angles').

    Returns dict like the pickle:
      intrinsicMat (3x3), distortion (1x5), rotation (3x3, world->camera, if present),
      translation (3,1 in mm, if present), imageSize (W,H)
    """
    root = ET.parse(str(xml_path)).getroot()
    cam = root if root.tag.lower() == "camera" else root.find(".//Camera")
    if cam is None:
        raise ValueError("XML must contain a <Camera> element.")

    geom = cam.find("Geometry")
    intr = cam.find("Intrinsic")
    extr = cam.find("Extrinsic")
    if geom is None or intr is None:
        raise ValueError("XML missing <Geometry> or <Intrinsic>.")

    W = int(float(geom.get("width")))
    H = int(float(geom.get("height")))

    # Pixel sizes (if present)
    dpx = float(geom.get("dpx", "1.0"))
    dpy = float(geom.get("dpy", "1.0"))

    focal = float(intr.get("focal"))
    sx = float(intr.get("sx", "1.0"))
    cx = float(intr.get("cx")); cy = float(intr.get("cy"))

    # Tsai -> OpenCV intrinsics
    fx = focal * sx / dpx
    fy = focal / dpy
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], float)

    k1 = float(intr.get("kappa1", "0"))
    # Map Tsai radial-only to Brown model
    dist = np.array([k1, 0.0, 0.0, 0.0, 0.0], float).reshape(1,5)

    R_w2c = None; t_w2c = None
    if extr is not None:
        tx = float(extr.get("tx")); ty = float(extr.get("ty")); tz = float(extr.get("tz"))
        t_w2c = np.array([tx, ty, tz], float).reshape(3,1)  # assume mm
        rx = float(extr.get("rx")); ry = float(extr.get("ry")); rz = float(extr.get("rz"))

        if angles == "rodrigues":
            # Interpret (rx,ry,rz) as Rodrigues vector
            R_w2c, _ = cv2.Rodrigues(np.array([rx,ry,rz], float).reshape(3,1))
        elif angles == "euler-xyz":
            R_w2c = euler_matrix(rx, ry, rz, order="xyz")
        elif angles == "euler-zyx":
            R_w2c = euler_matrix(rx, ry, rz, order="zyx")
        else:
            # auto: try Rodrigues by default
            R_w2c, _ = cv2.Rodrigues(np.array([rx,ry,rz], float).reshape(3,1))

    out = {
        "intrinsicMat": K,
        "distortion": dist,
        "imageSize": (W, H),
    }
    if R_w2c is not None and t_w2c is not None:
        out["rotation"] = R_w2c
        out["translation"] = t_w2c
    return out

def load_calibration_any(path_like, angles="auto"):
    p = Path(path_like)
    if p.suffix.lower() in [".pkl", ".pickle"]:
        with open(p, "rb") as pf:
            return _pkl.load(pf)
    elif p.suffix.lower() == ".xml":
        return parse_xml_calibration_tsai(p, angles=angles)
    else:
        raise ValueError(f"Unsupported calibration file: {p}")



def export_trc_from_series(frames, out_path, source_key="keypoints_cam_mm_abs", rate_hz=60.0, kp_names=None):
    # Collect F,K,3 into array (NaN where missing)
    seq = []
    Kmax = 0
    for o in frames:
        persons = o.get("persons") or []
        if not persons:
            seq.append(None); continue
        p = persons[0]
        if source_key not in p:
            seq.append(None); continue
        arr = np.asarray(p[source_key], dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            seq.append(None); continue
        seq.append(arr)
        Kmax = max(Kmax, arr.shape[0])
    F = len(seq); K = Kmax
    data = np.full((F, K, 3), np.nan, dtype=float)
    for i,a in enumerate(seq):
        if a is None: continue
        data[i,:a.shape[0],:] = a[:K]
    # Write TRC
    with open(out_path, "w", encoding="utf-8") as fout:
        fout.write(f"PathFileType\t4\t(X/Y/Z)\t{out_path}\n")
        fout.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        fout.write(f"{rate_hz:.1f}\t{rate_hz:.1f}\t{F}\t{K}\tmm\t{rate_hz:.1f}\t1\t{F}\n")
        names = (kp_names or [])[:K]
        if len(names) < K:
            names = names + [f"J{i+1}" for i in range(len(names), K)]
        fout.write("Frame#\tTime\t" + "\t\t\t".join(names) + "\t\t\t\n")
        xyz_cols = []
        for i in range(1, K+1):
            xyz_cols.extend([f"X{i}", f"Y{i}", f"Z{i}"])
        fout.write("\t" + "\t".join(xyz_cols) + "\n\n")
        dt = 1.0/float(rate_hz) if rate_hz>0 else 0.0
        for fi in range(F):
            t = fi*dt
            row = [str(fi+1), f"{t:.8f}"]
            for k in range(K):
                x,y,z = data[fi,k,:]
                row.extend([f"{x:.5f}" if np.isfinite(x) else "",
                            f"{y:.5f}" if np.isfinite(y) else "",
                            f"{z:.5f}" if np.isfinite(z) else ""])
            fout.write("\t".join(row) + "\n")

def _undistort_points(u_px, K, dist):
    import cv2
    u = np.asarray(u_px, dtype=np.float32).reshape(-1,1,2)
    out = cv2.undistortPoints(u, K, dist, P=K)
    return out.reshape(-1,2)

def _solve_pnp_fixed_scale(u_px, X_mm, K, dist, rvec_init=None, tvec_init=None, do_undistort=True):
    """Return R(3x3), t_mm(3,), reproj_rmse_px. Keeps X in mm -> t will be in mm."""
    import cv2
    u_ud = (_undistort_points(u_px, K, dist).astype(np.float32) if do_undistort else np.asarray(u_px, dtype=np.float32))
    X_obj = np.asarray(X_mm, dtype=np.float32)  # mm
    # Mask invalids
    vis = np.isfinite(u_ud).all(axis=1) & np.isfinite(X_obj).all(axis=1)
    u = u_ud[vis]; X = X_obj[vis]
    if u.shape[0] < 6:
        return None, None, np.nan
    # Initial solve
    flags = cv2.SOLVEPNP_ITERATIVE
    if rvec_init is None or tvec_init is None:
        ok, rvec, tvec = cv2.solvePnP(X, u, K, dist, flags=flags)
    else:
        ok, rvec, tvec = cv2.solvePnP(X, u, K, dist, rvec_init, tvec_init, True, flags=flags)
    if not ok:
        # Fallback: use EPnP without guess
        ok, rvec, tvec = cv2.solvePnP(X, u, K, dist, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            return None, None, np.nan
    R,_ = cv2.Rodrigues(rvec)
    # Reprojection RMSE
    proj,_ = cv2.projectPoints(X, rvec, tvec, K, dist)
    reproj = proj.reshape(-1,2) - u
    rmse = float(np.sqrt(np.mean(np.sum(reproj**2, axis=1))))
    return R, tvec.reshape(-1), rmse

def _select_joint_indices(K):
    """Return a robust subset of indices (COCO body 17 assumed at 0..16)."""
    idxs = []
    # H3WB: shoulders(5,6), hips(11,12), knees(13,14), ankles(15,16)
    for i in [5,6,11,12,13,14,15,16]:
        if i < K: idxs.append(i)
    return np.array(idxs, dtype=int)

def _cam_to_world(X_cam_mm, R_wc, t_wc_mm):
    return (R_wc @ X_cam_mm.T + t_wc_mm.reshape(3,1)).T

def _invert_extrinsics(R, t):
    """Given world->cam R,t, return cam->world R^T, -R^T t."""
    Rcw = R.T
    tcw = -Rcw @ t.reshape(3)
    return Rcw, tcw

def load_jsonl(p):
    frames = []
    with open(p, "r", encoding="utf-8") as f:
        for ln in f:
            frames.append(json.loads(ln))
    return frames

def save_jsonl(frames, p):
    with open(p, "w", encoding="utf-8") as f:
        for o in frames:
            f.write(json.dumps(o) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Make RTMW3D poses absolute using calibration (PnP with fixed scale)." )
    ap.add_argument("-m", "--manifest", required=True)
    ap.add_argument("-p", "--paths", required=True)
    ap.add_argument("--trial", required=True)
    #ap.add_argument("--in", dest="inp", required=True, help="Input preds_metric.jsonl")
    #ap.add_argument("--pickle", dest="pickle", required=True, help="cameraIntrinsicsExtrinsics.pickle")
    #ap.add_argument("--out", dest="outp", required=True, help="Output preds_metric_abs.jsonl")
    ap.add_argument("--rate", type=float, default=60.0, help="TRC sample rate")
    ap.add_argument("--trc-cam", action="store_true", help="Optional TRC (camera frame, mm)")
    ap.add_argument("--trc-world", action="store_true", help="Optional TRC (world frame, mm). Uses checkerboard extrinsics.")
    ap.add_argument("--min-score", type=float, default=0.0, help="Drop joints below this score when solving PnP.")
    ap.add_argument("--undistort", choices=["auto","on","off"], default="auto", help="Undistort 2D keypoints before PnP.")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA smoothing on translation (0=no smoothing, e.g., 0.2)")
    ap.add_argument("--xml-angles",
                choices=["auto","rodrigues","euler-xyz","euler-zyx"],
                default="euler-xyz",
                help="Interpretation of <Extrinsic rx,ry,rz> in Tsai XML.")

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
        raise log_err(f"Trial '{args.trial}' not found in manifest.")
    
    # Paths
    base = manifest.get('output_dir')
    if not base:
        subj = manifest.get('subject_id', 'subject')
        sess = manifest.get('session', 'Session')
        cam  = manifest.get('camera', 'Cam')
        base = Path(manifest.get('outputs_root', Path.cwd() / "outputs")) / subj / sess / cam
    trial_root = Path(base) / trial['id']
    rtmw3d_dir = trial_root / "rtmw3d"
    preds_metric = rtmw3d_dir / "preds_metric.jsonl"
    meta_path  = trial_root / "meta.json"
    intrinsics_extrinsics = trial.get("intrinsics_extrinsics")

    log_info(f"Trial root : {trial_root}")
    log_info(f"Preds metric path : {preds_metric}")
    log_info(f"meta.json  : {meta_path}")
    log_info(f"cameraIntrinsicsExtrinsics : {intrinsics_extrinsics}")

    output_path = rtmw3d_dir / "preds_metric_abs.jsonl"
    output_path_trc_world = rtmw3d_dir / "rtmw3d_world.trc"
    output_path_trc_cam = rtmw3d_dir / "rtmw3d_cam.trc"

    meta = json.load(open(meta_path, 'r', encoding='utf-8'))
    kp_names = meta.get("keypoint_names") or []

    # Load calibration (pickle or Tsai XML)
    calib = load_calibration_any(intrinsics_extrinsics, angles=args.xml_angles)

    K = np.asarray(calib["intrinsicMat"], float)
    dist = np.asarray(calib.get("distortion", np.zeros((1,5), float)), float).reshape(-1)

    R_wc0 = None; t_wc0_mm = None
    if "rotation" in calib and "translation" in calib:
        R_w2c = np.asarray(calib["rotation"], float)
        t_w2c = np.asarray(calib["translation"], float).reshape(3)
        # Invert to camera->world for exporting world TRC
        R_wc0, t_wc0_mm = _invert_extrinsics(R_w2c, t_w2c)

    frames = load_jsonl(preds_metric)

    # Image size (if present) for bounds checks
    W = H = None
    if "imageSize" in calib:
        W, H = int(calib["imageSize"][0]), int(calib["imageSize"][1])


    # Sample a few frames to decide undistort
    do_undistort = True
    if args.undistort != "on":
        sample = []
        for o in frames[:50]:
            persons = o.get("persons") or []
            if not persons: continue
            u = np.asarray(persons[0].get("keypoints_px"))
            if u.size: sample.append(u)
        if sample:
            u_sample = np.concatenate([s.reshape(-1,2) for s in sample], axis=0)
            if args.undistort == "auto":
                do_undistort = _should_undistort(u_sample, K, dist, thresh_px=0.1)
            elif args.undistort == "off":
                do_undistort = False
    log_info(f"Undistort policy: {args.undistort} -> do_undistort={do_undistort}")


    prev_rvec = None; prev_tvec = None
    ema = float(args.ema); t_ema = None

    out_frames = []
    for o in frames:
        persons = o.get("persons") or []
        new_persons = []
        for p in persons:
            X_mm = np.asarray(p.get("keypoints_xyz_mm"), dtype=float)
            u_px = np.asarray(p.get("keypoints_px"), dtype=float)
            scores = np.asarray(p.get("keypoint_scores"), dtype=float) if p.get("keypoint_scores") is not None else None
            p2 = dict(p)
            if X_mm.ndim==2 and X_mm.shape[1]==3 and u_px.ndim==2 and u_px.shape[1]==2:
                Kp = X_mm.shape[0]
                sel = _select_joint_indices(Kp)
                vis = np.ones(Kp, dtype=bool)
                m = _scores_to_mean(scores, Kp) if scores is not None else None
                if m is not None:
                    vis &= np.isfinite(m) & (m >= args.min_score)
                # Intersect with sel
                keep = np.zeros(Kp, dtype=bool)
                keep[sel] = True
                keep &= vis
                if keep.sum() >= 6:
                    if W is not None and H is not None:
                        u_all = u_px
                        if np.any((u_all[:,0] < -1) | (u_all[:,0] > W+1) | (u_all[:,1] < -1) | (u_all[:,1] > H+1)):
                            # This suggests keypoints might be in crop coords; we still run PnP but warn.
                            if "_bounds_warned" not in globals():
                                log_warn(f"keypoints_px exceed imageSize ({W}x{H}). They may be in crop coordinates. Consider mapping back to full-frame.")
                                globals()["_bounds_warned"] = True
                        # Solve PnP with previous as init if available
                        import cv2
                        if prev_rvec is not None and prev_tvec is not None:
                            R_cam, t_cam_mm, rmse = _solve_pnp_fixed_scale(u_px[keep], X_mm[keep], K, dist, prev_rvec, prev_tvec, do_undistort)
                        else:
                            R_cam, t_cam_mm, rmse = _solve_pnp_fixed_scale(u_px[keep], X_mm[keep], K, dist, None, None, do_undistort)
                        if R_cam is not None:
                            # cache for next frame
                            rvec,_ = cv2.Rodrigues(R_cam)
                            prev_rvec, prev_tvec = rvec, t_cam_mm.reshape(3,1)
                            # optional EMA on t
                            if ema > 0.0:
                                if t_ema is None: t_ema = t_cam_mm.copy()
                                else: t_ema = (1-ema)*t_ema + ema*t_cam_mm
                                t_use = t_ema
                            else:
                                t_use = t_cam_mm
                            # Absolute camera-frame keypoints in mm
                            X_abs = (R_cam @ X_mm.T).T + t_use.reshape(1,3)
                            p2["R_cam"] = R_cam.tolist()
                            p2["t_cam_mm"] = t_use.tolist()
                            p2["keypoints_cam_mm_abs"] = X_abs.tolist()
                            p2["reproj_rmse_px"] = rmse
                            # World coords if available
                            if R_wc0 is not None and t_wc0_mm is not None:
                                X_world = _cam_to_world(np.asarray(X_abs), R_wc0, t_wc0_mm)
                                p2["keypoints_world_mm_abs"] = X_world.tolist()
            new_persons.append(p2)
        o2 = dict(o); o2["persons"] = new_persons
        out_frames.append(o2)

    # Save jsonl
    with open(output_path, "w", encoding="utf-8") as f:
        for o in out_frames:
            f.write(json.dumps(o) + "\n")

    # Optional TRCs
    if args.trc_cam:
        export_trc_from_series(out_frames, output_path_trc_cam, source_key="keypoints_cam_mm_abs", rate_hz=float(args.rate), kp_names=kp_names)
    if args.trc_world:
        if R_wc0 is None:
            log_warn("--trc-world requested but checkerboard extrinsics not in pickle; skipping.", file=sys.stderr)
        else:
            # Build world series temporarily by transforming camera series
            frames_world = []
            for o in out_frames:
                persons = o.get("persons") or []
                new_persons = []
                for p in persons:
                    p2 = dict(p)
                    Xc = np.asarray(p.get("keypoints_cam_mm_abs"))
                    if Xc.size:
                        Xw = _cam_to_world(Xc, R_wc0, t_wc0_mm)
                        p2["__tmp_world_export__"] = Xw.tolist()
                    new_persons.append(p2)
                o2 = dict(o); o2["persons"] = new_persons
                frames_world.append(o2)
            export_trc_from_series(frames_world, output_path_trc_world, source_key="__tmp_world_export__", rate_hz=float(args.rate), kp_names=kp_names)

if __name__ == "__main__":
    main()
