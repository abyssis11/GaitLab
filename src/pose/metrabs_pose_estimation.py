import os, zipfile, shutil, tempfile, tensorflow as tf
import tensorflow_hub as tfhub
from IO.load_manifest import load_manifest
from pathlib import Path
import argparse
import tensorflow_io as tfio
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import pickle


# ---------- Logging ----------
def log_step(msg): print(f"[STEP] {msg}")
def log_info(msg): print(f"[INFO] {msg}")
def log_warn(msg): print(f"[WARN] {msg}")
def log_err (msg): print(f"[ERROR] {msg}")
def log_done(msg): print(f"[DONE] {msg}")

# ---------- IO helpers ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def project_cam(P_cam, K):
    X,Y,Z = P_cam[:,0], P_cam[:,1], P_cam[:,2] + 1e-9
    u = K[0,0]*X/Z + K[0,2]
    v = K[1,1]*Y/Z + K[1,2]
    return np.stack([u,v],1)

def download_model(model_type: str) -> str:
    server_prefix = 'https://omnomnom.vision.rwth-aachen.de/data/metrabs'
    fname = f'{model_type}_20211019.zip'
    zip_path = tf.keras.utils.get_file(
        fname=fname, origin=f'{server_prefix}/{fname}',
        cache_subdir='models', extract=False
    )
    zip_path = Path(zip_path)
    out_dir = zip_path.parent / model_type
    if (out_dir / 'saved_model.pb').exists():
        return str(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    with zipfile.ZipFile(zip_path, 'r') as zf, tempfile.TemporaryDirectory(dir=str(zip_path.parent)) as tmpdir:
        zf.extractall(tmpdir)
        top = Path(tmpdir)
        candidate = None
        for p in top.iterdir():
            if p.is_dir() and (p / 'saved_model.pb').exists():
                candidate = p; break
        if candidate is None and (top / 'saved_model.pb').exists():
            candidate = top
        if candidate is None:
            raise RuntimeError("Downloaded archive does not contain a SavedModel.")
        shutil.move(str(candidate), str(out_dir))
    if not (out_dir / 'saved_model.pb').exists():
        raise RuntimeError("Model extraction failed (saved_model.pb missing).")
    return str(out_dir)

# ---------- Calibration (Tsai-style) ----------
def _euler_to_R(rx, ry, rz, order='XYZ', angles_in_degrees=True):
    if angles_in_degrees:
        rx, ry, rz = np.deg2rad([rx, ry, rz])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=np.float64)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float64)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=np.float64)
    ops = {'X': Rx, 'Y': Ry, 'Z': Rz}
    R = np.eye(3, dtype=np.float64)
    for ch in order: R = ops[ch] @ R
    return R

def load_calibration_from_xml(xml_path, img_w:int, img_h:int,
                              euler_order='XYZ', angles_in_degrees=True):
    root = ET.parse(str(xml_path)).getroot()
    geo = root.find(".//Geometry"); intr = root.find(".//Intrinsic"); extr = root.find(".//Extrinsic")
    if geo is None or intr is None or extr is None:
        raise ValueError("Calibration XML missing Geometry/Intrinsic/Extrinsic nodes")

    Wc = int(float(geo.attrib.get("width"))); Hc = int(float(geo.attrib.get("height")))
    dpx = float(geo.attrib.get("dpx", geo.attrib.get("dx", "1.0")))
    dpy = float(geo.attrib.get("dpy", geo.attrib.get("dy", "1.0")))

    focal = float(intr.attrib.get("focal"))
    cx    = float(intr.attrib.get("cx")); cy = float(intr.attrib.get("cy"))
    sx    = float(intr.attrib.get("sx", 1.0))
    k1    = float(intr.attrib.get("kappa1", 0.0))

    if (dpx != 1.0) or (dpy != 1.0):
        fx = (sx * focal) / dpx; fy = focal / dpy
    else:
        fx = sx * focal; fy = focal
    K = np.array([[fx, 0.0, cx],[0.0, fy, cy],[0.0, 0.0, 1.0]], dtype=np.float32)
    dist = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    rx = float(extr.attrib.get("rx")); ry = float(extr.attrib.get("ry")); rz = float(extr.attrib.get("rz"))
    R  = _euler_to_R(rx, ry, rz, order=euler_order, angles_in_degrees=angles_in_degrees)
    t  = np.array([float(extr.attrib.get("tx")), float(extr.attrib.get("ty")), float(extr.attrib.get("tz"))], dtype=np.float64)

    T = np.eye(4, dtype=np.float32); T[:3,:3] = R.astype(np.float32); T[:3,3] = t.astype(np.float32)

    if (Wc != img_w) or (Hc != img_h):
        S = np.array([[img_w/float(Wc), 0, 0],[0, img_h/float(Hc), 0],[0, 0, 1]], dtype=np.float32)
        K = (S @ K).astype(np.float32)
    return K, dist, T

# ---------- Heel synthesis from ankle/toe/knee ----------
def synthesize_heel(ankle, toe, knee):
    fwd = toe - ankle; L = np.linalg.norm(fwd) + 1e-6; fwd /= L
    up  = ankle - knee; up /= (np.linalg.norm(up) + 1e-6)
    lat = np.cross(up, fwd); lat /= (np.linalg.norm(lat) + 1e-6)
    heel = ankle - 0.30 * L * fwd  # ~30% of ankle-to-toe behind the ankle
    return heel

# ---------- TRC writer ----------
def write_trc(path, fps, num_frames, marker_names, frames_xyz):
    """
    path: output .trc file path
    fps: float
    num_frames: int
    marker_names: list of 16 names (strings)
    frames_xyz: list length=num_frames; each is (16,3) np array (mm). Use np.nan for missing.
    """
    with open(path, 'w') as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{Path(path).name}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps:.2f}\t{fps:.2f}\t{num_frames}\t{len(marker_names)}\tmm\t{fps:.2f}\t1\t{num_frames}\n")
        # Header names
        f.write("Frame#\tTime\t" + "\t".join(marker_names) + "\t\n")
        # Component headers
        comps = []
        for i in range(1, len(marker_names)+1):
            comps += [f"X{i}", f"Y{i}", f"Z{i}"]
        f.write("\t\t" + "\t".join(comps) + "\n")
        # Data
        for i, xyz in enumerate(frames_xyz, start=1):
            t = (i-1)/fps
            flat = xyz.reshape(-1)
            vals = [f"{v:.5f}" if np.isfinite(v) else "" for v in flat]
            f.write(f"{i}\t{t:.5f}\t" + "\t".join(vals) + "\n")

def load_calibration_tsai(xml_path, img_w, img_h):
    root = ET.parse(str(xml_path)).getroot()
    geo = root.find(".//Geometry")
    intr = root.find(".//Intrinsic")
    extr = root.find(".//Extrinsic")
    if geo is None or intr is None or extr is None:
        raise ValueError("Calibration XML missing Geometry/Intrinsic/Extrinsic nodes")

    # --- Intrinsics ---
    Wc = int(float(geo.attrib.get("width")))
    Hc = int(float(geo.attrib.get("height")))
    dpx = float(geo.attrib.get("dpx", geo.attrib.get("dx", "1.0")))
    dpy = float(geo.attrib.get("dpy", geo.attrib.get("dy", "1.0")))

    focal = float(intr.attrib.get("focal"))   # ≈ 536.266
    sx    = float(intr.attrib.get("sx", 1.0)) # ≈ 1.00233
    cx    = float(intr.attrib.get("cx"))      # 480
    cy    = float(intr.attrib.get("cy"))      # 270
    k1    = float(intr.attrib.get("kappa1", 0.0))

    # Respect pixel pitch if provided (here it's 1.0, so no change)
    fx = (sx * focal) / dpx
    fy = (focal)      / dpy

    # If your current image size differs, scale K
    if (Wc != img_w) or (Hc != img_h):
        sx_scale = img_w / float(Wc)
        sy_scale = img_h / float(Hc)
        fx *= sx_scale
        fy *= sy_scale
        cx *= sx_scale
        cy *= sy_scale

    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,   0,  1]], dtype=np.float32)

    dist = np.array([k1, 0, 0, 0, 0], dtype=np.float32)

    # --- Extrinsics ---
    # Interpret rx,ry,rz as a Rodrigues rotation vector (radians), and
    # tx,ty,tz as the CAMERA CENTER C in world coordinates (mm).
    rvec = np.array([
        float(extr.attrib.get("rx")),
        float(extr.attrib.get("ry")),
        float(extr.attrib.get("rz")),
    ], dtype=np.float64)

    R, _ = cv2.Rodrigues(rvec)  # 3x3

    C = np.array([
        float(extr.attrib.get("tx")),
        float(extr.attrib.get("ty")),
        float(extr.attrib.get("tz")),
    ], dtype=np.float64)  # mm

    # Build world->camera transform: t = -R @ C
    t = (-R @ C.reshape(3,1)).ravel()

    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return K, dist, T

def load_calibration_pickle(pkl_path, img_w, img_h,
                            rot_is_world_to_cam=True,
                            units='mm'):
    """
    pkl contains:
      - 'intrinsicMat': (3,3)
      - 'distortion'  : (1,5)
      - 'imageSize'   : (2,1) [width, height]
      - 'rotation'    : (3,3)
      - 'translation' : (3,1)
    Returns K (3x3), dist (5,), T (4x4) in float32.
    T maps world->camera: X_cam = R * X_world + t
    Scales K if image size differs from current (img_w,img_h).
    """
    with open(pkl_path, 'rb') as f:
        calib = pickle.load(f)

    K0   = np.asarray(calib['intrinsicMat'], dtype=np.float64)
    dist = np.asarray(calib['distortion'],   dtype=np.float64).reshape(-1)
    size = np.asarray(calib['imageSize'],    dtype=np.float64).reshape(-1)
    R    = np.asarray(calib['rotation'],     dtype=np.float64)
    t    = np.asarray(calib['translation'],  dtype=np.float64).reshape(3)

    # Units
    if units.lower().startswith('m'):  # meters -> millimetres
        t = t * 1000.0

    # If rotation is cam->world, invert to world->cam
    if not rot_is_world_to_cam:
        # X_world = R * X_cam + t  ->  X_cam = R^T X_world - R^T t
        R = R.T
        t = -R @ t

    # Scale intrinsics if image size differs
    # size = [width, height]
    W0, H0 = int(round(size[0])), int(round(size[1]))
    if (W0 != img_w) or (H0 != img_h):
        sx = img_w / float(W0)
        sy = img_h / float(H0)
        K_scaled = np.array([[K0[0,0]*sx, 0.0,        K0[0,2]*sx],
                             [0.0,        K0[1,1]*sy, K0[1,2]*sy],
                             [0.0,        0.0,        1.0      ]], dtype=np.float64)
    else:
        K_scaled = K0

    # Pack outputs
    K_out   = K_scaled.astype(np.float32)
    dist_out= dist.astype(np.float32)
    T       = np.eye(4, dtype=np.float32)
    T[:3,:3]= R.astype(np.float32)
    T[:3, 3]= t.astype(np.float32)
    return K_out, dist_out, T


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--manifest", required=True, help="Path to manifest.yaml")
    ap.add_argument("-p", "--paths", required=True, help="Path to paths.yaml")
    ap.add_argument("--trial", required=True)
    ap.add_argument("--video-field", choices=["video_sync", "video_raw"], default="video_sync")
    ap.add_argument("--skeleton", default="smpl_24")
    ap.add_argument("--out-trc", default=None, help="Output TRC path (defaults next to video)")

    ap.add_argument("--calib-pickle", action="store_true", help="camera calibration pickle")
    ap.add_argument("--pkl-rot-cam2world", action="store_true",
                    help="Set if pickle rotation maps camera->world (we'll invert to world->camera)")
    ap.add_argument("--pkl-units-m", action="store_true",
                    help="Set if pickle translation is in meters (we'll convert to mm)")

    args = ap.parse_args()

    skeleton = args.skeleton
    log_step("Loading and resolving manifest")
    manifest = load_manifest(args.manifest, args.paths)

    # Find trial
    log_step(f"Finding trial '{args.trial}'")
    trial = None
    for subset, trials in manifest.get("trials", {}).items():
        for t in trials:
            if t.get("id") == args.trial:
                trial = t; break
        if trial: break
    if trial is None:
        raise SystemExit(f"[ERROR] Trial '{args.trial}' not found in manifest.")

    # Paths
    base = manifest.get('output_dir')
    trial_root = Path(base) / trial['id']
    metrabs_dir = trial_root / "metrabs"; 
    ensure_dir(metrabs_dir)
    vid_path = trial.get(args.video_field)
    if not vid_path: raise log_err("video path missing in manifest")
    calib = manifest.get('calibration', {}).get('intrinsics_extrinsics')
    if not calib: raise log_err("calibration.intrinsics_extrinsics missing in manifest")



    prediction_trc = metrabs_dir / "metrabs_prediction.trc"
    log_info(f"Video path: {vid_path}")
    log_info(f"Calibration: {calib}")

    # Video
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Cannot open video: {vid_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 100.0
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Grab first frame to initialize shapes/K
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        raise SystemExit("[ERROR] Could not read first frame.")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    H, W = frame_rgb.shape[:2]

    # Calibration
    # Calibration (pickled if provided, else XML)
    log_step("Loading calibration")
    if args.calib_pickle:
        K_np, dist_np, T_np = load_calibration_pickle(
            Path(calib),
            img_w=W, img_h=H,
            rot_is_world_to_cam=not args.pkl_rot_cam2world,
            units=('m' if args.pkl_units_m else 'mm')
        )
    else:
        K_np, dist_np, T_np = load_calibration_tsai(Path(calib), img_w=W, img_h=H)

    K   = tf.constant(K_np,   tf.float32)
    dist= tf.constant(dist_np, tf.float32)
    T   = tf.constant(T_np,   tf.float32)

    # Model
    log_step("Loading model")
    model = tf.saved_model.load(download_model('metrabs_eff2l_y4'))
    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str).tolist()
    name_to_idx = {n: i for i, n in enumerate(joint_names)}

    # Marker mapping (edit if you have better pelvis landmarks for RASI/LASI)
    MARKER_ORDER = [
        "RSHO","LSHO","RASI","LASI","RKNE","LKNE","RANK","LANK",
        "RHEE","LHEE","RTOE","LTOE","RELB","LELB","RWRB","LWRB"
    ]
    MARKER_MAP = {
        "RSHO": "rsho", "LSHO": "lsho",
        "RASI": "rhip", "LASI": "lhip",       # proxies for ASIS
        "RKNE": "rkne", "LKNE": "lkne",
        "RANK": "rank", "LANK": "lank",
        "RTOE": "rtoe", "LTOE": "ltoe",
        "RELB": "relb", "LELB": "lelb",
        "RWRB": "rwri", "LWRB": "lwri",
        # Heels are synthesized per-frame (no direct map)
    }

    # Prepare TRC accumulation
    frames_xyz = []
    # Process first frame we already read
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    log_step(f"Estimating pose on {num_frames} frames @ {fps:.2f} Hz")

    for fi in range(num_frames):
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            # pad with NaNs if video ended early
            xyz = np.full((len(MARKER_ORDER), 3), np.nan, dtype=np.float32)
            frames_xyz.append(xyz); continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_tf = tf.convert_to_tensor(frame_rgb, dtype=tf.uint8)

        pred = model.detect_poses(
            image_tf, skeleton=skeleton,
            intrinsic_matrix=K, 
            distortion_coeffs=dist, 
            extrinsic_matrix=T
        )

        if pred['boxes'].shape[0] == 0:
            xyz = np.full((len(MARKER_ORDER), 3), np.nan, dtype=np.float32)
            frames_xyz.append(xyz); continue

        boxes = pred['boxes'].numpy()
        idx_top = int(np.argmax(boxes[:, -1]))  # highest detection score
        P = pred['poses3d'].numpy()[idx_top]    # (J,3) in mm

        # Synthesize heels
        def J(name): return P[name_to_idx[name]]
        heel_r = synthesize_heel(J("rank"), J("rtoe"), J("rkne"))
        heel_l = synthesize_heel(J("lank"), J("ltoe"), J("lkne"))

        # Assemble markers in required order
        xyz = []
        for m in MARKER_ORDER:
            if m in ("RHEE","LHEE"):
                xyz.append(heel_r if m=="RHEE" else heel_l)
            else:
                jname = MARKER_MAP[m]
                if jname not in name_to_idx:
                    xyz.append(np.array([np.nan, np.nan, np.nan], dtype=np.float32))
                else:
                    xyz.append(J(jname))
        xyz = np.stack(xyz, axis=0).astype(np.float32)  # (16,3)
        frames_xyz.append(xyz)

        if (fi+1) % 50 == 0:
            log_info(f"Processed {fi+1}/{num_frames} frames")

    cap.release()

    # Write TRC
    write_trc(prediction_trc, fps=fps, num_frames=num_frames, marker_names=MARKER_ORDER, frames_xyz=frames_xyz)
    log_done(f"TRC written: {prediction_trc}")

    k = int(tf.argmax(pred['detection_scores'], axis=0)) if 'detection_scores' in pred else 0
    P3d_cam = pred['poses3d'][k].numpy()        # (J,3), camera coords in mm
    p2d_pred = pred['poses2d'][k].numpy()       # (J,2), px
    p2d_proj = project_cam(P3d_cam, K_np)       # K_np = EXACT matrix you passed

    err = np.linalg.norm(p2d_proj - p2d_pred, axis=1).mean()
    print("Mean 2D reprojection error [px]:", err)
    print("Z stats (mm):", P3d_cam[:,2].min(), P3d_cam[:,2].mean(), P3d_cam[:,2].max())
    print(joint_names)

if __name__ == "__main__":
    # make TF not pre-allocate all VRAM
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        pass
    main()
