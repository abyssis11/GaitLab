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
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

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

# ---------- Heel & small-toe synthesis ----------
def synthesize_heel(ankle, toe, knee):
    fwd = toe - ankle; L = np.linalg.norm(fwd) + 1e-6; fwd /= L
    up  = ankle - knee; up /= (np.linalg.norm(up) + 1e-6)
    lat = np.cross(up, fwd); lat /= (np.linalg.norm(lat) + 1e-6)
    heel = ankle - 0.30 * L * fwd  # ~30% of ankle-to-toe behind the ankle
    return heel

def synthesize_small_toe(ankle, big_toe, knee, side: str):
    """
    Create a synthetic 'small toe' marker using a lateral offset from the big toe.
    side: 'right' or 'left'
    """
    fwd = big_toe - ankle; L = np.linalg.norm(fwd) + 1e-6; fwd /= L
    up  = ankle - knee; up /= (np.linalg.norm(up) + 1e-6)
    lat = np.cross(up, fwd); lat /= (np.linalg.norm(lat) + 1e-6)
    # Lateral offset ~15% of foot length. Sign: + for right, - for left (consistent RH rule).
    sgn = +1.0 if side.lower().startswith('r') else -1.0
    small_toe = big_toe + sgn * 0.15 * L * lat
    return small_toe

# ---------- TRC writer ----------
def write_trc(path, fps, num_frames, marker_names, frames_xyz):
    """
    path: output .trc file path
    fps: float
    num_frames: int
    marker_names: list of N names (strings)
    frames_xyz: list length=num_frames; each is (N,3) np array (mm). Use np.nan for missing.
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

def visualize(image, detections, poses3d, poses2d, edges, save_to=None,
              use_extrinsics=False, R=None, t=None):
    fig = plt.figure(figsize=(10, 5.2))
    image_ax = fig.add_subplot(1, 2, 1)
    image_ax.imshow(image)
    for x, y, w, h in detections[:, :4]:
        image_ax.add_patch(Rectangle((x, y), w, h, fill=False))

    pose_ax = fig.add_subplot(1, 2, 2, projection='3d')
    pose_ax.view_init(5, -85)

    # Prepare a COPY for visualization (don't mutate inputs).
    poses3d_vis = poses3d.copy()
    # If extrinsics were used, the model returned WORLD coords.
    # For visualization, convert WORLD -> CAMERA, then use the usual camera->plot mapping.
    if use_extrinsics and (R is not None) and (t is not None):
        # poses3d_vis: (D, J, 3)
        Rt = R.astype(np.float64)
        tt = np.asarray(t, dtype=np.float64).reshape(3, 1)
        poses3d_vis = (Rt @ poses3d_vis.transpose(0, 2, 1) + tt).transpose(0, 2, 1)

    # Now poses3d_vis is in CAMERA frame (OpenCV: x right, y down, z forward).
    # Map to matplotlib-friendly coords with Z up.
    y = poses3d_vis[..., 1].copy()
    z = poses3d_vis[..., 2].copy()
    poses3d_vis[..., 1] = z
    poses3d_vis[..., 2] = -y
    # Auto-center & equal-scale the axes around the visible points.
    pts = poses3d_vis.reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    if finite.any():
        pts = pts[finite]
        ctr = pts.mean(axis=0)
        span = np.ptp(pts, axis=0).max()
        r = max(300.0, span * 0.6)  # at least a 300 mm cube
        pose_ax.set_xlim(ctr[0]-r, ctr[0]+r)
        pose_ax.set_ylim(ctr[1]-r, ctr[1]+r)
        pose_ax.set_zlim(ctr[2]-r, ctr[2]+r)
        try:
            pose_ax.set_box_aspect((1,1,1))
        except Exception:
            pass
    for pose3d, pose2d in zip(poses3d_vis, poses2d):
        for i_start, i_end in edges:
            image_ax.plot(*zip(pose2d[i_start], pose2d[i_end]), marker='o', markersize=2)
            pose_ax.plot(*zip(pose3d[i_start], pose3d[i_end]), marker='o', markersize=2)
        image_ax.scatter(*pose2d.T, s=2)
        pose_ax.scatter(*pose3d.T, s=2)

    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=200, bbox_inches="tight")
        print(f"[DONE] Saved: {save_to}")

def fit_plane(points):
    # points: (N,3), returns unit normal and point-on-plane (centroid)
    C = points.mean(axis=0)
    U, S, Vt = np.linalg.svd(points - C, full_matrices=False)
    n = Vt[-1]                  # plane normal (unit)
    if n[1] < 0: n = -n         # prefer pointing upward-ish (positive Y)
    return n/np.linalg.norm(n), C

def estimate_osim_basis(frames_xyz, marker_names):
    # Collect foot points across frames to detect the floor
    foot_labels = [l for l in marker_names
                   if any(k in l.lower() for k in ["ank", "toe", "heel"])]
    idxs = [marker_names.index(l) for l in foot_labels if l in marker_names]
    foot_samples = []
    for F in frames_xyz:
        P = F[idxs]                          # (k,3)
        P = P[np.isfinite(P).all(axis=1)]    # drop NaNs
        if len(P) >= 3:
            foot_samples.append(P)
    if not foot_samples:
        raise RuntimeError("No foot samples to estimate ground plane.")
    foot_all = np.concatenate(foot_samples, axis=0)

    # Up = plane normal (Y_osim)
    up, _ = fit_plane(foot_all)              # (3,)

    # Forward = pelvis/midHip displacement projected onto plane (X_osim)
    # pick your pelvis label here:
    cand = ["midHip","pelv","pelvis","pelvis_mid","pelvismarker"]
    pelvis_name = next((c for c in cand if c in marker_names), None)
    if pelvis_name is None:
        raise RuntimeError("No pelvis/midHip marker found for forward estimation.")
    ip = marker_names.index(pelvis_name)
    traj = np.array([F[ip] for F in frames_xyz])
    traj = traj[np.isfinite(traj).all(axis=1)]
    if len(traj) < 2:
        raise RuntimeError("Not enough pelvis samples for forward estimation.")
    disp = traj[-1] - traj[0]
    # Project onto ground plane
    disp_proj = disp - np.dot(disp, up) * up
    if np.linalg.norm(disp_proj) < 1e-6:
        # fallback: use mid-stance foot direction if no translation (e.g., static)
        # take vector toe->heel average direction
        foot_dir = foot_all[:, :]  # already have lots of points
        # crude fallback: X along global X if we cannot infer
        disp_proj = np.array([1.0, 0.0, 0.0])
    fwd = disp_proj / (np.linalg.norm(disp_proj) + 1e-12)

    # Right = cross(Forward, Up) (Z_osim)
    right = np.cross(fwd, up)
    right /= (np.linalg.norm(right) + 1e-12)

    # Orthonormalize: recompute forward = right x up to ensure orthonormal frame
    fwd = np.cross(up, right)
    fwd /= (np.linalg.norm(fwd) + 1e-12)

    # Build rotation R such that: v_osim = R @ v_lab
    # Columns of R are the images of lab basis eX,eY,eZ in OSIM coords,
    # but we want rows that pick lab vector components into OSIM axes.
    # Using basis vectors directly: R rows are [fwd; up; right] in lab coords.
    R_lab_to_osim = np.stack([fwd, up, right], axis=0)   # 3x3
    if np.linalg.det(R_lab_to_osim) < 0:
        # fix potential left-handedness
        right = -right
        R_lab_to_osim = np.stack([fwd, up, right], axis=0)
    return R_lab_to_osim


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

    # NEW: Write TRC using OpenSim-style target joint names/order
    ap.add_argument("--for-opensim", action="store_true",
                    help="If set, write TRC with OpenSim-style joint names and synthesized small toes")
    ap.add_argument("--swap-yz-in-loop", action="store_true",
                    help="Convert world->camera then swap y/z (y:=z, z:=-y) before assembling markers")
    

    args = ap.parse_args()

    first_img = None
    first_boxes = None
    first_p3d = None
    first_p2d = None

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
    metrabs_dir = trial_root / "metrabs"
    ensure_dir(metrabs_dir)
    vid_path = trial.get(args.video_field)
    if not vid_path: raise log_err("video path missing in manifest")
    calib = manifest.get('calibration', {}).get('intrinsics_extrinsics')
    if not calib: raise log_err("calibration.intrinsics_extrinsics missing in manifest")

    # Choose default TRC output name if not provided
    default_name = "metrabs_opensim_prediction.trc" if args.for_opensim else "metrabs_prediction.trc"
    prediction_trc = Path(args.out_trc) if args.out_trc else (metrabs_dir / default_name)

    log_info(f"Video path: {vid_path}")
    log_info(f"Calibration: {calib}")
    log_info(f"Output TRC:  {prediction_trc}")

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

    # -------- Marker configurations --------
    # Original/source-style
    MARKER_ORDER_SRC = [
        "RSHO","LSHO","RASI","LASI","RKNE","LKNE","RANK","LANK",
        "RHEE","LHEE","RTOE","LTOE","RELB","LELB","RWRB","LWRB", "midHip"
    ]
    MARKER_MAP_SRC = {
        "RSHO": "rsho", "LSHO": "lsho",
        "RASI": "rhip", "LASI": "lhip",       # proxies for ASIS
        "RKNE": "rkne", "LKNE": "lkne",
        "RANK": "rank", "LANK": "lank",
        "RTOE": "rtoe", "LTOE": "ltoe",
        "RELB": "relb", "LELB": "lelb",
        "RWRB": "rwri", "LWRB": "lwri", "midHip":"pelv"
        # Heels synthesized per-frame
    }

    # OpenSim-style target joints (requested)
    MARKER_ORDER_OS = [
        "neck","right_shoulder","left_shoulder","right_hip","left_hip", "right_elbow", "left_elbow", "right_wrist", "left_wrist",
        "right_knee","left_knee","right_ankle","left_ankle",
        "right_heel","left_heel","right_small_toe","left_small_toe",
        "right_big_toe","left_big_toe","midHip"
    ]
    MARKER_MAP_OS = {
        "right_elbow":"relb",
        "left_elbow":"lelb",
        "right_wrist":"rwri",
        "left_wrist":"lwri",
        "neck": "neck",
        "right_shoulder": "rsho", "left_shoulder": "lsho",
        "right_hip": "rhip", "left_hip": "lhip",
        "right_knee": "rkne", "left_knee": "lkne",
        "right_ankle": "rank", "left_ankle": "lank",
        # heels & small toes synthesized
        "right_big_toe": "rtoe", "left_big_toe": "ltoe",
        "midHip": "pelv",
    }

    # Select active set
    if args.for_opensim:
        MARKER_ORDER = MARKER_ORDER_OS
        MARKER_MAP = MARKER_MAP_OS
        log_info("OpenSim mode: writing TRC with target joint names & synthesized small toes")
    else:
        MARKER_ORDER = MARKER_ORDER_SRC
        MARKER_MAP = MARKER_MAP_SRC

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
            #default_fov_degrees = 90,
            intrinsic_matrix=K,
            distortion_coeffs=dist,
            extrinsic_matrix=T,
            suppress_implausible_poses=False
        )

        if pred['boxes'].shape[0] == 0:
            xyz = np.full((len(MARKER_ORDER), 3), np.nan, dtype=np.float32)
            frames_xyz.append(xyz); continue

        boxes = pred['boxes'].numpy()
        idx_top = int(np.argmax(boxes[:, -1]))  # highest detection score
        P = pred['poses3d'].numpy()[idx_top]    # (J,3) in mm (WORLD if extrinsic_matrix was passed)
        # If requested, transform WORLD -> CAMERA and apply the camera->plot swap (y:=z, z:=-y)
        if args.swap_yz_in_loop:
            y_ = P[:, 1].copy()
            z_ = P[:, 2].copy()
            P[:, 1] = z_
            P[:, 2] = -y_

        # Helper to fetch a joint
        def J(name): return P[name_to_idx[name]]

        # Synthesize heels
        heel_r = synthesize_heel(J("rank"), J("rtoe"), J("rkne"))
        heel_l = synthesize_heel(J("lank"), J("ltoe"), J("lkne"))

        if fi == 60:
            first_img = image_tf.numpy()                  # RGB HxWx3, uint8
            first_boxes = pred['boxes'].numpy()           # (D, 5) or similar
            first_p3d   = pred['poses3d'].numpy()         # (D, J, 3)
            first_p2d   = pred['poses2d'].numpy()         # (D, J, 2)

        # Synthesize small toes (OpenSim mode only; ignored otherwise)
        if args.for_opensim:
            small_toe_r = synthesize_small_toe(J("rank"), J("rtoe"), J("rkne"), side="right")
            small_toe_l = synthesize_small_toe(J("lank"), J("ltoe"), J("lkne"), side="left")

        # Assemble markers in required order
        xyz_list = []
        for m in MARKER_ORDER:
            if not args.for_opensim:
                # Source-style set
                if m in ("RHEE","LHEE"):
                    xyz_list.append(heel_r if m=="RHEE" else heel_l)
                else:
                    jname = MARKER_MAP[m]
                    if jname not in name_to_idx:
                        xyz_list.append(np.array([np.nan, np.nan, np.nan], dtype=np.float32))
                    else:
                        xyz_list.append(J(jname))
            else:
                # OpenSim-style set
                if m == "right_heel":
                    xyz_list.append(heel_r)
                elif m == "left_heel":
                    xyz_list.append(heel_l)
                elif m == "right_small_toe":
                    xyz_list.append(small_toe_r)
                elif m == "left_small_toe":
                    xyz_list.append(small_toe_l)
                else:
                    jname = MARKER_MAP.get(m, None)
                    if (jname is None) or (jname not in name_to_idx):
                        xyz_list.append(np.array([np.nan, np.nan, np.nan], dtype=np.float32))
                    else:
                        xyz_list.append(J(jname))

        xyz = np.stack(xyz_list, axis=0).astype(np.float32)
        frames_xyz.append(xyz)

        if (fi+1) % 50 == 0:
            log_info(f"Processed {fi+1}/{num_frames} frames")

    cap.release()

    R_lab_to_osim = estimate_osim_basis(frames_xyz, MARKER_ORDER)
    frames_xyz = [ (R_lab_to_osim @ F.T).T.astype(np.float32) for F in frames_xyz ]

    # Write TRC
    write_trc(prediction_trc, fps=fps, num_frames=num_frames, marker_names=MARKER_ORDER, frames_xyz=frames_xyz)
    log_done(f"TRC written: {prediction_trc}")

    # Debug stats
    k = int(tf.argmax(pred['detection_scores'], axis=0)) if 'detection_scores' in pred else 0
    R = T_np[:3, :3]
    t = T_np[:3, 3:4]
    P3d_pred = pred['poses3d'][k].numpy()  
    P3d_cam = (R @ P3d_pred.T + t).T
    #P3d_cam = pred['poses3d'][k].numpy()        # (J,3), camera coords in mm
    p2d_pred = pred['poses2d'][k].numpy()       # (J,2), px
    p2d_proj = project_cam(P3d_cam, K_np)       # K_np = EXACT matrix you passed

    err = np.linalg.norm(p2d_proj - p2d_pred, axis=1).mean()
    print("Mean 2D reprojection error [px]:", err)
    print("Z stats (mm):", P3d_cam[:,2].min(), P3d_cam[:,2].mean(), P3d_cam[:,2].max())
    print(joint_names)

    visualize(
        first_img,
        first_boxes,
        first_p3d,
        first_p2d,
        model.per_skeleton_joint_edges[skeleton].numpy(),
        save_to=str("visual5.png"),
        # If you passed extrinsic_matrix=T above, set True so we DON'T apply camera-frame swap.
        use_extrinsics=True,
        R=T_np[:3, :3],
        t=T_np[:3, 3]
     )

if __name__ == "__main__":
    # make TF not pre-allocate all VRAM
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        pass
    main()
