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
import poseviz
import cameravision
import time

# ---------- Logging ----------
def log_step(msg): print(f"[STEP] {msg}")
def log_info(msg): print(f"[INFO] {msg}")
def log_warn(msg): print(f"[WARN] {msg}")
def log_err (msg): print(f"[ERROR] {msg}")
def log_done(msg): print(f"[DONE] {msg}")

# ---------- IO helpers ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

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

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise log_err(f"Cannot open video: {vid_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 100.0
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    log_info(f"Num of frames {num_frames}")

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

    model = tf.saved_model.load(download_model('metrabs_eff2l_y4'))
    # --- names: force to plain Python list[str] ---
    _raw_names = model.per_skeleton_joint_names[skeleton].numpy()
    joint_names = [
        (n.decode("utf-8") if isinstance(n, (bytes, np.bytes_)) else str(n)).strip().lower()
        for n in np.ravel(_raw_names).tolist()
    ]

    # --- edges: force to list[tuple[int,int]] with 0-based indices ---
    edges_np = np.asarray(model.per_skeleton_joint_edges[skeleton].numpy())
    edges_np = edges_np.astype(np.int64)

    camera = cameravision.Camera(intrinsic_matrix=K, world_up=(0, 1, 0))

    # Accept both (N,2) and (2,N)
    if edges_np.ndim == 2 and edges_np.shape[0] == 2 and edges_np.shape[1] != 2:
        edges_np = edges_np.T
    edges_np = edges_np.reshape(-1, 2)

    joint_edges = [(int(i), int(j)) for i, j in edges_np.tolist()]
    print(joint_names)
    print(joint_edges)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    out_path = Path("./random_outputs/viz.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with poseviz.PoseViz(joint_names=joint_names, joint_edges=joint_edges, camera_type="free", viz_fps=25) as viz:
        viz.new_sequence_output(str(out_path), fps=25)
        for fi in range(num_frames):
            ok, frame_bgr = cap.read()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image_tf = tf.convert_to_tensor(frame_rgb, dtype=tf.uint8)
            log_info(f"Prediction for frame {fi}")
            pred = model.detect_poses(
                image_tf, skeleton=skeleton,
                #default_fov_degrees = 90,
                intrinsic_matrix=K,
                distortion_coeffs=dist,
                #extrinsic_matrix=T,
                suppress_implausible_poses=False
            )

            boxes = pred['boxes'].numpy()           # (D, 5) or similar
            p3d   = pred['poses3d'].numpy()         # (D, J, 3)
            p2d   = pred['poses2d'].numpy()         # (D, J, 2)

            log_info(f"Viz for frame {fi}")
            viz.update(frame=image_tf, boxes=boxes, poses=p3d, camera=camera)
            #time.sleep(1000)

if __name__ == "__main__":
    # make TF not pre-allocate all VRAM
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        pass
    main()