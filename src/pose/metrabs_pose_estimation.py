import os, zipfile, shutil, tempfile, tensorflow as tf
import tensorflow_hub as tfhub
from IO.load_manifest import load_manifest
from pathlib import Path
import argparse
import tensorflow_io as tfio
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
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

def download_model(model_type: str) -> str:
    """
    Downloads metrabs '<model_type>_20211019.zip' into ~/.keras/datasets/models
    and extracts it to a clean folder, but *skips* extraction if a valid model already exists.
    Returns the extracted model directory path.
    """
    server_prefix = 'https://omnomnom.vision.rwth-aachen.de/data/metrabs'
    fname = f'{model_type}_20211019.zip'

    # 1) Download zip (no auto-extract)
    zip_path = tf.keras.utils.get_file(
        fname=fname,
        origin=f'{server_prefix}/{fname}',
        cache_subdir='models',
        extract=False
    )
    zip_path = Path(zip_path)
    out_dir = zip_path.parent / model_type
    sentinel = out_dir / 'saved_model.pb'  # minimal check that extraction completed

    # 2) If already extracted and looks valid, reuse it
    if sentinel.exists():
        return str(out_dir)

    # 3) Clean any partial leftovers, then extract safely
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # extract to a temp dir first, then move into place atomically
        with tempfile.TemporaryDirectory(dir=str(zip_path.parent)) as tmpdir:
            zf.extractall(tmpdir)
            # some zips contain a top-level folder named model_type; handle both cases
            top = Path(tmpdir)
            # find first dir containing saved_model.pb
            candidate = None
            for p in top.iterdir():
                if p.is_dir() and (p / 'saved_model.pb').exists():
                    candidate = p
                    break
            if candidate is None and (top / 'saved_model.pb').exists():
                candidate = top
            if candidate is None:
                raise RuntimeError("Downloaded archive does not contain a SavedModel.")

            shutil.move(str(candidate), str(out_dir))

    if not sentinel.exists():
        raise RuntimeError("Model extraction failed (saved_model.pb missing).")
    return str(out_dir)

def visualize(image, detections, poses3d, poses2d, edges, save_to=None):
    fig = plt.figure(figsize=(10, 5.2))
    image_ax = fig.add_subplot(1, 2, 1)
    image_ax.imshow(image)
    for x, y, w, h in detections[:, :4]:
        image_ax.add_patch(Rectangle((x, y), w, h, fill=False))

    pose_ax = fig.add_subplot(1, 2, 2, projection='3d')
    pose_ax.view_init(5, -85)
    pose_ax.set_xlim3d(-1500, 1500)
    pose_ax.set_zlim3d(-1500, 1500)
    pose_ax.set_ylim3d(0, 3000)

    poses3d[..., 1], poses3d[..., 2] = poses3d[..., 2], -poses3d[..., 1]
    for pose3d, pose2d in zip(poses3d, poses2d):
        for i_start, i_end in edges:
            image_ax.plot(*zip(pose2d[i_start], pose2d[i_end]), marker='o', markersize=2)
            pose_ax.plot(*zip(pose3d[i_start], pose3d[i_end]), marker='o', markersize=2)
        image_ax.scatter(*pose2d.T, s=2)
        pose_ax.scatter(*pose3d.T, s=2)

    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=200, bbox_inches="tight")
        print(f"[DONE] Saved: {save_to}")

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
    R = np.eye(3, dtype=np.float64)           # <-- fixed
    for ch in order:                          # apply in the given order
        R = ops[ch] @ R
    return R

def load_calibration_from_xml(xml_path, img_w:int, img_h:int,
                              euler_order='XYZ', angles_in_degrees=True):
    root = ET.parse(str(xml_path)).getroot()

    geo = root.find(".//Geometry")
    intr = root.find(".//Intrinsic")
    extr = root.find(".//Extrinsic")
    if geo is None or intr is None or extr is None:
        raise ValueError("Calibration XML missing Geometry/Intrinsic/Extrinsic nodes")

    Wc = int(float(geo.attrib.get("width")))
    Hc = int(float(geo.attrib.get("height")))
    dpx = float(geo.attrib.get("dpx", geo.attrib.get("dx", "1.0")))
    dpy = float(geo.attrib.get("dpy", geo.attrib.get("dy", "1.0")))

    focal = float(intr.attrib.get("focal"))
    cx    = float(intr.attrib.get("cx"))
    cy    = float(intr.attrib.get("cy"))
    sx    = float(intr.attrib.get("sx", 1.0))
    k1    = float(intr.attrib.get("kappa1", 0.0))

    if (dpx != 1.0) or (dpy != 1.0):
        fx = (sx * focal) / dpx
        fy = focal / dpy
    else:
        fx = sx * focal
        fy = focal

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    dist = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    rx = float(extr.attrib.get("rx"))
    ry = float(extr.attrib.get("ry"))
    rz = float(extr.attrib.get("rz"))
    R  = _euler_to_R(rx, ry, rz, order=euler_order, angles_in_degrees=angles_in_degrees)

    t = np.array([float(extr.attrib.get("tx")),
                  float(extr.attrib.get("ty")),
                  float(extr.attrib.get("tz"))], dtype=np.float64)

    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)

    if (Wc != img_w) or (Hc != img_h):
        S = np.array([[img_w/float(Wc), 0, 0],
                      [0, img_h/float(Hc), 0],
                      [0, 0, 1]], dtype=np.float32)
        K = (S @ K).astype(np.float32)

    return K, dist, T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--manifest", required=True, help="Path to manifest.yaml")
    ap.add_argument("-p", "--paths", required=True, help="Path to paths.yaml")
    ap.add_argument("--trial", required=True)
    ap.add_argument("--video-field", choices=["video_sync", "video_raw"], default="video_sync")

    args = ap.parse_args()
    skeleton = 'smpl_24'
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
    subj = manifest.get('subject_id', 'subject')
    sess = manifest.get('session', 'Session')
    cam  = manifest.get('camera', 'Cam')
    calibration = manifest.get('calibration')
    trial_root = Path(base) / trial['id']
    metrabs = trial_root / "metrabs"
    vid_path = trial.get(args.video_field)
    ensure_dir(metrabs)

    calib = calibration.get('intrinsics_extrinsics') if calibration else None
    log_info(f"Trial root: {trial_root}")
    log_info(f"metrabs dir: {metrabs}")
    log_info(f"Video path: {vid_path}")
    log_info(f"Calibration path: {calib}")

    # ---- Grab a frame ----
    cap = cv2.VideoCapture(vid_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise SystemExit("[ERROR] Could not read a frame from the video.")

    # Convert to RGB uint8 (H,W,3)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    H, W = frame_rgb.shape[:2]
    image = tf.convert_to_tensor(frame_rgb, dtype=tf.uint8)

    # ---- Load calibration from XML ----
    if calib is None:
        raise SystemExit("[ERROR] No 'calibration.intrinsics_extrinsics' path in manifest.")
    calib_path = Path(calib)
    if not calib_path.exists():
        raise SystemExit(f"[ERROR] Calibration XML not found: {calib_path}")

    K_np, dist_np, T_np = load_calibration_from_xml(calib_path, img_w=W, img_h=H)
    K   = tf.constant(K_np,   tf.float32)     # (3,3)
    dist= tf.constant(dist_np, tf.float32)    # (5,)
    T   = tf.constant(T_np,   tf.float32)     # (4,4)

    # ---- Run MeTRAbs ----
    model = tf.saved_model.load(download_model('metrabs_eff2l_y4'))
    pred = model.detect_poses(
        image,
        skeleton=skeleton,
        intrinsic_matrix=K,
        extrinsic_matrix=T,
        distortion_coeffs=dist
    )

    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
    print(joint_names)
    print(pred['poses3d'])

    visualize(
        image.numpy(), 
        pred['boxes'].numpy(),
        pred['poses3d'].numpy(),
        pred['poses2d'].numpy(),
        model.per_skeleton_joint_edges[skeleton].numpy(),
        save_to=str(metrabs / "visual.png")
    )

if __name__ == "__main__":
    main()
