
#!/usr/bin/env python3
'''
python src/pose/canonicalize_preds_metric.py \
    -m ./manifests/OpenCapDataset/subject2.yaml \
    -p ./config/paths.yaml \
    --trial static2 \
    --export-trc \
    --rate 60 \
    --mode first-frame
'''
import argparse, json
from pathlib import Path
import numpy as np
from IO.load_manifest import load_manifest
import os

# ---------- Logging ----------
def log_step(msg): print(f"[STEP] {msg}")
def log_info(msg): print(f"[INFO] {msg}")
def log_warn(msg): print(f"[WARN] {msg}")
def log_err (msg): print(f"[ERROR] {msg}")
def log_done(msg): print(f"[DONE] {msg}")

# ---------- IO helpers ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _normalize(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > eps, v / n, 0.0*v)

def _infer_indices_coarse(K: int):
    """
    Best-effort indices for COCO-WholeBody-133 layout.
    Returns dict with indices: left_shoulder, right_shoulder, left_hip, right_hip.
    """
    if K >= 17:
        # H3WB ordering
        return {"left_shoulder": 5, "right_shoulder": 6, "left_hip": 11, "right_hip": 12}
    else:
        return {}

def _compute_neck_midhip(kps, idxs):
    LSh = kps[idxs["left_shoulder"]] if idxs.get("left_shoulder") is not None else None
    RSh = kps[idxs["right_shoulder"]] if idxs.get("right_shoulder") is not None else None
    left_hip = kps[idxs["left_hip"]] if idxs.get("left_hip") is not None else None
    right_hip = kps[idxs["right_hip"]] if idxs.get("right_hip") is not None else None

    neck = None
    midhip = None
    if LSh is not None and RSh is not None and np.isfinite(LSh).all() and np.isfinite(RSh).all():
        neck = 0.5*(LSh + RSh)
    if left_hip is not None and right_hip is not None and np.isfinite(left_hip).all() and np.isfinite(right_hip).all():
        midhip = 0.5*(left_hip + right_hip)
    return neck, midhip

def _canonical_rotation(neck, midhip, left_hip, right_hip):
    """
    Build a right-handed body-centric rotation R such that:
      e_y (world up) = normalize(neck - midhip)
      e_z (world right) = normalize(right_hip - left_hip)
      e_x (world forward) = normalize(cross(e_y, e_z))
    Then re-orthonormalize to be safe.
    Returns R_world_from_cam (3x3).
    """
    up = neck - midhip
    right = right_hip - left_hip
    e_y = _normalize(up)
    e_z = _normalize(right)
    e_x = np.cross(e_y, e_z)
    e_x = _normalize(e_x)
    # recompute e_z to ensure orthonormality
    e_z = _normalize(np.cross(e_x, e_y))
    R = np.stack([e_x, e_y, e_z], axis=0)  # rows are world axes in camera basis
    return R

def _apply_rot_center(kps, center, R):
    """
    kps: (K,3) in camera metric (mm), center: (3,), R: (3,3) rows world axes in camera basis.
    Return canonical coords: (kps - center) in camera, then rotated into world via R.
    Given R rows are world axes in camera basis, x_world = e_x · v_cam, so v_world = R @ v_cam
    """
    v = (kps - center[None, :])
    return (R @ v.T).T

def load_frames(in_path):
    frames = []
    with open(in_path, "r", encoding="utf-8") as f:
        for ln in f:
            o = json.loads(ln)
            frames.append(o)
    return frames

def write_frames(frames, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for o in frames:
            f.write(json.dumps(o) + "\n")

def export_trc(frames, out_path, kp_names: list | None , source_key="keypoints_xyz_mm_canonical",
               rate_hz=60.0):
    # collect (F,K,3)
    seq = []
    for o in frames:
        persons = o.get("persons") or []
        if not persons: 
            seq.append(None); continue
        p = persons[0]
        if source_key not in p: 
            seq.append(None); continue
        arr = np.asarray(p[source_key], dtype=float)
        seq.append(arr if arr.ndim==2 and arr.shape[1]==3 else None)
    F = len(seq)
    K = max((a.shape[0] for a in seq if a is not None), default=0)
    data = np.full((F, K, 3), np.nan, dtype=float)
    for i, a in enumerate(seq):
        if a is None: 
            continue
        data[i,:a.shape[0],:] = a[:K]
    # add synthetic neck & midHip if possible
    # We assume COCO-17 indices are at 0..16 for body; shoulders 5,6; hips 11,12
    add_neck = K > 6
    add_midhip = K > 12
    if add_neck:
        neck = 0.5 * (data[:,5,:] + data[:,6,:])
        data = np.concatenate([data, neck[:,None,:]], axis=1)
        K += 1
        kp_names.append("neck")
    if add_midhip:
        midhip = 0.5 * (data[:,11,:] + data[:,12,:])
        data = np.concatenate([data, midhip[:,None,:]], axis=1)
        K += 1
        kp_names.append("midHip")
    # headers
    with open(out_path, "w", encoding="utf-8") as fout:
        fout.write(f"PathFileType\t4\t(X/Y/Z)\t{out_path}\n")
        fout.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        fout.write(f"{rate_hz:.1f}\t{rate_hz:.1f}\t{F}\t{K}\tmm\t{rate_hz:.1f}\t1\t{F}\n")
        # names (fallback generic)
        if kp_names and len(kp_names) >= K:
            names = kp_names[:K]
        else:
            names = [f"J{i+1}" for i in range(K)]
        labels = "\t\t\t".join(names)
        fout.write(f"Frame#\tTime\t{labels}\t\t\t\n")
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

def canonicalize(in_path, out_path, mode="first-frame", use_abs = False):
    frames = load_frames(in_path)
    # detect K
    first_p = None
    for o in frames:
        persons = o.get("persons") or []
        if persons:
            first_p = persons[0]
            break
    if not first_p:
        log_err("No persons in input.")
    kps0 = np.asarray(first_p.get(f"keypoints_{'cam_mm_abs' if use_abs else 'xyz_mm'}"), dtype=float)
    if kps0.ndim != 2 or kps0.shape[1] != 3:
       log_err("keypoints missing or invalid shape.")
    K = kps0.shape[0]
    idxs = _infer_indices_coarse(K)
    if not idxs:
        log_err(f"Unsupported K={K}; cannot infer hips/shoulders without keypoint_names.")

    centers = []
    Rs = []
    for o in frames:
        persons = o.get("persons") or []
        if not persons:
            centers.append(None); Rs.append(None); continue
        A = np.asarray(persons[0].get(f"keypoints_{'cam_mm_abs' if use_abs else 'xyz_mm'}"), dtype=float)
        if A.ndim != 2 or A.shape[1] != 3 or A.shape[0] < K:
            centers.append(None); Rs.append(None); continue

        neck, midhip = _compute_neck_midhip(A, idxs)
        if neck is None or midhip is None:
            centers.append(None); Rs.append(None); continue

        left_hip = A[idxs["left_hip"]]
        right_hip = A[idxs["right_hip"]]
        R = _canonical_rotation(neck, midhip, left_hip, right_hip)
        centers.append(midhip)
        Rs.append(R)

    if mode == "first-frame":
        R0 = next((R for R in Rs if R is not None), None)
        if R0 is None:
            log_err("Could not compute a single canonical rotation from any frame.")
        Rs = [R0 if R is not None else None for R in Rs]

    out_frames = []
    for o, center, R in zip(frames, centers, Rs):
        persons = o.get("persons") or []
        new_persons = []
        for p in persons:
            A = np.asarray(p.get(f"keypoints_{'cam_mm_abs' if use_abs else 'xyz_mm'}"), dtype=float)
            if A.ndim != 2 or A.shape[1] != 3:
                continue
            p2 = dict(p)
            if center is not None:
                centered = A - center[None,:]
                p2["keypoints_xyz_mm_centered"] = centered.tolist()
                if R is not None:
                    canon = _apply_rot_center(A, center, R)
                    p2["keypoints_xyz_mm_canonical"] = canon.tolist()
            new_persons.append(p2)
        o2 = dict(o)
        o2["persons"] = new_persons
        if center is not None:
            o2["canonical_center_mm"] = center.tolist()
        if R is not None:
            o2["canonical_R_world_from_cam_rows"] = R.tolist()
        out_frames.append(o2)

    with open(out_path, "w", encoding="utf-8") as f:
        for o in out_frames:
            f.write(json.dumps(o) + "\n")

    return out_frames
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--manifest", required=True)
    ap.add_argument("-p", "--paths", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--in", dest="inp", help="Input preds_metric.jsonl")
    ap.add_argument("--out", dest="outp", help="Output preds_metric_canon.jsonl")
    ap.add_argument("--mode", choices=["first-frame","per-frame"], default="first-frame",
                    help="Use single rotation from first valid frame, or per-frame rotations.")
    ap.add_argument("--export-trc", action="store_true", help="Optional TRC export option")
    ap.add_argument("--rate", dest="rate", type=float, default=60.0, help="TRC sample rate assumption.")
    ap.add_argument("--use-abs", action="store_true", default=False)
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
        raise SystemExit(f"[ERROR] Trial '{args.trial}' not found in manifest.")
    
    # Paths
    base = manifest.get('output_dir')
    if not base:
        subj = manifest.get('subject_id', 'subject')
        sess = manifest.get('session', 'Session')
        cam  = manifest.get('camera', 'Cam')
        base = Path(manifest.get('outputs_root', Path.cwd() / "outputs")) / subj / sess / cam
    trial_root = Path(base) / trial['id']
    rtmw3d_dir = trial_root / "rtmw3d"
    preds_metric = rtmw3d_dir / f"preds_metric{'_abs' if args.use_abs else ''}.jsonl"
    meta_path  = trial_root / "meta.json"

    log_info(f"Trial root : {trial_root}")
    log_info(f"Preds metric path : {preds_metric}")
    log_info(f"meta.json  : {meta_path}")

    output_path = rtmw3d_dir / f"preds_metric{'_abs' if args.use_abs else ''}_canon.jsonl"

    meta = json.load(open(meta_path, 'r', encoding='utf-8'))
    kp_names = meta.get("keypoint_names") or []

    log_step(f"Running canonicalization")
    log_step(f"Using abs: {args.use_abs}")
    frames = canonicalize(preds_metric, output_path, mode=args.mode, use_abs=args.use_abs)

    if args.export_trc:
        log_step(f"Exporting in TRC")
        trc_outpath  = rtmw3d_dir / f"rtmw3d{'_abs' if args.use_abs else ''}_cannonical.trc"
        export_trc(frames, trc_outpath, kp_names=kp_names, source_key="keypoints_xyz_mm_canonical", rate_hz=args.rate)
        log_step(f"TRC exported to {trc_outpath}")

if __name__ == "__main__":
    main()
