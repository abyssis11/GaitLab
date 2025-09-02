#!/usr/bin/env python3
# run_marker_enhancer.py
"""
Prepare OCAP20 3D keypoints for the Marker-Enhancer and (optionally) run it.

What this script does
---------------------
1) Loads your manifest via IO.load_manifest.load_manifest
2) Locates:
   - <trial>/rtmw3d/preds_metric.jsonl       (REQUIRED: must contain keypoints_xyz_mm)
   - <trial>/meta.json                       (subject height_m, mass_kg, keypoint_names)
3) Extracts the OCAP20 joints from RTMW3D, resamples to 60 Hz, and builds the
   enhancer input features exactly as in the Enhancer/OpenCap papers:
   - Root-centered by mid_hip
   - Height-normalized (divide by subject height)
   - Body inputs: 15 joints * 3 + [height_m, mass_kg] = 47
   - Arms inputs:  7 joints * 3 + [height_m, mass_kg] = 23
4) Saves a prep cache:
   - <trial>/enhancer/enh_input_60hz.npz
     (t_60, ocap20_names, ocap20_centered_mm, midhip_mm, X_body_47, X_arms_23,
      subject_height_m, subject_mass_kg)
   - and CSVs with feature rows (for convenience)
5) Optional: runs Marker-Enhancer LSTM models and writes predictions:
   - <trial>/enhancer/body_pred_mm_Tx35x3.npz
   - <trial>/enhancer/arms_pred_mm_Tx8x3.npz
   (Outputs are de-normalized to mm and re-rooted at mid_hip per-frame.)
   **Now includes names** inside each NPZ under key 'names'.

Notes
-----
- We DO NOT auto-scale. If no 'keypoints_xyz_mm' is found in preds_metric.jsonl,
  we log and exit.
- OCAP20 list matches OpenCap’s 20 “IK” joints we’ve been using.
- LSTM models must include model.json, weights.h5, mean.npy, std.npy, metadata.json.

Example (prep-only)
-------------------
python run_marker_enhancer.py \
  -m manifests/OpenCapDataset/subject2.yaml \
  -p config/paths.yaml \
  --trial walking1

Example (run Keras models)
--------------------------
python run_marker_enhancer.py \
  -m manifests/OpenCapDataset/subject2.yaml \
  -p config/paths.yaml \
  --trial walking1 \
  --body-model ./models/marker_enhancer/body \
  --arms-model ./models/marker_enhancer/arm \
  --write-trc
"""

import argparse
import json
from pathlib import Path
import numpy as np
from IO.load_manifest import load_manifest

# ---------- Logging ----------
def log_step(msg): print(f"[STEP] {msg}")
def log_info(msg): print(f"[INFO] {msg}")
def log_warn(msg): print(f"[WARN] {msg}")
def log_err (msg): print(f"[ERROR] {msg}")
def log_done(msg): print(f"[DONE] {msg}")

# ---------- IO helpers ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def jsonl_iter(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                yield json.loads(ln)

# ---------- TRC ----------
def write_trc(path: Path, time_s, names, X_mm, rate_hz):
    """
    time_s: (T,), X_mm: (T,K,3) mm, names: list of length K
    """
    T, K, _ = X_mm.shape
    header = []
    header.append("PathFileType\t4\t(X/Y/Z)\t{}".format(path))
    header.append("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames")
    header.append(f"{rate_hz:.1f}\t{rate_hz:.1f}\t{T}\t{K}\tmm\t{rate_hz:.1f}\t1\t{T}")
    header_names = ["Frame#", "Time"] + [n for n in names]
    header.append("\t".join(header_names))
    comps = []
    for i in range(K):
        comps += [f"X{i+1}", f"Y{i+1}", f"Z{i+1}"]
    header.append("\t" + "\t".join(comps))

    lines = []
    for i in range(T):
        row = [str(i+1), f"{time_s[i]:.8f}"]
        for k in range(K):
            x, y, z = X_mm[i, k]
            row += [f"{x:.5f}", f"{y:.5f}", f"{z:.5f}"]
        lines.append("\t".join(row))

    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        f.write("\n".join(lines) + "\n")

# ---------- OCAP20 ----------
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

OCAP20_REQUIRED_MIN = [
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

def compose_ocap20_from_frame(kp_names, kps_xyz_mm, *, warn_once_state):
    """
    kp_names: list[str] (len K_total)
    kps_xyz_mm: (K_total,3)
    Returns dict name->(3,), with synthesized mid_hip/neck if needed.
    """
    name2i = {n: i for i, n in enumerate(kp_names)}
    out = {}

    def has(n): return n in name2i
    def get(n):
        return kps_xyz_mm[name2i[n]] if has(n) else None

    # Upper limbs
    for n in ["left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist"]:
        v = get(n)
        if v is not None:
            out[n] = v
    # Lower limbs
    for n in ["left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]:
        v = get(n)
        if v is not None:
            out[n] = v
    # Feet
    for n in ["left_heel","right_heel","left_small_toe","right_small_toe","left_big_toe","right_big_toe"]:
        v = get(n)
        if v is not None:
            out[n] = v

    # mid_hip
    if "mid_hip" in name2i:
        out["mid_hip"] = get("mid_hip")
    else:
        if has("left_hip") and has("right_hip"):
            out["mid_hip"] = 0.5 * (get("left_hip") + get("right_hip"))
            if not warn_once_state.get("mid_hip"):
                log_info("Synthesizing 'mid_hip' = 0.5*(left_hip+right_hip)")
                warn_once_state["mid_hip"] = True
        else:
            if not warn_once_state.get("no_mid_hip"):
                log_warn("Cannot synthesize 'mid_hip' (need both left_hip and right_hip)")
                warn_once_state["no_mid_hip"] = True

    # neck
    if "neck" in name2i:
        out["neck"] = get("neck")
    else:
        if has("left_shoulder") and has("right_shoulder"):
            out["neck"] = 0.5 * (get("left_shoulder") + get("right_shoulder"))
            if not warn_once_state.get("neck"):
                log_info("Synthesizing 'neck' = 0.5*(left_shoulder+right_shoulder)")
                warn_once_state["neck"] = True
        else:
            if not warn_once_state.get("no_neck"):
                log_warn("Cannot synthesize 'neck' (need both left_shoulder and right_shoulder)")
                warn_once_state["no_neck"] = True

    return out

# ---------- Resampling ----------
def resample_series(times_s, X, t_grid):
    """
    X: (T,K,3) -> (Tg,K,3), linear per-dim.
    NaNs are propagated if outside support.
    """
    T, K, _ = X.shape
    Y = np.full((len(t_grid), K, 3), np.nan, dtype=float)
    for k in range(K):
        for d in range(3):
            Y[:, k, d] = np.interp(t_grid, times_s, X[:, k, d], left=np.nan, right=np.nan)
    return Y

# ---------- Enhancer inputs (47 & 23) ----------
JOINTS_BODY15 = [
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip", "mid_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_small_toe", "right_small_toe",
    "left_big_toe", "right_big_toe",
]

JOINTS_ARMS7 = [
    "neck",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]

def _name_to_index(names):
    return {n: i for i, n in enumerate(names)}

def _center_by_midhip(X, names):
    n2i = _name_to_index(names)
    mi = n2i.get("mid_hip", None)
    if mi is None:
        raise RuntimeError("'mid_hip' required for root-centering but missing.")
    root = X[:, mi:mi+1, :]  # (T,1,3)
    return X - root, root[:, 0, :]  # centered, midhip_mm (T,3)

def _interp_nan_timewise(Y):
    """Linear interpolate NaNs along time for each joint/axis; fallback zeros."""
    T, K, D = Y.shape
    for k in range(K):
        for d in range(D):
            y = Y[:, k, d]
            if np.isnan(y).any():
                t = np.arange(T)
                mask = ~np.isnan(y)
                if mask.sum() >= 2:
                    y[~mask] = np.interp(t[~mask], t[mask], y[mask])
                else:
                    y[~mask] = 0.0
            Y[:, k, d] = y
    return Y

def build_enhancer_inputs(ocap20_xyz_mm, ocap20_names, height_m, mass_kg):
    """
    ocap20_xyz_mm: (T, K20, 3) in mm (resampled).
    Returns:
      X_body: (T, 47)  = flatten(JOINTS_BODY15)*3 + [h, w]
      X_arms: (T, 23)  = flatten(JOINTS_ARMS7)*3  + [h, w]
      X_centered_mm: (T, K20, 3) pelvis-centered
      midhip_mm: (T, 3)
    """
    n2i = _name_to_index(ocap20_names)

    Xc, midhip_mm = _center_by_midhip(ocap20_xyz_mm, ocap20_names)
    Xc = _interp_nan_timewise(Xc)

    height_mm = float(height_m) * 1000.0
    Xn = Xc / (height_mm + 1e-8)

    def stack_named(names_list):
        idxs = [n2i[n] for n in names_list if n in n2i]
        if len(idxs) != len(names_list):
            missing = [n for n in names_list if n not in n2i]
            raise RuntimeError(f"Required joints missing for enhancer inputs: {missing}")
        return Xn[:, idxs, :].reshape((Xn.shape[0], -1))  # (T, len*3)

    body_flat = stack_named(JOINTS_BODY15)  # (T, 45)
    arms_flat = stack_named(JOINTS_ARMS7)   # (T, 21)

    h_col = np.full((Xn.shape[0], 1), float(height_m), dtype=float)
    w_col = np.full((Xn.shape[0], 1), float(mass_kg),  dtype=float)

    X_body = np.concatenate([body_flat, h_col, w_col], axis=1)  # (T, 47)
    X_arms = np.concatenate([arms_flat, h_col, w_col], axis=1)  # (T, 23)
    return X_body, X_arms, Xc, midhip_mm

# ---------- CSV writers ----------
def save_csv(path: Path, arr: np.ndarray, header: list[str] | None):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write(",".join(header) + "\n")
        np.savetxt(f, arr, delimiter=",", fmt="%.8f")

def feature_headers(prefixes, height_weight=True):
    cols = []
    for p in prefixes:
        cols += [f"{p}_x", f"{p}_y", f"{p}_z"]
    if height_weight:
        cols += ["height_m", "mass_kg"]
    return cols

# ---------- Marker-Enhancer fixed output orders ----------
# Defaults taken from the repo helper functions (getMarkersAugmenter_*).
BODY35_DEFAULT = [
    'RASIS_augmenter','LASIS_augmenter','RPSIS_augmenter','LPSIS_augmenter',
    'RKnee_augmenter','RMKnee_augmenter','RAnkle_augmenter','RMAnkle_augmenter',
    'RToe_augmenter','R5meta_augmenter','RCalc_augmenter',
    'LKnee_augmenter','LMKnee_augmenter','LAnkle_augmenter','LMAnkle_augmenter',
    'LToe_augmenter','LCalc_augmenter','L5meta_augmenter',
    'RShoulder_augmenter','LShoulder_augmenter','C7_augmenter',
    'RThigh1_augmenter','RThigh2_augmenter','RThigh3_augmenter',
    'LThigh1_augmenter','LThigh2_augmenter','LThigh3_augmenter',
    'RSh1_augmenter','RSh2_augmenter','RSh3_augmenter',
    'LSh1_augmenter','LSh2_augmenter','LSh3_augmenter',
    'RHJC_augmenter','LHJC_augmenter',
]
ARMS8_DEFAULT = [
    'RElbow_augmenter','RMElbow_augmenter','RWrist_augmenter','RMWrist_augmenter',
    'LElbow_augmenter','LMElbow_augmenter','LWrist_augmenter','LMWrist_augmenter',
]

def _pick_output_names(meta, default_names, expected_len: int):
    """Prefer output marker names from model metadata.json if present."""
    names = None
    if isinstance(meta, dict):
        for k in ("response_markers", "output_markers", "marker_names", "names"):
            v = meta.get(k)
            if isinstance(v, (list, tuple)) and len(v) == expected_len:
                names = [str(x) for x in v]
                break
    return names or default_names

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Prepare OCAP20 from RTMW3D (metric) and run Marker-Enhancer.")
    ap.add_argument("-m", "--manifest", required=True)
    ap.add_argument("-p", "--paths", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--outputs-root", default=None)

    # LSTM model dirs (optional)
    ap.add_argument("--body-model", default=None, help="Directory containing model.json, weights.h5, mean.npy, std.npy, metadata.json")
    ap.add_argument("--arms-model", default=None, help="Directory containing model.json, weights.h5, mean.npy, std.npy, metadata.json")

    # Rate / export
    ap.add_argument("--output-rate", type=float, default=60.0, help="Resample RTMW3D series to this Hz for enhancer (default 60)")
    ap.add_argument("--write-trc", action="store_true", help="Write OCAP20 TRC with resampled (pelvis-centered) series")

    args = ap.parse_args()

    # Load manifest
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
    eval_dir = trial_root / "rtmw3d_eval"
    enh_dir  = trial_root / "enhancer"
    ensure_dir(eval_dir); ensure_dir(enh_dir)

    preds_path = rtmw3d_dir / "preds_metric.jsonl"
    meta_path  = trial_root / "meta.json"

    log_info(f"Trial root : {trial_root}")
    log_info(f"Preds path : {preds_path}")
    log_info(f"meta.json  : {meta_path}")

    if not preds_path.exists():
        raise SystemExit("[ERROR] preds_metric.jsonl not found. Run the height scaler first.")
    if not meta_path.exists():
        raise SystemExit("[ERROR] meta.json not found (need keypoint_names, subject info).")

    # Subject info
    log_step("Reading meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    subj = meta.get("subject") or {}
    height_m = subj.get("height_m", None)
    mass_kg  = subj.get("mass_kg", None)
    if not (isinstance(height_m, (int, float)) and height_m > 0):
        raise SystemExit("[ERROR] subject.height_m missing/invalid in meta.json")
    if not (isinstance(mass_kg, (int, float)) and mass_kg > 0):
        raise SystemExit("[ERROR] subject.mass_kg missing/invalid in meta.json")
    log_info(f"Subject info: height={height_m} m ({height_m*1000:.1f} mm), mass={mass_kg} kg")

    kp_names = meta.get("keypoint_names") or []
    if not kp_names:
        raise SystemExit("[ERROR] meta.keypoint_names empty")

    # Read preds (metric only)
    log_step("Reading RTMW3D metric predictions (keypoints_xyz_mm only; no auto-scale)")
    times = []
    frames = []
    warn_once = {}
    for o in jsonl_iter(preds_path):
        persons = o.get("persons") or []
        if not persons:
            continue
        p = persons[0]  # assume single-person
        kmm = p.get("keypoints_xyz_mm", None)
        if kmm is None:
            log_err("Found frame without 'keypoints_xyz_mm' (aborting; no auto-scale).")
            raise SystemExit(1)
        kmm = np.asarray(kmm, dtype=float)
        d = compose_ocap20_from_frame(kp_names, kmm, warn_once_state=warn_once)

        if not all(j in d for j in OCAP20_REQUIRED_MIN):
            continue

        row = np.full((len(OCAP20), 3), np.nan, dtype=float)
        for j, name in enumerate(OCAP20):
            if name in d:
                row[j] = d[name]
        frames.append(row)
        times.append(float(o.get("time_sec", 0.0)))

    if not frames:
        raise SystemExit("[ERROR] No usable frames with metric keypoints and minimum joints.")

    X = np.asarray(frames, dtype=float)   # (T,20,3)
    t = np.asarray(times, dtype=float)
    log_info(f"Frames read: {len(t)} from keypoints_xyz_mm")
    K_avail = int(np.isfinite(X).all(axis=2).any(axis=0).sum())
    log_info(f"OCAP20 available K={K_avail}")

    # Resample to enhancer rate
    rate = float(args.output_rate)
    log_step(f"Resampling to {rate:.1f} Hz")
    t0, t1 = t.min(), t.max()
    t_grid = np.arange(t0, t1 + 1e-9, 1.0 / rate)
    Xr = resample_series(t, X, t_grid)
    log_info(f"Output time range: {t_grid.min():.3f}..{t_grid.max():.3f} s (T={len(t_grid)})")

    # Build enhancer inputs (center->normalize->flatten + [h,w])
    log_step("Building enhancer input features (body=47, arms=23)")
    X_body, X_arms, X_centered_mm, midhip_mm = build_enhancer_inputs(
        ocap20_xyz_mm=Xr,
        ocap20_names=OCAP20,
        height_m=float(height_m),
        mass_kg=float(mass_kg),
    )
    log_info(f"Body inputs shape  : {X_body.shape}  (T x 47)")
    log_info(f"Arms inputs shape  : {X_arms.shape}  (T x 23)")

    # Save prep cache + CSVs
    log_step("Saving prep cache (.npz) and CSVs")
    np.savez_compressed(
        enh_dir / "enh_input_60hz.npz",
        t_60=t_grid,
        ocap20_names=np.array(OCAP20, dtype=object),
        ocap20_centered_mm=X_centered_mm,
        midhip_mm=midhip_mm,
        X_body_47=X_body,
        X_arms_23=X_arms,
        subject_height_m=float(height_m),
        subject_mass_kg=float(mass_kg),
    )

    # feature CSV headers
    body_hdr = feature_headers(JOINTS_BODY15, height_weight=True)
    arms_hdr = feature_headers(JOINTS_ARMS7,  height_weight=True)
    save_csv(enh_dir / "body_features_47.csv", X_body, body_hdr)
    save_csv(enh_dir / "arms_features_23.csv", X_arms, arms_hdr)
    log_done(f"Saved: {enh_dir/'enh_input_60hz.npz'}")
    log_done(f"Saved: {enh_dir/'body_features_47.csv'}")
    log_done(f"Saved: {enh_dir/'arms_features_23.csv'}")

    # Optional: write TRC for OCAP20 (centered)
    if args.write_trc:
        trc_out = eval_dir / f"ocap20_resampled_{int(round(rate))}hz.trc"
        write_trc(trc_out, t_grid, OCAP20, X_centered_mm, rate)
        log_done(f"Wrote TRC : {trc_out}")

    # ---------- Optional: Run Marker-Enhancer LSTMs ----------
    if args.body_model or args.arms_model:
        try:
            import json as _json
            from tensorflow.keras.models import model_from_json
        except Exception as e:
            log_warn(f"Keras/TensorFlow not available: {e}")
            log_warn("Skipping LSTM inference; prep is complete.")
            return

        def load_lstm_bundle(model_dir: Path):
            req = ["model.json","weights.h5","mean.npy","std.npy","metadata.json"]
            missing = [r for r in req if not (model_dir / r).exists()]
            if missing:
                raise FileNotFoundError(f"Missing in {model_dir}: {missing}")
            arch = _json.loads((model_dir/"model.json").read_text())
            mdl  = model_from_json(_json.dumps(arch))
            mdl.load_weights(str(model_dir/"weights.h5"))
            mean = np.load(model_dir/"mean.npy")
            std  = np.load(model_dir/"std.npy")
            meta = _json.loads((model_dir/"metadata.json").read_text())
            return mdl, mean, std, meta

        # BODY
        if args.body_model:
            log_step("Running body model")
            body_dir = Path(args.body_model)
            body_mdl, body_mean, body_std, body_meta = load_lstm_bundle(body_dir)
            if body_mean.shape != (47,) or body_std.shape != (47,):
                raise RuntimeError(f"Body mean/std must be (47,), got {body_mean.shape}/{body_std.shape}")
            Xb = (X_body - body_mean.reshape(1, -1)) / (body_std.reshape(1, -1) + 1e-8)
            Yb = body_mdl.predict(Xb[None, ...], verbose=0)[0]  # (T, 105)
            if Yb.shape[-1] != 105:
                raise RuntimeError(f"Body model output last dim expected 105, got {Yb.shape[-1]}")
            Yb = Yb.reshape((-1, 35, 3))                        # (T, 35, 3) normalized (height units)
            # de-normalize and add root back to mm
            Yb_mm = Yb * (float(height_m) * 1000.0) + midhip_mm[:, None, :]
            body_names = _pick_output_names(body_meta, BODY35_DEFAULT, expected_len=35)
            np.savez_compressed(
                enh_dir / "body_pred_mm_Tx35x3.npz",
                t_60=t_grid,
                pred_mm=Yb_mm,
                names=np.array(body_names, dtype=object),
            )
            log_done("Body model outputs saved (body_pred_mm_Tx35x3.npz)")

        # ARMS
        if args.arms_model:
            log_step("Running arms model")
            arms_dir = Path(args.arms_model)
            arms_mdl, arms_mean, arms_std, arms_meta = load_lstm_bundle(arms_dir)
            if arms_mean.shape != (23,) or arms_std.shape != (23,):
                raise RuntimeError(f"Arms mean/std must be (23,), got {arms_mean.shape}/{arms_std.shape}")
            Xa = (X_arms - arms_mean.reshape(1, -1)) / (arms_std.reshape(1, -1) + 1e-8)
            Ya = arms_mdl.predict(Xa[None, ...], verbose=0)[0]  # (T, 24)
            if Ya.shape[-1] != 24:
                raise RuntimeError(f"Arms model output last dim expected 24, got {Ya.shape[-1]}")
            Ya = Ya.reshape((-1, 8, 3))                         # (T, 8, 3) normalized
            Ya_mm = Ya * (float(height_m) * 1000.0) + midhip_mm[:, None, :]
            arms_names = _pick_output_names(arms_meta, ARMS8_DEFAULT, expected_len=8)
            np.savez_compressed(
                enh_dir / "arms_pred_mm_Tx8x3.npz",
                t_60=t_grid,
                pred_mm=Ya_mm,
                names=np.array(arms_names, dtype=object),
            )
            log_done("Arms model outputs saved (arms_pred_mm_Tx8x3.npz)")

    log_done("Marker-enhancer prep complete.")

if __name__ == "__main__":
    main()
