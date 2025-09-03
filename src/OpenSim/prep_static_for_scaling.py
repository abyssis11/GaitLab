#!/usr/bin/env python3
"""
prep_static_for_scaling.py

Run your static trial through the usual pipeline (RTMW3D -> height scale -> marker enhancer),
collapse to a single static pose, and write a TRC for OpenSim ScaleTool.
By default, auto-rename Enhancer's 43 markers to the OpenSim (LaiUhlrich2022) '_study' names.
Optionally call OpenSim's ScaleTool (Python API) and visualize.

Layout assumptions (same as your other scripts):
  <outputs_root or manifest.output_dir>/<subject>/<session>/<camera>/<trial>/
    meta.json
    rtmw3d/
      preds.jsonl            (from rtmw3d)
      preds_metric.jsonl     (from rtmw3d_scale_from_height)
    enhancer/
      body_pred_mm_Tx35x3.npz
      arms_pred_mm_Tx8x3.npz
      enh_input_60hz.npz

Outputs here:
  <trial>/opensim/scale/static_enhanced_markers.trc
  <trial>/opensim/scale/static_enhanced_markers_renamed.trc
  <trial>/opensim/scale/scale_report.json
  (optional) scaled model files if --opensim-model and --run-opensim are provided.

Usage (end-to-end on a static trial):
  python prep_static_for_scaling.py \
    -m manifests/OpenCapDataset/subject2.yaml \
    -p config/paths.yaml \
    --trial static1 \
    --run-rtmw3d --run-scale --run-enhancer \
    --body-model ./models/marker_enhancer/body \
    --arms-model ./models/marker_enhancer/arm \
    --opensim-model models/opensim/LaiUhlrich2022.osim \
    --run-opensim --visualize

If you already produced enhancer NPZs for the static trial, just do:
  python prep_static_for_scaling.py -m ... -p ... --trial static1
"""

import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import sys
import shutil
import time

# ---- Small IO/log helpers ----

def log_step(msg): print(f"[STEP] {msg}")
def log_info(msg): print(f"[INFO] {msg}")
def log_warn(msg): print(f"[WARN] {msg}")
def log_done(msg): print(f"[DONE] {msg}")
def log_err(msg):  print(f"[ERROR] {msg}")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_jsonable(x):
    import numpy as _np
    from pathlib import Path as _Path
    if isinstance(x, dict): return {k: to_jsonable(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [to_jsonable(v) for v in x]
    if isinstance(x, _np.ndarray): return x.tolist()
    if isinstance(x, (_np.floating, _np.integer)): return x.item()
    if isinstance(x, _Path): return str(x)
    return x

# ---- Default mapping: Enhancer (augmenter) → OpenSim '..._study' marker names ----
# These names match LaiUhlrich2022_markers_augmenter.xml and the markers inside LaiUhlrich2022.osim.
def default_augmenter_to_osim_map() -> dict[str, str]:
    return {
        # Pelvis (ASIS/PSIS)
        "RASIS_augmenter": "r.ASIS_study",
        "LASIS_augmenter": "L.ASIS_study",
        "RPSIS_augmenter": "r.PSIS_study",
        "LPSIS_augmenter": "L.PSIS_study",

        # Knees (lat/med)
        "RKnee_augmenter":  "r_knee_study",
        "RMKnee_augmenter": "r_mknee_study",
        "LKnee_augmenter":  "L_knee_study",
        "LMKnee_augmenter": "L_mknee_study",

        # Ankles (lat/med)
        "RAnkle_augmenter":  "r_ankle_study",
        "RMAnkle_augmenter": "r_mankle_study",
        "LAnkle_augmenter":  "L_ankle_study",
        "LMAnkle_augmenter": "L_mankle_study",

        # Feet (toe, 5th met, calc)
        "RToe_augmenter":   "r_toe_study",
        "R5meta_augmenter": "r_5meta_study",
        "RCalc_augmenter":  "r_calc_study",
        "LToe_augmenter":   "L_toe_study",
        "L5meta_augmenter": "L_5meta_study",
        "LCalc_augmenter":  "L_calc_study",

        # Shoulders + C7
        "RShoulder_augmenter": "r_shoulder_study",
        "LShoulder_augmenter": "L_shoulder_study",
        "C7_augmenter":         "C7_study",

        # Elbows (lat/med)
        "RElbow_augmenter":  "r_lelbow_study",
        "RMElbow_augmenter": "r_melbow_study",
        "LElbow_augmenter":  "L_lelbow_study",
        "LMElbow_augmenter": "L_melbow_study",

        # Wrists (lat/med → radius/ulna)
        "RWrist_augmenter":  "r_lwrist_study",
        "RMWrist_augmenter": "r_mwrist_study",
        "LWrist_augmenter":  "L_lwrist_study",
        "LMWrist_augmenter": "L_mwrist_study",

        # Thigh clusters (3 per side)
        "RThigh1_augmenter": "r_thigh1_study",
        "RThigh2_augmenter": "r_thigh2_study",
        "RThigh3_augmenter": "r_thigh3_study",
        "LThigh1_augmenter": "L_thigh1_study",
        "LThigh2_augmenter": "L_thigh2_study",
        "LThigh3_augmenter": "L_thigh3_study",

        # Shank clusters (3 per side)
        "RSh1_augmenter": "r_sh1_study",
        "RSh2_augmenter": "r_sh2_study",
        "RSh3_augmenter": "r_sh3_study",
        "LSh1_augmenter": "L_sh1_study",
        "LSh2_augmenter": "L_sh2_study",
        "LSh3_augmenter": "L_sh3_study",

        # Hip joint centers (regressed)
        "RHJC_augmenter": "RHJC_study",
        "LHJC_augmenter": "LHJC_study",
    }

# ---- Paths: mirror your other scripts ----

def decide_trial_root(manifest: dict, trial: dict, outputs_root_cli: str | None):
    base = manifest.get('output_dir')
    if not base:
        outputs_root = outputs_root_cli or Path.cwd() / "outputs"
        subj = manifest.get('subject_id', 'subject')
        sess = manifest.get('session', 'Session')
        cam  = manifest.get('camera', 'Cam')
        base = Path(outputs_root) / subj / sess / cam
    return Path(base) / trial['id']

def load_manifest(manifest_path: str, paths_path: str) -> dict:
    try:
        from IO.load_manifest import load_manifest as _lm
        return _lm(manifest_path, paths_path)  # type: ignore
    except Exception as e:
        log_warn(f"Using fallback YAML loader (IO.load_manifest not importable): {e}")
        import yaml
        paths = yaml.safe_load(Path(paths_path).read_text(encoding='utf-8'))
        root = paths.get('root', '')
        text = Path(manifest_path).read_text(encoding='utf-8')
        text = text.replace('${paths.root}', str(root))
        return yaml.safe_load(text)

def find_trial(manifest: dict, trial_id: str):
    for subset, trials in manifest.get('trials', {}).items():
        for t in trials:
            if t.get('id') == trial_id:
                return subset, t
    return None, None

# ---- Load enhancer NPZ with names ----

def load_npz_with_names(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    keys = set(getattr(d, 'files', []))
    data_key = None
    for k in ("pred_mm","markers_mm","outputs","Y","pred","arr_0"):
        if k in keys:
            data_key = k; break
    if data_key is None:
        raise RuntimeError(f"No prediction array in {npz_path} (looked for pred_mm/markers_mm/outputs/Y/pred/arr_0).")

    X = np.asarray(d[data_key])  # (T, M, 3), in mm
    names = None
    for nk in ("names","marker_names","markers_names","joint_names"):
        if nk in keys:
            raw = d[nk]
            try:
                names = [str(x) for x in list(raw)]
            except Exception:
                names = [str(x) for x in np.asarray(raw).tolist()]
            break
    return X, (names or [])

# ---- Make a static TRC ----

def write_trc(path: Path, time_s, names, X_mm, rate_hz):
    """
    TRC writer (mm), as used in other parts of your code.
    time_s: (T,), X_mm: (T,K,3), names: list[str] length K.
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

# ---- Collapsing strategies for a static trial ----

def collapse_static_pose(X_tmk3: np.ndarray, mode: str = "median") -> np.ndarray:
    """
    X_tmk3: (T, M, 3) in mm. Returns (1, M, 3).
    mode:
      - 'median': robust per-marker median across time
      - 'stillest': pick the frame with minimal sum of per-marker velocities
    """
    T, M, _ = X_tmk3.shape
    if T == 1 or mode == "median":
        med = np.nanmedian(X_tmk3, axis=0, keepdims=True)
        return med

    # 'stillest' → minimal ||ΔX|| across frames
    V = np.linalg.norm(np.diff(X_tmk3, axis=0), axis=(1,2))  # (T-1,)
    idx = int(np.argmin(V))
    idx = min(idx+1, T-1)
    return X_tmk3[idx:idx+1, :, :]

# ---- Renaming of TRC columns ----

def apply_mapping(names: list[str], mapping: dict[str, str]) -> list[str]:
    """Apply name mapping, avoiding duplicate output names by suffixing _2, _3, ..."""
    out = []
    used = set()
    for n in names:
        new = mapping.get(n, n)
        if new in used:
            i = 2
            candidate = f"{new}_{i}"
            while candidate in used:
                i += 1
                candidate = f"{new}_{i}"
            new = candidate
        out.append(new)
        used.add(new)
    return out

def load_mapping_json(mapping_json: Path | None) -> dict[str, str]:
    if not mapping_json:
        return {}
    try:
        mp = json.loads(mapping_json.read_text(encoding="utf-8"))
        if not isinstance(mp, dict):
            log_warn(f"{mapping_json} did not contain a JSON object; ignoring.")
            return {}
        # force str->str
        clean = {str(k): str(v) for k, v in mp.items()}
        return clean
    except Exception as e:
        log_warn(f"Failed to load rename map '{mapping_json}': {e}")
        return {}

# ---- Subprocess wrappers to run upstream pipeline pieces ----

def run_cmd(cmd: list[str]):
    log_info("→ " + " ".join(cmd))
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise SystemExit(f"Command failed ({ret.returncode}): {' '.join(cmd)}")

def run_rtmw3d_pose(manifest, paths, trial_id, device="cuda:0",
                    config=None, checkpoint=None, video_field="video_sync",
                    metainfo_from_file=None, outputs_root=None):
    exe = shutil.which("python") or "python"
    script = Path(__file__).parent / "rtmw3d_pose_estimation.py"
    if not script.exists():
        alt = Path(__file__).parent / "rtmw3d.py"
        script = alt if alt.exists() else script
    cmd = [exe, str(script),
           "-m", str(manifest),
           "-p", str(paths),
           "--subset", "all",
           "--trials", trial_id,
           "--video-field", video_field,
           "--device", device]
    if outputs_root: cmd += ["--outputs-root", str(outputs_root)]
    if config:       cmd += ["--config", str(config)]
    if checkpoint:   cmd += ["--checkpoint", str(checkpoint)]
    if metainfo_from_file: cmd += ["--metainfo-from-file", str(metainfo_from_file)]
    run_cmd(cmd)

def run_height_scale(manifest, paths, trial_id, outputs_root=None,
                     height_mm=None, head="nose",
                     foot="left_heel,right_heel,left_big_toe,right_big_toe,left_ankle,right_ankle"):
    exe = shutil.which("python") or "python"
    script = Path(__file__).parent / "rtmw3d_scale_from_height.py"
    cmd = [exe, str(script),
           "-m", str(manifest),
           "-p", str(paths),
           "--trial", trial_id]
    if outputs_root: cmd += ["--outputs-root", str(outputs_root)]
    if height_mm:    cmd += ["--height-mm", f"{float(height_mm):.3f}"]
    cmd += ["--head-joints", head, "--foot-joints", foot]
    run_cmd(cmd)

def run_marker_enhancer(manifest, paths, trial_id, outputs_root=None,
                        body_model=None, arms_model=None, output_rate=60.0):
    exe = shutil.which("python") or "python"
    script = Path(__file__).parent / "marker_enhancer.py"
    cmd = [exe, str(script),
           "-m", str(manifest),
           "-p", str(paths),
           "--trial", trial_id,
           "--output-rate", str(float(output_rate))]
    if outputs_root: cmd += ["--outputs-root", str(outputs_root)]
    if body_model:   cmd += ["--body-model", str(body_model)]
    if arms_model:   cmd += ["--arms-model", str(arms_model)]
    run_cmd(cmd)

# ---- OpenSim Scale/Visualize (optional) ----

def try_opensim_scale(model_path: Path, trc_path: Path, out_dir: Path,
                      mass_kg: float | None, visualize: bool):
    try:
        import opensim as osim
    except Exception as e:
        log_warn(f"OpenSim Python API not available: {e}")
        return False

    ensure_dir(out_dir)
    scale = osim.ScaleTool()
    scale.setName("static_scale")
    if mass_kg and mass_kg > 0:
        scale.setSubjectMass(float(mass_kg))

    # Model maker
    scale.getGenericModelMaker().setModelFileName(str(model_path))

    # ModelScaler: scale by marker distances in the static TRC
    ms = scale.getModelScaler()
    ms.setApply(True)
    ms.setMarkerFileName(str(trc_path))
    arr = osim.ArrayDouble()
    arr.append(0.0); arr.append(0.0)  # single-frame
    ms.setTimeRange(arr)
    ms.setOutputModelFileName(str(out_dir / "scaled_model.osim"))

    # MarkerPlacer: place model markers onto data
    mp = scale.getMarkerPlacer()
    mp.setApply(True)
    mp.setStaticPoseFileName(str(trc_path))
    mp.setTimeRange(arr)
    mp.setOutputModelFileName(str(out_dir / "scaled_marker_placed.osim"))
    mp.setOutputMarkerFileName(str(out_dir / "placed_markers.trc"))

    # Execute
    log_step("Running OpenSim ScaleTool")
    scale.run()

    if visualize:
        log_step("Launching OpenSim visualizer (scaled_marker_placed.osim)")
        model = osim.Model(str(out_dir / "scaled_marker_placed.osim"))
        model.setUseVisualizer(True)
        state = model.initSystem()
        time.sleep(0.5)
        print("[INFO] Close the visualizer window or Ctrl-C to quit.")
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass

    return True

# ---- Main ----

def main():
    ap = argparse.ArgumentParser(description="Create a static TRC for OpenSim scaling from enhancer markers (and optionally run OpenSim scale).")
    ap.add_argument("-m", "--manifest", required=True)
    ap.add_argument("-p", "--paths",    required=True)
    ap.add_argument("--trial", required=True, help="Static trial ID in manifest (e.g., static1)")
    ap.add_argument("--outputs-root", default=None)

    # Upstream pipeline flags
    ap.add_argument("--run-rtmw3d", action="store_true", help="Run RTMW3D on the static trial")
    ap.add_argument("--run-scale",  action="store_true", help="Run height-based scaling on RTMW3D outputs")
    ap.add_argument("--run-enhancer", action="store_true", help="Run marker-enhancer on the static trial")
    ap.add_argument("--body-model", default=None, help="Enhancer body LSTM dir")
    ap.add_argument("--arms-model", default=None, help="Enhancer arms LSTM dir")

    # Collapsing / TRC
    ap.add_argument("--collapse", choices=["median","stillest"], default="median", help="How to collapse frames into one static pose")
    ap.add_argument("--fps-trc", type=float, default=60.0, help="DataRate for the TRC (default 60)")
    ap.add_argument("--rename-map", default=None, help="JSON mapping {enhancerName: modelMarkerName} to rename TRC columns")
    ap.add_argument("--out-name", default="static_enhanced_markers", help="Base name for output TRC(s)")
    ap.add_argument("--no-auto-map", action="store_true",
                    help="Disable built-in mapping from Enhancer to OpenSim '_study' names")

    # Optional OpenSim
    ap.add_argument("--opensim-model", default=None, help="Path to base OpenSim model (.osim)")
    ap.add_argument("--run-opensim", action="store_true", help="Run OpenSim ScaleTool")
    ap.add_argument("--visualize", action="store_true", help="Visualize the scaled model (OpenSim visualizer)")

    # Optional RTMW3D config bits (only used if --run-rtmw3d)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rtmw3d-config", default=None)
    ap.add_argument("--rtmw3d-checkpoint", default=None)
    ap.add_argument("--metainfo-from-file", default=None)
    ap.add_argument("--video-field", choices=["video_sync","video_raw"], default="video_sync")

    args = ap.parse_args()

    # Load manifest & find static trial
    manifest = load_manifest(args.manifest, args.paths)
    subset, trial = find_trial(manifest, args.trial)
    if trial is None:
        raise SystemExit(f"Trial '{args.trial}' not found in manifest.")

    trial_root = decide_trial_root(manifest, trial, args.outputs_root)
    enh_dir    = trial_root / "enhancer"
    os_dir     = trial_root / "opensim" / "scale"
    ensure_dir(os_dir)

    # Optional upstream runs
    if args.run_rtmw3d:
        log_step("Running RTMW3D on static trial")
        run_rtmw3d_pose(args.manifest, args.paths, args.trial,
                        device=args.device,
                        config=args.rtmw3d_config,
                        checkpoint=args.rtmw3d_checkpoint,
                        video_field=args.video_field,
                        metainfo_from_file=args.metainfo_from_file,
                        outputs_root=args.outputs_root)

    if args.run_scale:
        log_step("Scaling RTMW3D outputs to metric (mm) via subject height")
        run_height_scale(args.manifest, args.paths, args.trial, outputs_root=args.outputs_root)

    if args.run_enhancer:
        log_step("Running marker enhancer on static trial")
        run_marker_enhancer(args.manifest, args.paths, args.trial,
                            outputs_root=args.outputs_root,
                            body_model=args.body_model,
                            arms_model=args.arms_model,
                            output_rate=60.0)

    # Load enhancer outputs
    body_npz = enh_dir / "body_pred_mm_Tx35x3.npz"
    arms_npz = enh_dir / "arms_pred_mm_Tx8x3.npz"
    if not body_npz.exists():
        raise SystemExit(f"Missing enhancer output: {body_npz}")
    Xb, names_b = load_npz_with_names(body_npz)
    Xa, names_a = (None, [])
    if arms_npz.exists():
        Xa, names_a = load_npz_with_names(arms_npz)

    # Combine markers
    if Xa is not None and Xa.shape[0] != Xb.shape[0]:
        raise SystemExit("Body and arms enhancer outputs have different T.")
    X_all = Xb if Xa is None else np.concatenate([Xb, Xa], axis=1)
    names_all = (names_b or []) + (names_a or [])
    if len(names_all) != X_all.shape[1]:
        log_warn("Names length != markers; synthesizing generic names for missing.")
        if not names_all:
            names_all = [f"M{i+1}" for i in range(X_all.shape[1])]

    log_info(f"Static enhancer series loaded: T={X_all.shape[0]} markers={X_all.shape[1]}")

    # Collapse to one static pose
    X1 = collapse_static_pose(X_all, mode=args.collapse)  # (1, M, 3)
    t1 = np.array([0.0], dtype=float)
    base = args.out_name

    # Write native-name TRC
    trc_native = os_dir / f"{base}.trc"
    write_trc(trc_native, t1, names_all, X1, rate_hz=float(args.fps_trc))
    log_done(f"Wrote TRC: {trc_native}")

    # Build mapping (priority: user JSON > built-in default unless disabled)
    mapping = {}
    user_map = load_mapping_json(Path(args.rename_map)) if args.rename_map else {}
    if user_map:
        mapping = user_map
        log_info(f"Using user mapping with {len(mapping)} entries")
    elif not args.no_auto_map:
        mapping = default_augmenter_to_osim_map()
        log_info(f"Using built-in Enhancer→OpenSim mapping ({len(mapping)} entries)")
    else:
        log_info("No renaming applied (native Enhancer names kept).")

    # Apply mapping and write renamed TRC
    if mapping:
        new_names = apply_mapping(names_all, mapping)
        trc_renamed = os_dir / f"{base}_renamed.trc"
        write_trc(trc_renamed, t1, new_names, X1, rate_hz=float(args.fps_trc))
        log_done(f"Wrote TRC (renamed for model): {trc_renamed}")
    else:
        trc_renamed = trc_native

    # Small report
    meta_json = (trial_root / "meta.json")
    subj_mass = None
    if meta_json.exists():
        try:
            meta = json.loads(meta_json.read_text(encoding='utf-8'))
            subj_mass = float(meta.get("subject", {}).get("mass_kg", "nan"))
        except Exception:
            pass

    report = {
        "trial_id": args.trial,
        "markers_written_native": names_all,
        "markers_written_renamed": new_names if mapping else None,
        "collapsed_mode": args.collapse,
        "trc_native": str(trc_native),
        "trc_renamed": str(trc_renamed),
        "subject_mass_kg": subj_mass,
        "used_mapping": ("user_json" if user_map else ("built_in_default" if mapping else "none")),
    }
    (os_dir / "scale_report.json").write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
    log_done(f"Report: {os_dir/'scale_report.json'}")

    # Optional: run OpenSim scaling + visualize
    if args.run_opensim:
        if not args.opensim_model:
            log_err("--run-opensim requires --opensim-model")
        else:
            ok = try_opensim_scale(Path(args.opensim_model), trc_renamed, os_dir,
                                   mass_kg=subj_mass, visualize=args.visualize)
            if ok:
                log_done("OpenSim scaling completed.")
            else:
                log_warn("OpenSim scaling did not run (see warnings above).")

if __name__ == "__main__":
    main()
