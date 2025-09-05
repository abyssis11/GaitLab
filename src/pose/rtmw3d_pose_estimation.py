#!/usr/bin/env python3
"""
Run RTMW3D on videos listed in a manifest (no detector), and write
preds.jsonl(.gz) + meta.json with fully populated keypoint names/skeleton.

Key improvements vs your original:
- Robustly populates model.dataset_meta using the model config and/or a dataset
  metainfo file (so keypoint_names won't be null).
- Explicit full-frame bbox to inference_topdown (no detector).
- Faster + safer: torch.inference_mode(), optional AMP (CUDA), frame stride.
- Accurate timestamps using CAP_PROP_POS_MSEC fallback to frame_idx / fps.
- Optional gzip output for long videos.
- meta.json includes extra provenance: root_idx, sigmas, config/ckpt names.

Usage
-----
python src/pose/rtmw3d_pose_estimation.py   
    -m manifests/OpenCapDataset/subject2-2.yaml   \
    -p config/paths.yaml   \
    --trials walking1   \
    --video-field video_sync    \
    --metainfo-from-file /home/denik/projects/GaitLab/external/datasets_config/h3wb.py   \
    --debug-metainfo 
"""
import os
import cv2
import json
import gzip
import argparse
import warnings
from contextlib import nullcontext
from pathlib import Path
import yaml, re
import numpy as np
import torch

from mmengine.config import Config, ConfigDict
from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

# your loader in a separate file
from IO.load_manifest import load_manifest


# ----------------------------
# Helpers
# ----------------------------
def _norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(k).lower())

def _dig(d, keys_norm):
    """Find first value in possibly nested dict `d` whose normalized key is in keys_norm.
    Looks also under common nesting like 'subject', 'participant', 'patient'."""
    if not isinstance(d, dict):
        return None
    for k, v in d.items():
        if _norm_key(k) in keys_norm:
            return v
    for nest in ("subject", "participant", "patient", "person"):
        v = d.get(nest)
        if isinstance(v, dict):
            got = _dig(v, keys_norm)
            if got is not None:
                return got
    return None

def _parse_number_with_units(x):
    if x is None:
        return None, None
    if isinstance(x, (int, float)):
        return float(x), None
    if isinstance(x, str):
        s = x.strip().lower()
        m = re.match(r"([-+]?\d*\.?\d+)\s*([a-z%]*)", s)
        if m:
            val = float(m.group(1))
            unit = m.group(2) or None
            return val, unit
    return None, None

def extract_subject_from_session_metadata(session_meta: dict):
    """Return dict with keys: sex, height_m, mass_kg if discoverable.
    Heuristics handle cm/mm/g units and common key aliases."""
    out = {"sex": None, "height_m": None, "mass_kg": None}
    if not isinstance(session_meta, dict):
        return out

    # sex / gender
    sex_raw = _dig(session_meta, {"sex", "gender", "biologicalsex"})
    if sex_raw is not None:
        s = str(sex_raw).strip().lower()
        if s in {"m", "male", "man"}:
            out["sex"] = "male"
        elif s in {"f", "female", "woman"}:
            out["sex"] = "female"
        else:
            out["sex"] = str(sex_raw)

    # height -> meters (accept m / cm / mm or bare numbers)
    h_raw = _dig(session_meta, {"heightm", "height", "stature", "bodyheight", "bodyheightm", "staturem", "bodyheightmeter"})
    hv, hu = _parse_number_with_units(h_raw)
    if hv is not None:
        height_m = None
        if hu in {"m", None}:
            if hu == "m" or (0.5 <= hv <= 2.6):        # looks like meters
                height_m = hv
            elif 50.0 <= hv <= 260.0:                   # likely centimeters
                height_m = hv / 100.0
            elif 500.0 <= hv <= 2600.0:                 # likely millimeters
                height_m = hv / 1000.0
        elif hu == "cm":
            height_m = hv / 100.0
        elif hu == "mm":
            height_m = hv / 1000.0
        out["height_m"] = height_m

    # mass -> kg (accept kg / g or bare numbers)
    m_raw = _dig(session_meta, {"masskg","mass","weightkg","weight","bodymass","bodyweight","bodymasskg"})
    mv, mu = _parse_number_with_units(m_raw)
    if mv is not None:
        mass_kg = None
        if mu in {"kg", None}:
            if mu == "kg" or (20.0 <= mv <= 300.0):     # reasonable adult mass in kg
                mass_kg = mv
            elif 20000.0 <= mv <= 300000.0:             # grams accidentally
                mass_kg = mv / 1000.0
        elif mu == "g":
            mass_kg = mv / 1000.0
        out["mass_kg"] = mass_kg

    return out

def np_to_py(x):
    if x is None:
        return None
    x = np.asarray(x)
    if x.dtype.kind in 'fc':
        x = x.astype(np.float32)
    elif x.dtype.kind in 'iu':
        x = x.astype(np.int32)
    return x.tolist()



def to_jsonable(obj):
    """Recursively convert numpy/Path/etc. to JSON-serializable Python types."""
    import numpy as _np
    from pathlib import Path as _Path
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_np.floating, _np.integer)):
        return obj.item()
    if isinstance(obj, _Path):
        return str(obj)
    return obj


def get_keypoint_names(dataset_meta: dict):
    if not isinstance(dataset_meta, dict):
        return None
    # Preferred: keypoint_info (dict or list)
    kinfo = dataset_meta.get('keypoint_info')
    if isinstance(kinfo, dict):
        names = [None] * len(kinfo)
        for v in kinfo.values():
            names[v['id']] = v['name']
        return names
    if isinstance(kinfo, list):
        return [e['name'] for e in sorted(kinfo, key=lambda x: x['id'])]
    # Fallbacks occasionally seen
    id2name = dataset_meta.get('keypoint_id2name')
    if isinstance(id2name, (list, tuple)):
        return list(id2name)
    names = dataset_meta.get('dataset_keypoints')
    if isinstance(names, (list, tuple)):
        return list(names)
    return None


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def pick_trials(manifest: dict, subset: str, trial_ids: list | None):
    trials_root = manifest['trials']
    subsets = [subset] if subset in trials_root else list(trials_root.keys())
    for s in subsets:
        for t in trials_root[s]:
            if (trial_ids is None) or (t['id'] in trial_ids):
                yield s, t


def decide_output_dir(manifest: dict, trial: dict, outputs_root_cli: str | None):
    base = manifest.get('output_dir')
    if not base:
        outputs_root = outputs_root_cli or Path.cwd() / "outputs"
        subj = manifest.get('subject_id', 'subject')
        sess = manifest.get('session', 'Session')
        cam = manifest.get('camera', 'Cam')
        base = Path(outputs_root) / subj / sess / cam
    return Path(base) / trial['id'] / "rtmw3d"


def open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps):
        fps = 30.0
    return cap, float(fps)


def whole_frame_bboxes(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    # MMPose expects xyxy by default
    return np.array([[0.0, 0.0, float(w - 1), float(h - 1)]], dtype=np.float32)


# ----------------------------
# Dataset metainfo resolution
# ----------------------------

import importlib.util as _importlib_util

def _load_dataset_info_py(py_path: str):
    """Load a dataset_info dict by executing a metainfo .py file (like h3wb.py).
    Returns None if the file can't be loaded or doesn't define dataset_info.
    """
    try:
        spec = _importlib_util.spec_from_file_location("_mmpose_metainfo_tmp", py_path)
        if spec is None or spec.loader is None:
            return None
        mod = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return getattr(mod, 'dataset_info', None)
    except Exception:
        return None


def _names_from_dataset_info(dataset_info: dict):
    if not isinstance(dataset_info, dict):
        return None
    kinfo = dataset_info.get('keypoint_info')
    if isinstance(kinfo, dict) and kinfo:
        # Build by id to ensure correct order
        try:
            K = 1 + max(int(v['id']) for v in kinfo.values())
            names = [None] * K
            for v in kinfo.values():
                idx = int(v['id'])
                names[idx] = str(v['name'])
            if all(isinstance(n, str) for n in names):
                return names
        except Exception:
            pass
    return None

# ----------------------------
# Dataset metainfo resolution
# ----------------------------

def resolve_dataset_meta(args, model):
    """Populate model.dataset_meta so names/skeleton are never None.
    Priority: --metainfo-from-file > config.metainfo/dataloader.metainfo > fallback.
    Adds verbose logging to help diagnose missing names.
    """
    def log(*x):
        if getattr(args, 'debug_metainfo', False):
            print('[META]', *x)

    # If already set with keypoint_info, keep it
    dm = getattr(model, 'dataset_meta', None)
    names = get_keypoint_names(dm or {})
    if names:
        log('model.dataset_meta already has names (K=', len(names), ')')
        return dm

    cfg = Config.fromfile(args.config)

    def _maybe(mi):
        return mi if isinstance(mi, (dict, ConfigDict)) else None

    candidates = [
        _maybe(cfg.get('metainfo')),
        _maybe(cfg.get('dataset_info')),
        _maybe(cfg.get('test_dataloader', {}).get('dataset', {}).get('metainfo')),
        _maybe(cfg.get('val_dataloader', {}).get('dataset', {}).get('metainfo')),
        _maybe(cfg.get('train_dataloader', {}).get('dataset', {}).get('metainfo')),
    ]

    metainfo_src = None

    # Highest priority: explicit CLI
    if getattr(args, 'metainfo_from_file', None):
        metainfo_src = dict(from_file=str(Path(args.metainfo_from_file).resolve()))
        log('using --metainfo-from-file =', metainfo_src['from_file'])
    else:
        # First non-empty candidate
        for c in candidates:
            if c:
                metainfo_src = dict(c)
                break
        # Resolve relative from_file against config dir
        if metainfo_src and 'from_file' in metainfo_src:
            base = Path(args.config).resolve().parent
            p = Path(metainfo_src['from_file'])
            metainfo_src['from_file'] = str((base / p).resolve())
            log('metainfo from config =', metainfo_src['from_file'])

    # Try parse if we have a source
    if metainfo_src:
        model.dataset_meta = parse_pose_metainfo(metainfo_src)
        dm = model.dataset_meta
        names = get_keypoint_names(dm or {})
        log('parsed metainfo keys =', sorted(list((dm or {}).keys())))
        if names:
            log('extracted keypoint_names K =', len(names))
            return dm
        else:
            log('no keypoint_names in provided metainfo; will try WholeBody fallback if appropriate')

    # Fallback: if dataset looks like WholeBody-133 (H3WB etc.), try coco_wholebody.py
    def find_coco_wholebody_py():
        try:
            import mmpose, os
            pkg = os.path.dirname(mmpose.__file__)
            p = os.path.join(pkg, 'configs', '_base_', 'datasets', 'coco_wholebody.py')
            return p if os.path.isfile(p) else None
        except Exception:
            return None

    ds_name = ((dm or {}).get('dataset_name') or '').lower()
    sigmas = (dm or {}).get('sigmas')
    looks_wholebody = (ds_name in {'h3wb','coco_wholebody','coco-wholebody'}) or (isinstance(sigmas, (list, tuple)) and len(sigmas) == 133)
    if looks_wholebody:
        cw_path = find_coco_wholebody_py()
        if cw_path:
            log('fallback to coco_wholebody metainfo at', cw_path)
            model.dataset_meta = parse_pose_metainfo(dict(from_file=cw_path))
            return model.dataset_meta
        else:
            warnings.warn(
                'Keypoint names missing. Could not auto-find coco_wholebody.py in your site-packages.'
                'Please pass --metainfo-from-file /ABS/PATH/to/mmpose/configs/_base_/datasets/coco_wholebody.py',
                RuntimeWarning)
            return dm

    # Nothing else we can do
    warnings.warn(
        'No dataset metainfo with keypoint names found. meta.json will have keypoint_names=null.'
        'Pass --metainfo-from-file to a dataset metainfo that contains keypoint_info.',
        RuntimeWarning,
    )
    return dm


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Run RTMW3D on videos listed in a manifest (no detector).")
    ap.add_argument("-m", "--manifest", required=True, help="Path to manifest.yaml")
    ap.add_argument("-p", "--paths", required=True, help="Path to paths.yaml")
    ap.add_argument("--subset", choices=["healthy", "pathological", "static", "all"], default="all")
    ap.add_argument("--trials", type=str, default=None, help="Comma-separated trial IDs (default: all)")
    ap.add_argument("--video-field", choices=["video_sync", "video_raw"], default="video_sync")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    ap.add_argument("--config", default="models/rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288.py")
    ap.add_argument("--checkpoint", default="models/rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth")
    ap.add_argument("--metainfo-from-file", default=None, help="Absolute path to dataset metainfo .py (e.g., cocktail14.py)")
    ap.add_argument("--outputs-root", default=None, help="Override outputs root if manifest.output_dir is absent")
    ap.add_argument("--print-every", type=int, default=50)
    # Performance / IO
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    ap.add_argument("--min-mean-score", type=float, default=0.0, help="Drop persons below this mean keypoint score")
    ap.add_argument("--gzip", action="store_true", help="Write preds.jsonl.gz instead of plain jsonl")
    ap.add_argument("--amp", action="store_true", help="CUDA only: mixed-precision inference")
    ap.add_argument("--debug-metainfo", action="store_true", help="Print detailed metainfo resolution logs")
    args = ap.parse_args()

    # Load and resolve the manifest via your loader
    manifest = load_manifest(args.manifest, args.paths)
    
    # Build RTMW3D once
    import importlib
    importlib.import_module('rtmpose3d.rtmpose3d.pose_estimator')
    importlib.import_module('rtmpose3d.rtmpose3d.rtmw3d_head')
    importlib.import_module('rtmpose3d.rtmpose3d.loss')           # if used by the config
    importlib.import_module('rtmpose3d.rtmpose3d.simcc_3d_label')  # if used by the config

    from mmpose.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    model = init_model(args.config, args.checkpoint, device=args.device)

    # Ensure dataset_meta is populated so keypoint_names/skeleton are valid
    dataset_meta = resolve_dataset_meta(args, model) or {}

    # Try to extract keypoint names; if absent, try reading the .py metainfo directly
    kp_names = get_keypoint_names(dataset_meta)
    if not kp_names and getattr(args, 'metainfo_from_file', None):
        alt_di = _load_dataset_info_py(str(Path(args.metainfo_from_file).resolve()))
        alt_names = _names_from_dataset_info(alt_di)
        if alt_names:
            kp_names = alt_names
            if args.debug_metainfo:
                print('[META] keypoint_names extracted directly from', args.metainfo_from_file, f'(K={len(kp_names)})')
        elif args.debug_metainfo:
            print('[META] could not extract names from', args.metainfo_from_file)

    # Write meta (names/skeleton) once per subject/session/camera tree
    # (kp_names was resolved above)
    skeleton = (
        dataset_meta.get('skeleton_links')
        or dataset_meta.get('skeleton_info')
        or dataset_meta.get('skeleton')
    )

    wrote_meta_roots = set()

    trial_ids = [t.strip() for t in args.trials.split(",")] if args.trials else None
    subsets = ["healthy", "pathological"] if args.subset == "all" else [args.subset]

    for subset_name in subsets:
        for subset_name2, trial in pick_trials(manifest, subset_name, trial_ids):
            vid_path = trial.get(args.video_field)
            if not vid_path:
                print(f"[SKIP] {trial['id']} has no '{args.video_field}'.")
                continue

            out_dir = decide_output_dir(manifest, trial, args.outputs_root)
            ensure_dir(out_dir)
            out_path = out_dir / ("preds.jsonl.gz" if args.gzip else "preds.jsonl")
            meta_path = out_dir.parent / "meta.json"

            # Write meta once per subject/session/camera
            meta_root = meta_path.parent
            if str(meta_root) not in wrote_meta_roots:
                ensure_dir(meta_root)
                # Load session metadata (path or dict) to auto-fill subject info
                session_meta_src = manifest.get("session_metadata")
                session_meta = None
                if isinstance(session_meta_src, (str, os.PathLike)) and Path(session_meta_src).exists():
                    try:
                        with open(session_meta_src, 'r', encoding='utf-8') as sf:
                            session_meta = yaml.safe_load(sf)
                    except Exception as e:
                        if args.debug_metainfo:
                            print('[META] failed to read session_metadata:', e)
                elif isinstance(session_meta_src, dict):
                    session_meta = session_meta_src

                subj_auto = extract_subject_from_session_metadata(session_meta or {})
                meta_payload = {
                    "dataset_name": dataset_meta.get("dataset_name"),
                    "keypoint_names": kp_names,
                    "skeleton": skeleton,
                    "root_idx": dataset_meta.get("root_idx"),
                    "sigmas": dataset_meta.get("sigmas"),
                    "note": "RTMW3D keypoints_xyz are (x,y,z) root-relative, scale-ambiguous; keypoints_px are pixel coords.",
                    "subject": {
                        "id": manifest.get("subject_id"),
                        "sex": subj_auto.get("sex"),
                        "height_m": subj_auto.get("height_m"),
                        "mass_kg": subj_auto.get("mass_kg"),
                    },
                    "provenance": {
                        "config": os.path.basename(args.config),
                        "checkpoint": os.path.basename(args.checkpoint),
                    },
                    "calibration": {
                        "intrinsics_extrinsics": manifest["calibration"].get("intrinsics_extrinsics"),
                        "mocap_to_video": manifest["calibration"].get("mocap_to_video"),
                        "extrinsic_video": manifest["calibration"].get("extrinsic_video"),
                    },
                    "session_metadata": manifest.get("session_metadata"),
                }
                with open(meta_path, "w", encoding="utf-8") as mf:
                    json.dump(to_jsonable(meta_payload), mf, indent=2)
                wrote_meta_roots.add(str(meta_root))

            # Open video
            cap, fps_file = open_video(vid_path)
            fps_manifest = manifest.get('fps_video', 'auto')
            fps = float(fps_file) if (fps_manifest == 'auto') else float(fps_manifest)

            print(f"[INFO] {trial['id']} ({args.video_field}) → {out_path}")
            frame_idx = 0

            open_fn = (lambda p, m: gzip.open(p, m)) if args.gzip else (lambda p, m: open(p, m, encoding='utf-8'))
            amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                       if (args.amp and isinstance(args.device, str) and args.device.startswith("cuda"))
                       else nullcontext())

            with open_fn(out_path, "wt") as f, torch.inference_mode():
                while True:
                    ok, frame_bgr = cap.read()
                    if not ok:
                        break

                    # Skip by stride
                    if frame_idx % args.stride != 0:
                        frame_idx += 1
                        continue

                    # Prefer accurate timestamps (VFR-safe)
                    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    time_sec = (pos_msec / 1000.0) if (pos_msec and not np.isnan(pos_msec) and pos_msec > 0) else (frame_idx / fps)

                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    bboxes = whole_frame_bboxes(frame_rgb)

                    with amp_ctx:
                        results = inference_topdown(model, frame_rgb, bboxes=bboxes)

                    persons = []
                    if results:
                        ds = results[0]
                        pred = ds.pred_instances
                        kps_xyz = getattr(pred, 'keypoints', None)                # (P, K, 3)
                        scores = getattr(pred, 'keypoint_scores', None)           # (P, K)
                        kps_px = getattr(pred, 'transformed_keypoints', None)     # (P, K, 2)

                        if kps_xyz is not None:
                            kps_xyz = np.asarray(kps_xyz)
                            P = kps_xyz.shape[0]
                            for p in range(P):
                                mean_score = float(np.asarray(scores)[p].mean()) if scores is not None else 1.0
                                if mean_score < args.min_mean_score:
                                    continue
                                persons.append({
                                    "keypoints_xyz": np_to_py(kps_xyz[p]),
                                    "keypoint_scores": np_to_py(np.asarray(scores)[p]) if scores is not None else None,
                                    "keypoints_px": np_to_py(np.asarray(kps_px)[p]) if kps_px is not None else None,
                                    "mean_score": mean_score,
                                })

                    f.write(json.dumps({
                        "trial_id": trial["id"],
                        "subset": subset_name2,
                        "frame_index": frame_idx,
                        "time_sec": time_sec,
                        "persons": persons
                    }) + "\n")

                    if frame_idx % args.print_every == 0:
                        print(f"  [{trial['id']}] frame {frame_idx}, persons={len(persons)}")
                    frame_idx += 1

            cap.release()
            print(f"[DONE] {trial['id']} → {out_path}")

    print("All selected trials processed.")


if __name__ == "__main__":
    main()
