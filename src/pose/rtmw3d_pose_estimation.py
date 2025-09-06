#!/usr/bin/env python3
"""
Run RTMW3D on videos listed in a manifest with **good crops** and optional detector,
then write preds.jsonl(.gz) + meta.json with fully populated keypoint names/skeleton.

Key fixes & features
--------------------
-  **Correct color space**: keep frames in **BGR** for MMPose.
-  **Tight person crops**:
  - Optional **MMDetection** person detector (run every N frames), or
  - **Detector-free** temporal tracking using the keypoint bbox from the previous frame.
-  **bbox_format='xyxy'** set explicitly for clarity.
-  `model.eval()` and `torch.inference_mode()` for correctness & speed.
-  Optional AMP (`--amp`) on CUDA.
-  Accurate timestamps with fallback to frame_idx / fps.
-  Optional two-pass refine (`--refine-pass`) to re-run on kp-tightened bbox.

Usage examples
--------------
# 1) No detector, temporal kp-bbox tracking (fast, works well for single subject)
python src/pose/rtmw3d_pose_estimation.py \
  -m manifests/OpenCapDataset/subject2-2.yaml -p config/paths.yaml \
  --trials walking1 --video-field video_sync --stride 1\
  --metainfo-from-file external/datasets_config/h3wb.py --refine-pass

# 2) With MMDetection person detector every 10 frames (robust to big motion)
python rtmw3d_pose_estimation_fixed.py \
  -m manifests/OpenCapDataset/subject2-2.yaml -p config/paths.yaml \
  --trials walking1 --video-field video_sync \
  --detector mmdet \
  --det-config mmdetection/configs/yolo/yolov3_d53_320_273e_coco.py \
  --det-checkpoint weights/yolov3_d53_320_273e_coco-421362b6.pth \
  --det-stride 10 --det-score-thr 0.4 --refine-pass

Notes
-----
- This script writes **top-down** predictions. `keypoints_xyz` are root-relative,
  scale-ambiguous; `keypoints_px` are in image pixels.
- If you don't have MMDetection installed, keep `--detector none` (default) and the
  tracker will still produce good crops after the first frame.
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

# your loader in a separate file
from IO.load_manifest import load_manifest

# Optional MMDetection
_HAS_MMDET = False
try:
    from mmdet.apis import init_detector as mmdet_init_detector
    from mmdet.apis import inference_detector as mmdet_inference_detector
    _HAS_MMDET = True
except Exception:
    _HAS_MMDET = False


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
    # height -> meters
    h_raw = _dig(session_meta, {"heightm", "height", "stature", "bodyheight", "bodyheightm", "staturem", "bodyheightmeter"})
    hv, hu = _parse_number_with_units(h_raw)
    if hv is not None:
        height_m = None
        if hu in {"m", None}:
            if hu == "m" or (0.5 <= hv <= 2.6):
                height_m = hv
            elif 50.0 <= hv <= 260.0:
                height_m = hv / 100.0
            elif 500.0 <= hv <= 2600.0:
                height_m = hv / 1000.0
        elif hu == "cm":
            height_m = hv / 100.0
        elif hu == "mm":
            height_m = hv / 1000.0
        out["height_m"] = height_m
    # mass -> kg
    m_raw = _dig(session_meta, {"masskg", "mass", "weightkg", "weight", "bodymass", "bodyweight", "bodymasskg"})
    mv, mu = _parse_number_with_units(m_raw)
    if mv is not None:
        mass_kg = None
        if mu in {"kg", None}:
            if mu == "kg" or (20.0 <= mv <= 300.0):
                mass_kg = mv
            elif 20000.0 <= mv <= 300000.0:
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
    kinfo = dataset_meta.get('keypoint_info')
    if isinstance(kinfo, dict):
        names = [None] * len(kinfo)
        for v in kinfo.values():
            names[v['id']] = v['name']
        return names
    if isinstance(kinfo, list):
        return [e['name'] for e in sorted(kinfo, key=lambda x: x['id'])]
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
        fps = 60.0
    return cap, float(fps)


# ----------------------------
# Dataset metainfo resolution
# ----------------------------

import importlib.util as _importlib_util


def _load_dataset_info_py(py_path: str):
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


def resolve_dataset_meta(args, model):
    """Populate model.dataset_meta so names/skeleton are never None.
    Priority: --metainfo-from-file > config.metainfo/dataloader.metainfo > fallback.
    """
    def log(*x):
        if getattr(args, 'debug_metainfo', False):
            print('[META]', *x)

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

    if getattr(args, 'metainfo_from_file', None):
        metainfo_src = dict(from_file=str(Path(args.metainfo_from_file).resolve()))
        log('using --metainfo-from-file =', metainfo_src['from_file'])
    else:
        for c in candidates:
            if c:
                metainfo_src = dict(c)
                break
        if metainfo_src and 'from_file' in metainfo_src:
            base = Path(args.config).resolve().parent
            p = Path(metainfo_src['from_file'])
            metainfo_src['from_file'] = str((base / p).resolve())
            log('metainfo from config =', metainfo_src['from_file'])

    if metainfo_src:
        model.dataset_meta = parse_pose_metainfo(metainfo_src)
        dm = model.dataset_meta
        names = get_keypoint_names(dm or {})
        if names:
            log('extracted keypoint_names K =', len(names))
            return dm
        else:
            log('no keypoint_names in provided metainfo; considering WholeBody fallback')

    # WholeBody fallback (common for H3WB)
    def find_coco_wholebody_py():
        try:
            import mmpose, os as _os
            pkg = _os.path.dirname(mmpose.__file__)
            p = _os.path.join(pkg, 'configs', '_base_', 'datasets', 'coco_wholebody.py')
            return p if _os.path.isfile(p) else None
        except Exception:
            return None

    ds_name = ((dm or {}).get('dataset_name') or '').lower()
    sigmas = (dm or {}).get('sigmas')
    looks_wholebody = (ds_name in {'h3wb','coco_wholebody','coco-wholebody'}) or (isinstance(sigmas, (list, tuple)) and len(sigmas) == 133)
    if looks_wholebody:
        cw_path = find_coco_wholebody_py()
        if cw_path:
            model.dataset_meta = parse_pose_metainfo(dict(from_file=cw_path))
            return model.dataset_meta
        else:
            warnings.warn('Keypoint names missing and WholeBody metainfo not auto-found. Please pass --metainfo-from-file.', RuntimeWarning)
            return dm

    warnings.warn('No dataset metainfo with keypoint names found. meta.json will have keypoint_names=null. Pass --metainfo-from-file.', RuntimeWarning)
    return dm


# ----------------------------
# BBox helpers (detector + tracker)
# ----------------------------

def _xyxy_from_keypoints(kp_px, img_w, img_h, pad=0.2, min_side=128):
    kp = np.asarray(kp_px)
    vis = np.isfinite(kp).all(axis=1)
    if not vis.any():
        return None
    x1, y1 = kp[vis].min(axis=0)
    x2, y2 = kp[vis].max(axis=0)
    w = x2 - x1
    h = y2 - y1
    pad_x = max(pad * w, (min_side - w) / 2)
    pad_y = max(pad * h, (min_side - h) / 2)
    x1 = float(max(0, x1 - pad_x)); y1 = float(max(0, y1 - pad_y))
    x2 = float(min(img_w - 1, x2 + pad_x)); y2 = float(min(img_h - 1, y2 + pad_y))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _heuristic_center_crop(h, w):
    side = int(0.9 * min(h, w))
    cx, cy = w // 2, int(h * 0.55)  # bias lower for gait
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - int(side * 0.6))
    x2 = min(w - 1, x1 + side)
    y2 = min(h - 1, y1 + int(side * 1.2))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _pick_det_bbox(dets_xyxy, scores, img_shape, prev_bbox=None):
    """Pick one bbox from mmdet outputs. Prefer near previous center; else largest area."""
    H, W = img_shape[:2]
    if dets_xyxy.size == 0:
        return None
    if prev_bbox is not None:
        pcx = 0.5 * (prev_bbox[0] + prev_bbox[2])
        pcy = 0.5 * (prev_bbox[1] + prev_bbox[3])
        centers = 0.5 * (dets_xyxy[:, :2] + dets_xyxy[:, 2:])
        d2 = (centers[:, 0] - pcx) ** 2 + (centers[:, 1] - pcy) ** 2
        idx = int(np.argmin(d2))
        return dets_xyxy[idx]
    # largest area
    areas = (dets_xyxy[:, 2] - dets_xyxy[:, 0]) * (dets_xyxy[:, 3] - dets_xyxy[:, 1])
    idx = int(np.argmax(areas))
    return dets_xyxy[idx]


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Run RTMW3D on videos listed in a manifest (tight crops; optional detector).")
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

    # Detector options
    ap.add_argument("--detector", choices=["none", "mmdet"], default="none")
    ap.add_argument("--det-config", default=None, help="MMDetection config for person detector")
    ap.add_argument("--det-checkpoint", default=None, help="MMDetection checkpoint for person detector")
    ap.add_argument("--det-score-thr", type=float, default=0.3, help="Score threshold for person detections")
    ap.add_argument("--det-stride", type=int, default=8, help="Run detector every N frames (tracking in between)")

    # Refinement
    ap.add_argument("--refine-pass", action="store_true", help="Re-run pose on kp-tightened bbox for better crops")

    args = ap.parse_args()

    # Load and resolve the manifest via your loader
    manifest = load_manifest(args.manifest, args.paths)

    # Build RTMW3D once
    import importlib
    importlib.import_module('rtmpose3d.rtmpose3d.pose_estimator')
    importlib.import_module('rtmpose3d.rtmpose3d.rtmw3d_head')
    importlib.import_module('rtmpose3d.rtmpose3d.loss')
    importlib.import_module('rtmpose3d.rtmpose3d.simcc_3d_label')

    from mmpose.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    model = init_model(args.config, args.checkpoint, device=args.device)
    model.eval()

    # Ensure dataset_meta is populated so keypoint_names/skeleton are valid
    dataset_meta = resolve_dataset_meta(args, model) or {}

    # Try to extract keypoint names; if absent, try reading the .py metainfo directly
    kp_names = get_keypoint_names(dataset_meta)
    if not kp_names and getattr(args, 'metainfo_from_file', None):
        alt_di = _load_dataset_info_py(str(Path(args.metainfo_from_file).resolve()))
        alt_names = _names_from_dataset_info(alt_di)
        if alt_names:
            kp_names = alt_names

    skeleton = (
        dataset_meta.get('skeleton_links')
        or dataset_meta.get('skeleton_info')
        or dataset_meta.get('skeleton')
    )

    # Optional detector
    det_model = None
    if args.detector == 'mmdet':
        if not _HAS_MMDET:
            warnings.warn('MMDetection not installed; falling back to detector=none', RuntimeWarning)
        elif not (args.det_config and args.det_checkpoint):
            warnings.warn('Detector config/checkpoint not provided; falling back to detector=none', RuntimeWarning)
        else:
            det_model = mmdet_init_detector(args.det_config, args.det_checkpoint, device=args.device)

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
                session_meta_src = manifest.get("session_metadata")
                session_meta = None
                if isinstance(session_meta_src, (str, os.PathLike)) and Path(session_meta_src).exists():
                    try:
                        with open(session_meta_src, 'r', encoding='utf-8') as sf:
                            session_meta = yaml.safe_load(sf)
                    except Exception as e:
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

            cap, fps_file = open_video(vid_path)
            fps_manifest = manifest.get('fps_video', 'auto')
            fps = float(fps_file) if (fps_manifest == 'auto') else float(fps_manifest)

            print(f"[INFO] {trial['id']} ({args.video_field}) → {out_path}")
            frame_idx = 0

            open_fn = (lambda p, m: gzip.open(p, m)) if args.gzip else (lambda p, m: open(p, m, encoding='utf-8'))
            amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                       if (args.amp and isinstance(args.device, str) and args.device.startswith("cuda"))
                       else nullcontext())

            prev_bbox_xyxy = None
            pad = 0.2
            min_side = 128

            with open_fn(out_path, "wt") as f, torch.inference_mode():
                while True:
                    ok, frame_bgr = cap.read()
                    if not ok:
                        break

                    if frame_idx % args.stride != 0:
                        frame_idx += 1
                        continue

                    # Timestamp (VFR-safe when POS_MSEC is valid)
                    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    time_sec = (pos_msec / 1000.0) if (pos_msec and not np.isnan(pos_msec) and pos_msec > 0) else (frame_idx / fps)

                    H, W = frame_bgr.shape[:2]

                    # --- Pick bbox ---
                    bbox_xyxy = None

                    # 1) Detector (periodic)
                    use_detector = det_model is not None and (frame_idx % max(1, args.det_stride) == 0)
                    if use_detector:
                        det_out = mmdet_inference_detector(det_model, frame_bgr)
                        # MMDet returns list per class or DetDataSample depending on version
                        det_bboxes = []
                        det_scores = []
                        try:
                            # new-style: DetDataSample
                            pred_instances = det_out.pred_instances
                            b = pred_instances.bboxes.cpu().numpy()
                            s = pred_instances.scores.cpu().numpy()
                            l = pred_instances.labels.cpu().numpy()
                            person_mask = (l == 0)  # COCO: class 0 = person
                            b = b[person_mask]; s = s[person_mask]
                            keep = s >= float(args.det_score_thr)
                            det_bboxes = b[keep]
                            det_scores = s[keep]
                        except Exception:
                            # old-style: list of arrays per class
                            try:
                                person_dets = det_out[0]  # class 0
                                person_dets = np.asarray(person_dets)
                                if person_dets.size > 0:
                                    det_bboxes = person_dets[:, :4]
                                    det_scores = person_dets[:, 4]
                                    keep = det_scores >= float(args.det_score_thr)
                                    det_bboxes = det_bboxes[keep]
                                    det_scores = det_scores[keep]
                            except Exception:
                                det_bboxes = []
                                det_scores = []

                        if isinstance(det_bboxes, np.ndarray) and det_bboxes.size > 0:
                            bbox_xyxy = _pick_det_bbox(det_bboxes, det_scores, frame_bgr.shape, prev_bbox=prev_bbox_xyxy)

                    # 2) Tracker (prev kp-bbox)
                    if bbox_xyxy is None and prev_bbox_xyxy is not None:
                        bbox_xyxy = prev_bbox_xyxy.copy()

                    # 3) First-frame heuristic
                    if bbox_xyxy is None:
                        bbox_xyxy = _heuristic_center_crop(H, W)

                    # --- Pose inference (first pass) ---
                    with amp_ctx:
                        results = inference_topdown(model, frame_bgr, bboxes=bbox_xyxy[None, :], bbox_format='xyxy')

                    persons = []
                    best_kp_for_refine = None
                    best_score_for_refine = -1.0

                    if results:
                        ds = results[0]
                        pred = ds.pred_instances
                        kps_xyz = getattr(pred, 'keypoints', None)
                        scores = getattr(pred, 'keypoint_scores', None)
                        kps_px = getattr(pred, 'transformed_keypoints', None)

                        if kps_xyz is not None:
                            kps_xyz = np.asarray(kps_xyz)
                            P = kps_xyz.shape[0]
                            # choose best person for tracking/refine
                            scores_np = np.asarray(scores) if scores is not None else None
                            for p in range(P):
                                mean_score = float(scores_np[p].mean()) if scores_np is not None else 1.0
                                if mean_score < args.min_mean_score:
                                    continue
                                px = np.asarray(kps_px)[p] if kps_px is not None else None
                                persons.append({
                                    "keypoints_xyz": np_to_py(kps_xyz[p]),
                                    "keypoint_scores": np_to_py(scores_np[p]) if scores_np is not None else None,
                                    "keypoints_px": np_to_py(px) if px is not None else None,
                                    "mean_score": mean_score,
                                })
                                if px is not None and mean_score > best_score_for_refine:
                                    best_score_for_refine = mean_score
                                    best_kp_for_refine = px

                    # --- Optional refine pass with kp-tightened bbox ---
                    if args.refine_pass and best_kp_for_refine is not None:
                        rb = _xyxy_from_keypoints(best_kp_for_refine, W, H, pad=pad, min_side=min_side)
                        if rb is not None:
                            with amp_ctx:
                                results2 = inference_topdown(model, frame_bgr, bboxes=rb[None, :], bbox_format='xyxy')
                            persons = []
                            if results2:
                                ds2 = results2[0]
                                pred2 = ds2.pred_instances
                                kps_xyz2 = getattr(pred2, 'keypoints', None)
                                scores2 = getattr(pred2, 'keypoint_scores', None)
                                kps_px2 = getattr(pred2, 'transformed_keypoints', None)
                                if kps_xyz2 is not None:
                                    kps_xyz2 = np.asarray(kps_xyz2)
                                    P2 = kps_xyz2.shape[0]
                                    scores2_np = np.asarray(scores2) if scores2 is not None else None
                                    for p in range(P2):
                                        mean_score = float(scores2_np[p].mean()) if scores2_np is not None else 1.0
                                        if mean_score < args.min_mean_score:
                                            continue
                                        px2 = np.asarray(kps_px2)[p] if kps_px2 is not None else None
                                        persons.append({
                                            "keypoints_xyz": np_to_py(kps_xyz2[p]),
                                            "keypoint_scores": np_to_py(scores2_np[p]) if scores2_np is not None else None,
                                            "keypoints_px": np_to_py(px2) if px2 is not None else None,
                                            "mean_score": mean_score,
                                        })
                                    # update best_kp_for_refine for tracking
                                    if kps_px2 is not None and len(kps_px2) > 0:
                                        best_kp_for_refine = np.asarray(kps_px2)[int(np.argmax([pp['mean_score'] for pp in persons]))]
                                        rb2 = _xyxy_from_keypoints(best_kp_for_refine, W, H, pad=pad, min_side=min_side)
                                        if rb2 is not None:
                                            prev_bbox_xyxy = rb2

                    # If no refine or refine not triggered, still update tracker bbox
                    if prev_bbox_xyxy is None and best_kp_for_refine is not None:
                        tb = _xyxy_from_keypoints(best_kp_for_refine, W, H, pad=pad, min_side=min_side)
                        if tb is not None:
                            prev_bbox_xyxy = tb

                    # Write output
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
