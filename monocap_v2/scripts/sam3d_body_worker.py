#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monocap_v2.core.logging_utils import write_json
from monocap_v2.core.sam3d_body_adapter import build_sam3d_pose_artifact
from monocap_v2.core.video_io import read_video_frames_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official SAM3D Body inference for monocap_v2.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-pose", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--settings-json", default="{}")
    parser.add_argument("--camera-json", default="{}")
    parser.add_argument("--subject-json", default="{}")
    parser.add_argument("--cfg-json", default="{}")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    progress = _Progress(args.report, started)
    try:
        progress.write("parse_settings")
        settings = json.loads(args.settings_json or "{}")
        camera = json.loads(args.camera_json or "{}")
        subject = json.loads(args.subject_json or "{}")
        cfg = json.loads(args.cfg_json or "{}")
        frame_results, video_report = run_sam3d_body_video(args.video, settings, camera, progress)
        progress.write("build_artifact")
        artifact, qc = build_sam3d_pose_artifact(
            frame_results,
            fps=float(video_report.get("fps") or 30.0),
            camera=camera,
            subject=subject,
            cfg=cfg,
            source_video=str(args.video),
            inference_video=str(args.video),
            video_report=video_report,
        )
        qc["runtime_sec"] = time.perf_counter() - started
        args.output_pose.parent.mkdir(parents=True, exist_ok=True)
        with args.output_pose.open("wb") as f:
            pickle.dump(artifact, f)
        progress.write("write_qc")
        write_json(args.report, qc)
        return 0
    except Exception as exc:
        report = {
            "stage": "pose3d_sam3d_body",
            "status": "failed",
            "backend": "sam3d_body",
            "error": str(exc),
            "runtime_sec": time.perf_counter() - started,
        }
        try:
            write_json(args.report, report)
        except Exception:
            pass
        print(f"[sam3d-body-worker] failed: {exc}", file=sys.stderr, flush=True)
        return 1


def run_sam3d_body_video(
    video_path: Path, settings: dict[str, Any], camera: dict[str, Any] | None = None, progress: "_Progress | None" = None
) -> tuple[list[dict[str, Any] | None], dict[str, Any]]:
    repo_path = Path(str(settings.get("repo_path") or "external/sam-3d-body")).resolve()
    checkpoint_path = Path(str(settings.get("checkpoint_path") or "models/sam3d_body/sam-3d-body-dinov3/model.ckpt")).resolve()
    mhr_model_path = Path(str(settings.get("mhr_model_path") or "models/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt")).resolve()
    _progress(progress, "validate_paths")
    if not repo_path.exists():
        raise FileNotFoundError(f"SAM3D Body repo not found: {repo_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM3D Body checkpoint not found: {checkpoint_path}")
    if not mhr_model_path.exists():
        raise FileNotFoundError(f"SAM3D Body MHR model not found: {mhr_model_path}")
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    _progress(progress, "read_video_start", video=str(video_path), max_frames=_max_frames(settings))
    frames, report = read_video_frames_bgr(video_path, max_frames=_max_frames(settings))
    _progress(progress, "read_video_done", frames=len(frames), fps=report.get("fps"))
    estimator = _load_estimator(settings, checkpoint_path, mhr_model_path, progress)
    _progress(progress, "load_estimator_done")
    cam_int = _camera_intrinsics_tensor(camera or {}, settings, progress)
    outputs: list[dict[str, Any] | None] = []
    for index, frame in enumerate(frames):
        _progress(progress, "frame_start", frame_index=index, total_frames=len(frames))
        rgb = frame.bgr[:, :, ::-1].copy()
        result = estimator.process_one_image(
            rgb,
            bbox_thr=float(settings.get("bbox_threshold", settings.get("bbox_thresh", 0.8))),
            use_mask=bool(settings.get("use_mask", False)),
            inference_type=str(settings.get("inference_type", "full")),
            cam_int=cam_int,
        )
        normalized = _normalize_sam3d_result(result)
        _attach_faces(normalized, getattr(estimator, "faces", None))
        outputs.append(normalized)
        _progress(progress, "frame_done", frame_index=index, total_frames=len(frames), has_detection=normalized is not None)
    return outputs, report


def _load_estimator(settings: dict[str, Any], checkpoint_path: Path, mhr_model_path: Path, progress: "_Progress | None" = None) -> Any:
    _progress(progress, "import_sam3d_start")
    import torch
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    requested = str(settings.get("device") or "auto")
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    _progress(progress, "load_body_model_start", device=str(device), checkpoint=str(checkpoint_path), mhr_model=str(mhr_model_path))
    model, model_cfg = load_sam_3d_body(str(checkpoint_path), device=device, mhr_path=str(mhr_model_path))
    _progress(progress, "load_body_model_done")
    human_detector = None
    human_segmentor = None
    fov_estimator = None
    detector_name = settings.get("detector_name")
    segmentor_name = settings.get("segmentor_name")
    fov_name = settings.get("fov_name")
    if detector_name:
        _progress(progress, "load_detector_start", detector=str(detector_name))
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(name=str(detector_name), device=device, path=str(settings.get("detector_path") or ""))
        _progress(progress, "load_detector_done", detector=str(detector_name))
    if segmentor_name:
        _progress(progress, "load_segmentor_start", segmentor=str(segmentor_name))
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(name=str(segmentor_name), device=device, path=str(settings.get("segmentor_path") or ""))
        _progress(progress, "load_segmentor_done", segmentor=str(segmentor_name))
    if fov_name:
        _progress(progress, "load_fov_start", fov=str(fov_name))
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=str(fov_name), device=device, path=str(settings.get("fov_path") or ""))
        _progress(progress, "load_fov_done", fov=str(fov_name))
    return SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )


def _camera_intrinsics_tensor(camera: dict[str, Any], settings: dict[str, Any], progress: "_Progress | None" = None) -> Any:
    if not bool(settings.get("use_camera_intrinsics", False)):
        return None
    import torch

    required = ["fx", "fy", "cx", "cy"]
    missing = [key for key in required if camera.get(key) is None]
    if missing:
        raise ValueError(f"SAM3D camera intrinsics were requested, but camera is missing: {', '.join(missing)}")
    k = torch.tensor(
        [
            [
                [float(camera["fx"]), 0.0, float(camera["cx"])],
                [0.0, float(camera["fy"]), float(camera["cy"])],
                [0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    _progress(progress, "camera_intrinsics_ready", fx=float(camera["fx"]), fy=float(camera["fy"]), cx=float(camera["cx"]), cy=float(camera["cy"]))
    return k


def _normalize_sam3d_result(result: Any) -> dict[str, Any] | None:
    if result is None:
        return None
    if isinstance(result, tuple) and result:
        result = result[0]
    if isinstance(result, list):
        return {"detections": [_normalize_detection(item) for item in result if item is not None]}
    if isinstance(result, dict):
        if "detections" in result or "people" in result:
            return result
        count = _first_count(result)
        if count is None:
            return _normalize_detection(result)
        return {"detections": [_slice_detection(result, idx) for idx in range(count)]}
    return None


def _normalize_detection(detection: Any) -> dict[str, Any]:
    if isinstance(detection, dict):
        return detection
    return {}


def _attach_faces(result: dict[str, Any] | None, faces: Any) -> None:
    if result is None or faces is None:
        return
    face_array = _to_numpy(faces).astype(np.int32)
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        return
    detections = result.get("detections")
    if isinstance(detections, list):
        for detection in detections:
            if isinstance(detection, dict) and "faces" not in detection:
                detection["faces"] = face_array
    elif "faces" not in result:
        result["faces"] = face_array


def _first_count(result: dict[str, Any]) -> int | None:
    for key in ("pred_keypoints_3d", "pred_vertices", "pred_keypoints_2d", "bbox", "boxes"):
        if key not in result or result[key] is None:
            continue
        arr = _to_numpy(result[key])
        if arr.ndim >= 3:
            return int(arr.shape[0])
        if key in {"bbox", "boxes"} and arr.ndim == 2:
            return int(arr.shape[0])
    return None


def _slice_detection(result: dict[str, Any], idx: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in result.items():
        if value is None:
            continue
        arr = _to_numpy(value)
        if arr.ndim >= 1 and arr.shape[0] > idx:
            out[key] = arr[idx]
        else:
            out[key] = value
    return out


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _max_frames(settings: dict[str, Any]) -> int | None:
    value = settings.get("max_frames")
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


class _Progress:
    def __init__(self, report_path: Path, started: float):
        self.report_path = report_path
        self.started = started

    def write(self, phase: str, **extra: Any) -> None:
        report = {
            "stage": "pose3d_sam3d_body",
            "status": "running",
            "backend": "sam3d_body",
            "phase": phase,
            "runtime_sec": time.perf_counter() - self.started,
            **extra,
        }
        try:
            write_json(self.report_path, report)
        except Exception:
            pass


def _progress(progress: _Progress | None, phase: str, **extra: Any) -> None:
    if progress is not None:
        progress.write(phase, **extra)


if __name__ == "__main__":
    raise SystemExit(main())
