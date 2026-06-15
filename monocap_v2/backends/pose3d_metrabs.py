from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from monocap_v2.core.video_io import read_video_frames_bgr


def run_pose3d(video_path: Path, pose2d: dict | None, camera: dict, subject: dict, cfg: dict) -> dict:
    """Run MeTRAbs and return the backend-neutral joints-only artifact.

    MeTRAbs reports 3D joints in millimetres. monocap_v2 stores internal 3D
    coordinates in metres, so this adapter converts before returning.
    """
    tf = _import_tensorflow()
    _configure_tf_memory_growth(tf)

    metrabs_cfg = cfg.get("config", {}).get("metrabs", {})
    model = _load_model(tf, metrabs_cfg)
    skeleton = str(metrabs_cfg.get("skeleton", "smpl_24"))
    requested_names = _skeleton_names(model, skeleton)

    video_info = cfg.get("video_info", {})
    fps = float(video_info.get("fps") or 30.0)
    max_frames = metrabs_cfg.get("max_frames")
    max_frames_int = int(max_frames) if max_frames is not None else None
    max_detections = int(metrabs_cfg.get("max_detections", 1))
    suppress_implausible = bool(metrabs_cfg.get("suppress_implausible_poses", False))

    K = _camera_intrinsic_matrix(camera)
    dist = _distortion_coeffs(camera)

    frames_3d_mm: list[np.ndarray | None] = []
    frames_2d_px: list[np.ndarray | None] = []
    frame_scores: list[float] = []

    video_frames, video_read_report = read_video_frames_bgr(video_path, max_frames=max_frames_int)
    for _frame_index, frame_rgb in _iter_video_rgb(video_frames):
        pred = model.detect_poses(
            tf.convert_to_tensor(frame_rgb, dtype=tf.uint8),
            skeleton=skeleton,
            intrinsic_matrix=tf.constant(K, tf.float32),
            distortion_coeffs=tf.constant(dist, tf.float32),
            max_detections=max_detections,
            suppress_implausible_poses=suppress_implausible,
        )
        pose3d_mm, pose2d_px, score = _select_best_detection(pred)
        frames_3d_mm.append(pose3d_mm)
        frames_2d_px.append(pose2d_px)
        frame_scores.append(score)

    joint_count = _infer_joint_count(frames_3d_mm, requested_names)
    joint_names = _resolve_joint_names(model, skeleton, joint_count)
    joints_m = _stack_frames(frames_3d_mm, joint_count, dims=3) / 1000.0
    xy = _stack_frames(frames_2d_px, joint_count, dims=2)
    confidence = np.repeat(np.asarray(frame_scores, dtype=np.float32)[:, None], joint_count, axis=1)
    confidence[~np.isfinite(xy).all(axis=2)] = 0.0

    return {
        "representation": "joints",
        "backend": "metrabs",
        "fps": fps,
        "units": "m",
        "joint_names": joint_names,
        "joints_3d": joints_m.astype(np.float32),
        "pose2d": {
            "xy": xy.astype(np.float32),
            "confidence": confidence.astype(np.float32),
            "names": joint_names,
            "fps": fps,
            "backend": "metrabs",
        },
        "camera": {
            "intrinsics": camera,
            "extrinsics": None,
            "is_assumed": bool(camera.get("is_assumed", True)),
        },
        "source_video": str(video_path),
        "subject": {"id": subject.get("id"), "height_m": subject.get("height_m"), "mass_kg": subject.get("mass_kg")},
        "backend_meta": {
            "skeleton_requested": skeleton,
            "model_type": metrabs_cfg.get("model_type", "metrabs_eff2l_y4"),
            "max_frames": max_frames_int,
            "num_frames": int(joints_m.shape[0]),
            "video_read_report": video_read_report,
            "frame_indices": [int(frame.index) for frame in video_frames],
            "num_detections": int(np.count_nonzero(np.asarray(frame_scores) > 0)),
        },
    }


def _import_tensorflow():
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - exercised only when env is wrong
        raise RuntimeError("TensorFlow is required for the MeTRAbs backend. Use the monocap-v2 conda env.") from exc
    return tf


def _configure_tf_memory_growth(tf) -> None:
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def _load_model(tf, metrabs_cfg: dict[str, Any]):
    model_path = metrabs_cfg.get("model_path")
    if model_path and (Path(model_path) / "saved_model.pb").exists():
        return tf.saved_model.load(str(model_path))
    model_type = str(metrabs_cfg.get("model_type", "metrabs_eff2l_y4"))
    return tf.saved_model.load(_download_model(tf, model_type))


def _download_model(tf, model_type: str) -> str:
    server_prefix = "https://omnomnom.vision.rwth-aachen.de/data/metrabs"
    fname = f"{model_type}_20211019.zip"
    zip_path = Path(tf.keras.utils.get_file(fname=fname, origin=f"{server_prefix}/{fname}", cache_subdir="models", extract=False))
    out_dir = zip_path.parent / model_type
    if (out_dir / "saved_model.pb").exists():
        return str(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    with zipfile.ZipFile(zip_path, "r") as zf, tempfile.TemporaryDirectory(dir=str(zip_path.parent)) as tmpdir:
        zf.extractall(tmpdir)
        top = Path(tmpdir)
        candidate = next((p for p in top.iterdir() if p.is_dir() and (p / "saved_model.pb").exists()), None)
        if candidate is None and (top / "saved_model.pb").exists():
            candidate = top
        if candidate is None:
            raise RuntimeError("Downloaded MeTRAbs archive does not contain a SavedModel.")
        shutil.move(str(candidate), str(out_dir))
    return str(out_dir)


def _iter_video_rgb(video_frames):
    import cv2

    for frame in video_frames:
        yield frame.index, cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB)


def _camera_intrinsic_matrix(camera: dict[str, Any]) -> np.ndarray:
    return np.array(
        [[float(camera["fx"]), 0.0, float(camera["cx"])], [0.0, float(camera["fy"]), float(camera["cy"])], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _distortion_coeffs(camera: dict[str, Any]) -> np.ndarray:
    distortion = camera.get("distortion") or {}
    if not distortion.get("enabled", False):
        return np.zeros((5,), dtype=np.float32)
    coeffs = distortion.get("coeffs") or distortion.get("coefficients") or [0, 0, 0, 0, 0]
    out = np.zeros((5,), dtype=np.float32)
    coeffs_arr = np.asarray(coeffs, dtype=np.float32).reshape(-1)
    out[: min(5, coeffs_arr.size)] = coeffs_arr[:5]
    return out


def _select_best_detection(pred: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    boxes = _to_numpy(pred.get("boxes"))
    poses3d = _to_numpy(pred.get("poses3d"))
    poses2d = _to_numpy(pred.get("poses2d"))
    if boxes is None or poses3d is None or boxes.shape[0] == 0 or poses3d.shape[0] == 0:
        return None, None, 0.0
    scores = boxes[:, -1] if boxes.ndim == 2 and boxes.shape[1] > 0 else np.ones((boxes.shape[0],), dtype=np.float32)
    idx = int(np.nanargmax(scores))
    score = float(scores[idx]) if np.isfinite(scores[idx]) else 0.0
    pose3d = poses3d[idx] if idx < poses3d.shape[0] else None
    pose2d = poses2d[idx] if poses2d is not None and idx < poses2d.shape[0] else None
    return pose3d, pose2d, score


def _to_numpy(value) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _infer_joint_count(frames: list[np.ndarray | None], requested_names: list[str]) -> int:
    for frame in frames:
        if frame is not None and frame.ndim == 2 and frame.shape[1] == 3:
            return int(frame.shape[0])
    return len(requested_names)


def _stack_frames(frames: list[np.ndarray | None], joint_count: int, dims: int) -> np.ndarray:
    out = np.full((len(frames), joint_count, dims), np.nan, dtype=np.float32)
    for i, frame in enumerate(frames):
        if frame is None:
            continue
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != dims:
            continue
        n = min(joint_count, arr.shape[0])
        out[i, :n, :] = arr[:n, :]
    return out


def _skeleton_names(model, skeleton: str) -> list[str]:
    try:
        return [str(n) for n in model.per_skeleton_joint_names[skeleton].numpy().astype(str).tolist()]
    except Exception:
        return []


def _resolve_joint_names(model, skeleton: str, joint_count: int) -> list[str]:
    requested = _skeleton_names(model, skeleton)
    if len(requested) == joint_count:
        return requested
    fallback = _skeleton_names(model, "")
    if len(fallback) == joint_count:
        return fallback
    return [f"J{i:03d}" for i in range(joint_count)]
