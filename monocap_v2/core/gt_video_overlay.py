from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from monocap_v2.core.logging_utils import read_yaml, write_json
from monocap_v2.core.mocap_eval import resample_timeseries
from monocap_v2.core.wham_timeline import estimate_raw_sync_alignment, video_metadata


GT_LABELS = {
    "pelvis": "PELV",
    "left_hip": "LHIP",
    "right_hip": "RHIP",
    "left_knee": "LKNE",
    "right_knee": "RKNE",
    "left_ankle": "LANK",
    "right_ankle": "RANK",
    "left_mtp": "LMTP",
    "right_mtp": "RMTP",
}


def render_gt_labeled_video_overlay(
    run_dir: Path,
    reference_npz: Path,
    out_mp4: Path,
    out_json: Path,
    video_space: str = "raw",
    preview_fps: float | None = None,
) -> dict[str, Any]:
    import cv2

    run_dir = Path(run_dir)
    cfg = read_yaml(run_dir / "run_config.yaml")
    trial = cfg.get("trial") or {}
    calibration = (cfg.get("manifest_summary") or {}).get("calibration") or {}
    camera_path = Path(str(calibration.get("intrinsics_extrinsics") or ""))
    if not camera_path.exists():
        raise FileNotFoundError(f"OpenCap camera calibration is unavailable: {camera_path}")

    reference = _load_reference(reference_npz)
    sync_video = Path(str(trial.get("video_sync") or cfg.get("raw_video") or ""))
    raw_video = Path(str(trial.get("video_raw") or sync_video))
    if not sync_video.exists():
        raise FileNotFoundError(f"Synced video is unavailable: {sync_video}")
    if video_space == "raw" and not raw_video.exists():
        raise FileNotFoundError(f"Raw video is unavailable: {raw_video}")

    sync_fps, sync_count, _, _ = video_metadata(sync_video)
    if sync_fps <= 0 or sync_count <= 0:
        raise RuntimeError(f"Could not read synced video metadata: {sync_video}")
    alignment = _resolve_video_alignment(raw_video, sync_video, video_space)
    source_video = raw_video if video_space == "raw" and alignment.get("status") != "skipped" else sync_video
    raw_offset = int(alignment.get("best_raw_offset") or 0) if source_video == raw_video else 0

    sync_indices = np.arange(int(sync_count), dtype=int)
    timestamps = sync_indices.astype(float) / float(sync_fps)
    joints = resample_timeseries(reference["time_s"], reference["joints_m"], timestamps)
    projected, valid = project_opensim_fk_to_pixels(joints, camera_path)

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {source_video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    out_fps = float(preview_fps or min(float(sync_fps), 30.0))
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open GT overlay writer: {out_mp4}")

    names = [str(name) for name in reference["joint_names"]]
    labels = [GT_LABELS.get(name, name.upper()) for name in names]
    written = 0
    try:
        for sync_idx, timestamp in zip(sync_indices.tolist(), timestamps.tolist()):
            source_frame_idx = raw_offset + int(sync_idx) if source_video == raw_video else int(sync_idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            _draw_labeled_gt_markers(frame, projected[int(sync_idx)], valid[int(sync_idx)], labels)
            header = (
                f"GT OpenSim FK labels only | {video_space} frame {source_frame_idx} | "
                f"sync frame {int(sync_idx)} | t={timestamp:.3f}s"
            )
            _draw_text_with_outline(frame, header, (20, 36), 0.72, (255, 255, 255), thickness=2)
            _draw_text_with_outline(frame, "left=red  right=blue  center=white", (20, 68), 0.60, (230, 230, 230), thickness=2)
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        cap.release()

    inside = _inside_frame(projected, valid, width, height)
    report = {
        "status": "ok" if written else "failed",
        "run_dir": str(run_dir),
        "reference_npz": str(reference_npz),
        "camera_calibration": str(camera_path),
        "video_space": video_space,
        "source_video": str(source_video),
        "sync_video": str(sync_video),
        "raw_video": str(raw_video) if raw_video else None,
        "raw_sync_alignment": alignment,
        "frames_requested": int(sync_count),
        "frames_written": int(written),
        "sync_fps": float(sync_fps),
        "preview_fps": float(out_fps),
        "frame_size": [int(width), int(height)],
        "joint_names": names,
        "labels": labels,
        "projected_valid_ratio": float(np.mean(valid)) if valid.size else 0.0,
        "inside_frame_ratio": float(np.mean(inside)) if inside.size else 0.0,
        "outputs": {"mp4": str(out_mp4), "qc": str(out_json)},
        "notes": [
            "GT points are mocap-derived OpenSim FK anatomical joint centers projected with the OpenCap camera calibration.",
            "This is not a raw optical marker overlay.",
            "Mocap/OpenSim data are resampled from the FK timebase to synced-video timestamps.",
        ],
    }
    if alignment.get("status") == "warning":
        report["status"] = "warning"
        report["warnings"] = list(alignment.get("warnings") or [])
    write_json(out_json, report)
    return report


def project_opensim_fk_to_pixels(joints_m: np.ndarray, camera_pickle: Path | str) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    camera_pickle = Path(camera_pickle)
    with camera_pickle.open("rb") as f:
        calibration = pickle.load(f)
    K = np.asarray(calibration["intrinsicMat"], dtype=float)
    R = np.asarray(calibration["rotation"], dtype=float)
    t = np.asarray(calibration["translation"], dtype=float).reshape(3)
    dist = np.asarray(calibration.get("distortion", np.zeros((1, 5))), dtype=float).reshape(-1)
    rvec, _ = cv2.Rodrigues(R)

    joints = np.asarray(joints_m, dtype=float)
    flat_m = joints.reshape(-1, 3)
    finite = np.isfinite(flat_m).all(axis=1)
    points_mm = np.where(finite[:, None], flat_m * 1000.0, 0.0).astype(np.float64)
    uv, _ = cv2.projectPoints(points_mm, rvec, t.astype(np.float64), K, dist)
    uv = uv.reshape(joints.shape[0], joints.shape[1], 2)
    camera_points = (R @ points_mm.T).T + t[None, :]
    depth = camera_points[:, 2].reshape(joints.shape[0], joints.shape[1])
    valid = finite.reshape(joints.shape[0], joints.shape[1]) & np.isfinite(uv).all(axis=2) & (depth > 0)
    uv[~valid] = np.nan
    return uv.astype(np.float32), valid


def _load_reference(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {
        "time_s": np.asarray(data["time_s"], dtype=float),
        "joints_m": np.asarray(data["joints_m"], dtype=float),
        "joint_names": [str(name) for name in data["joint_names"].tolist()],
    }


def _resolve_video_alignment(raw_video: Path, sync_video: Path, video_space: str) -> dict[str, Any]:
    if video_space == "sync":
        fps, count, width, height = video_metadata(sync_video)
        return {
            "status": "skipped",
            "reason": "Overlay requested in synced-video frame space.",
            "sync_video": str(sync_video),
            "sync_frame_count": int(count),
            "sync_fps": float(fps),
            "sync_width": int(width),
            "sync_height": int(height),
            "best_raw_offset": 0,
            "warnings": [],
        }
    if video_space != "raw":
        raise ValueError("video_space must be 'raw' or 'sync'.")
    return estimate_raw_sync_alignment(raw_video, sync_video)


def _draw_labeled_gt_markers(frame: np.ndarray, xy: np.ndarray, valid: np.ndarray, labels: list[str]) -> None:
    import cv2

    height, width = frame.shape[:2]
    for idx, label in enumerate(labels):
        if idx >= xy.shape[0] or not bool(valid[idx]):
            continue
        x, y = np.round(xy[idx]).astype(int).tolist()
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        color = _label_color_bgr(label)
        cv2.circle(frame, (x, y), 7, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, color, -1, cv2.LINE_AA)
        dx = -82 if label.startswith("R") else 10
        dy = -9 if "ANK" not in label and "MTP" not in label else 18
        tx = int(np.clip(x + dx, 4, max(4, width - 86)))
        ty = int(np.clip(y + dy, 18, max(18, height - 8)))
        cv2.line(frame, (x, y), (tx, ty), color, 1, cv2.LINE_AA)
        _draw_text_with_outline(frame, label, (tx, ty), 0.52, color, thickness=1)


def _draw_text_with_outline(frame: np.ndarray, text: str, origin: tuple[int, int], scale: float, color: tuple[int, int, int], thickness: int = 1) -> None:
    import cv2

    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _label_color_bgr(label: str) -> tuple[int, int, int]:
    if label.startswith("L"):
        return (70, 70, 255)
    if label.startswith("R"):
        return (255, 120, 60)
    return (255, 255, 255)


def _inside_frame(projected: np.ndarray, valid: np.ndarray, width: int, height: int) -> np.ndarray:
    uv = np.asarray(projected, dtype=float)
    return (
        np.asarray(valid, dtype=bool)
        & np.isfinite(uv).all(axis=2)
        & (uv[..., 0] >= 0)
        & (uv[..., 0] < float(width))
        & (uv[..., 1] >= 0)
        & (uv[..., 1] < float(height))
    )
