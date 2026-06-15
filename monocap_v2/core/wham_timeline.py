from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def wham_timebase_report(pose: dict, cfg: dict | None = None) -> dict[str, Any]:
    if pose.get("backend") != "wham":
        return {"status": "skipped", "reason": "Pose artifact is not from WHAM."}

    meta = pose.get("backend_meta") or {}
    frame_ids = np.asarray(meta.get("frame_ids", []), dtype=int).reshape(-1)
    if frame_ids.size == 0:
        return {"status": "skipped", "reason": "WHAM frame_ids are missing."}

    fps = float(pose.get("fps") or meta.get("fps") or 30.0)
    start_frame = int(meta.get("start_frame") or 0)
    raw_frame_ids = frame_ids + start_frame
    source_frame_count = _int_or_none(meta.get("source_frame_count"))
    diffs = np.diff(raw_frame_ids)
    warnings: list[str] = []
    if diffs.size and not np.all(diffs == 1):
        warnings.append("WHAM frame_ids are not contiguous.")
    if source_frame_count and raw_frame_ids[-1] >= source_frame_count:
        warnings.append("WHAM frame_ids extend beyond the reported raw source video frame count.")

    coverage_ratio = float(len(np.unique(raw_frame_ids)) / source_frame_count) if source_frame_count else None
    span_ratio = float((raw_frame_ids[-1] - raw_frame_ids[0] + 1) / source_frame_count) if source_frame_count else None
    return {
        "status": "warning" if warnings else "ok",
        "backend": "wham",
        "source_video": meta.get("source_video") or pose.get("source_video"),
        "inference_video": meta.get("inference_video"),
        "fps": fps,
        "source_frame_count": source_frame_count,
        "inference_frame_count": _int_or_none(meta.get("inference_frame_count")),
        "selected_track_id": meta.get("selected_track_id"),
        "num_output_frames": int(frame_ids.size),
        "frame_ids": raw_frame_ids.astype(int).tolist(),
        "inference_frame_ids": frame_ids.astype(int).tolist(),
        "frame_start": int(raw_frame_ids[0]),
        "frame_end": int(raw_frame_ids[-1]),
        "time_start_s": float(raw_frame_ids[0] / fps),
        "time_end_s": float(raw_frame_ids[-1] / fps),
        "wham_relative_time_start_s": 0.0,
        "wham_relative_time_end_s": float((raw_frame_ids[-1] - raw_frame_ids[0]) / fps),
        "coverage_ratio": coverage_ratio,
        "span_ratio": span_ratio,
        "local_only": bool(meta.get("local_only", True)),
        "dpvo_available": bool(meta.get("dpvo_available", False)),
        "warnings": warnings,
    }


def build_wham_timeline_report(pose: dict, cfg: dict | None = None) -> dict[str, Any]:
    timebase = wham_timebase_report(pose, cfg)
    if timebase.get("status") == "skipped":
        return {"status": "skipped", "timebase": timebase, "reason": timebase.get("reason")}

    raw_video, sync_video = _timeline_video_paths(pose, cfg or {})
    alignment = estimate_raw_sync_alignment(raw_video, sync_video) if raw_video and sync_video else {
        "status": "skipped",
        "reason": "Missing raw or synced video path.",
        "raw_video": str(raw_video) if raw_video else None,
        "sync_video": str(sync_video) if sync_video else None,
    }

    warnings = list(timebase.get("warnings") or [])
    warnings.extend(alignment.get("warnings") or [])
    overlap = _wham_sync_overlap(timebase, alignment)
    if overlap.get("status") == "warning":
        warnings.extend(overlap.get("warnings") or [])

    status = "ok"
    if timebase.get("status") == "warning" or alignment.get("status") == "warning" or overlap.get("status") == "warning":
        status = "warning"
    if alignment.get("status") == "skipped":
        status = "warning"

    return {
        "status": status,
        "timebase": timebase,
        "raw_sync_alignment": alignment,
        "overlap": overlap,
        "warnings": warnings,
    }


def estimate_raw_sync_alignment(
    raw_video: Path | str,
    sync_video: Path | str,
    max_samples: int = 12,
    match_threshold: float = 0.90,
    resize: tuple[int, int] = (64, 64),
) -> dict[str, Any]:
    raw_path = Path(raw_video)
    sync_path = Path(sync_video)
    if raw_path.resolve() == sync_path.resolve():
        fps, count, width, height = video_metadata(raw_path)
        return {
            "status": "ok",
            "raw_video": str(raw_path),
            "sync_video": str(sync_path),
            "raw_frame_count": count,
            "sync_frame_count": count,
            "raw_fps": fps,
            "sync_fps": fps,
            "raw_width": width,
            "raw_height": height,
            "sync_width": width,
            "sync_height": height,
            "best_raw_offset": 0,
            "best_match_score": 1.0,
            "score_metric": "1 - mean_abs_pixel_error/255",
            "sample_indices": [0],
            "warnings": [],
        }

    raw_fps, raw_count, raw_width, raw_height = video_metadata(raw_path)
    sync_fps, sync_count, sync_width, sync_height = video_metadata(sync_path)
    if raw_count <= 0 or sync_count <= 0:
        return _alignment_warning(raw_path, sync_path, raw_count, sync_count, "Could not read video frame counts.")
    if raw_count < sync_count:
        return _alignment_warning(raw_path, sync_path, raw_count, sync_count, "Raw video is shorter than synced video.")

    sample_indices = _sample_indices(sync_count, max_samples)
    sync_frames = _read_resized_frames(sync_path, sample_indices, resize)
    if len(sync_frames) != len(sample_indices):
        return _alignment_warning(raw_path, sync_path, raw_count, sync_count, "Could not read synced-video sample frames.")

    best_offset = 0
    best_score = -np.inf
    max_offset = raw_count - sync_count
    for offset in range(max_offset + 1):
        raw_frames = _read_resized_frames(raw_path, [offset + idx for idx in sample_indices], resize)
        if len(raw_frames) != len(sample_indices):
            continue
        score = _mean_frame_similarity(raw_frames, sync_frames)
        if score > best_score:
            best_score = score
            best_offset = offset

    warnings: list[str] = []
    status = "ok"
    if best_score < match_threshold:
        status = "warning"
        warnings.append(f"Best raw/sync visual match score {best_score:.3f} is below threshold {match_threshold:.3f}.")
    if abs(raw_fps - sync_fps) > 1e-3:
        warnings.append(f"Raw FPS {raw_fps:.3f} differs from synced FPS {sync_fps:.3f}.")
    return {
        "status": status,
        "raw_video": str(raw_path),
        "sync_video": str(sync_path),
        "raw_frame_count": int(raw_count),
        "sync_frame_count": int(sync_count),
        "raw_fps": float(raw_fps),
        "sync_fps": float(sync_fps),
        "raw_width": int(raw_width),
        "raw_height": int(raw_height),
        "sync_width": int(sync_width),
        "sync_height": int(sync_height),
        "best_raw_offset": int(best_offset),
        "best_match_score": float(best_score),
        "match_threshold": float(match_threshold),
        "score_metric": "1 - mean_abs_pixel_error/255",
        "sample_indices": sample_indices,
        "warnings": warnings,
    }


def video_metadata(path: Path | str) -> tuple[float, int, int, int]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return 0.0, 0, 0, 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return fps, count, width, height
    finally:
        cap.release()


def _timeline_video_paths(pose: dict, cfg: dict) -> tuple[Path | None, Path | None]:
    trial = cfg.get("trial") or {}
    meta = pose.get("backend_meta") or {}
    raw = meta.get("source_video") or pose.get("source_video") or trial.get("video_raw")
    sync = trial.get("video_sync") or cfg.get("raw_video")
    raw_path = Path(str(raw)) if raw else None
    sync_path = Path(str(sync)) if sync else None
    return raw_path, sync_path


def _wham_sync_overlap(timebase: dict, alignment: dict) -> dict[str, Any]:
    if alignment.get("status") == "skipped" or alignment.get("best_raw_offset") is None:
        return {"status": "skipped", "reason": "No raw/sync alignment available."}
    sync_start = int(alignment["best_raw_offset"])
    sync_count = int(alignment.get("sync_frame_count") or 0)
    sync_end = sync_start + sync_count - 1
    wham_start = int(timebase["frame_start"])
    wham_end = int(timebase["frame_end"])
    overlap_start = max(sync_start, wham_start)
    overlap_end = min(sync_end, wham_end)
    overlap_frames = max(0, overlap_end - overlap_start + 1)
    wham_span = max(1, wham_end - wham_start + 1)
    sync_span = max(1, sync_count)
    warnings: list[str] = []
    if overlap_frames == 0:
        warnings.append("WHAM selected segment does not overlap the estimated synced-video window.")
    return {
        "status": "warning" if warnings else "ok",
        "sync_raw_frame_start": sync_start,
        "sync_raw_frame_end": sync_end,
        "wham_frame_start": wham_start,
        "wham_frame_end": wham_end,
        "overlap_frame_start": overlap_start if overlap_frames else None,
        "overlap_frame_end": overlap_end if overlap_frames else None,
        "overlap_frames": int(overlap_frames),
        "overlap_ratio_of_wham_span": float(overlap_frames / wham_span),
        "overlap_ratio_of_sync_span": float(overlap_frames / sync_span),
        "warnings": warnings,
    }


def _sample_indices(frame_count: int, max_samples: int) -> list[int]:
    n = max(1, min(int(max_samples), int(frame_count)))
    return sorted(set(int(round(v)) for v in np.linspace(0, frame_count - 1, n)))


def _read_resized_frames(path: Path, indices: list[int], size: tuple[int, int]) -> list[np.ndarray]:
    import cv2

    frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return frames
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                return frames
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32))
    finally:
        cap.release()
    return frames


def _mean_frame_similarity(a_frames: list[np.ndarray], b_frames: list[np.ndarray]) -> float:
    scores = []
    for a, b in zip(a_frames, b_frames):
        scores.append(1.0 - float(np.mean(np.abs(a - b))) / 255.0)
    return float(np.mean(scores)) if scores else 0.0


def _alignment_warning(raw_path: Path, sync_path: Path, raw_count: int, sync_count: int, reason: str) -> dict[str, Any]:
    return {
        "status": "warning",
        "raw_video": str(raw_path),
        "sync_video": str(sync_path),
        "raw_frame_count": int(raw_count),
        "sync_frame_count": int(sync_count),
        "best_raw_offset": None,
        "best_match_score": None,
        "warnings": [reason],
    }


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None
