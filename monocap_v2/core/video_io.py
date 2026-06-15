from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


CaptureFactory = Callable[[str], Any]


@dataclass(frozen=True)
class VideoFrame:
    index: int
    bgr: np.ndarray


def video_metadata(video_path: Path | str, capture_factory: CaptureFactory | None = None) -> dict[str, Any]:
    cap = _open_capture(video_path, capture_factory)
    try:
        return _metadata_from_capture(video_path, cap)
    finally:
        cap.release()


def read_video_frames_bgr(
    video_path: Path | str,
    max_frames: int | None = None,
    recover_tail: bool = True,
    capture_factory: CaptureFactory | None = None,
) -> tuple[list[VideoFrame], dict[str, Any]]:
    """Read frames, recovering random-access-readable tail frames after early EOF."""
    meta_cap = _open_capture(video_path, capture_factory)
    try:
        meta = _metadata_from_capture(video_path, meta_cap)
    finally:
        meta_cap.release()

    metadata_count = int(meta.get("metadata_frame_count") or 0)
    requested = _requested_count(metadata_count, max_frames)
    frames: list[VideoFrame] = []
    sequential_indices: list[int] = []

    cap = _open_capture(video_path, capture_factory)
    try:
        idx = 0
        while requested is None or idx < requested:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(VideoFrame(index=idx, bgr=np.asarray(frame)))
            sequential_indices.append(idx)
            idx += 1
    finally:
        cap.release()

    recovered_indices: list[int] = []
    if recover_tail and requested is not None and len(sequential_indices) < requested:
        seen = {frame.index for frame in frames}
        cap = _open_capture(video_path, capture_factory)
        try:
            for idx in range(len(sequential_indices), requested):
                if idx in seen:
                    continue
                cap.set(_cv2_prop("CAP_PROP_POS_FRAMES"), idx)
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(VideoFrame(index=idx, bgr=np.asarray(frame)))
                    recovered_indices.append(idx)
                    seen.add(idx)
        finally:
            cap.release()

    frames.sort(key=lambda item: item.index)
    expected_indices = list(range(requested if requested is not None else len(frames)))
    present = {frame.index for frame in frames}
    missing = [idx for idx in expected_indices if idx not in present]
    report = {
        **meta,
        "requested_frame_count": requested,
        "sequential_decoded_frame_count": len(sequential_indices),
        "sequential_frame_indices": sequential_indices,
        "random_access_recovered_count": len(recovered_indices),
        "random_access_recovered_indices": recovered_indices,
        "random_access_frame_count": len(sequential_indices) + len(recovered_indices),
        "usable_frame_count": len(frames),
        "missing_frame_indices": missing,
        "early_sequential_eof": requested is not None and len(sequential_indices) < requested,
        "recovered_tail": bool(recovered_indices),
        "complete": len(missing) == 0,
    }
    return frames, report


def probe_video_frames(
    video_path: Path | str,
    max_frames: int | None = None,
    recover_tail: bool = True,
    capture_factory: CaptureFactory | None = None,
) -> dict[str, Any]:
    _frames, report = read_video_frames_bgr(video_path, max_frames=max_frames, recover_tail=recover_tail, capture_factory=capture_factory)
    return report


def write_video_frames_bgr(frames: list[VideoFrame], target: Path | str, fps: float, size: tuple[int, int] | None = None) -> dict[str, Any]:
    import cv2

    target = Path(target)
    if not frames:
        raise RuntimeError(f"No frames available to write video: {target}")
    first = frames[0].bgr
    height, width = first.shape[:2]
    if size is not None:
        width, height = int(size[0]), int(size[1])
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), float(fps or 30.0), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {target}")
    try:
        for item in frames:
            frame = np.asarray(item.bgr)
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    return {"path": str(target), "frames": len(frames), "fps": float(fps or 30.0), "width": int(width), "height": int(height)}


def _requested_count(metadata_count: int, max_frames: int | None) -> int | None:
    if max_frames is not None and max_frames > 0:
        return min(int(max_frames), metadata_count) if metadata_count > 0 else int(max_frames)
    return metadata_count if metadata_count > 0 else None


def _open_capture(video_path: Path | str, capture_factory: CaptureFactory | None):
    if capture_factory is not None:
        cap = capture_factory(str(video_path))
    else:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return cap


def _metadata_from_capture(video_path: Path | str, cap) -> dict[str, Any]:
    width = int(cap.get(_cv2_prop("CAP_PROP_FRAME_WIDTH")) or 0)
    height = int(cap.get(_cv2_prop("CAP_PROP_FRAME_HEIGHT")) or 0)
    fps = float(cap.get(_cv2_prop("CAP_PROP_FPS")) or 0.0)
    frame_count = int(cap.get(_cv2_prop("CAP_PROP_FRAME_COUNT")) or 0)
    return {
        "source": str(video_path),
        "width": width,
        "height": height,
        "fps": fps,
        "metadata_frame_count": frame_count,
        "metadata_duration_sec": float(frame_count / fps) if fps > 0 and frame_count > 0 else 0.0,
    }


def _cv2_prop(name: str) -> int:
    try:
        import cv2

        return int(getattr(cv2, name))
    except Exception:
        fallback = {
            "CAP_PROP_FRAME_WIDTH": 3,
            "CAP_PROP_FRAME_HEIGHT": 4,
            "CAP_PROP_FPS": 5,
            "CAP_PROP_POS_FRAMES": 1,
            "CAP_PROP_FRAME_COUNT": 7,
        }
        return fallback[name]
