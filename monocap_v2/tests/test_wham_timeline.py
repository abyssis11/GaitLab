from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from monocap_v2.core.wham_timeline import build_wham_timeline_report, estimate_raw_sync_alignment, wham_timebase_report
from monocap_v2.pipeline import stage_12_visualize


def test_raw_sync_matching_recovers_exact_subclip_offset(tmp_path: Path) -> None:
    frames = [_pattern_frame(i, (48, 64)) for i in range(18)]
    raw = _write_video(tmp_path / "raw.mp4", frames)
    sync = _write_video(tmp_path / "sync.mp4", frames[5:11])

    report = estimate_raw_sync_alignment(raw, sync, max_samples=4, match_threshold=0.80)

    assert report["status"] == "ok"
    assert report["best_raw_offset"] == 5
    assert report["best_match_score"] > 0.90


def test_raw_sync_matching_warns_when_no_direct_match(tmp_path: Path) -> None:
    raw = _write_video(tmp_path / "raw.mp4", [_pattern_frame(i, (48, 64)) for i in range(12)])
    sync = _write_video(tmp_path / "sync.mp4", [np.full((48, 64, 3), 255, dtype=np.uint8) for _ in range(4)])

    report = estimate_raw_sync_alignment(raw, sync, max_samples=3, match_threshold=0.98)

    assert report["status"] == "warning"
    assert report["best_match_score"] < 0.98
    assert report["warnings"]


def test_raw_sync_matching_handles_different_resolutions(tmp_path: Path) -> None:
    frames = [_pattern_frame(i, (48, 64)) for i in range(14)]
    raw = _write_video(tmp_path / "raw.mp4", frames)
    sync = _write_video(tmp_path / "sync.mp4", [cv2.resize(frame, (32, 24)) for frame in frames[4:9]], size=(32, 24))

    report = estimate_raw_sync_alignment(raw, sync, max_samples=4, match_threshold=0.80)

    assert report["status"] == "ok"
    assert report["best_raw_offset"] == 4


def test_wham_timebase_metadata_and_noncontiguous_warning() -> None:
    pose = _wham_pose(frame_ids=[10, 12, 13])

    report = wham_timebase_report(pose)

    assert report["status"] == "warning"
    assert report["time_start_s"] == 1.0
    assert report["time_end_s"] == 1.3
    assert "not contiguous" in report["warnings"][0]


def test_wham_timeline_skips_non_wham_pose() -> None:
    report = build_wham_timeline_report({"backend": "metrabs"})

    assert report["status"] == "skipped"


def test_wham_overlay_and_timeline_plot_are_written(tmp_path: Path) -> None:
    raw = _write_video(tmp_path / "raw.mp4", [_pattern_frame(i, (64, 80)) for i in range(5)], size=(80, 64))
    pose = _wham_pose(frame_ids=[1, 2, 3], source_video=raw)
    pose["pose2d"] = {
        "xy": np.asarray(
            [
                [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]],
                [[12.0, 10.0], [22.0, 20.0], [32.0, 30.0]],
                [[14.0, 10.0], [24.0, 20.0], [34.0, 30.0]],
            ],
            dtype=np.float32,
        ),
        "confidence": np.ones((3, 3), dtype=np.float32),
        "names": ["a", "b", "c"],
        "fps": 10.0,
        "backend": "wham",
    }
    timeline = build_wham_timeline_report(pose, {"trial": {"video_raw": str(raw), "video_sync": str(raw)}})

    overlay = tmp_path / "overlay.mp4"
    labeled_overlay = tmp_path / "labeled_overlay.mp4"
    plot = tmp_path / "timeline.png"
    stage_12_visualize._write_wham_raw_overlay(pose, overlay, {"preview_fps": 10})
    stage_12_visualize._write_wham_labeled_overlay(pose, labeled_overlay, {"preview_fps": 10})
    stage_12_visualize._write_wham_timeline_plot(timeline, plot)

    assert overlay.exists()
    assert overlay.stat().st_size > 0
    assert labeled_overlay.exists()
    assert labeled_overlay.stat().st_size > 0
    assert plot.exists()
    assert plot.stat().st_size > 0


def _wham_pose(frame_ids: list[int], source_video: Path | None = None) -> dict:
    return {
        "backend": "wham",
        "representation": "hybrid",
        "fps": 10.0,
        "joint_names": ["pelv"],
        "joints_3d": np.zeros((len(frame_ids), 1, 3), dtype=np.float32),
        "backend_meta": {
            "source_video": str(source_video) if source_video else "raw.mp4",
            "inference_video": str(source_video) if source_video else "raw.mp4",
            "source_frame_count": 20,
            "inference_frame_count": 20,
            "selected_track_id": "1",
            "frame_ids": np.asarray(frame_ids, dtype=np.int64),
            "local_only": True,
            "dpvo_available": False,
            "start_frame": 0,
        },
    }


def _write_video(path: Path, frames: list[np.ndarray], size: tuple[int, int] | None = None) -> Path:
    if size is None:
        h, w = frames[0].shape[:2]
        size = (w, h)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    for frame in frames:
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size)
        writer.write(frame)
    writer.release()
    return path


def _pattern_frame(idx: int, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    y, x = np.indices((h, w))
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[..., 0] = (x * 3 + idx * 17) % 255
    frame[..., 1] = (y * 5 + idx * 29) % 255
    frame[..., 2] = ((x + y) * 2 + idx * 41) % 255
    return frame
