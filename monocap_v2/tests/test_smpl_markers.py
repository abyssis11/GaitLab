from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.mocap_eval import parse_trc
from monocap_v2.core.smpl_markers import apply_marker_time_window, extract_virtual_markers, load_marker_set, marker_qc
from monocap_v2.pipeline import stage_08_extract_virtual_markers, stage_09_export_trc, stage_12_visualize


def test_marker_map_loads_default_debug_set() -> None:
    marker_set = load_marker_set(Path("monocap_v2/configs/marker_map_smpl_to_opensim.yaml"))

    assert marker_set["name"] == "wham_smpl_debug_v1"
    assert marker_set["debug"] is True
    assert len(marker_set["markers"]) >= 12
    assert all(str(marker["name"]).startswith("DBG_") for marker in marker_set["markers"])


def test_extract_virtual_markers_resolves_explicit_and_nearest_vertices() -> None:
    marker_set = {
        "name": "unit_debug",
        "debug": True,
        "coordinate_space": "test_space",
        "markers": [
            {"name": "DBG_EXPLICIT", "vertex_index": 3},
            {"name": "DBG_LHIP", "nearest_vertex_to_joint": "lhip"},
        ],
    }
    vertices = np.zeros((2, 5, 3), dtype=np.float32)
    vertices[:, 3, :] = [3.0, 0.0, 0.0]
    vertices[:, 4, :] = [10.0, 0.0, 0.0]
    joints = np.zeros((2, 1, 3), dtype=np.float32)
    joints[:, 0, :] = [10.1, 0.0, 0.0]
    pose = {
        "backend": "wham",
        "fps": 60.0,
        "joint_names": ["lhip"],
        "joints_3d": joints,
        "smpl": {"vertices": vertices, "vertices_coordinate_space": "wham_camera_local"},
        "backend_meta": {"frame_ids": np.array([7, 8]), "raw_frame_ids": np.array([107, 108])},
    }

    payload = extract_virtual_markers(pose, marker_set)
    qc = marker_qc(payload)

    assert payload["markers_m"].shape == (2, 2, 3)
    assert payload["marker_names"] == ["DBG_EXPLICIT", "DBG_LHIP"]
    assert payload["vertex_indices"] == [3, 4]
    assert qc["status"] == "ok"


def test_stage_08_and_stage_09_write_markers_and_trc(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_standard_dirs()
    pose = _debug_pose()
    with registry.ensure_parent("pose3d_refined").open("wb") as f:
        pickle.dump(pose, f)

    first = stage_08_extract_virtual_markers.run(tmp_path, {}, force=False)
    cached = stage_08_extract_virtual_markers.run(tmp_path, {}, force=False)
    trc_result = stage_09_export_trc.run(tmp_path, {}, force=True)

    assert first["status"] == "ok"
    assert cached["status"] == "cached"
    assert registry.get("virtual_markers").exists()
    assert registry.get("virtual_markers_qc").exists()
    assert trc_result["status"] == "ok"
    assert registry.get("smpl_markers_trc").exists()
    assert registry.get("smpl_markers_qc").exists()

    trc = parse_trc(registry.get("smpl_markers_trc"))
    assert trc.units == "m"
    assert trc.data_rate == 60.0
    assert "DBG_PELV" in trc.marker_names
    assert np.allclose(trc.markers["DBG_PELV"][0], pose["smpl"]["vertices"][0, 0])


def test_marker_jump_diagnostics_and_visuals(tmp_path: Path) -> None:
    marker_path = tmp_path / "virtual_markers.pkl"
    markers = np.zeros((5, 3, 3), dtype=np.float32)
    markers[:, 0, :] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    markers[:, 1, :] = np.array([-0.2, 0.0, 0.0], dtype=np.float32)
    markers[:, 2, :] = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    markers[3:, :, :] += np.array([0.8, 0.0, 0.0], dtype=np.float32)
    payload = {
        "marker_set": "unit_debug",
        "debug": True,
        "coordinate_space": "wham_camera_local",
        "fps": 60.0,
        "units": "m",
        "marker_names": ["DBG_PELV", "DBG_LHIP", "DBG_RHIP"],
        "markers_m": markers,
        "raw_frame_ids": np.arange(10, 15, dtype=np.int64),
        "raw_video_time_s": np.arange(10, 15, dtype=np.float32) / 60.0,
    }
    with marker_path.open("wb") as f:
        pickle.dump(payload, f)

    diagnostics = stage_12_visualize._marker_jump_diagnostics(marker_path)
    plot_path = tmp_path / "jump.png"
    preview_path = tmp_path / "jump.mp4"
    stage_12_visualize._write_marker_jump_plot(marker_path, diagnostics, plot_path)
    stage_12_visualize._write_marker_jump_preview(
        marker_path,
        diagnostics,
        preview_path,
        {"preview_width": 320, "preview_height": 240, "marker_jump_preview_fps": 4},
    )

    assert diagnostics["status"] == "warning"
    assert diagnostics["classification"] == "whole_body_translation_or_root_jump"
    assert diagnostics["top_jumps"][0]["from_frame"] == 2
    assert diagnostics["top_jumps"][0]["raw_frame_from"] == 12
    assert plot_path.exists() and plot_path.stat().st_size > 0
    assert preview_path.exists() and preview_path.stat().st_size > 0


def test_marker_time_window_uses_wham_sync_overlap(tmp_path: Path) -> None:
    raw_frames = [_pattern_frame(i, (40, 52)) for i in range(10)]
    raw = _write_video(tmp_path / "raw.mp4", raw_frames)
    sync = _write_video(tmp_path / "sync.mp4", raw_frames[4:8])
    markers = np.zeros((8, 2, 3), dtype=np.float32)
    markers[:, :, 0] = np.arange(8, dtype=np.float32)[:, None]
    payload = {
        "marker_set": "unit_debug",
        "debug": True,
        "coordinate_space": "wham_camera_local",
        "fps": 10.0,
        "units": "m",
        "marker_names": ["DBG_PELV", "DBG_HEAD"],
        "markers_m": markers,
        "raw_frame_ids": np.arange(2, 10, dtype=np.int64),
        "frame_ids": np.arange(2, 10, dtype=np.int64),
        "raw_video_time_s": np.arange(2, 10, dtype=np.float32) / 10.0,
        "wham_relative_time_s": np.arange(8, dtype=np.float32) / 10.0,
    }
    pose = {
        "backend": "wham",
        "representation": "hybrid",
        "fps": 10.0,
        "backend_meta": {
            "source_video": str(raw),
            "inference_video": str(raw),
            "source_frame_count": 10,
            "inference_frame_count": 10,
            "selected_track_id": "1",
            "frame_ids": np.arange(2, 10, dtype=np.int64),
            "start_frame": 0,
        },
    }

    trimmed, report = apply_marker_time_window(payload, pose, {"trial": {"video_raw": str(raw), "video_sync": str(sync)}, "config": {}})

    assert report["status"] == "applied"
    assert report["raw_frame_start"] == 4
    assert report["raw_frame_end"] == 7
    assert report["kept_frames"] == 4
    assert report["dropped_before"] == 2
    assert report["dropped_after"] == 2
    assert trimmed["markers_m"].shape[0] == 4
    assert np.asarray(trimmed["raw_frame_ids"]).tolist() == [4, 5, 6, 7]
    assert np.allclose(trimmed["wham_relative_time_s"], [0.0, 0.1, 0.2, 0.3])


def test_smpl_vertices_preview_uses_marker_time_window(tmp_path: Path) -> None:
    pose = _debug_pose()
    vertices = np.repeat(pose["smpl"]["vertices"], 4, axis=1)
    pose["smpl"]["vertices"] = vertices
    pose["backend_meta"]["frame_ids"] = np.array([210, 211, 212], dtype=np.int64)
    marker_payload = {
        "marker_names": ["DBG_PELV"],
        "markers_m": vertices[1:2, :1, :],
        "time_window": {
            "status": "applied",
            "raw_frame_start": 211,
            "raw_frame_end": 211,
        },
    }
    marker_path = tmp_path / "markers.pkl"
    with marker_path.open("wb") as f:
        pickle.dump(marker_payload, f)

    out_path = tmp_path / "smpl.mp4"
    report = stage_12_visualize._write_smpl_vertices_preview(
        pose,
        out_path,
        {"preview_width": 320, "preview_height": 240, "smpl_preview_fps": 4, "smpl_preview_max_vertices": 20},
        marker_path,
    )

    assert report["status"] == "ok"
    assert report["render_mode"] == "vertex_cloud"
    assert report["frames_rendered"] == 1
    assert report["time_window"]["status"] == "applied"
    assert out_path.exists() and out_path.stat().st_size > 0


def _debug_pose() -> dict:
    joint_names = [
        "pelv",
        "spi1",
        "spi3",
        "neck",
        "head",
        "lhip",
        "rhip",
        "lkne",
        "rkne",
        "lank",
        "rank",
        "left_heel",
        "right_heel",
        "left_big_toe",
        "right_big_toe",
        "left_toe",
        "right_toe",
        "lsho",
        "rsho",
        "lelb",
        "relb",
        "lwri",
        "rwri",
    ]
    t = 3
    joints = np.zeros((t, len(joint_names), 3), dtype=np.float32)
    for idx in range(len(joint_names)):
        joints[:, idx, :] = np.array([idx * 0.02, idx * 0.01, idx * 0.03], dtype=np.float32)
        joints[:, idx, 0] += np.arange(t, dtype=np.float32) * 0.001
    vertices = joints.copy()
    return {
        "representation": "hybrid",
        "backend": "wham",
        "fps": 60.0,
        "units": "m",
        "joint_names": joint_names,
        "joints_3d": joints,
        "smpl": {"vertices": vertices, "vertices_coordinate_space": "wham_camera_local"},
        "backend_meta": {
            "frame_ids": np.array([210, 211, 212], dtype=np.int64),
            "raw_frame_ids": np.array([210, 211, 212], dtype=np.int64),
            "raw_video_time_s": np.array([3.5, 3.5166667, 3.5333333], dtype=np.float32),
            "wham_relative_time_s": np.array([0.0, 1.0 / 60.0, 2.0 / 60.0], dtype=np.float32),
        },
    }


def _write_video(path: Path, frames: list[np.ndarray]) -> Path:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for frame in frames:
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
