from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from monocap_v2.core.smpl_mesh import load_smpl_faces, validate_smpl_faces
from monocap_v2.pipeline import stage_12_visualize


def test_load_smpl_faces_from_synthetic_pickle(tmp_path: Path) -> None:
    model = tmp_path / "model.pkl"
    faces = np.asarray([[0, 1, 2], [2, 3, 0]], dtype=np.uint32)
    with model.open("wb") as f:
        pickle.dump({"f": faces}, f)
    cfg = {
        "repo_root": str(tmp_path),
        "config": {
            "smpl": {
                "model_dir": str(tmp_path),
                "gender": "neutral",
                "file_map": {"neutral": "model.pkl"},
            }
        },
    }

    info = load_smpl_faces(cfg, vertex_count=4)

    assert info["source_type"] == "smpl_pickle_f"
    assert np.array_equal(info["faces"], faces.astype(np.int32))


def test_validate_smpl_faces_rejects_bad_shape_and_out_of_range() -> None:
    try:
        validate_smpl_faces(np.asarray([0, 1, 2]), vertex_count=3)
    except ValueError as exc:
        assert "shape" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected invalid shape to fail.")

    try:
        validate_smpl_faces(np.asarray([[0, 1, 5]], dtype=np.int32), vertex_count=5)
    except ValueError as exc:
        assert "only 5 vertices" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected out-of-range face to fail.")


def test_smpl_mesh_preview_writes_video_and_aligns_markers(tmp_path: Path) -> None:
    faces_path = tmp_path / "faces.npy"
    np.save(faces_path, np.asarray([[0, 1, 2], [0, 2, 3], [0, 3, 4], [1, 3, 4]], dtype=np.int32))
    pose = _mesh_pose()
    marker_path = tmp_path / "markers.pkl"
    marker_payload = {
        "marker_names": ["DBG_PELV", "DBG_LHIP"],
        "markers_m": pose["smpl"]["vertices"][1:2, :2, :],
        "raw_frame_ids": np.asarray([11], dtype=np.int64),
        "time_window": {"status": "applied", "raw_frame_start": 11, "raw_frame_end": 11},
    }
    with marker_path.open("wb") as f:
        pickle.dump(marker_payload, f)
    cfg = {
        "repo_root": str(tmp_path),
        "config": {
            "smpl": {"faces_path": str(faces_path)},
            "visualization": {
                "preview_width": 320,
                "preview_height": 240,
                "preview_fps": 4,
                "smpl_mesh": {"enabled": True, "max_faces": 3, "show_markers": True, "show_marker_labels": True},
            },
        },
    }

    out_path = tmp_path / "mesh.mp4"
    report = stage_12_visualize._write_smpl_mesh_preview(pose, out_path, cfg, marker_path)

    assert report["status"] == "ok"
    assert report["render_mode"] == "mesh"
    assert report["face_source_type"] == "faces_path"
    assert report["frames_rendered"] == 1
    assert report["faces_rendered"] == 3
    assert report["marker_overlay_status"] == "ok"
    assert report["marker_frames_aligned"] == 1
    assert out_path.exists() and out_path.stat().st_size > 0


def _mesh_pose() -> dict:
    vertices = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.1, 0.0, 0.3], [0.0, 0.2, 0.1], [0.2, 0.2, 0.1]],
            [[0.1, 0.0, 0.0], [0.3, 0.0, 0.0], [0.2, 0.0, 0.3], [0.1, 0.2, 0.1], [0.3, 0.2, 0.1]],
            [[0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [0.3, 0.0, 0.3], [0.2, 0.2, 0.1], [0.4, 0.2, 0.1]],
        ],
        dtype=np.float32,
    )
    return {
        "representation": "hybrid",
        "backend": "wham",
        "fps": 10.0,
        "units": "m",
        "joint_names": ["pelv", "lhip"],
        "joints_3d": vertices[:, :2, :],
        "smpl": {"vertices": vertices, "vertices_coordinate_space": "wham_camera_local"},
        "backend_meta": {"frame_ids": np.asarray([10, 11, 12], dtype=np.int64), "start_frame": 0},
    }
