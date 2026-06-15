from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from monocap_v2.core.smpl_marker_placement import LOWER_LIMB_MARKERS, build_lower_limb_marker_map_proposal, lower_limb_marker_placement_qc
from monocap_v2.pipeline import stage_12_visualize


def test_lower_limb_marker_placement_qc_detects_good_and_bad_markers() -> None:
    pose, payload = _placement_pose_and_markers()
    good = lower_limb_marker_placement_qc(pose, payload, {})

    bad_payload = dict(payload)
    bad_markers = np.asarray(payload["markers_m"]).copy()
    ltoe_idx = payload["marker_names"].index("DBG_LTOE")
    bad_markers[:, ltoe_idx, :] += np.asarray([0.8, 0.0, 0.0], dtype=np.float32)
    bad_payload["markers_m"] = bad_markers
    bad = lower_limb_marker_placement_qc(pose, bad_payload, {})

    assert good["status"] == "ok"
    assert good["frames"] == 3
    assert good["per_marker"]["DBG_LANK"]["distance_to_source_joint_m"]["median"] < 0.02
    assert bad["status"] == "warning"
    assert any("DBG_LTOE" in warning for warning in bad["warnings"])


def test_lower_limb_marker_map_proposal_uses_explicit_vertex_indices() -> None:
    _, payload = _placement_pose_and_markers()

    proposal = build_lower_limb_marker_map_proposal(payload)
    marker_set = proposal["marker_sets"]["wham_smpl_debug_v2"]
    markers = marker_set["markers"]

    assert [marker["name"] for marker in markers] == LOWER_LIMB_MARKERS
    assert all("vertex_index" in marker for marker in markers)
    assert all("nearest_vertex_to_joint" not in marker for marker in markers)
    assert markers[0]["vertex_index"] == payload["vertex_indices"][0]


def test_marker_placement_plot_and_preview_are_written(tmp_path: Path) -> None:
    pose, payload = _placement_pose_and_markers()
    marker_path = tmp_path / "markers.pkl"
    with marker_path.open("wb") as f:
        pickle.dump(payload, f)
    faces_path = tmp_path / "faces.npy"
    np.save(faces_path, np.asarray([[0, 1, 2], [1, 3, 5], [2, 4, 6], [7, 9, 10], [8, 9, 10]], dtype=np.int32))

    report = lower_limb_marker_placement_qc(pose, payload, {})
    plot_path = tmp_path / "placement.png"
    preview_path = tmp_path / "placement.mp4"
    stage_12_visualize._write_marker_placement_plot(report, plot_path)
    mesh_report = stage_12_visualize._write_smpl_mesh_preview(
        pose,
        preview_path,
        {
            "repo_root": str(tmp_path),
            "config": {
                "smpl": {"faces_path": str(faces_path)},
                "visualization": {
                    "preview_width": 320,
                    "preview_height": 240,
                    "preview_fps": 4,
                    "smpl_mesh": {"max_faces": 5, "show_markers": True, "show_marker_labels": True},
                },
            },
        },
        marker_path,
        set(LOWER_LIMB_MARKERS),
    )

    assert plot_path.exists() and plot_path.stat().st_size > 0
    assert preview_path.exists() and preview_path.stat().st_size > 0
    assert mesh_report["render_mode"] == "mesh"
    assert mesh_report["marker_overlay_status"] == "ok"


def _placement_pose_and_markers() -> tuple[dict, dict]:
    names = [
        "pelv",
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
    ]
    base = np.asarray(
        [
            [0.0, -1.00, 0.00],
            [-0.12, -0.88, 0.00],
            [0.12, -0.88, 0.00],
            [-0.12, -0.50, 0.00],
            [0.12, -0.50, 0.00],
            [-0.12, -0.10, 0.00],
            [0.12, -0.10, 0.00],
            [-0.12, -0.05, -0.12],
            [0.12, -0.05, -0.12],
            [-0.12, -0.04, 0.20],
            [0.12, -0.04, 0.20],
        ],
        dtype=np.float32,
    )
    joints = np.stack([base + np.asarray([idx * 0.01, 0.0, 0.0], dtype=np.float32) for idx in range(3)], axis=0)
    vertices = joints.copy()
    marker_names = LOWER_LIMB_MARKERS.copy()
    markers = vertices.copy()
    pose = {
        "backend": "wham",
        "representation": "hybrid",
        "fps": 60.0,
        "units": "m",
        "joint_names": names,
        "joints_3d": joints,
        "smpl": {"vertices": vertices, "vertices_coordinate_space": "wham_camera_local"},
        "backend_meta": {"frame_ids": np.asarray([10, 11, 12], dtype=np.int64), "start_frame": 0},
    }
    payload = {
        "marker_set": "unit_debug",
        "debug": True,
        "coordinate_space": "wham_camera_local",
        "fps": 60.0,
        "units": "m",
        "marker_names": marker_names,
        "markers_m": markers,
        "vertex_indices": list(range(len(marker_names))),
        "resolved_from": {marker: {"type": "nearest_vertex_to_joint", "value": joint} for marker, joint in zip(marker_names, names)},
        "raw_frame_ids": np.asarray([10, 11, 12], dtype=np.int64),
        "time_window": {"status": "applied", "raw_frame_start": 10, "raw_frame_end": 12},
    }
    return pose, payload
