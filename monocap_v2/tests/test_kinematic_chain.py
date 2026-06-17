from __future__ import annotations

import numpy as np

from monocap_v2.core.kinematic_chain import apply_kinematic_chain_to_pose, build_chain_edges
from monocap_v2.core.temporal_smoothing import EdgeLength, temporal_smoothing_metrics


def test_kinematic_chain_reduces_direction_jitter_and_preserves_lengths() -> None:
    frames = 31
    names = ["pelv", "lhip", "lkne"]
    joints = np.zeros((frames, len(names), 3), dtype=np.float32)
    joints[:, 0, :] = [0.0, 0.0, 4.0]
    joints[:, 1, :] = [-0.1, 0.0, 4.0]
    joints[:, 2, :] = [-0.1, -0.5, 4.0]
    joints[:, 2, 0] += 0.05 * ((np.arange(frames) % 2) * 2 - 1)
    pose = {"representation": "joints", "backend": "test", "fps": 60.0, "joint_names": names, "joints_3d": joints}

    corrected, report = apply_kinematic_chain_to_pose(
        pose,
        {"method": "moving_average", "window_frames": 9, "smooth_roots": False, "max_joint_displacement_m": None},
    )
    targets = [EdgeLength(edge.parent, edge.child, edge.target_m) for edge in build_chain_edges(joints, names)]
    before = temporal_smoothing_metrics(joints, names, 60.0, target_lengths=targets)
    after = temporal_smoothing_metrics(corrected["joints_3d"], names, 60.0, target_lengths=targets)

    assert report["status"] == "ok"
    assert after["mean_second_diff_m"] < before["mean_second_diff_m"]
    assert after["median_bone_length_error_mm"] < 1e-3


def test_kinematic_chain_supports_coco_style_lower_limb_names() -> None:
    names = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_big_toe", "right_big_toe"]
    joints = np.zeros((5, len(names), 3), dtype=np.float32)
    joints[:, names.index("left_knee"), 1] = -0.4
    joints[:, names.index("right_knee"), 1] = -0.4
    joints[:, names.index("left_ankle"), 1] = -0.8
    joints[:, names.index("right_ankle"), 1] = -0.8
    joints[:, names.index("left_big_toe"), :] = [0.0, -0.8, 0.2]
    joints[:, names.index("right_big_toe"), :] = [0.0, -0.8, 0.2]

    edges = build_chain_edges(joints, names)

    assert len(edges) >= 6


def test_kinematic_chain_skips_hybrid_artifacts() -> None:
    pose = {
        "representation": "hybrid",
        "backend": "wham",
        "joint_names": ["pelv", "lhip"],
        "joints_3d": np.zeros((3, 2, 3), dtype=np.float32),
    }

    corrected, report = apply_kinematic_chain_to_pose(pose, {})

    assert report["status"] == "skipped"
    assert "hybrid" in report["reason"]
    np.testing.assert_allclose(corrected["joints_3d"], pose["joints_3d"])
