from __future__ import annotations

import numpy as np
import pytest

from monocap_v2.core.temporal_smoothing import (
    apply_temporal_smoothing_to_pose,
    estimate_edge_lengths,
    project_bones_to_lengths,
    temporal_smoothing_metrics,
)


def test_short_edge_aliases_support_coco_style_joint_names() -> None:
    names = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_big_toe", "right_big_toe"]
    joints = np.zeros((3, len(names), 3), dtype=np.float32)
    joints[:, names.index("left_knee"), 1] = -0.4
    joints[:, names.index("right_knee"), 1] = -0.4
    joints[:, names.index("left_ankle"), 1] = -0.8
    joints[:, names.index("right_ankle"), 1] = -0.8
    joints[:, names.index("left_big_toe"), :] = [0.0, -0.8, 0.2]
    joints[:, names.index("right_big_toe"), :] = [0.0, -0.8, 0.2]

    targets = estimate_edge_lengths(joints, names)

    assert len(targets) >= 6


def test_temporal_smoothing_reduces_synthetic_jitter() -> None:
    frames = 31
    names = ["pelv", "lhip", "lkne"]
    joints = np.zeros((frames, len(names), 3), dtype=np.float32)
    t = np.linspace(0.0, 1.0, frames)
    joints[:, 0, 0] = t
    joints[:, 1, :] = np.stack([t, np.full(frames, -0.1), np.zeros(frames)], axis=1)
    joints[:, 2, :] = np.stack([t, np.full(frames, -0.55), np.zeros(frames)], axis=1)
    joints[:, :, 1] += (0.02 * np.sin(np.arange(frames) * np.pi))[:, None]
    joints[::2, :, 2] += 0.03
    pose = {
        "representation": "joints",
        "backend": "test",
        "fps": 60.0,
        "units": "m",
        "joint_names": names,
        "joints_3d": joints,
    }

    corrected, report = apply_temporal_smoothing_to_pose(
        pose,
        {"method": "moving_average", "window_frames": 5, "preserve_bones": False, "max_joint_displacement_m": None},
    )

    assert report["status"] == "ok"
    before = report["metrics_before"]["mean_second_diff_m"]
    after = report["metrics_after"]["mean_second_diff_m"]
    assert after < before
    assert corrected["joints_3d"].shape == joints.shape


def test_bone_projection_preserves_original_edge_lengths() -> None:
    frames = 4
    names = ["pelv", "lhip", "lkne"]
    reference = np.zeros((frames, 3, 3), dtype=float)
    reference[:, 1, :] = [-0.1, 0.0, 0.0]
    reference[:, 2, :] = [-0.1, -0.5, 0.0]
    stretched = reference.copy()
    stretched[:, 2, :] = [-0.1, -0.8, 0.0]
    targets = estimate_edge_lengths(reference, names)

    projected = project_bones_to_lengths(stretched, reference, names, targets=targets, iterations=2, blend=1.0)
    length = np.linalg.norm(projected[:, 2, :] - projected[:, 1, :], axis=1)

    assert np.nanmedian(length) == pytest.approx(0.5)


def test_temporal_smoothing_with_bone_preservation_keeps_bone_error_small() -> None:
    frames = 9
    names = ["pelv", "lhip", "lkne"]
    joints = np.zeros((frames, 3, 3), dtype=np.float32)
    joints[:, 1, :] = [-0.1, 0.0, 0.0]
    joints[:, 2, :] = [-0.1, -0.5, 0.0]
    joints[4, 2, 0] += 0.08
    pose = {
        "representation": "joints",
        "backend": "test",
        "fps": 60.0,
        "units": "m",
        "joint_names": names,
        "joints_3d": joints,
    }

    corrected, report = apply_temporal_smoothing_to_pose(
        pose,
        {"method": "moving_average", "window_frames": 5, "preserve_bones": True, "bone_projection_iterations": 3},
    )
    targets = estimate_edge_lengths(joints, names)
    metrics = temporal_smoothing_metrics(corrected["joints_3d"], names, 60.0, target_lengths=targets)

    assert report["status"] == "ok"
    assert metrics["median_bone_length_error_mm"] < 1.0


def test_temporal_smoothing_skips_hybrid_artifacts() -> None:
    pose = {
        "representation": "hybrid",
        "backend": "wham",
        "joint_names": ["pelv"],
        "joints_3d": np.zeros((3, 1, 3), dtype=np.float32),
    }

    corrected, report = apply_temporal_smoothing_to_pose(pose, {"method": "moving_average"})

    assert report["status"] == "skipped"
    assert "hybrid" in report["reason"]
    np.testing.assert_allclose(corrected["joints_3d"], pose["joints_3d"])
