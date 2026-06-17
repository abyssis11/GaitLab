from __future__ import annotations

import numpy as np
import pytest

from monocap_v2.core.geometry import project_points
from monocap_v2.core.reprojection_consistency import (
    align_pose2d_to_joints,
    apply_reprojection_consistency_to_pose,
    reprojection_metrics,
)


def test_depth_preserving_reprojection_reduces_pixel_error_and_keeps_depth() -> None:
    camera = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
    joints = np.array(
        [
            [[0.00, 0.00, 4.0], [0.20, 0.10, 4.0]],
            [[0.02, 0.00, 4.0], [0.22, 0.10, 4.0]],
        ],
        dtype=np.float32,
    )
    observed = project_points(joints, camera)
    observed[..., 0] += 20.0
    observed[..., 1] -= 10.0
    pose = {
        "representation": "joints",
        "backend": "dummy",
        "fps": 60.0,
        "units": "m",
        "joint_names": ["pelv", "lhip"],
        "joints_3d": joints,
    }
    pose2d = {"xy": observed, "confidence": np.ones(observed.shape[:2]), "names": ["pelv", "lhip"], "fps": 60.0}

    corrected, report = apply_reprojection_consistency_to_pose(
        pose,
        camera,
        pose2d,
        {"blend": 1.0, "confidence_threshold": 0.1, "preserve_bones": False},
    )

    assert report["status"] == "ok"
    assert report["metrics_after"]["mean_reprojection_error_px"] < report["metrics_before"]["mean_reprojection_error_px"]
    assert np.allclose(corrected["joints_3d"][..., 2], joints[..., 2])
    assert corrected["joints_3d"][0, 0, 0] > joints[0, 0, 0]
    assert corrected["joints_3d"][0, 0, 1] < joints[0, 0, 1]


def test_reprojection_confidence_threshold_leaves_low_confidence_joint_unchanged() -> None:
    camera = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
    joints = np.array([[[0.0, 0.0, 3.0], [0.1, 0.0, 3.0]]], dtype=np.float32)
    observed = project_points(joints, camera)
    observed[:, 0, 0] += 30.0
    observed[:, 1, 0] += 30.0
    pose = {
        "representation": "joints",
        "backend": "dummy",
        "joint_names": ["pelv", "lhip"],
        "joints_3d": joints,
    }
    pose2d = {"xy": observed, "confidence": np.array([[1.0, 0.1]]), "names": ["pelv", "lhip"]}

    corrected, report = apply_reprojection_consistency_to_pose(
        pose,
        camera,
        pose2d,
        {"blend": 1.0, "confidence_threshold": 0.5, "preserve_bones": False},
    )

    assert report["valid_observations"] == 1
    assert corrected["joints_3d"][0, 0, 0] > joints[0, 0, 0]
    assert corrected["joints_3d"][0, 1, 0] == pytest.approx(float(joints[0, 1, 0]))


def test_reprojection_preserve_bones_keeps_original_median_lengths() -> None:
    camera = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
    joints = np.array(
        [
            [[0.0, 0.0, 3.0], [0.3, 0.0, 3.0]],
            [[0.0, 0.0, 3.0], [0.3, 0.0, 3.0]],
        ],
        dtype=np.float32,
    )
    observed = project_points(joints, camera)
    observed[:, 1, 0] += 100.0
    pose = {
        "representation": "joints",
        "backend": "dummy",
        "joint_names": ["pelvis", "left_hip"],
        "joints_3d": joints,
    }
    pose2d = {"xy": observed, "confidence": np.ones(observed.shape[:2]), "names": ["pelvis", "left_hip"]}

    corrected, report = apply_reprojection_consistency_to_pose(
        pose,
        camera,
        pose2d,
        {"blend": 1.0, "confidence_threshold": 0.1, "preserve_bones": True, "bone_projection_iterations": 4},
    )

    assert report["status"] == "ok"
    lengths = np.linalg.norm(corrected["joints_3d"][:, 1, :] - corrected["joints_3d"][:, 0, :], axis=1)
    assert np.nanmedian(lengths) == pytest.approx(0.3, abs=1e-4)


def test_reprojection_skips_smpl_or_hybrid_artifacts() -> None:
    pose = {
        "representation": "hybrid",
        "backend": "wham",
        "joint_names": ["pelv"],
        "joints_3d": np.zeros((2, 1, 3), dtype=np.float32),
        "smpl": {"vertices": np.zeros((2, 10, 3), dtype=np.float32)},
    }

    corrected, report = apply_reprojection_consistency_to_pose(
        pose,
        {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        {"xy": np.zeros((2, 1, 2)), "confidence": np.ones((2, 1)), "names": ["pelv"]},
        {"blend": 1.0},
    )

    assert report["status"] == "skipped"
    assert "hybrid" in report["reason"]
    assert corrected["smpl"]["vertices"].shape == (2, 10, 3)


def test_align_pose2d_by_names_when_shapes_differ() -> None:
    xy = np.zeros((3, 3, 2), dtype=float)
    xy[:, 2, :] = [10.0, 20.0]
    conf = np.ones((3, 3), dtype=float)

    aligned_xy, aligned_conf, report = align_pose2d_to_joints(
        {"xy": xy, "confidence": conf, "names": ["nose", "right_knee", "left_knee"]},
        ["left_knee"],
        (2, 1),
    )

    assert report["status"] == "ok"
    assert report["mode"] == "name_mapping"
    assert aligned_xy.shape == (2, 1, 2)
    assert aligned_conf.shape == (2, 1)
    assert np.allclose(aligned_xy[:, 0, :], [10.0, 20.0])


def test_reprojection_metrics_accept_nested_camera_intrinsics() -> None:
    joints = np.array([[[0.0, 0.0, 2.0]]], dtype=float)
    camera = {"intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 40.0}}
    xy = project_points(joints, camera["intrinsics"])

    metrics = reprojection_metrics(joints, camera, xy, np.ones((1, 1)))

    assert metrics["mean_reprojection_error_px"] == pytest.approx(0.0)
