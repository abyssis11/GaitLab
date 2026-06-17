from __future__ import annotations

import numpy as np
import pytest

from monocap_v2.core.pose_prior import (
    apply_pose_prior_to_pose,
    joint_angles_deg,
    project_child_to_angle_limit,
)


def test_project_child_to_angle_limit_raises_too_acute_angle() -> None:
    parent = np.array([0.0, 1.0, 0.0])
    center = np.array([0.0, 0.0, 0.0])
    child = np.array([0.1, 0.1, 0.0])

    corrected = project_child_to_angle_limit(parent, center, child, min_angle_deg=90.0, max_angle_deg=180.0)

    assert corrected is not None
    angle = joint_angles_deg(parent[None], center[None], corrected[None])[0]
    assert angle == pytest.approx(90.0)
    assert np.linalg.norm(corrected - center) == pytest.approx(np.linalg.norm(child - center))


def test_pose_prior_corrects_knee_violation_and_moves_toe_with_ankle() -> None:
    names = ["pelvis", "left_hip", "left_knee", "left_ankle", "left_big_toe"]
    joints = np.zeros((2, len(names), 3), dtype=np.float32)
    joints[:, names.index("left_hip"), :] = [0.0, 1.0, 0.0]
    joints[:, names.index("left_knee"), :] = [0.0, 0.0, 0.0]
    joints[:, names.index("left_ankle"), :] = [0.1, 0.1, 0.0]
    joints[:, names.index("left_big_toe"), :] = [0.2, 0.1, 0.0]
    before_toe_delta = joints[:, names.index("left_big_toe"), :] - joints[:, names.index("left_ankle"), :]
    pose = {
        "representation": "joints",
        "backend": "dummy",
        "fps": 60.0,
        "units": "m",
        "joint_names": names,
        "joints_3d": joints,
    }

    corrected, report = apply_pose_prior_to_pose(
        pose,
        {"limit_set": "walking_knee90", "blend": 1.0, "max_joint_displacement_m": None},
    )

    assert report["status"] == "ok"
    assert report["total_corrections"] == 2
    angle = joint_angles_deg(
        corrected["joints_3d"][:, names.index("left_hip"), :],
        corrected["joints_3d"][:, names.index("left_knee"), :],
        corrected["joints_3d"][:, names.index("left_ankle"), :],
    )
    assert np.nanmin(angle) == pytest.approx(90.0, abs=1e-4)
    after_toe_delta = corrected["joints_3d"][:, names.index("left_big_toe"), :] - corrected["joints_3d"][:, names.index("left_ankle"), :]
    assert np.allclose(after_toe_delta, before_toe_delta)


def test_pose_prior_max_displacement_limits_correction() -> None:
    names = ["left_hip", "left_knee", "left_ankle"]
    joints = np.zeros((1, len(names), 3), dtype=np.float32)
    joints[:, names.index("left_hip"), :] = [0.0, 1.0, 0.0]
    joints[:, names.index("left_knee"), :] = [0.0, 0.0, 0.0]
    joints[:, names.index("left_ankle"), :] = [0.1, 0.1, 0.0]
    pose = {
        "representation": "joints",
        "backend": "dummy",
        "joint_names": names,
        "joints_3d": joints,
    }

    corrected, report = apply_pose_prior_to_pose(
        pose,
        {"limit_set": "walking_knee90", "blend": 1.0, "max_joint_displacement_m": 0.01},
    )

    assert report["status"] == "ok"
    displacement = np.linalg.norm(corrected["joints_3d"] - joints, axis=-1)
    assert np.nanmax(displacement) == pytest.approx(0.01)


def test_pose_prior_skips_hybrid_artifacts() -> None:
    pose = {
        "representation": "hybrid",
        "backend": "wham",
        "joint_names": ["pelvis"],
        "joints_3d": np.zeros((2, 1, 3), dtype=np.float32),
        "smpl": {"vertices": np.zeros((2, 10, 3), dtype=np.float32)},
    }

    corrected, report = apply_pose_prior_to_pose(pose, {"limit_set": "walking_knee90"})

    assert report["status"] == "skipped"
    assert "hybrid" in report["reason"]
    assert corrected["smpl"]["vertices"].shape == (2, 10, 3)
