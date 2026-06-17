from __future__ import annotations

import numpy as np
import pytest

from monocap_v2.core.subject_scale import (
    apply_subject_scale_to_pose,
    height_based_targets,
    measure_segment_lengths,
    static_based_targets,
)


NAMES = ["pelv", "lhip", "rhip", "lkne", "rkne", "lank", "rank", "ltoe", "rtoe"]


def test_measure_segment_lengths_handles_missing_and_nans() -> None:
    joints = _pose_array(thigh=0.3, shank=0.3, foot=0.1)
    joints[1, NAMES.index("lkne"), :] = np.nan

    measured = measure_segment_lengths(joints, NAMES)
    missing = measure_segment_lengths(joints[:, :2, :], NAMES[:2])

    assert measured["left_thigh"]["valid_frames"] == 1
    assert measured["right_thigh"]["valid_frames"] == 2
    assert missing["right_thigh"]["status"] == "missing_joints"


def test_height_targets_are_deterministic_meters() -> None:
    targets = height_based_targets(2.0)

    assert targets["left_thigh"] == pytest.approx(0.49)
    assert targets["right_shank"] == pytest.approx(0.492)
    assert targets["left_foot"] == pytest.approx(0.304)


def test_static_targets_are_normalized_separately_from_height_targets() -> None:
    static = _pose_array(thigh=0.4, shank=0.4, foot=0.2)
    targets, report = static_based_targets(static, NAMES, 2.0)
    height_targets = height_based_targets(2.0)

    assert report["static_normalization_scale"] == pytest.approx((2.0 * (0.245 + 0.246)) / 0.8)
    assert targets["left_thigh"] == pytest.approx(0.491)
    assert targets["left_foot"] != pytest.approx(height_targets["left_foot"])


def test_global_scale_preserves_shape_and_changes_lengths() -> None:
    pose = _pose(thigh=0.245, shank=0.246, foot=0.152)
    corrected, report = apply_subject_scale_to_pose(
        pose,
        {"height_m": 2.0},
        {"mode": "height_global", "min_scale": 0.5, "max_scale": 3.0},
    )

    measured = measure_segment_lengths(np.asarray(corrected["joints_3d"]), NAMES)
    assert report["status"] == "ok"
    assert report["global_scale"]["scale"] == pytest.approx(2.0)
    assert measured["left_thigh"]["median_m"] == pytest.approx(0.49)
    assert measured["left_shank"]["median_m"] == pytest.approx(0.492)
    assert np.allclose(corrected["joints_3d"][:, NAMES.index("lhip"), :], pose["joints_3d"][:, NAMES.index("lhip"), :] * np.array([2.0, 2.0, 2.0]))


def test_bone_retarget_reaches_lower_limb_targets_from_hip_outward() -> None:
    pose = _pose(thigh=0.3, shank=0.2, foot=0.1)
    corrected, report = apply_subject_scale_to_pose(pose, {"height_m": 2.0}, {"mode": "height_bone"})

    measured = measure_segment_lengths(np.asarray(corrected["joints_3d"]), NAMES)
    assert report["status"] == "ok"
    assert measured["left_thigh"]["median_m"] == pytest.approx(0.49)
    assert measured["left_shank"]["median_m"] == pytest.approx(0.492)
    assert measured["left_foot"]["median_m"] == pytest.approx(0.304)


def test_hybrid_artifacts_skip_without_changing_joints() -> None:
    pose = _pose(thigh=0.3, shank=0.2, foot=0.1)
    pose["representation"] = "hybrid"
    pose["smpl"] = {"vertices": np.zeros((2, 4, 3), dtype=np.float32)}

    corrected, report = apply_subject_scale_to_pose(pose, {"height_m": 2.0}, {"mode": "height_global"})

    assert report["status"] == "skipped"
    assert "joints artifacts" in report["reason"]
    assert np.allclose(corrected["joints_3d"], pose["joints_3d"])


def _pose(thigh: float, shank: float, foot: float) -> dict:
    return {
        "representation": "joints",
        "backend": "test",
        "fps": 60.0,
        "units": "m",
        "joint_names": list(NAMES),
        "joints_3d": _pose_array(thigh, shank, foot).astype(np.float32),
    }


def _pose_array(thigh: float, shank: float, foot: float) -> np.ndarray:
    joints = np.zeros((2, len(NAMES), 3), dtype=float)
    joints[:, NAMES.index("lhip"), :] = [-0.1, 0.0, 0.0]
    joints[:, NAMES.index("rhip"), :] = [0.1, 0.0, 0.0]
    joints[:, NAMES.index("lkne"), :] = [-0.1, -thigh, 0.0]
    joints[:, NAMES.index("rkne"), :] = [0.1, -thigh, 0.0]
    joints[:, NAMES.index("lank"), :] = [-0.1, -(thigh + shank), 0.0]
    joints[:, NAMES.index("rank"), :] = [0.1, -(thigh + shank), 0.0]
    joints[:, NAMES.index("ltoe"), :] = [-0.1, -(thigh + shank), foot]
    joints[:, NAMES.index("rtoe"), :] = [0.1, -(thigh + shank), foot]
    return joints
