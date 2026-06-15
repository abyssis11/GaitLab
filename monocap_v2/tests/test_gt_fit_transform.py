from __future__ import annotations

import numpy as np

from monocap_v2.core.gt_fit_transform import _apply_mapping, fit_sequence_transform, temporal_holdout_fit


JOINT_NAMES = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]


def test_gt_fit_rigid_recovers_sequence_transform() -> None:
    pred = _synthetic_pose()
    theta = np.deg2rad(25.0)
    rot = np.asarray(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ],
        dtype=float,
    )
    trans = np.asarray([0.4, -0.2, 0.1], dtype=float)
    ref = pred @ rot.T + trans[None, None, :]

    fit = fit_sequence_transform(pred, ref, JOINT_NAMES, mode="rigid")

    assert fit["mpjpe_mm"] < 1e-6
    assert fit["transform"]["scale"] == 1.0
    assert fit["transform"]["rotation_determinant"] > 0.999


def test_gt_fit_similarity_recovers_scale() -> None:
    pred = _synthetic_pose()
    ref = pred * 1.25 + np.asarray([0.1, 0.2, -0.3])[None, None, :]

    rigid = fit_sequence_transform(pred, ref, JOINT_NAMES, mode="rigid")
    similarity = fit_sequence_transform(pred, ref, JOINT_NAMES, mode="similarity")

    assert rigid["mpjpe_mm"] > 1.0
    assert similarity["mpjpe_mm"] < 1e-6
    assert abs(float(similarity["transform"]["scale"]) - 1.25) < 1e-9


def test_gt_fit_mapping_can_swap_gt_labels() -> None:
    pred = _synthetic_pose()
    ref = _swap_lr(pred)

    no_pred, no_ref = _apply_mapping(pred, ref, JOINT_NAMES, "no_swap")
    swap_pred, swap_ref = _apply_mapping(pred, ref, JOINT_NAMES, "swap_gt_only")
    no_fit = fit_sequence_transform(no_pred, no_ref, JOINT_NAMES, mode="none")
    swap_fit = fit_sequence_transform(swap_pred, swap_ref, JOINT_NAMES, mode="none")

    assert no_fit["mpjpe_mm"] > 100.0
    assert swap_fit["mpjpe_mm"] < 1e-6


def test_temporal_holdout_reports_second_half_error() -> None:
    pred = _synthetic_pose(frames=8)
    ref = pred + np.asarray([0.05, 0.0, 0.0])[None, None, :]

    report = temporal_holdout_fit(pred, ref, JOINT_NAMES, mode="translation")

    assert report["status"] == "ok"
    assert report["first_half_fit_mpjpe_mm"] < 1e-6
    assert report["second_half_eval_mpjpe_mm"] < 1e-6


def _synthetic_pose(frames: int = 5) -> np.ndarray:
    base = np.asarray(
        [
            [-0.1, 0.9, 0.0],
            [0.1, 0.9, 0.0],
            [-0.1, 0.5, 0.05],
            [0.1, 0.5, -0.05],
            [-0.1, 0.1, 0.02],
            [0.1, 0.1, -0.02],
        ],
        dtype=float,
    )
    out = np.repeat(base[None, :, :], frames, axis=0)
    out[:, :, 2] += np.linspace(0.0, 0.2, frames)[:, None]
    return out


def _swap_lr(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[:, [0, 1], :] = out[:, [1, 0], :]
    out[:, [2, 3], :] = out[:, [3, 2], :]
    out[:, [4, 5], :] = out[:, [5, 4], :]
    return out
