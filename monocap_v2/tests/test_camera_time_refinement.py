from __future__ import annotations

import numpy as np
import pytest

from monocap_v2.core.camera_time_refinement import (
    apply_camera_delta,
    apply_camera_time_refinement_to_pose,
    reprojection_time_metrics,
)
from monocap_v2.core.geometry import project_points


def test_time_search_recovers_known_positive_offset_from_reprojection() -> None:
    camera = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}
    fps = 10.0
    joints = _moving_joints(20)
    projected = project_points(joints, camera)
    xy = projected.copy()
    xy[:-1] = projected[1:]
    xy[-1] = projected[-1]
    pose = _pose(joints, fps=fps)
    pose2d = {"xy": xy, "confidence": np.ones(xy.shape[:2]), "names": pose["joint_names"], "fps": fps}

    _corrected, report = apply_camera_time_refinement_to_pose(
        pose,
        camera,
        pose2d,
        {
            "time_search": {"enabled": True, "offsets_s": [-0.1, 0.0, 0.1]},
            "camera_delta": {"enabled": False},
        },
    )

    assert report["status"] == "ok"
    assert report["selected_time_offset_s"] == pytest.approx(0.1)
    assert report["metrics_after"]["mean_reprojection_error_px"] < report["metrics_before"]["mean_reprojection_error_px"]


def test_camera_delta_reduces_reprojection_error() -> None:
    camera = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}
    joints = _moving_joints(12)
    target = apply_camera_delta(joints, np.array([0.0, 0.0, np.deg2rad(2.0)]), np.array([0.02, -0.01, 0.0]))
    xy = project_points(target, camera)
    pose = _pose(joints)
    pose2d = {"xy": xy, "confidence": np.ones(xy.shape[:2]), "names": pose["joint_names"], "fps": 60.0}

    corrected, report = apply_camera_time_refinement_to_pose(
        pose,
        camera,
        pose2d,
        {
            "time_search": {"enabled": False},
            "camera_delta": {
                "enabled": True,
                "max_rotation_deg": 5.0,
                "max_translation_m": 0.05,
                "max_nfev": 30,
                "rotation_prior_weight": 0.0,
                "translation_prior_weight": 0.0,
            },
        },
    )

    assert report["status"] == "ok"
    assert report["camera_delta"]["status"] == "ok"
    assert report["metrics_after"]["mean_reprojection_error_px"] < report["metrics_before"]["mean_reprojection_error_px"]
    assert not np.allclose(corrected["joints_3d"], joints)


def test_reprojection_time_metrics_reports_motion_correlation() -> None:
    camera = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
    joints = _moving_joints(8)
    xy = project_points(joints, camera)

    metrics = reprojection_time_metrics(joints, camera, xy, np.ones(xy.shape[:2]), 60.0, 60.0, 0.0)

    assert metrics["mean_reprojection_error_px"] == pytest.approx(0.0)
    assert metrics["motion_correlation"] == pytest.approx(1.0)


def test_camera_time_refinement_skips_hybrid_artifacts() -> None:
    pose = _pose(_moving_joints(3))
    pose["representation"] = "hybrid"
    pose["smpl"] = {"vertices": np.zeros((3, 10, 3), dtype=np.float32)}

    corrected, report = apply_camera_time_refinement_to_pose(
        pose,
        {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        {"xy": np.zeros((3, 2, 2)), "confidence": np.ones((3, 2)), "names": pose["joint_names"]},
        {},
    )

    assert report["status"] == "skipped"
    assert "hybrid" in report["reason"]
    assert corrected["smpl"]["vertices"].shape == (3, 10, 3)


def _moving_joints(frames: int) -> np.ndarray:
    joints = np.zeros((frames, 2, 3), dtype=np.float32)
    joints[:, :, 2] = 3.0
    joints[:, 0, 0] = np.linspace(0.0, 0.2, frames)
    joints[:, 1, 0] = np.linspace(0.2, 0.4, frames)
    joints[:, 1, 1] = 0.1
    return joints


def _pose(joints: np.ndarray, fps: float = 60.0) -> dict:
    return {
        "representation": "joints",
        "backend": "dummy",
        "fps": fps,
        "units": "m",
        "joint_names": ["pelv", "lhip"],
        "joints_3d": joints.astype(np.float32),
    }
