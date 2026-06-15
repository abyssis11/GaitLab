from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.opensim_io import write_trc
from monocap_v2.pipeline import stage_12_mocap_validation


MARKERS = ["L_HJC", "R_HJC", "L_knee", "r_knee", "L_ankle", "r_ankle", "L_toe", "r_toe"]


def test_stage_writes_outputs_and_respects_cache(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_standard_dirs()
    eval_coords = _eval_coords()
    pose = {
        "representation": "joints",
        "backend": "test",
        "fps": 30.0,
        "units": "m",
        "joint_names": ["lhip", "rhip", "lkne", "rkne", "lank", "rank", "ltoe", "rtoe"],
        "joints_3d": _eval_to_camera(eval_coords),
    }
    for key in ("pose3d_initial", "pose3d_refined"):
        with registry.ensure_parent(key).open("wb") as f:
            pickle.dump(pose, f)
    trc_path = tmp_path / "mocap.trc"
    write_trc(trc_path, MARKERS, _eval_to_mocap_raw(eval_coords), fps=30.0)
    cfg = {"mocap_trc": str(trc_path), "config": {"mocap_validation": {"preview_fps": 10}}}

    first = stage_12_mocap_validation.run(tmp_path, cfg, force=False)
    second = stage_12_mocap_validation.run(tmp_path, cfg, force=False)
    forced = stage_12_mocap_validation.run(tmp_path, cfg, force=True)

    assert first["status"] == "ok"
    assert second["status"] == "cached"
    assert forced["status"] == "ok"
    for key in [
        "mocap_validation",
        "mocap_validation_series",
        "mocap_joint_errors_plot",
        "mocap_lower_limb_overlay",
        "mocap_foot_trajectories_plot",
        "mocap_segment_lengths_plot",
        "mocap_only_native_qc",
        "mocap_only_native_overlay",
        "mocap_only_native_representative_frame",
    ]:
        assert registry.get(key).exists()
        assert registry.get(key).stat().st_size > 0
    cap = cv2.VideoCapture(str(registry.get("mocap_lower_limb_overlay")))
    try:
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(eval_coords)
    finally:
        cap.release()
    cap = cv2.VideoCapture(str(registry.get("mocap_only_native_overlay")))
    try:
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(eval_coords)
    finally:
        cap.release()
    assert first["mocap_only_native"]["coordinate_space"] == "raw_trc_native"
    assert first["mocap_only_native"]["alignment"] == "none"
    assert first["mocap_only_native"]["resampling"] == "none"


def test_stage_skips_cleanly_without_mocap(tmp_path: Path) -> None:
    result = stage_12_mocap_validation.run(tmp_path, {}, force=False)
    assert result["status"] == "skipped"
    assert "mocap_trc" in result["reason"]


def _eval_coords() -> np.ndarray:
    frame = np.array(
        [
            [-0.1, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [-0.12, 0.05, 0.55],
            [0.12, -0.05, 0.55],
            [-0.13, 0.10, 0.10],
            [0.13, -0.10, 0.10],
            [-0.14, 0.25, 0.05],
            [0.14, 0.05, 0.05],
        ],
        dtype=float,
    )
    return np.stack([frame + np.array([0.0, 0.03 * idx, 0.0]) for idx in range(6)])


def _eval_to_camera(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values)
    out[..., 0] = values[..., 0]
    out[..., 1] = -values[..., 2]
    out[..., 2] = values[..., 1]
    return out


def _eval_to_mocap_raw(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values)
    out[..., 0] = values[..., 1]
    out[..., 1] = -values[..., 0]
    out[..., 2] = values[..., 2]
    return out
