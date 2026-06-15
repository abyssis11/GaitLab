from __future__ import annotations

import numpy as np

from monocap_v2.core.geometry import apply_axis_expr
from monocap_v2.core.level_a_audit import run_axis_sweep, run_timing_sweep


def test_axis_sweep_detects_known_axis_expression() -> None:
    reference = _reference()
    pose = _pose(apply_axis_expr(reference["joints_m"], "x,-y,z"))
    report = run_axis_sweep(pose, reference, {}, "rtmw3d", axis_expressions=["identity", "x,-y,z"])
    assert report["best"]["axis"] == "x,-y,z"
    assert report["best"]["primary_root_centered_mpjpe_mm"] < 1e-3


def test_axis_sweep_reports_left_right_swap_as_diagnostic_only() -> None:
    reference = _reference()
    swapped = reference["joints_m"].copy()
    for a, b in [(1, 2), (3, 4), (5, 6)]:
        swapped[:, [a, b], :] = swapped[:, [b, a], :]
    report = run_axis_sweep(_pose(swapped), reference, {}, "rtmw3d", axis_expressions=["identity"])
    assert report["best"]["left_right_swapped"] is True
    assert report["best"]["primary_root_centered_mpjpe_mm"] < 1e-3


def test_timing_sweep_detects_synthetic_offset() -> None:
    times = np.arange(20, dtype=float) / 10.0
    reference = _reference(times)
    pose = _pose(apply_axis_expr(_values(times - 0.2), "x,-y,z"), fps=10.0)
    report = run_timing_sweep(pose, reference, {}, "rtmw3d", time_offsets=[-0.2, 0.0, 0.2])
    assert report["best"]["time_offset_s"] == 0.2
    assert report["best"]["primary_root_centered_mpjpe_mm"] < 1e-3


def _reference(times: np.ndarray | None = None) -> dict:
    times = np.arange(10, dtype=float) / 10.0 if times is None else np.asarray(times, dtype=float)
    return {
        "time_s": times,
        "joints_m": _values(times),
        "joint_names": ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
        "source": "synthetic",
    }


def _pose(joints: np.ndarray, fps: float = 10.0) -> dict:
    return {
        "representation": "joints",
        "backend": "rtmw3d",
        "fps": fps,
        "units": "m",
        "joint_names": ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
        "joints_3d": np.asarray(joints, dtype=np.float32),
    }


def _values(times: np.ndarray) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    out = np.zeros((t.shape[0], 7, 3), dtype=float)
    base = np.array(
        [
            [0.0, 0.0, 1.0],
            [-0.11, 0.02, 0.9],
            [0.11, -0.02, 0.9],
            [-0.13, 0.05, 0.55],
            [0.13, -0.04, 0.55],
            [-0.15, 0.08, 0.15],
            [0.15, -0.07, 0.15],
        ],
        dtype=float,
    )
    out[:] = base[None, :, :]
    phase = 2.0 * np.pi * t
    out[:, 3, 0] += 0.06 * np.sin(phase)
    out[:, 5, 0] += 0.10 * np.sin(phase)
    out[:, 4, 0] -= 0.06 * np.sin(phase)
    out[:, 6, 0] -= 0.10 * np.sin(phase)
    out[:, 5, 2] += 0.03 * np.cos(phase)
    out[:, 6, 2] -= 0.03 * np.cos(phase)
    return out
