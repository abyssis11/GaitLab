from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from monocap_v2.core.mocap_eval import evaluate_pose_against_mocap_with_series
from monocap_v2.core.mocap_marker_compare import compare_virtual_markers_to_mocap, metric_to_display_coords
from monocap_v2.core.opensim_io import write_trc
from monocap_v2.pipeline.stage_12_mocap_validation import _write_mocap_smpl_marker_overlay


JOINT_MARKERS = ["L_HJC", "R_HJC", "L_knee", "r_knee", "L_ankle", "r_ankle", "L_toe", "r_toe"]
EXTRA_MARKERS = ["L_calc", "r_calc", "L.ASIS", "r.ASIS", "L.PSIS", "r.PSIS"]
DEBUG_NAMES = ["DBG_PELV", "DBG_LHIP", "DBG_RHIP", "DBG_LKNE", "DBG_RKNE", "DBG_LANK", "DBG_RANK", "DBG_LHEE", "DBG_RHEE", "DBG_LTOE", "DBG_RTOE"]


def test_marker_comparison_reports_zero_for_matching_debug_anchors_and_duplicate_warning(tmp_path: Path) -> None:
    pose_eval, mocap_eval, debug_eval = _synthetic_values()
    pose = _pose(pose_eval)
    trc_path = tmp_path / "mocap.trc"
    write_trc(trc_path, JOINT_MARKERS + EXTRA_MARKERS, _eval_to_mocap_raw(mocap_eval), fps=30.0)
    eval_report, eval_series = evaluate_pose_against_mocap_with_series(pose, trc_path)
    payload = {
        "marker_set": "test_debug",
        "marker_names": DEBUG_NAMES,
        "markers_m": _eval_to_camera(debug_eval),
        "vertex_indices": [0, 1, 1, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    report, series = compare_virtual_markers_to_mocap(pose, payload, trc_path, eval_report, eval_series)
    assert report["overall_median_residual_mm"] < 1e-5
    assert any("DBG_LHIP and DBG_RHIP resolve to the same SMPL vertex" in warning for warning in report["warnings"])
    assert series["smpl_markers_aligned_m"].shape == (6, len(DEBUG_NAMES), 3)
    assert report["timebase"]["comparison_rate_hz"] == 30.0
    assert report["per_marker"]["DBG_LHIP"]["correspondence_quality"] == "non_equivalent"


def test_marker_comparison_resamples_prediction_to_native_100hz_mocap_grid(tmp_path: Path) -> None:
    pose_eval, mocap_eval, debug_eval = _synthetic_values()
    pose = _pose(pose_eval)
    trc_path = tmp_path / "mocap_100hz.trc"
    time_30 = np.arange(len(mocap_eval)) / 30.0
    time_100 = np.arange(17) / 100.0
    mocap_100 = np.stack([np.interp(time_100, time_30, mocap_eval[:, idx, coord]) for idx in range(mocap_eval.shape[1]) for coord in range(3)])
    mocap_100 = mocap_100.reshape(mocap_eval.shape[1], 3, len(time_100)).transpose(2, 0, 1)
    write_trc(trc_path, JOINT_MARKERS + EXTRA_MARKERS, _eval_to_mocap_raw(mocap_100), fps=100.0)
    eval_report, eval_series = evaluate_pose_against_mocap_with_series(pose, trc_path)
    payload = {"marker_set": "test_debug", "marker_names": DEBUG_NAMES, "markers_m": _eval_to_camera(debug_eval), "vertex_indices": list(range(11))}
    report, series = compare_virtual_markers_to_mocap(pose, payload, trc_path, eval_report, eval_series)
    assert report["timebase"]["comparison_rate_hz"] == 100.0
    assert report["timebase"]["source_prediction_rate_hz"] == 30.0
    assert series["smpl_markers_aligned_m"].shape[0] == 14


def test_metric_to_display_coords_moves_historical_vertical_axis_to_up() -> None:
    metric = np.asarray([[-2.0, 3.0, 4.0]])
    assert np.allclose(metric_to_display_coords(metric), [[4.0, 3.0, 2.0]])


def test_marker_comparison_mesh_overlay_smoke(tmp_path: Path) -> None:
    pose_eval, mocap_eval, debug_eval = _synthetic_values()
    pose = _pose(pose_eval)
    pose["smpl"] = {
        "vertices": _eval_to_camera(
            np.stack(
                [
                    np.array([[-0.3, 0.0, 0.0], [0.3, 0.0, 0.0], [-0.3, 0.0, 1.0], [0.3, 0.0, 1.0]]) + np.array([0.0, 0.03 * idx, 0.0])
                    for idx in range(6)
                ]
            )
        )
    }
    trc_path = tmp_path / "mocap.trc"
    write_trc(trc_path, JOINT_MARKERS + EXTRA_MARKERS, _eval_to_mocap_raw(mocap_eval), fps=30.0)
    eval_report, eval_series = evaluate_pose_against_mocap_with_series(pose, trc_path)
    payload = {"marker_set": "test_debug", "marker_names": DEBUG_NAMES, "markers_m": _eval_to_camera(debug_eval), "vertex_indices": list(range(11))}
    _, series = compare_virtual_markers_to_mocap(pose, payload, trc_path, eval_report, eval_series)
    faces = tmp_path / "faces.npy"
    np.save(faces, np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int32))
    overlay = tmp_path / "overlay.mp4"
    frame = tmp_path / "frame.png"
    cfg = {
        "repo_root": str(tmp_path),
        "config": {
            "smpl": {"faces_path": str(faces)},
            "mocap_validation": {"preview_fps": 10, "marker_comparison": {"max_faces": 100}},
        },
    }
    report = _write_mocap_smpl_marker_overlay(pose, series, eval_report, cfg, overlay, frame)
    assert report["status"] == "ok"
    assert overlay.exists() and overlay.stat().st_size > 0
    assert frame.exists() and frame.stat().st_size > 0
    cap = cv2.VideoCapture(str(overlay))
    try:
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    finally:
        cap.release()


def _synthetic_values() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joints = np.array(
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
    extras = np.array(
        [
            [-0.14, 0.00, 0.04],
            [0.14, 0.00, 0.04],
            [-0.1, 0.02, 1.0],
            [0.1, 0.02, 1.0],
            [-0.1, -0.02, 1.0],
            [0.1, -0.02, 1.0],
        ],
        dtype=float,
    )
    debug = np.array(
        [
            [0.0, 0.0, 1.0],
            *joints[:6],
            extras[0],
            extras[1],
            joints[6],
            joints[7],
        ]
    )
    offsets = np.asarray([[0.0, 0.03 * idx, 0.0] for idx in range(6)])
    pose_eval = joints[None, :, :] + offsets[:, None, :]
    mocap_eval = np.concatenate([joints, extras], axis=0)[None, :, :] + offsets[:, None, :]
    debug_eval = debug[None, :, :] + offsets[:, None, :]
    return pose_eval, mocap_eval, debug_eval


def _pose(eval_coords: np.ndarray) -> dict:
    return {
        "representation": "joints",
        "backend": "test",
        "fps": 30.0,
        "units": "m",
        "joint_names": ["lhip", "rhip", "lkne", "rkne", "lank", "rank", "ltoe", "rtoe"],
        "joints_3d": _eval_to_camera(eval_coords),
    }


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
