from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from monocap_v2.core.mocap_eval import (
    evaluate_pose_against_mocap,
    parse_trc,
    resample_timeseries,
    resolve_prediction_timebase,
    validate_mocap_to_video_transform,
)


MARKERS = ["L_HJC", "R_HJC", "L_knee", "r_knee", "L_ankle", "r_ankle", "L_toe", "r_toe"]


def test_parse_trc_units_and_timestamps(tmp_path: Path) -> None:
    trc_path = tmp_path / "test.trc"
    _write_trc(trc_path, np.zeros((2, len(MARKERS), 3), dtype=float), fps=100.0)
    trc = parse_trc(trc_path)
    assert trc.units == "m"
    assert trc.data_rate == 100.0
    assert trc.marker_names == MARKERS
    assert np.allclose(trc.time, [0.0, 0.01])


def test_resample_timeseries_linear() -> None:
    source = np.array([0.0, 1.0])
    values = np.array([[[0.0, 0.0, 0.0]], [[2.0, 4.0, 6.0]]])
    out = resample_timeseries(source, values, np.array([0.5]))
    assert np.allclose(out[0, 0], [1.0, 2.0, 3.0])


def test_evaluate_pose_against_mocap_root_centered_zero_for_matching_data(tmp_path: Path) -> None:
    eval_coords = np.array(
        [
            [
                [-0.1, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [-0.15, 0.0, 0.55],
                [0.15, 0.0, 0.55],
                [-0.15, 0.0, 0.1],
                [0.15, 0.0, 0.1],
                [-0.15, 0.15, 0.05],
                [0.15, 0.15, 0.05],
            ],
            [
                [-0.1, 0.1, 1.0],
                [0.1, 0.1, 1.0],
                [-0.15, 0.1, 0.55],
                [0.15, 0.1, 0.55],
                [-0.15, 0.1, 0.1],
                [0.15, 0.1, 0.1],
                [-0.15, 0.25, 0.05],
                [0.15, 0.25, 0.05],
            ],
        ],
        dtype=float,
    )
    pose = {
        "representation": "joints",
        "backend": "test",
        "fps": 100.0,
        "units": "m",
        "joint_names": ["lhip", "rhip", "lkne", "rkne", "lank", "rank", "ltoe", "rtoe"],
        "joints_3d": _eval_to_camera(eval_coords),
    }
    trc_path = tmp_path / "mocap.trc"
    _write_trc(trc_path, _eval_to_mocap_raw(eval_coords), fps=100.0)
    report = evaluate_pose_against_mocap(pose, trc_path)
    assert report["primary_root_centered_rigid_mpjpe_m"] < 1e-8
    assert report["primary_root_centered_rigid_mpjpe_mm"] < 1e-5
    assert report["pa_similarity_mpjpe_m"] < 1e-8
    assert report["resampling"] == "mocap_resampled_to_prediction_timestamps"


def test_evaluate_pose_against_mocap_resamples_mocap_to_video_fps(tmp_path: Path) -> None:
    eval_coords = np.zeros((3, len(MARKERS), 3), dtype=float)
    eval_coords[:, :, 1] = np.array([0.0, 0.02, 0.04])[:, None]
    eval_coords[:, :, 2] = 1.0
    pose = {
        "representation": "joints",
        "backend": "test",
        "fps": 50.0,
        "units": "m",
        "joint_names": ["lhip", "rhip", "lkne", "rkne", "lank", "rank", "ltoe", "rtoe"],
        "joints_3d": _eval_to_camera(eval_coords),
    }
    mocap_eval_coords = np.zeros((5, len(MARKERS), 3), dtype=float)
    mocap_eval_coords[:, :, 1] = np.array([0.0, 0.01, 0.02, 0.03, 0.04])[:, None]
    mocap_eval_coords[:, :, 2] = 1.0
    trc_path = tmp_path / "mocap_100hz.trc"
    _write_trc(trc_path, _eval_to_mocap_raw(mocap_eval_coords), fps=100.0)
    report = evaluate_pose_against_mocap(pose, trc_path)
    assert report["prediction_fps"] == 50.0
    assert report["mocap_data_rate_hz"] == 100.0
    assert report["frames"] == 3
    assert report["primary_root_centered_rigid_mpjpe_mm"] < 1e-5


def test_wham_timebase_maps_raw_frames_to_synced_timestamps() -> None:
    pose = {
        "backend": "wham",
        "fps": 60.0,
        "joints_3d": np.zeros((10, len(MARKERS), 3)),
        "backend_meta": {"frame_ids": np.arange(100, 110)},
    }
    timeline = {
        "status": "ok",
        "raw_sync_alignment": {"status": "ok", "best_raw_offset": 102, "sync_frame_count": 5, "sync_fps": 60.0},
        "overlap": {"status": "ok"},
    }
    result = resolve_prediction_timebase(pose, timeline_report=timeline)
    assert result["pose_indices"].tolist() == [2, 3, 4, 5, 6]
    assert result["raw_frame_ids"].tolist() == [102, 103, 104, 105, 106]
    assert result["sync_frame_ids"].tolist() == [0, 1, 2, 3, 4]
    assert np.allclose(result["prediction_timestamps_s"], np.arange(5) / 60.0)


def test_wham_timebase_rejects_uncertain_alignment() -> None:
    pose = {
        "backend": "wham",
        "fps": 60.0,
        "joints_3d": np.zeros((10, len(MARKERS), 3)),
        "backend_meta": {"frame_ids": np.arange(100, 110)},
    }
    timeline = {
        "status": "warning",
        "raw_sync_alignment": {"status": "warning", "best_raw_offset": 102, "sync_frame_count": 5, "sync_fps": 60.0},
        "overlap": {"status": "ok"},
    }
    with pytest.raises(ValueError, match="uncertain"):
        resolve_prediction_timebase(pose, timeline_report=timeline)


def test_normal_and_rigid_mpjpe_separate_rotation_mismatch(tmp_path: Path) -> None:
    reference = _walking_eval_coords()
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    prediction = reference @ rotation.T
    report = evaluate_pose_against_mocap(_pose(prediction), _write_reference_trc(tmp_path, reference))
    assert report["normal_root_centered_mpjpe_mm"] > 10.0
    assert report["root_centered_rigid_mpjpe_mm"] < 1e-5


def test_normalized_and_pa_mpjpe_recover_scale_mismatch(tmp_path: Path) -> None:
    reference = _walking_eval_coords()
    prediction = reference * 1.5
    report = evaluate_pose_against_mocap(_pose(prediction), _write_reference_trc(tmp_path, reference))
    assert report["normal_root_centered_mpjpe_mm"] > 10.0
    assert report["root_centered_n_mpjpe_mm"] < 1e-5
    assert report["pa_mpjpe_mm"] < 1e-5


def test_singular_mocap_transform_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "transform.yaml"
    path.write_text(
        "R_fromMocap_toVideo:\n"
        "  - [0, 0, -1]\n"
        "  - [0, -1, 0]\n"
        "  - [0, 0, -1]\n"
        "position_fromMocapOrigin_toVideoOrigin: [0, 0, 0]\n",
        encoding="utf-8",
    )
    report = validate_mocap_to_video_transform(path)
    assert report["status"] == "warning"
    assert report["determinant"] == 0.0
    assert any("orthonormal" in warning for warning in report["warnings"])


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


def _walking_eval_coords() -> np.ndarray:
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
    return np.stack([frame, frame + np.array([0.0, 0.1, 0.0]), frame + np.array([0.0, 0.2, 0.0])])


def _pose(eval_coords: np.ndarray) -> dict:
    return {
        "representation": "joints",
        "backend": "test",
        "fps": 100.0,
        "units": "m",
        "joint_names": ["lhip", "rhip", "lkne", "rkne", "lank", "rank", "ltoe", "rtoe"],
        "joints_3d": _eval_to_camera(eval_coords),
    }


def _write_reference_trc(tmp_path: Path, eval_coords: np.ndarray) -> Path:
    path = tmp_path / "reference.trc"
    _write_trc(path, _eval_to_mocap_raw(eval_coords), fps=100.0)
    return path


def _write_trc(path: Path, values_m: np.ndarray, fps: float) -> None:
    values_mm = values_m * 1000.0
    with path.open("w", encoding="utf-8") as f:
        f.write("PathFileType\t4\t(X/Y/Z)\ttest.trc\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps:.1f}\t{fps:.1f}\t{values_mm.shape[0]}\t{values_mm.shape[1]}\tmm\t{fps:.1f}\t1\t{values_mm.shape[0]}\n")
        f.write("Frame#\tTime\t" + "\t\t\t".join(MARKERS) + "\t\t\t\n")
        f.write("\t" + "\t".join(f"{axis}{i + 1}" for i in range(len(MARKERS)) for axis in ["X", "Y", "Z"]) + "\n\n")
        for frame_idx, frame in enumerate(values_mm):
            row = [str(frame_idx + 1), f"{frame_idx / fps:.8f}"]
            for xyz in frame:
                row.extend(f"{v:.6f}" for v in xyz)
            f.write("\t".join(row) + "\n")
