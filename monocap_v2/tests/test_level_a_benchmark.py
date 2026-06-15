from __future__ import annotations

import json
import pickle
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from monocap_v2.scripts.benchmark_backends_level_a import _pipeline_python
from monocap_v2.core.level_a_benchmark import (
    aggregate_level_a,
    backend_run_name,
    compare_pose_to_opensim_reference,
)
from monocap_v2.core.opensim_fk import parse_storage_table, read_ik_marker_error_summary
from monocap_v2.core.opensim_fk_visualization import render_opensim_fk_preview
from monocap_v2.core.level_a_visualization import render_level_a_overlay
from monocap_v2.core.wham_conventions import (
    LEGACY_PROFILE,
    OPENCAP_CAMERA_FULL_PROFILE,
    OPENCAP_CAMERA_LR_PROFILE,
    OPENCAP_CAMERA_PROFILE,
    apply_level_a_convention,
    axis_expr_determinant,
    swap_left_right_names,
)


def test_parse_opensim_storage_table_reads_time_and_degrees(tmp_path: Path) -> None:
    path = tmp_path / "ik.mot"
    path.write_text(
        "\n".join(
            [
                "Coordinates",
                "inDegrees=yes",
                "endheader",
                "time pelvis_tilt pelvis_tx",
                "0.00 10.0 1.0",
                "0.01 11.0 1.1",
            ]
        ),
        encoding="utf-8",
    )
    table = parse_storage_table(path)
    assert table.in_degrees is True
    assert table.columns == ["pelvis_tilt", "pelvis_tx"]
    assert np.allclose(table.time, [0.0, 0.01])
    assert np.allclose(table.data[:, 0], [10.0, 11.0])


def test_ik_marker_error_summary_reports_millimeters(tmp_path: Path) -> None:
    path = tmp_path / "errors.sto"
    path.write_text(
        "\n".join(
            [
                "Model Marker Errors from IK",
                "endheader",
                "time marker_error_RMS marker_error_max",
                "0.00 0.010 0.030",
                "0.01 0.020 0.040",
            ]
        ),
        encoding="utf-8",
    )
    summary = read_ik_marker_error_summary(path)
    assert summary["status"] == "ok"
    assert summary["rms_median_mm"] == 15.0
    assert summary["max_mean_mm"] == 35.0


def test_level_a_matching_synthetic_pose_has_zero_primary_error() -> None:
    reference = _reference()
    pose = _pose(reference["joints_m"].copy(), backend="rtmw3d", axis_space="identity")
    report, _ = compare_pose_to_opensim_reference(pose, reference, axis_map={"rtmw3d": "identity"})
    assert report["primary_root_centered_mpjpe_mm"] < 1e-3
    assert report["root_centered_rigid_mpjpe_mm"] < 1e-3
    assert report["primary_joint_count"] == 6


def test_level_a_rotation_mismatch_has_large_normal_small_rigid_and_preserves_hip_root() -> None:
    reference = _reference()
    rot_z_90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pred = reference["joints_m"] @ rot_z_90.T
    pose = _pose(pred, backend="rtmw3d", axis_space="identity")
    report, series = compare_pose_to_opensim_reference(pose, reference, axis_map={"rtmw3d": "identity"})
    names = [str(name) for name in series["joint_names"].tolist()]
    left = names.index("left_hip")
    right = names.index("right_hip")
    fitted_hip_midpoint = 0.5 * (
        series["prediction_root_centered_rigid_m"][:, left, :]
        + series["prediction_root_centered_rigid_m"][:, right, :]
    )

    assert report["primary_root_centered_mpjpe_mm"] > 100.0
    assert report["root_centered_rigid_mpjpe_mm"] < 1e-3
    assert np.allclose(fitted_hip_midpoint, 0.0, atol=1e-9)


def test_level_a_scale_mismatch_is_recovered_by_normalized_metrics() -> None:
    reference = _reference()
    pose = _pose(reference["joints_m"] * 1.25, backend="rtmw3d", axis_space="identity")
    report, series = compare_pose_to_opensim_reference(pose, reference, axis_map={"rtmw3d": "identity"})
    assert report["primary_root_centered_mpjpe_mm"] > 10.0
    assert report["root_centered_n_mpjpe_mm"] < 1e-6
    assert report["pa_mpjpe_mm"] < 1e-6
    assert "prediction_pa_root_centered_m" in series


def test_level_a_optional_evaluation_hz_resamples_prediction_and_reference() -> None:
    reference = _reference()
    pose = _pose(reference["joints_m"].copy(), backend="rtmw3d", axis_space="identity")
    report, series = compare_pose_to_opensim_reference(
        pose,
        reference,
        axis_map={"rtmw3d": "identity"},
        evaluation_hz=120.0,
    )

    assert report["resampling"] == "prediction_and_reference_resampled_to_uniform_timestamps"
    assert report["evaluation_hz"] == 120.0
    assert report["timebase"]["evaluation"]["evaluation_hz"] == 120.0
    assert np.allclose(np.diff(series["time_s"]), 1.0 / 120.0)
    assert report["primary_root_centered_mpjpe_mm"] < 1e-3


def test_level_a_hip_midpoint_root_is_robust_to_pelvis_origin_difference() -> None:
    reference = _reference()
    pred = reference["joints_m"].copy()
    pred[:, 0, 0] += 0.25
    pose = _pose(pred, backend="rtmw3d", axis_space="identity")

    hip_report, hip_series = compare_pose_to_opensim_reference(pose, reference, axis_map={"rtmw3d": "identity"})
    pelvis_report, _ = compare_pose_to_opensim_reference(
        pose,
        reference,
        axis_map={"rtmw3d": "identity"},
        root_mode="pelvis",
    )

    assert hip_report["root_mode"] == "hip_midpoint"
    assert hip_series["root_name"] == "hip_midpoint"
    assert hip_report["primary_root_centered_mpjpe_mm"] < 1e-3
    assert pelvis_report["primary_root_centered_mpjpe_mm"] > 200.0


def test_wham_default_level_a_convention_remains_legacy() -> None:
    reference = _reference()
    wham_camera_like = reference["joints_m"].copy()
    wham_camera_like[:, :, 1] *= -1.0
    pose = _wham_pose(wham_camera_like)
    report, _ = compare_pose_to_opensim_reference(pose, reference, timeline_report=_wham_timeline())

    assert report["convention_profile"] == LEGACY_PROFILE
    assert report["axis_mode"] == "x,-y,z"
    assert report["linear_transform_determinant"] == pytest.approx(-1.0)
    assert report["default_eligible"] is False
    assert report["primary_root_centered_mpjpe_mm"] < 1e-3


def test_wham_opencap_camera_profile_maps_camera_vectors_back_to_lab(tmp_path: Path) -> None:
    reference = _reference()
    theta = np.deg2rad(37.0)
    rotation_lab_to_camera = np.asarray(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ],
        dtype=float,
    )
    pre_axis_camera = reference["joints_m"] @ rotation_lab_to_camera.T
    wham_camera_like = pre_axis_camera.copy()
    wham_camera_like[:, :, 1] *= -1.0
    camera_path = tmp_path / "cameraIntrinsicsExtrinsics.pickle"
    with camera_path.open("wb") as f:
        pickle.dump({"rotation": rotation_lab_to_camera}, f)
    run_config = {"manifest_summary": {"calibration": {"intrinsics_extrinsics": str(camera_path)}}}

    report, _ = compare_pose_to_opensim_reference(
        _wham_pose(wham_camera_like),
        reference,
        run_config=run_config,
        convention_profile=OPENCAP_CAMERA_PROFILE,
        timeline_report=_wham_timeline(),
    )

    assert report["convention_profile"] == OPENCAP_CAMERA_PROFILE
    assert report["convention_source"] == "opencap_camera_calibration"
    assert report["camera_rotation_source"] == str(camera_path)
    assert report["left_right_swap"] is False
    assert report["primary_root_centered_mpjpe_mm"] < 1e-3


def test_wham_opencap_full_extrinsic_profile_uses_translation_for_global_metric(tmp_path: Path) -> None:
    reference = _reference()
    theta = np.deg2rad(-24.0)
    rotation_lab_to_camera = np.asarray(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ],
        dtype=float,
    )
    translation_mm = np.asarray([1000.0, -2000.0, 500.0], dtype=float)
    translation_m = translation_mm / 1000.0
    pre_axis_camera = reference["joints_m"] @ rotation_lab_to_camera.T + translation_m[None, None, :]
    wham_camera_like = pre_axis_camera.copy()
    wham_camera_like[:, :, 1] *= -1.0
    camera_path = tmp_path / "cameraIntrinsicsExtrinsics.pickle"
    with camera_path.open("wb") as f:
        pickle.dump({"rotation": rotation_lab_to_camera, "translation": translation_mm.reshape(3, 1)}, f)
    run_config = {"manifest_summary": {"calibration": {"intrinsics_extrinsics": str(camera_path)}}}

    full_report, _ = compare_pose_to_opensim_reference(
        _wham_pose(wham_camera_like),
        reference,
        run_config=run_config,
        convention_profile=OPENCAP_CAMERA_FULL_PROFILE,
        timeline_report=_wham_timeline(),
    )
    rotation_only_report, _ = compare_pose_to_opensim_reference(
        _wham_pose(wham_camera_like),
        reference,
        run_config=run_config,
        convention_profile=OPENCAP_CAMERA_PROFILE,
        timeline_report=_wham_timeline(),
    )

    assert full_report["uses_camera_translation"] is True
    assert full_report["camera_translation_units"] == "mm_to_m_inferred"
    assert full_report["primary_root_centered_mpjpe_mm"] < 1e-3
    assert full_report["global_no_align_mpjpe_mm"] < 1e-3
    assert rotation_only_report["primary_root_centered_mpjpe_mm"] < 1e-3
    assert rotation_only_report["global_no_align_mpjpe_mm"] > 1000.0


def test_wham_calibrated_convention_requires_camera_calibration() -> None:
    reference = _reference()
    wham_camera_like = reference["joints_m"].copy()
    wham_camera_like[:, :, 1] *= -1.0
    pose = _wham_pose(wham_camera_like)
    legacy, _ = compare_pose_to_opensim_reference(pose, reference, timeline_report=_wham_timeline())
    assert legacy["convention_profile"] == LEGACY_PROFILE

    with pytest.raises(ValueError, match="calibration"):
        compare_pose_to_opensim_reference(
            pose,
            reference,
            convention_profile=OPENCAP_CAMERA_PROFILE,
            timeline_report=_wham_timeline(),
        )


def test_wham_lr_convention_swaps_names_not_numeric_arrays(tmp_path: Path) -> None:
    names = ["pelvis", "left_hip", "right_knee", "lank", "rtoe"]
    assert swap_left_right_names(names) == ["pelvis", "right_hip", "left_knee", "rank", "ltoe"]
    values = np.arange(2 * len(names) * 3, dtype=float).reshape(2, len(names), 3)
    camera_path = tmp_path / "cameraIntrinsicsExtrinsics.pickle"
    with camera_path.open("wb") as f:
        pickle.dump({"rotation": np.eye(3)}, f)
    run_config = {"manifest_summary": {"calibration": {"intrinsics_extrinsics": str(camera_path)}}}

    base_names, base_values, _base_meta = apply_level_a_convention(
        "wham",
        names,
        values,
        "x,-y,z",
        run_config=run_config,
        convention_profile=OPENCAP_CAMERA_PROFILE,
    )
    lr_names, lr_values, lr_meta = apply_level_a_convention(
        "wham",
        names,
        values,
        "x,-y,z",
        run_config=run_config,
        convention_profile=OPENCAP_CAMERA_LR_PROFILE,
    )

    assert base_names == names
    assert lr_names == swap_left_right_names(names)
    assert np.allclose(lr_values, base_values)
    assert lr_meta["left_right_swap"] is True
    assert lr_meta["diagnostic_only"] is True


def test_wham_yaw180_candidate_is_proper_rotation() -> None:
    assert axis_expr_determinant("-x,y,-z") == pytest.approx(1.0)

    names = ["pelvis", "left_hip", "right_hip"]
    values = np.arange(2 * len(names) * 3, dtype=float).reshape(2, len(names), 3)
    run_config = {
        "config": {
            "level_a": {
                "wham_convention_profiles": {
                    "yaw180_test": {
                        "label": "Yaw 180 test",
                        "pre_axis": "identity",
                        "post_axis": "-x,y,-z",
                        "diagnostic_only": True,
                    }
                }
            }
        }
    }

    _out_names, _out_values, meta = apply_level_a_convention(
        "wham",
        names,
        values,
        "identity",
        run_config=run_config,
        convention_profile="yaw180_test",
    )

    assert meta["post_transform_determinant"] == pytest.approx(1.0)
    assert meta["proper_post_transform"] is True
    assert meta["diagnostic_only"] is True


def test_wham_single_axis_mirror_is_forced_diagnostic_only() -> None:
    assert axis_expr_determinant("-x,y,z") == pytest.approx(-1.0)

    names = ["pelvis", "left_hip", "right_hip"]
    values = np.zeros((2, len(names), 3), dtype=float)
    run_config = {
        "config": {
            "level_a": {
                "wham_convention_profiles": {
                    "single_axis_mirror": {
                        "label": "Single-axis mirror",
                        "pre_axis": "identity",
                        "post_axis": "-x,y,z",
                        "diagnostic_only": False,
                    }
                }
            }
        }
    }

    _out_names, _out_values, meta = apply_level_a_convention(
        "wham",
        names,
        values,
        "identity",
        run_config=run_config,
        convention_profile="single_axis_mirror",
    )

    assert meta["post_transform_determinant"] == pytest.approx(-1.0)
    assert meta["proper_post_transform"] is False
    assert meta["diagnostic_only"] is True
    assert meta["default_eligible"] is False


def test_wham_time_offset_profile_shifts_native_and_100hz_timestamps() -> None:
    reference = _reference(frame_count=30)
    wham_camera_like = reference["joints_m"].copy()
    wham_camera_like[:, :, 1] *= -1.0
    run_config = {
        "config": {
            "level_a": {
                "wham_convention_profiles": {
                    "offset120": {
                        "label": "Offset 120 ms",
                        "pre_axis": "x,-y,z",
                        "time_offset_s": 0.120,
                        "diagnostic_only": True,
                    }
                }
            }
        }
    }

    native_report, native_series = compare_pose_to_opensim_reference(
        _wham_pose(wham_camera_like),
        reference,
        run_config=run_config,
        convention_profile="offset120",
        timeline_report=_wham_timeline(frame_count=30),
    )
    hz_report, hz_series = compare_pose_to_opensim_reference(
        _wham_pose(wham_camera_like),
        reference,
        run_config=run_config,
        convention_profile="offset120",
        timeline_report=_wham_timeline(frame_count=30),
        evaluation_hz=100.0,
    )

    assert native_report["time_offset_s"] == pytest.approx(0.120)
    assert native_report["time_start_s"] == pytest.approx(native_series["time_s"][0])
    assert native_report["time_start_s"] == pytest.approx(8 / 60.0 - 0.120)
    assert native_report["timebase"]["raw_frame_ids"][0] == 8
    assert hz_report["time_offset_s"] == pytest.approx(0.120)
    assert hz_report["evaluation_hz"] == pytest.approx(100.0)
    assert np.allclose(np.diff(hz_series["time_s"]), 0.01)


def test_backend_run_name_adds_backend_suffix() -> None:
    manifest = {"subject_id": "subject7", "session": "Session1", "camera": "Cam1"}
    assert backend_run_name(manifest, "walking1", "wham") == "subject7_Session1_Cam1_walking1__wham"


def test_level_a_aggregation_ranks_and_ties() -> None:
    rows = [
        _row("metrabs", "walking1", 100.0),
        _row("metrabs", "walking2", 110.0),
        _row("rtmw3d", "walking1", 104.0),
        _row("rtmw3d", "walking2", 112.0),
        _row("wham", "walking1", 160.0),
        _row("wham", "walking2", 170.0),
    ]
    agg = aggregate_level_a(rows, tie_threshold_mm=5.0)
    assert agg["status"] == "ok"
    assert agg["backends"]["metrabs"]["rank"] == 1
    assert agg["backends"]["rtmw3d"]["rank"] == 1
    assert agg["backends"]["wham"]["rank"] == 3


def test_level_a_aggregation_warns_when_some_backend_rows_fail() -> None:
    rows = [_row("metrabs", "walking1", 100.0), {"backend": "wham", "trial": "walking1", "status": "pipeline_failed"}]
    agg = aggregate_level_a(rows)
    assert agg["status"] == "warning"
    assert agg["failure_count"] == 1


def test_benchmark_runner_uses_wham_python_for_wham() -> None:
    args = Namespace(wham_python=Path("/envs/wham/bin/python"))
    assert _pipeline_python(args, "wham") == Path("/envs/wham/bin/python")
    assert _pipeline_python(args, "metrabs") == Path(sys.executable)


def test_level_a_cli_skip_inference_with_fake_cache(tmp_path: Path) -> None:
    repo_root = tmp_path
    manifest = tmp_path / "manifest.yaml"
    paths = tmp_path / "paths.yaml"
    paths.write_text("datasets:\n  opencap_root: /tmp/opencap\n", encoding="utf-8")
    manifest.write_text(
        "\n".join(
            [
                "subject_id: subject7",
                "session: Session1",
                "camera: Cam1",
                "paths:",
                "  root: ${datasets.opencap_root}/subject7",
                "trials:",
                "  healthy:",
                "    - id: walking1",
                "      video_sync: ${paths.root}/walking1.avi",
            ]
        ),
        encoding="utf-8",
    )
    out_dir = repo_root / "monocap_v2" / "benchmarks" / "level_a"
    ref_dir = out_dir / "reference"
    ref_dir.mkdir(parents=True)
    reference = _reference()
    np.savez_compressed(ref_dir / "opensim_fk_walking1.npz", time_s=reference["time_s"], joints_m=reference["joints_m"], joint_names=np.asarray(reference["joint_names"]))
    (ref_dir / "opensim_fk_walking1.json").write_text(json.dumps({"status": "ok", "ik_marker_errors": {"rms_median_mm": 12.0}}), encoding="utf-8")
    run_dir = repo_root / "monocap_v2" / "runs" / "subject7_Session1_Cam1_walking1__rtmw3d"
    (run_dir / "pose3d_initial").mkdir(parents=True)
    with (run_dir / "pose3d_initial" / "pose3d_initial.pkl").open("wb") as f:
        pickle.dump(_pose(reference["joints_m"], backend="rtmw3d", axis_space="identity"), f)
    (run_dir / "run_config.yaml").write_text("config: {}\n", encoding="utf-8")

    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_backends_level_a.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest),
            "--paths",
            str(paths),
            "--trials",
            "walking1",
            "--backends",
            "rtmw3d",
            "--skip-inference",
            "--out",
            str(out_dir),
            "--repo-root",
            str(repo_root),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((out_dir / "level_a_summary.json").read_text(encoding="utf-8"))
    assert summary["aggregate"]["backends"]["rtmw3d"]["median_primary_root_centered_mpjpe_mm"] < 1e-3
    assert (out_dir / "per_backend_trial_metrics.csv").exists()


def test_opensim_fk_visualization_writes_preview(tmp_path: Path) -> None:
    reference = _reference()
    mp4 = tmp_path / "opensim_fk_preview.mp4"
    png = tmp_path / "opensim_fk_frame.png"
    report = render_opensim_fk_preview(reference, mp4, png, fps=5.0, width=640, height=360, max_frames=3)
    assert report["status"] == "ok"
    assert report["frames_rendered"] == 3
    assert mp4.exists()
    assert mp4.stat().st_size > 0
    assert png.exists()
    assert png.stat().st_size > 0


def test_level_a_overlay_visualization_writes_preview(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "bench"
    run_dir = tmp_path / "run"
    ref = _reference()
    (benchmark_dir / "reference").mkdir(parents=True)
    np.savez_compressed(
        benchmark_dir / "reference" / "opensim_fk_walking1.npz",
        time_s=ref["time_s"],
        joints_m=ref["joints_m"],
        joint_names=np.asarray(ref["joint_names"]),
    )
    (benchmark_dir / "reference" / "opensim_fk_walking1.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    (run_dir / "pose3d_initial").mkdir(parents=True)
    with (run_dir / "pose3d_initial" / "pose3d_initial.pkl").open("wb") as f:
        pickle.dump(_pose(ref["joints_m"], backend="rtmw3d", axis_space="identity"), f)
    (run_dir / "run_config.yaml").write_text("config: {}\n", encoding="utf-8")
    (benchmark_dir / "level_a_summary.json").write_text(
        json.dumps({"rows": [{"backend": "rtmw3d", "trial": "walking1", "status": "valid", "run_dir": str(run_dir)}]}),
        encoding="utf-8",
    )
    mp4 = benchmark_dir / "vis.mp4"
    png = benchmark_dir / "vis.png"
    qc = benchmark_dir / "vis.json"
    report = render_level_a_overlay(benchmark_dir, "walking1", ["rtmw3d"], mp4, png, qc, preview_fps=5.0, width=800, height=420)
    assert report["status"] == "ok"
    assert "pa_mpjpe_per_frame_similarity" in report["panels"]
    assert mp4.exists() and mp4.stat().st_size > 0
    assert png.exists() and png.stat().st_size > 0


def _reference(frame_count: int = 6) -> dict:
    time = np.arange(frame_count, dtype=float) / 60.0
    base = np.array(
        [
            [0.0, 0.0, 1.0],
            [-0.1, 0.0, 0.9],
            [0.1, 0.0, 0.9],
            [-0.1, 0.0, 0.5],
            [0.1, 0.0, 0.5],
            [-0.1, 0.0, 0.1],
            [0.1, 0.0, 0.1],
            [-0.1, 0.1, 0.0],
            [0.1, 0.1, 0.0],
        ],
        dtype=float,
    )
    joints = np.repeat(base[None, :, :], len(time), axis=0)
    joints[:, :, 1] += np.linspace(0.0, 0.05, len(time))[:, None]
    return {
        "time_s": time,
        "joints_m": joints,
        "joint_names": [
            "pelvis",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_mtp",
            "right_mtp",
        ],
        "source": "synthetic",
    }


def _pose(joints: np.ndarray, backend: str, axis_space: str) -> dict:
    if axis_space != "identity":
        raise ValueError(axis_space)
    return {
        "representation": "joints",
        "backend": backend,
        "fps": 60.0,
        "units": "m",
        "joint_names": [
            "pelvis",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_mtp",
            "right_mtp",
        ],
        "joints_3d": np.asarray(joints, dtype=np.float32),
    }


def _wham_pose(joints: np.ndarray) -> dict:
    pose = _pose(joints, backend="wham", axis_space="identity")
    pose["representation"] = "hybrid"
    pose["backend_meta"] = {"raw_frame_ids": list(range(np.asarray(joints).shape[0])), "frame_ids": list(range(np.asarray(joints).shape[0]))}
    pose["smpl"] = {"model_type": "smpl", "betas": np.zeros(10)}
    return pose


def _wham_timeline(frame_count: int = 6) -> dict:
    return {
        "status": "ok",
        "raw_sync_alignment": {"status": "ok", "best_raw_offset": 0, "sync_frame_count": frame_count, "sync_fps": 60.0},
        "overlap": {"status": "ok"},
    }


def _row(backend: str, trial: str, primary: float) -> dict:
    return {
        "backend": backend,
        "trial": trial,
        "status": "valid",
        "primary_root_centered_mpjpe_mm": primary,
    }
