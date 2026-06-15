from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from monocap_v2.core.wham_convention_matrix import (
    VISUAL_CANDIDATE_PROFILE,
    aggregate_matrix_rows,
    generate_matrix_candidates,
    matrix_profiles_for_viewer,
    run_wham_convention_matrix_audit,
    signed_axis_expressions,
    time_offsets,
)
from monocap_v2.core.wham_conventions import axis_expr_determinant
from monocap_v2.core.wham_gt_interactive import build_wham_gt_payload


def test_signed_axis_expressions_cover_48_unique_transforms() -> None:
    exprs = signed_axis_expressions()
    assert len(exprs) == 48
    assert len(set(exprs)) == 48
    dets = [round(axis_expr_determinant(expr)) for expr in exprs]
    assert dets.count(1) == 24
    assert dets.count(-1) == 24
    assert axis_expr_determinant("-x,y,-z") == pytest.approx(1.0)
    assert axis_expr_determinant("-x,y,z") == pytest.approx(-1.0)


def test_matrix_candidate_generation_marks_diagnostic_candidates() -> None:
    candidates = generate_matrix_candidates("physical", time_offsets(0.0, 0.12, 0.12))
    lookup = {candidate["profile"]: candidate for candidate in candidates}

    assert VISUAL_CANDIDATE_PROFILE in lookup
    visual = lookup[VISUAL_CANDIDATE_PROFILE]
    assert visual["time_offset_s"] == pytest.approx(0.12)
    assert visual["left_right_swap"] is True
    assert visual["post_transform_determinant"] == pytest.approx(1.0)
    assert visual["diagnostic_only"] is True
    assert visual["promotable_by_definition"] is False

    full = generate_matrix_candidates("full", [0.0])
    single_axis = next(candidate for candidate in full if candidate["pre_axis"] == "-x,y,z" and not candidate["left_right_swap"])
    assert single_axis["linear_transform_determinant_without_camera"] == pytest.approx(-1.0)
    assert single_axis["diagnostic_only"] is True


def test_matrix_rankings_separate_normal_rigid_pa_and_stable() -> None:
    rows = [
        _row("normal_winner", "walking1", normal=40.0, rigid=90.0, pa=90.0, gap=5.0),
        _row("normal_winner", "walking2", normal=44.0, rigid=92.0, pa=91.0, gap=6.0),
        _row("rigid_winner", "walking1", normal=100.0, rigid=20.0, pa=30.0, gap=80.0),
        _row("rigid_winner", "walking2", normal=110.0, rigid=22.0, pa=31.0, gap=88.0),
        _row("pa_winner", "walking1", normal=120.0, rigid=40.0, pa=10.0, gap=80.0),
        _row("pa_winner", "walking2", normal=130.0, rigid=42.0, pa=11.0, gap=88.0),
    ]
    aggregate = aggregate_matrix_rows(rows, expected_trials=["walking1", "walking2"])

    assert aggregate["rankings"]["normal"][0]["profile"] == "normal_winner"
    assert aggregate["rankings"]["rigid"][0]["profile"] == "rigid_winner"
    assert aggregate["rankings"]["pa"][0]["profile"] == "pa_winner"
    assert aggregate["rankings"]["stable"][0]["profile"] == "normal_winner"


def test_matrix_audit_runs_on_cached_wham_artifacts_and_viewer_loads_candidates(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmark"
    run_dir = tmp_path / "run"
    reference = _reference(frame_count=30)
    camera_path = tmp_path / "cameraIntrinsicsExtrinsics.pickle"
    with camera_path.open("wb") as f:
        pickle.dump({"rotation": np.eye(3), "translation": np.zeros((3, 1))}, f)

    (benchmark_dir / "reference").mkdir(parents=True)
    np.savez_compressed(
        benchmark_dir / "reference" / "opensim_fk_walking1.npz",
        time_s=reference["time_s"],
        joints_m=reference["joints_m"],
        joint_names=np.asarray(reference["joint_names"]),
    )
    (benchmark_dir / "reference" / "opensim_fk_walking1.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    (benchmark_dir / "level_a_summary.json").write_text(
        json.dumps({"rows": [{"backend": "wham", "trial": "walking1", "status": "valid", "run_dir": str(run_dir)}]}),
        encoding="utf-8",
    )

    (run_dir / "pose3d_initial").mkdir(parents=True)
    wham_camera_like = reference["joints_m"].copy()
    wham_camera_like[:, :, 1] *= -1.0
    with (run_dir / "pose3d_initial" / "pose3d_initial.pkl").open("wb") as f:
        pickle.dump(_wham_pose(wham_camera_like), f)
    (run_dir / "pose3d_initial" / "pose3d_initial_qc.json").write_text(json.dumps({"wham_timeline": _wham_timeline(frame_count=30)}), encoding="utf-8")
    (run_dir / "run_config.yaml").write_text(
        f"manifest_summary:\n  calibration:\n    intrinsics_extrinsics: {camera_path}\nconfig: {{}}\n",
        encoding="utf-8",
    )

    report = run_wham_convention_matrix_audit(
        benchmark_dir,
        trials=["walking1"],
        profile_set="physical",
        evaluation_hz_values=[None, 100.0],
        time_offset_min=0.0,
        time_offset_max=0.12,
        time_offset_step=0.12,
        out_dir=benchmark_dir / "matrix",
    )

    assert report["status"] == "ok"
    outputs = report["outputs"]
    for key in ["json", "summary_md", "csv", "top_normal_csv", "top_rigid_csv", "top_pa_csv", "top_stable_csv"]:
        assert Path(outputs[key]).exists()
    assert Path(outputs["plot"]).exists()
    assert report["aggregate"]["rankings"]["normal"]
    assert report["viewer_candidates"]["presets"]["best_normal"]

    profiles, presets = matrix_profiles_for_viewer(Path(outputs["json"]), evaluation_hz=100.0)
    assert profiles
    assert presets["visual_candidate"] == VISUAL_CANDIDATE_PROFILE

    payload = build_wham_gt_payload(
        benchmark_dir,
        "walking1",
        evaluation_hz=100.0,
        matrix_audit_path=Path(outputs["json"]),
    )
    assert payload["matrix_presets"]["best_normal"]
    assert any(option["profile"] == payload["matrix_presets"]["best_normal"] for option in payload["convention_profiles"])


def _row(profile: str, trial: str, normal: float, rigid: float, pa: float, gap: float) -> dict:
    return {
        "status": "ok",
        "profile": profile,
        "label": profile,
        "trial": trial,
        "evaluation_label": "native",
        "left_right_swap": False,
        "time_offset_s": 0.0,
        "uses_camera_translation": False,
        "diagnostic_only": False,
        "promotable": True,
        "proper_post_transform": True,
        "proper_linear_transform": True,
        "linear_transform_determinant": 1.0,
        "primary_root_centered_mpjpe_mm": normal,
        "root_centered_rigid_mpjpe_mm": rigid,
        "pa_mpjpe_mm": pa,
        "global_no_align_mpjpe_mm": normal,
        "normal_minus_rigid_gap_mm": gap,
    }


def _reference(frame_count: int = 6) -> dict:
    time = np.arange(frame_count, dtype=float) / 60.0
    base = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [-0.1, 0.0, 0.9],
            [0.1, 0.0, 0.9],
            [-0.1, 0.0, 0.5],
            [0.1, 0.0, 0.5],
            [-0.1, 0.0, 0.1],
            [0.1, 0.0, 0.1],
        ],
        dtype=float,
    )
    joints = np.repeat(base[None, :, :], len(time), axis=0)
    joints[:, :, 1] += np.linspace(0.0, 0.05, len(time))[:, None]
    return {
        "time_s": time,
        "joints_m": joints,
        "joint_names": ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
        "source": "synthetic",
    }


def _wham_pose(joints: np.ndarray) -> dict:
    return {
        "representation": "hybrid",
        "backend": "wham",
        "fps": 60.0,
        "units": "m",
        "joint_names": ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
        "joints_3d": np.asarray(joints, dtype=np.float32),
        "backend_meta": {"raw_frame_ids": list(range(np.asarray(joints).shape[0])), "frame_ids": list(range(np.asarray(joints).shape[0]))},
        "smpl": {"model_type": "smpl", "betas": np.zeros(10)},
    }


def _wham_timeline(frame_count: int = 6) -> dict:
    return {
        "status": "ok",
        "raw_sync_alignment": {"status": "ok", "best_raw_offset": 0, "sync_frame_count": frame_count, "sync_fps": 60.0},
        "overlap": {"status": "ok"},
    }
