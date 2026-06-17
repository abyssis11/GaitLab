from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from monocap_v2.core import backend_convention_audit as audit


def test_axis_candidate_generation_returns_48_signed_permutations() -> None:
    axes = audit.generate_axis_candidates()

    assert len(axes) == 48
    assert len(set(axes)) == 48
    assert axes[:2] == ["x,-y,z", "x,-y,-z"]
    dets = [audit.axis_expr_determinant(expr) for expr in axes]
    assert sum(1 for det in dets if det == pytest.approx(1.0)) == 24
    assert sum(1 for det in dets if det == pytest.approx(-1.0)) == 24


def test_physical_axis_candidates_are_no_permutation_subset() -> None:
    axes = audit.generate_physical_axis_candidates()

    assert axes[:2] == ["x,-y,z", "x,-y,-z"]
    assert len(axes) == 8
    assert set(axes).issubset(set(audit.generate_axis_candidates()))
    assert all(expr.replace("-", "").split(",") == ["x", "y", "z"] for expr in axes)


def test_time_offset_generation_uses_requested_step() -> None:
    offsets = audit.generate_time_offsets(-0.30, 0.30, 0.02)

    assert len(offsets) == 31
    assert offsets[0] == pytest.approx(-0.30)
    assert offsets[15] == pytest.approx(0.0)
    assert offsets[-1] == pytest.approx(0.30)


def test_model_left_right_relabel_changes_names_not_numeric_arrays() -> None:
    joints = np.arange(2 * 7 * 3, dtype=float).reshape(2, 7, 3)
    pose = {
        "joint_names": ["pelvis", "left_hip", "right_hip", "lkne", "rkne", "lank", "rank"],
        "joints_3d": joints,
        "pose2d": {"names": ["left_knee", "right_knee"], "xy": np.zeros((2, 2, 2))},
    }

    relabeled = audit.relabel_pose_left_right(pose)

    assert relabeled["joint_names"] == ["pelvis", "right_hip", "left_hip", "rkne", "lkne", "rank", "lank"]
    assert relabeled["joints_3d"] is joints
    assert np.shares_memory(relabeled["joints_3d"], pose["joints_3d"])
    assert relabeled["pose2d"]["names"] == ["right_knee", "left_knee"]
    assert relabeled["pose2d"]["xy"] is pose["pose2d"]["xy"]


def test_backend_convention_audit_writes_outputs_with_fake_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark_dir = tmp_path / "bench"
    benchmark_dir.mkdir()
    run_dir = tmp_path / "run__wham"
    _write_summary(benchmark_dir, run_dir)

    def fake_load_pose(path: Path) -> dict:
        return {
            "backend": "metrabs",
            "joint_names": ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
            "joints_3d": np.zeros((10, 7, 3), dtype=float),
            "fps": 60.0,
        }

    def fake_compare(
        pose: dict,
        reference: dict,
        run_config: dict | None = None,
        axis_map: dict | None = None,
        evaluation_hz: float | None = None,
        diagnostic_time_offset_s: float | None = None,
        swap_reference_lr: bool = False,
        **_kwargs,
    ):
        axis = next(iter((axis_map or {"wham": "identity"}).values()))
        model_lr = str(pose["joint_names"][1]).startswith("right")
        offset = float(diagnostic_time_offset_s or 0.0)
        raw = 200.0
        if axis == "x,-y,z":
            raw -= 40.0
        if swap_reference_lr:
            raw -= 20.0
        if model_lr:
            raw -= 10.0
        raw += abs(offset) * 100.0
        if evaluation_hz == 100.0:
            raw += 1.0
        report = {
            "status": "ok",
            "primary_root_centered_mpjpe_mm": raw,
            "root_centered_rigid_mpjpe_mm": raw * 0.75,
            "pa_mpjpe_mm": raw * 0.50,
            "normal_minus_rigid_gap_mm": raw * 0.25,
            "overlap_frames": 10,
            "time_start_s": 0.0,
            "time_end_s": 0.15,
            "warnings": [],
        }
        return report, {}

    monkeypatch.setattr(audit, "load_pose_artifact", fake_load_pose)
    monkeypatch.setattr(audit, "load_run_config", lambda _path: {})
    monkeypatch.setattr(audit, "load_cached_wham_timeline", lambda _path: None)
    monkeypatch.setattr(
        audit,
        "load_opensim_reference",
        lambda *_args, **_kwargs: {
            "time_s": np.arange(10) / 60.0,
            "joint_names": ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
            "joints_m": np.zeros((10, 7, 3), dtype=float),
        },
    )
    monkeypatch.setattr(audit, "compare_pose_to_opensim_reference", fake_compare)

    report = audit.run_backend_convention_audit(
        benchmark_dirs={"Cam1": benchmark_dir},
        trials=["walking1"],
        backends=["wham", "metrabs"],
        evaluation_hz_values=[60.0, 100.0],
        out_dir=tmp_path / "audit",
        time_offset_min=0.0,
        time_offset_max=0.02,
        time_offset_step=0.02,
        axis_candidates=["x,-y,z", "x,-y,-z"],
    )

    out_dir = tmp_path / "audit"
    assert report["candidate_count"] == 33
    assert (out_dir / "convention_audit_rows.csv").exists()
    assert (out_dir / "top_by_case.csv").exists()
    assert (out_dir / "stability_summary.json").exists()
    assert (out_dir / "stability_summary.md").exists()
    assert (out_dir / "walking_trial_tables.md").exists()

    rows = list(csv.DictReader((out_dir / "convention_audit_rows.csv").open()))
    valid = [row for row in rows if row["status"] == "valid"]
    failed = [row for row in rows if row["status"] != "valid"]
    assert len(valid) == 32
    assert len(failed) == 1
    assert any(row["reference_left_right_swap"] == "True" and row["promotable_candidate"] == "False" for row in valid)
    assert any(row["time_offset_s"] == "0.02" and row["promotable_candidate"] == "False" for row in valid)

    summary = json.loads((out_dir / "stability_summary.json").read_text())
    assert summary["valid_candidate_count"] == 32
    assert summary["failed_candidate_count"] == 1
    assert summary["best_pa_case_count"] == 2


def test_backend_convention_audit_can_compare_initial_and_refined_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark_dir = tmp_path / "bench"
    benchmark_dir.mkdir()
    run_dir = tmp_path / "run__metrabs"
    payload = {
        "rows": [
            {"backend": "metrabs_initial", "trial": "walking1", "status": "valid", "run_dir": str(run_dir)},
            {"backend": "metrabs_refined", "trial": "walking1", "status": "valid", "run_dir": str(run_dir)},
        ]
    }
    (benchmark_dir / "level_a_summary.json").write_text(json.dumps(payload), encoding="utf-8")

    pose = {
        "backend": "metrabs",
        "joint_names": ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
        "joints_3d": np.zeros((10, 7, 3), dtype=float),
        "fps": 60.0,
    }
    reference = {
        "time_s": np.arange(10) / 60.0,
        "joint_names": ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
        "joints_m": np.zeros((10, 7, 3), dtype=float),
    }

    monkeypatch.setattr(audit, "_load_pose_for_source", lambda _path, source: {**pose, "metadata": {"pose_source": source}})
    monkeypatch.setattr(audit, "load_run_config", lambda _path: {})
    monkeypatch.setattr(audit, "load_cached_wham_timeline", lambda _path: None)
    monkeypatch.setattr(audit, "load_opensim_reference", lambda *_args, **_kwargs: reference)
    monkeypatch.setattr(
        audit,
        "compare_pose_to_opensim_reference",
        lambda *_args, **_kwargs: (
            {
                "status": "ok",
                "root_centered_rigid_mpjpe_mm": 0.0,
                "pa_mpjpe_mm": 0.0,
            },
            {},
        ),
    )

    report = audit.run_backend_convention_audit(
        benchmark_dirs={"Cam0": benchmark_dir},
        trials=["walking1"],
        backends=["metrabs"],
        evaluation_hz_values=[60.0],
        out_dir=tmp_path / "audit",
        time_offset_min=0.0,
        time_offset_max=0.0,
        time_offset_step=0.02,
        axis_candidates=["x,-y,z"],
        reference_lr_values=(False,),
        model_lr_values=(False,),
        pose_sources=("initial", "refined"),
    )

    rows = list(csv.DictReader((tmp_path / "audit" / "convention_audit_rows.csv").open()))
    assert report["valid_candidate_count"] == 2
    assert {row["pose_source"] for row in rows} == {"initial", "refined"}
    top_rows = list(csv.DictReader((tmp_path / "audit" / "top_by_case.csv").open()))
    assert {row["pose_source"] for row in top_rows} == {"initial", "refined"}


def test_wham_timeline_is_built_once_for_candidate_sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark_dir = tmp_path / "bench"
    benchmark_dir.mkdir()
    run_dir = tmp_path / "run__wham"
    _write_summary(benchmark_dir, run_dir)
    build_calls = []

    pose = {
        "backend": "wham",
        "joint_names": ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
        "joints_3d": np.zeros((10, 7, 3), dtype=float),
        "fps": 60.0,
        "backend_meta": {},
    }
    reference = {
        "time_s": np.arange(10) / 60.0,
        "joint_names": ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
        "joints_m": np.zeros((10, 7, 3), dtype=float),
    }
    timeline = {
        "status": "ok",
        "raw_sync_alignment": {"status": "ok", "best_raw_offset": 0, "sync_frame_count": 10, "sync_fps": 60.0},
        "overlap": {"status": "ok", "overlap_frames": 10},
    }

    monkeypatch.setattr(audit, "load_pose_artifact", lambda _path: pose)
    monkeypatch.setattr(audit, "load_run_config", lambda _path: {})
    monkeypatch.setattr(audit, "load_cached_wham_timeline", lambda _path: None)
    monkeypatch.setattr(audit, "load_opensim_reference", lambda *_args, **_kwargs: reference)

    def fake_build(_pose: dict, _cfg: dict) -> dict:
        build_calls.append(1)
        return timeline

    monkeypatch.setattr(audit, "build_wham_timeline_report", fake_build)

    audit.run_backend_convention_audit(
        benchmark_dirs={"Cam1": benchmark_dir},
        trials=["walking1"],
        backends=["wham"],
        evaluation_hz_values=[60.0],
        out_dir=tmp_path / "audit",
        time_offset_min=0.0,
        time_offset_max=0.02,
        time_offset_step=0.02,
        axis_candidates=["x,-y,z", "x,-y,-z"],
    )

    assert build_calls == [1]
    assert (run_dir / "reports" / "wham_timeline_qc.json").exists()


def _write_summary(benchmark_dir: Path, run_dir: Path) -> None:
    payload = {
        "rows": [
            {
                "backend": "wham",
                "trial": "walking1",
                "status": "valid",
                "run_dir": str(run_dir),
            }
        ]
    }
    (benchmark_dir / "level_a_summary.json").write_text(json.dumps(payload), encoding="utf-8")
