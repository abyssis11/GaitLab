from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from monocap_v2.core.benchmark import aggregate_trials, parse_mocap_validation_report, parse_stage07_report


def test_parse_stage07_report_valid(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_report(run_dir, initial=220.0, refined=215.0)
    _write_camera(run_dir)
    row = parse_stage07_report(run_dir, "walking1")
    assert row["status"] == "valid"
    assert row["camera_source"] == "opencap_manifest"
    assert row["camera_distortion_enabled"] is True
    assert row["initial_primary_mpjpe_mm"] == 220.0
    assert row["refined_primary_mpjpe_mm"] == 215.0
    assert row["primary_improvement_mm"] == 5.0
    assert row["foot_speed_contact_before_mps"] == 0.8


def test_aggregate_status_ok_warning_failed() -> None:
    ok_rows = [
        _row("walking1", 220.0, 215.0),
        _row("walking2", 230.0, 226.0),
        _row("walking3", 210.0, 209.0),
    ]
    assert aggregate_trials(ok_rows)["status"] == "ok"

    warning_rows = [
        _row("walking1", 220.0, 210.0),
        _row("walking2", 230.0, 225.0),
        _row("walking3", 210.0, 214.0),
    ]
    warning = aggregate_trials(warning_rows, regression_threshold_mm=2.0)
    assert warning["status"] == "warning"
    assert warning["large_regression_trials"] == ["walking3"]

    failed_rows = [
        _row("walking1", 220.0, 225.0),
        _row("walking2", 230.0, 235.0),
        _row("walking3", 210.0, 212.0),
    ]
    assert aggregate_trials(failed_rows)["status"] == "failed"


def test_benchmark_cli_skip_inference_smoke(tmp_path: Path) -> None:
    repo_root = tmp_path
    manifest = tmp_path / "manifest.yaml"
    paths = tmp_path / "paths.yaml"
    paths.write_text("datasets:\n  opencap_root: /tmp/opencap\noutputs_root: /tmp/out\n", encoding="utf-8")
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
    run_dir = repo_root / "monocap_v2" / "runs" / "subject7_Session1_Cam1_walking1"
    _write_report(run_dir, initial=220.0, refined=219.0)
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_opencap.py"
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
            "--activity",
            "walking",
            "--skip-inference",
            "--repo-root",
            str(repo_root),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (repo_root / "monocap_v2" / "benchmarks" / "subject7_walking" / "benchmark_summary.json").exists()


def test_parse_mocap_validation_report_uses_normal_mpjpe(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_mocap_validation_report(run_dir, initial=700.0, refined=700.0)
    row = parse_mocap_validation_report(run_dir, "walking1")
    assert row["status"] == "valid"
    assert row["report_source"] == "mocap_validation"
    assert row["initial_primary_mpjpe_mm"] == 700.0
    assert row["refined_root_centered_rigid_mpjpe_mm"] == 220.0
    assert row["refined_pa_mpjpe_mm"] == 150.0


def test_parse_mocap_validation_report_rejects_wrong_backend(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_mocap_validation_report(run_dir, initial=700.0, refined=700.0, backend="metrabs")
    row = parse_mocap_validation_report(run_dir, "walking1", expected_backend="wham")
    assert row["status"] == "backend_mismatch"
    assert "Expected pose3d backend" in row["error"]


def test_benchmark_cli_mocap_validation_supports_separate_output(tmp_path: Path) -> None:
    repo_root = tmp_path
    manifest = tmp_path / "manifest.yaml"
    paths = tmp_path / "paths.yaml"
    paths.write_text("datasets:\n  opencap_root: /tmp/opencap\noutputs_root: /tmp/out\n", encoding="utf-8")
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
    run_dir = repo_root / "monocap_v2" / "runs" / "subject7_Session1_Cam1_walking1"
    _write_mocap_validation_report(run_dir, initial=700.0, refined=700.0)
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_opencap.py"
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
            "--skip-inference",
            "--report-source",
            "mocap_validation",
            "--out",
            "monocap_v2/benchmarks/subject7_walking_wham",
            "--repo-root",
            str(repo_root),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = repo_root / "monocap_v2" / "benchmarks" / "subject7_walking_wham" / "benchmark_summary.json"
    assert summary.exists()
    assert json.loads(summary.read_text())["aggregate"]["aggregation_mode"] == "descriptive"


def _row(trial: str, initial: float, refined: float) -> dict:
    return {
        "trial": trial,
        "status": "valid",
        "initial_primary_mpjpe_mm": initial,
        "refined_primary_mpjpe_mm": refined,
        "primary_improvement_mm": initial - refined,
    }


def _write_report(run_dir: Path, initial: float, refined: float) -> None:
    report_dir = run_dir / "optimization"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "ok",
        "metrics_before": {"mean_foot_speed_during_contact_mps": 0.8},
        "metrics_after": {"mean_foot_speed_during_contact_mps": 0.5},
        "warnings": [],
        "mocap_evaluation": {
            "status": "ok",
            "primary_improvement_mm": initial - refined,
            "initial": {
                "primary_root_centered_rigid_mpjpe_mm": initial,
                "sequence_similarity_mpjpe_mm": initial + 20.0,
                "pa_similarity_mpjpe_mm": initial - 40.0,
            },
            "refined": {
                "primary_root_centered_rigid_mpjpe_mm": refined,
                "sequence_similarity_mpjpe_mm": refined + 20.0,
                "pa_similarity_mpjpe_mm": refined - 40.0,
            },
        },
    }
    (report_dir / "opt_stage2_report.json").write_text(json.dumps(report), encoding="utf-8")


def _write_camera(run_dir: Path) -> None:
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "camera_qc.json").write_text(
        json.dumps(
            {
                "camera_source": "opencap_manifest",
                "mode": "opencap_manifest",
                "is_assumed": False,
                "distortion_enabled": True,
            }
        ),
        encoding="utf-8",
    )


def _write_mocap_validation_report(run_dir: Path, initial: float, refined: float, backend: str = "wham") -> None:
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "warning",
        "backend": backend,
        "warnings": ["Absolute transform unavailable."],
        "initial": {
            "normal_root_centered_mpjpe_mm": initial,
            "root_centered_rigid_mpjpe_mm": 220.0,
            "pa_mpjpe_mm": 150.0,
            "global_sequence_similarity_mpjpe_mm": 290.0,
        },
        "refined": {
            "normal_root_centered_mpjpe_mm": refined,
            "root_centered_rigid_mpjpe_mm": 220.0,
            "pa_mpjpe_mm": 150.0,
            "global_sequence_similarity_mpjpe_mm": 290.0,
        },
    }
    (report_dir / "mocap_validation.json").write_text(json.dumps(report), encoding="utf-8")
