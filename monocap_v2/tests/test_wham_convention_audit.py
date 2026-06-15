from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from monocap_v2.core.wham_convention_audit import run_wham_convention_audit
from monocap_v2.core.wham_conventions import LEGACY_PROFILE, OPENCAP_CAMERA_PROFILE


def test_wham_convention_audit_runs_on_cached_artifacts(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmark"
    run_dir = tmp_path / "run"
    reference = _reference()
    camera_path = tmp_path / "cameraIntrinsicsExtrinsics.pickle"
    with camera_path.open("wb") as f:
        pickle.dump({"rotation": np.eye(3)}, f)

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
    (run_dir / "pose3d_initial" / "pose3d_initial_qc.json").write_text(json.dumps({"wham_timeline": _wham_timeline()}), encoding="utf-8")
    (run_dir / "run_config.yaml").write_text(
        f"manifest_summary:\n  calibration:\n    intrinsics_extrinsics: {camera_path}\nconfig: {{}}\n",
        encoding="utf-8",
    )

    report = run_wham_convention_audit(
        benchmark_dir,
        trials=["walking1"],
        profiles=[LEGACY_PROFILE, OPENCAP_CAMERA_PROFILE],
    )

    assert report["status"] == "ok"
    audit_dir = benchmark_dir / "audit"
    assert (audit_dir / "wham_convention_audit.json").exists()
    assert (audit_dir / "wham_convention_audit.csv").exists()
    assert (audit_dir / "wham_convention_audit.md").exists()
    csv_text = (audit_dir / "wham_convention_audit.csv").read_text(encoding="utf-8")
    assert LEGACY_PROFILE in csv_text
    assert OPENCAP_CAMERA_PROFILE in csv_text
    assert "opencap_camera_calibration" in csv_text


def _reference() -> dict:
    time = np.arange(6, dtype=float) / 60.0
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


def _wham_timeline() -> dict:
    return {
        "status": "ok",
        "raw_sync_alignment": {"status": "ok", "best_raw_offset": 0, "sync_frame_count": 6, "sync_fps": 60.0},
        "overlap": {"status": "ok"},
    }
