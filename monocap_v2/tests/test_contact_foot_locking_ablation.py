from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

from monocap_v2.core.level_a_benchmark import backend_run_name
from monocap_v2.core.logging_utils import write_json, write_yaml
from monocap_v2.scripts import benchmark_contact_foot_locking_ablation as script


def test_contact_foot_locking_ablation_runs_on_fake_cached_artifacts(tmp_path: Path, monkeypatch) -> None:
    manifest_path, paths_path, manifest = _manifest(tmp_path)
    run_dir = tmp_path / "monocap_v2" / "runs" / backend_run_name(manifest, "walking1", "metrabs")
    _write_pose_run(run_dir)
    ref_npz, ref_json = _write_reference(tmp_path)

    def fake_reference(*_args, **_kwargs):
        return {"npz": ref_npz, "json": ref_json}

    monkeypatch.setattr(script, "_ensure_reference", fake_reference)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_contact_foot_locking_ablation.py",
            "--manifest",
            str(manifest_path),
            "--paths",
            str(paths_path),
            "--trials",
            "walking1",
            "--backends",
            "metrabs",
            "--variants",
            "baseline,root_toe",
            "--repo-root",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
        ],
    )

    assert script.main() == 0
    assert (tmp_path / "out" / "contact_foot_locking_ablation.csv").exists()
    summary = json.loads((tmp_path / "out" / "contact_foot_locking_ablation.json").read_text(encoding="utf-8"))
    variants = {row["variant"] for row in summary["rows"]}
    assert variants == {"baseline", "root_toe"}
    assert all(row["status"] == "valid" for row in summary["rows"])


def _manifest(tmp_path: Path) -> tuple[Path, Path, dict]:
    root = tmp_path / "dataset" / "subjectX"
    manifest = {
        "subject_id": "subjectX",
        "session": "Session1",
        "camera": "Cam0",
        "session_metadata": str(root / "sessionMetadata.yaml"),
        "paths": {"root": "${datasets.opencap_root}/subjectX"},
        "trials": {"healthy": [{"id": "walking1", "video_sync": str(root / "video.avi")}]},
    }
    manifest_path = tmp_path / "manifest.yaml"
    paths_path = tmp_path / "paths.yaml"
    write_yaml(manifest_path, manifest)
    write_yaml(paths_path, {"datasets": {"opencap_root": str(tmp_path / "dataset")}})
    resolved = dict(manifest)
    resolved["paths"] = {"root": str(root)}
    return manifest_path, paths_path, resolved


def _write_pose_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    names = ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_big_toe", "right_big_toe"]
    ref = _reference_joints()
    camera_like = ref.copy()
    camera_like[:, :, 1] *= -1.0
    pose = {
        "representation": "joints",
        "backend": "metrabs",
        "fps": 60.0,
        "units": "m",
        "joint_names": names,
        "joints_3d": camera_like.astype(np.float32),
    }
    (run_dir / "pose3d_initial").mkdir()
    with (run_dir / "pose3d_initial" / "pose3d_initial.pkl").open("wb") as f:
        pickle.dump(pose, f)
    write_yaml(run_dir / "run_config.yaml", {"config": {"optimization": {"contact_foot_locking": {"max_correction_m": 0.2}}}})


def _write_reference(tmp_path: Path) -> tuple[Path, Path]:
    path = tmp_path / "reference.npz"
    names = np.asarray(["pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"], dtype=object)
    np.savez(path, time_s=np.arange(6, dtype=float) / 60.0, joints_m=_reference_joints()[:, :7, :], joint_names=names)
    meta = tmp_path / "reference.json"
    write_json(meta, {"ik_marker_errors": {}})
    return path, meta


def _reference_joints() -> np.ndarray:
    joints = np.zeros((6, 9, 3), dtype=float)
    joints[:, 1, :] = [-0.1, 0.0, 0.0]
    joints[:, 2, :] = [0.1, 0.0, 0.0]
    joints[:, 3, :] = [-0.1, -0.45, 0.0]
    joints[:, 4, :] = [0.1, -0.45, 0.0]
    joints[:, 5, :] = [-0.1, -0.9, 0.0]
    joints[:, 6, :] = [0.1, -0.9, 0.0]
    joints[:, 7, :] = [-0.1, -0.9, 0.1]
    joints[:, 8, :] = [0.1, -0.9, 0.1]
    joints[:, :, 0] += np.linspace(0.0, 0.1, 6)[:, None]
    return joints
