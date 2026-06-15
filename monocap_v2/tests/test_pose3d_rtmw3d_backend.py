from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from monocap_v2.backends import pose3d_rtmw3d
from monocap_v2.core.logging_utils import write_yaml
from monocap_v2.core.schemas import validate_pose3d_artifact


def test_rtmw3d_metric_jsonl_converts_to_artifact_arrays(tmp_path: Path) -> None:
    meta = {"fps": 60.0, "keypoint_names": ["pelvis", "left_hip"]}
    preds = tmp_path / "preds_metric.jsonl"
    rows = [
        {
            "frame_index": 0,
            "time_sec": 0.0,
            "persons": [
                {
                    "mean_score": 0.5,
                    "keypoints_xyz_mm": [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]],
                    "keypoints_px": [[10.0, 11.0], [20.0, 21.0]],
                    "keypoint_scores": [0.7, 0.8],
                }
            ],
        },
        {"frame_index": 1, "time_sec": 1.0 / 60.0, "persons": []},
    ]
    preds.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    parsed = pose3d_rtmw3d.read_rtmw3d_metric_predictions(preds, meta)

    assert parsed["joint_names"] == ["pelvis", "left_hip"]
    assert parsed["joints_3d_m"].shape == (2, 2, 3)
    assert np.allclose(parsed["joints_3d_m"][0, 0], [0.1, 0.2, 0.3])
    assert np.isnan(parsed["joints_3d_m"][1]).all()
    assert np.allclose(parsed["xy"][0, 1], [20.0, 21.0])
    assert parsed["confidence"][0, 0] == np.float32(0.7)
    assert parsed["confidence"][1].sum() == 0.0
    assert parsed["valid_person_frames"] == 1


def test_rtmw3d_backend_runs_subprocess_adapter_and_writes_local_manifest(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    run_dir = repo / "monocap_v2" / "runs" / "run"
    run_dir.mkdir(parents=True)
    (repo / "src" / "pose").mkdir(parents=True)
    (repo / "src" / "pose" / "rtmw3d_pose_estimation.py").write_text("# fake\n", encoding="utf-8")
    (repo / "src" / "pose" / "rtmw3d_scale_from_height.py").write_text("# fake\n", encoding="utf-8")
    (repo / "models" / "rtmw3d").mkdir(parents=True)
    (repo / "models" / "rtmw3d" / "cfg.py").write_text("# fake\n", encoding="utf-8")
    (repo / "models" / "rtmw3d" / "ckpt.pth").write_text("fake\n", encoding="utf-8")
    (repo / "external" / "datasets_config").mkdir(parents=True)
    (repo / "external" / "datasets_config" / "h3wb.py").write_text("dataset_info={}\n", encoding="utf-8")
    paths_yaml = repo / "paths.yaml"
    write_yaml(paths_yaml, {"datasets": {"opencap_root": str(repo), "gpjatk_root": str(repo)}})
    manifest = {
        "subject_id": "subject7",
        "session": "Session1",
        "camera": "Cam1",
        "paths": {"root": str(repo)},
        "trials": {"healthy": [{"id": "walking1", "video_sync": str(repo / "sync.mp4")}]},
    }
    write_yaml(run_dir / "manifest_resolved.yaml", manifest)
    video = _write_tiny_video(repo / "sync.mp4")
    calls = []

    def fake_run(cmd, cwd, env):
        calls.append(cmd)
        work_root = run_dir / "pose3d_initial" / "rtmw3d_work"
        trial_root = work_root / "walking1"
        out_dir = trial_root / "rtmw3d"
        out_dir.mkdir(parents=True, exist_ok=True)
        if "rtmw3d_pose_estimation.py" in cmd[1]:
            (trial_root / "meta.json").write_text(
                json.dumps({"fps": 10.0, "keypoint_names": ["pelvis", "left_hip"]}),
                encoding="utf-8",
            )
            (out_dir / "preds.jsonl").write_text("{}\n", encoding="utf-8")
        else:
            row = {
                "frame_index": 0,
                "time_sec": 0.0,
                "persons": [
                    {
                        "mean_score": 0.9,
                        "keypoints_xyz_mm": [[0.0, 1000.0, 2000.0], [3000.0, 4000.0, 5000.0]],
                        "keypoints_px": [[1.0, 2.0], [3.0, 4.0]],
                        "keypoint_scores": [0.9, 0.8],
                    }
                ],
            }
            (out_dir / "preds_metric.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    monkeypatch.setattr(pose3d_rtmw3d, "_run_command", fake_run)
    artifact = pose3d_rtmw3d.run_pose3d(
        video,
        None,
        _camera(),
        {"id": "subject7", "height_m": 1.75},
        {
            "repo_root": str(repo),
            "run_dir": str(run_dir),
            "paths_path": str(paths_yaml),
            "trial_id": "walking1",
            "trial_subset": "healthy",
            "trial": {"id": "walking1", "video_sync": str(video)},
            "video_field": "video_sync",
            "force": True,
            "config": {
                "rtmw3d": {
                    "python": "/fake/python",
                    "config_path": "models/rtmw3d/cfg.py",
                    "checkpoint_path": "models/rtmw3d/ckpt.pth",
                    "metainfo_from_file": "external/datasets_config/h3wb.py",
                    "video_field": "video_sync",
                    "device": "cpu",
                    "refine_pass": True,
                }
            },
            "video_info": {"fps": 10.0},
        },
    )

    validate_pose3d_artifact(artifact)
    assert artifact["backend"] == "rtmw3d"
    assert artifact["representation"] == "joints"
    assert artifact["joints_3d"].shape == (1, 2, 3)
    assert np.allclose(artifact["joints_3d"][0, 1], [3.0, 4.0, 5.0])
    assert artifact["pose2d"]["xy"].shape == (1, 2, 2)
    assert len(calls) == 2
    local_manifest = run_dir / "pose3d_initial" / "rtmw3d_work" / "manifest_rtmw3d.yaml"
    assert local_manifest.exists()
    assert str(run_dir / "pose3d_initial" / "rtmw3d_work") in local_manifest.read_text(encoding="utf-8")


def _write_tiny_video(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    for i in range(2):
        writer.write(np.full((24, 32, 3), i * 40, dtype=np.uint8))
    writer.release()
    return path.resolve()


def _camera() -> dict:
    return {
        "fx": 38.4,
        "fy": 38.4,
        "cx": 16.0,
        "cy": 12.0,
        "width": 32,
        "height": 24,
        "distortion": {"enabled": False},
        "is_assumed": True,
    }
