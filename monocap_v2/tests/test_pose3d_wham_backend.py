from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from monocap_v2.backends import pose3d_wham
from monocap_v2.core.schemas import validate_pose3d_artifact


def test_wham_backend_converts_fake_result_to_hybrid_artifact(tmp_path: Path, monkeypatch) -> None:
    repo = _fake_wham_repo(tmp_path)
    selected_video = _write_tiny_video(tmp_path / "sync.mp4")
    raw_video = _write_tiny_video(tmp_path / "raw.mp4")
    called = {}

    def fake_run(wham_repo, video, output_dir, run_global, visualize):
        called["video"] = Path(video)
        called["run_global"] = run_global
        results = {
            2: _fake_subject(frames=[7]),
            1: _fake_subject(frames=[3, 5]),
        }
        tracking = {
            1: {
                "frame_id": np.array([2, 3, 5], dtype=np.int64),
                "keypoints": np.array(
                    [
                        [[1.0, 1.5, 0.1], [2.0, 2.5, 0.2]],
                        [[3.0, 3.5, 0.3], [4.0, 4.5, 0.4]],
                        [[5.0, 5.5, 0.5], [6.0, 6.5, 0.6]],
                    ],
                    dtype=np.float32,
                ),
            }
        }
        return results, tracking, None, False

    monkeypatch.setattr(pose3d_wham, "_run_wham_api", fake_run)

    artifact = pose3d_wham.run_pose3d(
        selected_video,
        None,
        _camera(),
        {"id": "subject7", "sex": "female", "height_m": 1.7},
        {
            "repo_root": str(tmp_path),
            "run_dir": str(tmp_path / "run"),
            "force": True,
            "trial": {"video_raw": str(raw_video)},
            "config": {
                "wham": {
                    "repo_path": str(repo),
                    "video_field": "video_raw",
                    "local_only": True,
                    "output_dir_name": "wham_work",
                }
            },
        },
    )

    validate_pose3d_artifact(artifact)
    assert called["video"] == raw_video.resolve()
    assert called["run_global"] is False
    assert artifact["representation"] == "hybrid"
    assert artifact["backend"] == "wham"
    assert artifact["smpl"]["gender"] == "female"
    assert artifact["smpl"]["vertices"].shape == (2, 5, 3)
    assert artifact["joints_3d"].shape == (2, 6, 3)
    assert artifact["joint_names"][-4:] == ["left_big_toe", "left_heel", "right_big_toe", "right_heel"]
    assert artifact["backend_meta"]["selected_track_id"] == "1"
    assert artifact["backend_meta"]["joint_regression"]["sources"][0]["type"] == "wham_regressor_generic"
    assert artifact["backend_meta"]["frame_ids"].tolist() == [3, 5]
    assert artifact["pose2d"]["xy"].shape == (2, 2, 2)
    assert np.allclose(artifact["pose2d"]["xy"][0, 0], [3.0, 3.5])
    assert np.allclose(artifact["smpl"]["body_pose"], 0.0)


def test_wham_source_video_falls_back_to_stage_video(tmp_path: Path) -> None:
    selected_video = _write_tiny_video(tmp_path / "sync.mp4")
    out = pose3d_wham._select_source_video(
        selected_video,
        {"trial": {"video_raw": str(tmp_path / "missing.mp4")}},
        {"video_field": "video_raw"},
        tmp_path,
    )
    assert out == selected_video.resolve()


def _fake_wham_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "WHAM"
    body_models = repo / "dataset" / "body_models"
    body_models.mkdir(parents=True)
    (repo / "wham_api.py").write_text("# fake\n", encoding="utf-8")
    wham_regressor = np.zeros((2, 5), dtype=np.float32)
    wham_regressor[0, 0] = 1.0
    wham_regressor[1, 1] = 1.0
    feet_regressor = np.zeros((4, 5), dtype=np.float32)
    feet_regressor[0, 2] = 1.0
    feet_regressor[1, 3] = 1.0
    feet_regressor[2, 4] = 1.0
    feet_regressor[3, 0] = 1.0
    np.save(body_models / "J_regressor_wham.npy", wham_regressor)
    np.save(body_models / "J_regressor_feet.npy", feet_regressor)
    return repo


def _fake_subject(frames: list[int]) -> dict:
    count = len(frames)
    vertices = np.arange(count * 5 * 3, dtype=np.float32).reshape(count, 5, 3) / 100.0
    ident_body = np.broadcast_to(np.eye(3, dtype=np.float32), (count, 23, 3, 3)).copy()
    ident_root = np.broadcast_to(np.eye(3, dtype=np.float32), (count, 1, 3, 3)).copy()
    return {
        "frame_id": np.asarray(frames, dtype=np.int64),
        "verts_cam": vertices,
        "poses_body": ident_body,
        "poses_root_cam": ident_root,
        "betas": np.zeros((count, 10), dtype=np.float32),
        "trans_world": np.zeros((count, 3), dtype=np.float32),
    }


def _write_tiny_video(path: Path) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    for i in range(3):
        writer.write(np.full((24, 32, 3), i * 20, dtype=np.uint8))
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
