from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from monocap_v2.backends import pose3d_metrabs
from monocap_v2.core.schemas import validate_pose3d_artifact


class _FakeNameTensor:
    def __init__(self, names):
        self._names = np.asarray(names, dtype=object)

    def numpy(self):
        return self._names


class _FakeTF:
    uint8 = np.uint8
    float32 = np.float32

    class config:
        @staticmethod
        def list_physical_devices(kind):
            return []

        class experimental:
            @staticmethod
            def set_memory_growth(gpu, value):
                return None

    @staticmethod
    def convert_to_tensor(value, dtype=None):
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def constant(value, dtype=None):
        return np.asarray(value, dtype=dtype)


class _DetectingModel:
    per_skeleton_joint_names = {"smpl_24": _FakeNameTensor(["pelvis", "left_hip"])}

    def detect_poses(self, image, **kwargs):
        return {
            "boxes": np.asarray([[0, 0, 10, 10, 0.75]], dtype=np.float32),
            "poses3d": np.asarray([[[1000, 2000, 3000], [4000, 5000, 6000]]], dtype=np.float32),
            "poses2d": np.asarray([[[11, 12], [21, 22]]], dtype=np.float32),
        }


class _NoDetectionModel:
    per_skeleton_joint_names = {"smpl_24": _FakeNameTensor(["pelvis", "left_hip"])}

    def detect_poses(self, image, **kwargs):
        return {
            "boxes": np.zeros((0, 5), dtype=np.float32),
            "poses3d": np.zeros((0, 2, 3), dtype=np.float32),
            "poses2d": np.zeros((0, 2, 2), dtype=np.float32),
        }


def test_metrabs_backend_artifact_units_and_max_frames(tmp_path: Path, monkeypatch) -> None:
    video = _write_tiny_video(tmp_path / "tiny.mp4", frames=3)
    monkeypatch.setattr(pose3d_metrabs, "_import_tensorflow", lambda: _FakeTF)
    monkeypatch.setattr(pose3d_metrabs, "_load_model", lambda tf, cfg: _DetectingModel())

    artifact = pose3d_metrabs.run_pose3d(video, None, _camera(), {"id": "s1"}, _cfg(max_frames=2))

    validate_pose3d_artifact(artifact)
    assert artifact["representation"] == "joints"
    assert artifact["backend"] == "metrabs"
    assert artifact["units"] == "m"
    assert artifact["joint_names"] == ["pelvis", "left_hip"]
    assert artifact["joints_3d"].shape == (2, 2, 3)
    assert np.allclose(artifact["joints_3d"][0, 0], [1.0, 2.0, 3.0])
    assert artifact["pose2d"]["xy"].shape == (2, 2, 2)
    assert np.allclose(artifact["pose2d"]["confidence"], 0.75)


def test_metrabs_backend_no_detection_keeps_nan_frames(tmp_path: Path, monkeypatch) -> None:
    video = _write_tiny_video(tmp_path / "tiny.mp4", frames=2)
    monkeypatch.setattr(pose3d_metrabs, "_import_tensorflow", lambda: _FakeTF)
    monkeypatch.setattr(pose3d_metrabs, "_load_model", lambda tf, cfg: _NoDetectionModel())

    artifact = pose3d_metrabs.run_pose3d(video, None, _camera(), {"id": "s1"}, _cfg(max_frames=None))

    validate_pose3d_artifact(artifact)
    assert artifact["joints_3d"].shape == (2, 2, 3)
    assert np.isnan(artifact["joints_3d"]).all()
    assert artifact["pose2d"]["confidence"].sum() == 0.0


def _write_tiny_video(path: Path, frames: int) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    for i in range(frames):
        writer.write(np.full((24, 32, 3), i * 20, dtype=np.uint8))
    writer.release()
    return path


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


def _cfg(max_frames) -> dict:
    return {
        "video_info": {"fps": 10.0},
        "config": {
            "metrabs": {
                "skeleton": "smpl_24",
                "max_frames": max_frames,
                "max_detections": 1,
                "suppress_implausible_poses": False,
            }
        },
    }

