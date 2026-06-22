from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Protocol


class BackendUnavailable(RuntimeError):
    pass


class Pose2DBackend(Protocol):
    def run_pose2d(self, video_path: Path, camera: dict, subject: dict, cfg: dict) -> dict:
        ...


class Pose3DBackend(Protocol):
    def run_pose3d(
        self,
        video_path: Path,
        pose2d: dict | None,
        camera: dict,
        subject: dict,
        cfg: dict,
    ) -> dict:
        ...


POSE2D_BACKENDS = {
    "none": "monocap_v2.backends.pose2d_dummy",
    "dummy": "monocap_v2.backends.pose2d_dummy",
    "mediapipe": "monocap_v2.backends.pose2d_mediapipe",
    "mmpose": "monocap_v2.backends.pose2d_mmpose",
}

POSE3D_BACKENDS = {
    "dummy": "monocap_v2.backends.pose3d_dummy",
    "metrabs": "monocap_v2.backends.pose3d_metrabs",
    "rtmw3d": "monocap_v2.backends.pose3d_rtmw3d",
    "sam3d_body": "monocap_v2.backends.pose3d_sam3d_body",
    "wham": "monocap_v2.backends.pose3d_wham",
}


def load_backend(kind: str, name: str) -> Any:
    if kind == "pose2d":
        registry = POSE2D_BACKENDS
    elif kind == "pose3d":
        registry = POSE3D_BACKENDS
    else:
        raise KeyError(f"Unknown backend kind: {kind}")
    if name not in registry:
        known = ", ".join(sorted(registry))
        raise KeyError(f"Unknown {kind} backend '{name}'. Known: {known}")
    return importlib.import_module(registry[name])


def run_pose2d_backend(name: str, video_path: Path, camera: dict, subject: dict, cfg: dict) -> dict:
    backend = load_backend("pose2d", name)
    return backend.run_pose2d(video_path, camera, subject, cfg)


def run_pose3d_backend(
    name: str,
    video_path: Path,
    pose2d: dict | None,
    camera: dict,
    subject: dict,
    cfg: dict,
) -> dict:
    backend = load_backend("pose3d", name)
    return backend.run_pose3d(video_path, pose2d, camera, subject, cfg)
