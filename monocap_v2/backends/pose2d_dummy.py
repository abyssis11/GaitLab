from __future__ import annotations

from pathlib import Path

import numpy as np


def run_pose2d(video_path: Path, camera: dict, subject: dict, cfg: dict) -> dict:
    raise RuntimeError("Pose2D backend is disabled. Use a real backend or a 3D backend that provides 2D keypoints.")


def dummy_pose2d(frame_count: int, fps: float, width: int, height: int) -> dict:
    names = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    xy = np.zeros((frame_count, len(names), 2), dtype=np.float32)
    xy[..., 0] = width / 2.0
    xy[..., 1] = height / 2.0
    confidence = np.ones((frame_count, len(names)), dtype=np.float32)
    return {"xy": xy, "confidence": confidence, "names": names, "fps": float(fps), "backend": "dummy"}

