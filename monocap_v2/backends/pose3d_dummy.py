from __future__ import annotations

from pathlib import Path

import numpy as np


def run_pose3d(video_path: Path, pose2d: dict | None, camera: dict, subject: dict, cfg: dict) -> dict:
    frame_count = int(cfg.get("video_info", {}).get("frame_count") or 30)
    fps = float(cfg.get("video_info", {}).get("fps") or 30.0)
    names = [
        "pelvis",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_big_toe",
        "right_big_toe",
        "left_shoulder",
        "right_shoulder",
    ]
    t = np.linspace(0, 1, frame_count, dtype=np.float32)
    joints = np.zeros((frame_count, len(names), 3), dtype=np.float32)
    joints[:, :, 0] = t[:, None] * 0.5
    joints[:, :, 1] = np.linspace(0.0, 1.5, len(names), dtype=np.float32)[None, :] * 0.02
    joints[:, :, 2] = np.linspace(-0.3, 0.3, len(names), dtype=np.float32)[None, :]
    return {
        "representation": "joints",
        "backend": "dummy",
        "fps": fps,
        "units": "m",
        "joint_names": names,
        "joints_3d": joints,
        "camera": {"intrinsics": camera, "extrinsics": None, "is_assumed": bool(camera.get("is_assumed", True))},
        "source_video": str(video_path),
    }

