from __future__ import annotations

from pathlib import Path

import numpy as np

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.backend_registry import run_pose2d_backend
from monocap_v2.core.logging_utils import read_json, write_json
from monocap_v2.core.stage_utils import cached, stage_result


STAGE = "stage_03_pose2d"


def run(run_dir: Path, cfg: dict, force: bool = False) -> dict:
    registry = ArtifactRegistry(run_dir)
    out_path = registry.ensure_parent("keypoints_2d")
    if cached(out_path, force):
        return stage_result(STAGE, "cached", output=str(out_path))

    backend = cfg.get("config", {}).get("backends", {}).get("pose2d", "none")
    if backend in {"none", None}:
        result = stage_result(STAGE, "skipped", reason="pose2d backend disabled; pose3d backend may provide 2D data")
        write_json(registry.ensure_parent("pose2d_qc"), result)
        return result

    video_path = _active_video_path(registry)
    camera = read_json(registry.get("camera_assumed"))
    subject = read_json(registry.get("subject_info"))
    pose2d = run_pose2d_backend(backend, video_path, camera, subject, cfg)
    np.savez_compressed(
        out_path,
        xy=pose2d["xy"],
        confidence=pose2d["confidence"],
        names=np.asarray(pose2d["names"], dtype=object),
        fps=float(pose2d["fps"]),
        backend=str(pose2d["backend"]),
    )
    qc = {
        "stage": STAGE,
        "status": "ok",
        "backend": pose2d["backend"],
        "frames": int(pose2d["xy"].shape[0]),
        "joints": int(pose2d["xy"].shape[1]),
        "mean_confidence": float(np.nanmean(pose2d["confidence"])),
    }
    write_json(registry.ensure_parent("pose2d_qc"), qc)
    return stage_result(STAGE, "ok", output=str(out_path), backend=pose2d["backend"])


def _active_video_path(registry: ArtifactRegistry) -> Path:
    info = read_json(registry.get("video_info"))
    return Path(info.get("preprocessed_video") or info["raw_video"])

