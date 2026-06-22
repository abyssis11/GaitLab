from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.backend_registry import run_pose3d_backend
from monocap_v2.core.logging_utils import read_json, write_json
from monocap_v2.core.schemas import validate_pose3d_artifact
from monocap_v2.core.stage_utils import cached, stage_result
from monocap_v2.core.wham_timeline import wham_timebase_report


STAGE = "stage_04_pose3d_initial"


def run(run_dir: Path, cfg: dict, force: bool = False) -> dict:
    registry = ArtifactRegistry(run_dir)
    out_path = registry.ensure_parent("pose3d_initial")
    backend = cfg.get("config", {}).get("backends", {}).get("pose3d", "metrabs")
    if cached(out_path, force) and _cached_backend_matches(out_path, backend):
        return stage_result(STAGE, "cached", output=str(out_path), backend=backend)
    video_info = read_json(registry.get("video_info"))
    backend_cfg = {**cfg, "video_info": video_info}
    video_path = Path(video_info.get("preprocessed_video") or video_info["raw_video"])
    camera = read_json(registry.get("camera_assumed"))
    subject = read_json(registry.get("subject_info"))
    pose2d = _read_pose2d(registry)
    artifact = run_pose3d_backend(backend, video_path, pose2d, camera, subject, backend_cfg)
    validate_pose3d_artifact(artifact)

    with out_path.open("wb") as f:
        pickle.dump(artifact, f)

    pose2d_written = _write_pose2d_if_present(registry, artifact, cfg)
    joints = np.asarray(artifact["joints_3d"])
    qc = {
        "stage": STAGE,
        "status": "ok",
        "backend": artifact["backend"],
        "representation": artifact["representation"],
        "frames": int(joints.shape[0]),
        "joints": int(joints.shape[1]),
        "units": artifact["units"],
        "finite_ratio": float(np.isfinite(joints).mean()),
        "has_smpl": "smpl" in artifact,
        "has_smpl_vertices": bool(isinstance(artifact.get("smpl"), dict) and artifact["smpl"].get("vertices") is not None),
        "has_mesh": "mesh" in artifact,
        "has_mesh_vertices": bool(isinstance(artifact.get("mesh"), dict) and artifact["mesh"].get("vertices") is not None),
        "source_video": artifact.get("source_video"),
        "backend_meta": artifact.get("backend_meta", {}),
        "pose2d_written": pose2d_written,
    }
    if artifact.get("backend") == "wham":
        qc["wham_timebase"] = wham_timebase_report(artifact, cfg)
    if artifact.get("backend") == "sam3d_body":
        meta = artifact.get("backend_meta") or {}
        canonical = meta.get("canonical_joint_indices") or {}
        qc["canonical_joint_coverage"] = {
            "configured": list(canonical.keys()),
            "available": [name for name in canonical if name in artifact.get("joint_names", [])],
            "missing": [name for name in canonical if name not in artifact.get("joint_names", [])],
            "selected_indices": canonical,
            "mapping_source": meta.get("mapping_source"),
            "mapping_note": meta.get("mapping_note"),
        }
    write_json(registry.ensure_parent("pose3d_initial_qc"), qc)
    return stage_result(STAGE, "ok", output=str(out_path), backend=artifact["backend"], representation=artifact["representation"])


def _cached_backend_matches(path: Path, expected_backend: str) -> bool:
    try:
        with path.open("rb") as f:
            artifact = pickle.load(f)
    except Exception:
        return False
    return artifact.get("backend") == expected_backend


def _read_pose2d(registry: ArtifactRegistry) -> dict | None:
    path = registry.get("keypoints_2d")
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "xy": data["xy"],
        "confidence": data["confidence"],
        "names": data["names"].tolist(),
        "fps": float(data["fps"]),
        "backend": str(data["backend"]),
    }


def _write_pose2d_if_present(registry: ArtifactRegistry, artifact: dict, cfg: dict) -> bool:
    pose2d = artifact.get("pose2d")
    if not isinstance(pose2d, dict):
        return False
    if not cfg.get("config", {}).get("metrabs", {}).get("save_pose2d", True):
        return False
    out_path = registry.ensure_parent("keypoints_2d")
    np.savez_compressed(
        out_path,
        xy=np.asarray(pose2d["xy"], dtype=np.float32),
        confidence=np.asarray(pose2d["confidence"], dtype=np.float32),
        names=np.asarray(pose2d["names"], dtype=object),
        fps=float(pose2d["fps"]),
        backend=str(pose2d["backend"]),
    )
    confidence = np.asarray(pose2d["confidence"], dtype=float)
    qc = {
        "stage": STAGE,
        "status": "ok",
        "source": "pose3d_backend",
        "backend": pose2d["backend"],
        "frames": int(np.asarray(pose2d["xy"]).shape[0]),
        "joints": int(np.asarray(pose2d["xy"]).shape[1]),
        "mean_confidence": float(np.nanmean(confidence)) if confidence.size else 0.0,
    }
    write_json(registry.ensure_parent("pose2d_qc"), qc)
    return True
