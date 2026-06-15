from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.logging_utils import write_json
from monocap_v2.pipeline import stage_04_pose3d_initial, stage_06_optimize_extrinsics, stage_07_optimize_pose, stage_08_extract_virtual_markers


def test_stage_cache_behavior(tmp_path: Path) -> None:
    first = stage_06_optimize_extrinsics.run(tmp_path, {}, force=False)
    second = stage_06_optimize_extrinsics.run(tmp_path, {}, force=False)
    assert first["status"] == "skipped"
    assert second["status"] == "cached"


def test_pose3d_cache_requires_matching_backend(tmp_path: Path) -> None:
    path = tmp_path / "pose.pkl"
    with path.open("wb") as f:
        pickle.dump({"backend": "metrabs"}, f)
    assert stage_04_pose3d_initial._cached_backend_matches(path, "metrabs")
    assert not stage_04_pose3d_initial._cached_backend_matches(path, "wham")


def test_smpl_dependent_stage_skips_for_joints_only(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_standard_dirs()
    pose = {
        "representation": "joints",
        "backend": "dummy",
        "fps": 30.0,
        "units": "m",
        "joint_names": ["pelvis"],
        "joints_3d": np.zeros((3, 1, 3), dtype=np.float32),
    }
    with registry.ensure_parent("pose3d_refined").open("wb") as f:
        pickle.dump(pose, f)
    result = stage_08_extract_virtual_markers.run(tmp_path, {}, force=False)
    assert result["status"] == "skipped"
    assert "No SMPL vertices" in result["reason"]


def test_stage_07_writes_refined_pose_and_uses_cache(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_standard_dirs()
    pose = {
        "representation": "joints",
        "backend": "dummy",
        "fps": 30.0,
        "units": "m",
        "joint_names": ["pelv", "lhip", "rhip"],
        "joints_3d": np.array(
            [
                [[0.0, 0.0, 4.0], [-0.1, 0.2, 4.0], [0.1, 0.2, 4.0]],
                [[0.0, 0.0, 4.1], [-0.1, 0.2, 4.1], [0.1, 0.2, 4.1]],
                [[0.0, 0.0, 4.2], [-0.1, 0.2, 4.2], [0.1, 0.2, 4.2]],
            ],
            dtype=np.float32,
        ),
    }
    with registry.ensure_parent("pose3d_initial").open("wb") as f:
        pickle.dump(pose, f)
    write_json(
        registry.ensure_parent("camera_assumed"),
        {"width": 640, "height": 480, "fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0},
    )
    cfg = {
        "activity": "walking",
        "config": {
            "optimization": {
                "joints_only": {
                    "max_nfev": 3,
                    "robust_loss": "linear",
                    "weights": {"fidelity": 1.0, "bone": 1.0, "smoothness": 1.0},
                }
            }
        },
    }
    first = stage_07_optimize_pose.run(tmp_path, cfg, force=False)
    second = stage_07_optimize_pose.run(tmp_path, cfg, force=False)
    assert first["status"] in {"ok", "warning"}
    assert first["method"] == "scipy_least_squares_joints_only"
    assert second["status"] == "cached"
    assert registry.get("pose3d_refined").exists()
    assert registry.get("opt_stage2_report").exists()


def test_stage_07_skips_hybrid_smpl_by_default(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_standard_dirs()
    pose = {
        "representation": "hybrid",
        "backend": "wham",
        "fps": 30.0,
        "units": "m",
        "joint_names": ["pelv"],
        "joints_3d": np.zeros((2, 1, 3), dtype=np.float32),
        "smpl": {"vertices": np.zeros((2, 5, 3), dtype=np.float32)},
    }
    with registry.ensure_parent("pose3d_initial").open("wb") as f:
        pickle.dump(pose, f)
    result = stage_07_optimize_pose.run(tmp_path, {"config": {"optimization": {"joints_only": {"enabled": True}}}}, force=False)
    assert result["status"] == "skipped"
    assert "SMPL" in result["reason"]
    with registry.get("pose3d_refined").open("rb") as f:
        refined = pickle.load(f)
    assert refined["refinement"]["status"] == "passthrough"
