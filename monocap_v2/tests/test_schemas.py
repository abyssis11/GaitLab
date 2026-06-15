from __future__ import annotations

import numpy as np
import pytest

from monocap_v2.core.schemas import has_smpl_vertices, validate_pose3d_artifact


def test_pose3d_joints_schema() -> None:
    artifact = {
        "representation": "joints",
        "backend": "dummy",
        "fps": 30.0,
        "units": "m",
        "joint_names": ["pelvis", "left_hip"],
        "joints_3d": np.zeros((3, 2, 3), dtype=np.float32),
    }
    validate_pose3d_artifact(artifact)
    assert not has_smpl_vertices(artifact)


def test_pose3d_hybrid_schema_requires_smpl() -> None:
    artifact = {
        "representation": "hybrid",
        "backend": "wham",
        "fps": 30.0,
        "units": "m",
        "joint_names": ["pelvis"],
        "joints_3d": np.zeros((3, 1, 3), dtype=np.float32),
    }
    with pytest.raises(ValueError):
        validate_pose3d_artifact(artifact)


def test_pose3d_smpl_vertices_detection() -> None:
    artifact = {
        "representation": "smpl",
        "backend": "wham",
        "fps": 30.0,
        "units": "m",
        "joint_names": ["pelvis"],
        "joints_3d": np.zeros((3, 1, 3), dtype=np.float32),
        "smpl": {"vertices": np.zeros((3, 6890, 3), dtype=np.float32)},
    }
    validate_pose3d_artifact(artifact)
    assert has_smpl_vertices(artifact)

