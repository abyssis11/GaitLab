from __future__ import annotations

from typing import Any


VALID_REPRESENTATIONS = {"joints", "smpl", "hybrid"}
VALID_UNITS = {"m"}


def validate_pose3d_artifact(artifact: dict[str, Any]) -> None:
    representation = artifact.get("representation")
    if representation not in VALID_REPRESENTATIONS:
        raise ValueError(f"Invalid pose3d representation: {representation}")
    if artifact.get("units") not in VALID_UNITS:
        raise ValueError("pose3d artifacts must use internal units='m'")
    if not artifact.get("backend"):
        raise ValueError("pose3d artifact missing backend")
    if "joints_3d" not in artifact:
        raise ValueError("pose3d artifact missing joints_3d")

    import numpy as np

    joints = np.asarray(artifact["joints_3d"])
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError("joints_3d must have shape [T, J, 3]")
    names = artifact.get("joint_names") or []
    if len(names) != joints.shape[1]:
        raise ValueError("joint_names length must match joints_3d.shape[1]")

    if representation in {"smpl", "hybrid"} and "smpl" not in artifact:
        raise ValueError("SMPL/hybrid pose3d artifact missing smpl payload")


def has_smpl_vertices(artifact: dict[str, Any]) -> bool:
    smpl = artifact.get("smpl")
    return isinstance(smpl, dict) and smpl.get("vertices") is not None

