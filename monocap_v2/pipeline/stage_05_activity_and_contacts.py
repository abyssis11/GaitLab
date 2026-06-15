from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from monocap_v2.backends.activity_manual import activity_payload
from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.contact_utils import estimate_contacts_from_joints
from monocap_v2.core.logging_utils import write_json
from monocap_v2.core.stage_utils import cached, stage_result


STAGE = "stage_05_activity_and_contacts"


def run(run_dir: Path, cfg: dict, force: bool = False) -> dict:
    registry = ArtifactRegistry(run_dir)
    out_path = registry.ensure_parent("contacts")
    if cached(out_path, force):
        return stage_result(STAGE, "cached", output=str(out_path))

    with registry.get("pose3d_initial").open("rb") as f:
        pose3d = pickle.load(f)
    activity = cfg.get("activity", "other")
    contacts = pose3d.get("contacts") or estimate_contacts_from_joints(pose3d, activity)
    np.savez_compressed(
        out_path,
        left_heel=contacts["left_heel"],
        left_toe=contacts["left_toe"],
        right_heel=contacts["right_heel"],
        right_toe=contacts["right_toe"],
        backend=str(contacts.get("backend", "unknown")),
        activity=activity,
    )
    write_json(registry.ensure_parent("activity"), activity_payload(activity))
    qc = {
        "stage": STAGE,
        "status": "ok",
        "backend": contacts.get("backend", "unknown"),
        "activity": activity,
        "mean_contact_probability": {
            key: float(np.nanmean(contacts[key])) for key in ["left_heel", "left_toe", "right_heel", "right_toe"]
        },
    }
    write_json(registry.ensure_parent("contacts_qc"), qc)
    return stage_result(STAGE, "ok", output=str(out_path), backend=contacts.get("backend", "unknown"))

