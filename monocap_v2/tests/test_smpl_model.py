from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from monocap_v2.core.smpl_model import SMPL_24_JOINT_NAMES, regress_smpl_24_joints


def test_regress_smpl_24_joints_from_synthetic_pickle(tmp_path: Path) -> None:
    model = tmp_path / "model.pkl"
    regressor = np.zeros((len(SMPL_24_JOINT_NAMES), 6), dtype=np.float32)
    regressor[0, 2] = 1.0
    regressor[1, 3] = 1.0
    with model.open("wb") as f:
        pickle.dump({"J_regressor": regressor}, f)

    vertices = np.arange(2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3)
    info = regress_smpl_24_joints(
        vertices,
        {
            "repo_root": str(tmp_path),
            "config": {
                "smpl": {
                    "model_dir": str(tmp_path),
                    "gender": "neutral",
                    "file_map": {"neutral": "model.pkl"},
                }
            },
        },
    )

    assert info["joint_names"] == SMPL_24_JOINT_NAMES
    assert info["joints"].shape == (2, len(SMPL_24_JOINT_NAMES), 3)
    assert np.allclose(info["joints"][:, 0], vertices[:, 2])
    assert np.allclose(info["joints"][:, 1], vertices[:, 3])
