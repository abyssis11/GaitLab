from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.logging_utils import write_yaml
from monocap_v2.core import wham_smpl_root_refinement as root_refine


def test_wham_smpl_root_refinement_skips_missing_smpl_fields(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_standard_dirs()
    pose = {
        "representation": "hybrid",
        "backend": "wham",
        "units": "m",
        "joint_names": ["pelv"],
        "joints_3d": np.zeros((2, 1, 3), dtype=np.float32),
        "smpl": {"vertices": np.zeros((2, 5, 3), dtype=np.float32)},
    }

    refined, report = root_refine.apply_wham_smpl_root_refinement(registry, pose, {}, {"enabled": True})

    assert report["status"] == "skipped"
    assert "missing SMPL fields" in report["reason"]
    np.testing.assert_allclose(refined["joints_3d"], pose["joints_3d"])


def test_wham_smpl_root_refinement_accepts_mocked_worker_output(tmp_path: Path, monkeypatch) -> None:
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_standard_dirs()
    pose = _wham_pose()
    with registry.ensure_parent("pose3d_initial").open("wb") as f:
        pickle.dump(pose, f)
    write_yaml(
        registry.ensure_parent("run_config"),
        {"repo_root": str(Path(__file__).resolve().parents[2]), "config": {"smpl": {"model_dir": "models/smpl"}}},
    )

    def fake_run(cmd, cwd, text, capture_output):
        output_pose = Path(cmd[cmd.index("--output-pose") + 1])
        output_report = Path(cmd[cmd.index("--report") + 1])
        refined = _wham_pose()
        refined["smpl"]["transl"][:, 0] = 0.2
        refined["refinement"] = {"status": "ok", "method": "wham_smpl_root_refinement"}
        with output_pose.open("wb") as f:
            pickle.dump(refined, f)
        root_refine.write_json(
            output_report,
            {
                "status": "ok",
                "method": "wham_smpl_root_refinement",
                "optimized_parameters": ["smpl.transl", "smpl.global_orient"],
                "mocap_used_in_objective": False,
            },
        )

        class Proc:
            returncode = 0
            stdout = ""
            stderr = ""

        return Proc()

    monkeypatch.setattr(root_refine.subprocess, "run", fake_run)

    refined, report = root_refine.apply_wham_smpl_root_refinement(
        registry,
        pose,
        {"repo_root": str(Path(__file__).resolve().parents[2])},
        {"enabled": True, "python": sys.executable},
    )

    assert report["status"] == "ok"
    assert report["method"] == "wham_smpl_root_refinement"
    assert refined["smpl"]["transl"][0, 0] == np.float32(0.2)
    assert refined["refinement"]["wham_smpl_root_refinement"]["status"] == "ok"


def _wham_pose() -> dict:
    frames = 2
    return {
        "representation": "hybrid",
        "backend": "wham",
        "fps": 30.0,
        "units": "m",
        "joint_names": ["pelv"],
        "joints_3d": np.zeros((frames, 1, 3), dtype=np.float32),
        "smpl": {
            "vertices": np.zeros((frames, 5, 3), dtype=np.float32),
            "betas": np.zeros((frames, 10), dtype=np.float32),
            "body_pose": np.zeros((frames, 23, 3), dtype=np.float32),
            "global_orient": np.zeros((frames, 3), dtype=np.float32),
            "transl": np.zeros((frames, 3), dtype=np.float32),
        },
    }
