from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.geometry import project_points
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
    result = stage_07_optimize_pose.run(
        tmp_path,
        {"config": {"optimization": {"joints_only": {"enabled": True}, "subject_scale": {"enabled": True, "mode": "height_global"}}}},
        force=False,
    )
    assert result["status"] == "skipped"
    assert "SMPL" in result["reason"]
    assert result["subject_scale"]["status"] == "skipped"
    with registry.get("pose3d_refined").open("rb") as f:
        refined = pickle.load(f)
    assert refined["refinement"]["status"] == "passthrough"


def test_stage_07_subject_scale_only_when_optimizer_disabled(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_standard_dirs()
    names = ["pelv", "lhip", "rhip", "lkne", "rkne", "lank", "rank", "ltoe", "rtoe"]
    joints = np.zeros((2, len(names), 3), dtype=np.float32)
    joints[:, names.index("lhip"), :] = [-0.1, 0.0, 4.0]
    joints[:, names.index("rhip"), :] = [0.1, 0.0, 4.0]
    joints[:, names.index("lkne"), :] = [-0.1, -0.245, 4.0]
    joints[:, names.index("rkne"), :] = [0.1, -0.245, 4.0]
    joints[:, names.index("lank"), :] = [-0.1, -0.491, 4.0]
    joints[:, names.index("rank"), :] = [0.1, -0.491, 4.0]
    joints[:, names.index("ltoe"), :] = [-0.1, -0.491, 4.152]
    joints[:, names.index("rtoe"), :] = [0.1, -0.491, 4.152]
    pose = {
        "representation": "joints",
        "backend": "dummy",
        "fps": 30.0,
        "units": "m",
        "joint_names": names,
        "joints_3d": joints,
    }
    with registry.ensure_parent("pose3d_initial").open("wb") as f:
        pickle.dump(pose, f)
    write_json(registry.ensure_parent("subject_info"), {"height_m": 2.0})
    cfg = {
        "config": {
            "optimization": {
                "joints_only": {"enabled": False},
                "subject_scale": {"enabled": True, "mode": "height_global", "min_scale": 0.5, "max_scale": 3.0},
            }
        }
    }

    result = stage_07_optimize_pose.run(tmp_path, cfg, force=True)

    assert result["status"] == "ok"
    assert result["method"] == "subject_scale_only"
    with registry.get("pose3d_refined").open("rb") as f:
        refined = pickle.load(f)
    assert refined["refinement"]["subject_scale"]["status"] == "ok"
    assert refined["refinement"]["subject_scale"]["global_scale"]["scale"] == pytest.approx(2.0)


def test_stage_07_named_refinement_profile_auto_resolves_static_pose(tmp_path: Path) -> None:
    run_dir = tmp_path / "subjectX_Session1_Cam0_walking1__dummy"
    static_dir = tmp_path / "subjectX_Session1_Cam0_static2__dummy"
    registry = ArtifactRegistry(run_dir)
    static_registry = ArtifactRegistry(static_dir)
    registry.ensure_standard_dirs()
    static_registry.ensure_standard_dirs()
    names = ["pelv", "lhip", "rhip", "lkne", "rkne", "lank", "rank", "ltoe", "rtoe"]
    joints = np.zeros((9, len(names), 3), dtype=np.float32)
    joints[:, names.index("lhip"), :] = [-0.1, 0.0, 4.0]
    joints[:, names.index("rhip"), :] = [0.1, 0.0, 4.0]
    joints[:, names.index("lkne"), :] = [-0.1, -0.35, 4.0]
    joints[:, names.index("rkne"), :] = [0.1, -0.35, 4.0]
    joints[:, names.index("lank"), :] = [-0.1, -0.8, 4.0]
    joints[:, names.index("rank"), :] = [0.1, -0.8, 4.0]
    joints[:, names.index("ltoe"), :] = np.column_stack([np.linspace(-0.1, 0.1, 9), np.full(9, -0.8), np.full(9, 4.15)])
    joints[:, names.index("rtoe"), :] = [0.1, -0.8, 4.15]
    pose = {
        "representation": "joints",
        "backend": "dummy",
        "fps": 30.0,
        "units": "m",
        "joint_names": names,
        "joints_3d": joints,
    }
    static_pose = {**pose, "joints_3d": joints.copy()}
    static_pose["joints_3d"][:, names.index("lkne"), :] = [-0.1, -0.45, 4.0]
    static_pose["joints_3d"][:, names.index("rkne"), :] = [0.1, -0.45, 4.0]
    with registry.ensure_parent("pose3d_initial").open("wb") as f:
        pickle.dump(pose, f)
    with static_registry.ensure_parent("pose3d_initial").open("wb") as f:
        pickle.dump(static_pose, f)
    write_json(registry.ensure_parent("subject_info"), {"height_m": 1.8})
    np.savez_compressed(
        registry.ensure_parent("contacts"),
        left_heel=np.zeros(9, dtype=np.float32),
        left_toe=np.ones(9, dtype=np.float32),
        right_heel=np.zeros(9, dtype=np.float32),
        right_toe=np.zeros(9, dtype=np.float32),
        backend="test",
        activity="walking",
    )
    cfg = {
        "manifest_summary": {"subject_id": "subjectX", "session": "Session1", "camera": "Cam0"},
        "config": {
            "optimization": {
                "refinement_profile": "joints_static_chain_contact_v1",
                "refinement_profiles": {
                    "joints_static_chain_contact_v1": {
                        "label": "test profile",
                        "version": 1,
                        "representation": "joints",
                        "stage_order": ["subject_scale", "kinematic_chain", "contact_foot_locking"],
                        "mocap_used_in_objective": False,
                        "optimization": {
                            "subject_scale": {
                                "enabled": True,
                                "mode": "static_bone",
                                "static_trial": "auto",
                                "static_trial_priority": ["static2"],
                            },
                            "kinematic_chain": {
                                "enabled": True,
                                "method": "moving_average",
                                "window_frames": 3,
                                "direction_window_frames": 3,
                                "root_window_frames": 3,
                                "smooth_roots": False,
                                "max_joint_displacement_m": None,
                            },
                            "contact_foot_locking": {
                                "enabled": True,
                                "mode": "root_translation",
                                "feet": "toes",
                                "contact_threshold": 0.5,
                                "smooth_correction_window_frames": 1,
                            },
                            "joints_only": {"enabled": False},
                        },
                    }
                },
            }
        },
    }

    result = stage_07_optimize_pose.run(run_dir, cfg, force=True)

    assert result["status"] == "ok"
    assert result["refinement_profile"]["name"] == "joints_static_chain_contact_v1"
    assert result["subject_scale"]["static_resolution"]["selected_static_trial"] == "static2"
    with registry.get("pose3d_refined").open("rb") as f:
        refined = pickle.load(f)
    assert refined["refinement"]["refinement_profile"]["name"] == "joints_static_chain_contact_v1"
    assert refined["refinement"]["mocap_used_in_objective"] is False


def test_stage_07_temporal_smoothing_only_when_optimizer_disabled(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_standard_dirs()
    names = ["pelv", "lhip", "lkne"]
    joints = np.zeros((9, len(names), 3), dtype=np.float32)
    joints[:, names.index("lhip"), :] = [-0.1, 0.0, 4.0]
    joints[:, names.index("lkne"), :] = [-0.1, -0.5, 4.0]
    joints[4, :, 2] += 0.08
    pose = {
        "representation": "joints",
        "backend": "dummy",
        "fps": 30.0,
        "units": "m",
        "joint_names": names,
        "joints_3d": joints,
    }
    with registry.ensure_parent("pose3d_initial").open("wb") as f:
        pickle.dump(pose, f)
    cfg = {
        "config": {
            "optimization": {
                "joints_only": {"enabled": False},
                "temporal_smoothing": {
                    "enabled": True,
                    "method": "moving_average",
                    "window_frames": 5,
                    "preserve_bones": True,
                },
            }
        }
    }

    result = stage_07_optimize_pose.run(tmp_path, cfg, force=True)

    assert result["status"] == "ok"
    assert result["method"] == "temporal_smoothing_only"
    with registry.get("pose3d_refined").open("rb") as f:
        refined = pickle.load(f)
    assert refined["refinement"]["temporal_smoothing"]["status"] == "ok"
    assert "temporal_smoothing" in result


def test_stage_07_reprojection_consistency_only_when_optimizer_disabled(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_standard_dirs()
    camera = {"width": 640, "height": 480, "fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
    names = ["pelv", "lhip"]
    joints = np.array(
        [
            [[0.0, 0.0, 3.0], [0.2, 0.0, 3.0]],
            [[0.0, 0.0, 3.0], [0.2, 0.0, 3.0]],
        ],
        dtype=np.float32,
    )
    xy = project_points(joints, camera)
    xy[..., 0] += 20.0
    pose = {
        "representation": "joints",
        "backend": "dummy",
        "fps": 30.0,
        "units": "m",
        "joint_names": names,
        "joints_3d": joints,
        "pose2d": {
            "xy": xy.astype(np.float32),
            "confidence": np.ones(xy.shape[:2], dtype=np.float32),
            "names": names,
            "fps": 30.0,
            "backend": "test",
        },
    }
    with registry.ensure_parent("pose3d_initial").open("wb") as f:
        pickle.dump(pose, f)
    write_json(registry.ensure_parent("camera_assumed"), camera)
    cfg = {
        "config": {
            "optimization": {
                "joints_only": {"enabled": False},
                "reprojection_consistency": {
                    "enabled": True,
                    "blend": 1.0,
                    "confidence_threshold": 0.1,
                    "preserve_bones": False,
                },
            }
        }
    }

    result = stage_07_optimize_pose.run(tmp_path, cfg, force=True)

    assert result["status"] == "ok"
    assert result["method"] == "reprojection_consistency_only"
    with registry.get("pose3d_refined").open("rb") as f:
        refined = pickle.load(f)
    report = refined["refinement"]["reprojection_consistency"]
    assert report["status"] == "ok"
    assert report["metrics_after"]["mean_reprojection_error_px"] < report["metrics_before"]["mean_reprojection_error_px"]


def test_stage_07_camera_time_refinement_only_when_optimizer_disabled(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_standard_dirs()
    camera = {"width": 640, "height": 480, "fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
    names = ["pelv", "lhip"]
    joints = np.zeros((5, len(names), 3), dtype=np.float32)
    joints[:, :, 2] = 3.0
    joints[:, names.index("pelv"), 0] = np.linspace(0.0, 0.2, 5)
    joints[:, names.index("lhip"), 0] = np.linspace(0.2, 0.4, 5)
    xy = project_points(joints, camera)
    xy[:-1] = xy[1:]
    pose = {
        "representation": "joints",
        "backend": "dummy",
        "fps": 30.0,
        "units": "m",
        "joint_names": names,
        "joints_3d": joints,
        "pose2d": {
            "xy": xy.astype(np.float32),
            "confidence": np.ones(xy.shape[:2], dtype=np.float32),
            "names": names,
            "fps": 30.0,
            "backend": "test",
        },
    }
    with registry.ensure_parent("pose3d_initial").open("wb") as f:
        pickle.dump(pose, f)
    write_json(registry.ensure_parent("camera_assumed"), camera)
    cfg = {
        "config": {
            "optimization": {
                "joints_only": {"enabled": False},
                "camera_time_refinement": {
                    "enabled": True,
                    "time_search": {"enabled": True, "offsets_s": [0.0, 1.0 / 30.0]},
                    "camera_delta": {"enabled": False},
                },
            }
        }
    }

    result = stage_07_optimize_pose.run(tmp_path, cfg, force=True)

    assert result["status"] == "ok"
    assert result["method"] == "camera_time_refinement_only"
    with registry.get("pose3d_refined").open("rb") as f:
        refined = pickle.load(f)
    report = refined["refinement"]["camera_time_refinement"]
    assert report["status"] == "ok"
    assert report["selected_time_offset_s"] == pytest.approx(1.0 / 30.0, abs=1e-5)


def test_stage_07_kinematic_chain_only_when_optimizer_disabled(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_standard_dirs()
    names = ["pelv", "lhip", "lkne"]
    joints = np.zeros((9, len(names), 3), dtype=np.float32)
    joints[:, names.index("lhip"), :] = [-0.1, 0.0, 4.0]
    joints[:, names.index("lkne"), :] = [-0.1, -0.5, 4.0]
    joints[::2, names.index("lkne"), 0] += 0.05
    pose = {
        "representation": "joints",
        "backend": "dummy",
        "fps": 30.0,
        "units": "m",
        "joint_names": names,
        "joints_3d": joints,
    }
    with registry.ensure_parent("pose3d_initial").open("wb") as f:
        pickle.dump(pose, f)
    cfg = {
        "config": {
            "optimization": {
                "joints_only": {"enabled": False},
                "kinematic_chain": {
                    "enabled": True,
                    "method": "moving_average",
                    "window_frames": 5,
                    "smooth_roots": False,
                    "max_joint_displacement_m": None,
                },
            }
        }
    }

    result = stage_07_optimize_pose.run(tmp_path, cfg, force=True)

    assert result["status"] == "ok"
    assert result["method"] == "kinematic_chain_only"
    with registry.get("pose3d_refined").open("rb") as f:
        refined = pickle.load(f)
    assert refined["refinement"]["kinematic_chain"]["status"] == "ok"


def test_stage_07_pose_prior_only_when_optimizer_disabled(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_standard_dirs()
    names = ["lhip", "lkne", "lank"]
    joints = np.zeros((2, len(names), 3), dtype=np.float32)
    joints[:, names.index("lhip"), :] = [0.0, 1.0, 0.0]
    joints[:, names.index("lkne"), :] = [0.0, 0.0, 0.0]
    joints[:, names.index("lank"), :] = [0.1, 0.1, 0.0]
    pose = {
        "representation": "joints",
        "backend": "dummy",
        "fps": 30.0,
        "units": "m",
        "joint_names": names,
        "joints_3d": joints,
    }
    with registry.ensure_parent("pose3d_initial").open("wb") as f:
        pickle.dump(pose, f)
    cfg = {
        "config": {
            "optimization": {
                "joints_only": {"enabled": False},
                "pose_prior": {
                    "enabled": True,
                    "limit_set": "walking_knee90",
                    "max_joint_displacement_m": None,
                },
            }
        }
    }

    result = stage_07_optimize_pose.run(tmp_path, cfg, force=True)

    assert result["status"] == "ok"
    assert result["method"] == "pose_prior_only"
    with registry.get("pose3d_refined").open("rb") as f:
        refined = pickle.load(f)
    report = refined["refinement"]["pose_prior"]
    assert report["status"] == "ok"
    assert report["total_corrections"] == 2


def test_stage_07_contact_foot_locking_only_when_optimizer_disabled(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_standard_dirs()
    names = ["pelv", "ltoe", "rtoe"]
    joints = np.zeros((8, len(names), 3), dtype=np.float32)
    joints[:, names.index("ltoe"), 0] = np.linspace(0.0, 0.3, 8)
    joints[:, names.index("rtoe"), 0] = 0.2
    pose = {
        "representation": "joints",
        "backend": "dummy",
        "fps": 30.0,
        "units": "m",
        "joint_names": names,
        "joints_3d": joints,
    }
    with registry.ensure_parent("pose3d_initial").open("wb") as f:
        pickle.dump(pose, f)
    np.savez_compressed(
        registry.ensure_parent("contacts"),
        left_heel=np.zeros(8, dtype=np.float32),
        left_toe=np.ones(8, dtype=np.float32),
        right_heel=np.zeros(8, dtype=np.float32),
        right_toe=np.zeros(8, dtype=np.float32),
        backend="test",
        activity="walking",
    )
    cfg = {
        "config": {
            "optimization": {
                "joints_only": {"enabled": False},
                "contact_foot_locking": {
                    "enabled": True,
                    "mode": "root_translation",
                    "feet": "toes",
                    "contact_threshold": 0.5,
                    "smooth_correction_window_frames": 1,
                },
            }
        }
    }

    result = stage_07_optimize_pose.run(tmp_path, cfg, force=True)

    assert result["status"] == "ok"
    assert result["method"] == "contact_foot_locking_only"
    with registry.get("pose3d_refined").open("rb") as f:
        refined = pickle.load(f)
    assert refined["refinement"]["contact_foot_locking"]["status"] == "ok"
