from __future__ import annotations

import numpy as np

from monocap_v2.core.foot_locking import apply_contact_foot_locking_to_pose, foot_locking_metrics


def test_root_translation_foot_locking_reduces_contact_sliding_and_preserves_local_pose() -> None:
    frames = 12
    names = ["pelv", "ltoe", "rtoe"]
    joints = np.zeros((frames, len(names), 3), dtype=np.float32)
    joints[:, 0, :] = [0.0, 0.0, 4.0]
    joints[:, 1, :] = [0.0, 0.8, 4.0]
    joints[:, 2, :] = [0.2, 0.8, 4.0]
    joints[:, :, 0] += np.linspace(0.0, 0.4, frames)[:, None]
    contacts = {
        "left_toe": np.ones(frames, dtype=np.float32),
        "right_toe": np.zeros(frames, dtype=np.float32),
        "left_heel": np.zeros(frames, dtype=np.float32),
        "right_heel": np.zeros(frames, dtype=np.float32),
    }
    pose = {
        "representation": "joints",
        "backend": "test",
        "fps": 30.0,
        "units": "m",
        "joint_names": names,
        "joints_3d": joints,
    }
    before = foot_locking_metrics(joints, names, contacts, fps=30.0, contact_threshold=0.5)

    corrected, report = apply_contact_foot_locking_to_pose(
        pose,
        contacts,
        {"mode": "root_translation", "feet": "toes", "contact_threshold": 0.5, "smooth_correction_window_frames": 1},
    )
    after = foot_locking_metrics(corrected["joints_3d"], names, contacts, fps=30.0, contact_threshold=0.5)

    assert report["status"] == "ok"
    assert after["mean_contact_horizontal_speed_mps"] < before["mean_contact_horizontal_speed_mps"]
    np.testing.assert_allclose(
        corrected["joints_3d"][:, 1, :] - corrected["joints_3d"][:, 0, :],
        joints[:, 1, :] - joints[:, 0, :],
        atol=1e-6,
    )


def test_endpoint_foot_locking_moves_only_contact_endpoint() -> None:
    frames = 8
    names = ["pelv", "ltoe"]
    joints = np.zeros((frames, len(names), 3), dtype=np.float32)
    joints[:, 1, 0] = np.linspace(0.0, 0.2, frames)
    contacts = {
        "left_toe": np.ones(frames, dtype=np.float32),
        "right_toe": np.zeros(frames, dtype=np.float32),
        "left_heel": np.zeros(frames, dtype=np.float32),
        "right_heel": np.zeros(frames, dtype=np.float32),
    }
    pose = {"representation": "joints", "backend": "test", "fps": 30.0, "joint_names": names, "joints_3d": joints}

    corrected, report = apply_contact_foot_locking_to_pose(
        pose,
        contacts,
        {"mode": "endpoint", "feet": "toes", "contact_threshold": 0.5},
    )

    assert report["status"] == "ok"
    np.testing.assert_allclose(corrected["joints_3d"][:, 0, :], joints[:, 0, :])
    assert np.nanstd(corrected["joints_3d"][:, 1, 0]) < np.nanstd(joints[:, 1, 0])


def test_contact_foot_locking_root_translation_updates_hybrid_smpl_consistently() -> None:
    frames = 6
    joints = np.zeros((frames, 2, 3), dtype=np.float32)
    joints[:, 1, 0] = np.linspace(0.0, 0.2, frames)
    vertices = np.zeros((frames, 3, 3), dtype=np.float32)
    vertices[:, :, 0] = joints[:, :1, 0] + np.array([0.0, 0.1, 0.2], dtype=np.float32)
    transl = np.zeros((frames, 3), dtype=np.float32)
    pose = {
        "representation": "hybrid",
        "backend": "wham",
        "joint_names": ["pelv", "ltoe"],
        "joints_3d": joints,
        "smpl": {"vertices": vertices.copy(), "transl": transl.copy()},
    }
    contacts = {"left_toe": np.ones(frames, dtype=np.float32)}

    corrected, report = apply_contact_foot_locking_to_pose(
        pose,
        contacts,
        {"mode": "root_translation", "feet": "toes", "contact_threshold": 0.5, "smooth_correction_window_frames": 1},
    )

    assert report["status"] == "ok"
    assert report["smpl_consistency"]["status"] == "ok"
    assert "smpl.vertices" in report["smpl_consistency"]["updated_fields"]
    assert "smpl.transl" in report["smpl_consistency"]["updated_fields"]
    correction = corrected["joints_3d"][:, 0, :] - joints[:, 0, :]
    np.testing.assert_allclose(corrected["smpl"]["transl"], transl + correction, atol=1e-6)
    np.testing.assert_allclose(corrected["smpl"]["vertices"], vertices + correction[:, None, :], atol=1e-6)


def test_contact_foot_locking_endpoint_skips_hybrid_artifacts() -> None:
    pose = {
        "representation": "hybrid",
        "backend": "wham",
        "joint_names": ["pelv", "ltoe"],
        "joints_3d": np.zeros((3, 2, 3), dtype=np.float32),
    }

    corrected, report = apply_contact_foot_locking_to_pose(pose, {"left_toe": np.ones(3)}, {"mode": "endpoint"})

    assert report["status"] == "skipped"
    assert "SMPL-consistent" in report["reason"]
    np.testing.assert_allclose(corrected["joints_3d"], pose["joints_3d"])
