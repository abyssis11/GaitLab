from __future__ import annotations

import numpy as np

from monocap_v2.core.refinement import compute_refinement_metrics, optimize_joints_only


def test_optimizer_reduces_synthetic_contact_sliding() -> None:
    frames = 8
    joints = np.zeros((frames, 3, 3), dtype=np.float32)
    joints[:, 0, :] = np.array([0.0, 0.2, 4.0])
    joints[:, 1, 0] = np.linspace(0.0, 0.5, frames)
    joints[:, 1, 1] = 1.1
    joints[:, 1, 2] = 4.0
    joints[:, 2, :] = np.array([0.2, 1.1, 4.0])
    pose = {
        "representation": "joints",
        "backend": "test",
        "fps": 30.0,
        "units": "m",
        "joint_names": ["pelv", "ltoe", "rtoe"],
        "joints_3d": joints,
    }
    contacts = {
        "left_toe": np.ones(frames, dtype=np.float32),
        "right_toe": np.zeros(frames, dtype=np.float32),
    }
    cfg = {
        "max_nfev": 25,
        "robust_loss": "linear",
        "f_scale": 0.1,
        "contact_threshold": 0.5,
        "min_contact_segment_frames": 2,
        "weights": {
            "fidelity": 0.05,
            "bone": 0.0,
            "smoothness": 0.1,
            "contact_velocity": 10.0,
            "contact_position": 100.0,
            "flat_floor": 0.0,
        },
    }
    before = compute_refinement_metrics(joints, pose["joint_names"], pose["fps"], contacts, contact_threshold=0.5)
    refined, report = optimize_joints_only(pose, {}, None, contacts, {}, cfg)
    after = compute_refinement_metrics(refined, pose["joint_names"], pose["fps"], contacts, contact_threshold=0.5)
    assert report["n_variables"] == int(np.isfinite(joints).sum())
    assert report["final_cost"] < report["initial_cost"]
    assert after["mean_foot_speed_during_contact_mps"] < before["mean_foot_speed_during_contact_mps"]
