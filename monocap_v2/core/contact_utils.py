from __future__ import annotations

import numpy as np


FOOT_NAME_HINTS = {
    "left_heel": ["left_heel", "lheel", "lhe", "lhee"],
    "left_toe": ["left_big_toe", "left_toe", "ltoe"],
    "right_heel": ["right_heel", "rheel", "rhe", "rhee"],
    "right_toe": ["right_big_toe", "right_toe", "rtoe"],
}


def estimate_contacts_from_joints(pose3d: dict, activity: str) -> dict:
    names = [str(n).lower() for n in pose3d.get("joint_names", [])]
    joints = np.asarray(pose3d["joints_3d"], dtype=float)
    fps = float(pose3d.get("fps") or 30.0)
    contacts = {}
    for contact_name, hints in FOOT_NAME_HINTS.items():
        idx = _find_index(names, hints)
        if idx is None:
            contacts[contact_name] = np.zeros(joints.shape[0], dtype=np.float32)
            continue
        foot = joints[:, idx, :]
        # MeTRAbs-style camera coordinates use +Y down, so convert to +Up
        # before applying a low-and-slow contact heuristic.
        vertical = -foot[:, 1]
        velocity = np.linalg.norm(np.gradient(foot, axis=0) * fps, axis=1)
        height_score = 1.0 - _normalize01(vertical)
        vel_score = 1.0 - _normalize01(velocity)
        prob = np.clip(0.65 * height_score + 0.35 * vel_score, 0.0, 1.0)
        if activity in {"squat", "sit_to_stand"}:
            prob = np.maximum(prob, 0.75)
        contacts[contact_name] = prob.astype(np.float32)
    contacts["backend"] = "heuristic"
    contacts["activity"] = activity
    return contacts


def _find_index(names: list[str], hints: list[str]) -> int | None:
    canon = [n.replace("_", "").replace("-", "") for n in names]
    for hint in hints:
        h = hint.replace("_", "").replace("-", "")
        for i, name in enumerate(canon):
            if name == h:
                return i
    for hint in hints:
        h = hint.replace("_", "").replace("-", "")
        for i, name in enumerate(canon):
            if h in name:
                return i
    return None


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = np.nanpercentile(x, 5)
    hi = np.nanpercentile(x, 95)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)
