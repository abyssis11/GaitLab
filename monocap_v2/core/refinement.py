from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from monocap_v2.core.geometry import edge_indices, find_joint, project_points


FOOT_CONTACTS = [
    ("left_toe", ("ltoe", "left_toe", "left_big_toe")),
    ("right_toe", ("rtoe", "right_toe", "right_big_toe")),
]

HORIZONTAL_COORDS = (0, 2)


@dataclass(frozen=True)
class ResidualSpec:
    kind: str
    weight: float
    deps: tuple[int, ...]
    args: tuple[Any, ...]


def optimize_joints_only(
    pose: dict,
    camera: dict,
    pose2d: dict | None,
    contacts: dict | None,
    activity_weights: dict,
    optimizer_cfg: dict,
) -> tuple[np.ndarray, dict]:
    from scipy.optimize import least_squares

    initial = np.asarray(pose["joints_3d"], dtype=float)
    names = [str(name) for name in pose.get("joint_names", [])]
    valid = np.isfinite(initial)
    var_index = np.full(initial.shape, -1, dtype=int)
    var_index[valid] = np.arange(int(np.count_nonzero(valid)))
    x0 = initial[valid].astype(float)

    specs = build_residual_specs(initial, names, var_index, camera, pose2d, contacts, activity_weights, optimizer_cfg, float(pose.get("fps") or 30.0))
    if not specs:
        return initial.astype(np.float32), {
            "status": "skipped",
            "reason": "No finite optimization variables or residuals.",
            "n_variables": int(x0.size),
            "n_residuals": 0,
        }

    def residual_fn(x: np.ndarray) -> np.ndarray:
        joints = unpack_variables(initial, valid, x)
        return evaluate_specs(specs, joints, camera)

    jac_sparsity = build_jac_sparsity(specs, x0.size)
    r0 = residual_fn(x0)
    result = least_squares(
        residual_fn,
        x0,
        jac_sparsity=jac_sparsity,
        loss=str(optimizer_cfg.get("robust_loss", "soft_l1")),
        f_scale=float(optimizer_cfg.get("f_scale", 0.05)),
        max_nfev=int(optimizer_cfg.get("max_nfev", 30)),
        x_scale="jac",
        verbose=0,
    )
    refined = unpack_variables(initial, valid, result.x).astype(np.float32)
    r1 = residual_fn(result.x)
    initial_cost = float(0.5 * np.sum(r0**2))
    final_cost = float(0.5 * np.sum(r1**2))
    improved = bool(np.isfinite(final_cost) and final_cost <= initial_cost)
    return refined, {
        "status": "ok" if improved else "warning",
        "success": bool(improved),
        "convergence_success": bool(result.success),
        "message": str(result.message),
        "scipy_status": int(result.status),
        "nfev": int(result.nfev),
        "njev": int(result.njev) if result.njev is not None else None,
        "n_variables": int(x0.size),
        "n_residuals": int(r0.size),
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "cost_reduction": float(initial_cost - final_cost),
        "residual_groups": dict(Counter(spec.kind for spec in specs)),
    }


def build_residual_specs(
    initial: np.ndarray,
    names: list[str],
    var_index: np.ndarray,
    camera: dict,
    pose2d: dict | None,
    contacts: dict | None,
    activity_weights: dict,
    optimizer_cfg: dict,
    fps: float,
) -> list[ResidualSpec]:
    specs: list[ResidualSpec] = []
    weights = optimizer_cfg.get("weights", {})
    fidelity_w = float(weights.get("fidelity", 8.0))
    bone_w = float(weights.get("bone", 30.0))
    reproj_w = float(activity_weights.get("reprojection", weights.get("reprojection", 10.0)))
    smooth_w = float(activity_weights.get("smoothness", weights.get("smoothness", 1.0)))
    contact_velocity_w = float(activity_weights.get("contact_velocity", weights.get("contact_velocity", 1.0)))
    contact_position_w = float(activity_weights.get("contact_position", weights.get("contact_position", 10.0)))
    flat_floor_w = float(activity_weights.get("flat_floor", weights.get("flat_floor", 1.0)))
    contact_threshold = float(optimizer_cfg.get("contact_threshold", 0.75))

    _add_fidelity_specs(specs, initial, var_index, np.sqrt(fidelity_w))
    _add_reprojection_specs(specs, initial, var_index, camera, pose2d, np.sqrt(reproj_w))
    _add_bone_specs(specs, initial, names, var_index, np.sqrt(bone_w))
    _add_smoothness_specs(specs, initial, var_index, np.sqrt(smooth_w))
    _add_contact_specs(
        specs,
        initial,
        names,
        var_index,
        contacts,
        fps,
        np.sqrt(contact_velocity_w),
        np.sqrt(contact_position_w),
        np.sqrt(flat_floor_w),
        contact_threshold,
        int(optimizer_cfg.get("min_contact_segment_frames", 3)),
    )
    return specs


def unpack_variables(template: np.ndarray, valid: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.asarray(template, dtype=float).copy()
    out[valid] = x
    return out


def evaluate_specs(specs: list[ResidualSpec], joints: np.ndarray, camera: dict) -> np.ndarray:
    values = np.empty((len(specs),), dtype=float)
    projections_cache: dict[int, np.ndarray] = {}
    for row, spec in enumerate(specs):
        w = spec.weight
        if spec.kind == "fidelity":
            t, j, c, target = spec.args
            values[row] = w * (joints[t, j, c] - target)
        elif spec.kind == "reprojection":
            t, j, c, target, diag = spec.args
            if t not in projections_cache:
                projections_cache[t] = project_points(joints[t], camera)
            values[row] = w * (projections_cache[t][j, c] - target) / diag
        elif spec.kind == "bone":
            t, a, b, target = spec.args
            values[row] = w * (np.linalg.norm(joints[t, a] - joints[t, b]) - target)
        elif spec.kind == "smoothness":
            t, j, c = spec.args
            values[row] = w * (joints[t - 1, j, c] - 2.0 * joints[t, j, c] + joints[t + 1, j, c])
        elif spec.kind == "contact_velocity":
            t, j, c, fps = spec.args
            values[row] = w * (joints[t + 1, j, c] - joints[t, j, c]) * fps
        elif spec.kind == "contact_anchor":
            t, j, c, target = spec.args
            values[row] = w * (joints[t, j, c] - target)
        elif spec.kind == "flat_floor":
            t, j, target_y_down = spec.args
            values[row] = w * (joints[t, j, 1] - target_y_down)
        else:  # pragma: no cover - defensive guard
            raise ValueError(f"Unknown residual kind: {spec.kind}")
    return values


def build_jac_sparsity(specs: list[ResidualSpec], n_variables: int):
    from scipy.sparse import lil_matrix

    mat = lil_matrix((len(specs), n_variables), dtype=int)
    for row, spec in enumerate(specs):
        for dep in spec.deps:
            if dep >= 0:
                mat[row, dep] = 1
    return mat.tocsr()


def compute_refinement_metrics(joints: np.ndarray, names: list[str], fps: float, contacts: dict | None = None, contact_threshold: float = 0.75) -> dict:
    joints = np.asarray(joints, dtype=float)
    pelvis_idx = find_joint(names, ("pelv", "pelvis", "root"))
    metrics: dict[str, Any] = {"finite_ratio": float(np.isfinite(joints).mean()) if joints.size else 0.0}
    if pelvis_idx is not None and joints.shape[0] > 0:
        pelvis = joints[:, pelvis_idx, :]
        finite = np.isfinite(pelvis).all(axis=1)
        drift = np.full((joints.shape[0],), np.nan, dtype=float)
        if np.any(finite):
            first = pelvis[np.flatnonzero(finite)[0]]
            drift[finite] = np.linalg.norm(pelvis[finite][:, HORIZONTAL_COORDS] - first[list(HORIZONTAL_COORDS)], axis=1)
        metrics["pelvis_drift_m"] = {
            "end": _safe_float(drift[-1]),
            "max": _safe_float(np.nanmax(drift)),
            "mean": _safe_float(np.nanmean(drift)),
        }
    speeds = []
    contact_speeds = []
    for contact_key, candidates in FOOT_CONTACTS:
        idx = find_joint(names, candidates)
        if idx is None:
            continue
        foot = joints[:, idx, :]
        speed = np.linalg.norm(np.gradient(foot, axis=0) * fps, axis=1)
        speeds.append(speed)
        prob = _contact_prob(contacts, contact_key, joints.shape[0])
        mask = prob > contact_threshold
        if np.any(mask):
            contact_speeds.append(speed[mask])
    if speeds:
        metrics["mean_joint_speed_mps"] = _safe_float(np.nanmean(np.concatenate(speeds)))
    if contact_speeds:
        metrics["mean_foot_speed_during_contact_mps"] = _safe_float(np.nanmean(np.concatenate(contact_speeds)))
    else:
        metrics["mean_foot_speed_during_contact_mps"] = None
    return metrics


def _add_fidelity_specs(specs: list[ResidualSpec], initial: np.ndarray, var_index: np.ndarray, weight: float) -> None:
    for t, j, c in np.argwhere(var_index >= 0):
        dep = int(var_index[t, j, c])
        specs.append(ResidualSpec("fidelity", weight, (dep,), (int(t), int(j), int(c), float(initial[t, j, c]))))


def _add_reprojection_specs(
    specs: list[ResidualSpec],
    initial: np.ndarray,
    var_index: np.ndarray,
    camera: dict,
    pose2d: dict | None,
    weight: float,
) -> None:
    if not pose2d or pose2d.get("xy") is None:
        return
    xy = np.asarray(pose2d.get("xy"), dtype=float)
    conf = np.asarray(pose2d.get("confidence", np.ones(xy.shape[:2])), dtype=float)
    if xy.shape[:2] != initial.shape[:2]:
        return
    diag = float(np.hypot(float(camera.get("width", 1)), float(camera.get("height", 1)))) or 1.0
    for t in range(initial.shape[0]):
        for j in range(initial.shape[1]):
            deps = tuple(int(v) for v in var_index[t, j, :] if v >= 0)
            if len(deps) != 3 or not np.isfinite(xy[t, j]).all():
                continue
            c_weight = weight * float(np.sqrt(max(conf[t, j], 0.0)))
            if c_weight <= 0:
                continue
            specs.append(ResidualSpec("reprojection", c_weight, deps, (t, j, 0, float(xy[t, j, 0]), diag)))
            specs.append(ResidualSpec("reprojection", c_weight, deps, (t, j, 1, float(xy[t, j, 1]), diag)))


def _add_bone_specs(specs: list[ResidualSpec], initial: np.ndarray, names: list[str], var_index: np.ndarray, weight: float) -> None:
    for a, b in edge_indices(names):
        lengths = np.linalg.norm(initial[:, a, :] - initial[:, b, :], axis=1)
        target = float(np.nanmedian(lengths))
        if not np.isfinite(target) or target <= 0:
            continue
        for t in range(initial.shape[0]):
            deps = tuple(int(v) for v in np.concatenate([var_index[t, a, :], var_index[t, b, :]]) if v >= 0)
            if len(deps) == 6:
                specs.append(ResidualSpec("bone", weight, deps, (t, a, b, target)))


def _add_smoothness_specs(specs: list[ResidualSpec], initial: np.ndarray, var_index: np.ndarray, weight: float) -> None:
    if initial.shape[0] < 3:
        return
    for t in range(1, initial.shape[0] - 1):
        for j in range(initial.shape[1]):
            for c in range(3):
                deps = (int(var_index[t - 1, j, c]), int(var_index[t, j, c]), int(var_index[t + 1, j, c]))
                if min(deps) >= 0:
                    specs.append(ResidualSpec("smoothness", weight, deps, (t, j, c)))


def _add_contact_specs(
    specs: list[ResidualSpec],
    initial: np.ndarray,
    names: list[str],
    var_index: np.ndarray,
    contacts: dict | None,
    fps: float,
    velocity_weight: float,
    position_weight: float,
    floor_weight: float,
    threshold: float,
    min_segment_frames: int,
) -> None:
    if not contacts:
        return
    floor_targets = []
    for contact_key, candidates in FOOT_CONTACTS:
        idx = find_joint(names, candidates)
        if idx is None:
            continue
        prob = _contact_prob(contacts, contact_key, initial.shape[0])
        mask = prob > threshold
        if np.any(mask):
            floor_targets.extend(initial[mask, idx, 1].tolist())
        for t in range(initial.shape[0] - 1):
            if not (mask[t] and mask[t + 1]):
                continue
            for c in HORIZONTAL_COORDS:
                deps = (int(var_index[t, idx, c]), int(var_index[t + 1, idx, c]))
                if min(deps) >= 0:
                    specs.append(ResidualSpec("contact_velocity", velocity_weight, deps, (t, idx, c, fps)))
        for segment in _contiguous_segments(mask, min_segment_frames):
            anchor = np.nanmedian(initial[segment, idx, :][:, HORIZONTAL_COORDS], axis=0)
            for t in segment:
                for local_idx, c in enumerate(HORIZONTAL_COORDS):
                    dep = int(var_index[t, idx, c])
                    if dep >= 0 and np.isfinite(anchor[local_idx]):
                        specs.append(ResidualSpec("contact_anchor", position_weight, (dep,), (t, idx, c, float(anchor[local_idx]))))
    if floor_targets and floor_weight > 0:
        target_y_down = float(np.nanpercentile(np.asarray(floor_targets, dtype=float), 95))
        for contact_key, candidates in FOOT_CONTACTS:
            idx = find_joint(names, candidates)
            if idx is None:
                continue
            mask = _contact_prob(contacts, contact_key, initial.shape[0]) > threshold
            for t in np.flatnonzero(mask):
                dep = int(var_index[t, idx, 1])
                if dep >= 0:
                    specs.append(ResidualSpec("flat_floor", floor_weight, (dep,), (int(t), idx, target_y_down)))


def _contact_prob(contacts: dict | None, key: str, n_frames: int) -> np.ndarray:
    if not contacts or key not in contacts:
        return np.zeros((n_frames,), dtype=float)
    arr = np.asarray(contacts[key], dtype=float).reshape(-1)
    out = np.zeros((n_frames,), dtype=float)
    out[: min(n_frames, arr.size)] = arr[: min(n_frames, arr.size)]
    return out


def _contiguous_segments(mask: np.ndarray, min_len: int) -> list[np.ndarray]:
    segments = []
    start = None
    for idx, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            if idx - start >= min_len:
                segments.append(np.arange(start, idx, dtype=int))
            start = None
    return segments


def _safe_float(value) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None
