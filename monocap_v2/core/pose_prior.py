from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from monocap_v2.core.geometry import find_joint
from monocap_v2.core.temporal_smoothing import estimate_edge_lengths, limit_joint_displacement, temporal_smoothing_metrics


@dataclass(frozen=True)
class JointAngleLimit:
    name: str
    proximal: str
    center: str
    distal: str
    min_angle_deg: float
    max_angle_deg: float
    descendants: tuple[str, ...] = ()


DEFAULT_LOWER_LIMB_LIMITS = (
    JointAngleLimit("left_knee", "left_hip", "left_knee", "left_ankle", 35.0, 180.0, ("left_toe",)),
    JointAngleLimit("right_knee", "right_hip", "right_knee", "right_ankle", 35.0, 180.0, ("right_toe",)),
    JointAngleLimit("left_ankle", "left_knee", "left_ankle", "left_toe", 45.0, 170.0, ()),
    JointAngleLimit("right_ankle", "right_knee", "right_ankle", "right_toe", 45.0, 170.0, ()),
)

JOINT_ALIASES = {
    "left_hip": ("left_hip", "lhip"),
    "right_hip": ("right_hip", "rhip"),
    "left_knee": ("left_knee", "lkne"),
    "right_knee": ("right_knee", "rkne"),
    "left_ankle": ("left_ankle", "lank"),
    "right_ankle": ("right_ankle", "rank"),
    "left_toe": ("left_toe", "left_big_toe", "left_mtp", "ltoe", "lmtp"),
    "right_toe": ("right_toe", "right_big_toe", "right_mtp", "rtoe", "rmtp"),
}


def apply_pose_prior_to_pose(
    pose: dict[str, Any],
    cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply diagnostic joint-angle limits to joints-only artifacts.

    This is not a learned body prior. It is a conservative geometric prior that
    detects and projects lower-limb angle violations without using mocap/GT.
    """

    cfg = cfg or {}
    out_pose = copy.deepcopy(pose)
    representation = str(pose.get("representation") or "")
    report: dict[str, Any] = {
        "status": "skipped",
        "representation": representation,
        "mocap_used_in_objective": False,
    }
    if representation != "joints":
        report["reason"] = f"Pose prior only applies to joints artifacts, got {representation!r}."
        return out_pose, report

    joints = np.asarray(pose.get("joints_3d"), dtype=float)
    names = [str(name) for name in pose.get("joint_names", [])]
    if joints.ndim != 3 or joints.shape[-1] != 3 or joints.shape[0] == 0 or not names:
        report["reason"] = "Pose artifact has no valid [T, J, 3] joints and joint names."
        return out_pose, report

    limits = resolve_limits(cfg)
    resolved = [_resolve_limit(limit, names) for limit in limits]
    resolved = [item for item in resolved if item is not None]
    if not resolved:
        report["reason"] = "No configured joint-angle limits matched this skeleton."
        return out_pose, report

    blend = float(np.clip(float(cfg.get("blend", 1.0)), 0.0, 1.0))
    passes = max(1, int(cfg.get("passes", 1)))
    max_displacement_m = _finite_positive_or_none(cfg.get("max_joint_displacement_m"))
    before_angles = angle_limit_metrics(joints, names, limits)
    corrected = joints.copy()
    correction_counts: dict[str, int] = {limit.name: 0 for limit, *_rest in resolved}
    for _ in range(passes):
        corrected, counts = project_angle_limits(corrected, names, resolved, blend=blend)
        for key, value in counts.items():
            correction_counts[key] = correction_counts.get(key, 0) + int(value)
    if max_displacement_m is not None:
        corrected = limit_joint_displacement(corrected, joints, max_displacement_m)
    corrected[~np.isfinite(joints)] = np.nan
    after_angles = angle_limit_metrics(corrected, names, limits)

    target_lengths = estimate_edge_lengths(joints, names)
    before_metrics = temporal_smoothing_metrics(joints, names, fps=float(pose.get("fps") or 30.0), target_lengths=target_lengths)
    after_metrics = temporal_smoothing_metrics(corrected, names, fps=float(pose.get("fps") or 30.0), target_lengths=target_lengths)
    after_metrics["mean_joint_displacement_m"] = _mean_joint_displacement(corrected, joints)
    after_metrics["max_joint_displacement_m"] = _max_joint_displacement(corrected, joints)

    out_pose["joints_3d"] = corrected.astype(np.float32)
    prior_report = {
        "status": "ok",
        "method": "geometric_joint_angle_limits",
        "limit_set": str(cfg.get("limit_set") or "lower_limb_conservative"),
        "blend": blend,
        "passes": passes,
        "max_joint_displacement_m": max_displacement_m,
        "limits": [limit_to_dict(limit) for limit, *_rest in resolved],
        "correction_counts": correction_counts,
        "total_corrections": int(sum(correction_counts.values())),
        "angle_metrics_before": before_angles,
        "angle_metrics_after": after_angles,
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
        "warnings": _warnings(correction_counts, after_metrics),
        "mocap_used_in_objective": False,
    }
    out_pose["pose_prior"] = prior_report
    return out_pose, prior_report


def resolve_limits(cfg: dict[str, Any] | None = None) -> list[JointAngleLimit]:
    cfg = cfg or {}
    configured = cfg.get("limits")
    if configured:
        return [
            JointAngleLimit(
                name=str(item["name"]),
                proximal=str(item["proximal"]),
                center=str(item["center"]),
                distal=str(item["distal"]),
                min_angle_deg=float(item.get("min_angle_deg", 0.0)),
                max_angle_deg=float(item.get("max_angle_deg", 180.0)),
                descendants=tuple(str(v) for v in item.get("descendants", [])),
            )
            for item in configured
        ]
    limit_set = str(cfg.get("limit_set") or "lower_limb_conservative").lower()
    if limit_set in {"lower_limb_conservative", "conservative"}:
        return list(DEFAULT_LOWER_LIMB_LIMITS)
    if limit_set in {"walking_knee90", "knee90"}:
        return [
            JointAngleLimit("left_knee", "left_hip", "left_knee", "left_ankle", 90.0, 180.0, ("left_toe",)),
            JointAngleLimit("right_knee", "right_hip", "right_knee", "right_ankle", 90.0, 180.0, ("right_toe",)),
        ]
    if limit_set in {"walking_knee105", "knee105"}:
        return [
            JointAngleLimit("left_knee", "left_hip", "left_knee", "left_ankle", 105.0, 180.0, ("left_toe",)),
            JointAngleLimit("right_knee", "right_hip", "right_knee", "right_ankle", 105.0, 180.0, ("right_toe",)),
        ]
    if limit_set in {"walking_lower_limb", "lower_limb_walking"}:
        return [
            JointAngleLimit("left_knee", "left_hip", "left_knee", "left_ankle", 105.0, 180.0, ("left_toe",)),
            JointAngleLimit("right_knee", "right_hip", "right_knee", "right_ankle", 105.0, 180.0, ("right_toe",)),
            JointAngleLimit("left_ankle", "left_knee", "left_ankle", "left_toe", 55.0, 155.0, ()),
            JointAngleLimit("right_ankle", "right_knee", "right_ankle", "right_toe", 55.0, 155.0, ()),
        ]
    raise ValueError(f"Unsupported pose-prior limit set: {limit_set}")


def project_angle_limits(
    joints: np.ndarray,
    joint_names: list[str],
    resolved_limits: list[tuple[JointAngleLimit, int, int, int, list[int]]],
    blend: float = 1.0,
) -> tuple[np.ndarray, dict[str, int]]:
    values = np.asarray(joints, dtype=float)
    out = values.copy()
    blend = float(np.clip(blend, 0.0, 1.0))
    counts: dict[str, int] = {limit.name: 0 for limit, *_rest in resolved_limits}
    if blend <= 0.0:
        return out, counts
    for t in range(out.shape[0]):
        for limit, proximal_idx, center_idx, distal_idx, descendant_indices in resolved_limits:
            parent = out[t, proximal_idx]
            center = out[t, center_idx]
            child = out[t, distal_idx]
            corrected_child = project_child_to_angle_limit(
                parent,
                center,
                child,
                min_angle_deg=limit.min_angle_deg,
                max_angle_deg=limit.max_angle_deg,
            )
            if corrected_child is None:
                continue
            delta = blend * (corrected_child - child)
            if not np.isfinite(delta).all() or np.linalg.norm(delta) < 1e-12:
                continue
            out[t, distal_idx] = child + delta
            for desc_idx in descendant_indices:
                if np.isfinite(out[t, desc_idx]).all():
                    out[t, desc_idx] = out[t, desc_idx] + delta
            counts[limit.name] = counts.get(limit.name, 0) + 1
    return out, counts


def project_child_to_angle_limit(
    parent: np.ndarray,
    center: np.ndarray,
    child: np.ndarray,
    min_angle_deg: float,
    max_angle_deg: float,
) -> np.ndarray | None:
    if not (np.isfinite(parent).all() and np.isfinite(center).all() and np.isfinite(child).all()):
        return None
    a = parent - center
    b = child - center
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm < 1e-12 or b_norm < 1e-12:
        return None
    a_dir = a / a_norm
    b_dir = b / b_norm
    angle = angle_between_deg(a_dir, b_dir)
    if angle is None:
        return None
    target = float(np.clip(angle, float(min_angle_deg), float(max_angle_deg)))
    if abs(target - angle) < 1e-9:
        return None
    perp = b_dir - np.dot(b_dir, a_dir) * a_dir
    perp_norm = float(np.linalg.norm(perp))
    if perp_norm < 1e-9:
        perp = _orthogonal_unit(a_dir)
    else:
        perp = perp / perp_norm
    theta = np.deg2rad(target)
    target_dir = np.cos(theta) * a_dir + np.sin(theta) * perp
    norm = np.linalg.norm(target_dir)
    if not np.isfinite(norm) or norm < 1e-12:
        return None
    return center + b_norm * target_dir / norm


def angle_limit_metrics(joints: np.ndarray, joint_names: list[str], limits: list[JointAngleLimit]) -> dict[str, Any]:
    values = np.asarray(joints, dtype=float)
    out: dict[str, Any] = {}
    for limit in limits:
        resolved = _resolve_limit(limit, joint_names)
        if resolved is None:
            out[limit.name] = {"status": "missing_joints"}
            continue
        _limit, proximal_idx, center_idx, distal_idx, _desc = resolved
        angles = joint_angles_deg(values[:, proximal_idx, :], values[:, center_idx, :], values[:, distal_idx, :])
        finite = np.isfinite(angles)
        violations = finite & ((angles < limit.min_angle_deg) | (angles > limit.max_angle_deg))
        valid = angles[finite]
        out[limit.name] = {
            "status": "ok",
            "min_angle_deg": float(limit.min_angle_deg),
            "max_angle_deg": float(limit.max_angle_deg),
            "valid_frames": int(np.count_nonzero(finite)),
            "violation_frames": int(np.count_nonzero(violations)),
            "violation_ratio": float(np.count_nonzero(violations) / finite.size) if finite.size else 0.0,
            "median_deg": _safe_float(np.nanmedian(valid)) if valid.size else None,
            "min_deg": _safe_float(np.nanmin(valid)) if valid.size else None,
            "max_deg": _safe_float(np.nanmax(valid)) if valid.size else None,
            "p05_deg": _safe_float(np.nanpercentile(valid, 5)) if valid.size else None,
            "p95_deg": _safe_float(np.nanpercentile(valid, 95)) if valid.size else None,
        }
    return out


def joint_angles_deg(parent: np.ndarray, center: np.ndarray, child: np.ndarray) -> np.ndarray:
    a = np.asarray(parent, dtype=float) - np.asarray(center, dtype=float)
    b = np.asarray(child, dtype=float) - np.asarray(center, dtype=float)
    dot = np.sum(a * b, axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cos = np.full(dot.shape, np.nan, dtype=float)
    valid = np.isfinite(dot) & np.isfinite(denom) & (denom > 1e-12)
    cos[valid] = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
    return np.rad2deg(np.arccos(cos))


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not np.isfinite(denom) or denom < 1e-12:
        return None
    cos = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cos)))


def limit_to_dict(limit: JointAngleLimit) -> dict[str, Any]:
    return {
        "name": limit.name,
        "proximal": limit.proximal,
        "center": limit.center,
        "distal": limit.distal,
        "min_angle_deg": float(limit.min_angle_deg),
        "max_angle_deg": float(limit.max_angle_deg),
        "descendants": list(limit.descendants),
    }


def _resolve_limit(
    limit: JointAngleLimit,
    joint_names: list[str],
) -> tuple[JointAngleLimit, int, int, int, list[int]] | None:
    proximal = _joint_index(joint_names, limit.proximal)
    center = _joint_index(joint_names, limit.center)
    distal = _joint_index(joint_names, limit.distal)
    if proximal is None or center is None or distal is None:
        return None
    descendants = []
    for name in limit.descendants:
        idx = _joint_index(joint_names, name)
        if idx is not None:
            descendants.append(idx)
    return limit, proximal, center, distal, descendants


def _joint_index(joint_names: list[str], canonical_name: str) -> int | None:
    return find_joint(joint_names, JOINT_ALIASES.get(canonical_name, (canonical_name,)))


def _orthogonal_unit(vec: np.ndarray) -> np.ndarray:
    base = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(vec, base))) > 0.9:
        base = np.array([0.0, 1.0, 0.0], dtype=float)
    out = base - np.dot(base, vec) * vec
    norm = float(np.linalg.norm(out))
    return out / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=float)


def _warnings(correction_counts: dict[str, int], metrics_after: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if sum(correction_counts.values()) == 0:
        warnings.append("No joint-limit violations were corrected; this prior likely has no effect for this artifact.")
    max_disp = metrics_after.get("max_joint_displacement_m")
    if max_disp is not None and float(max_disp) > 0.15:
        warnings.append("Pose-prior correction displaced at least one joint by more than 0.15 m.")
    return warnings


def _mean_joint_displacement(a: np.ndarray, b: np.ndarray) -> float | None:
    dist = np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), axis=-1)
    return _safe_float(np.nanmean(dist))


def _max_joint_displacement(a: np.ndarray, b: np.ndarray) -> float | None:
    dist = np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), axis=-1)
    return _safe_float(np.nanmax(dist))


def _finite_positive_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out) or out <= 0:
        return None
    return out


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None
