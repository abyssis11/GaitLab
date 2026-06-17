from __future__ import annotations

import copy
from typing import Any

import numpy as np

from monocap_v2.core.geometry import find_joint, project_points
from monocap_v2.core.temporal_smoothing import (
    estimate_edge_lengths,
    limit_joint_displacement,
    project_bones_to_lengths,
)


def apply_reprojection_consistency_to_pose(
    pose: dict[str, Any],
    camera: dict[str, Any],
    pose2d: dict[str, Any] | None,
    cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pull 3D joints toward their observed 2D rays while keeping depth fixed.

    This is a diagnostic joints-only correction. It never uses mocap/OpenSim
    data. It assumes the 3D joints are in a camera-like coordinate system where
    pinhole projection through `camera` is meaningful.
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
        report["reason"] = f"Reprojection consistency only applies to joints artifacts, got {representation!r}."
        return out_pose, report
    if not pose2d:
        pose2d = pose.get("pose2d") if isinstance(pose.get("pose2d"), dict) else None
    if not pose2d:
        report["reason"] = "No 2D keypoints are available."
        return out_pose, report

    joints = np.asarray(pose.get("joints_3d"), dtype=float)
    joint_names = [str(name) for name in pose.get("joint_names", [])]
    if joints.ndim != 3 or joints.shape[-1] != 3 or not joint_names:
        report["reason"] = "Pose artifact has no valid [T, J, 3] joints and joint names."
        return out_pose, report

    xy, conf, align_report = align_pose2d_to_joints(pose2d, joint_names, joints.shape[:2])
    if xy is None or conf is None:
        report["reason"] = align_report.get("reason", "2D keypoints could not be aligned to 3D joints.")
        report["pose2d_alignment"] = align_report
        return out_pose, report

    blend = float(np.clip(float(cfg.get("blend", 0.5)), 0.0, 1.0))
    confidence_threshold = float(cfg.get("confidence_threshold", 0.25))
    min_depth_m = float(cfg.get("min_depth_m", 0.1))
    preserve_bones = bool(cfg.get("preserve_bones", True))
    bone_iterations = max(0, int(cfg.get("bone_projection_iterations", 2)))
    bone_blend = float(np.clip(float(cfg.get("bone_preservation_blend", 1.0)), 0.0, 1.0))
    max_displacement_m = _finite_positive_or_none(cfg.get("max_joint_displacement_m"))

    before_metrics = reprojection_metrics(joints, camera, xy, conf, confidence_threshold, min_depth_m)
    target = depth_preserving_reprojection_targets(joints, xy, camera)
    valid = (
        np.isfinite(target).all(axis=-1)
        & np.isfinite(joints).all(axis=-1)
        & (joints[..., 2] > min_depth_m)
        & np.isfinite(conf)
        & (conf >= confidence_threshold)
    )
    corrected = joints.copy()
    alpha = blend * np.clip(conf, 0.0, 1.0)
    corrected[valid, 0] = joints[valid, 0] + alpha[valid] * (target[valid, 0] - joints[valid, 0])
    corrected[valid, 1] = joints[valid, 1] + alpha[valid] * (target[valid, 1] - joints[valid, 1])

    target_lengths = estimate_edge_lengths(joints, joint_names)
    if preserve_bones and target_lengths and bone_iterations > 0:
        corrected = project_bones_to_lengths(
            corrected,
            joints,
            joint_names,
            targets=target_lengths,
            iterations=bone_iterations,
            blend=bone_blend,
        )
    if max_displacement_m is not None:
        corrected = limit_joint_displacement(corrected, joints, max_displacement_m)
    corrected[~np.isfinite(joints)] = np.nan
    after_metrics = reprojection_metrics(corrected, camera, xy, conf, confidence_threshold, min_depth_m)

    out_pose["joints_3d"] = corrected.astype(np.float32)
    reproj_report = {
        "status": "ok",
        "method": "depth_preserving_reprojection_consistency",
        "blend": blend,
        "confidence_threshold": confidence_threshold,
        "min_depth_m": min_depth_m,
        "preserve_bones": preserve_bones,
        "bone_projection_iterations": bone_iterations,
        "bone_preservation_blend": bone_blend,
        "max_joint_displacement_m": max_displacement_m,
        "pose2d_alignment": align_report,
        "valid_observations": int(np.count_nonzero(valid)),
        "valid_observation_ratio": float(np.count_nonzero(valid) / valid.size) if valid.size else 0.0,
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
        "mocap_used_in_objective": False,
    }
    out_pose["reprojection_consistency"] = reproj_report
    return out_pose, reproj_report


def depth_preserving_reprojection_targets(joints: np.ndarray, xy: np.ndarray, camera: dict[str, Any]) -> np.ndarray:
    values = np.asarray(joints, dtype=float)
    obs = np.asarray(xy, dtype=float)
    target = np.full_like(values, np.nan, dtype=float)
    z = values[..., 2]
    intrinsics = _camera_intrinsics(camera)
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    target[..., 0] = (obs[..., 0] - cx) * z / fx
    target[..., 1] = (obs[..., 1] - cy) * z / fy
    target[..., 2] = z
    return target


def align_pose2d_to_joints(
    pose2d: dict[str, Any],
    joint_names: list[str],
    target_shape: tuple[int, int],
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    xy_raw = np.asarray(pose2d.get("xy"), dtype=float)
    if xy_raw.ndim != 3 or xy_raw.shape[-1] != 2:
        return None, None, {"status": "failed", "reason": f"pose2d xy has invalid shape {xy_raw.shape}."}
    conf_raw = np.asarray(pose2d.get("confidence", np.ones(xy_raw.shape[:2])), dtype=float)
    pose2d_names = [str(name) for name in pose2d.get("names", [])]
    frames, joints = target_shape
    xy = np.full((frames, joints, 2), np.nan, dtype=float)
    conf = np.zeros((frames, joints), dtype=float)
    if xy_raw.shape[:2] == target_shape:
        xy[:] = xy_raw[:frames, :joints]
        conf[:] = conf_raw[:frames, :joints]
        return xy, conf, {"status": "ok", "mode": "same_shape", "matched_joints": int(joints), "matched_frames": int(frames)}
    if not pose2d_names:
        return None, None, {"status": "failed", "reason": "pose2d names are unavailable and shape does not match."}
    matched = 0
    frame_count = min(frames, xy_raw.shape[0])
    for target_idx, name in enumerate(joint_names):
        source_idx = find_joint(pose2d_names, (name,))
        if source_idx is None or source_idx >= xy_raw.shape[1]:
            continue
        xy[:frame_count, target_idx, :] = xy_raw[:frame_count, source_idx, :]
        conf[:frame_count, target_idx] = conf_raw[:frame_count, source_idx]
        matched += 1
    if matched == 0:
        return None, None, {"status": "failed", "reason": "No 2D joint names matched 3D joint names."}
    return xy, conf, {"status": "ok", "mode": "name_mapping", "matched_joints": int(matched), "matched_frames": int(frame_count)}


def reprojection_metrics(
    joints: np.ndarray,
    camera: dict[str, Any],
    xy: np.ndarray,
    confidence: np.ndarray,
    confidence_threshold: float = 0.25,
    min_depth_m: float = 0.1,
) -> dict[str, Any]:
    values = np.asarray(joints, dtype=float)
    obs = np.asarray(xy, dtype=float)
    conf = np.asarray(confidence, dtype=float)
    projection = project_points(values, _camera_intrinsics(camera))
    err = np.linalg.norm(projection - obs, axis=-1)
    valid = (
        np.isfinite(err)
        & np.isfinite(conf)
        & (conf >= float(confidence_threshold))
        & np.isfinite(values).all(axis=-1)
        & (values[..., 2] > float(min_depth_m))
    )
    valid_err = err[valid]
    displacement_to_ray = np.linalg.norm(depth_preserving_reprojection_targets(values, obs, camera) - values, axis=-1)
    valid_disp = displacement_to_ray[valid]
    return {
        "valid_observations": int(np.count_nonzero(valid)),
        "valid_observation_ratio": float(np.count_nonzero(valid) / valid.size) if valid.size else 0.0,
        "mean_reprojection_error_px": _safe_float(np.nanmean(valid_err)) if valid_err.size else None,
        "median_reprojection_error_px": _safe_float(np.nanmedian(valid_err)) if valid_err.size else None,
        "p90_reprojection_error_px": _safe_float(np.nanpercentile(valid_err, 90)) if valid_err.size else None,
        "mean_depth_preserving_displacement_m": _safe_float(np.nanmean(valid_disp)) if valid_disp.size else None,
        "median_depth_preserving_displacement_m": _safe_float(np.nanmedian(valid_disp)) if valid_disp.size else None,
    }


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


def _camera_intrinsics(camera: dict[str, Any]) -> dict[str, Any]:
    if all(key in camera for key in ("fx", "fy", "cx", "cy")):
        return camera
    nested = camera.get("intrinsics") if isinstance(camera.get("intrinsics"), dict) else {}
    if all(key in nested for key in ("fx", "fy", "cx", "cy")):
        return nested
    missing = [key for key in ("fx", "fy", "cx", "cy") if key not in camera and key not in nested]
    raise KeyError(f"Camera intrinsics missing required fields: {', '.join(missing)}")
