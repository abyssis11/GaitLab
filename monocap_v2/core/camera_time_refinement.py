from __future__ import annotations

import copy
from typing import Any

import numpy as np

from monocap_v2.core.geometry import project_points
from monocap_v2.core.reprojection_consistency import align_pose2d_to_joints


def apply_camera_time_refinement_to_pose(
    pose: dict[str, Any],
    camera: dict[str, Any],
    pose2d: dict[str, Any] | None,
    cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Estimate a small non-GT camera/time correction from 2D reprojection.

    The selected time offset is stored as metadata; the pose sequence is not
    resampled here. The camera delta, when enabled, is applied directly to the
    joints because it is an explicit backend-to-camera correction.
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
        report["reason"] = f"Camera/time refinement only applies to joints artifacts, got {representation!r}."
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

    fps = float(pose.get("fps") or pose2d.get("fps") or 30.0)
    pose2d_fps = float(pose2d.get("fps") or fps)
    confidence_threshold = float(cfg.get("confidence_threshold", 0.25))
    min_depth_m = float(cfg.get("min_depth_m", 0.1))
    time_cfg = cfg.get("time_search") if isinstance(cfg.get("time_search"), dict) else {}
    camera_cfg = cfg.get("camera_delta") if isinstance(cfg.get("camera_delta"), dict) else {}
    time_enabled = bool(time_cfg.get("enabled", cfg.get("time_search_enabled", True)))
    camera_enabled = bool(camera_cfg.get("enabled", cfg.get("optimize_camera", True)))
    if not time_enabled and not camera_enabled:
        report["reason"] = "Both time search and camera-delta optimization are disabled."
        return out_pose, report

    offsets = _offset_candidates(time_cfg, cfg) if time_enabled else [float(cfg.get("time_offset_s", 0.0))]
    before_zero = reprojection_time_metrics(joints, camera, xy, conf, fps, pose2d_fps, 0.0, confidence_threshold, min_depth_m)
    candidate_reports = [
        reprojection_time_metrics(joints, camera, xy, conf, fps, pose2d_fps, offset, confidence_threshold, min_depth_m)
        for offset in offsets
    ]
    valid_candidates = [item for item in candidate_reports if item.get("mean_reprojection_error_px") is not None]
    if not valid_candidates:
        report.update(
            {
                "reason": "No valid 2D/3D observations were available for time/camera refinement.",
                "pose2d_alignment": align_report,
                "metrics_before": before_zero,
                "time_candidates": candidate_reports,
            }
        )
        return out_pose, report
    selected = min(valid_candidates, key=lambda item: float(item["mean_reprojection_error_px"]))
    selected_offset = float(selected["time_offset_s"])
    corrected = joints.copy()
    camera_delta_report: dict[str, Any] = {"status": "skipped", "enabled": camera_enabled}
    if camera_enabled:
        corrected, camera_delta_report = optimize_camera_delta(
            joints,
            camera,
            xy,
            conf,
            fps,
            pose2d_fps,
            selected_offset,
            camera_cfg,
            confidence_threshold=confidence_threshold,
            min_depth_m=min_depth_m,
        )

    after = reprojection_time_metrics(corrected, camera, xy, conf, fps, pose2d_fps, selected_offset, confidence_threshold, min_depth_m)
    warnings = _refinement_warnings(selected_offset, offsets, time_enabled, camera_delta_report)
    out_pose["joints_3d"] = corrected.astype(np.float32)
    refinement_report = {
        "status": "ok",
        "method": "camera_time_reprojection",
        "pose2d_alignment": align_report,
        "time_search_enabled": time_enabled,
        "camera_delta_enabled": camera_enabled,
        "selected_time_offset_s": selected_offset,
        "selected_time_offset_ms": selected_offset * 1000.0,
        "time_offset_applied_to_sequence": False,
        "time_offset_for_evaluation_s": selected_offset,
        "confidence_threshold": confidence_threshold,
        "min_depth_m": min_depth_m,
        "metrics_before": before_zero,
        "metrics_after_time_only": selected,
        "metrics_after": after,
        "time_candidates": candidate_reports,
        "camera_delta": camera_delta_report,
        "warnings": warnings,
        "mocap_used_in_objective": False,
    }
    out_pose["camera_time_refinement"] = refinement_report
    return out_pose, refinement_report


def optimize_camera_delta(
    joints: np.ndarray,
    camera: dict[str, Any],
    xy: np.ndarray,
    confidence: np.ndarray,
    fps: float,
    pose2d_fps: float,
    time_offset_s: float,
    cfg: dict[str, Any] | None = None,
    confidence_threshold: float = 0.25,
    min_depth_m: float = 0.1,
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.optimize import least_squares

    cfg = cfg or {}
    max_rotation_rad = np.deg2rad(float(cfg.get("max_rotation_deg", 8.0)))
    max_translation_m = float(cfg.get("max_translation_m", 0.25))
    max_nfev = int(cfg.get("max_nfev", 40))
    reproj_weight = float(cfg.get("reprojection_weight", 1.0))
    rotation_prior_weight = float(cfg.get("rotation_prior_weight", 0.2))
    translation_prior_weight = float(cfg.get("translation_prior_weight", 0.5))
    depth_penalty_weight = float(cfg.get("depth_penalty_weight", 2.0))
    robust_loss = str(cfg.get("robust_loss", "soft_l1"))
    f_scale = float(cfg.get("f_scale", 0.05))
    diag = _image_diag(camera)
    base_projection = _project_resampled(joints, camera, fps, xy.shape[0], pose2d_fps, time_offset_s)
    base_valid = (
        np.isfinite(xy).all(axis=-1)
        & np.isfinite(confidence)
        & (confidence >= confidence_threshold)
        & np.isfinite(base_projection).all(axis=-1)
    )
    if np.count_nonzero(base_valid) < 3:
        return joints.copy(), {
            "status": "skipped",
            "reason": "Fewer than three valid 2D/3D observations are available for camera-delta optimization.",
            "enabled": True,
        }

    def residual(params: np.ndarray) -> np.ndarray:
        rotvec = params[:3]
        trans = params[3:6]
        transformed = apply_camera_delta(joints, rotvec, trans)
        projected = _project_resampled(transformed, camera, fps, xy.shape[0], pose2d_fps, time_offset_s)
        reproj = (projected - xy) / diag
        reproj_res = np.sqrt(reproj_weight) * reproj[base_valid].reshape(-1)
        reproj_res = np.where(np.isfinite(reproj_res), reproj_res, 10.0)
        depth_violation = np.maximum(0.0, min_depth_m - transformed[..., 2]).reshape(-1)
        prior = np.concatenate(
            [
                np.sqrt(rotation_prior_weight) * rotvec,
                np.sqrt(translation_prior_weight) * trans,
                np.sqrt(depth_penalty_weight) * depth_violation,
            ]
        )
        return np.concatenate([reproj_res, prior])

    x0 = np.zeros(6, dtype=float)
    lower = np.array([-max_rotation_rad, -max_rotation_rad, -max_rotation_rad, -max_translation_m, -max_translation_m, -max_translation_m])
    upper = -lower
    r0 = residual(x0)
    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        loss=robust_loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        x_scale="jac",
        verbose=0,
    )
    corrected = apply_camera_delta(joints, result.x[:3], result.x[3:6])
    r1 = residual(result.x)
    initial_cost = float(0.5 * np.sum(r0**2))
    final_cost = float(0.5 * np.sum(r1**2))
    rot_deg = np.rad2deg(result.x[:3])
    trans = result.x[3:6]
    return corrected, {
        "status": "ok" if final_cost <= initial_cost else "warning",
        "enabled": True,
        "success": bool(result.success and final_cost <= initial_cost),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "cost_reduction": initial_cost - final_cost,
        "rotation_vector_rad": [_safe_float(v) for v in result.x[:3]],
        "rotation_vector_deg": [_safe_float(v) for v in rot_deg],
        "rotation_magnitude_deg": _safe_float(np.linalg.norm(rot_deg)),
        "translation_m": [_safe_float(v) for v in trans],
        "translation_magnitude_m": _safe_float(np.linalg.norm(trans)),
        "bounds": {"max_rotation_deg": np.rad2deg(max_rotation_rad), "max_translation_m": max_translation_m},
        "weights": {
            "reprojection": reproj_weight,
            "rotation_prior": rotation_prior_weight,
            "translation_prior": translation_prior_weight,
            "depth_penalty": depth_penalty_weight,
        },
    }


def _refinement_warnings(
    selected_offset_s: float,
    offsets: list[float],
    time_enabled: bool,
    camera_delta_report: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if time_enabled and offsets:
        min_offset = min(offsets)
        max_offset = max(offsets)
        if abs(float(selected_offset_s) - min_offset) < 1e-9 or abs(float(selected_offset_s) - max_offset) < 1e-9:
            warnings.append(
                "Selected time offset is on the search boundary; treat this as diagnostic and expand or disable time search before promotion."
            )
    bounds = camera_delta_report.get("bounds") or {}
    rot = camera_delta_report.get("rotation_vector_deg") or []
    max_rot = bounds.get("max_rotation_deg")
    if max_rot is not None and rot:
        if max(abs(float(v)) for v in rot if v is not None) >= 0.99 * float(max_rot):
            warnings.append("Camera rotation delta is at its component bound; correction may be absorbing a convention error.")
    trans = camera_delta_report.get("translation_m") or []
    max_trans = bounds.get("max_translation_m")
    if max_trans is not None and trans:
        if max(abs(float(v)) for v in trans if v is not None) >= 0.99 * float(max_trans):
            warnings.append("Camera translation delta is at its component bound; correction may be absorbing a convention or scale error.")
    return warnings


def apply_camera_delta(joints: np.ndarray, rotvec: np.ndarray, translation: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    values = np.asarray(joints, dtype=float)
    rot = Rotation.from_rotvec(np.asarray(rotvec, dtype=float)).as_matrix()
    trans = np.asarray(translation, dtype=float)
    flat = values.reshape(-1, 3)
    out = flat @ rot.T + trans
    return out.reshape(values.shape)


def reprojection_time_metrics(
    joints: np.ndarray,
    camera: dict[str, Any],
    xy: np.ndarray,
    confidence: np.ndarray,
    fps: float,
    pose2d_fps: float,
    time_offset_s: float,
    confidence_threshold: float = 0.25,
    min_depth_m: float = 0.1,
) -> dict[str, Any]:
    projected = _project_resampled(joints, camera, fps, xy.shape[0], pose2d_fps, time_offset_s)
    values = np.asarray(joints, dtype=float)
    obs = np.asarray(xy, dtype=float)
    conf = np.asarray(confidence, dtype=float)
    err = np.linalg.norm(projected - obs, axis=-1)
    valid = (
        np.isfinite(err)
        & np.isfinite(conf)
        & (conf >= confidence_threshold)
        & np.isfinite(projected).all(axis=-1)
        & np.isfinite(obs).all(axis=-1)
    )
    finite_depth = np.isfinite(values[..., 2]) & (values[..., 2] > min_depth_m)
    valid_err = err[valid]
    return {
        "time_offset_s": float(time_offset_s),
        "time_offset_ms": float(time_offset_s) * 1000.0,
        "valid_observations": int(np.count_nonzero(valid)),
        "valid_observation_ratio": float(np.count_nonzero(valid) / valid.size) if valid.size else 0.0,
        "positive_depth_ratio": float(np.count_nonzero(finite_depth) / finite_depth.size) if finite_depth.size else 0.0,
        "mean_reprojection_error_px": _safe_float(np.nanmean(valid_err)) if valid_err.size else None,
        "median_reprojection_error_px": _safe_float(np.nanmedian(valid_err)) if valid_err.size else None,
        "p90_reprojection_error_px": _safe_float(np.nanpercentile(valid_err, 90)) if valid_err.size else None,
        "motion_correlation": _safe_float(_motion_correlation(projected, obs, conf, confidence_threshold)),
    }


def _project_resampled(
    joints: np.ndarray,
    camera: dict[str, Any],
    fps: float,
    obs_frames: int,
    obs_fps: float,
    time_offset_s: float,
) -> np.ndarray:
    values = np.asarray(joints, dtype=float)
    pred_t = np.arange(values.shape[0], dtype=float) / float(fps) - float(time_offset_s)
    obs_t = np.arange(int(obs_frames), dtype=float) / float(obs_fps)
    projected = project_points(values, _camera_intrinsics(camera))
    out = np.full((obs_frames, values.shape[1], 2), np.nan, dtype=float)
    if values.shape[0] == 0 or obs_frames == 0:
        return out
    for joint_idx in range(values.shape[1]):
        for coord in range(2):
            series = projected[:, joint_idx, coord]
            finite = np.isfinite(series) & np.isfinite(pred_t)
            if np.count_nonzero(finite) < 2:
                continue
            inside = (obs_t >= pred_t[finite][0]) & (obs_t <= pred_t[finite][-1])
            out[inside, joint_idx, coord] = np.interp(obs_t[inside], pred_t[finite], series[finite])
    return out


def _offset_candidates(time_cfg: dict[str, Any], cfg: dict[str, Any]) -> list[float]:
    explicit = time_cfg.get("offsets_s") or cfg.get("time_offsets_s")
    if explicit is not None:
        offsets = [float(v) for v in explicit]
    else:
        min_s = float(time_cfg.get("min_offset_s", cfg.get("time_offset_min_s", -0.25)))
        max_s = float(time_cfg.get("max_offset_s", cfg.get("time_offset_max_s", 0.25)))
        step_s = float(time_cfg.get("step_s", cfg.get("time_offset_step_s", 0.02)))
        if step_s <= 0:
            step_s = 0.02
        count = int(np.floor((max_s - min_s) / step_s + 0.5)) + 1
        offsets = [min_s + idx * step_s for idx in range(max(count, 1))]
    offsets.append(0.0)
    return sorted({round(float(v), 6) for v in offsets})


def _motion_correlation(projected: np.ndarray, observed: np.ndarray, confidence: np.ndarray, threshold: float) -> float | None:
    if projected.shape[0] < 3:
        return None
    pred_v = np.diff(projected, axis=0)
    obs_v = np.diff(observed, axis=0)
    conf = np.minimum(confidence[:-1], confidence[1:])
    valid = (
        np.isfinite(pred_v).all(axis=-1)
        & np.isfinite(obs_v).all(axis=-1)
        & np.isfinite(conf)
        & (conf >= threshold)
    )
    if np.count_nonzero(valid) < 3:
        return None
    a = pred_v[valid].reshape(-1)
    b = obs_v[valid].reshape(-1)
    if np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _camera_intrinsics(camera: dict[str, Any]) -> dict[str, Any]:
    if all(key in camera for key in ("fx", "fy", "cx", "cy")):
        return camera
    nested = camera.get("intrinsics") if isinstance(camera.get("intrinsics"), dict) else {}
    if all(key in nested for key in ("fx", "fy", "cx", "cy")):
        return nested
    missing = [key for key in ("fx", "fy", "cx", "cy") if key not in camera and key not in nested]
    raise KeyError(f"Camera intrinsics missing required fields: {', '.join(missing)}")


def _image_diag(camera: dict[str, Any]) -> float:
    intr = _camera_intrinsics(camera)
    width = float(camera.get("width") or intr.get("width") or max(float(intr["cx"]) * 2.0, 1.0))
    height = float(camera.get("height") or intr.get("height") or max(float(intr["cy"]) * 2.0, 1.0))
    return float(np.hypot(width, height)) or 1.0


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None
