from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from monocap_v2.core.geometry import edge_indices, find_joint


@dataclass(frozen=True)
class EdgeLength:
    parent: int
    child: int
    target_m: float


def apply_temporal_smoothing_to_pose(
    pose: dict[str, Any],
    cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Smooth joints over time while preserving the backend skeleton lengths.

    This helper is diagnostic/refinement-only. It never reads mocap or OpenSim
    ground truth and it skips SMPL/hybrid artifacts so SMPL params and vertices
    cannot become inconsistent with edited joints.
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
        report["reason"] = f"Temporal smoothing only applies to joints artifacts, got {representation!r}."
        return out_pose, report

    joints = np.asarray(pose.get("joints_3d"), dtype=float)
    names = [str(name) for name in pose.get("joint_names", [])]
    if joints.ndim != 3 or joints.shape[-1] != 3 or joints.shape[0] < 3 or not names:
        report["reason"] = "Pose artifact has no valid [T, J, 3] sequence with at least 3 frames."
        return out_pose, report

    method = str(cfg.get("method") or "moving_average")
    window = _odd_window(int(cfg.get("window_frames", 7)), joints.shape[0])
    passes = max(1, int(cfg.get("passes", 1)))
    preserve_bones = bool(cfg.get("preserve_bones", True))
    smooth_root = bool(cfg.get("smooth_root", True))
    bone_iterations = max(0, int(cfg.get("bone_projection_iterations", 2)))
    bone_blend = float(np.clip(float(cfg.get("bone_preservation_blend", 1.0)), 0.0, 1.0))
    max_displacement_m = _finite_positive_or_none(cfg.get("max_joint_displacement_m"))

    filled, fill_report = fill_nan_linear(joints)
    smoothed = filled.copy()
    for _ in range(passes):
        smoothed = smooth_joint_sequence(
            smoothed,
            method=method,
            window_frames=window,
            polyorder=int(cfg.get("savgol_polyorder", 2)),
        )

    root_idx = find_joint(names, ("pelvis", "pelv", "root"))
    if not smooth_root and root_idx is not None:
        smoothed[:, root_idx, :] = filled[:, root_idx, :]

    targets = estimate_edge_lengths(filled, names)
    projected = smoothed
    if preserve_bones and targets and bone_iterations > 0:
        projected = project_bones_to_lengths(
            projected,
            filled,
            names,
            targets=targets,
            iterations=bone_iterations,
            blend=bone_blend,
        )

    if max_displacement_m is not None:
        projected = limit_joint_displacement(projected, filled, max_displacement_m)

    # Keep originally missing values missing; smoothing should not invent data.
    projected[~np.isfinite(joints)] = np.nan

    before_metrics = temporal_smoothing_metrics(filled, names, fps=float(pose.get("fps") or 30.0), target_lengths=targets)
    after_metrics = temporal_smoothing_metrics(projected, names, fps=float(pose.get("fps") or 30.0), target_lengths=targets)

    out_pose["joints_3d"] = projected.astype(np.float32)
    smoothing_report = {
        "status": "ok",
        "method": method,
        "window_frames": int(window),
        "passes": int(passes),
        "preserve_bones": preserve_bones,
        "smooth_root": smooth_root,
        "bone_projection_iterations": int(bone_iterations),
        "bone_preservation_blend": bone_blend,
        "max_joint_displacement_m": max_displacement_m,
        "nan_fill": fill_report,
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
        "mocap_used_in_objective": False,
    }
    out_pose["temporal_smoothing"] = smoothing_report
    return out_pose, smoothing_report


def smooth_joint_sequence(
    joints: np.ndarray,
    method: str = "moving_average",
    window_frames: int = 7,
    polyorder: int = 2,
) -> np.ndarray:
    values = np.asarray(joints, dtype=float)
    window = _odd_window(int(window_frames), values.shape[0])
    if window < 3:
        return values.copy()
    method = method.strip().lower()
    if method in {"moving_average", "mean", "ma"}:
        return _moving_average(values, window)
    if method in {"savgol", "savitzky_golay", "savitzky-golay"}:
        try:
            from scipy.signal import savgol_filter
        except Exception:
            return _moving_average(values, window)
        order = min(max(1, int(polyorder)), window - 1)
        return savgol_filter(values, window_length=window, polyorder=order, axis=0, mode="interp")
    raise ValueError(f"Unsupported temporal smoothing method: {method}")


def fill_nan_linear(joints: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(joints, dtype=float)
    out = values.copy()
    total = int(out.size)
    missing = int(np.count_nonzero(~np.isfinite(out)))
    if missing == 0:
        return out, {"missing_values": 0, "filled_values": 0}
    frame_idx = np.arange(out.shape[0], dtype=float)
    filled = 0
    for j in range(out.shape[1]):
        for c in range(3):
            series = out[:, j, c]
            valid = np.isfinite(series)
            if not np.any(valid):
                continue
            if np.count_nonzero(valid) == 1:
                series[~valid] = series[valid][0]
            else:
                series[~valid] = np.interp(frame_idx[~valid], frame_idx[valid], series[valid])
            filled += int(np.count_nonzero(~valid))
    return out, {"missing_values": missing, "filled_values": filled, "total_values": total}


def estimate_edge_lengths(joints: np.ndarray, joint_names: list[str]) -> list[EdgeLength]:
    values = np.asarray(joints, dtype=float)
    targets: list[EdgeLength] = []
    for parent, child in edge_indices(joint_names):
        lengths = np.linalg.norm(values[:, child, :] - values[:, parent, :], axis=1)
        finite = np.isfinite(lengths) & (lengths > 1e-9)
        if not np.any(finite):
            continue
        targets.append(EdgeLength(parent=parent, child=child, target_m=float(np.nanmedian(lengths[finite]))))
    return targets


def project_bones_to_lengths(
    joints: np.ndarray,
    reference: np.ndarray,
    joint_names: list[str],
    targets: list[EdgeLength] | None = None,
    iterations: int = 2,
    blend: float = 1.0,
) -> np.ndarray:
    out = np.asarray(joints, dtype=float).copy()
    ref = np.asarray(reference, dtype=float)
    targets = targets if targets is not None else estimate_edge_lengths(ref, joint_names)
    blend = float(np.clip(blend, 0.0, 1.0))
    if not targets or iterations <= 0 or blend <= 0.0:
        return out
    for _ in range(iterations):
        for t in range(out.shape[0]):
            for edge in targets:
                parent = out[t, edge.parent]
                child = out[t, edge.child]
                if not np.isfinite(parent).all():
                    continue
                direction = child - parent
                norm = float(np.linalg.norm(direction))
                if not np.isfinite(norm) or norm < 1e-9:
                    direction = ref[t, edge.child] - ref[t, edge.parent]
                    norm = float(np.linalg.norm(direction))
                if not np.isfinite(norm) or norm < 1e-9:
                    continue
                target_child = parent + edge.target_m * direction / norm
                if np.isfinite(child).all():
                    out[t, edge.child] = child + blend * (target_child - child)
                else:
                    out[t, edge.child] = target_child
    return out


def limit_joint_displacement(joints: np.ndarray, reference: np.ndarray, max_displacement_m: float) -> np.ndarray:
    out = np.asarray(joints, dtype=float).copy()
    ref = np.asarray(reference, dtype=float)
    delta = out - ref
    dist = np.linalg.norm(delta, axis=-1)
    mask = np.isfinite(dist) & (dist > float(max_displacement_m)) & (dist > 1e-12)
    out[mask] = ref[mask] + delta[mask] * (float(max_displacement_m) / dist[mask])[:, None]
    return out


def temporal_smoothing_metrics(
    joints: np.ndarray,
    joint_names: list[str],
    fps: float,
    target_lengths: list[EdgeLength] | None = None,
) -> dict[str, Any]:
    values = np.asarray(joints, dtype=float)
    metrics: dict[str, Any] = {"finite_ratio": float(np.isfinite(values).mean()) if values.size else 0.0}
    if values.shape[0] >= 2:
        velocity = np.diff(values, axis=0) * float(fps)
        metrics["mean_velocity_mps"] = _safe_float(np.nanmean(np.linalg.norm(velocity, axis=-1)))
    if values.shape[0] >= 3:
        second = values[:-2] - 2.0 * values[1:-1] + values[2:]
        second_norm = np.linalg.norm(second, axis=-1)
        metrics["mean_second_diff_m"] = _safe_float(np.nanmean(second_norm))
        metrics["rms_second_diff_m"] = _safe_float(np.sqrt(np.nanmean(second_norm**2)))
        metrics["mean_acceleration_mps2"] = _safe_float(np.nanmean(second_norm) * float(fps) ** 2)
    if target_lengths is None:
        target_lengths = estimate_edge_lengths(values, joint_names)
    bone_errors = []
    for edge in target_lengths:
        lengths = np.linalg.norm(values[:, edge.child, :] - values[:, edge.parent, :], axis=1)
        finite = np.isfinite(lengths)
        if np.any(finite):
            bone_errors.extend(np.abs(lengths[finite] - edge.target_m).tolist())
    if bone_errors:
        arr = np.asarray(bone_errors, dtype=float)
        metrics["median_bone_length_error_mm"] = _safe_float(np.nanmedian(arr) * 1000.0)
        metrics["max_bone_length_error_mm"] = _safe_float(np.nanmax(arr) * 1000.0)
    return metrics


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    kernel = np.ones((window,), dtype=float) / float(window)
    flat = padded.reshape(padded.shape[0], -1)
    smoothed = np.empty((values.shape[0], flat.shape[1]), dtype=float)
    for idx in range(flat.shape[1]):
        smoothed[:, idx] = np.convolve(flat[:, idx], kernel, mode="valid")
    return smoothed.reshape(values.shape)


def _odd_window(window: int, n_frames: int) -> int:
    if n_frames < 3:
        return 1
    window = max(3, int(window))
    window = min(window, n_frames if n_frames % 2 == 1 else n_frames - 1)
    if window % 2 == 0:
        window -= 1
    return max(3, window)


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
