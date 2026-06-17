from __future__ import annotations

import copy
from typing import Any

import numpy as np

from monocap_v2.core.geometry import find_joint


CONTACT_JOINTS = {
    "left_heel": ("left_heel", "lheel"),
    "left_toe": ("left_toe", "left_big_toe", "ltoe"),
    "right_heel": ("right_heel", "rheel"),
    "right_toe": ("right_toe", "right_big_toe", "rtoe"),
}

HORIZONTAL_COORDS = (0, 2)


def apply_contact_foot_locking_to_pose(
    pose: dict[str, Any],
    contacts: dict[str, Any] | None,
    cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Lock contact feet using only monocular contacts and predicted joints.

    `root_translation` applies a per-frame translation to every joint, so local
    pose and bone lengths are preserved. `endpoint` locks foot joints directly
    and is useful as a diagnostic, but can alter lower-limb bone lengths.
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
        report["reason"] = f"Contact foot locking only applies to joints artifacts, got {representation!r}."
        return out_pose, report
    if not contacts:
        report["reason"] = "No contact probabilities are available."
        return out_pose, report

    joints = np.asarray(pose.get("joints_3d"), dtype=float)
    names = [str(name) for name in pose.get("joint_names", [])]
    if joints.ndim != 3 or joints.shape[-1] != 3 or joints.shape[0] < 2 or not names:
        report["reason"] = "Pose artifact has no valid [T, J, 3] joints."
        return out_pose, report

    mode = str(cfg.get("mode") or "root_translation")
    contact_threshold = float(cfg.get("contact_threshold", 0.85))
    min_segment_frames = int(cfg.get("min_segment_frames", 3))
    lock_vertical = bool(cfg.get("lock_vertical", False))
    blend = float(np.clip(float(cfg.get("blend", 1.0)), 0.0, 1.0))
    max_correction_m = _finite_positive_or_none(cfg.get("max_correction_m"))
    smooth_window = int(cfg.get("smooth_correction_window_frames", 1))
    foot_keys = _selected_foot_keys(str(cfg.get("feet") or "toes"))
    coords = (0, 1, 2) if lock_vertical else HORIZONTAL_COORDS

    specs = _build_lock_specs(joints, names, contacts, foot_keys, contact_threshold, min_segment_frames, coords)
    if not specs:
        report.update(
            {
                "reason": "No usable contact segments and foot joints were found.",
                "contact_threshold": contact_threshold,
                "min_segment_frames": min_segment_frames,
                "feet": foot_keys,
            }
        )
        return out_pose, report

    fps = float(pose.get("fps") or 30.0)
    before_metrics = foot_locking_metrics(joints, names, contacts, fps, contact_threshold, min_segment_frames, foot_keys)
    if mode == "root_translation":
        locked, correction = _apply_root_translation_lock(joints, specs, coords, blend, max_correction_m, smooth_window)
    elif mode == "endpoint":
        locked, correction = _apply_endpoint_lock(joints, specs, coords, blend, max_correction_m)
    else:
        report["reason"] = f"Unsupported contact foot-locking mode: {mode}"
        return out_pose, report
    locked[~np.isfinite(joints)] = np.nan
    after_metrics = foot_locking_metrics(locked, names, contacts, fps, contact_threshold, min_segment_frames, foot_keys)

    out_pose["joints_3d"] = locked.astype(np.float32)
    lock_report = {
        "status": "ok",
        "method": "contact_foot_locking",
        "mode": mode,
        "feet": foot_keys,
        "contact_threshold": contact_threshold,
        "min_segment_frames": min_segment_frames,
        "lock_vertical": lock_vertical,
        "blend": blend,
        "max_correction_m": max_correction_m,
        "smooth_correction_window_frames": smooth_window,
        "segments": _public_specs(specs),
        "locked_segment_count": len(specs),
        "used_contact_keys": sorted({spec["contact_key"] for spec in specs}),
        "correction_summary": _correction_summary(correction),
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
        "mocap_used_in_objective": False,
    }
    out_pose["contact_foot_locking"] = lock_report
    return out_pose, lock_report


def foot_locking_metrics(
    joints: np.ndarray,
    joint_names: list[str],
    contacts: dict[str, Any] | None,
    fps: float = 30.0,
    contact_threshold: float = 0.85,
    min_segment_frames: int = 3,
    foot_keys: list[str] | None = None,
) -> dict[str, Any]:
    values = np.asarray(joints, dtype=float)
    metrics: dict[str, Any] = {"finite_ratio": float(np.isfinite(values).mean()) if values.size else 0.0}
    if foot_keys is None:
        foot_keys = _selected_foot_keys("toes")
    speeds = []
    horizontal_speeds = []
    segment_stds = []
    for spec in _build_lock_specs(values, joint_names, contacts, foot_keys, contact_threshold, min_segment_frames, HORIZONTAL_COORDS):
        idx = int(spec["joint_index"])
        segment = np.asarray(spec["frames"], dtype=int)
        if segment.size < 2:
            continue
        foot = values[segment, idx, :]
        speed = np.linalg.norm(np.diff(foot, axis=0) * fps, axis=1)
        h_speed = np.linalg.norm(np.diff(foot[:, HORIZONTAL_COORDS], axis=0) * fps, axis=1)
        speeds.extend(speed[np.isfinite(speed)].tolist())
        horizontal_speeds.extend(h_speed[np.isfinite(h_speed)].tolist())
        h_std = np.nanmean(np.nanstd(foot[:, HORIZONTAL_COORDS], axis=0))
        if np.isfinite(h_std):
            segment_stds.append(float(h_std))
    metrics["mean_contact_speed_mps"] = _safe_float(np.nanmean(speeds)) if speeds else None
    metrics["mean_contact_horizontal_speed_mps"] = _safe_float(np.nanmean(horizontal_speeds)) if horizontal_speeds else None
    metrics["mean_contact_horizontal_position_std_m"] = _safe_float(np.nanmean(segment_stds)) if segment_stds else None
    metrics["contact_segment_count"] = len(segment_stds)
    return metrics


def _build_lock_specs(
    joints: np.ndarray,
    joint_names: list[str],
    contacts: dict[str, Any] | None,
    foot_keys: list[str],
    threshold: float,
    min_segment_frames: int,
    coords: tuple[int, ...],
) -> list[dict[str, Any]]:
    if not contacts:
        return []
    specs: list[dict[str, Any]] = []
    for key in foot_keys:
        idx = find_joint(joint_names, CONTACT_JOINTS[key])
        if idx is None:
            continue
        prob = _contact_prob(contacts, key, joints.shape[0])
        for frames in _contiguous_segments(prob >= threshold, min_segment_frames):
            foot = joints[frames, idx, :]
            finite = np.isfinite(foot[:, coords]).all(axis=1)
            if np.count_nonzero(finite) < min_segment_frames:
                continue
            anchor = np.full((3,), np.nan, dtype=float)
            anchor[list(coords)] = np.nanmedian(foot[finite][:, coords], axis=0)
            specs.append(
                {
                    "contact_key": key,
                    "joint_index": int(idx),
                    "joint_name": str(joint_names[idx]),
                    "frames": frames,
                    "anchor": anchor,
                    "mean_probability": float(np.nanmean(prob[frames])),
                }
            )
    return specs


def _apply_root_translation_lock(
    joints: np.ndarray,
    specs: list[dict[str, Any]],
    coords: tuple[int, ...],
    blend: float,
    max_correction_m: float | None,
    smooth_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(joints, dtype=float).copy()
    correction = np.zeros((out.shape[0], 3), dtype=float)
    weights = np.zeros((out.shape[0],), dtype=float)
    for spec in specs:
        idx = int(spec["joint_index"])
        anchor = np.asarray(spec["anchor"], dtype=float)
        for t in np.asarray(spec["frames"], dtype=int):
            delta = np.zeros((3,), dtype=float)
            current = out[t, idx, :]
            if not np.isfinite(current[list(coords)]).all():
                continue
            delta[list(coords)] = anchor[list(coords)] - current[list(coords)]
            correction[t] += delta * float(spec.get("mean_probability", 1.0))
            weights[t] += float(spec.get("mean_probability", 1.0))
    valid = weights > 0
    correction[valid] /= weights[valid, None]
    correction = _limit_correction(correction, max_correction_m)
    if smooth_window >= 3:
        correction = _smooth_correction(correction, smooth_window)
        correction = _limit_correction(correction, max_correction_m)
    out += blend * correction[:, None, :]
    return out, blend * correction


def _apply_endpoint_lock(
    joints: np.ndarray,
    specs: list[dict[str, Any]],
    coords: tuple[int, ...],
    blend: float,
    max_correction_m: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(joints, dtype=float).copy()
    correction = np.zeros((out.shape[0], 3), dtype=float)
    counts = np.zeros((out.shape[0],), dtype=float)
    for spec in specs:
        idx = int(spec["joint_index"])
        anchor = np.asarray(spec["anchor"], dtype=float)
        for t in np.asarray(spec["frames"], dtype=int):
            delta = np.zeros((3,), dtype=float)
            current = out[t, idx, :]
            if not np.isfinite(current[list(coords)]).all():
                continue
            delta[list(coords)] = anchor[list(coords)] - current[list(coords)]
            delta = _limit_correction(delta[None, :], max_correction_m)[0]
            out[t, idx, :] += blend * delta
            correction[t] += blend * delta
            counts[t] += 1.0
    valid = counts > 0
    correction[valid] /= counts[valid, None]
    return out, correction


def _selected_foot_keys(value: str) -> list[str]:
    key = value.strip().lower()
    if key in {"toe", "toes"}:
        return ["left_toe", "right_toe"]
    if key in {"heel", "heels"}:
        return ["left_heel", "right_heel"]
    if key in {"toe_heel", "heel_toe", "toes_and_heels", "heels_and_toes", "all"}:
        return ["left_heel", "left_toe", "right_heel", "right_toe"]
    raise ValueError(f"Unsupported foot selection: {value}")


def _contact_prob(contacts: dict[str, Any] | None, key: str, n_frames: int) -> np.ndarray:
    if not contacts or key not in contacts:
        return np.zeros((n_frames,), dtype=float)
    arr = np.asarray(contacts[key], dtype=float).reshape(-1)
    out = np.zeros((n_frames,), dtype=float)
    out[: min(n_frames, arr.size)] = arr[: min(n_frames, arr.size)]
    return out


def _contiguous_segments(mask: np.ndarray, min_len: int) -> list[np.ndarray]:
    segments = []
    start = None
    values = np.asarray(mask, dtype=bool).tolist() + [False]
    for idx, value in enumerate(values):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            if idx - start >= int(min_len):
                segments.append(np.arange(start, idx, dtype=int))
            start = None
    return segments


def _limit_correction(correction: np.ndarray, max_correction_m: float | None) -> np.ndarray:
    out = np.asarray(correction, dtype=float).copy()
    if max_correction_m is None:
        return out
    norm = np.linalg.norm(out, axis=-1)
    mask = np.isfinite(norm) & (norm > max_correction_m) & (norm > 1e-12)
    out[mask] *= (float(max_correction_m) / norm[mask])[..., None]
    return out


def _smooth_correction(correction: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window < 3:
        return correction
    if window % 2 == 0:
        window -= 1
    pad = window // 2
    padded = np.pad(correction, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones((window,), dtype=float) / float(window)
    out = np.empty_like(correction)
    for c in range(3):
        out[:, c] = np.convolve(padded[:, c], kernel, mode="valid")
    return out


def _public_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for spec in specs:
        frames = np.asarray(spec["frames"], dtype=int)
        out.append(
            {
                "contact_key": spec["contact_key"],
                "joint_name": spec["joint_name"],
                "frame_start": int(frames[0]),
                "frame_end": int(frames[-1]),
                "frame_count": int(frames.size),
                "anchor_m": _json_list(spec["anchor"]),
                "mean_probability": float(spec["mean_probability"]),
            }
        )
    return out


def _correction_summary(correction: np.ndarray) -> dict[str, Any]:
    norm = np.linalg.norm(np.asarray(correction, dtype=float), axis=1)
    finite = norm[np.isfinite(norm)]
    if finite.size == 0:
        return {"mean_m": None, "max_m": None, "active_frames": 0}
    return {
        "mean_m": float(np.nanmean(finite)),
        "median_m": float(np.nanmedian(finite)),
        "max_m": float(np.nanmax(finite)),
        "active_frames": int(np.count_nonzero(finite > 1e-12)),
    }


def _json_list(value: np.ndarray) -> list[float | None]:
    out = []
    for item in np.asarray(value, dtype=float).tolist():
        out.append(float(item) if np.isfinite(item) else None)
    return out


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
