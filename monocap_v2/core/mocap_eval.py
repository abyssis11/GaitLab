from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from monocap_v2.core.geometry import apply_axis_expr, apply_similarity, camera_to_eval_coords, find_joint, kabsch_align
from monocap_v2.core.logging_utils import read_yaml


PREDICTION_JOINTS = [
    ("left_hip", ("lhip", "left_hip")),
    ("right_hip", ("rhip", "right_hip")),
    ("left_knee", ("lkne", "left_knee")),
    ("right_knee", ("rkne", "right_knee")),
    ("left_ankle", ("lank", "left_ankle")),
    ("right_ankle", ("rank", "right_ankle")),
    ("left_toe", ("ltoe", "left_toe", "left_big_toe")),
    ("right_toe", ("rtoe", "right_toe", "right_big_toe")),
]

MOCAP_MARKERS = [
    ("left_hip", ("L_HJC", "L_HJC_reg", "L.ASIS", "LASI")),
    ("right_hip", ("R_HJC", "R_HJC_reg", "r.ASIS", "RASI")),
    ("left_knee", ("L_knee", "LKNE")),
    ("right_knee", ("r_knee", "RKNE")),
    ("left_ankle", ("L_ankle", "LANK")),
    ("right_ankle", ("r_ankle", "RANK")),
    ("left_toe", ("L_toe", "LTOE")),
    ("right_toe", ("r_toe", "RTOE")),
]


@dataclass
class TRCData:
    time: np.ndarray
    markers: dict[str, np.ndarray]
    marker_names: list[str]
    units: str
    data_rate: float | None


def parse_trc(path: Path | str) -> TRCData:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    frame_header_idx = None
    for idx, line in enumerate(lines[:300]):
        if re.search(r"\bFrame", line, re.I) and re.search(r"\bTime\b", line, re.I):
            frame_header_idx = idx
            break
    if frame_header_idx is None:
        raise ValueError(f"Could not find TRC Frame/Time header: {path}")

    units = "m"
    data_rate = None
    for idx in range(max(0, frame_header_idx - 10), frame_header_idx):
        cols = _split_header(lines[idx])
        if not cols or not any(c.lower() == "units" for c in cols):
            continue
        vals = _split_header(lines[idx + 1]) if idx + 1 < len(lines) else []
        lower = [c.lower() for c in cols]
        try:
            units = vals[lower.index("units")].lower()
        except Exception:
            pass
        try:
            data_rate = float(vals[lower.index("datarate")])
        except Exception:
            pass
        break

    marker_names = _parse_marker_names(lines[frame_header_idx])
    if not marker_names:
        raise ValueError(f"No marker names found in TRC header: {path}")

    time: list[float] = []
    buffers = {name: [] for name in marker_names}
    for line in lines[frame_header_idx + 2 :]:
        if not line.strip():
            continue
        parts = re.split(r"[\s,\t]+", line.strip())
        if len(parts) < 2:
            continue
        try:
            time.append(float(parts[1]))
        except ValueError:
            break
        for marker_idx, marker_name in enumerate(marker_names):
            base = 2 + 3 * marker_idx
            buffers[marker_name].append(
                (
                    _parse_float(parts, base),
                    _parse_float(parts, base + 1),
                    _parse_float(parts, base + 2),
                )
            )

    if not time:
        raise ValueError(f"No numeric rows found in TRC: {path}")
    markers = {name: np.asarray(values, dtype=float) for name, values in buffers.items()}
    out_units = units
    if "mm" in units.lower():
        markers = {name: values / 1000.0 for name, values in markers.items()}
        out_units = "m"
    return TRCData(time=np.asarray(time, dtype=float), markers=markers, marker_names=marker_names, units=out_units, data_rate=data_rate)


def evaluate_pose_against_mocap(
    pose: dict,
    mocap_trc: Path | str,
    mocap_axis: str = "-y,x,z",
) -> dict:
    report, _ = evaluate_pose_against_mocap_with_series(pose, mocap_trc, mocap_axis=mocap_axis)
    return report


def evaluate_pose_against_mocap_with_series(
    pose: dict,
    mocap_trc: Path | str,
    mocap_axis: str = "-y,x,z",
    cfg: dict | None = None,
    timeline_report: dict | None = None,
    mocap_to_video_transform: dict | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    trc = parse_trc(mocap_trc)
    pred_names, pred_all = prediction_lower_limb(pose)
    timebase = resolve_prediction_timebase(pose, cfg=cfg, timeline_report=timeline_report)
    indices = np.asarray(timebase["pose_indices"], dtype=int)
    pred_time = np.asarray(timebase["prediction_timestamps_s"], dtype=float)
    pred = pred_all[indices]
    ref, marker_mapping = mocap_lower_limb_with_mapping(trc, pred_names, mocap_axis=mocap_axis)
    ref_rs = resample_timeseries(trc.time, ref, pred_time)
    valid_time = (pred_time >= np.nanmin(trc.time)) & (pred_time <= np.nanmax(trc.time))
    pred = pred[valid_time]
    ref_rs = ref_rs[valid_time]
    time = pred_time[valid_time]
    if pred.shape[0] < 1:
        raise ValueError("No prediction timestamps overlap the mocap time range.")

    pred_centered = root_center(pred, pred_names)
    ref_centered = root_center(ref_rs, pred_names)
    normal = sequence_mpjpe(pred_centered, ref_centered, pred_names, align="none")
    rigid, rigid_transform, pred_rigid = sequence_mpjpe_with_transform(
        pred_centered,
        ref_centered,
        pred_names,
        align="rigid",
        allow_translation=False,
    )
    normalized, normalized_transform, pred_normalized = normalized_mpjpe(pred_centered, ref_centered, pred_names)
    root_similarity, root_similarity_transform, pred_root_similarity = sequence_mpjpe_with_transform(
        pred_centered,
        ref_centered,
        pred_names,
        align="similarity",
        allow_translation=False,
    )
    global_similarity, global_similarity_transform, pred_global_similarity = sequence_mpjpe_with_transform(
        pred, ref_rs, pred_names, align="similarity"
    )
    pa_similarity = per_frame_aligned_mpjpe(pred, ref_rs, pred_names, align="similarity")
    segment_lengths = summarize_segment_lengths(pred, ref_rs, pred_names)
    foot_series = foot_trajectory_series(pred, ref_rs, pred_names, time)
    warnings = []
    gap_threshold = float(((cfg or {}).get("config", {}).get("mocap_validation", {}) or {}).get("warn_normal_rigid_gap_mm", 100.0))
    if normal.get("mpjpe_mm") is not None and rigid.get("mpjpe_mm") is not None:
        gap = float(normal["mpjpe_mm"] - rigid["mpjpe_mm"])
        if gap > gap_threshold:
            warnings.append(
                f"Normal MPJPE exceeds rigid MPJPE by {gap:.3f} mm; inspect coordinate conventions before absolute interpretation."
            )
    absolute_metric = _absolute_camera_metric(
        pose,
        trc,
        pred_names,
        indices[valid_time],
        time,
        marker_mapping,
        mocap_to_video_transform,
    )

    report = {
        "status": "ok",
        "mocap_trc": str(mocap_trc),
        "joint_names": pred_names,
        "mocap_marker_mapping": marker_mapping,
        "frames": int(pred.shape[0]),
        "prediction_fps": float(pose.get("fps") or 30.0),
        "mocap_data_rate_hz": trc.data_rate,
        "resampling": "mocap_resampled_to_prediction_timestamps",
        "time_start": float(time[0]) if time.size else None,
        "time_end": float(time[-1]) if time.size else None,
        "mocap_axis": mocap_axis,
        "timebase": _public_timebase(timebase, valid_time),
        "normal_root_centered_mpjpe_m": normal["mpjpe_m"],
        "normal_root_centered_mpjpe_mm": normal["mpjpe_mm"],
        "root_centered_rigid_mpjpe_m": rigid["mpjpe_m"],
        "root_centered_rigid_mpjpe_mm": rigid["mpjpe_mm"],
        "root_centered_n_mpjpe_m": normalized["mpjpe_m"],
        "root_centered_n_mpjpe_mm": normalized["mpjpe_mm"],
        "root_centered_similarity_mpjpe_m": root_similarity["mpjpe_m"],
        "root_centered_similarity_mpjpe_mm": root_similarity["mpjpe_mm"],
        "pa_mpjpe_m": pa_similarity["mpjpe_m"],
        "pa_mpjpe_mm": pa_similarity["mpjpe_mm"],
        "global_sequence_similarity_mpjpe_m": global_similarity["mpjpe_m"],
        "global_sequence_similarity_mpjpe_mm": global_similarity["mpjpe_mm"],
        # Backward-compatible names used by the existing joints-only benchmark.
        "primary_root_centered_rigid_mpjpe_m": rigid["mpjpe_m"],
        "primary_root_centered_rigid_mpjpe_mm": rigid["mpjpe_mm"],
        "sequence_similarity_mpjpe_m": global_similarity["mpjpe_m"],
        "sequence_similarity_mpjpe_mm": global_similarity["mpjpe_mm"],
        "pa_similarity_mpjpe_m": pa_similarity["mpjpe_m"],
        "pa_similarity_mpjpe_mm": pa_similarity["mpjpe_mm"],
        "per_joint_normal_root_centered_mpjpe_m": normal["per_joint_mpjpe_m"],
        "per_joint_normal_root_centered_mpjpe_mm": normal["per_joint_mpjpe_mm"],
        "per_joint_root_centered_rigid_mpjpe_m": rigid["per_joint_mpjpe_m"],
        "per_joint_root_centered_rigid_mpjpe_mm": rigid["per_joint_mpjpe_mm"],
        "segment_lengths": segment_lengths,
        "fitted_transforms": {
            "root_centered_rigid": rigid_transform,
            "root_centered_scale_only": normalized_transform,
            "root_centered_similarity": root_similarity_transform,
            "global_sequence_similarity": global_similarity_transform,
        },
        "absolute_camera_frame": absolute_metric,
        "warnings": warnings,
    }
    series = {
        "time_s": time,
        "joint_names": np.asarray(pred_names, dtype=object),
        "prediction_eval_m": pred,
        "mocap_eval_m": ref_rs,
        "prediction_root_centered_m": pred_centered,
        "mocap_root_centered_m": ref_centered,
        "prediction_root_centered_rigid_m": pred_rigid,
        "prediction_root_centered_normalized_m": pred_normalized,
        "prediction_root_centered_similarity_m": pred_root_similarity,
        "prediction_global_similarity_m": pred_global_similarity,
        **foot_series,
    }
    if "raw_frame_ids" in timebase:
        series["raw_frame_ids"] = np.asarray(timebase["raw_frame_ids"], dtype=int)[valid_time]
    if "sync_frame_ids" in timebase:
        series["sync_frame_ids"] = np.asarray(timebase["sync_frame_ids"], dtype=int)[valid_time]
    return report, series


def resolve_prediction_timebase(
    pose: dict,
    cfg: dict | None = None,
    timeline_report: dict | None = None,
    min_overlap_frames: int | None = None,
) -> dict[str, Any]:
    fps = float(pose.get("fps") or 30.0)
    frame_count = int(np.asarray(pose["joints_3d"]).shape[0])
    validation_cfg = ((cfg or {}).get("config", {}).get("mocap_validation", {}) or {})
    min_frames = int(min_overlap_frames or validation_cfg.get("min_overlap_frames", 5))
    if pose.get("backend") != "wham":
        indices = np.arange(frame_count, dtype=int)
        return {
            "mode": "synced_sequence",
            "pose_indices": indices,
            "prediction_timestamps_s": indices.astype(float) / fps,
            "prediction_fps": fps,
            "frames": frame_count,
        }

    if timeline_report is None:
        from monocap_v2.core.wham_timeline import build_wham_timeline_report

        timeline_report = build_wham_timeline_report(pose, cfg)
    alignment = timeline_report.get("raw_sync_alignment") or {}
    overlap = timeline_report.get("overlap") or {}
    if timeline_report.get("status") != "ok" or alignment.get("status") != "ok" or overlap.get("status") != "ok":
        raise ValueError("WHAM raw/sync timeline alignment is uncertain; mocap validation was skipped.")
    offset = alignment.get("best_raw_offset")
    sync_count = alignment.get("sync_frame_count")
    sync_fps = float(alignment.get("sync_fps") or fps)
    if offset is None or sync_count is None:
        raise ValueError("WHAM raw/sync timeline alignment is missing offset or synced frame count.")
    meta = pose.get("backend_meta") or {}
    raw_frame_ids = np.asarray(meta.get("raw_frame_ids", []), dtype=int).reshape(-1)
    if raw_frame_ids.size == 0:
        frame_ids = np.asarray(meta.get("frame_ids", []), dtype=int).reshape(-1)
        raw_frame_ids = frame_ids + int(meta.get("start_frame") or 0)
    if raw_frame_ids.size != frame_count:
        raise ValueError("WHAM frame IDs do not match the pose frame count.")
    keep = (raw_frame_ids >= int(offset)) & (raw_frame_ids < int(offset) + int(sync_count))
    indices = np.flatnonzero(keep)
    if indices.size < min_frames:
        raise ValueError(f"WHAM/synced overlap contains {indices.size} frames; at least {min_frames} are required.")
    kept_raw = raw_frame_ids[indices]
    sync_frame_ids = kept_raw - int(offset)
    return {
        "mode": "wham_raw_to_synced_overlap",
        "pose_indices": indices,
        "raw_frame_ids": kept_raw,
        "sync_frame_ids": sync_frame_ids,
        "prediction_timestamps_s": sync_frame_ids.astype(float) / sync_fps,
        "prediction_fps": fps,
        "sync_fps": sync_fps,
        "best_raw_offset": int(offset),
        "frames": int(indices.size),
    }


def prediction_lower_limb(pose: dict) -> tuple[list[str], np.ndarray]:
    source_names = [str(n) for n in pose.get("joint_names", [])]
    source = camera_to_eval_coords(np.asarray(pose["joints_3d"], dtype=float))
    return _select_prediction_joints(source_names, source)


def prediction_lower_limb_camera(pose: dict) -> tuple[list[str], np.ndarray]:
    source_names = [str(n) for n in pose.get("joint_names", [])]
    source = np.asarray(pose["joints_3d"], dtype=float)
    return _select_prediction_joints(source_names, source)


def _select_prediction_joints(source_names: list[str], source: np.ndarray) -> tuple[list[str], np.ndarray]:
    names = []
    arrays = []
    for target, candidates in PREDICTION_JOINTS:
        idx = find_joint(source_names, candidates)
        if idx is None:
            continue
        names.append(target)
        arrays.append(source[:, idx, :])
    if not arrays:
        raise ValueError("No comparable lower-limb prediction joints found.")
    return names, np.stack(arrays, axis=1)


def mocap_lower_limb(trc: TRCData, target_names: Iterable[str], mocap_axis: str = "-y,x,z") -> np.ndarray:
    ref, _ = mocap_lower_limb_with_mapping(trc, target_names, mocap_axis=mocap_axis)
    return ref


def mocap_lower_limb_with_mapping(
    trc: TRCData,
    target_names: Iterable[str],
    mocap_axis: str | None = "-y,x,z",
) -> tuple[np.ndarray, dict[str, str | None]]:
    marker_lookup = {_canon(name): values for name, values in trc.markers.items()}
    original_names = {_canon(name): name for name in trc.markers}
    arrays = []
    selected = {}
    mapping = dict(MOCAP_MARKERS)
    for target in target_names:
        candidates = mapping.get(target, ())
        arr = None
        selected_name = None
        for candidate in candidates:
            arr = marker_lookup.get(_canon(candidate))
            if arr is not None:
                selected_name = original_names[_canon(candidate)]
                break
        if arr is None:
            arr = np.full((trc.time.shape[0], 3), np.nan, dtype=float)
        arrays.append(arr)
        selected[target] = selected_name
    ref = np.stack(arrays, axis=1)
    return (apply_axis_expr(ref, mocap_axis) if mocap_axis else ref), selected


def resample_timeseries(t_src: np.ndarray, values: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full((len(t_dst), values.shape[1], 3), np.nan, dtype=float)
    for joint_idx in range(values.shape[1]):
        for coord_idx in range(3):
            series = values[:, joint_idx, coord_idx]
            valid = np.isfinite(series) & np.isfinite(t_src)
            if np.count_nonzero(valid) < 2:
                continue
            ts = t_src[valid]
            xs = series[valid]
            interp = np.interp(t_dst, ts, xs)
            interp[(t_dst < ts.min()) | (t_dst > ts.max())] = np.nan
            out[:, joint_idx, coord_idx] = interp
    return out


def root_centered_sequence_mpjpe(pred: np.ndarray, ref: np.ndarray, joint_names: list[str], align: str = "rigid") -> dict:
    pred_centered = root_center(pred, joint_names)
    ref_centered = root_center(ref, joint_names)
    metrics, _, _ = sequence_mpjpe_with_transform(
        pred_centered,
        ref_centered,
        joint_names,
        align=align,
        allow_translation=align not in {"rigid", "similarity"},
    )
    return metrics


def sequence_mpjpe(pred: np.ndarray, ref: np.ndarray, joint_names: list[str], align: str = "none") -> dict:
    metrics, _, _ = sequence_mpjpe_with_transform(pred, ref, joint_names, align=align)
    return metrics


def sequence_mpjpe_with_transform(
    pred: np.ndarray,
    ref: np.ndarray,
    joint_names: list[str],
    align: str = "none",
    allow_translation: bool = True,
) -> tuple[dict, dict[str, Any], np.ndarray]:
    pred_eval = np.asarray(pred, dtype=float).copy()
    ref_eval = np.asarray(ref, dtype=float)
    rot = np.eye(3)
    trans = np.zeros(3)
    scale = 1.0
    if align in {"rigid", "similarity"}:
        mask = np.isfinite(pred_eval).all(axis=2) & np.isfinite(ref_eval).all(axis=2)
        if np.count_nonzero(mask) >= 3:
            rot, trans, scale = kabsch_align(pred_eval[mask], ref_eval[mask], mode=align)
            if not allow_translation:
                trans = np.zeros(3)
            pred_eval = apply_similarity(pred_eval, rot, trans, scale)
    return mpjpe(pred_eval, ref_eval, joint_names), _transform_dict(rot, trans, scale), pred_eval


def normalized_mpjpe(pred: np.ndarray, ref: np.ndarray, joint_names: list[str]) -> tuple[dict, dict[str, Any], np.ndarray]:
    pred_eval = np.asarray(pred, dtype=float).copy()
    ref_eval = np.asarray(ref, dtype=float)
    mask = np.isfinite(pred_eval).all(axis=2) & np.isfinite(ref_eval).all(axis=2)
    denom = float(np.sum(pred_eval[mask] ** 2)) if np.any(mask) else 0.0
    scale = float(np.sum(pred_eval[mask] * ref_eval[mask]) / denom) if denom > 1e-12 else 1.0
    pred_eval *= scale
    return mpjpe(pred_eval, ref_eval, joint_names), _transform_dict(np.eye(3), np.zeros(3), scale), pred_eval


def per_frame_aligned_pose(pred: np.ndarray, ref: np.ndarray, align: str = "similarity") -> np.ndarray:
    pred_eval = np.asarray(pred, dtype=float).copy()
    ref_eval = np.asarray(ref, dtype=float)
    valid = np.isfinite(pred_eval).all(axis=2) & np.isfinite(ref_eval).all(axis=2)
    if pred_eval.ndim == 3 and pred_eval.shape == ref_eval.shape and valid.size and np.all(valid):
        pred_mean = pred_eval.mean(axis=1)
        ref_mean = ref_eval.mean(axis=1)
        pred_centered = pred_eval - pred_mean[:, None, :]
        ref_centered = ref_eval - ref_mean[:, None, :]
        cov = np.einsum("tji,tjk->tik", pred_centered, ref_centered)
        u, s, vt = np.linalg.svd(cov)
        rot = np.einsum("tji,tkj->tik", vt, u)
        reflected = np.linalg.det(rot) < 0
        if np.any(reflected):
            vt = vt.copy()
            vt[reflected, -1, :] *= -1
            rot = np.einsum("tji,tkj->tik", vt, u)
        scale = np.ones(pred_eval.shape[0], dtype=float)
        if align == "similarity":
            denom = np.sum(pred_centered**2, axis=(1, 2)) + 1e-12
            scale = np.sum(s, axis=1) / denom
        elif align != "rigid":
            raise ValueError("align must be 'rigid' or 'similarity'")
        trans = ref_mean - scale[:, None] * np.einsum("tij,tj->ti", rot, pred_mean)
        return scale[:, None, None] * np.einsum("tjc,tic->tji", pred_eval, rot) + trans[:, None, :]
    for frame_idx in range(pred_eval.shape[0]):
        mask = valid[frame_idx]
        if np.count_nonzero(mask) < 3:
            pred_eval[frame_idx] = np.nan
            continue
        rot, trans, scale = kabsch_align(pred_eval[frame_idx, mask], ref_eval[frame_idx, mask], mode=align)
        pred_eval[frame_idx] = apply_similarity(pred_eval[frame_idx], rot, trans, scale)
    return pred_eval


def per_frame_aligned_mpjpe(pred: np.ndarray, ref: np.ndarray, joint_names: list[str], align: str = "similarity") -> dict:
    pred_eval = per_frame_aligned_pose(pred, ref, align=align)
    ref_eval = np.asarray(ref, dtype=float)
    return mpjpe(pred_eval, ref_eval, joint_names)


def mpjpe(pred: np.ndarray, ref: np.ndarray, joint_names: list[str]) -> dict:
    diff = np.asarray(pred, dtype=float) - np.asarray(ref, dtype=float)
    dist = np.linalg.norm(diff, axis=2)
    valid = np.isfinite(dist)
    per_frame = np.where(valid, dist, np.nan)
    per_joint = {}
    for idx, name in enumerate(joint_names):
        per_joint[name] = float(np.nanmean(per_frame[:, idx])) if np.isfinite(per_frame[:, idx]).any() else None
    return {
        "mpjpe_m": _m_value(np.nanmean(per_frame)) if np.isfinite(per_frame).any() else None,
        "mpjpe_mm": _mm_value(np.nanmean(per_frame)) if np.isfinite(per_frame).any() else None,
        "per_joint_mpjpe_m": per_joint,
        "per_joint_mpjpe_mm": {name: _mm_value(value) for name, value in per_joint.items()},
        "valid_ratio": float(valid.mean()) if valid.size else 0.0,
    }


def root_center(values: np.ndarray, joint_names: list[str]) -> np.ndarray:
    left = joint_names.index("left_hip")
    right = joint_names.index("right_hip")
    root = 0.5 * (values[:, left, :] + values[:, right, :])
    centered = values - root[:, None, :]
    invalid = ~np.isfinite(root).all(axis=1)
    centered[invalid] = np.nan
    return centered


def summarize_segment_lengths(pred: np.ndarray, ref: np.ndarray, joint_names: list[str]) -> dict[str, Any]:
    segments = {
        "left_thigh": ("left_hip", "left_knee"),
        "right_thigh": ("right_hip", "right_knee"),
        "left_shank": ("left_knee", "left_ankle"),
        "right_shank": ("right_knee", "right_ankle"),
    }
    out = {}
    for name, (a_name, b_name) in segments.items():
        if a_name not in joint_names or b_name not in joint_names:
            continue
        a = joint_names.index(a_name)
        b = joint_names.index(b_name)
        pred_lengths = np.linalg.norm(pred[:, a] - pred[:, b], axis=1)
        ref_lengths = np.linalg.norm(ref[:, a] - ref[:, b], axis=1)
        pred_median = _finite_median(pred_lengths)
        ref_median = _finite_median(ref_lengths)
        out[name] = {
            "prediction_median_mm": _mm_value(pred_median),
            "prediction_std_mm": _mm_value(_finite_std(pred_lengths)),
            "mocap_median_mm": _mm_value(ref_median),
            "mocap_std_mm": _mm_value(_finite_std(ref_lengths)),
            "prediction_minus_mocap_median_mm": _mm_value(pred_median - ref_median)
            if pred_median is not None and ref_median is not None
            else None,
        }
    return out


def foot_trajectory_series(pred: np.ndarray, ref: np.ndarray, joint_names: list[str], time_s: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        name = f"{side}_toe"
        if name not in joint_names:
            continue
        idx = joint_names.index(name)
        out[f"{side}_foot_prediction_height_m"] = pred[:, idx, 2]
        out[f"{side}_foot_mocap_height_m"] = ref[:, idx, 2]
        out[f"{side}_foot_prediction_speed_mps"] = _trajectory_speed(pred[:, idx], time_s)
        out[f"{side}_foot_mocap_speed_mps"] = _trajectory_speed(ref[:, idx], time_s)
    return out


def validate_mocap_to_video_transform(path: Path | str | None) -> dict[str, Any]:
    if not path:
        return {"status": "skipped", "reason": "Manifest mocap_to_video transform is unavailable."}
    transform_path = Path(path)
    if not transform_path.exists():
        return {"status": "warning", "source": str(transform_path), "warnings": ["Manifest mocap_to_video transform does not exist."]}
    try:
        payload = read_yaml(transform_path)
        rotation = np.asarray(payload.get("R_fromMocap_toVideo"), dtype=float)
        translation = np.asarray(payload.get("position_fromMocapOrigin_toVideoOrigin"), dtype=float).reshape(-1)
    except Exception as exc:
        return {"status": "warning", "source": str(transform_path), "warnings": [f"Could not parse mocap_to_video transform: {exc}"]}
    warnings = []
    if rotation.shape != (3, 3):
        warnings.append(f"Rotation shape must be [3, 3], got {list(rotation.shape)}.")
    if translation.shape != (3,):
        warnings.append(f"Translation shape must be [3], got {list(translation.shape)}.")
    det = None
    orthonormal_error = None
    if rotation.shape == (3, 3):
        det = float(np.linalg.det(rotation))
        orthonormal_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
        if orthonormal_error > 1e-3:
            warnings.append(f"Rotation is not orthonormal (error={orthonormal_error:.6g}).")
        if abs(det - 1.0) > 1e-3:
            warnings.append(f"Rotation determinant must be near +1, got {det:.6g}.")
    return {
        "status": "warning" if warnings else "ok",
        "source": str(transform_path),
        "rotation_matrix": rotation.tolist() if rotation.shape == (3, 3) else None,
        "translation": translation.tolist() if translation.shape == (3,) else None,
        "translation_units": "m",
        "determinant": det,
        "orthonormality_error": orthonormal_error,
        "warnings": warnings,
    }


def _absolute_camera_metric(
    pose: dict,
    trc: TRCData,
    pred_names: list[str],
    pose_indices: np.ndarray,
    time_s: np.ndarray,
    marker_mapping: dict[str, str | None],
    transform: dict | None,
) -> dict[str, Any]:
    if not transform or transform.get("status") != "ok":
        return {
            "status": "unavailable",
            "reason": "A valid manifest mocap_to_video transform is required for absolute camera-frame MPJPE.",
        }
    rotation = np.asarray(transform["rotation_matrix"], dtype=float)
    translation = np.asarray(transform["translation"], dtype=float)
    _, pred_camera_all = prediction_lower_limb_camera(pose)
    ref_mocap, _ = mocap_lower_limb_with_mapping(trc, pred_names, mocap_axis=None)
    ref_camera = ref_mocap @ rotation.T + translation
    ref_camera_rs = resample_timeseries(trc.time, ref_camera, time_s)
    metrics = mpjpe(pred_camera_all[pose_indices], ref_camera_rs, pred_names)
    return {
        "status": "ok",
        "metric": "absolute_camera_frame_mpjpe",
        "mpjpe_m": metrics["mpjpe_m"],
        "mpjpe_mm": metrics["mpjpe_mm"],
        "mocap_marker_mapping": marker_mapping,
    }


def _public_timebase(timebase: dict[str, Any], valid_time: np.ndarray) -> dict[str, Any]:
    out = {key: value for key, value in timebase.items() if key not in {"pose_indices", "prediction_timestamps_s"}}
    out["frames_after_mocap_overlap"] = int(np.count_nonzero(valid_time))
    if "raw_frame_ids" in out:
        out["raw_frame_ids"] = np.asarray(out["raw_frame_ids"], dtype=int)[valid_time].tolist()
    if "sync_frame_ids" in out:
        out["sync_frame_ids"] = np.asarray(out["sync_frame_ids"], dtype=int)[valid_time].tolist()
    return out


def _trajectory_speed(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    speed = np.full(len(values), np.nan, dtype=float)
    if len(values) < 2:
        return speed
    dt = np.diff(time_s)
    delta = np.linalg.norm(np.diff(values, axis=0), axis=1)
    valid = np.isfinite(delta) & np.isfinite(dt) & (dt > 0)
    speed[1:][valid] = delta[valid] / dt[valid]
    return speed


def _transform_dict(rot: np.ndarray, trans: np.ndarray, scale: float) -> dict[str, Any]:
    return {"rotation_matrix": np.asarray(rot).tolist(), "translation": np.asarray(trans).tolist(), "scale": float(scale)}


def _finite_median(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    return float(np.nanmedian(finite)) if np.isfinite(finite).any() else None


def _finite_std(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    return float(np.nanstd(finite)) if np.isfinite(finite).any() else None


def _split_header(line: str) -> list[str]:
    if "\t" in line:
        return [part.strip() for part in line.split("\t") if part.strip()]
    return [part.strip() for part in re.split(r"[\s,]+", line.strip()) if part.strip()]


def _parse_marker_names(line: str) -> list[str]:
    parts = _split_header(line)
    if len(parts) >= 3:
        return parts[2:]
    return []


def _parse_float(parts: list[str], idx: int) -> float:
    if idx >= len(parts):
        return float("nan")
    try:
        return float(parts[idx])
    except ValueError:
        return float("nan")


def _canon(name: str) -> str:
    return name.strip().lower().replace("_", "").replace("-", "").replace(".", "")


def _m_value(value) -> float | None:
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)


def _mm_value(value) -> float | None:
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value) * 1000.0
