from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from monocap_v2.core.geometry import apply_axis_expr, apply_similarity, camera_to_eval_coords
from monocap_v2.core.mocap_eval import parse_trc, resample_timeseries


LOWER_LIMB_MARKER_COMPARISONS = [
    {
        "virtual_marker": "DBG_PELV",
        "mocap_candidates": ["L.ASIS", "r.ASIS", "L.PSIS", "r.PSIS"],
        "combine": "mean_available",
        "anatomical_role": "pelvis_surface_centroid",
        "correspondence_quality": "proxy_composite",
        "interpretation": "SMPL pelvis debug anchor versus mean available ASIS/PSIS surface markers.",
    },
    {
        "virtual_marker": "DBG_LHIP",
        "mocap_candidates": ["L_HJC", "L_HJC_reg"],
        "anatomical_role": "left_hip_joint_center",
        "correspondence_quality": "non_equivalent",
        "interpretation": "SMPL surface vertex nearest hip joint versus mocap hip joint center.",
    },
    {
        "virtual_marker": "DBG_RHIP",
        "mocap_candidates": ["R_HJC", "R_HJC_reg"],
        "anatomical_role": "right_hip_joint_center",
        "correspondence_quality": "non_equivalent",
        "interpretation": "SMPL surface vertex nearest hip joint versus mocap hip joint center.",
    },
    *[
        {
            "virtual_marker": virtual,
            "mocap_candidates": [mocap],
            "anatomical_role": role,
            "correspondence_quality": "approximate_surface_landmark",
            "interpretation": "Nearest-joint SMPL debug vertex versus external mocap surface marker.",
        }
        for virtual, mocap, role in [
            ("DBG_LKNE", "L_knee", "left_knee"),
            ("DBG_RKNE", "r_knee", "right_knee"),
            ("DBG_LANK", "L_ankle", "left_ankle"),
            ("DBG_RANK", "r_ankle", "right_ankle"),
            ("DBG_LHEE", "L_calc", "left_calcaneus"),
            ("DBG_RHEE", "r_calc", "right_calcaneus"),
            ("DBG_LTOE", "L_toe", "left_toe"),
            ("DBG_RTOE", "r_toe", "right_toe"),
        ]
    ],
]

LEFT_RIGHT_PAIRS = [
    ("DBG_LHIP", "DBG_RHIP"),
    ("DBG_LKNE", "DBG_RKNE"),
    ("DBG_LANK", "DBG_RANK"),
    ("DBG_LHEE", "DBG_RHEE"),
    ("DBG_LTOE", "DBG_RTOE"),
]


def compare_virtual_markers_to_mocap(
    pose: dict[str, Any],
    marker_payload: dict[str, Any],
    mocap_trc: Path | str,
    evaluation_report: dict[str, Any],
    evaluation_series: dict[str, np.ndarray],
    mocap_axis: str = "-y,x,z",
    warn_median_residual_mm: float = 250.0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    markers = np.asarray(marker_payload.get("markers_m"), dtype=float)
    marker_names = [str(name) for name in marker_payload.get("marker_names", [])]
    if markers.ndim != 3 or markers.shape[-1] != 3 or len(marker_names) != markers.shape[1]:
        raise ValueError("virtual_markers.pkl must contain markers_m [T, M, 3] and matching marker_names.")

    source_time = np.asarray(evaluation_series["time_s"], dtype=float)
    marker_indices, series_indices = _aligned_marker_series_indices(marker_payload, evaluation_series, markers.shape[0])
    if marker_indices.size < 1:
        raise ValueError("Could not align virtual markers to mocap-validation timestamps.")
    source_time = source_time[series_indices]
    markers = markers[marker_indices]
    prediction_joints = np.asarray(evaluation_series["prediction_eval_m"], dtype=float)[series_indices]
    mocap_joints = np.asarray(evaluation_series["mocap_eval_m"], dtype=float)[series_indices]
    joint_names = [str(name) for name in evaluation_series["joint_names"]]
    prediction_root = _pelvis_root(prediction_joints, joint_names)
    mocap_root = _pelvis_root(mocap_joints, joint_names)

    transform = ((evaluation_report.get("fitted_transforms") or {}).get("root_centered_rigid") or {})
    rotation = np.asarray(transform.get("rotation_matrix") or np.eye(3), dtype=float)
    translation = np.asarray(transform.get("translation") or np.zeros(3), dtype=float)
    scale = float(transform.get("scale", 1.0))
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Root-centered rigid joint transform is malformed.")

    trc = parse_trc(mocap_trc)
    comparison_time = _native_mocap_time_grid(trc.time, source_time)
    virtual_source_aligned = apply_similarity(camera_to_eval_coords(markers) - prediction_root[:, None, :], rotation, translation, scale)
    virtual_aligned = _resample_points_linear(source_time, virtual_source_aligned, comparison_time)
    mocap_root_comparison = _resample_points_linear(source_time, mocap_root[:, None, :], comparison_time)[:, 0, :]
    mocap_aligned, mappings = _resampled_mocap_markers(trc, comparison_time, mocap_root_comparison, mocap_axis)
    names = [entry["virtual_marker"] for entry in mappings]
    marker_lookup = {name: idx for idx, name in enumerate(marker_names)}
    virtual_selected = np.stack([virtual_aligned[:, marker_lookup[name], :] for name in names], axis=1)
    errors = np.linalg.norm(virtual_selected - mocap_aligned, axis=2)
    finite = np.isfinite(errors)
    warnings = _duplicate_vertex_warnings(marker_payload)
    per_marker = {}
    for idx, mapping in enumerate(mappings):
        values = errors[:, idx]
        valid = values[np.isfinite(values)]
        summary = {
            **mapping,
            "anatomically_equivalent": False,
            "valid_frames": int(valid.size),
            "median_residual_mm": _mm(np.nanmedian(valid)) if valid.size else None,
            "mean_residual_mm": _mm(np.nanmean(valid)) if valid.size else None,
            "p95_residual_mm": _mm(np.nanpercentile(valid, 95)) if valid.size else None,
            "max_residual_mm": _mm(np.nanmax(valid)) if valid.size else None,
        }
        if summary["median_residual_mm"] is not None and summary["median_residual_mm"] > warn_median_residual_mm:
            warnings.append(
                f"{mapping['virtual_marker']} median residual {summary['median_residual_mm']:.3f} mm exceeds "
                f"{warn_median_residual_mm:.3f} mm."
            )
        per_marker[mapping["virtual_marker"]] = summary

    valid_errors = errors[finite]
    quality_counts = {}
    for mapping in mappings:
        quality = mapping["correspondence_quality"]
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    report = {
        "status": "warning" if warnings else "ok",
        "comparison_mode": "root_centered_rigid_from_lower_limb_joints",
        "alignment_fitted_from": "SMPL and mocap lower-limb joints only; marker residuals are evaluation-only.",
        "timebase": {
            "mode": "shared_native_mocap_grid",
            "source_prediction_rate_hz": _rate_hz(source_time),
            "comparison_rate_hz": float(trc.data_rate or _rate_hz(comparison_time)),
            "source_prediction_frames": int(source_time.size),
            "comparison_frames": int(comparison_time.size),
            "time_start_s": float(comparison_time[0]),
            "time_end_s": float(comparison_time[-1]),
        },
        "marker_set": marker_payload.get("marker_set"),
        "debug": True,
        "limitations": [
            "Current SMPL markers are debug anchors, not validated anatomical OpenSim markers.",
            "Residuals guide marker-map design and must not be interpreted as final marker-placement accuracy.",
        ],
        "frames": int(errors.shape[0]),
        "marker_count": int(errors.shape[1]),
        "marker_names": names,
        "correspondence_summary": {
            "exact_anatomical_matches": 0,
            "quality_counts": quality_counts,
            "all_correspondences_are_debug_approximations": True,
        },
        "overall_median_residual_mm": _mm(np.nanmedian(valid_errors)) if valid_errors.size else None,
        "overall_mean_residual_mm": _mm(np.nanmean(valid_errors)) if valid_errors.size else None,
        "overall_p95_residual_mm": _mm(np.nanpercentile(valid_errors, 95)) if valid_errors.size else None,
        "valid_ratio": float(finite.mean()) if finite.size else 0.0,
        "per_marker": per_marker,
        "warnings": list(dict.fromkeys(warnings)),
    }
    comparison_series = {
        "time_s": comparison_time,
        "source_time_s": source_time,
        "marker_names": np.asarray(names, dtype=object),
        "smpl_markers_aligned_m": virtual_selected,
        "mocap_markers_aligned_m": mocap_aligned,
        "residual_mm": errors * 1000.0,
        "prediction_root_source_eval_m": prediction_root,
        "mocap_root_source_eval_m": mocap_root,
        "source_pose_indices": _pose_indices_for_series(pose, evaluation_series, series_indices),
    }
    if "raw_frame_ids" in evaluation_series:
        source_raw = np.asarray(evaluation_series["raw_frame_ids"], dtype=float)[series_indices]
        comparison_series["source_raw_frame_ids"] = source_raw.astype(int)
        comparison_series["raw_frame_positions"] = np.interp(comparison_time, source_time, source_raw)
    return report, comparison_series


def aligned_smpl_mesh_vertices(
    pose: dict[str, Any],
    comparison_series: dict[str, np.ndarray],
    evaluation_report: dict[str, Any],
) -> np.ndarray:
    vertices = np.asarray((pose.get("smpl") or {}).get("vertices"), dtype=float)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError("SMPL vertices are unavailable for marker comparison mesh rendering.")
    pose_indices = np.asarray(comparison_series["source_pose_indices"], dtype=int)
    roots = np.asarray(comparison_series["prediction_root_source_eval_m"], dtype=float)
    source_time = np.asarray(comparison_series["source_time_s"], dtype=float)
    comparison_time = np.asarray(comparison_series["time_s"], dtype=float)
    transform = ((evaluation_report.get("fitted_transforms") or {}).get("root_centered_rigid") or {})
    rotation = np.asarray(transform.get("rotation_matrix") or np.eye(3), dtype=float)
    translation = np.asarray(transform.get("translation") or np.zeros(3), dtype=float)
    aligned_source = apply_similarity(
        camera_to_eval_coords(vertices[pose_indices]) - roots[:, None, :],
        rotation,
        translation,
        float(transform.get("scale", 1.0)),
    )
    return _resample_points_linear(source_time, aligned_source, comparison_time)


def _resampled_mocap_markers(trc, time_s: np.ndarray, mocap_root: np.ndarray, mocap_axis: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    lookup = {_canon(name): (name, values) for name, values in trc.markers.items()}
    arrays = []
    mappings = []
    for definition in LOWER_LIMB_MARKER_COMPARISONS:
        available = [lookup[_canon(name)] for name in definition["mocap_candidates"] if _canon(name) in lookup]
        if not available:
            continue
        if definition.get("combine") == "mean_available":
            values = np.nanmean(np.stack([value for _, value in available], axis=0), axis=0)
            selected = [name for name, _ in available]
        else:
            selected_name, values = available[0]
            selected = [selected_name]
        values = apply_axis_expr(values, mocap_axis)
        arrays.append(resample_timeseries(trc.time, values[:, None, :], time_s)[:, 0, :] - mocap_root)
        mappings.append(
            {
                "virtual_marker": definition["virtual_marker"],
                "mocap_sources": selected,
                "anatomical_role": definition["anatomical_role"],
                "correspondence_quality": definition["correspondence_quality"],
                "interpretation": definition["interpretation"],
            }
        )
    if not arrays:
        raise ValueError("No comparable lower-limb mocap markers were available.")
    return np.stack(arrays, axis=1), mappings


def _aligned_marker_series_indices(marker_payload: dict[str, Any], evaluation_series: dict[str, np.ndarray], marker_count: int) -> tuple[np.ndarray, np.ndarray]:
    marker_raw = _optional_1d(marker_payload.get("raw_frame_ids"))
    series_raw = _optional_1d(evaluation_series.get("raw_frame_ids"))
    if marker_raw.size == marker_count and series_raw.size:
        series_lookup = {int(raw): idx for idx, raw in enumerate(series_raw.astype(int).tolist())}
        pairs = [(idx, series_lookup[int(raw)]) for idx, raw in enumerate(marker_raw.astype(int).tolist()) if int(raw) in series_lookup]
        if pairs:
            marker_indices, series_indices = zip(*pairs)
            return np.asarray(marker_indices, dtype=int), np.asarray(series_indices, dtype=int)
    count = min(marker_count, len(np.asarray(evaluation_series["time_s"])))
    return np.arange(count, dtype=int), np.arange(count, dtype=int)


def _pose_indices_for_series(pose: dict[str, Any], evaluation_series: dict[str, np.ndarray], series_indices: np.ndarray) -> np.ndarray:
    pose_count = int(np.asarray(pose["joints_3d"]).shape[0])
    pose_raw = _optional_1d((pose.get("backend_meta") or {}).get("raw_frame_ids"))
    if pose_raw.size == 0:
        frame_ids = _optional_1d((pose.get("backend_meta") or {}).get("frame_ids"))
        pose_raw = frame_ids + int((pose.get("backend_meta") or {}).get("start_frame") or 0) if frame_ids.size else pose_raw
    series_raw = _optional_1d(evaluation_series.get("raw_frame_ids"))
    if pose_raw.size == pose_count and series_raw.size:
        lookup = {int(raw): idx for idx, raw in enumerate(pose_raw.astype(int).tolist())}
        return np.asarray([lookup[int(raw)] for raw in series_raw[series_indices].astype(int).tolist()], dtype=int)
    return np.arange(len(series_indices), dtype=int)


def _duplicate_vertex_warnings(marker_payload: dict[str, Any]) -> list[str]:
    names = [str(name) for name in marker_payload.get("marker_names", [])]
    vertices = [int(idx) for idx in marker_payload.get("vertex_indices", [])]
    lookup = dict(zip(names, vertices))
    warnings = []
    for left, right in LEFT_RIGHT_PAIRS:
        if left in lookup and right in lookup and lookup[left] == lookup[right]:
            warnings.append(f"{left} and {right} resolve to the same SMPL vertex {lookup[left]}.")
    return warnings


def metric_to_display_coords(values: np.ndarray) -> np.ndarray:
    """Map historical metric axes [-raw_y, raw_x, raw_z] to lateral, progression, up."""
    arr = np.asarray(values, dtype=float)
    out = np.empty_like(arr)
    out[..., 0] = arr[..., 2]
    out[..., 1] = arr[..., 1]
    out[..., 2] = -arr[..., 0]
    return out


def _native_mocap_time_grid(mocap_time: np.ndarray, source_time: np.ndarray) -> np.ndarray:
    mocap = np.asarray(mocap_time, dtype=float)
    source = np.asarray(source_time, dtype=float)
    keep = np.isfinite(mocap) & (mocap >= np.nanmin(source) - 1e-7) & (mocap <= np.nanmax(source) + 1e-7)
    out = mocap[keep]
    if out.size < 2:
        raise ValueError("Native mocap time grid has fewer than two samples inside the trusted prediction window.")
    return out


def _resample_points_linear(source_time: np.ndarray, values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    source = np.asarray(source_time, dtype=float)
    target = np.asarray(target_time, dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.shape[0] != source.size:
        raise ValueError("Time series first dimension must match source_time.")
    if source.size < 2:
        raise ValueError("At least two source frames are required for interpolation.")
    right = np.searchsorted(source, target, side="right")
    right = np.clip(right, 1, source.size - 1)
    left = right - 1
    span = source[right] - source[left]
    weight = np.divide(target - source[left], span, out=np.zeros_like(target), where=np.abs(span) > 1e-12)
    shape = (target.size,) + (1,) * (arr.ndim - 1)
    out = arr[left] * (1.0 - weight.reshape(shape)) + arr[right] * weight.reshape(shape)
    outside = (target < source[0] - 1e-7) | (target > source[-1] + 1e-7)
    out[outside] = np.nan
    return out


def _rate_hz(time_s: np.ndarray) -> float | None:
    values = np.asarray(time_s, dtype=float)
    delta = np.diff(values)
    valid = delta[np.isfinite(delta) & (delta > 1e-12)]
    return float(1.0 / np.median(valid)) if valid.size else None


def _pelvis_root(joints: np.ndarray, names: list[str]) -> np.ndarray:
    return 0.5 * (joints[:, names.index("left_hip"), :] + joints[:, names.index("right_hip"), :])


def _optional_1d(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    return np.asarray(value).reshape(-1)


def _canon(name: str) -> str:
    return name.strip().lower().replace("_", "").replace("-", "").replace(".", "")


def _mm(value: float) -> float:
    return float(value) * 1000.0
