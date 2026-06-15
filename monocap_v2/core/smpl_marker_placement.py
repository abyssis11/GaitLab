from __future__ import annotations

from typing import Any

import numpy as np

from monocap_v2.core.geometry import find_joint


LOWER_LIMB_MARKERS = [
    "DBG_PELV",
    "DBG_LHIP",
    "DBG_RHIP",
    "DBG_LKNE",
    "DBG_RKNE",
    "DBG_LANK",
    "DBG_RANK",
    "DBG_LHEE",
    "DBG_RHEE",
    "DBG_LTOE",
    "DBG_RTOE",
]

MARKER_TO_JOINT = {
    "DBG_PELV": "pelv",
    "DBG_LHIP": "lhip",
    "DBG_RHIP": "rhip",
    "DBG_LKNE": "lkne",
    "DBG_RKNE": "rkne",
    "DBG_LANK": "lank",
    "DBG_RANK": "rank",
    "DBG_LHEE": "left_heel",
    "DBG_RHEE": "right_heel",
    "DBG_LTOE": "left_big_toe",
    "DBG_RTOE": "right_big_toe",
}


def lower_limb_marker_placement_qc(pose3d: dict[str, Any], marker_payload: dict[str, Any], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    qc_cfg = cfg or {}
    thresholds = {
        "max_joint_distance_m": float(qc_cfg.get("max_joint_distance_m", 0.20)),
        "min_ankle_to_foot_marker_m": float(qc_cfg.get("min_ankle_to_foot_marker_m", 0.03)),
        "max_ankle_to_foot_marker_m": float(qc_cfg.get("max_ankle_to_foot_marker_m", 0.40)),
        "max_marker_jump_m": float(qc_cfg.get("max_marker_jump_m", 0.20)),
    }
    markers = np.asarray(marker_payload.get("markers_m"), dtype=float)
    marker_names = [str(name) for name in marker_payload.get("marker_names", [])]
    marker_lookup = {name: idx for idx, name in enumerate(marker_names)}
    joints = np.asarray(pose3d.get("joints_3d"), dtype=float)
    joint_names = [str(name) for name in pose3d.get("joint_names", [])]
    if markers.ndim != 3 or markers.shape[-1] != 3:
        raise ValueError("markers_m must have shape [T, M, 3].")
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError("pose3d joints_3d must have shape [T, J, 3].")

    marker_frame_ids = _timebase_array(marker_payload.get("raw_frame_ids"))
    if marker_frame_ids.size == 0:
        marker_frame_ids = _timebase_array(marker_payload.get("frame_ids"))
    pose_frame_ids = _pose_raw_frame_ids(pose3d, joints.shape[0])
    marker_indices, pose_indices = _aligned_frame_indices(marker_frame_ids, pose_frame_ids, markers.shape[0], joints.shape[0])
    if marker_indices.size == 0:
        raise ValueError("Could not align marker frames to pose frames for placement QC.")

    aligned_markers = markers[marker_indices]
    aligned_joints = joints[pose_indices]
    per_marker = {}
    warnings: list[str] = []
    worst_marker = None
    worst_score = -np.inf
    for name in LOWER_LIMB_MARKERS:
        if name not in marker_lookup:
            warnings.append(f"Required lower-limb marker {name} is missing.")
            continue
        joint_name = _marker_source_joint(name, marker_payload)
        joint_idx = find_joint(joint_names, [joint_name])
        marker_idx = marker_lookup[name]
        entry: dict[str, Any] = {"source_joint": joint_name, "marker_index": marker_idx}
        if joint_idx is None:
            entry["status"] = "warning"
            entry["warnings"] = [f"Could not resolve source joint {joint_name}."]
            warnings.extend(entry["warnings"])
            per_marker[name] = entry
            continue
        values = aligned_markers[:, marker_idx, :]
        joint_values = aligned_joints[:, joint_idx, :]
        distances = np.linalg.norm(values - joint_values, axis=1)
        jumps = np.linalg.norm(np.diff(values, axis=0), axis=1) if values.shape[0] > 1 else np.zeros((0,), dtype=float)
        entry_warnings = []
        median_distance = _nanmedian(distances)
        max_jump = _nanmax_or_zero(jumps)
        if median_distance > thresholds["max_joint_distance_m"]:
            entry_warnings.append(f"Median marker-to-joint distance {median_distance:.3f} m exceeds threshold.")
        if max_jump > thresholds["max_marker_jump_m"]:
            entry_warnings.append(f"Max marker jump {max_jump:.3f} m exceeds threshold.")
        entry.update(
            {
                "status": "warning" if entry_warnings else "ok",
                "joint_index": int(joint_idx),
                "distance_to_source_joint_m": {
                    "first_frame": float(distances[0]) if distances.size else None,
                    "median": median_distance,
                    "max": _nanmax_or_zero(distances),
                },
                "trajectory": {
                    "max_jump_m": max_jump,
                    "median_jump_m": _nanmedian(jumps) if jumps.size else 0.0,
                    "smoothness_second_diff_m": _second_diff_metric(values),
                },
                "median_up_m": _nanmedian(_up(values)),
                "warnings": entry_warnings,
            }
        )
        warnings.extend(f"{name}: {warning}" for warning in entry_warnings)
        score = median_distance / max(thresholds["max_joint_distance_m"], 1e-9) + max_jump / max(thresholds["max_marker_jump_m"], 1e-9)
        if score > worst_score:
            worst_score = score
            worst_marker = name
        per_marker[name] = entry

    pair_checks = _pair_checks(aligned_markers, marker_lookup)
    segment_checks, segment_warnings = _segment_checks(aligned_markers, marker_lookup, thresholds)
    ordering_checks, ordering_warnings = _vertical_ordering_checks(aligned_markers, aligned_joints, marker_lookup, joint_names)
    warnings.extend(pair_checks["warnings"])
    warnings.extend(segment_warnings)
    warnings.extend(ordering_warnings)
    status = "warning" if warnings else "ok"
    return {
        "status": status,
        "group": "lower_limb",
        "marker_set": marker_payload.get("marker_set"),
        "debug": bool(marker_payload.get("debug", False)),
        "frames": int(marker_indices.size),
        "marker_frame_start": int(marker_indices[0]),
        "marker_frame_end": int(marker_indices[-1]),
        "raw_frame_start": int(marker_frame_ids[marker_indices[0]]) if marker_frame_ids.size else None,
        "raw_frame_end": int(marker_frame_ids[marker_indices[-1]]) if marker_frame_ids.size else None,
        "thresholds": thresholds,
        "markers_evaluated": sorted(per_marker),
        "per_marker": per_marker,
        "pair_checks": pair_checks,
        "segment_checks": segment_checks,
        "vertical_ordering": ordering_checks,
        "worst_marker": worst_marker,
        "warnings": warnings,
    }


def build_lower_limb_marker_map_proposal(marker_payload: dict[str, Any], proposal_name: str = "wham_smpl_debug_v2") -> dict[str, Any]:
    names = [str(name) for name in marker_payload.get("marker_names", [])]
    vertex_indices = [int(v) for v in marker_payload.get("vertex_indices", [])]
    if len(names) != len(vertex_indices):
        raise ValueError("marker_names and vertex_indices must have the same length for marker-map proposal.")
    lookup = dict(zip(names, vertex_indices))
    missing = [name for name in LOWER_LIMB_MARKERS if name not in lookup]
    if missing:
        raise ValueError(f"Cannot build lower-limb marker proposal; missing markers: {', '.join(missing)}")
    markers = []
    resolved_from = marker_payload.get("resolved_from") or {}
    for name in LOWER_LIMB_MARKERS:
        entry = {"name": name, "vertex_index": int(lookup[name])}
        source = resolved_from.get(name) or {}
        if source:
            entry["source"] = source
        markers.append(entry)
    return {
        "default_marker_set": proposal_name,
        "marker_sets": {
            proposal_name: {
                "description": "Proposed lower-limb fixed-vertex debug marker set generated from WHAM SMPL placement QC.",
                "debug": True,
                "coordinate_space": marker_payload.get("coordinate_space", "wham_camera_local"),
                "source_marker_set": marker_payload.get("marker_set"),
                "markers": markers,
            }
        },
    }


def _marker_source_joint(marker_name: str, marker_payload: dict[str, Any]) -> str:
    source = (marker_payload.get("resolved_from") or {}).get(marker_name) or {}
    if source.get("type") == "nearest_vertex_to_joint" and source.get("value"):
        return str(source["value"])
    return MARKER_TO_JOINT[marker_name]


def _pose_raw_frame_ids(pose3d: dict[str, Any], frame_count: int) -> np.ndarray:
    meta = pose3d.get("backend_meta") or {}
    raw = _timebase_array(meta.get("raw_frame_ids"))
    if raw.size == frame_count:
        return raw.astype(int)
    frame_ids = _timebase_array(meta.get("frame_ids"))
    if frame_ids.size == frame_count:
        start_frame = int(meta.get("start_frame") or 0)
        return frame_ids.astype(int) + start_frame
    return np.arange(frame_count, dtype=int)


def _aligned_frame_indices(marker_frame_ids: np.ndarray, pose_frame_ids: np.ndarray, marker_count: int, pose_count: int) -> tuple[np.ndarray, np.ndarray]:
    if marker_frame_ids.size == marker_count and pose_frame_ids.size == pose_count:
        pose_lookup = {int(raw): idx for idx, raw in enumerate(pose_frame_ids.tolist())}
        marker_indices = []
        pose_indices = []
        for marker_idx, raw in enumerate(marker_frame_ids.astype(int).tolist()):
            pose_idx = pose_lookup.get(int(raw))
            if pose_idx is not None:
                marker_indices.append(marker_idx)
                pose_indices.append(pose_idx)
        return np.asarray(marker_indices, dtype=int), np.asarray(pose_indices, dtype=int)
    n = min(marker_count, pose_count)
    return np.arange(n, dtype=int), np.arange(n, dtype=int)


def _pair_checks(markers: np.ndarray, lookup: dict[str, int]) -> dict[str, Any]:
    pairs = [("DBG_LHIP", "DBG_RHIP"), ("DBG_LKNE", "DBG_RKNE"), ("DBG_LANK", "DBG_RANK"), ("DBG_LHEE", "DBG_RHEE"), ("DBG_LTOE", "DBG_RTOE")]
    out = {}
    warnings = []
    for left, right in pairs:
        if left not in lookup or right not in lookup:
            continue
        diff = markers[:, lookup[left], :] - markers[:, lookup[right], :]
        separation = np.linalg.norm(diff, axis=1)
        up_diff = np.abs(_up(markers[:, lookup[left], :]) - _up(markers[:, lookup[right], :]))
        entry = {
            "median_separation_m": _nanmedian(separation),
            "median_abs_up_difference_m": _nanmedian(up_diff),
            "status": "ok",
            "warnings": [],
        }
        if entry["median_separation_m"] < 0.02:
            warning = f"{left}/{right} median separation is very small."
            entry["status"] = "warning"
            entry["warnings"].append(warning)
            warnings.append(warning)
        if entry["median_abs_up_difference_m"] > 0.25:
            warning = f"{left}/{right} median vertical asymmetry is large."
            entry["status"] = "warning"
            entry["warnings"].append(warning)
            warnings.append(warning)
        out[f"{left}_{right}"] = entry
    return {"status": "warning" if warnings else "ok", "pairs": out, "warnings": warnings}


def _segment_checks(markers: np.ndarray, lookup: dict[str, int], thresholds: dict[str, float]) -> tuple[dict[str, Any], list[str]]:
    checks = {}
    warnings = []
    for side, ankle, heel, toe in [("left", "DBG_LANK", "DBG_LHEE", "DBG_LTOE"), ("right", "DBG_RANK", "DBG_RHEE", "DBG_RTOE")]:
        side_entry = {}
        for foot_name in [heel, toe]:
            if ankle not in lookup or foot_name not in lookup:
                continue
            distances = np.linalg.norm(markers[:, lookup[ankle], :] - markers[:, lookup[foot_name], :], axis=1)
            median_distance = _nanmedian(distances)
            entry_warnings = []
            if median_distance < thresholds["min_ankle_to_foot_marker_m"]:
                entry_warnings.append(f"{ankle}->{foot_name} distance is too small: {median_distance:.3f} m.")
            if median_distance > thresholds["max_ankle_to_foot_marker_m"]:
                entry_warnings.append(f"{ankle}->{foot_name} distance is too large: {median_distance:.3f} m.")
            side_entry[foot_name] = {"median_distance_m": median_distance, "status": "warning" if entry_warnings else "ok", "warnings": entry_warnings}
            warnings.extend(entry_warnings)
        checks[side] = side_entry
    return {"status": "warning" if warnings else "ok", "sides": checks}, warnings


def _vertical_ordering_checks(markers: np.ndarray, joints: np.ndarray, marker_lookup: dict[str, int], joint_names: list[str]) -> tuple[dict[str, Any], list[str]]:
    checks = {}
    warnings = []
    for side, hip, knee, ankle, heel, toe in [
        ("left", "lhip", "lkne", "lank", "DBG_LHEE", "DBG_LTOE"),
        ("right", "rhip", "rkne", "rank", "DBG_RHEE", "DBG_RTOE"),
    ]:
        hip_idx = find_joint(joint_names, [hip])
        knee_idx = find_joint(joint_names, [knee])
        ankle_idx = find_joint(joint_names, [ankle])
        if hip_idx is None or knee_idx is None or ankle_idx is None:
            continue
        hip_up = _nanmedian(_up(joints[:, hip_idx, :]))
        knee_up = _nanmedian(_up(joints[:, knee_idx, :]))
        ankle_up = _nanmedian(_up(joints[:, ankle_idx, :]))
        side_checks = {
            "hip_above_knee_m": float(hip_up - knee_up),
            "knee_above_ankle_m": float(knee_up - ankle_up),
            "status": "ok",
            "warnings": [],
        }
        if hip_up <= knee_up:
            side_checks["warnings"].append(f"{side} hip is not above knee.")
        if knee_up <= ankle_up:
            side_checks["warnings"].append(f"{side} knee is not above ankle.")
        for foot_marker in [heel, toe]:
            if foot_marker not in marker_lookup:
                continue
            foot_up = _nanmedian(_up(markers[:, marker_lookup[foot_marker], :]))
            key = f"{foot_marker}_below_knee_m"
            side_checks[key] = float(knee_up - foot_up)
            side_checks[f"{foot_marker}_relative_to_ankle_m"] = float(ankle_up - foot_up)
            if knee_up - foot_up < 0.10:
                side_checks["warnings"].append(f"{foot_marker} is not sufficiently below knee.")
            if foot_up > ankle_up + 0.15:
                side_checks["warnings"].append(f"{foot_marker} is too high relative to ankle.")
        if side_checks["warnings"]:
            side_checks["status"] = "warning"
            warnings.extend(side_checks["warnings"])
        checks[side] = side_checks
    return {"status": "warning" if warnings else "ok", "sides": checks}, warnings


def _up(values: np.ndarray) -> np.ndarray:
    return -np.asarray(values, dtype=float)[..., 1]


def _second_diff_metric(values: np.ndarray) -> float:
    if values.shape[0] < 3:
        return 0.0
    second = np.diff(values, n=2, axis=0)
    return _nanmedian(np.linalg.norm(second, axis=1))


def _timebase_array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    try:
        return np.asarray(value).reshape(-1)
    except Exception:
        return np.asarray([])


def _nanmedian(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmedian(arr)) if arr.size and np.isfinite(arr).any() else 0.0


def _nanmax_or_zero(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmax(arr)) if arr.size and np.isfinite(arr).any() else 0.0
