from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from monocap_v2.core.geometry import find_joint
from monocap_v2.core.logging_utils import read_yaml
from monocap_v2.core.wham_timeline import build_wham_timeline_report


def load_marker_set(path: Path, marker_set_name: str | None = None) -> dict[str, Any]:
    data = read_yaml(path)
    marker_sets = data.get("marker_sets")
    if isinstance(marker_sets, dict):
        name = marker_set_name or data.get("default_marker_set")
        if not name:
            raise ValueError(f"Marker map {path} has marker_sets but no default_marker_set.")
        if name not in marker_sets:
            known = ", ".join(sorted(str(key) for key in marker_sets))
            raise ValueError(f"Unknown marker set '{name}'. Known marker sets: {known}")
        marker_set = dict(marker_sets[name] or {})
        marker_set["name"] = str(name)
    else:
        marker_set = {
            "name": marker_set_name or data.get("name") or "legacy_marker_set",
            "debug": bool(data.get("debug", False)),
            "coordinate_space": data.get("coordinate_space"),
            "markers": data.get("markers", []),
        }
    validate_marker_set(marker_set)
    return marker_set


def validate_marker_set(marker_set: dict[str, Any]) -> None:
    markers = marker_set.get("markers")
    if not isinstance(markers, list) or not markers:
        raise ValueError("Marker set must contain a non-empty markers list.")
    seen = set()
    for idx, marker in enumerate(markers):
        if not isinstance(marker, dict):
            raise ValueError(f"Marker definition #{idx} must be a mapping.")
        name = marker.get("name")
        if not name:
            raise ValueError(f"Marker definition #{idx} is missing name.")
        if str(name) in seen:
            raise ValueError(f"Duplicate marker name: {name}")
        seen.add(str(name))
        has_vertex = marker.get("vertex_index") is not None
        has_joint = marker.get("nearest_vertex_to_joint") is not None
        if has_vertex == has_joint:
            raise ValueError(f"Marker {name} must define exactly one of vertex_index or nearest_vertex_to_joint.")
        if has_vertex and int(marker["vertex_index"]) < 0:
            raise ValueError(f"Marker {name} has negative vertex_index.")


def extract_virtual_markers(pose3d: dict, marker_set: dict[str, Any]) -> dict[str, Any]:
    smpl = pose3d.get("smpl") or {}
    vertices = np.asarray(smpl.get("vertices"), dtype=np.float32)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError("SMPL vertices must have shape [T, V, 3].")
    joints = np.asarray(pose3d.get("joints_3d"), dtype=np.float32)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError("pose3d joints_3d must have shape [T, J, 3] for nearest-joint marker resolution.")
    joint_names = [str(name) for name in pose3d.get("joint_names", [])]
    frame_idx = _first_valid_frame(vertices, joints)
    marker_names = []
    vertex_indices = []
    resolved_from = {}
    for marker in marker_set["markers"]:
        name = str(marker["name"])
        marker_names.append(name)
        if marker.get("vertex_index") is not None:
            vertex_idx = int(marker["vertex_index"])
            source = {"type": "vertex_index", "value": vertex_idx}
        else:
            joint_name = str(marker["nearest_vertex_to_joint"])
            vertex_idx = _nearest_vertex_to_joint(vertices[frame_idx], joints[frame_idx], joint_names, joint_name)
            source = {"type": "nearest_vertex_to_joint", "value": joint_name}
        if vertex_idx >= vertices.shape[1]:
            raise ValueError(f"Marker {name} resolved vertex_index {vertex_idx}, but vertices only contain {vertices.shape[1]} vertices.")
        vertex_indices.append(vertex_idx)
        resolved_from[name] = source

    marker_array = vertices[:, np.asarray(vertex_indices, dtype=int), :].astype(np.float32)
    meta = pose3d.get("backend_meta") or {}
    frame_ids = meta.get("frame_ids")
    raw_frame_ids = meta.get("raw_frame_ids")
    if raw_frame_ids is None:
        raw_frame_ids = frame_ids
    raw_video_time_s = meta.get("raw_video_time_s")
    if raw_video_time_s is None and raw_frame_ids is not None:
        raw_video_time_s = np.asarray(raw_frame_ids, dtype=float) / float(pose3d.get("fps") or 30.0)
    wham_relative_time_s = meta.get("wham_relative_time_s")
    if wham_relative_time_s is None and raw_frame_ids is not None:
        raw_arr = np.asarray(raw_frame_ids, dtype=float)
        wham_relative_time_s = (raw_arr - raw_arr[0]) / float(pose3d.get("fps") or 30.0)

    return {
        "marker_set": marker_set["name"],
        "debug": bool(marker_set.get("debug", False)),
        "description": marker_set.get("description"),
        "coordinate_space": marker_set.get("coordinate_space") or smpl.get("vertices_coordinate_space") or "unknown",
        "backend": pose3d.get("backend"),
        "fps": float(pose3d.get("fps") or 30.0),
        "units": "m",
        "marker_names": marker_names,
        "markers_m": marker_array,
        "vertex_indices": vertex_indices,
        "resolved_from": resolved_from,
        "resolution_frame_index": int(frame_idx),
        "frame_ids": frame_ids,
        "raw_frame_ids": raw_frame_ids,
        "raw_video_time_s": raw_video_time_s,
        "wham_relative_time_s": wham_relative_time_s,
        "source_pose_backend_meta": meta,
    }


def apply_marker_time_window(payload: dict[str, Any], pose3d: dict, cfg: dict | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trim time-varying marker arrays to the configured WHAM/sync overlap window."""
    cfg = cfg or {}
    window_cfg = (cfg.get("config", {}).get("markers", {}).get("time_window") or {})
    mode = str(window_cfg.get("mode", "sync_overlap_for_wham"))
    if mode in {"none", "off", "full", "full_sequence"}:
        report = {"status": "skipped", "mode": mode, "reason": "Marker time windowing disabled."}
        payload["time_window"] = report
        return payload, report

    markers = np.asarray(payload.get("markers_m"), dtype=float)
    if markers.ndim != 3 or markers.shape[0] == 0:
        report = {"status": "skipped", "mode": mode, "reason": "Marker payload has no time dimension."}
        payload["time_window"] = report
        return payload, report

    if mode == "manual_raw_frames":
        start = _maybe_int(window_cfg.get("raw_frame_start"))
        end = _maybe_int(window_cfg.get("raw_frame_end"))
        source = "manual_raw_frames"
        timeline = None
    elif mode in {"sync_overlap", "sync_overlap_for_wham"}:
        if mode == "sync_overlap_for_wham" and pose3d.get("backend") != "wham":
            report = {"status": "skipped", "mode": mode, "reason": "Pose backend is not WHAM."}
            payload["time_window"] = report
            return payload, report
        timeline = build_wham_timeline_report(pose3d, cfg)
        overlap = timeline.get("overlap") or {}
        start = _maybe_int(overlap.get("overlap_frame_start"))
        end = _maybe_int(overlap.get("overlap_frame_end"))
        source = "raw_sync_overlap"
    else:
        report = {"status": "skipped", "mode": mode, "reason": f"Unknown marker time-window mode: {mode}"}
        payload["time_window"] = report
        return payload, report

    if start is None or end is None or end < start:
        report = {
            "status": "skipped",
            "mode": mode,
            "source": source,
            "reason": "No valid marker raw-frame window was available.",
        }
        if timeline is not None:
            report["timeline_status"] = timeline.get("status")
            report["timeline_warnings"] = timeline.get("warnings", [])
        payload["time_window"] = report
        return payload, report

    raw_frame_ids = _timebase_array(payload.get("raw_frame_ids"))
    if raw_frame_ids.size == 0:
        raw_frame_ids = _timebase_array(payload.get("frame_ids"))
    if raw_frame_ids.size != markers.shape[0]:
        report = {
            "status": "skipped",
            "mode": mode,
            "source": source,
            "reason": "Marker raw/frame ids are missing or do not match marker frame count.",
            "requested_raw_frame_start": int(start),
            "requested_raw_frame_end": int(end),
        }
        payload["time_window"] = report
        return payload, report

    mask = (raw_frame_ids >= start) & (raw_frame_ids <= end)
    selected = np.flatnonzero(mask)
    min_frames = int(window_cfg.get("min_frames", 2))
    if selected.size < min_frames:
        report = {
            "status": "skipped",
            "mode": mode,
            "source": source,
            "reason": f"Selected marker window has only {int(selected.size)} frame(s), below min_frames={min_frames}.",
            "requested_raw_frame_start": int(start),
            "requested_raw_frame_end": int(end),
            "original_frames": int(markers.shape[0]),
        }
        payload["time_window"] = report
        return payload, report

    trimmed = dict(payload)
    for key in ["markers_m", "frame_ids", "raw_frame_ids", "raw_video_time_s", "wham_relative_time_s"]:
        value = payload.get(key)
        arr = _timebase_array(value) if key != "markers_m" else np.asarray(value)
        if arr.shape[0] == markers.shape[0]:
            trimmed[key] = arr[selected].copy()
    if trimmed.get("wham_relative_time_s") is not None:
        rel = np.asarray(trimmed["wham_relative_time_s"], dtype=float).reshape(-1)
        if rel.size and np.isfinite(rel[0]):
            trimmed["wham_relative_time_s"] = rel - rel[0]

    report = {
        "status": "applied",
        "mode": mode,
        "source": source,
        "requested_raw_frame_start": int(start),
        "requested_raw_frame_end": int(end),
        "raw_frame_start": int(raw_frame_ids[selected[0]]),
        "raw_frame_end": int(raw_frame_ids[selected[-1]]),
        "local_frame_start": int(selected[0]),
        "local_frame_end": int(selected[-1]),
        "original_frames": int(markers.shape[0]),
        "kept_frames": int(selected.size),
        "dropped_before": int(selected[0]),
        "dropped_after": int(markers.shape[0] - selected[-1] - 1),
    }
    if timeline is not None:
        alignment = timeline.get("raw_sync_alignment") or {}
        report["timeline_status"] = timeline.get("status")
        report["sync_raw_frame_start"] = (timeline.get("overlap") or {}).get("sync_raw_frame_start")
        report["sync_raw_frame_end"] = (timeline.get("overlap") or {}).get("sync_raw_frame_end")
        report["raw_sync_match_score"] = alignment.get("best_match_score")
        report["raw_sync_offset"] = alignment.get("best_raw_offset")
        report["timeline_warnings"] = timeline.get("warnings", [])
    trimmed["time_window"] = report
    return trimmed, report


def marker_qc(payload: dict[str, Any]) -> dict[str, Any]:
    markers = np.asarray(payload.get("markers_m"), dtype=float)
    names = [str(name) for name in payload.get("marker_names", [])]
    warnings: list[str] = []
    if markers.ndim != 3 or markers.shape[-1] != 3:
        raise ValueError("markers_m must have shape [T, M, 3].")
    finite = np.isfinite(markers).all(axis=2)
    finite_ratio = float(finite.mean()) if finite.size else 0.0
    if finite_ratio < 1.0:
        warnings.append(f"Marker finite ratio is {finite_ratio:.3f}.")
    ranges = {}
    for idx, name in enumerate(names):
        values = markers[:, idx, :]
        if np.isfinite(values).any():
            ranges[name] = {
                "range_xyz_m": _range(values),
                "trajectory_span_m": float(np.linalg.norm(np.nanmax(values, axis=0) - np.nanmin(values, axis=0))),
            }
        else:
            ranges[name] = {"range_xyz_m": None, "trajectory_span_m": None}
            warnings.append(f"Marker {name} has no finite samples.")

    jumps = np.linalg.norm(np.diff(markers, axis=0), axis=2) if markers.shape[0] > 1 else np.zeros((0, markers.shape[1]))
    max_jump = float(np.nanmax(jumps)) if jumps.size and np.isfinite(jumps).any() else 0.0
    if max_jump > 0.5:
        warnings.append(f"Maximum frame-to-frame marker jump is large: {max_jump:.3f} m.")
    warnings.extend(_left_right_warnings(names, markers))
    return {
        "status": "warning" if warnings else "ok",
        "marker_set": payload.get("marker_set"),
        "debug": bool(payload.get("debug", False)),
        "debug_warning": "Debug SMPL marker set; not final anatomical OpenSim markers." if payload.get("debug") else None,
        "backend": payload.get("backend"),
        "coordinate_space": payload.get("coordinate_space"),
        "units": payload.get("units", "m"),
        "fps": payload.get("fps"),
        "frames": int(markers.shape[0]),
        "markers": int(markers.shape[1]),
        "finite_ratio": finite_ratio,
        "max_frame_to_frame_jump_m": max_jump,
        "vertex_indices": payload.get("vertex_indices"),
        "marker_names": names,
        "marker_ranges": ranges,
        "time_window": payload.get("time_window"),
        "warnings": warnings,
    }


def _nearest_vertex_to_joint(vertices_frame: np.ndarray, joints_frame: np.ndarray, joint_names: list[str], joint_name: str) -> int:
    joint_idx = find_joint(joint_names, [joint_name])
    if joint_idx is None:
        raise ValueError(f"Could not resolve joint '{joint_name}' in pose3d joint_names.")
    joint = joints_frame[joint_idx]
    if not np.isfinite(joint).all():
        raise ValueError(f"Joint '{joint_name}' is not finite in marker resolution frame.")
    valid_vertices = np.isfinite(vertices_frame).all(axis=1)
    if not np.any(valid_vertices):
        raise ValueError("No finite vertices in marker resolution frame.")
    distances = np.full((vertices_frame.shape[0],), np.inf, dtype=float)
    distances[valid_vertices] = np.linalg.norm(vertices_frame[valid_vertices] - joint[None, :], axis=1)
    return int(np.nanargmin(distances))


def _first_valid_frame(vertices: np.ndarray, joints: np.ndarray) -> int:
    frame_count = min(vertices.shape[0], joints.shape[0])
    for idx in range(frame_count):
        if np.isfinite(vertices[idx]).all() and np.isfinite(joints[idx]).any():
            return idx
    raise ValueError("Could not find a valid frame for marker resolution.")


def _range(values: np.ndarray) -> list[float]:
    return [float(v) for v in (np.nanmax(values, axis=0) - np.nanmin(values, axis=0)).tolist()]


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _timebase_array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    try:
        return np.asarray(value).reshape(-1)
    except Exception:
        return np.asarray([])


def _left_right_warnings(names: list[str], markers: np.ndarray) -> list[str]:
    warnings = []
    pairs = [("DBG_LHIP", "DBG_RHIP"), ("DBG_LKNE", "DBG_RKNE"), ("DBG_LANK", "DBG_RANK"), ("DBG_LHEE", "DBG_RHEE"), ("DBG_LTOE", "DBG_RTOE")]
    lookup = {name: idx for idx, name in enumerate(names)}
    for left, right in pairs:
        if left not in lookup or right not in lookup:
            continue
        left_x = np.nanmedian(markers[:, lookup[left], 0])
        right_x = np.nanmedian(markers[:, lookup[right], 0])
        if np.isfinite(left_x) and np.isfinite(right_x) and abs(left_x - right_x) < 1e-4:
            warnings.append(f"Left/right marker pair {left}/{right} has nearly identical median X coordinates.")
    return warnings
