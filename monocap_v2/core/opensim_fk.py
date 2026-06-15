from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from monocap_v2.core.logging_utils import write_json


REFERENCE_JOINT_QUERIES = [
    {"name": "pelvis", "kind": "body", "opensim_name": "pelvis"},
    {"name": "left_hip", "kind": "joint_child", "opensim_name": "hip_l"},
    {"name": "right_hip", "kind": "joint_child", "opensim_name": "hip_r"},
    {"name": "left_knee", "kind": "joint_child", "opensim_name": "walker_knee_l"},
    {"name": "right_knee", "kind": "joint_child", "opensim_name": "walker_knee_r"},
    {"name": "left_ankle", "kind": "joint_child", "opensim_name": "ankle_l"},
    {"name": "right_ankle", "kind": "joint_child", "opensim_name": "ankle_r"},
    {"name": "left_mtp", "kind": "joint_child", "opensim_name": "mtp_l"},
    {"name": "right_mtp", "kind": "joint_child", "opensim_name": "mtp_r"},
]


@dataclass(frozen=True)
class StorageTable:
    time: np.ndarray
    columns: list[str]
    data: np.ndarray
    in_degrees: bool | None
    path: Path


def parse_storage_table(path: Path | str) -> StorageTable:
    """Parse OpenSim .mot/.sto storage files into a numeric table."""
    storage_path = Path(path)
    lines = storage_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_degrees = _parse_in_degrees(lines)
    header_idx = _find_storage_header(lines)
    columns = _split_row(lines[header_idx])
    if not columns or columns[0].lower() != "time":
        raise ValueError(f"OpenSim storage header must start with time: {storage_path}")

    rows: list[list[float]] = []
    for line in lines[header_idx + 1 :]:
        parts = _split_row(line)
        if len(parts) < len(columns):
            continue
        try:
            rows.append([float(value) for value in parts[: len(columns)]])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"No numeric rows found in OpenSim storage file: {storage_path}")

    data = np.asarray(rows, dtype=float)
    return StorageTable(time=data[:, 0], columns=columns[1:], data=data[:, 1:], in_degrees=in_degrees, path=storage_path)


def read_ik_marker_error_summary(path: Path | str | None) -> dict[str, Any]:
    if not path:
        return {"status": "skipped", "reason": "IK marker error file is unavailable."}
    error_path = Path(path)
    if not error_path.exists():
        return {"status": "missing", "source": str(error_path), "reason": "IK marker error file does not exist."}
    try:
        table = parse_storage_table(error_path)
    except Exception as exc:
        return {"status": "failed", "source": str(error_path), "error": str(exc)}
    lookup = {name: idx for idx, name in enumerate(table.columns)}
    out: dict[str, Any] = {"status": "ok", "source": str(error_path), "frames": int(table.time.shape[0])}
    for col, label in [("marker_error_RMS", "rms"), ("marker_error_max", "max")]:
        idx = lookup.get(col)
        if idx is None:
            continue
        values_m = table.data[:, idx]
        finite = values_m[np.isfinite(values_m)]
        if finite.size:
            out[f"{label}_median_mm"] = float(np.nanmedian(finite) * 1000.0)
            out[f"{label}_mean_mm"] = float(np.nanmean(finite) * 1000.0)
            out[f"{label}_p95_mm"] = float(np.nanpercentile(finite, 95) * 1000.0)
    return out


def export_opensim_fk_reference(
    model_path: Path | str,
    ik_mot_path: Path | str,
    out_npz: Path | str,
    out_json: Path | str,
    marker_errors_path: Path | str | None = None,
) -> dict[str, Any]:
    """Export mocap-derived OpenSim joint centers by evaluating IK coordinates."""
    import opensim as osim  # type: ignore

    model_path = Path(model_path)
    ik_mot_path = Path(ik_mot_path)
    out_npz = Path(out_npz)
    out_json = Path(out_json)
    table = parse_storage_table(ik_mot_path)

    model = osim.Model(str(model_path))
    state = model.initSystem()
    coordinates = _coordinate_lookup(model)
    joint_getters, missing_queries = _reference_joint_getters(model)

    joints = np.full((table.time.shape[0], len(REFERENCE_JOINT_QUERIES), 3), np.nan, dtype=np.float64)
    coordinate_columns = {name: idx for idx, name in enumerate(table.columns) if name in coordinates}
    for row_idx, row in enumerate(table.data):
        for name, col_idx in coordinate_columns.items():
            value = float(row[col_idx])
            coord = coordinates[name]
            if table.in_degrees and _coordinate_is_rotational(coord, name):
                value = np.deg2rad(value)
            coord.setValue(state, value, False)
        model.realizePosition(state)
        for joint_idx, getter in enumerate(joint_getters):
            if getter is None:
                continue
            joints[row_idx, joint_idx] = _vec3_to_array(getter(state))

    joint_names = [query["name"] for query in REFERENCE_JOINT_QUERIES]
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, time_s=table.time.astype(np.float64), joints_m=joints.astype(np.float32), joint_names=np.asarray(joint_names))
    marker_error_summary = read_ik_marker_error_summary(marker_errors_path)
    metadata = {
        "status": "ok" if not missing_queries else "warning",
        "model": str(model_path),
        "ik_mot": str(ik_mot_path),
        "out_npz": str(out_npz),
        "joint_names": joint_names,
        "frames": int(joints.shape[0]),
        "time_start_s": float(table.time[0]),
        "time_end_s": float(table.time[-1]),
        "in_degrees": table.in_degrees,
        "coordinate_count": len(coordinate_columns),
        "missing_reference_queries": missing_queries,
        "ik_marker_errors": marker_error_summary,
    }
    write_json(out_json, metadata)
    return metadata


def _parse_in_degrees(lines: list[str]) -> bool | None:
    for line in lines[:100]:
        match = re.match(r"\s*inDegrees\s*=\s*(yes|no|true|false|0|1)\s*$", line, re.I)
        if not match:
            continue
        value = match.group(1).lower()
        return value in {"yes", "true", "1"}
    return None


def _find_storage_header(lines: list[str]) -> int:
    for idx, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            for candidate in range(idx + 1, len(lines)):
                parts = _split_row(lines[candidate])
                if parts and parts[0].lower() == "time":
                    return candidate
    for idx, line in enumerate(lines[:300]):
        parts = _split_row(line)
        if parts and parts[0].lower() == "time":
            return idx
    raise ValueError("Could not find OpenSim storage column header.")


def _split_row(line: str) -> list[str]:
    return [part.strip() for part in re.split(r"[\s,\t]+", line.strip()) if part.strip()]


def _coordinate_lookup(model) -> dict[str, Any]:
    coord_set = model.getCoordinateSet()
    out = {}
    for idx in range(coord_set.getSize()):
        coord = coord_set.get(idx)
        out[str(coord.getName())] = coord
    return out


def _reference_joint_getters(model) -> tuple[list[Any | None], list[dict[str, str]]]:
    body_set = model.getBodySet()
    joint_set = model.getJointSet()
    getters = []
    missing = []
    for query in REFERENCE_JOINT_QUERIES:
        name = str(query["opensim_name"])
        try:
            if query["kind"] == "body":
                body = body_set.get(name)
                getters.append(lambda state, body=body: body.getPositionInGround(state))
            else:
                joint = joint_set.get(name)
                frame = joint.getChildFrame()
                getters.append(lambda state, frame=frame: frame.getPositionInGround(state))
        except Exception:
            getters.append(None)
            missing.append({"name": str(query["name"]), "opensim_name": name, "kind": str(query["kind"])})
    return getters, missing


def _coordinate_is_rotational(coord, name: str) -> bool:
    try:
        motion_type = str(coord.getMotionType()).lower()
        if "trans" in motion_type:
            return False
        if "rot" in motion_type:
            return True
    except Exception:
        pass
    return not name.endswith(("_tx", "_ty", "_tz"))


def _vec3_to_array(vec) -> np.ndarray:
    try:
        return np.asarray([float(vec.get(0)), float(vec.get(1)), float(vec.get(2))], dtype=float)
    except Exception:
        return np.asarray([float(vec[0]), float(vec[1]), float(vec[2])], dtype=float)
