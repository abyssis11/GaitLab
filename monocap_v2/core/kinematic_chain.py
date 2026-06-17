from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from monocap_v2.core.geometry import edge_indices, find_joint
from monocap_v2.core.temporal_smoothing import (
    EdgeLength,
    estimate_edge_lengths,
    fill_nan_linear,
    limit_joint_displacement,
    smooth_joint_sequence,
    temporal_smoothing_metrics,
)


@dataclass(frozen=True)
class ChainEdge:
    parent: int
    child: int
    target_m: float


def apply_kinematic_chain_to_pose(
    pose: dict[str, Any],
    cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refine joints through a fixed-length skeleton tree.

    This is a joints-only diagnostic pass. It does not use mocap/OpenSim data.
    Instead of optimizing/editing free joint coordinates, it smooths root joints
    and parent-child unit directions, then reconstructs children by forward
    kinematics with fixed median segment lengths.
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
        report["reason"] = f"Kinematic-chain refinement only applies to joints artifacts, got {representation!r}."
        return out_pose, report

    joints = np.asarray(pose.get("joints_3d"), dtype=float)
    names = [str(name) for name in pose.get("joint_names", [])]
    if joints.ndim != 3 or joints.shape[-1] != 3 or joints.shape[0] < 3 or not names:
        report["reason"] = "Pose artifact has no valid [T, J, 3] sequence with at least 3 frames."
        return out_pose, report

    filled, fill_report = fill_nan_linear(joints)
    edges = build_chain_edges(filled, names)
    if not edges:
        report["reason"] = "No usable skeleton edges were found for this joint set."
        return out_pose, report

    method = str(cfg.get("method") or "moving_average")
    window = _odd_window(int(cfg.get("window_frames", 9)), filled.shape[0])
    direction_window = _odd_window(int(cfg.get("direction_window_frames", window)), filled.shape[0])
    root_window = _odd_window(int(cfg.get("root_window_frames", window)), filled.shape[0])
    smooth_roots = bool(cfg.get("smooth_roots", True))
    direction_blend = float(np.clip(float(cfg.get("direction_blend", 1.0)), 0.0, 1.0))
    output_blend = float(np.clip(float(cfg.get("output_blend", 1.0)), 0.0, 1.0))
    max_displacement_m = _finite_positive_or_none(cfg.get("max_joint_displacement_m"))

    root_indices = chain_root_indices(filled.shape[1], edges)
    roots = smooth_roots_sequence(
        filled,
        root_indices,
        method=method,
        window_frames=root_window,
        polyorder=int(cfg.get("savgol_polyorder", 2)),
        enabled=smooth_roots,
    )
    directions = smooth_edge_directions(
        filled,
        edges,
        method=method,
        window_frames=direction_window,
        polyorder=int(cfg.get("savgol_polyorder", 2)),
        blend=direction_blend,
    )
    reconstructed = reconstruct_from_chain(filled, edges, directions, roots, root_indices)
    if output_blend < 1.0:
        reconstructed = filled + output_blend * (reconstructed - filled)
    if max_displacement_m is not None:
        reconstructed = limit_joint_displacement(reconstructed, filled, max_displacement_m)
    reconstructed[~np.isfinite(joints)] = np.nan

    target_lengths = [EdgeLength(edge.parent, edge.child, edge.target_m) for edge in edges]
    before_metrics = temporal_smoothing_metrics(filled, names, fps=float(pose.get("fps") or 30.0), target_lengths=target_lengths)
    after_metrics = temporal_smoothing_metrics(reconstructed, names, fps=float(pose.get("fps") or 30.0), target_lengths=target_lengths)
    after_metrics["mean_joint_displacement_m"] = _mean_joint_displacement(reconstructed, filled)
    after_metrics["max_joint_displacement_m"] = _max_joint_displacement(reconstructed, filled)

    out_pose["joints_3d"] = reconstructed.astype(np.float32)
    chain_report = {
        "status": "ok",
        "method": "kinematic_chain_direction_smoothing",
        "smoothing_method": method,
        "window_frames": int(window),
        "direction_window_frames": int(direction_window),
        "root_window_frames": int(root_window),
        "smooth_roots": smooth_roots,
        "direction_blend": direction_blend,
        "output_blend": output_blend,
        "max_joint_displacement_m": max_displacement_m,
        "edge_count": len(edges),
        "root_indices": [int(idx) for idx in root_indices],
        "root_names": [names[idx] for idx in root_indices],
        "nan_fill": fill_report,
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
        "mocap_used_in_objective": False,
    }
    out_pose["kinematic_chain"] = chain_report
    return out_pose, chain_report


def build_chain_edges(joints: np.ndarray, joint_names: list[str]) -> list[ChainEdge]:
    values = np.asarray(joints, dtype=float)
    out: list[ChainEdge] = []
    for parent, child in edge_indices(joint_names):
        lengths = np.linalg.norm(values[:, child, :] - values[:, parent, :], axis=1)
        finite = np.isfinite(lengths) & (lengths > 1e-9)
        if np.any(finite):
            out.append(ChainEdge(parent=int(parent), child=int(child), target_m=float(np.nanmedian(lengths[finite]))))
    return _ordered_edges(out, joint_names)


def chain_root_indices(n_joints: int, edges: list[ChainEdge]) -> list[int]:
    parents = {edge.parent for edge in edges}
    children = {edge.child for edge in edges}
    roots = sorted(parents - children)
    if roots:
        return roots
    return [0] if n_joints else []


def smooth_roots_sequence(
    joints: np.ndarray,
    root_indices: list[int],
    method: str,
    window_frames: int,
    polyorder: int = 2,
    enabled: bool = True,
) -> np.ndarray:
    roots = np.asarray(joints, dtype=float)[:, root_indices, :].copy()
    if not enabled or roots.shape[0] < 3:
        return roots
    return smooth_joint_sequence(roots, method=method, window_frames=window_frames, polyorder=polyorder)


def smooth_edge_directions(
    joints: np.ndarray,
    edges: list[ChainEdge],
    method: str,
    window_frames: int,
    polyorder: int = 2,
    blend: float = 1.0,
) -> dict[tuple[int, int], np.ndarray]:
    values = np.asarray(joints, dtype=float)
    out: dict[tuple[int, int], np.ndarray] = {}
    blend = float(np.clip(blend, 0.0, 1.0))
    for edge in edges:
        raw = values[:, edge.child, :] - values[:, edge.parent, :]
        raw_unit = _normalize_vectors(raw)
        smoothed = smooth_joint_sequence(raw_unit[:, None, :], method=method, window_frames=window_frames, polyorder=polyorder)[:, 0, :]
        blended = _normalize_vectors(raw_unit + blend * (smoothed - raw_unit))
        out[(edge.parent, edge.child)] = blended
    return out


def reconstruct_from_chain(
    reference: np.ndarray,
    edges: list[ChainEdge],
    directions: dict[tuple[int, int], np.ndarray],
    roots: np.ndarray,
    root_indices: list[int],
) -> np.ndarray:
    ref = np.asarray(reference, dtype=float)
    out = ref.copy()
    root_lookup = {idx: pos for pos, idx in enumerate(root_indices)}
    for idx, pos in root_lookup.items():
        out[:, idx, :] = roots[:, pos, :]
    for edge in edges:
        direction = directions[(edge.parent, edge.child)]
        parent = out[:, edge.parent, :]
        fallback = ref[:, edge.child, :]
        child = parent + edge.target_m * direction
        finite = np.isfinite(parent).all(axis=1) & np.isfinite(direction).all(axis=1)
        out[finite, edge.child, :] = child[finite]
        out[~finite, edge.child, :] = fallback[~finite]
    return out


def _ordered_edges(edges: list[ChainEdge], joint_names: list[str]) -> list[ChainEdge]:
    if not edges:
        return []
    remaining = list(edges)
    ordered: list[ChainEdge] = []
    known = set(chain_root_indices(len(joint_names), edges))
    # Prefer pelvis/hip/shoulder roots when available, but keep disconnected
    # components such as COCO left/right legs usable.
    pelvis = find_joint(joint_names, ("pelvis", "pelv", "root"))
    if pelvis is not None:
        known.add(pelvis)
    while remaining:
        progressed = False
        for edge in list(remaining):
            if edge.parent in known:
                ordered.append(edge)
                known.add(edge.child)
                remaining.remove(edge)
                progressed = True
        if not progressed:
            edge = remaining.pop(0)
            ordered.append(edge)
            known.add(edge.parent)
            known.add(edge.child)
    return ordered


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 2 or values.shape[-1] != 3:
        raise ValueError(f"Expected [T, 3] direction vectors, got {values.shape}.")
    norms = np.linalg.norm(values, axis=-1)
    valid = np.isfinite(norms) & (norms > 1e-12) & np.isfinite(values).all(axis=-1)
    out = np.zeros_like(values, dtype=float)
    if np.any(valid):
        out[valid] = values[valid] / norms[valid, None]
    idx = np.arange(values.shape[0], dtype=float)
    for c in range(3):
        series = out[:, c]
        if np.count_nonzero(valid) == 0:
            series[:] = 0.0
        elif np.count_nonzero(valid) == 1:
            series[~valid] = series[valid][0]
        else:
            series[~valid] = np.interp(idx[~valid], idx[valid], series[valid])
    norms = np.linalg.norm(out, axis=-1)
    valid = np.isfinite(norms) & (norms > 1e-12)
    out[valid] = out[valid] / norms[valid, None]
    return out


def _odd_window(window: int, n_frames: int) -> int:
    if n_frames < 3:
        return 1
    window = max(3, int(window))
    window = min(window, n_frames if n_frames % 2 == 1 else n_frames - 1)
    if window % 2 == 0:
        window -= 1
    return max(3, window)


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
