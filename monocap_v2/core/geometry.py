from __future__ import annotations

from typing import Iterable

import numpy as np


SKELETON_EDGES = [
    ("pelv", "lhip"),
    ("lhip", "lkne"),
    ("lkne", "lank"),
    ("lank", "ltoe"),
    ("pelv", "rhip"),
    ("rhip", "rkne"),
    ("rkne", "rank"),
    ("rank", "rtoe"),
    ("pelv", "spi1"),
    ("spi1", "spi2"),
    ("spi2", "spi3"),
    ("spi3", "neck"),
    ("neck", "head"),
    ("neck", "lcla"),
    ("lcla", "lsho"),
    ("lsho", "lelb"),
    ("lelb", "lwri"),
    ("lwri", "lhan"),
    ("neck", "rcla"),
    ("rcla", "rsho"),
    ("rsho", "relb"),
    ("relb", "rwri"),
    ("rwri", "rhan"),
]


def project_points(points_m, camera: dict) -> np.ndarray:
    points = np.asarray(points_m, dtype=float)
    z = points[..., 2]
    out = np.full(points.shape[:-1] + (2,), np.nan, dtype=float)
    valid = np.isfinite(points).all(axis=-1) & (np.abs(z) > 1e-9)
    out[..., 0] = np.where(valid, float(camera["fx"]) * points[..., 0] / z + float(camera["cx"]), np.nan)
    out[..., 1] = np.where(valid, float(camera["fy"]) * points[..., 1] / z + float(camera["cy"]), np.nan)
    return out


def camera_to_eval_coords(points_m) -> np.ndarray:
    """Convert camera coordinates with +Y down to evaluation coordinates with +Z up."""
    points = np.asarray(points_m, dtype=float)
    out = np.empty_like(points, dtype=float)
    out[..., 0] = points[..., 0]
    out[..., 1] = points[..., 2]
    out[..., 2] = -points[..., 1]
    return out


def apply_axis_expr(arr: np.ndarray, expr: str) -> np.ndarray:
    expr = expr.strip().lower()
    parts = [p.strip() for p in expr.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Axis expression must have 3 components, got {expr!r}")
    basis = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    matrix = np.zeros((3, 3), dtype=float)
    used = set()
    for row, token in enumerate(parts):
        sign = -1.0 if token.startswith("-") else 1.0
        axis = token.lstrip("+-")
        if axis not in basis:
            raise ValueError(f"Invalid axis token {token!r}")
        if axis in used:
            raise ValueError(f"Axis {axis!r} used more than once")
        used.add(axis)
        matrix[row, :] = sign * basis[axis]
    flat = np.asarray(arr, dtype=float).reshape(-1, 3) @ matrix.T
    return flat.reshape(np.asarray(arr).shape)


def edge_indices(joint_names: Iterable[str], edges: Iterable[tuple[str, str]] = SKELETON_EDGES) -> list[tuple[int, int]]:
    aliases = joint_aliases(joint_names)
    out = []
    for a_name, b_name in edges:
        a = aliases.get(_canon(a_name))
        b = aliases.get(_canon(b_name))
        if a is not None and b is not None:
            out.append((a, b))
    return out


def joint_aliases(joint_names: Iterable[str]) -> dict[str, int]:
    aliases: dict[str, int] = {}
    for idx, name in enumerate(joint_names):
        norm = _canon(str(name))
        aliases[norm] = idx
        if norm.startswith("left"):
            aliases["l" + norm[4:]] = idx
        if norm.startswith("right"):
            aliases["r" + norm[5:]] = idx
        if norm in {"leftbigtoe", "lefttoe"}:
            aliases["ltoe"] = idx
        if norm in {"rightbigtoe", "righttoe"}:
            aliases["rtoe"] = idx
        if norm in {"leftheel"}:
            aliases["lheel"] = idx
        if norm in {"rightheel"}:
            aliases["rheel"] = idx
        if norm == "pelvis":
            aliases["pelv"] = idx
    return aliases


def find_joint(joint_names: Iterable[str], candidates: Iterable[str]) -> int | None:
    aliases = joint_aliases(joint_names)
    for candidate in candidates:
        idx = aliases.get(_canon(candidate))
        if idx is not None:
            return idx
    names = [_canon(str(n)) for n in joint_names]
    for candidate in candidates:
        c = _canon(candidate)
        for idx, name in enumerate(names):
            if c in name:
                return idx
    return None


def kabsch_align(pred: np.ndarray, ref: np.ndarray, mode: str = "rigid") -> tuple[np.ndarray, np.ndarray, float]:
    if mode not in {"rigid", "similarity"}:
        raise ValueError("mode must be 'rigid' or 'similarity'")
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if pred.shape[0] < 3:
        return np.eye(3), np.zeros(3), 1.0
    pred_mean = pred.mean(axis=0)
    ref_mean = ref.mean(axis=0)
    pred_centered = pred - pred_mean
    ref_centered = ref - ref_mean
    u, s, vt = np.linalg.svd(pred_centered.T @ ref_centered)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T
    scale = 1.0
    if mode == "similarity":
        scale = float(np.sum(s) / (np.sum(pred_centered**2) + 1e-12))
    trans = ref_mean - scale * (rot @ pred_mean)
    return rot, trans, scale


def apply_similarity(points: np.ndarray, rot: np.ndarray, trans: np.ndarray, scale: float = 1.0) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    out = points.reshape(-1, 3)
    out = scale * (out @ rot.T) + trans
    return out.reshape(points.shape)


def _canon(name: str) -> str:
    return name.strip().lower().replace("_", "").replace("-", "").replace(".", "")
