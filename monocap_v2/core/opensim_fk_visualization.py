from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


OPEN_SIM_EDGES = [
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_mtp"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_mtp"),
    ("left_hip", "right_hip"),
]


def load_fk_reference(path: Path | str) -> dict[str, Any]:
    data = np.load(Path(path), allow_pickle=True)
    return {
        "time_s": np.asarray(data["time_s"], dtype=float),
        "joints_m": np.asarray(data["joints_m"], dtype=float),
        "joint_names": [str(name) for name in data["joint_names"].tolist()],
    }


def render_opensim_fk_preview(
    reference: dict[str, Any],
    out_mp4: Path,
    out_png: Path,
    fps: float = 30.0,
    width: int = 1280,
    height: int = 720,
    max_frames: int | None = None,
) -> dict[str, Any]:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    time = np.asarray(reference["time_s"], dtype=float)
    joints = np.asarray(reference["joints_m"], dtype=float)
    names = [str(name) for name in reference["joint_names"]]
    edges = _edge_indices(names)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError("OpenSim FK joints must have shape [T, J, 3].")
    if joints.shape[0] == 0:
        raise ValueError("OpenSim FK reference has no frames.")

    frame_indices = _sample_indices(joints.shape[0], max_frames)
    display = _to_display_coords(joints)
    bounds = _axis_bounds(display)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {out_mp4}")

    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    canvas = FigureCanvasAgg(fig)
    axes = [
        fig.add_subplot(131, projection="3d"),
        fig.add_subplot(132),
        fig.add_subplot(133),
    ]
    representative_rgb = None
    representative_idx = frame_indices[len(frame_indices) // 2]
    try:
        for frame_idx in frame_indices:
            _draw_frame(fig, axes, display[frame_idx], names, edges, bounds, int(frame_idx), float(time[frame_idx]) if frame_idx < len(time) else None)
            fig.tight_layout(pad=0.8)
            canvas.draw()
            rgb = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if frame_idx == representative_idx:
                representative_rgb = rgb
    finally:
        writer.release()
        plt.close(fig)
    if representative_rgb is not None:
        cv2.imwrite(str(out_png), cv2.cvtColor(representative_rgb, cv2.COLOR_RGB2BGR))

    return {
        "status": "ok",
        "frames_total": int(joints.shape[0]),
        "frames_rendered": int(len(frame_indices)),
        "fps": float(fps),
        "joint_names": names,
        "edges": [list(edge) for edge in edges],
        "coordinate_space": "opensim_ground",
        "display_axes": {"x": "OpenSim X", "y": "OpenSim Z", "up": "OpenSim Y"},
        "out_mp4": str(out_mp4),
        "out_png": str(out_png),
    }


def _draw_frame(fig, axes, frame: np.ndarray, names: list[str], edges: list[tuple[int, int]], bounds: dict[str, tuple[float, float]], frame_idx: int, time_s: float | None) -> None:
    for ax in axes:
        ax.clear()
    ax3d, ax_top, ax_side = axes
    finite = np.isfinite(frame).all(axis=1)
    colors = [_joint_color(name) for name in names]

    ax3d.scatter(frame[finite, 0], frame[finite, 1], frame[finite, 2], s=24, c=[colors[i] for i, ok in enumerate(finite) if ok], depthshade=False)
    for a, b in edges:
        if finite[a] and finite[b]:
            color = _edge_color(names[a], names[b])
            ax3d.plot([frame[a, 0], frame[b, 0]], [frame[a, 1], frame[b, 1]], [frame[a, 2], frame[b, 2]], color=color, linewidth=2.2)
            ax_top.plot([frame[a, 0], frame[b, 0]], [frame[a, 1], frame[b, 1]], color=color, linewidth=2.0)
            ax_side.plot([frame[a, 0], frame[b, 0]], [frame[a, 2], frame[b, 2]], color=color, linewidth=2.0)
    for idx, name in enumerate(names):
        if not finite[idx]:
            continue
        short = _short_name(name)
        ax3d.text(frame[idx, 0], frame[idx, 1], frame[idx, 2], short, fontsize=7)
        ax_top.text(frame[idx, 0], frame[idx, 1], short, fontsize=7)
        ax_side.text(frame[idx, 0], frame[idx, 2], short, fontsize=7)

    ax3d.set_xlim(*bounds["x"])
    ax3d.set_ylim(*bounds["z"])
    ax3d.set_zlim(*bounds["up"])
    ax3d.set_xlabel("OpenSim X (m)")
    ax3d.set_ylabel("OpenSim Z (m)")
    ax3d.set_zlabel("OpenSim Y up (m)")
    ax3d.view_init(elev=18, azim=-70)
    ax3d.set_title("3D")

    ax_top.set_xlim(*bounds["x"])
    ax_top.set_ylim(*bounds["z"])
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.set_xlabel("OpenSim X (m)")
    ax_top.set_ylabel("OpenSim Z (m)")
    ax_top.set_title("Top view")

    ax_side.set_xlim(*bounds["x"])
    ax_side.set_ylim(*bounds["up"])
    ax_side.set_aspect("equal", adjustable="box")
    ax_side.set_xlabel("OpenSim X (m)")
    ax_side.set_ylabel("OpenSim Y up (m)")
    ax_side.set_title("Side view")

    title_time = "" if time_s is None else f" | t={time_s:.3f}s"
    fig.suptitle(f"OpenSim FK joint centers | frame {frame_idx:04d}{title_time}", fontsize=12)


def _to_display_coords(joints: np.ndarray) -> np.ndarray:
    """Display OpenSim ground as horizontal X/Z and vertical Y."""
    out = np.empty_like(joints, dtype=float)
    out[..., 0] = joints[..., 0]
    out[..., 1] = joints[..., 2]
    out[..., 2] = joints[..., 1]
    return out


def _axis_bounds(display: np.ndarray) -> dict[str, tuple[float, float]]:
    finite = display[np.isfinite(display).all(axis=2)]
    if finite.size == 0:
        return {"x": (-1, 1), "z": (-1, 1), "up": (-1, 1)}
    mins = np.nanpercentile(finite, 2, axis=0)
    maxs = np.nanpercentile(finite, 98, axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.nanmax(maxs - mins) / 2.0), 0.6)
    return {
        "x": (float(center[0] - radius), float(center[0] + radius)),
        "z": (float(center[1] - radius), float(center[1] + radius)),
        "up": (float(center[2] - radius), float(center[2] + radius)),
    }


def _sample_indices(frame_count: int, max_frames: int | None) -> np.ndarray:
    if max_frames is None or max_frames <= 0 or frame_count <= max_frames:
        return np.arange(frame_count, dtype=int)
    return np.unique(np.linspace(0, frame_count - 1, int(max_frames)).round().astype(int))


def _edge_indices(names: list[str]) -> list[tuple[int, int]]:
    lookup = {_canon(name): idx for idx, name in enumerate(names)}
    out = []
    for a_name, b_name in OPEN_SIM_EDGES:
        a = lookup.get(_canon(a_name))
        b = lookup.get(_canon(b_name))
        if a is not None and b is not None:
            out.append((a, b))
    return out


def _joint_color(name: str) -> str:
    n = name.lower()
    if "left" in n:
        return "#1f77b4"
    if "right" in n:
        return "#d62728"
    return "#2ca02c"


def _edge_color(a: str, b: str) -> str:
    if "left" in a.lower() or "left" in b.lower():
        return "#1f77b4"
    if "right" in a.lower() or "right" in b.lower():
        return "#d62728"
    return "#333333"


def _short_name(name: str) -> str:
    return (
        name.replace("left_", "L_")
        .replace("right_", "R_")
        .replace("pelvis", "PELV")
        .replace("_", "")
        .upper()
    )


def _canon(name: str) -> str:
    return name.strip().lower().replace("_", "").replace("-", "").replace(".", "")
