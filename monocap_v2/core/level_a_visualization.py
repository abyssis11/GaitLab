from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from monocap_v2.core.level_a_benchmark import compare_pose_to_opensim_reference, load_opensim_reference, load_pose_artifact, load_run_config
from monocap_v2.core.logging_utils import read_json
from monocap_v2.core.mocap_eval import resample_timeseries


COLORS = {
    "gt": "#111111",
    "metrabs": "#1f77b4",
    "rtmw3d": "#ff7f0e",
    "wham": "#2ca02c",
}

EDGES = [
    ("hip_midpoint", "left_hip"),
    ("hip_midpoint", "right_hip"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_hip", "right_hip"),
]


def render_level_a_overlay(
    benchmark_dir: Path,
    trial: str,
    backends: list[str],
    out_mp4: Path,
    out_png: Path,
    out_json: Path,
    preview_fps: float = 30.0,
    width: int = 2100,
    height: int = 820,
    evaluation_hz: float | None = None,
    wham_convention_profile: str | None = None,
    swap_reference_lr: bool = False,
) -> dict[str, Any]:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    summary = read_json(benchmark_dir / "level_a_summary.json")
    reference = load_opensim_reference(
        benchmark_dir / "reference" / f"opensim_fk_{trial}.npz",
        benchmark_dir / "reference" / f"opensim_fk_{trial}.json",
    )
    series_by_backend: dict[str, dict[str, np.ndarray]] = {}
    reports: dict[str, dict[str, Any]] = {}
    row_lookup = {(row.get("backend"), row.get("trial")): row for row in summary.get("rows", [])}
    for backend in backends:
        row = row_lookup.get((backend, trial))
        if not row or row.get("status") != "valid":
            raise ValueError(f"No valid Level A row for backend={backend!r}, trial={trial!r}.")
        run_dir = Path(str(row["run_dir"]))
        pose = load_pose_artifact(run_dir)
        report, series = compare_pose_to_opensim_reference(
            pose,
            reference,
            run_config=load_run_config(run_dir),
            timeline_report=_cached_wham_timeline(run_dir),
            evaluation_hz=evaluation_hz,
            convention_profile=wham_convention_profile if backend == "wham" else None,
            swap_reference_lr=swap_reference_lr,
        )
        reports[backend] = report
        series_by_backend[backend] = series

    common_time = _common_time(series_by_backend, preview_fps)
    first_series = next(iter(series_by_backend.values()))
    root_name = str(first_series.get("root_name") or "pelvis")
    names = [root_name] + [str(name) for name in first_series["joint_names"].tolist()]
    raw_by_backend = {
        backend: _add_pelvis(resample_timeseries(series["time_s"], series["prediction_root_centered_m"], common_time))
        for backend, series in series_by_backend.items()
    }
    rigid_by_backend = {
        backend: _add_pelvis(resample_timeseries(series["time_s"], series["prediction_root_centered_rigid_m"], common_time))
        for backend, series in series_by_backend.items()
    }
    pa_by_backend = {
        backend: _add_pelvis(resample_timeseries(series["time_s"], series["prediction_pa_root_centered_m"], common_time))
        for backend, series in series_by_backend.items()
    }
    gt = _add_pelvis(resample_timeseries(first_series["time_s"], first_series["reference_root_centered_m"], common_time))
    bounds = _axis_bounds([gt, *raw_by_backend.values(), *rigid_by_backend.values(), *pa_by_backend.values()])
    edge_indices = _edge_indices(names)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    canvas = FigureCanvasAgg(fig)
    axes = [fig.add_subplot(131, projection="3d"), fig.add_subplot(132, projection="3d"), fig.add_subplot(133, projection="3d")]
    writer = None
    video_size = None
    representative_rgb = None
    representative_idx = len(common_time) // 2
    try:
        for idx, time_s in enumerate(common_time):
            _draw_panel(axes[0], gt[idx], {b: raw_by_backend[b][idx] for b in backends}, names, edge_indices, bounds, "Raw root-centered", time_s)
            _draw_panel(
                axes[1],
                gt[idx],
                {b: rigid_by_backend[b][idx] for b in backends},
                names,
                edge_indices,
                bounds,
                "Sequence rigid-aligned",
                time_s,
            )
            _draw_panel(
                axes[2],
                gt[idx],
                {b: pa_by_backend[b][idx] for b in backends},
                names,
                edge_indices,
                bounds,
                "PA-MPJPE per-frame similarity",
                time_s,
            )
            fig.suptitle(f"Level A lower-limb comparison | {trial} | frame {idx:04d}", fontsize=13)
            fig.tight_layout(pad=0.8)
            canvas.draw()
            rgb = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
            if writer is None:
                frame_h, frame_w = rgb.shape[:2]
                video_size = (int(frame_w), int(frame_h))
                writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(preview_fps), video_size)
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer: {out_mp4}")
            elif video_size is not None and (rgb.shape[1], rgb.shape[0]) != video_size:
                rgb = cv2.resize(rgb, video_size, interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if idx == representative_idx:
                representative_rgb = rgb
    finally:
        if writer is not None:
            writer.release()
        plt.close(fig)
    if representative_rgb is not None:
        cv2.imwrite(str(out_png), cv2.cvtColor(representative_rgb, cv2.COLOR_RGB2BGR))

    qc = {
        "status": "ok",
        "trial": trial,
        "backends": backends,
        "frames_rendered": int(common_time.shape[0]),
        "preview_fps": float(preview_fps),
        "evaluation_hz": float(evaluation_hz) if evaluation_hz is not None else None,
        "wham_convention_profile": wham_convention_profile,
        "reference_left_right_swap": bool(swap_reference_lr),
        "frame_size": list(video_size) if video_size is not None else None,
        "time_start_s": float(common_time[0]),
        "time_end_s": float(common_time[-1]),
        "joint_names": names,
        "panels": ["raw_root_centered", "sequence_rigid_aligned", "pa_mpjpe_per_frame_similarity"],
        "metric_note": "Raw panel uses the Level A fixed backend axis maps with no fitted transform; rigid panel fits one sequence-level rotation per backend; PA panel fits one per-frame similarity transform.",
        "metrics": {
            backend: {
                "primary_root_centered_mpjpe_mm": reports[backend].get("primary_root_centered_mpjpe_mm"),
                "root_centered_rigid_mpjpe_mm": reports[backend].get("root_centered_rigid_mpjpe_mm"),
                "pa_mpjpe_mm": reports[backend].get("pa_mpjpe_mm"),
                "warnings": reports[backend].get("warnings"),
            }
            for backend in backends
        },
        "outputs": {"mp4": str(out_mp4), "representative_frame": str(out_png), "qc": str(out_json)},
    }
    return qc


def _draw_panel(ax, gt: np.ndarray, predictions: dict[str, np.ndarray], names: list[str], edges: list[tuple[int, int]], bounds: dict[str, tuple[float, float]], title: str, time_s: float) -> None:
    ax.clear()
    _draw_skeleton(ax, gt, names, edges, COLORS["gt"], "GT")
    for backend, values in predictions.items():
        _draw_skeleton(ax, values, names, edges, COLORS.get(backend, "#777777"), backend)
    ax.set_xlim(*bounds["x"])
    ax.set_ylim(*bounds["z"])
    ax.set_zlim(*bounds["up"])
    ax.set_xlabel("X")
    ax.set_ylabel("Z/display")
    ax.set_zlabel("Up/display")
    ax.view_init(elev=18, azim=-70)
    ax.set_title(f"{title} | t={time_s:.3f}s")
    ax.legend(loc="upper left", fontsize=8)


def _draw_skeleton(ax, values: np.ndarray, names: list[str], edges: list[tuple[int, int]], color: str, label: str) -> None:
    display = _to_display_coords(values)
    finite = np.isfinite(display).all(axis=1)
    if np.any(finite):
        ax.scatter(display[finite, 0], display[finite, 1], display[finite, 2], s=18, color=color, alpha=0.85, depthshade=False, label=label)
    for a, b in edges:
        if finite[a] and finite[b]:
            ax.plot([display[a, 0], display[b, 0]], [display[a, 1], display[b, 1]], [display[a, 2], display[b, 2]], color=color, linewidth=2.0, alpha=0.85)


def _to_display_coords(values: np.ndarray) -> np.ndarray:
    """Display Level A/OpenSim-ish coordinates with component 1 as up."""
    arr = np.asarray(values, dtype=float)
    out = np.empty_like(arr)
    out[..., 0] = arr[..., 0]
    out[..., 1] = arr[..., 2]
    out[..., 2] = arr[..., 1]
    return out


def _add_pelvis(values: np.ndarray) -> np.ndarray:
    pelvis = np.zeros((values.shape[0], 1, 3), dtype=float)
    return np.concatenate([pelvis, np.asarray(values, dtype=float)], axis=1)


def _common_time(series_by_backend: dict[str, dict[str, np.ndarray]], fps: float) -> np.ndarray:
    starts = [float(np.nanmin(series["time_s"])) for series in series_by_backend.values()]
    ends = [float(np.nanmax(series["time_s"])) for series in series_by_backend.values()]
    start = max(starts)
    end = min(ends)
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ValueError("Backend series do not share an overlapping time window.")
    count = max(2, int(np.floor((end - start) * float(fps))) + 1)
    return start + np.arange(count, dtype=float) / float(fps)


def _axis_bounds(arrays: list[np.ndarray]) -> dict[str, tuple[float, float]]:
    display = np.concatenate([_to_display_coords(arr).reshape(-1, 3) for arr in arrays], axis=0)
    finite = display[np.isfinite(display).all(axis=1)]
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


def _edge_indices(names: list[str]) -> list[tuple[int, int]]:
    lookup = {_canon(name): idx for idx, name in enumerate(names)}
    out = []
    for a, b in EDGES:
        ai = lookup.get(_canon(a))
        bi = lookup.get(_canon(b))
        if ai is not None and bi is not None:
            out.append((ai, bi))
    return out


def _cached_wham_timeline(run_dir: Path) -> dict[str, Any] | None:
    qc_path = run_dir / "pose3d_initial" / "pose3d_initial_qc.json"
    if not qc_path.exists():
        return None
    qc = read_json(qc_path)
    timeline = qc.get("wham_timeline") or qc.get("wham_timeline_report")
    if (
        isinstance(timeline, dict)
        and isinstance(timeline.get("raw_sync_alignment"), dict)
        and isinstance(timeline.get("overlap"), dict)
    ):
        return timeline
    # The lower-level wham_timebase QC block is useful for reporting, but
    # timestamp resolution needs the full raw-vs-sync alignment report.
    return None


def _canon(name: str) -> str:
    return name.strip().lower().replace("_", "").replace("-", "").replace(".", "")
