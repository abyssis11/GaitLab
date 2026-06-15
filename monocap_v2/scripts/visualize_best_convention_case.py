#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monocap_v2.core.backend_convention_audit import parse_benchmark_dirs, relabel_pose_left_right
from monocap_v2.core.level_a_benchmark import (
    compare_pose_to_opensim_reference,
    load_cached_wham_timeline,
    load_opensim_reference,
    load_pose_artifact,
    load_run_config,
)
from monocap_v2.core.level_a_visualization import (
    COLORS,
    _add_pelvis,
    _axis_bounds,
    _draw_skeleton,
    _edge_indices,
)
from monocap_v2.core.logging_utils import read_json, write_json
from monocap_v2.core.mocap_eval import resample_timeseries


DEFAULT_SUMMARY = Path("monocap_v2/benchmarks/cam0_cam1_walking_convention_summary/best_pa_by_case.csv")
DEFAULT_OUT_DIR = Path("monocap_v2/benchmarks/cam0_cam1_walking_convention_summary/visualizations")
DEFAULT_BENCHMARK_DIRS = (
    "Cam1=monocap_v2/benchmarks/subject7_walking_level_a,"
    "Cam0=monocap_v2/benchmarks/subject7_walking_level_a_cam0"
)
BACKENDS = ["wham", "metrabs", "rtmw3d"]
PANEL_KEYS = [
    ("prediction_root_centered_m", "Raw root-centered", "raw_primary_mm"),
    ("prediction_root_centered_rigid_m", "Sequence rigid", "rigid_mm"),
    ("prediction_pa_root_centered_m", "PA per-frame similarity", "pa_mm"),
]
PA_PANEL_KEYS = [("prediction_pa_root_centered_m", "PA per-frame similarity", "pa_mm")]


def main() -> int:
    args = parse_args()
    rows = _load_best_rows(args.summary_csv)
    selected = _select_case_rows(rows, args.camera, args.trial, args.evaluation_hz, args.backends.split(","))
    benchmark_dirs = parse_benchmark_dirs(args.benchmark_dirs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    camera = selected[0]["camera"]
    trial = selected[0]["trial"]
    hz = float(selected[0]["evaluation_hz"])
    if args.single_best:
        selected = [min(selected, key=lambda row: float(row["pa_mm"]))]
        suffix = f"best_global_{camera}_{trial}_{selected[0]['backend']}_{hz:g}hz"
    else:
        suffix = f"best_case_{camera}_{trial}_{hz:g}hz_all_backends"
    if args.panel_set == "pa":
        suffix = f"{suffix}_pa_only"

    out_mp4 = args.out_mp4 or out_dir / f"{suffix}.mp4"
    out_png = args.out_png or out_dir / f"{suffix}.png"
    out_json = args.out_json or out_dir / f"{suffix}.json"

    report = render_candidate_grid(
        selected_rows=selected,
        benchmark_dirs=benchmark_dirs,
        out_mp4=out_mp4,
        out_png=out_png,
        preview_fps=args.fps,
        width=args.width,
        row_height=args.row_height,
        panel_keys=PA_PANEL_KEYS if args.panel_set == "pa" else PANEL_KEYS,
    )
    write_json(out_json, {**report, "outputs": {"mp4": str(out_mp4), "png": str(out_png), "json": str(out_json)}})
    print(f"[best-convention-vis] {report['status']} -> {out_mp4}", flush=True)
    print(f"[best-convention-vis] frame -> {out_png}", flush=True)
    print(f"[best-convention-vis] qc -> {out_json}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize exact best convention-audit candidates.")
    ap.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--benchmark-dirs", default=DEFAULT_BENCHMARK_DIRS)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--trial", default=None)
    ap.add_argument("--evaluation-hz", type=float, default=None)
    ap.add_argument("--backends", default="wham,metrabs,rtmw3d")
    ap.add_argument("--single-best", action="store_true", help="Render only the lowest-PA backend row for the selected case.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--out-mp4", type=Path, default=None)
    ap.add_argument("--out-png", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=2100)
    ap.add_argument("--row-height", type=int, default=360)
    ap.add_argument("--panel-set", choices=["all", "pa"], default="all")
    return ap.parse_args()


def render_candidate_grid(
    selected_rows: list[dict[str, Any]],
    benchmark_dirs: dict[str, Path],
    out_mp4: Path,
    out_png: Path,
    preview_fps: float,
    width: int,
    row_height: int,
    panel_keys: list[tuple[str, str, str]],
) -> dict[str, Any]:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if not selected_rows:
        raise ValueError("No selected rows to render.")
    camera = selected_rows[0]["camera"]
    trial = selected_rows[0]["trial"]
    hz = float(selected_rows[0]["evaluation_hz"])
    benchmark_dir = Path(benchmark_dirs[camera])
    reference = load_opensim_reference(
        benchmark_dir / "reference" / f"opensim_fk_{trial}.npz",
        benchmark_dir / "reference" / f"opensim_fk_{trial}.json",
    )
    level_a_summary = read_json(benchmark_dir / "level_a_summary.json")
    run_lookup = {(row.get("backend"), row.get("trial")): Path(str(row["run_dir"])) for row in level_a_summary.get("rows", []) if row.get("status") == "valid"}

    items = []
    for row in selected_rows:
        backend = row["backend"]
        run_dir = run_lookup[(backend, trial)]
        pose = load_pose_artifact(run_dir)
        if _bool(row.get("model_left_right_swap")):
            pose = relabel_pose_left_right(pose)
        report, series = compare_pose_to_opensim_reference(
            pose,
            reference,
            run_config=load_run_config(run_dir),
            axis_map={backend: str(row["axis"])},
            evaluation_hz=hz,
            diagnostic_time_offset_s=float(row["time_offset_s"]),
            swap_reference_lr=_bool(row.get("reference_left_right_swap")),
            timeline_report=load_cached_wham_timeline(run_dir),
        )
        items.append({"row": row, "report": report, "series": series})

    common_time = _common_time([item["series"] for item in items], preview_fps)
    root_name = str(items[0]["series"].get("root_name") or "pelvis")
    joint_names = [root_name] + [str(name) for name in items[0]["series"]["joint_names"].tolist()]
    edge_indices = _edge_indices(joint_names)

    prepared = []
    bounds_arrays = []
    for item in items:
        series = item["series"]
        gt = _add_pelvis(resample_timeseries(series["time_s"], series["reference_root_centered_m"], common_time))
        panels = {}
        for key, _title, _metric in panel_keys:
            panels[key] = _add_pelvis(resample_timeseries(series["time_s"], series[key], common_time))
        prepared.append({**item, "gt": gt, "panels": panels})
        bounds_arrays.extend([gt, *panels.values()])
    bounds = _axis_bounds(bounds_arrays)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    height = max(360, int(row_height) * len(prepared))
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    canvas = FigureCanvasAgg(fig)
    col_count = len(panel_keys)
    axes = np.asarray(
        [fig.add_subplot(len(prepared), col_count, idx + 1, projection="3d") for idx in range(len(prepared) * col_count)],
        dtype=object,
    ).reshape(len(prepared), col_count)
    writer = None
    video_size = None
    representative_rgb = None
    representative_idx = len(common_time) // 2
    try:
        for frame_idx, time_s in enumerate(common_time):
            for row_idx, item in enumerate(prepared):
                backend = item["row"]["backend"]
                for col_idx, (key, title, metric_key) in enumerate(panel_keys):
                    ax = axes[row_idx, col_idx]
                    ax.clear()
                    _draw_skeleton(ax, item["gt"][frame_idx], joint_names, edge_indices, COLORS["gt"], "GT")
                    _draw_skeleton(
                        ax,
                        item["panels"][key][frame_idx],
                        joint_names,
                        edge_indices,
                        COLORS.get(backend, "#777777"),
                        backend,
                    )
                    ax.set_xlim(*bounds["x"])
                    ax.set_ylim(*bounds["z"])
                    ax.set_zlim(*bounds["up"])
                    ax.set_xlabel("X")
                    ax.set_ylabel("Z/display")
                    ax.set_zlabel("Up/display")
                    ax.view_init(elev=18, azim=-70)
                    metric_value = item["row"].get(metric_key)
                    ax.set_title(f"{backend} | {title} | {float(metric_value):.2f} mm", fontsize=10)
                    ax.legend(loc="upper left", fontsize=7)
            fig.suptitle(
                f"Best convention candidates | {camera} {trial} {hz:g} Hz | t={time_s:.3f}s",
                fontsize=13,
            )
            fig.tight_layout(pad=0.6)
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
            if frame_idx == representative_idx:
                representative_rgb = rgb
    finally:
        if writer is not None:
            writer.release()
        plt.close(fig)
    if representative_rgb is not None:
        cv2.imwrite(str(out_png), cv2.cvtColor(representative_rgb, cv2.COLOR_RGB2BGR))

    return {
        "status": "ok",
        "camera": camera,
        "trial": trial,
        "evaluation_hz": hz,
        "frames_rendered": int(common_time.shape[0]),
        "preview_fps": float(preview_fps),
        "time_start_s": float(common_time[0]),
        "time_end_s": float(common_time[-1]),
        "rows": [_public_row(item["row"]) for item in prepared],
        "reports": {item["row"]["backend"]: item["report"] for item in prepared},
        "panels": [title for _key, title, _metric in panel_keys],
    }


def _select_case_rows(
    rows: list[dict[str, Any]],
    camera: str | None,
    trial: str | None,
    evaluation_hz: float | None,
    backends: list[str],
) -> list[dict[str, Any]]:
    best = min(rows, key=lambda row: float(row["pa_mm"]))
    camera = camera or str(best["camera"])
    trial = trial or str(best["trial"])
    evaluation_hz = float(evaluation_hz if evaluation_hz is not None else best["evaluation_hz"])
    backend_set = {backend.strip() for backend in backends if backend.strip()}
    selected = [
        row
        for row in rows
        if row["camera"] == camera
        and row["trial"] == trial
        and abs(float(row["evaluation_hz"]) - evaluation_hz) < 1e-9
        and row["backend"] in backend_set
    ]
    if not selected:
        raise ValueError(f"No best-PA rows found for camera={camera}, trial={trial}, evaluation_hz={evaluation_hz}.")
    return sorted(selected, key=lambda row: BACKENDS.index(row["backend"]) if row["backend"] in BACKENDS else 99)


def _load_best_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("ranking") == "best_pa":
                rows.append(row)
    return rows


def _common_time(series_list: list[dict[str, np.ndarray]], fps: float) -> np.ndarray:
    starts = [float(np.nanmin(series["time_s"])) for series in series_list]
    ends = [float(np.nanmax(series["time_s"])) for series in series_list]
    start = max(starts)
    end = min(ends)
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ValueError("Selected series do not share an overlapping time window.")
    count = max(2, int(np.floor((end - start) * float(fps))) + 1)
    return start + np.arange(count, dtype=float) / float(fps)


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "camera",
        "trial",
        "backend",
        "evaluation_hz",
        "raw_primary_mm",
        "rigid_mm",
        "pa_mm",
        "axis",
        "reference_left_right_swap",
        "model_left_right_swap",
        "time_offset_s",
        "diagnostic_only",
        "source",
    ]
    return {key: row.get(key) for key in keys}


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


if __name__ == "__main__":
    raise SystemExit(main())
