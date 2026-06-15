from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.logging_utils import read_json, write_json
from monocap_v2.core.mocap_marker_compare import aligned_smpl_mesh_vertices, compare_virtual_markers_to_mocap, metric_to_display_coords
from monocap_v2.core.mocap_eval import evaluate_pose_against_mocap_with_series, parse_trc, validate_mocap_to_video_transform
from monocap_v2.core.smpl_mesh import load_smpl_faces
from monocap_v2.core.stage_utils import cached, stage_result
from monocap_v2.core.wham_timeline import build_wham_timeline_report


STAGE = "stage_12_mocap_validation"
LOWER_LIMB_EDGES = [
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_toe"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_toe"),
]
MOCAP_NATIVE_EDGES = [
    ("L_HJC", "R_HJC"),
    ("L_HJC", "L_knee"),
    ("L_knee", "L_ankle"),
    ("L_ankle", "L_toe"),
    ("L_ankle", "L_calc"),
    ("L_calc", "L_toe"),
    ("R_HJC", "r_knee"),
    ("r_knee", "r_ankle"),
    ("r_ankle", "r_toe"),
    ("r_ankle", "r_calc"),
    ("r_calc", "r_toe"),
    ("L.ASIS", "r.ASIS"),
    ("L.PSIS", "r.PSIS"),
    ("L.ASIS", "L.PSIS"),
    ("r.ASIS", "r.PSIS"),
]
MOCAP_NATIVE_LABELS = {
    "L_HJC",
    "R_HJC",
    "L_knee",
    "r_knee",
    "L_ankle",
    "r_ankle",
    "L_calc",
    "r_calc",
    "L_toe",
    "r_toe",
    "L.ASIS",
    "r.ASIS",
}


def run(run_dir: Path, cfg: dict, force: bool = False) -> dict:
    registry = ArtifactRegistry(run_dir)
    report_path = registry.ensure_parent("mocap_validation")
    if cached(report_path, force):
        return stage_result(STAGE, "cached", output=str(report_path))

    validation_cfg = cfg.get("config", {}).get("mocap_validation", {})
    if not validation_cfg.get("enabled", True):
        return _write_status(report_path, "skipped", reason="Mocap validation disabled in config.")
    mocap_path = Path(str(cfg.get("mocap_trc") or ""))
    if not cfg.get("mocap_trc") or not mocap_path.exists():
        return _write_status(report_path, "skipped", reason=f"mocap_trc is unavailable: {mocap_path}")

    initial_path = registry.get("pose3d_initial")
    refined_path = registry.get("pose3d_refined")
    if not initial_path.exists():
        return _write_status(report_path, "skipped", reason="pose3d_initial artifact is missing.")
    initial = _read_pose(initial_path)
    refined = _read_pose(refined_path) if refined_path.exists() else initial
    timeline = _timeline_report(initial, cfg, registry)
    transform_qc = validate_mocap_to_video_transform(
        ((cfg.get("manifest_summary") or {}).get("calibration") or {}).get("mocap_to_video")
    )
    mocap_axis = str(validation_cfg.get("mocap_axis", "-y,x,z"))

    try:
        initial_report, initial_series = evaluate_pose_against_mocap_with_series(
            initial,
            mocap_path,
            mocap_axis=mocap_axis,
            cfg=cfg,
            timeline_report=timeline,
            mocap_to_video_transform=transform_qc,
        )
        refined_report, refined_series = evaluate_pose_against_mocap_with_series(
            refined,
            mocap_path,
            mocap_axis=mocap_axis,
            cfg=cfg,
            timeline_report=timeline,
            mocap_to_video_transform=transform_qc,
        )
    except Exception as exc:
        return _write_status(
            report_path,
            "warning",
            reason=str(exc),
            mocap_trc=str(mocap_path),
            transform_qc=transform_qc,
            wham_timeline=timeline,
            mocap_used_in_objective=False,
        )

    outputs = {
        "mocap_validation": str(report_path),
        "mocap_validation_series": str(registry.ensure_parent("mocap_validation_series")),
        "mocap_joint_errors_plot": str(registry.ensure_parent("mocap_joint_errors_plot")),
        "mocap_lower_limb_overlay": str(registry.ensure_parent("mocap_lower_limb_overlay")),
        "mocap_foot_trajectories_plot": str(registry.ensure_parent("mocap_foot_trajectories_plot")),
        "mocap_segment_lengths_plot": str(registry.ensure_parent("mocap_segment_lengths_plot")),
    }
    _write_series(Path(outputs["mocap_validation_series"]), initial_series, refined_series)
    _write_joint_error_plot(initial_report, refined_report, Path(outputs["mocap_joint_errors_plot"]))
    _write_foot_trajectories_plot(refined_series, Path(outputs["mocap_foot_trajectories_plot"]))
    _write_segment_lengths_plot(refined_report, Path(outputs["mocap_segment_lengths_plot"]))
    _write_lower_limb_overlay(
        refined_series,
        Path(outputs["mocap_lower_limb_overlay"]),
        preview_fps=float(validation_cfg.get("preview_fps", 30.0)),
    )
    mocap_only_native = _write_mocap_only_native(registry, mocap_path, cfg, outputs)
    marker_comparison = _write_marker_comparison(
        registry,
        refined,
        mocap_path,
        refined_report,
        refined_series,
        cfg,
        mocap_axis,
        outputs,
    )

    warnings = list(transform_qc.get("warnings") or [])
    warnings.extend(str(w) for w in initial_report.get("warnings") or [])
    warnings.extend(str(w) for w in refined_report.get("warnings") or [])
    warnings.extend(str(w) for w in mocap_only_native.get("warnings") or [])
    warnings.extend(str(w) for w in marker_comparison.get("warnings") or [])
    warnings = list(dict.fromkeys(warnings))
    status = "warning" if warnings or transform_qc.get("status") == "warning" else "ok"
    report = stage_result(
        STAGE,
        status,
        output=str(report_path),
        outputs=outputs,
        mocap_trc=str(mocap_path),
        mocap_used_in_objective=False,
        backend=initial.get("backend"),
        representation=initial.get("representation"),
        timing=refined_report.get("timebase"),
        joint_count=len(refined_report.get("joint_names") or []),
        joint_names=refined_report.get("joint_names"),
        mocap_marker_mapping=refined_report.get("mocap_marker_mapping"),
        initial=initial_report,
        refined=refined_report,
        transform_qc=transform_qc,
        mocap_only_native=mocap_only_native,
        marker_comparison=marker_comparison,
        warnings=warnings,
    )
    write_json(report_path, report)
    return report


def _read_pose(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def _timeline_report(pose: dict, cfg: dict, registry: ArtifactRegistry) -> dict | None:
    if pose.get("backend") != "wham":
        return None
    path = registry.get("wham_timeline_qc")
    return read_json(path) if path.exists() else build_wham_timeline_report(pose, cfg)


def _write_status(path: Path, status: str, **extra: Any) -> dict[str, Any]:
    report = stage_result(STAGE, status, output=str(path), **extra)
    write_json(path, report)
    return report


def _write_series(path: Path, initial: dict[str, np.ndarray], refined: dict[str, np.ndarray]) -> None:
    payload = {f"initial_{key}": value for key, value in initial.items()}
    payload.update({f"refined_{key}": value for key, value in refined.items()})
    np.savez_compressed(path, **payload)


def _write_joint_error_plot(initial: dict, refined: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(refined["joint_names"])
    x = np.arange(len(names))
    width = 0.2
    initial_normal = _metric_vector(initial, names, "per_joint_normal_root_centered_mpjpe_mm")
    refined_normal = _metric_vector(refined, names, "per_joint_normal_root_centered_mpjpe_mm")
    initial_rigid = _metric_vector(initial, names, "per_joint_root_centered_rigid_mpjpe_mm")
    refined_rigid = _metric_vector(refined, names, "per_joint_root_centered_rigid_mpjpe_mm")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - 1.5 * width, initial_normal, width, label="Initial normal")
    ax.bar(x - 0.5 * width, refined_normal, width, label="Refined normal")
    ax.bar(x + 0.5 * width, initial_rigid, width, label="Initial rigid")
    ax.bar(x + 1.5 * width, refined_rigid, width, label="Refined rigid")
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_ylabel("MPJPE (mm)")
    ax.set_title("Lower-limb mocap joint errors")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _write_foot_trajectories_plot(series: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.asarray(series["time_s"], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for side, color in (("left", "#1f77b4"), ("right", "#d62728")):
        axes[0].plot(time, series[f"{side}_foot_prediction_height_m"] * 1000.0, color=color, label=f"{side} prediction")
        axes[0].plot(time, series[f"{side}_foot_mocap_height_m"] * 1000.0, color=color, linestyle="--", label=f"{side} mocap")
        axes[1].plot(time, series[f"{side}_foot_prediction_speed_mps"], color=color, label=f"{side} prediction")
        axes[1].plot(time, series[f"{side}_foot_mocap_speed_mps"], color=color, linestyle="--", label=f"{side} mocap")
    axes[0].set_ylabel("Foot height (mm)")
    axes[1].set_ylabel("Foot speed (m/s)")
    axes[1].set_xlabel("Synced time (s)")
    axes[0].set_title("Lower-limb foot trajectories")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _write_segment_lengths_plot(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    segments = report.get("segment_lengths") or {}
    names = list(segments)
    x = np.arange(len(names))
    width = 0.36
    pred = [segments[name].get("prediction_median_mm") for name in names]
    ref = [segments[name].get("mocap_median_mm") for name in names]
    pred_std = [segments[name].get("prediction_std_mm") for name in names]
    ref_std = [segments[name].get("mocap_std_mm") for name in names]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, pred, width, yerr=pred_std, capsize=3, label="Prediction")
    ax.bar(x + width / 2, ref, width, yerr=ref_std, capsize=3, label="Mocap")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylabel("Length (mm)")
    ax.set_title("Lower-limb segment lengths")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _write_lower_limb_overlay(series: dict[str, np.ndarray], path: Path, preview_fps: float = 30.0) -> None:
    import cv2

    pred = np.asarray(series["prediction_root_centered_rigid_m"], dtype=float)
    ref = np.asarray(series["mocap_root_centered_m"], dtype=float)
    names = [str(name) for name in series["joint_names"]]
    time = np.asarray(series["time_s"], dtype=float)
    width, height = 900, 700
    margin = 60
    points = np.concatenate([pred[..., [0, 2]], ref[..., [0, 2]]], axis=0).reshape(-1, 2)
    finite = points[np.isfinite(points).all(axis=1)]
    if not finite.size:
        raise ValueError("No finite lower-limb coordinates available for overlay.")
    lo = finite.min(axis=0)
    hi = finite.max(axis=0)
    pad = np.maximum((hi - lo) * 0.1, 0.05)
    lo -= pad
    hi += pad
    edges = [(names.index(a), names.index(b)) for a, b in LOWER_LIMB_EDGES if a in names and b in names]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(preview_fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for frame_idx in range(len(pred)):
            canvas = np.full((height, width, 3), 250, dtype=np.uint8)
            _draw_overlay_pose(canvas, ref[frame_idx], edges, lo, hi, margin, (70, 160, 70), "Mocap", (20, 34))
            _draw_overlay_pose(canvas, pred[frame_idx], edges, lo, hi, margin, (40, 90, 210), "WHAM/SMPL", (140, 34))
            cv2.putText(canvas, f"t={time[frame_idx]:.3f}s", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Pelvis-centered, sequence rigid-aligned QC view", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()


def _draw_overlay_pose(canvas, pose, edges, lo, hi, margin, color, label, label_xy) -> None:
    import cv2

    def pixel(point):
        x = margin + (point[0] - lo[0]) / max(hi[0] - lo[0], 1e-9) * (canvas.shape[1] - 2 * margin)
        y = canvas.shape[0] - margin - (point[2] - lo[1]) / max(hi[1] - lo[1], 1e-9) * (canvas.shape[0] - 2 * margin)
        return int(round(x)), int(round(y))

    cv2.putText(canvas, label, label_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    for a, b in edges:
        if np.isfinite(pose[[a, b]]).all():
            cv2.line(canvas, pixel(pose[a]), pixel(pose[b]), color, 2, cv2.LINE_AA)
    for point in pose:
        if np.isfinite(point).all():
            cv2.circle(canvas, pixel(point), 4, color, -1, cv2.LINE_AA)


def _metric_vector(report: dict, names: list[str], key: str) -> list[float]:
    values = report.get(key) or {}
    return [float(values.get(name) or 0.0) for name in names]


def _write_mocap_only_native(registry: ArtifactRegistry, mocap_path: Path, cfg: dict, outputs: dict[str, str]) -> dict:
    native_cfg = cfg.get("config", {}).get("mocap_validation", {}).get("mocap_only_native", {})
    qc_path = registry.ensure_parent("mocap_only_native_qc")
    outputs["mocap_only_native_qc"] = str(qc_path)
    if not native_cfg.get("enabled", True):
        report = {"status": "skipped", "reason": "Native mocap-only visualization disabled in config."}
        write_json(qc_path, report)
        return report
    try:
        trc = parse_trc(mocap_path)
        markers = np.stack([trc.markers[name] for name in trc.marker_names], axis=1)
        overlay_path = registry.ensure_parent("mocap_only_native_overlay")
        frame_path = registry.ensure_parent("mocap_only_native_representative_frame")
        fps = float(trc.data_rate or _series_rate_hz(trc.time) or 100.0)
        render = _write_mocap_only_native_overlay(
            markers,
            trc.marker_names,
            np.asarray(trc.time, dtype=float),
            overlay_path,
            frame_path,
            fps=fps,
            show_labels=bool(native_cfg.get("show_labels", True)),
            representative_frame=native_cfg.get("representative_frame", "mid"),
        )
        finite_ratio = float(np.isfinite(markers).all(axis=2).mean())
        warnings = []
        if finite_ratio < 1.0:
            warnings.append(f"Raw TRC contains non-finite marker samples; finite ratio is {finite_ratio:.6f}.")
        outputs.update(
            {
                "mocap_only_native_overlay": str(overlay_path),
                "mocap_only_native_representative_frame": str(frame_path),
            }
        )
        report = {
            "status": "warning" if warnings else "ok",
            "source_trc": str(mocap_path),
            "coordinate_space": "raw_trc_native",
            "axis_description": {"x": "progression", "y": "up", "z": "lateral"},
            "units": trc.units,
            "frames": int(markers.shape[0]),
            "marker_count": int(markers.shape[1]),
            "finite_ratio": finite_ratio,
            "data_rate_hz": fps,
            "alignment": "none",
            "resampling": "none",
            "render": render,
            "outputs": {
                "mocap_only_native_qc": str(qc_path),
                "mocap_only_native_overlay": str(overlay_path),
                "mocap_only_native_representative_frame": str(frame_path),
            },
            "warnings": warnings,
        }
        write_json(qc_path, report)
        return report
    except Exception as exc:
        report = {
            "status": "warning",
            "reason": f"Native mocap-only visualization could not be written: {exc}",
            "warnings": [str(exc)],
        }
        write_json(qc_path, report)
        return report


def _write_mocap_only_native_overlay(
    markers: np.ndarray,
    names: list[str],
    time_s: np.ndarray,
    path: Path,
    frame_path: Path,
    fps: float,
    show_labels: bool = True,
    representative_frame: str | int = "mid",
) -> dict:
    import cv2

    markers = np.asarray(markers, dtype=float)
    if markers.ndim != 3 or markers.shape[-1] != 3 or markers.shape[1] != len(names):
        raise ValueError("Native mocap overlay requires marker coordinates [T, M, 3] and matching names.")
    if markers.shape[0] < 1:
        raise ValueError("Native mocap overlay requires at least one frame.")
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    edges = [(name_to_idx[a], name_to_idx[b]) for a, b in MOCAP_NATIVE_EDGES if a in name_to_idx and b in name_to_idx]
    width, height = 1280, 720
    panel_margin = 55
    panel_gap = 34
    panel_top = 118
    panel_bottom = 58
    panel_width = (width - 2 * panel_margin - panel_gap) // 2
    panel_height = height - panel_top - panel_bottom
    panels = [
        {
            "rect": (panel_margin, panel_top, panel_width, panel_height),
            "title": "Front: lateral / up",
            "horizontal": "raw Z lateral",
            "axes": (2, 1),
            "bounds": _native_projection_bounds(markers, (2, 1)),
        },
        {
            "rect": (panel_margin + panel_width + panel_gap, panel_top, panel_width, panel_height),
            "title": "Side: progression / up",
            "horizontal": "raw X progression",
            "axes": (0, 1),
            "bounds": _native_projection_bounds(markers, (0, 1)),
        },
    ]
    representative_idx = (
        len(markers) // 2
        if str(representative_frame) == "mid"
        else min(max(int(representative_frame), 0), len(markers) - 1)
    )
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for frame_idx, frame in enumerate(markers):
            canvas = np.full((height, width, 3), 250, dtype=np.uint8)
            cv2.putText(canvas, "Raw mocap TRC only", (24, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (30, 30, 30), 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"native frame={frame_idx + 1}/{len(markers)}  t={time_s[frame_idx]:.3f}s  no alignment  no resampling",
                (24, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (70, 70, 70),
                1,
                cv2.LINE_AA,
            )
            for panel in panels:
                _draw_mocap_native_panel(canvas, frame, names, edges, panel, show_labels=show_labels)
            cv2.putText(
                canvas,
                "raw TRC axes: X=progression, Y=up, Z=lateral | gray=all mocap markers, blue=left, red=right",
                (24, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (65, 65, 65),
                1,
                cv2.LINE_AA,
            )
            writer.write(canvas)
            if frame_idx == representative_idx:
                cv2.imwrite(str(frame_path), canvas)
    finally:
        writer.release()
    return {
        "status": "ok",
        "render_mode": "raw_trc_native_front_side_2d",
        "frames_rendered": int(len(markers)),
        "render_fps": float(fps),
        "marker_count": int(len(names)),
        "skeleton_edges_rendered": int(len(edges)),
        "representative_frame_index": int(representative_idx),
    }


def _draw_mocap_native_panel(canvas, frame: np.ndarray, names: list[str], edges: list[tuple[int, int]], panel: dict, show_labels: bool) -> None:
    import cv2

    left, top, width, height = panel["rect"]
    lo, hi = panel["bounds"]
    axes = panel["axes"]

    def pixel(point):
        projected = point[list(axes)]
        x = left + (projected[0] - lo[0]) / max(hi[0] - lo[0], 1e-9) * width
        y = top + height - (projected[1] - lo[1]) / max(hi[1] - lo[1], 1e-9) * height
        return int(round(x)), int(round(y))

    cv2.rectangle(canvas, (left, top), (left + width, top + height), (205, 205, 205), 1)
    cv2.putText(canvas, panel["title"], (left, top - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.61, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{panel['horizontal']} | raw Y up", (left, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
    for a, b in edges:
        if np.isfinite(frame[[a, b]]).all():
            cv2.line(canvas, pixel(frame[a]), pixel(frame[b]), _mocap_native_color(names[a]), 2, cv2.LINE_AA)
    for idx, point in enumerate(frame):
        if not np.isfinite(point).all():
            continue
        color = _mocap_native_color(names[idx]) if names[idx] in MOCAP_NATIVE_LABELS else (145, 145, 145)
        radius = 4 if names[idx] in MOCAP_NATIVE_LABELS else 2
        cv2.circle(canvas, pixel(point), radius, color, -1, cv2.LINE_AA)
        if show_labels and names[idx] in MOCAP_NATIVE_LABELS:
            x, y = pixel(point)
            cv2.putText(canvas, names[idx], (x + 5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1, cv2.LINE_AA)


def _native_projection_bounds(markers: np.ndarray, axes: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(markers, dtype=float)[..., list(axes)].reshape(-1, 2)
    finite = points[np.isfinite(points).all(axis=1)]
    if not finite.size:
        raise ValueError("No finite raw TRC marker coordinates available for native mocap overlay.")
    lo = finite.min(axis=0)
    hi = finite.max(axis=0)
    pad = np.maximum((hi - lo) * 0.08, 0.05)
    return lo - pad, hi + pad


def _mocap_native_color(name: str) -> tuple[int, int, int]:
    lower = name.lower()
    if lower.startswith("l"):
        return  (210, 105, 35)
    if lower.startswith("r"):
        return (60, 60, 210)
    return (80, 150, 80)


def _write_marker_comparison(
    registry: ArtifactRegistry,
    pose: dict,
    mocap_path: Path,
    evaluation_report: dict,
    evaluation_series: dict[str, np.ndarray],
    cfg: dict,
    mocap_axis: str,
    outputs: dict[str, str],
) -> dict:
    marker_cfg = cfg.get("config", {}).get("mocap_validation", {}).get("marker_comparison", {})
    qc_path = registry.ensure_parent("mocap_smpl_marker_qc")
    outputs["mocap_smpl_marker_qc"] = str(qc_path)
    if not marker_cfg.get("enabled", True):
        report = {"status": "skipped", "reason": "Mocap/SMPL marker comparison disabled in config."}
        write_json(qc_path, report)
        return report
    marker_path = registry.get("virtual_markers")
    if not marker_path.exists():
        report = {"status": "skipped", "reason": "virtual_markers.pkl is unavailable; marker comparison requires SMPL markers."}
        write_json(qc_path, report)
        return report
    try:
        with marker_path.open("rb") as f:
            marker_payload = pickle.load(f)
        report, series = compare_virtual_markers_to_mocap(
            pose,
            marker_payload,
            mocap_path,
            evaluation_report,
            evaluation_series,
            mocap_axis=mocap_axis,
            warn_median_residual_mm=float(marker_cfg.get("warn_median_residual_mm", 250.0)),
        )
        series_path = registry.ensure_parent("mocap_smpl_marker_series")
        errors_plot = registry.ensure_parent("mocap_smpl_marker_errors_plot")
        overlay_path = registry.ensure_parent("mocap_smpl_marker_overlay")
        frame_path = registry.ensure_parent("mocap_smpl_marker_representative_frame")
        np.savez_compressed(series_path, **series)
        _write_marker_comparison_errors_plot(report, errors_plot)
        render_qc = _write_mocap_smpl_marker_overlay(pose, series, evaluation_report, cfg, overlay_path, frame_path)
        report["render"] = render_qc
        report["outputs"] = {
            "mocap_smpl_marker_qc": str(qc_path),
            "mocap_smpl_marker_series": str(series_path),
            "mocap_smpl_marker_overlay": str(overlay_path),
            "mocap_smpl_marker_errors_plot": str(errors_plot),
            "mocap_smpl_marker_representative_frame": str(frame_path),
        }
        outputs.update(report["outputs"])
        write_json(qc_path, report)
        return report
    except Exception as exc:
        report = {"status": "warning", "reason": f"Mocap/SMPL marker comparison could not be written: {exc}", "warnings": [str(exc)]}
        write_json(qc_path, report)
        return report


def _write_marker_comparison_errors_plot(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = report.get("per_marker") or {}
    names = list(entries)
    median = [entries[name].get("median_residual_mm") for name in names]
    p95 = [entries[name].get("p95_residual_mm") for name in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - 0.18, median, 0.36, label="Median")
    ax.bar(x + 0.18, p95, 0.36, label="P95")
    ax.set_xticks(x, [name.removeprefix("DBG_") for name in names], rotation=30, ha="right")
    ax.set_ylabel("Residual (mm)")
    ax.set_title("Approximate mocap vs SMPL debug-marker residuals")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _write_mocap_smpl_marker_overlay(
    pose: dict,
    series: dict[str, np.ndarray],
    evaluation_report: dict,
    cfg: dict,
    path: Path,
    frame_path: Path,
) -> dict:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    marker_cfg = cfg.get("config", {}).get("mocap_validation", {}).get("marker_comparison", {})
    mesh = metric_to_display_coords(aligned_smpl_mesh_vertices(pose, series, evaluation_report))
    faces_info = load_smpl_faces(cfg, vertex_count=mesh.shape[1])
    faces = _decimate_faces(np.asarray(faces_info["faces"], dtype=int), int(marker_cfg.get("max_faces", 1800)))
    smpl_markers = metric_to_display_coords(np.asarray(series["smpl_markers_aligned_m"], dtype=float))
    mocap_markers = metric_to_display_coords(np.asarray(series["mocap_markers_aligned_m"], dtype=float))
    residual_mm = np.asarray(series["residual_mm"], dtype=float)
    names = [str(name) for name in series["marker_names"]]
    time = np.asarray(series["time_s"], dtype=float)
    width, height = 1280, 640
    fps = float(_series_rate_hz(time) or cfg.get("config", {}).get("mocap_validation", {}).get("preview_fps", 30.0))
    bounds = _mesh_bounds(mesh)
    frame_choice = str(marker_cfg.get("representative_frame", "mid"))
    representative_idx = len(mesh) // 2 if frame_choice == "mid" else min(max(int(frame_choice), 0), len(mesh) - 1)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    canvas = FigureCanvasAgg(fig)
    axes = [fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d")]
    try:
        for frame_idx in range(len(mesh)):
            for ax, title, azim in zip(axes, ["Front: lateral / up", "Side: progression / up"], [-90, 0]):
                ax.clear()
                collection = Poly3DCollection(
                    mesh[frame_idx][faces],
                    alpha=0.28,
                    facecolor="#8f9ca8",
                    edgecolor="#d5d9dd",
                    linewidth=0.04,
                )
                ax.add_collection3d(collection)
                _draw_marker_comparison_3d(ax, smpl_markers[frame_idx], mocap_markers[frame_idx], residual_mm[frame_idx], names)
                _set_mesh_axes(ax, bounds, title, azim)
            fig.suptitle(
                f"Mocap vs SMPL debug markers | t={time[frame_idx]:.3f}s | "
                "blue=SMPL, green=mocap, orange=residual",
                fontsize=11,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            canvas.draw()
            frame = np.asarray(canvas.buffer_rgba())[..., :3]
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if frame_idx == representative_idx:
                fig.savefig(frame_path, dpi=140)
    finally:
        writer.release()
        plt.close(fig)
    return {
        "status": "ok",
        "render_mode": "triangulated_smpl_mesh_front_side",
        "frames_rendered": int(len(mesh)),
        "render_fps": fps,
        "display_axes": "lateral_x, progression_y, up_z",
        "faces_rendered": int(len(faces)),
        "face_source": faces_info.get("source"),
        "representative_frame_index": int(representative_idx),
    }


def _draw_marker_comparison_3d(ax, smpl_markers: np.ndarray, mocap_markers: np.ndarray, residual_mm: np.ndarray, names: list[str]) -> None:
    for idx, name in enumerate(names):
        smpl = smpl_markers[idx]
        mocap = mocap_markers[idx]
        if not np.isfinite(smpl).all() or not np.isfinite(mocap).all():
            continue
        ax.scatter(*smpl, s=20, c="#1f77b4", depthshade=False)
        ax.scatter(*mocap, s=20, c="#2ca02c", depthshade=False)
        ax.plot([smpl[0], mocap[0]], [smpl[1], mocap[1]], [smpl[2], mocap[2]], c="#ff7f0e", linewidth=1.1)
        ax.text(smpl[0], smpl[1], smpl[2], f"{name.removeprefix('DBG_')} {residual_mm[idx]:.0f}", fontsize=5, color="#174f85")


def _mesh_bounds(mesh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = mesh[np.isfinite(mesh).all(axis=2)]
    lo = np.nanmin(finite, axis=0)
    hi = np.nanmax(finite, axis=0)
    center = 0.5 * (lo + hi)
    half = max(float(np.max(hi - lo)) * 0.58, 0.5)
    return center - half, center + half


def _set_mesh_axes(ax, bounds: tuple[np.ndarray, np.ndarray], title: str, azim: float) -> None:
    lo, hi = bounds
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.view_init(elev=0, azim=azim)
    ax.set_title(title)
    ax.set_axis_off()
    ax.text2D(0.03, 0.96, "Up ^", transform=ax.transAxes, fontsize=9)


def _decimate_faces(faces: np.ndarray, max_faces: int) -> np.ndarray:
    if max_faces <= 0 or faces.shape[0] <= max_faces:
        return faces
    return faces[np.linspace(0, faces.shape[0] - 1, max_faces).round().astype(int)]


def _series_rate_hz(time_s: np.ndarray) -> float | None:
    delta = np.diff(np.asarray(time_s, dtype=float))
    valid = delta[np.isfinite(delta) & (delta > 1e-12)]
    return float(1.0 / np.median(valid)) if valid.size else None
