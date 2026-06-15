from __future__ import annotations

import csv
import json
import statistics
from itertools import permutations, product
from pathlib import Path
from typing import Any

import numpy as np

from monocap_v2.core.level_a_benchmark import (
    BACKEND_AXIS_MAP,
    PRIMARY_ERROR_JOINTS,
    PRIMARY_JOINTS,
    _canonical_prediction_joints,
    _canonical_reference_joints,
    _prediction_timebase,
    _resolve_root_mode,
    _root_series,
    _stack_named,
    _swap_left_right_values,
    _uniform_overlap_time,
    _validate_root_mode,
    compare_pose_to_opensim_reference,
    load_cached_wham_timeline,
    load_opensim_reference,
    load_pose_artifact,
    load_run_config,
)
from monocap_v2.core.logging_utils import write_json
from monocap_v2.core.mocap_eval import resample_timeseries
from monocap_v2.core.wham_conventions import axis_expr_determinant
from monocap_v2.core.wham_timeline import build_wham_timeline_report


DEFAULT_TIME_OFFSET_MIN = -0.30
DEFAULT_TIME_OFFSET_MAX = 0.30
DEFAULT_TIME_OFFSET_STEP = 0.02
DEFAULT_OUT_DIR = Path("monocap_v2/benchmarks/cam0_cam1_convention_audit")
KNOWN_AXIS_CANDIDATES = ["x,-y,z", "x,-y,-z"]
PHYSICAL_AXIS_CANDIDATES = [
    "x,-y,z",
    "x,-y,-z",
    "-x,-y,-z",
    "-x,-y,z",
    "x,y,z",
    "x,y,-z",
    "-x,y,z",
    "-x,y,-z",
]

AUDIT_ROW_FIELDS = [
    "camera",
    "trial",
    "backend",
    "evaluation_hz",
    "status",
    "run_dir",
    "axis",
    "axis_determinant",
    "proper_axis",
    "axis_unusual",
    "reference_left_right_swap",
    "model_left_right_swap",
    "time_offset_s",
    "time_gt_tuned",
    "diagnostic_only",
    "promotable_candidate",
    "raw_primary_mm",
    "rigid_mm",
    "pa_mm",
    "normal_minus_rigid_gap_mm",
    "overlap_frames",
    "time_start_s",
    "time_end_s",
    "warnings",
    "error",
]

TOP_ROW_FIELDS = [
    "camera",
    "trial",
    "backend",
    "evaluation_hz",
    "ranking",
    "metric_value_mm",
    "raw_primary_mm",
    "rigid_mm",
    "pa_mm",
    "axis",
    "axis_determinant",
    "proper_axis",
    "reference_left_right_swap",
    "model_left_right_swap",
    "time_offset_s",
    "diagnostic_only",
    "promotable_candidate",
    "overlap_frames",
    "time_start_s",
    "time_end_s",
]


def run_backend_convention_audit(
    benchmark_dirs: dict[str, Path],
    trials: list[str],
    backends: list[str],
    evaluation_hz_values: list[float],
    out_dir: Path = DEFAULT_OUT_DIR,
    time_offset_min: float = DEFAULT_TIME_OFFSET_MIN,
    time_offset_max: float = DEFAULT_TIME_OFFSET_MAX,
    time_offset_step: float = DEFAULT_TIME_OFFSET_STEP,
    axis_candidates: list[str] | None = None,
    reference_lr_values: tuple[bool, ...] = (False, True),
    model_lr_values: tuple[bool, ...] = (False, True),
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    axes = axis_candidates or generate_axis_candidates()
    offsets = generate_time_offsets(time_offset_min, time_offset_max, time_offset_step)

    rows: list[dict[str, Any]] = []
    for camera, benchmark_dir in benchmark_dirs.items():
        rows.extend(
            _evaluate_benchmark_dir(
                camera=str(camera),
                benchmark_dir=Path(benchmark_dir),
                trials=trials,
                backends=backends,
                evaluation_hz_values=evaluation_hz_values,
                axes=axes,
                offsets=offsets,
                reference_lr_values=reference_lr_values,
                model_lr_values=model_lr_values,
            )
        )

    top_rows = top_rows_by_case(rows)
    stability = summarize_stability(rows, top_rows)
    outputs = {
        "convention_audit_rows": str(out_dir / "convention_audit_rows.csv"),
        "top_by_case": str(out_dir / "top_by_case.csv"),
        "stability_summary_json": str(out_dir / "stability_summary.json"),
        "stability_summary_md": str(out_dir / "stability_summary.md"),
        "walking_trial_tables": str(out_dir / "walking_trial_tables.md"),
    }

    _write_csv(Path(outputs["convention_audit_rows"]), rows, AUDIT_ROW_FIELDS)
    _write_csv(Path(outputs["top_by_case"]), top_rows, TOP_ROW_FIELDS)
    write_json(Path(outputs["stability_summary_json"]), {**stability, "outputs": outputs})
    _write_stability_md(Path(outputs["stability_summary_md"]), stability)
    _write_walking_trial_tables(Path(outputs["walking_trial_tables"]), top_rows)

    return {
        "status": stability["status"],
        "candidate_count": len(rows),
        "valid_candidate_count": len([row for row in rows if row.get("status") == "valid"]),
        "outputs": outputs,
        "stability": stability,
    }


def generate_axis_candidates() -> list[str]:
    axes = ["x", "y", "z"]
    seen: set[str] = set()
    out: list[str] = []
    for expr in KNOWN_AXIS_CANDIDATES:
        if expr not in seen:
            out.append(expr)
            seen.add(expr)
    for perm in permutations(axes):
        for signs in product((1, -1), repeat=3):
            expr = ",".join(("-" if sign < 0 else "") + axis for sign, axis in zip(signs, perm))
            if expr not in seen:
                out.append(expr)
                seen.add(expr)
    return out


def generate_physical_axis_candidates() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for expr in PHYSICAL_AXIS_CANDIDATES:
        if expr not in seen:
            out.append(expr)
            seen.add(expr)
    return out


def generate_time_offsets(min_value: float, max_value: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("time offset step must be positive")
    count = int(np.floor((float(max_value) - float(min_value)) / float(step) + 0.5)) + 1
    values = [round(float(min_value) + idx * float(step), 6) for idx in range(max(count, 0))]
    return [value for value in values if value <= float(max_value) + 1e-9]


def relabel_pose_left_right(pose: dict[str, Any]) -> dict[str, Any]:
    out = dict(pose)
    out["joint_names"] = [_swap_name(str(name)) for name in pose.get("joint_names", [])]
    pose2d = pose.get("pose2d")
    if isinstance(pose2d, dict):
        pose2d_out = dict(pose2d)
        if pose2d_out.get("names"):
            pose2d_out["names"] = [_swap_name(str(name)) for name in pose2d_out.get("names", [])]
        out["pose2d"] = pose2d_out
    return out


def parse_benchmark_dirs(value: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in _csv_arg(value):
        if "=" not in item:
            raise ValueError(f"Benchmark dir entry must be NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        out[name.strip()] = Path(path.strip())
    if not out:
        raise ValueError("At least one benchmark directory is required")
    return out


def top_rows_by_case(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in rows if row.get("status") == "valid"]
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = {}
    for row in valid:
        key = (str(row["camera"]), str(row["trial"]), str(row["backend"]), float(row["evaluation_hz"]))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(grouped):
        candidates = grouped[key]
        for ranking, metric_key in [
            ("best_raw", "raw_primary_mm"),
            ("best_rigid", "rigid_mm"),
            ("best_pa", "pa_mm"),
        ]:
            best = _best(candidates, metric_key)
            if best:
                out.append(_top_row(best, ranking, metric_key))
        promotable = [row for row in candidates if row.get("promotable_candidate")]
        best_promotable = _best(promotable, "raw_primary_mm")
        if best_promotable:
            out.append(_top_row(best_promotable, "best_promotable_raw", "raw_primary_mm"))
    return out


def summarize_stability(rows: list[dict[str, Any]], top_rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "valid"]
    failed = [row for row in rows if row.get("status") != "valid"]
    best_pa = [row for row in top_rows if row.get("ranking") == "best_pa"]
    axis_stability = _axis_stability(best_pa)
    timing_stability = _timing_stability(best_pa)
    trial_outliers = _trial_outliers(best_pa)
    promotable = [row for row in top_rows if row.get("ranking") == "best_promotable_raw"]
    status = "failed" if not valid else "warning" if failed or trial_outliers else "ok"
    return {
        "status": status,
        "candidate_count": len(rows),
        "valid_candidate_count": len(valid),
        "failed_candidate_count": len(failed),
        "best_pa_case_count": len(best_pa),
        "promotable_case_count": len(promotable),
        "axis_stability_by_backend": axis_stability,
        "timing_stability_by_camera_backend": timing_stability,
        "trial_outliers": trial_outliers,
        "interpretation": _interpretation(axis_stability, timing_stability, trial_outliers, promotable),
    }


def _evaluate_benchmark_dir(
    camera: str,
    benchmark_dir: Path,
    trials: list[str],
    backends: list[str],
    evaluation_hz_values: list[float],
    axes: list[str],
    offsets: list[float],
    reference_lr_values: tuple[bool, ...],
    model_lr_values: tuple[bool, ...],
) -> list[dict[str, Any]]:
    summary_path = benchmark_dir / "level_a_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_rows = {(row.get("backend"), row.get("trial")): row for row in summary.get("rows", [])}
    rows: list[dict[str, Any]] = []
    for trial in trials:
        reference = load_opensim_reference(
            benchmark_dir / "reference" / f"opensim_fk_{trial}.npz",
            benchmark_dir / "reference" / f"opensim_fk_{trial}.json",
        )
        for backend in backends:
            source = summary_rows.get((backend, trial))
            if not source or source.get("status") != "valid":
                rows.append(_failed_row(camera, trial, backend, None, None, "missing_valid_row", "No valid Level A source row."))
                continue
            run_dir = Path(str(source["run_dir"]))
            try:
                pose = load_pose_artifact(run_dir)
                run_config = load_run_config(run_dir)
                timeline = _load_or_build_wham_timeline(run_dir, pose, run_config)
            except Exception as exc:
                rows.append(_failed_row(camera, trial, backend, None, None, "artifact_failed", str(exc), run_dir=run_dir))
                continue
            pose_by_lr = {False: pose, True: relabel_pose_left_right(pose)}
            representative_axis_by_det = _representative_axes_by_determinant(axes)
            alignment_cache: dict[tuple[float, bool, bool, float, float], dict[str, float | None]] = {}
            for hz in evaluation_hz_values:
                for axis in axes:
                    axis_meta = _axis_meta(axis, backend)
                    for reference_lr in reference_lr_values:
                        for model_lr in model_lr_values:
                            candidate_pose = pose_by_lr[model_lr]
                            for offset in offsets:
                                rows.append(
                                    _score_candidate(
                                        camera,
                                        trial,
                                        backend,
                                        run_dir,
                                        candidate_pose,
                                        reference,
                                        run_config,
                                        timeline,
                                        float(hz),
                                        axis,
                                        axis_meta,
                                        bool(reference_lr),
                                        bool(model_lr),
                                        float(offset),
                                        alignment_cache,
                                        representative_axis_by_det,
                                    )
                                )
    return rows


def _load_or_build_wham_timeline(run_dir: Path, pose: dict[str, Any], run_config: dict[str, Any]) -> dict[str, Any] | None:
    if pose.get("backend") != "wham":
        return None
    cached = load_cached_wham_timeline(run_dir)
    if cached is not None:
        return cached
    timeline = build_wham_timeline_report(pose, run_config)
    out_path = run_dir / "reports" / "wham_timeline_qc.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, timeline)
    return timeline


def _score_candidate(
    camera: str,
    trial: str,
    backend: str,
    run_dir: Path,
    pose: dict[str, Any],
    reference: dict[str, Any],
    run_config: dict[str, Any],
    timeline: dict[str, Any] | None,
    evaluation_hz: float,
    axis: str,
    axis_meta: dict[str, Any],
    reference_lr: bool,
    model_lr: bool,
    offset: float,
    alignment_cache: dict[tuple[float, bool, bool, float, float], dict[str, float | None]],
    representative_axis_by_det: dict[float, str],
) -> dict[str, Any]:
    diagnostic_only = _diagnostic_only(axis_meta, reference_lr, model_lr, offset)
    base = {
        "camera": camera,
        "trial": trial,
        "backend": backend,
        "evaluation_hz": float(evaluation_hz),
        "run_dir": str(run_dir),
        "axis": axis,
        **axis_meta,
        "reference_left_right_swap": bool(reference_lr),
        "model_left_right_swap": bool(model_lr),
        "time_offset_s": float(offset),
        "time_gt_tuned": bool(abs(float(offset)) > 1e-12),
        "diagnostic_only": diagnostic_only,
        "promotable_candidate": not diagnostic_only,
    }
    try:
        raw_metrics = _raw_candidate_metrics(
            pose,
            reference,
            run_config=run_config,
            backend=backend,
            axis=axis,
            timeline_report=timeline,
            evaluation_hz=float(evaluation_hz),
            offset=float(offset),
            swap_reference_lr=bool(reference_lr),
        )
    except Exception as exc:
        return {**base, "status": "failed", "error": str(exc)}
    alignment = _cached_alignment_metrics(
        alignment_cache,
        representative_axis_by_det,
        axis_meta,
        backend,
        pose,
        reference,
        run_config,
        timeline,
        evaluation_hz,
        reference_lr,
        model_lr,
        offset,
    )
    raw_primary = raw_metrics["raw_primary_mm"]
    rigid_mm = alignment.get("rigid_mm")
    normal_gap = float(raw_primary - rigid_mm) if raw_primary is not None and rigid_mm is not None else None
    warnings = list(raw_metrics.get("warnings") or [])
    if normal_gap is not None and normal_gap > 100.0:
        warnings.append(
            f"Normal MPJPE exceeds rigid MPJPE by {normal_gap:.3f} mm; inspect camera/reference extrinsics, "
            "axis/sign convention, left-right labels, and timing before interpreting absolute orientation."
        )
    return {
        **base,
        "status": "valid",
        "raw_primary_mm": raw_primary,
        "rigid_mm": rigid_mm,
        "pa_mm": alignment.get("pa_mm"),
        "normal_minus_rigid_gap_mm": normal_gap,
        "overlap_frames": raw_metrics.get("overlap_frames"),
        "time_start_s": raw_metrics.get("time_start_s"),
        "time_end_s": raw_metrics.get("time_end_s"),
        "warnings": "; ".join(warnings),
        "error": "",
    }


def _raw_candidate_metrics(
    pose: dict[str, Any],
    reference: dict[str, Any],
    run_config: dict[str, Any],
    backend: str,
    axis: str,
    timeline_report: dict[str, Any] | None,
    evaluation_hz: float,
    offset: float,
    swap_reference_lr: bool,
    min_overlap_frames: int = 5,
) -> dict[str, Any]:
    root_mode = _resolve_root_mode(run_config, None)
    warnings: list[str] = []
    pred_names, pred_all, _convention = _canonical_prediction_joints(
        pose,
        axis,
        run_config=run_config,
        axis_explicit=True,
    )
    ref_names, ref_all = _canonical_reference_joints(reference)
    if swap_reference_lr:
        ref_all = _swap_left_right_values(ref_all, ref_names)
    common_primary = [name for name in PRIMARY_JOINTS if name in pred_names and name in ref_names]
    missing_primary = [name for name in PRIMARY_JOINTS if name not in common_primary]
    if missing_primary:
        warnings.append(f"Missing primary joints: {', '.join(missing_primary)}")
    _validate_root_mode(root_mode, common_primary)
    error_joints = [name for name in PRIMARY_ERROR_JOINTS if name in common_primary]
    if len(error_joints) < 3:
        raise ValueError("At least three non-root lower-limb joints are required for Level A comparison.")

    timebase = _prediction_timebase(pose, run_config, min_overlap_frames=min_overlap_frames, timeline_report=timeline_report)
    pose_indices = np.asarray(timebase["pose_indices"], dtype=int)
    prediction_time_native = np.asarray(timebase["prediction_timestamps_s"], dtype=float)
    pred_time = prediction_time_native - float(offset)
    pred_native = _stack_named(pred_names, pred_all, common_primary)[pose_indices]
    ref_native = _stack_named(ref_names, ref_all, common_primary)
    ref_time = np.asarray(reference["time_s"], dtype=float)
    native_valid_time = (pred_time >= np.nanmin(ref_time)) & (pred_time <= np.nanmax(ref_time))
    if evaluation_hz is not None:
        eval_hz = float(evaluation_hz)
        if not np.isfinite(eval_hz) or eval_hz <= 0:
            raise ValueError(f"evaluation_hz must be positive when provided, got {evaluation_hz!r}.")
        time = _uniform_overlap_time(pred_time[native_valid_time], ref_time, eval_hz)
        pred = resample_timeseries(pred_time[native_valid_time], pred_native[native_valid_time], time)
        ref = resample_timeseries(ref_time, ref_native, time)
    else:
        ref = resample_timeseries(ref_time, ref_native, pred_time)
        pred = pred_native[native_valid_time]
        ref = ref[native_valid_time]
        time = pred_time[native_valid_time]
    if pred.shape[0] < min_overlap_frames:
        raise ValueError(f"Only {pred.shape[0]} frames overlap the OpenSim FK reference; need at least {min_overlap_frames}.")

    eval_indices = [common_primary.index(name) for name in error_joints]
    pred_root = _root_series(pred, common_primary, root_mode)
    ref_root = _root_series(ref, common_primary, root_mode)
    pred_centered = pred[:, eval_indices, :] - pred_root[:, None, :]
    ref_centered = ref[:, eval_indices, :] - ref_root[:, None, :]
    dist = np.linalg.norm(pred_centered - ref_centered, axis=2)
    raw_m = float(np.nanmean(dist)) if np.isfinite(dist).any() else None
    return {
        "raw_primary_mm": raw_m * 1000.0 if raw_m is not None else None,
        "overlap_frames": int(pred.shape[0]),
        "time_start_s": float(time[0]),
        "time_end_s": float(time[-1]),
        "warnings": warnings,
    }


def _cached_alignment_metrics(
    cache: dict[tuple[float, bool, bool, float, float], dict[str, float | None]],
    representative_axis_by_det: dict[float, str],
    axis_meta: dict[str, Any],
    backend: str,
    pose: dict[str, Any],
    reference: dict[str, Any],
    run_config: dict[str, Any],
    timeline: dict[str, Any] | None,
    evaluation_hz: float,
    reference_lr: bool,
    model_lr: bool,
    offset: float,
) -> dict[str, float | None]:
    det = float(axis_meta["axis_determinant"])
    det_key = float(round(det))
    key = (det_key, bool(reference_lr), bool(model_lr), float(evaluation_hz), round(float(offset), 6))
    if key in cache:
        return cache[key]
    representative_axis = representative_axis_by_det.get(det_key)
    if not representative_axis:
        cache[key] = {"rigid_mm": None, "pa_mm": None}
        return cache[key]
    try:
        report, _series = compare_pose_to_opensim_reference(
            pose,
            reference,
            run_config=run_config,
            axis_map={backend: representative_axis},
            timeline_report=timeline,
            evaluation_hz=float(evaluation_hz),
            diagnostic_time_offset_s=float(offset),
            swap_reference_lr=bool(reference_lr),
            compute_pa=True,
            minimal_metrics=True,
        )
    except Exception:
        cache[key] = {"rigid_mm": None, "pa_mm": None}
        return cache[key]
    cache[key] = {"rigid_mm": report.get("root_centered_rigid_mpjpe_mm"), "pa_mm": report.get("pa_mpjpe_mm")}
    return cache[key]


def _representative_axes_by_determinant(axes: list[str]) -> dict[float, str]:
    out: dict[float, str] = {}
    for axis in axes:
        det = float(round(float(axis_expr_determinant(axis))))
        out.setdefault(det, axis)
    return out


def _axis_meta(axis: str, backend: str) -> dict[str, Any]:
    det = axis_expr_determinant(axis)
    default_axis = BACKEND_AXIS_MAP.get(backend, "identity")
    return {
        "axis_determinant": float(det),
        "proper_axis": bool(np.isfinite(det) and abs(float(det) - 1.0) < 1e-9),
        "axis_unusual": axis != default_axis,
    }


def _diagnostic_only(axis_meta: dict[str, Any], reference_lr: bool, model_lr: bool, offset: float) -> bool:
    return bool(
        axis_meta.get("axis_unusual")
        or not axis_meta.get("proper_axis")
        or reference_lr
        or model_lr
        or abs(float(offset)) > 1e-12
    )


def _failed_row(
    camera: str,
    trial: str,
    backend: str,
    evaluation_hz: float | None,
    axis: str | None,
    status: str,
    error: str,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    return {
        "camera": camera,
        "trial": trial,
        "backend": backend,
        "evaluation_hz": evaluation_hz,
        "status": status,
        "run_dir": str(run_dir) if run_dir else "",
        "axis": axis,
        "error": error,
    }


def _best(rows: list[dict[str, Any]], metric_key: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get(metric_key) not in {None, ""}]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row[metric_key]))


def _top_row(row: dict[str, Any], ranking: str, metric_key: str) -> dict[str, Any]:
    return {field: row.get(field) for field in TOP_ROW_FIELDS if field in row} | {
        "ranking": ranking,
        "metric_value_mm": row.get(metric_key),
    }


def _axis_stability(best_pa_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for backend in sorted({str(row["backend"]) for row in best_pa_rows}):
        selected = [row for row in best_pa_rows if row.get("backend") == backend]
        axes = [str(row.get("axis")) for row in selected]
        axis, count = _mode_count(axes)
        out.append(
            {
                "backend": backend,
                "case_count": len(selected),
                "most_common_axis": axis,
                "most_common_count": count,
                "stable": bool(count == len(selected) and selected),
                "axis_counts": _counts(axes),
            }
        )
    return out


def _timing_stability(best_pa_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    keys = sorted({(str(row["camera"]), str(row["backend"])) for row in best_pa_rows})
    for camera, backend in keys:
        selected = [row for row in best_pa_rows if row.get("camera") == camera and row.get("backend") == backend]
        offsets = [f"{float(row.get('time_offset_s') or 0.0):+.3f}" for row in selected]
        offset, count = _mode_count(offsets)
        out.append(
            {
                "camera": camera,
                "backend": backend,
                "case_count": len(selected),
                "most_common_time_offset_s": offset,
                "most_common_count": count,
                "stable": bool(count == len(selected) and selected),
                "time_offset_counts": _counts(offsets),
            }
        )
    return out


def _trial_outliers(best_pa_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    medians = []
    for trial in sorted({str(row["trial"]) for row in best_pa_rows}):
        values = [float(row["pa_mm"]) for row in best_pa_rows if row.get("trial") == trial and row.get("pa_mm") not in {None, ""}]
        if values:
            medians.append({"trial": trial, "median_best_pa_mm": float(statistics.median(values))})
    if not medians:
        return []
    global_median = float(statistics.median([row["median_best_pa_mm"] for row in medians]))
    return [
        {**row, "global_median_best_pa_mm": global_median, "delta_from_global_median_mm": row["median_best_pa_mm"] - global_median}
        for row in medians
        if row["median_best_pa_mm"] > global_median + 25.0
    ]


def _interpretation(
    axis_stability: list[dict[str, Any]],
    timing_stability: list[dict[str, Any]],
    trial_outliers: list[dict[str, Any]],
    promotable: list[dict[str, Any]],
) -> list[str]:
    notes = []
    if not promotable:
        notes.append("No non-diagnostic promotable candidate won a case; current strong candidates depend on reflection, timing, or L/R choices.")
    if any(not row["stable"] for row in axis_stability):
        notes.append("Best PA axis choices are not stable across all cameras/trials/frequencies for at least one backend.")
    if any(not row["stable"] for row in timing_stability):
        notes.append("Best PA timing offsets are not stable across all walking trials for at least one camera/backend.")
    if trial_outliers:
        notes.append("At least one walking trial is an outlier under best PA diagnostics; inspect time/window/model quality before promoting conventions.")
    if not notes:
        notes.append("Best diagnostic choices are stable in this audit, but GT-tuned settings remain evaluation-only.")
    return notes


def _write_stability_md(path: Path, stability: dict[str, Any]) -> None:
    lines = [
        "# Backend Convention Audit Stability Summary",
        "",
        f"- Status: `{stability.get('status')}`",
        f"- Candidates: `{stability.get('candidate_count')}`",
        f"- Valid candidates: `{stability.get('valid_candidate_count')}`",
        f"- Best-PA cases: `{stability.get('best_pa_case_count')}`",
        f"- Promotable best cases: `{stability.get('promotable_case_count')}`",
        "",
        "## Interpretation",
        "",
    ]
    lines.extend(f"- {note}" for note in stability.get("interpretation") or [])
    lines.extend(["", "## Axis Stability By Backend", "", "| Backend | Stable | Most Common Axis | Count | Cases |", "|---|---:|---|---:|---:|"])
    for row in stability.get("axis_stability_by_backend") or []:
        lines.append(f"| {row['backend']} | {row['stable']} | `{row['most_common_axis']}` | {row['most_common_count']} | {row['case_count']} |")
    lines.extend(["", "## Timing Stability By Camera/Backend", "", "| Camera | Backend | Stable | Most Common Offset | Count | Cases |", "|---|---|---:|---:|---:|---:|"])
    for row in stability.get("timing_stability_by_camera_backend") or []:
        lines.append(
            f"| {row['camera']} | {row['backend']} | {row['stable']} | {row['most_common_time_offset_s']} | {row['most_common_count']} | {row['case_count']} |"
        )
    lines.extend(["", "## Trial Outliers", "", "| Trial | Median Best PA | Delta From Global Median |", "|---|---:|---:|"])
    for row in stability.get("trial_outliers") or []:
        lines.append(f"| {row['trial']} | {_fmt(row['median_best_pa_mm'])} | {_fmt(row['delta_from_global_median_mm'])} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_walking_trial_tables(path: Path, top_rows: list[dict[str, Any]]) -> None:
    rows = [row for row in top_rows if row.get("ranking") == "best_pa"]
    lines = [
        "# Walking Trial Best-PA Diagnostic Tables",
        "",
        "Each row is the lowest PA-MPJPE diagnostic candidate for that camera/trial/backend/frequency.",
    ]
    for trial in sorted({str(row["trial"]) for row in rows}):
        lines.extend(
            [
                "",
                f"## {trial}",
                "",
                "| Camera | Hz | Backend | Raw/root | Rigid | PA | Axis | Offset | Ref L/R | Model L/R |",
                "|---|---:|---|---:|---:|---:|---|---:|---:|---:|",
            ]
        )
        for row in [item for item in rows if item.get("trial") == trial]:
            lines.append(
                f"| {row['camera']} | {_fmt(row['evaluation_hz'])} | {row['backend']} | "
                f"{_fmt(row.get('raw_primary_mm'))} | {_fmt(row.get('rigid_mm'))} | {_fmt(row.get('pa_mm'))} | "
                f"`{row.get('axis')}` | {_fmt(row.get('time_offset_s'))} | {row.get('reference_left_right_swap')} | {row.get('model_left_right_swap')} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _mode_count(values: list[str]) -> tuple[str | None, int]:
    if not values:
        return None, 0
    counts = _counts(values)
    key = max(counts, key=lambda item: counts[item])
    return key, counts[key]


def _counts(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def _swap_name(name: str) -> str:
    lower = name.lower()
    if lower.startswith("left"):
        return "right" + name[4:]
    if lower.startswith("right"):
        return "left" + name[5:]
    if lower.startswith("l") and any(key in lower for key in ["hip", "kne", "ank", "toe", "mtp", "heel"]):
        return "r" + name[1:]
    if lower.startswith("r") and any(key in lower for key in ["hip", "kne", "ank", "toe", "mtp", "heel"]):
        return "l" + name[1:]
    return name


def _csv_arg(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    return value


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.2f}"
