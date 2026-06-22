from __future__ import annotations

import csv
import pickle
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from monocap_v2.core.geometry import find_joint
from monocap_v2.core.logging_utils import read_json, read_yaml, write_json
from monocap_v2.core.manifest import default_run_name
from monocap_v2.core.mocap_eval import (
    foot_trajectory_series,
    mpjpe,
    normalized_mpjpe,
    per_frame_aligned_pose,
    resample_timeseries,
    sequence_mpjpe,
    sequence_mpjpe_with_transform,
    summarize_segment_lengths,
)
from monocap_v2.core.opensim_fk import read_ik_marker_error_summary
from monocap_v2.core.wham_conventions import apply_level_a_convention


PRIMARY_JOINTS = ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
PRIMARY_ERROR_JOINTS = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
OPTIONAL_JOINTS = ["left_mtp", "right_mtp"]
# Level A compares against OpenSim FK joint centers. OpenSim ground uses Y-up,
# while the camera-space backends here report image-like coordinates with +Y
# downward and +Z depth, so use an OpenSim-Y-up conversion rather than the
# older Z-up camera_to_eval helper.
BACKEND_AXIS_MAP = {"metrabs": "x,-y,z", "wham": "x,-y,z", "rtmw3d": "x,-y,z", "sam3d_body": "x,-y,z"}
DEFAULT_ROOT_MODE = "hip_midpoint"
TIE_THRESHOLD_MM = 5.0

PREDICTION_ALIASES = {
    "pelvis": ("pelvis", "pelv", "root"),
    "left_hip": ("left_hip", "lhip"),
    "right_hip": ("right_hip", "rhip"),
    "left_knee": ("left_knee", "lkne"),
    "right_knee": ("right_knee", "rkne"),
    "left_ankle": ("left_ankle", "lank"),
    "right_ankle": ("right_ankle", "rank"),
    "left_mtp": ("left_mtp", "lmtp", "left_toe", "ltoe", "left_big_toe"),
    "right_mtp": ("right_mtp", "rmtp", "right_toe", "rtoe", "right_big_toe"),
}

LEVEL_A_CSV_FIELDS = [
    "backend",
    "trial",
    "status",
    "rank",
    "run_dir",
    "pose3d_backend",
    "representation",
    "axis_mode",
    "convention_profile",
    "convention_source",
    "pre_axis",
    "camera_rotation_source",
    "camera_translation_source",
    "camera_translation_units",
    "uses_camera_translation",
    "time_offset_s",
    "pre_axis_determinant",
    "post_transform_determinant",
    "linear_transform_determinant",
    "proper_post_transform",
    "proper_linear_transform",
    "default_eligible",
    "left_right_swap",
    "diagnostic_only",
    "root_mode",
    "resampling",
    "evaluation_hz",
    "frames",
    "overlap_frames",
    "valid_frame_ratio",
    "primary_joint_count",
    "primary_root_centered_mpjpe_mm",
    "root_centered_rigid_mpjpe_mm",
    "root_centered_n_mpjpe_mm",
    "root_centered_similarity_mpjpe_mm",
    "pa_mpjpe_mm",
    "global_no_align_mpjpe_mm",
    "global_sequence_similarity_mpjpe_mm",
    "normal_minus_rigid_gap_mm",
    "ik_marker_rms_median_mm",
    "ik_marker_rms_mean_mm",
    "warnings",
    "error",
]


def backend_run_name(manifest: dict[str, Any], trial: str, backend: str) -> str:
    return f"{default_run_name(manifest, trial)}__{_safe_backend(backend)}"


def load_pose_artifact(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "pose3d_initial" / "pose3d_initial.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing pose3d initial artifact: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.yaml"
    return read_yaml(path) if path.exists() else {}


def load_cached_wham_timeline(run_dir: Path) -> dict[str, Any] | None:
    for candidate in _run_dir_candidates(run_dir):
        for qc_path in [candidate / "reports" / "wham_timeline_qc.json", candidate / "pose3d_initial" / "pose3d_initial_qc.json"]:
            if not qc_path.exists():
                continue
            qc = read_json(qc_path)
            timeline = qc if "raw_sync_alignment" in qc else qc.get("wham_timeline") or qc.get("wham_timeline_report")
            if (
                isinstance(timeline, dict)
                and isinstance(timeline.get("raw_sync_alignment"), dict)
                and isinstance(timeline.get("overlap"), dict)
            ):
                return timeline
    return None


def load_opensim_reference(npz_path: Path | str, json_path: Path | str | None = None) -> dict[str, Any]:
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    out = {
        "time_s": np.asarray(data["time_s"], dtype=float),
        "joints_m": np.asarray(data["joints_m"], dtype=float),
        "joint_names": [str(name) for name in data["joint_names"].tolist()],
        "source": str(npz_path),
    }
    if json_path and Path(json_path).exists():
        out["metadata"] = read_json(Path(json_path))
    return out


def compare_pose_to_opensim_reference(
    pose: dict[str, Any],
    reference: dict[str, Any],
    run_config: dict[str, Any] | None = None,
    axis_map: dict[str, str] | None = None,
    root_mode: str | None = None,
    convention_profile: str | None = None,
    evaluation_hz: float | None = None,
    diagnostic_time_offset_s: float | None = None,
    swap_reference_lr: bool = False,
    compute_pa: bool = True,
    minimal_metrics: bool = False,
    min_overlap_frames: int = 5,
    timeline_report: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    backend = str(pose.get("backend") or "unknown")
    axis_lookup = axis_map or BACKEND_AXIS_MAP
    axis_mode = axis_lookup.get(backend, "identity")
    axis_explicit = axis_map is not None and backend in axis_map
    root_mode = _resolve_root_mode(run_config, root_mode)
    warnings: list[str] = []
    if backend not in axis_lookup:
        warnings.append(f"No axis mode configured for backend {backend!r}; using identity.")

    pred_names, pred_all, convention = _canonical_prediction_joints(
        pose,
        axis_mode,
        run_config=run_config,
        convention_profile=convention_profile,
        axis_explicit=axis_explicit,
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
    time_offset_s = float(convention.get("time_offset_s") or 0.0)
    time_offset_source = "convention"
    if diagnostic_time_offset_s is not None:
        time_offset_s = float(diagnostic_time_offset_s)
        time_offset_source = "diagnostic_override"
    pred_time = prediction_time_native - time_offset_s
    pred_native = _stack_named(pred_names, pred_all, common_primary)[pose_indices]
    ref_native = _stack_named(ref_names, ref_all, common_primary)
    ref_time = np.asarray(reference["time_s"], dtype=float)
    native_valid_time = (pred_time >= np.nanmin(ref_time)) & (pred_time <= np.nanmax(ref_time))
    resampling = "reference_resampled_to_prediction_timestamps"
    if evaluation_hz is not None:
        eval_hz = float(evaluation_hz)
        if not np.isfinite(eval_hz) or eval_hz <= 0:
            raise ValueError(f"evaluation_hz must be positive when provided, got {evaluation_hz!r}.")
        time = _uniform_overlap_time(pred_time[native_valid_time], ref_time, eval_hz)
        pred = resample_timeseries(pred_time[native_valid_time], pred_native[native_valid_time], time)
        ref = resample_timeseries(ref_time, ref_native, time)
        valid_time = np.ones(time.shape[0], dtype=bool)
        resampling = "prediction_and_reference_resampled_to_uniform_timestamps"
    else:
        ref = resample_timeseries(ref_time, ref_native, pred_time)
        valid_time = native_valid_time
        pred = pred_native[valid_time]
        ref = ref[valid_time]
        time = pred_time[valid_time]
    if pred.shape[0] < min_overlap_frames:
        raise ValueError(f"Only {pred.shape[0]} frames overlap the OpenSim FK reference; need at least {min_overlap_frames}.")

    eval_indices = [common_primary.index(name) for name in error_joints]
    pred_root = _root_series(pred, common_primary, root_mode)
    ref_root = _root_series(ref, common_primary, root_mode)
    pred_centered = pred[:, eval_indices, :] - pred_root[:, None, :]
    ref_centered = ref[:, eval_indices, :] - ref_root[:, None, :]
    pred_eval = pred[:, eval_indices, :]
    ref_eval = ref[:, eval_indices, :]

    normal = sequence_mpjpe(pred_centered, ref_centered, error_joints, align="none")
    rigid, rigid_transform, pred_rigid = sequence_mpjpe_with_transform(
        pred_centered,
        ref_centered,
        error_joints,
        align="rigid",
        allow_translation=False,
    )
    if minimal_metrics:
        empty_metric = {"mpjpe_mm": None, "mpjpe_m": None, "per_joint_mpjpe_mm": {}, "per_joint_mpjpe_m": {}, "valid_ratio": 0.0}
        empty_transform: dict[str, Any] = {}
        normalized, normalized_transform, pred_normalized = empty_metric, empty_transform, np.full_like(pred_centered, np.nan)
        root_similarity, root_similarity_transform, pred_root_similarity = empty_metric, empty_transform, np.full_like(
            pred_centered, np.nan
        )
        global_similarity, global_similarity_transform, pred_global_similarity = empty_metric, empty_transform, np.full_like(
            pred_eval, np.nan
        )
        global_no_align = empty_metric
        segment_lengths: dict[str, Any] = {}
        foot_series: dict[str, np.ndarray] = {}
    else:
        normalized, normalized_transform, pred_normalized = normalized_mpjpe(pred_centered, ref_centered, error_joints)
        root_similarity, root_similarity_transform, pred_root_similarity = sequence_mpjpe_with_transform(
            pred_centered,
            ref_centered,
            error_joints,
            align="similarity",
            allow_translation=False,
        )
        global_similarity, global_similarity_transform, pred_global_similarity = sequence_mpjpe_with_transform(
            pred_eval, ref_eval, error_joints, align="similarity"
        )
        global_no_align = sequence_mpjpe(pred_eval, ref_eval, error_joints, align="none")
        segment_lengths = summarize_segment_lengths(pred[:, eval_indices, :], ref[:, eval_indices, :], error_joints)
        foot_series = foot_trajectory_series(pred[:, eval_indices, :], ref[:, eval_indices, :], error_joints, time)
    if compute_pa:
        pred_pa = per_frame_aligned_pose(pred_eval, ref_eval, align="similarity")
        pa = mpjpe(pred_pa, ref_eval, error_joints)
        pred_pa_root_centered = pred_pa - ref_root[:, None, :]
    else:
        pa = {"mpjpe_mm": None, "mpjpe_m": None, "per_joint_mpjpe_mm": {}, "per_joint_mpjpe_m": {}, "valid_ratio": 0.0}
        pred_pa_root_centered = np.full_like(pred_eval, np.nan)

    valid_frame_ratio = _valid_frame_ratio(pred_centered, ref_centered)
    normal_gap = _metric_gap(normal, rigid)
    if normal_gap is not None and normal_gap > 100.0:
        warnings.append(
            f"Normal MPJPE exceeds rigid MPJPE by {normal_gap:.3f} mm; inspect camera/reference extrinsics, "
            "axis/sign convention, left-right labels, and timing before interpreting absolute orientation."
        )

    report = {
        "status": "ok",
        "backend": backend,
        "representation": pose.get("representation"),
        "axis_mode": convention.get("axis_mode", axis_mode),
        "convention_profile": convention.get("convention_profile"),
        "convention_source": convention.get("convention_source"),
        "pre_axis": convention.get("pre_axis"),
        "camera_rotation_source": convention.get("camera_rotation_source"),
        "camera_translation_source": convention.get("camera_translation_source"),
        "camera_translation_units": convention.get("camera_translation_units"),
        "camera_translation_m": convention.get("camera_translation_m"),
        "uses_camera_translation": convention.get("uses_camera_translation"),
        "time_offset_s": time_offset_s,
        "time_offset_source": time_offset_source,
        "pre_axis_determinant": convention.get("pre_axis_determinant"),
        "post_transform_determinant": convention.get("post_transform_determinant"),
        "camera_rotation_determinant": convention.get("camera_rotation_determinant"),
        "linear_transform_determinant": convention.get("linear_transform_determinant"),
        "proper_post_transform": convention.get("proper_post_transform"),
        "proper_linear_transform": convention.get("proper_linear_transform"),
        "default_eligible": convention.get("default_eligible"),
        "left_right_swap": convention.get("left_right_swap"),
        "diagnostic_only": convention.get("diagnostic_only"),
        "root_mode": root_mode,
        "resampling": resampling,
        "evaluation_hz": float(evaluation_hz) if evaluation_hz is not None else None,
        "reference_left_right_swap": bool(swap_reference_lr),
        "frames": int(np.asarray(pose["joints_3d"]).shape[0]),
        "overlap_frames": int(pred.shape[0]),
        "prediction_fps": float(pose.get("fps") or 30.0),
        "reference_source": reference.get("source"),
        "reference_joint_names": common_primary,
        "primary_joint_names": error_joints,
        "primary_joint_count": len(error_joints),
        "valid_frame_ratio": valid_frame_ratio,
        "time_start_s": float(time[0]),
        "time_end_s": float(time[-1]),
        "timebase": _public_timebase(timebase, native_valid_time, evaluation_hz=evaluation_hz, evaluation_time=time),
        "primary_root_centered_mpjpe_mm": normal["mpjpe_mm"],
        "normal_root_centered_mpjpe_mm": normal["mpjpe_mm"],
        "root_centered_rigid_mpjpe_mm": rigid["mpjpe_mm"],
        "root_centered_n_mpjpe_mm": normalized["mpjpe_mm"],
        "root_centered_similarity_mpjpe_mm": root_similarity["mpjpe_mm"],
        "pa_mpjpe_mm": pa["mpjpe_mm"],
        "global_no_align_mpjpe_mm": global_no_align["mpjpe_mm"],
        "global_sequence_similarity_mpjpe_mm": global_similarity["mpjpe_mm"],
        "normal_minus_rigid_gap_mm": normal_gap,
        "per_joint_primary_mpjpe_mm": normal["per_joint_mpjpe_mm"],
        "per_joint_root_centered_rigid_mpjpe_mm": rigid["per_joint_mpjpe_mm"],
        "segment_lengths": segment_lengths,
        "fitted_transforms": {
            "root_centered_rigid": rigid_transform,
            "root_centered_scale_only": normalized_transform,
            "root_centered_similarity": root_similarity_transform,
            "global_sequence_similarity": global_similarity_transform,
        },
        "warnings": warnings,
    }
    series = {
        "time_s": time,
        "joint_names": np.asarray(error_joints, dtype=object),
        "root_name": root_mode,
        "prediction_eval_m": pred_eval,
        "reference_eval_m": ref_eval,
        "prediction_root_centered_m": pred_centered,
        "reference_root_centered_m": ref_centered,
        "prediction_root_centered_rigid_m": pred_rigid,
        "prediction_root_centered_normalized_m": pred_normalized,
        "prediction_root_centered_similarity_m": pred_root_similarity,
        "prediction_global_similarity_m": pred_global_similarity,
        "prediction_pa_root_centered_m": pred_pa_root_centered,
        **foot_series,
    }
    return report, series


def row_from_report(backend: str, trial: str, run_dir: Path, report: dict[str, Any], reference_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    marker_errors = (reference_meta or {}).get("ik_marker_errors") or {}
    return {
        "backend": backend,
        "trial": trial,
        "status": "valid" if report.get("status") == "ok" else report.get("status"),
        "run_dir": str(run_dir),
        "pose3d_backend": report.get("backend"),
        "representation": report.get("representation"),
        "axis_mode": report.get("axis_mode"),
        "convention_profile": report.get("convention_profile"),
        "convention_source": report.get("convention_source"),
        "pre_axis": report.get("pre_axis"),
        "camera_rotation_source": report.get("camera_rotation_source"),
        "camera_translation_source": report.get("camera_translation_source"),
        "camera_translation_units": report.get("camera_translation_units"),
        "uses_camera_translation": report.get("uses_camera_translation"),
        "time_offset_s": report.get("time_offset_s"),
        "pre_axis_determinant": report.get("pre_axis_determinant"),
        "post_transform_determinant": report.get("post_transform_determinant"),
        "linear_transform_determinant": report.get("linear_transform_determinant"),
        "proper_post_transform": report.get("proper_post_transform"),
        "proper_linear_transform": report.get("proper_linear_transform"),
        "default_eligible": report.get("default_eligible"),
        "left_right_swap": report.get("left_right_swap"),
        "diagnostic_only": report.get("diagnostic_only"),
        "root_mode": report.get("root_mode"),
        "resampling": report.get("resampling"),
        "evaluation_hz": report.get("evaluation_hz"),
        "frames": report.get("frames"),
        "overlap_frames": report.get("overlap_frames"),
        "valid_frame_ratio": report.get("valid_frame_ratio"),
        "primary_joint_count": report.get("primary_joint_count"),
        "primary_root_centered_mpjpe_mm": report.get("primary_root_centered_mpjpe_mm"),
        "root_centered_rigid_mpjpe_mm": report.get("root_centered_rigid_mpjpe_mm"),
        "root_centered_n_mpjpe_mm": report.get("root_centered_n_mpjpe_mm"),
        "root_centered_similarity_mpjpe_mm": report.get("root_centered_similarity_mpjpe_mm"),
        "pa_mpjpe_mm": report.get("pa_mpjpe_mm"),
        "global_no_align_mpjpe_mm": report.get("global_no_align_mpjpe_mm"),
        "global_sequence_similarity_mpjpe_mm": report.get("global_sequence_similarity_mpjpe_mm"),
        "normal_minus_rigid_gap_mm": report.get("normal_minus_rigid_gap_mm"),
        "ik_marker_rms_median_mm": marker_errors.get("rms_median_mm"),
        "ik_marker_rms_mean_mm": marker_errors.get("rms_mean_mm"),
        "warnings": _join(report.get("warnings")),
        "error": None,
        "per_joint_primary_mpjpe_mm": report.get("per_joint_primary_mpjpe_mm"),
        "segment_lengths": report.get("segment_lengths"),
    }


def failure_row(backend: str, trial: str, run_dir: Path | None, status: str, error: str) -> dict[str, Any]:
    return {
        "backend": backend,
        "trial": trial,
        "status": status,
        "run_dir": str(run_dir) if run_dir else None,
        "error": error,
    }


def aggregate_level_a(rows: list[dict[str, Any]], tie_threshold_mm: float = TIE_THRESHOLD_MM) -> dict[str, Any]:
    backends = sorted({str(row.get("backend")) for row in rows if row.get("backend")})
    by_backend: dict[str, Any] = {}
    for backend in backends:
        backend_rows = [row for row in rows if row.get("backend") == backend]
        valid = [
            row
            for row in backend_rows
            if row.get("status") == "valid" and row.get("primary_root_centered_mpjpe_mm") is not None
        ]
        values = [float(row["primary_root_centered_mpjpe_mm"]) for row in valid]
        by_backend[backend] = {
            "status": "valid" if valid else "missing",
            "valid_trial_count": len(valid),
            "failure_count": len(backend_rows) - len(valid),
            "median_primary_root_centered_mpjpe_mm": float(statistics.median(values)) if values else None,
            "mean_primary_root_centered_mpjpe_mm": float(statistics.fmean(values)) if values else None,
            "trials": [row.get("trial") for row in valid],
        }

    ranked = [
        (backend, stats["median_primary_root_centered_mpjpe_mm"])
        for backend, stats in by_backend.items()
        if stats.get("median_primary_root_centered_mpjpe_mm") is not None
    ]
    ranked.sort(key=lambda item: float(item[1]))
    rank = 0
    last_value = None
    for idx, (backend, value) in enumerate(ranked, start=1):
        if last_value is None or abs(float(value) - float(last_value)) > tie_threshold_mm:
            rank = idx
            last_value = value
        by_backend[backend]["rank"] = rank
    for backend, stats in by_backend.items():
        stats.setdefault("rank", None)

    failure_count = len([row for row in rows if row.get("status") != "valid"])
    return {
        "status": "failed" if not ranked else "warning" if failure_count else "ok",
        "metric": f"primary_{DEFAULT_ROOT_MODE}_centered_lower_limb_mpjpe_mm",
        "tie_threshold_mm": float(tie_threshold_mm),
        "backend_count": len(backends),
        "valid_backend_count": len(ranked),
        "backends": by_backend,
        "ranking": [{"backend": backend, "rank": by_backend[backend]["rank"], "median_mm": value} for backend, value in ranked],
        "failure_count": failure_count,
    }


def write_level_a_outputs(out_dir: Path, rows: list[dict[str, Any]], aggregate: dict[str, Any], metadata: dict[str, Any]) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    ranked_rows = _rows_with_ranks(rows, aggregate)

    csv_path = out_dir / "per_backend_trial_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEVEL_A_CSV_FIELDS)
        writer.writeheader()
        for row in ranked_rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in LEVEL_A_CSV_FIELDS})

    summary = {"metadata": metadata, "aggregate": aggregate, "rows": ranked_rows}
    json_path = out_dir / "level_a_summary.json"
    write_json(json_path, summary)

    md_path = out_dir / "level_a_summary.md"
    _write_summary_md(md_path, ranked_rows, aggregate, metadata)
    plot_outputs = _write_plots(plots_dir, ranked_rows)
    outputs = {"summary_json": str(json_path), "summary_md": str(md_path), "per_backend_trial_csv": str(csv_path)}
    outputs.update(plot_outputs)
    return outputs


def resolve_mocap_opensim_paths(manifest: dict[str, Any], trial: str) -> dict[str, Path]:
    root = Path(str((manifest.get("paths") or {}).get("root") or ""))
    model = _first_existing(
        [
            Path(str((manifest.get("ik_models") or {}).get("mocap_scaled_osim") or "")),
            root / "OpenSimData" / "Mocap" / "Model" / "LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim",
        ]
    )
    if model is None:
        candidates = sorted((root / "OpenSimData" / "Mocap" / "Model").glob("*scaled.osim"))
        model = candidates[0] if candidates else root / "OpenSimData" / "Mocap" / "Model" / "LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim"
    return {
        "model": model,
        "ik_mot": root / "OpenSimData" / "Mocap" / "IK" / f"{trial}.mot",
        "ik_marker_errors": root / "OpenSimData" / "Mocap" / "IK" / f"{trial}_ik_marker_errors.sto",
    }


def _prediction_timebase(
    pose: dict[str, Any],
    run_config: dict[str, Any] | None,
    min_overlap_frames: int,
    timeline_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from monocap_v2.core.mocap_eval import resolve_prediction_timebase

    return resolve_prediction_timebase(pose, cfg=run_config or {}, timeline_report=timeline_report, min_overlap_frames=min_overlap_frames)


def _canonical_prediction_joints(
    pose: dict[str, Any],
    axis_mode: str,
    run_config: dict[str, Any] | None = None,
    convention_profile: str | None = None,
    axis_explicit: bool = False,
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    source_names = [str(name) for name in pose.get("joint_names", [])]
    names, joints, convention = apply_level_a_convention(
        str(pose.get("backend") or "unknown"),
        source_names,
        np.asarray(pose["joints_3d"], dtype=float),
        axis_mode,
        run_config=run_config,
        convention_profile=convention_profile,
        axis_explicit=axis_explicit,
    )
    selected_names, selected_joints = _canonical_select(names, joints, derive_pelvis=True)
    return selected_names, selected_joints, convention


def _canonical_reference_joints(reference: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    return _canonical_select([str(name) for name in reference["joint_names"]], np.asarray(reference["joints_m"], dtype=float), derive_pelvis=False)


def _canonical_select(source_names: list[str], values: np.ndarray, derive_pelvis: bool) -> tuple[list[str], np.ndarray]:
    names = []
    arrays = []
    pending: dict[str, np.ndarray] = {}
    for target in PRIMARY_JOINTS + OPTIONAL_JOINTS:
        idx = find_joint(source_names, PREDICTION_ALIASES[target])
        if idx is not None:
            pending[target] = values[:, idx, :]
    if derive_pelvis and "pelvis" not in pending and "left_hip" in pending and "right_hip" in pending:
        pending["pelvis"] = 0.5 * (pending["left_hip"] + pending["right_hip"])
    for target in PRIMARY_JOINTS + OPTIONAL_JOINTS:
        if target in pending:
            names.append(target)
            arrays.append(pending[target])
    if not arrays:
        raise ValueError("No comparable canonical joints found.")
    return names, np.stack(arrays, axis=1)


def _stack_named(names: list[str], values: np.ndarray, selected: list[str]) -> np.ndarray:
    lookup = {name: idx for idx, name in enumerate(names)}
    return np.stack([values[:, lookup[name], :] for name in selected], axis=1)


def _swap_left_right_values(values: np.ndarray, names: list[str]) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    lookup = {_canon_joint_name(name): idx for idx, name in enumerate(names)}
    for left, right in [
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
        ("left_mtp", "right_mtp"),
        ("left_toe", "right_toe"),
    ]:
        li = lookup.get(_canon_joint_name(left))
        ri = lookup.get(_canon_joint_name(right))
        if li is not None and ri is not None:
            out[:, [li, ri], :] = out[:, [ri, li], :]
    return out


def _canon_joint_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(".", "_")


def _resolve_root_mode(run_config: dict[str, Any] | None, requested: str | None) -> str:
    if requested:
        return _normalize_root_mode(requested)
    cfg = ((run_config or {}).get("config") or {}).get("level_a") or {}
    return _normalize_root_mode(str(cfg.get("root_mode") or DEFAULT_ROOT_MODE))


def _normalize_root_mode(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "pelvis": "pelvis",
        "pelv": "pelvis",
        "root": "pelvis",
        "hip_midpoint": "hip_midpoint",
        "hipmidpoint": "hip_midpoint",
        "mid_hip": "hip_midpoint",
        "midhip": "hip_midpoint",
        "hips": "hip_midpoint",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported Level A root mode {value!r}; expected 'hip_midpoint' or 'pelvis'.")
    return aliases[normalized]


def _validate_root_mode(root_mode: str, names: list[str]) -> None:
    if root_mode == "pelvis" and "pelvis" not in names:
        raise ValueError("A pelvis root is required for Level A comparison.")
    if root_mode == "hip_midpoint" and not {"left_hip", "right_hip"}.issubset(names):
        raise ValueError("A left/right hip pair is required for hip-midpoint Level A comparison.")


def _root_series(values: np.ndarray, names: list[str], root_mode: str) -> np.ndarray:
    if root_mode == "pelvis":
        return values[:, names.index("pelvis"), :]
    if root_mode == "hip_midpoint":
        left = values[:, names.index("left_hip"), :]
        right = values[:, names.index("right_hip"), :]
        return 0.5 * (left + right)
    raise ValueError(f"Unsupported root mode: {root_mode}")


def _valid_frame_ratio(pred: np.ndarray, ref: np.ndarray) -> float:
    valid = np.isfinite(pred).all(axis=2) & np.isfinite(ref).all(axis=2)
    return float(np.all(valid, axis=1).mean()) if valid.size else 0.0


def _metric_gap(normal: dict[str, Any], rigid: dict[str, Any]) -> float | None:
    if normal.get("mpjpe_mm") is None or rigid.get("mpjpe_mm") is None:
        return None
    return float(normal["mpjpe_mm"] - rigid["mpjpe_mm"])


def _public_timebase(
    timebase: dict[str, Any],
    valid_time: np.ndarray,
    evaluation_hz: float | None = None,
    evaluation_time: np.ndarray | None = None,
) -> dict[str, Any]:
    out = {key: value for key, value in timebase.items() if key not in {"pose_indices", "prediction_timestamps_s"}}
    out["frames_after_reference_overlap"] = int(np.count_nonzero(valid_time))
    for key in ("raw_frame_ids", "sync_frame_ids"):
        if key in out:
            out[key] = np.asarray(out[key], dtype=int)[valid_time].tolist()
    if evaluation_hz is not None and evaluation_time is not None:
        eval_time = np.asarray(evaluation_time, dtype=float)
        out["evaluation"] = {
            "mode": "uniform_resampled",
            "evaluation_hz": float(evaluation_hz),
            "evaluation_frame_count": int(eval_time.shape[0]),
            "time_start_s": float(eval_time[0]) if eval_time.size else None,
            "time_end_s": float(eval_time[-1]) if eval_time.size else None,
        }
    return out


def _uniform_overlap_time(pred_time: np.ndarray, ref_time: np.ndarray, evaluation_hz: float) -> np.ndarray:
    pred_time = np.asarray(pred_time, dtype=float)
    ref_time = np.asarray(ref_time, dtype=float)
    pred_finite = pred_time[np.isfinite(pred_time)]
    ref_finite = ref_time[np.isfinite(ref_time)]
    if pred_finite.size < 2 or ref_finite.size < 2:
        raise ValueError("At least two finite prediction and reference timestamps are required for uniform resampling.")
    start = max(float(np.nanmin(pred_finite)), float(np.nanmin(ref_finite)))
    end = min(float(np.nanmax(pred_finite)), float(np.nanmax(ref_finite)))
    if end < start:
        raise ValueError("Prediction and OpenSim FK reference timestamps do not overlap.")
    step = 1.0 / float(evaluation_hz)
    time = np.arange(start, end + step * 0.5, step, dtype=float)
    time = time[time <= end + 1e-9]
    if time.size == 0:
        return np.asarray([start], dtype=float)
    return time


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if str(path) not in {"", "."} and path.is_file():
            return path
    return None


def _run_dir_candidates(run_dir: Path) -> list[Path]:
    candidates = [run_dir]
    name = run_dir.name
    if "__" in name:
        candidates.append(run_dir.with_name(name.split("__", 1)[0]))
    return candidates


def _rows_with_ranks(rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    rank_lookup = {backend: stats.get("rank") for backend, stats in (aggregate.get("backends") or {}).items()}
    for row in rows:
        item = dict(row)
        item["rank"] = rank_lookup.get(row.get("backend"))
        out.append(item)
    return out


def _write_summary_md(path: Path, rows: list[dict[str, Any]], aggregate: dict[str, Any], metadata: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# monocap_v2 Level A Backend Benchmark\n\n")
        f.write(f"- Status: `{aggregate.get('status')}`\n")
        f.write("- Reference: `mocap-derived OpenSim FK joint centers`\n")
        f.write("- Primary metric: `hip-midpoint-root-centered lower-limb MPJPE`, mm, no fitted transform\n")
        f.write(f"- Trials: `{', '.join(metadata.get('trials') or [])}`\n")
        f.write(f"- Tie threshold: `{aggregate.get('tie_threshold_mm')} mm`\n\n")
        f.write("## Backend Ranking\n\n")
        f.write("| Rank | Backend | Valid Trials | Median Primary MPJPE (mm) |\n")
        f.write("|---:|---|---:|---:|\n")
        for item in aggregate.get("ranking") or []:
            stats = (aggregate.get("backends") or {}).get(item["backend"], {})
            f.write(f"| {item.get('rank')} | {item.get('backend')} | {stats.get('valid_trial_count')} | {_fmt(item.get('median_mm'))} |\n")
        f.write("\n## Per Trial\n\n")
        f.write("| Backend | Trial | Status | Convention | Eval Hz | Root | Primary | Rigid | N-MPJPE | PA | Global Raw | Similarity | IK RMS | Warnings |\n")
        f.write("|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row.get('backend')} | {row.get('trial')} | {row.get('status')} | "
                f"{row.get('convention_profile') or ''} | "
                f"{_fmt(row.get('evaluation_hz'))} | "
                f"{row.get('root_mode') or ''} | "
                f"{_fmt(row.get('primary_root_centered_mpjpe_mm'))} | {_fmt(row.get('root_centered_rigid_mpjpe_mm'))} | "
                f"{_fmt(row.get('root_centered_n_mpjpe_mm'))} | {_fmt(row.get('pa_mpjpe_mm'))} | "
                f"{_fmt(row.get('global_no_align_mpjpe_mm'))} | {_fmt(row.get('global_sequence_similarity_mpjpe_mm'))} | "
                f"{_fmt(row.get('ik_marker_rms_median_mm'))} | "
                f"{row.get('warnings') or row.get('error') or ''} |\n"
            )


def _write_plots(plots_dir: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    valid = [row for row in rows if row.get("status") == "valid" and row.get("primary_root_centered_mpjpe_mm") is not None]
    outputs = {}
    if valid:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [f"{row['backend']}:{row['trial']}" for row in valid]
        values = [float(row["primary_root_centered_mpjpe_mm"]) for row in valid]
        ax.bar(np.arange(len(values)), values, color="#4C78A8")
        ax.set_ylabel("Primary MPJPE (mm)")
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        fig.tight_layout()
        path = plots_dir / "backend_mpjpe_by_trial.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs["backend_mpjpe_by_trial_plot"] = str(path)

        per_joint: dict[str, list[float]] = {}
        for row in valid:
            for joint, value in (row.get("per_joint_primary_mpjpe_mm") or {}).items():
                if value is not None:
                    per_joint.setdefault(str(joint), []).append(float(value))
        if per_joint:
            fig, ax = plt.subplots(figsize=(7, 4))
            names = list(per_joint)
            medians = [float(np.nanmedian(per_joint[name])) for name in names]
            ax.bar(np.arange(len(names)), medians, color="#59A14F")
            ax.set_ylabel("Median per-joint MPJPE (mm)")
            ax.set_xticks(np.arange(len(names)))
            ax.set_xticklabels(names, rotation=35, ha="right")
            fig.tight_layout()
            path = plots_dir / "per_joint_errors.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            outputs["per_joint_errors_plot"] = str(path)
    return outputs


def _safe_backend(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-_") or "backend"


def _join(values: Any) -> str:
    if not values:
        return ""
    if isinstance(values, str):
        return values
    return "; ".join(str(value) for value in values)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return _join(value) if not isinstance(value, dict) else "; ".join(f"{k}={_fmt(v)}" for k, v in value.items())
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def reference_marker_error_summary(path: Path | None) -> dict[str, Any]:
    return read_ik_marker_error_summary(path)
