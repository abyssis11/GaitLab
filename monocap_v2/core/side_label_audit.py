from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from monocap_v2.core.backend_convention_audit import parse_benchmark_dirs, relabel_pose_left_right
from monocap_v2.core.geometry import find_joint
from monocap_v2.core.level_a_benchmark import (
    compare_pose_to_opensim_reference,
    load_cached_wham_timeline,
    load_opensim_reference,
    load_pose_artifact,
    load_run_config,
)
from monocap_v2.core.logging_utils import read_json, write_json
from monocap_v2.core.mocap_eval import resolve_prediction_timebase
from monocap_v2.core.side_phase_audit import compare_side_signal_sets, side_motion_signals_2d


DEFAULT_SUMMARY_CSV = Path("monocap_v2/benchmarks/cam0_cam1_walking_convention_summary/best_pa_by_case.csv")
DEFAULT_OUT_DIR = Path("monocap_v2/benchmarks/cam0_cam1_walking_side_label_audit")
DEFAULT_BENCHMARK_DIRS = (
    "Cam1=monocap_v2/benchmarks/subject7_walking_level_a,"
    "Cam0=monocap_v2/benchmarks/subject7_walking_level_a_cam0"
)

SIDE_JOINTS = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
MAPPING_SPECS = [
    ("no_swap", False, False),
    ("swap_model_only", True, False),
    ("swap_gt_only", False, True),
    ("swap_both", True, True),
]

SIDE_LABEL_ROW_FIELDS = [
    "camera",
    "trial",
    "backend",
    "evaluation_hz",
    "mapping",
    "swap_model_lr",
    "swap_reference_lr",
    "recommendation",
    "status",
    "evidence_score",
    "score_margin_to_next",
    "video_model_margin",
    "video_reference_margin",
    "model_reference_phase_margin",
    "model_lateral_dot",
    "reference_lateral_dot",
    "raw_primary_mm",
    "rigid_mm",
    "pa_mm",
    "axis",
    "time_offset_s",
    "overlap_frames",
    "warnings",
    "error",
]


def run_side_label_audit(
    summary_csv: Path = DEFAULT_SUMMARY_CSV,
    benchmark_dirs: dict[str, Path] | str = DEFAULT_BENCHMARK_DIRS,
    out_dir: Path = DEFAULT_OUT_DIR,
    camera: str | None = None,
    trial: str | None = None,
    backend: str | None = None,
    evaluation_hz: float | None = None,
    write_plots: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmark_lookup = parse_benchmark_dirs(benchmark_dirs) if isinstance(benchmark_dirs, str) else benchmark_dirs
    rows = _load_best_rows(Path(summary_csv), camera=camera, trial=trial, backend=backend, evaluation_hz=evaluation_hz)
    case_reports: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            report = run_side_label_case(row, benchmark_lookup, out_dir=out_dir, write_plot=write_plots)
        except Exception as exc:
            report = _failed_case_report(row, exc)
            write_json(out_dir / _case_json_name(row), report)
        case_reports.append(report)
        flat_rows.extend(report.get("rows") or [])

    outputs = {
        "rows_csv": str(out_dir / "side_label_rows.csv"),
        "summary_md": str(out_dir / "side_label_summary.md"),
    }
    _write_rows_csv(Path(outputs["rows_csv"]), flat_rows)
    summary = _summary(case_reports, outputs)
    _write_summary_md(Path(outputs["summary_md"]), summary, case_reports)
    write_json(out_dir / "side_label_summary.json", summary)
    return summary


def run_side_label_case(
    row: dict[str, Any],
    benchmark_dirs: dict[str, Path],
    out_dir: Path,
    write_plot: bool = True,
) -> dict[str, Any]:
    camera = str(row["camera"])
    trial = str(row["trial"])
    backend = str(row["backend"])
    hz = float(row["evaluation_hz"])
    axis = str(row["axis"])
    offset = float(row["time_offset_s"])
    benchmark_dir = Path(benchmark_dirs[camera])
    run_dir = _run_dir_for_row(benchmark_dir, trial, backend)

    pose = load_pose_artifact(run_dir)
    run_config = load_run_config(run_dir)
    timeline = load_cached_wham_timeline(run_dir)
    reference = load_opensim_reference(
        benchmark_dir / "reference" / f"opensim_fk_{trial}.npz",
        benchmark_dir / "reference" / f"opensim_fk_{trial}.json",
    )

    baseline_report, baseline_series = _compare_mapping(
        pose,
        reference,
        run_config,
        timeline,
        backend,
        axis,
        hz,
        offset,
        swap_model=False,
        swap_reference=False,
    )
    time_s = np.asarray(baseline_series["time_s"], dtype=float)
    joint_names = [str(name) for name in baseline_series["joint_names"].tolist()]
    pred = np.asarray(baseline_series["prediction_eval_m"], dtype=float)
    ref = np.asarray(baseline_series["reference_eval_m"], dtype=float)
    video_signals, video_report = video_side_signals_for_level_a_times(pose, run_config, timeline, time_s, offset)

    metrics_by_mapping: dict[str, dict[str, Any]] = {"no_swap": _metrics_from_report(baseline_report)}
    for mapping, swap_model, swap_reference in MAPPING_SPECS[1:]:
        try:
            report, _series = _compare_mapping(
                pose,
                reference,
                run_config,
                timeline,
                backend,
                axis,
                hz,
                offset,
                swap_model=swap_model,
                swap_reference=swap_reference,
            )
            metrics_by_mapping[mapping] = _metrics_from_report(report)
        except Exception as exc:
            metrics_by_mapping[mapping] = {"status": "failed", "error": str(exc)}

    decision = evaluate_side_label_mappings(
        time_s=time_s,
        joint_names=joint_names,
        prediction_eval_m=pred,
        reference_eval_m=ref,
        video_signals=video_signals if video_report.get("status") == "ok" else None,
        metrics_by_mapping=metrics_by_mapping,
    )

    case_id = _case_stem(row)
    plot_path = out_dir / f"{case_id}_phase.png"
    plot_report = write_side_label_plot(plot_path, decision) if write_plot else {"status": "skipped"}
    case_report = {
        "status": "warning" if _case_warnings(decision, video_report, plot_report) else "ok",
        "camera": camera,
        "trial": trial,
        "backend": backend,
        "evaluation_hz": hz,
        "run_dir": str(run_dir),
        "axis": axis,
        "time_offset_s": offset,
        "cached_best_row_swaps": {
            "model_left_right_swap": _bool(row.get("model_left_right_swap")),
            "reference_left_right_swap": _bool(row.get("reference_left_right_swap")),
        },
        "recommendation": decision["recommendation"],
        "recommendation_reason": decision["recommendation_reason"],
        "best_mapping": decision.get("best_mapping"),
        "score_margin_to_next": decision.get("score_margin_to_next"),
        "walking_direction": decision["walking_direction"],
        "video_side": video_report,
        "rows": [
            _case_row(row, item, decision["recommendation"], decision.get("score_margin_to_next"), decision.get("best_mapping"))
            for item in decision["rows"]
        ],
        "outputs": {
            "json": str(out_dir / f"{case_id}.json"),
            "plot": str(plot_path) if plot_path.exists() else None,
        },
        "warnings": _case_warnings(decision, video_report, plot_report),
    }
    write_json(out_dir / f"{case_id}.json", case_report)
    return case_report


def evaluate_side_label_mappings(
    time_s: np.ndarray,
    joint_names: list[str],
    prediction_eval_m: np.ndarray,
    reference_eval_m: np.ndarray,
    video_signals: dict[str, Any] | None = None,
    metrics_by_mapping: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    names = [str(name) for name in joint_names]
    pred = np.asarray(prediction_eval_m, dtype=float)
    ref = np.asarray(reference_eval_m, dtype=float)
    direction = walking_direction(ref, names)
    direction_source = "reference"
    if direction["status"] != "ok":
        direction = walking_direction(pred, names)
        direction_source = "prediction"

    rows: list[dict[str, Any]] = []
    for mapping, swap_model, swap_reference in MAPPING_SPECS:
        pred_item = swap_side_values(pred, names) if swap_model else pred
        ref_item = swap_side_values(ref, names) if swap_reference else ref
        if direction["status"] == "ok":
            forward = np.asarray(direction["forward"], dtype=float)
            model_signals = forward_motion_signals(pred_item, names, time_s, forward, source="model_3d")
            ref_signals = forward_motion_signals(ref_item, names, time_s, forward, source="reference_3d")
            model_geometry = hip_lateral_geometry(pred_item, names, forward)
            ref_geometry = hip_lateral_geometry(ref_item, names, forward)
            model_ref = compare_side_signal_sets(ref_signals, model_signals)
            video_model = compare_side_signal_sets(video_signals, model_signals) if video_signals is not None else _missing_agreement()
            video_ref = compare_side_signal_sets(video_signals, ref_signals) if video_signals is not None else _missing_agreement()
        else:
            model_signals = ref_signals = {}
            model_geometry = ref_geometry = {"status": "skipped", "reason": direction.get("reason")}
            model_ref = video_model = video_ref = _missing_agreement()
        metrics = (metrics_by_mapping or {}).get(mapping) or {}
        rows.append(
            {
                "mapping": mapping,
                "swap_model_lr": bool(swap_model),
                "swap_reference_lr": bool(swap_reference),
                "video_model_agreement": video_model,
                "video_reference_agreement": video_ref,
                "model_reference_phase_agreement": model_ref,
                "model_geometry": model_geometry,
                "reference_geometry": ref_geometry,
                "model_signals": _compact_signals(model_signals),
                "reference_signals": _compact_signals(ref_signals),
                "metrics": metrics,
                "evidence_score": evidence_score(video_model, video_ref, model_ref, model_geometry, ref_geometry),
            }
        )
    recommendation = recommend_side_label_mapping(rows)
    return {
        "recommendation": recommendation["recommendation"],
        "recommendation_reason": recommendation["reason"],
        "walking_direction": {**direction, "source": direction_source},
        "rows": rows,
        "best_mapping": recommendation.get("best_mapping"),
        "score_margin_to_next": recommendation.get("score_margin_to_next"),
    }


def video_side_signals_for_level_a_times(
    pose: dict[str, Any],
    run_config: dict[str, Any],
    timeline: dict[str, Any] | None,
    time_s: np.ndarray,
    time_offset_s: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    pose2d = pose.get("pose2d")
    if not isinstance(pose2d, dict):
        return None, {"status": "skipped", "reason": "pose artifact has no pose2d field"}
    xy = np.asarray(pose2d.get("xy"), dtype=float)
    if xy.ndim != 3 or xy.shape[-1] != 2:
        return None, {"status": "skipped", "reason": "pose2d xy is unavailable or not [T, J, 2]"}
    indices = _pose2d_side_indices(pose2d, xy.shape[1])
    if indices is None:
        return None, {"status": "skipped", "reason": "pose2d lacks hip/knee/ankle side labels"}
    conf = np.asarray(pose2d.get("confidence"), dtype=float)
    if conf.ndim != 2 or conf.shape[:2] != xy.shape[:2]:
        conf = np.ones(xy.shape[:2], dtype=float)
    timebase = resolve_prediction_timebase(pose, cfg=run_config, timeline_report=timeline)
    pose_indices = np.asarray(timebase["pose_indices"], dtype=int)
    native_time = np.asarray(timebase["prediction_timestamps_s"], dtype=float)
    valid = (pose_indices >= 0) & (pose_indices < xy.shape[0]) & np.isfinite(native_time)
    pose_indices = pose_indices[valid]
    native_time = native_time[valid]
    if pose_indices.size < 2:
        return None, {"status": "skipped", "reason": "fewer than two pose2d frames overlap the Level A timebase"}
    selected_xy = xy[pose_indices][:, indices, :]
    selected_conf = conf[pose_indices][:, indices]
    sample_time = np.asarray(time_s, dtype=float) + float(time_offset_s)
    sampled_xy = _resample_xy(native_time, selected_xy, sample_time)
    sampled_conf = _resample_xy(native_time, selected_conf[..., None], sample_time)[..., 0]
    sampled_xy[(~np.isfinite(sampled_conf)) | (sampled_conf < 0.2)] = np.nan
    try:
        signals = side_motion_signals_2d(sampled_xy, np.asarray(time_s, dtype=float))
    except Exception as exc:
        return None, {"status": "skipped", "reason": f"could not derive 2D side signals: {exc}"}
    return signals, {
        "status": "ok",
        "source": "pose3d_initial.pose2d",
        "joint_names": SIDE_JOINTS,
        "finite_ratio": float(np.isfinite(sampled_xy).all(axis=2).mean()) if sampled_xy.size else 0.0,
        "sampled_frames": int(sampled_xy.shape[0]),
        "sample_time_offset_s": float(time_offset_s),
    }


def walking_direction(values: np.ndarray, joint_names: list[str]) -> dict[str, Any]:
    root = _hip_midpoint(values, joint_names)
    if root is None:
        return {"status": "skipped", "reason": "missing left/right hips"}
    horizontal = np.asarray(root, dtype=float).copy()
    horizontal[:, 1] = 0.0
    finite = np.isfinite(horizontal).all(axis=1)
    if np.count_nonzero(finite) < 3:
        return {"status": "skipped", "reason": "fewer than three finite hip-midpoint frames"}
    pts = horizontal[finite]
    displacement = pts[-1] - pts[0]
    direction = displacement.copy()
    if float(np.linalg.norm(direction)) < 1e-6:
        centered = pts - np.nanmean(pts, axis=0)
        cov = centered.T @ centered
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            direction = eigvecs[:, int(np.nanargmax(eigvals))]
            if float(np.dot(direction, displacement)) < 0.0:
                direction = -direction
        except Exception:
            return {"status": "skipped", "reason": "could not estimate horizontal walking direction"}
    direction[1] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return {"status": "skipped", "reason": "horizontal hip-midpoint motion is too small"}
    forward = direction / norm
    return {
        "status": "ok",
        "forward": forward,
        "horizontal_displacement_m": float(np.linalg.norm(displacement)),
        "method": "hip midpoint first-to-last horizontal displacement with PCA fallback",
    }


def hip_lateral_geometry(values: np.ndarray, joint_names: list[str], forward: np.ndarray) -> dict[str, Any]:
    left_idx = find_joint(joint_names, ("left_hip", "lhip"))
    right_idx = find_joint(joint_names, ("right_hip", "rhip"))
    if left_idx is None or right_idx is None:
        return {"status": "skipped", "reason": "missing left/right hips"}
    lateral = np.asarray(values[:, right_idx, :] - values[:, left_idx, :], dtype=float)
    lateral[:, 1] = 0.0
    mean = np.nanmean(lateral[np.isfinite(lateral).all(axis=1)], axis=0) if np.isfinite(lateral).all(axis=1).any() else None
    if mean is None or not np.isfinite(mean).all() or float(np.linalg.norm(mean)) < 1e-9:
        return {"status": "skipped", "reason": "hip lateral vector is unavailable"}
    actual_right = mean / float(np.linalg.norm(mean))
    expected_right = np.cross(np.asarray([0.0, 1.0, 0.0]), np.asarray(forward, dtype=float))
    expected_right[1] = 0.0
    if float(np.linalg.norm(expected_right)) < 1e-9:
        return {"status": "skipped", "reason": "expected right vector is unavailable"}
    expected_right = expected_right / float(np.linalg.norm(expected_right))
    dot = float(np.dot(actual_right, expected_right))
    if dot >= 0.25:
        preference = "labels_consistent"
    elif dot <= -0.25:
        preference = "labels_inverted"
    else:
        preference = "uncertain"
    return {
        "status": "ok",
        "right_axis_dot": dot,
        "preference": preference,
        "actual_right": actual_right,
        "expected_right": expected_right,
        "method": "dot(mean(RHIP-LHIP), cross(Y_up, walking_forward))",
    }


def forward_motion_signals(
    values: np.ndarray,
    joint_names: list[str],
    time_s: np.ndarray,
    forward: np.ndarray,
    source: str,
) -> dict[str, Any]:
    names = [str(name) for name in joint_names]
    root = _hip_midpoint(values, names)
    if root is None:
        raise ValueError("left/right hips are required for forward motion signals")
    lookup = {name: find_joint(names, (name, name.replace("left_", "l").replace("right_", "r"))) for name in SIDE_JOINTS}
    if any(idx is None for idx in lookup.values()):
        missing = [name for name, idx in lookup.items() if idx is None]
        raise ValueError(f"missing side joints: {', '.join(missing)}")
    fwd = np.asarray(forward, dtype=float)
    left_knee = _forward_projection(values[:, lookup["left_knee"], :] - root, fwd)
    right_knee = _forward_projection(values[:, lookup["right_knee"], :] - root, fwd)
    left_ankle = _forward_projection(values[:, lookup["left_ankle"], :] - root, fwd)
    right_ankle = _forward_projection(values[:, lookup["right_ankle"], :] - root, fwd)
    left = _zscore_nan(left_knee) + _zscore_nan(left_ankle)
    right = _zscore_nan(right_knee) + _zscore_nan(right_ankle)
    return {
        "source": source,
        "left_signal": left,
        "right_signal": right,
        "finite_ratio": {
            "left": float(np.isfinite(left).mean()) if left.size else 0.0,
            "right": float(np.isfinite(right).mean()) if right.size else 0.0,
        },
        "method": "knee+ankle forward displacement along walking direction",
    }


def swap_side_values(values: np.ndarray, joint_names: list[str]) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    for left, right in [
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
        ("left_mtp", "right_mtp"),
    ]:
        li = find_joint(joint_names, (left, left.replace("left_", "l")))
        ri = find_joint(joint_names, (right, right.replace("right_", "r")))
        if li is not None and ri is not None:
            out[:, [li, ri], :] = out[:, [ri, li], :]
    return out


def evidence_score(
    video_model: dict[str, Any],
    video_reference: dict[str, Any],
    model_reference: dict[str, Any],
    model_geometry: dict[str, Any],
    reference_geometry: dict[str, Any],
) -> float | None:
    terms: list[float] = []
    for item, weight in [(video_model, 2.0), (video_reference, 2.0), (model_reference, 1.0)]:
        value = item.get("same_minus_opposite")
        if value is not None and np.isfinite(value):
            terms.append(weight * float(value))
    for item in [model_geometry, reference_geometry]:
        value = item.get("right_axis_dot")
        if value is not None and np.isfinite(value):
            terms.append(0.75 * float(value))
    return float(np.sum(terms)) if terms else None


def recommend_side_label_mapping(rows: list[dict[str, Any]], min_margin: float = 0.15) -> dict[str, Any]:
    valid = [row for row in rows if row.get("evidence_score") is not None and np.isfinite(row["evidence_score"])]
    if not valid:
        return {"recommendation": "unresolved", "reason": "no finite side-label evidence", "best_mapping": None}
    ordered = sorted(valid, key=lambda row: float(row["evidence_score"]), reverse=True)
    best = ordered[0]
    second = ordered[1] if len(ordered) > 1 else None
    margin = None if second is None else float(best["evidence_score"] - second["evidence_score"])
    if second is not None and margin is not None and margin < min_margin:
        return {
            "recommendation": "unresolved",
            "reason": f"best mapping {best['mapping']} beats next candidate by only {margin:.3f}",
            "best_mapping": best["mapping"],
            "score_margin_to_next": margin,
        }
    mapping = str(best["mapping"])
    if mapping == "swap_model_only":
        recommendation = "model_swap_likely"
        best_mapping = mapping
        reason = f"highest non-MPJPE evidence score is {mapping}"
    elif mapping == "swap_gt_only":
        recommendation = "gt_swap_likely"
        best_mapping = mapping
        reason = f"highest non-MPJPE evidence score is {mapping}"
    elif mapping in {"no_swap", "swap_both"}:
        recommendation = "both_or_neither_equivalent"
        best_mapping = "no_swap"
        reason = (
            f"highest non-MPJPE evidence score is {mapping}; policy prefers no_swap when both/neither are equivalent"
        )
    else:
        recommendation = "unresolved"
        best_mapping = mapping
        reason = f"highest non-MPJPE evidence score is {mapping}"
    return {
        "recommendation": recommendation,
        "reason": reason,
        "best_mapping": best_mapping,
        "highest_scoring_mapping": mapping,
        "score_margin_to_next": margin,
    }


def write_side_label_plot(path: Path, decision: dict[str, Any]) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "skipped", "reason": f"matplotlib unavailable: {exc}"}

    rows = decision.get("rows") or []
    if not rows:
        return {"status": "skipped", "reason": "no side-label rows to plot"}
    best = max([row for row in rows if row.get("evidence_score") is not None], key=lambda row: float(row["evidence_score"]), default=rows[0])
    model = best.get("model_signals") or {}
    ref = best.get("reference_signals") or {}
    if not model.get("left_signal") or not ref.get("left_signal"):
        return {"status": "skipped", "reason": "best row has no plottable phase signals"}
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(model["left_signal"]))
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    axes[0].plot(t, model["left_signal"], color="#cc3333", label="model left")
    axes[0].plot(t, model["right_signal"], color="#2f6fdd", label="model right")
    axes[0].set_title(f"Model phase, best mapping: {best.get('mapping')}")
    axes[1].plot(t, ref["left_signal"], color="#cc3333", label="GT left")
    axes[1].plot(t, ref["right_signal"], color="#2f6fdd", label="GT right")
    axes[1].set_title("OpenSim FK phase")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("evaluation frame")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"status": "ok", "output": str(path)}


def _compare_mapping(
    pose: dict[str, Any],
    reference: dict[str, Any],
    run_config: dict[str, Any],
    timeline: dict[str, Any] | None,
    backend: str,
    axis: str,
    evaluation_hz: float,
    offset: float,
    swap_model: bool,
    swap_reference: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    candidate_pose = relabel_pose_left_right(pose) if swap_model else pose
    return compare_pose_to_opensim_reference(
        candidate_pose,
        reference,
        run_config=run_config,
        axis_map={backend: axis},
        timeline_report=timeline,
        evaluation_hz=float(evaluation_hz),
        diagnostic_time_offset_s=float(offset),
        swap_reference_lr=bool(swap_reference),
        compute_pa=True,
        minimal_metrics=True,
    )


def _metrics_from_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "raw_primary_mm": report.get("primary_root_centered_mpjpe_mm"),
        "rigid_mm": report.get("root_centered_rigid_mpjpe_mm"),
        "pa_mm": report.get("pa_mpjpe_mm"),
        "overlap_frames": report.get("overlap_frames"),
        "warnings": report.get("warnings") or [],
    }


def _pose2d_side_indices(pose2d: dict[str, Any], joint_count: int) -> list[int] | None:
    names = [str(name) for name in pose2d.get("names") or []]
    indices = []
    for name in SIDE_JOINTS:
        idx = find_joint(names, (name, name.replace("left_", "l").replace("right_", "r")))
        if idx is None:
            break
        indices.append(idx)
    if len(indices) == len(SIDE_JOINTS):
        return indices
    if joint_count >= 17:
        return [11, 12, 13, 14, 15, 16]
    return None


def _resample_xy(t_src: np.ndarray, values: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full((len(t_dst), arr.shape[1], arr.shape[2]), np.nan, dtype=float)
    for joint_idx in range(arr.shape[1]):
        for coord_idx in range(arr.shape[2]):
            series = arr[:, joint_idx, coord_idx]
            valid = np.isfinite(series) & np.isfinite(t_src)
            if np.count_nonzero(valid) < 2:
                continue
            ts = np.asarray(t_src, dtype=float)[valid]
            xs = series[valid]
            interp = np.interp(t_dst, ts, xs)
            interp[(t_dst < ts.min()) | (t_dst > ts.max())] = np.nan
            out[:, joint_idx, coord_idx] = interp
    return out


def _hip_midpoint(values: np.ndarray, joint_names: list[str]) -> np.ndarray | None:
    left = find_joint(joint_names, ("left_hip", "lhip"))
    right = find_joint(joint_names, ("right_hip", "rhip"))
    if left is None or right is None:
        return None
    return 0.5 * (np.asarray(values, dtype=float)[:, left, :] + np.asarray(values, dtype=float)[:, right, :])


def _forward_projection(values: np.ndarray, forward: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(values, dtype=float) * np.asarray(forward, dtype=float)[None, :], axis=1)


def _zscore_nan(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.zeros(arr.shape, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return out
    std = float(np.nanstd(finite))
    if std <= 1e-12:
        return out
    out[np.isfinite(arr)] = (arr[np.isfinite(arr)] - float(np.nanmean(finite))) / std
    return out


def _missing_agreement() -> dict[str, Any]:
    return {"same_label_correlation": None, "opposite_label_correlation": None, "same_minus_opposite": None, "preferred": "unavailable"}


def _compact_signals(signals: dict[str, Any]) -> dict[str, Any]:
    if not signals:
        return {}
    return {
        "source": signals.get("source"),
        "method": signals.get("method"),
        "finite_ratio": signals.get("finite_ratio"),
        "left_signal": np.asarray(signals.get("left_signal"), dtype=float).tolist(),
        "right_signal": np.asarray(signals.get("right_signal"), dtype=float).tolist(),
    }


def _case_row(
    source_row: dict[str, Any],
    item: dict[str, Any],
    recommendation: str,
    score_margin_to_next: float | None,
    best_mapping: str | None,
) -> dict[str, Any]:
    metrics = item.get("metrics") or {}
    return {
        "camera": source_row.get("camera"),
        "trial": source_row.get("trial"),
        "backend": source_row.get("backend"),
        "evaluation_hz": source_row.get("evaluation_hz"),
        "mapping": item.get("mapping"),
        "swap_model_lr": item.get("swap_model_lr"),
        "swap_reference_lr": item.get("swap_reference_lr"),
        "recommendation": recommendation,
        "status": metrics.get("status") or "ok",
        "evidence_score": item.get("evidence_score"),
        "score_margin_to_next": score_margin_to_next if item.get("mapping") == best_mapping else None,
        "video_model_margin": (item.get("video_model_agreement") or {}).get("same_minus_opposite"),
        "video_reference_margin": (item.get("video_reference_agreement") or {}).get("same_minus_opposite"),
        "model_reference_phase_margin": (item.get("model_reference_phase_agreement") or {}).get("same_minus_opposite"),
        "model_lateral_dot": (item.get("model_geometry") or {}).get("right_axis_dot"),
        "reference_lateral_dot": (item.get("reference_geometry") or {}).get("right_axis_dot"),
        "raw_primary_mm": metrics.get("raw_primary_mm"),
        "rigid_mm": metrics.get("rigid_mm"),
        "pa_mm": metrics.get("pa_mm"),
        "axis": source_row.get("axis"),
        "time_offset_s": source_row.get("time_offset_s"),
        "overlap_frames": metrics.get("overlap_frames"),
        "warnings": "; ".join(str(w) for w in metrics.get("warnings") or []),
        "error": metrics.get("error") or "",
    }


def _case_warnings(decision: dict[str, Any], video_report: dict[str, Any], plot_report: dict[str, Any]) -> list[str]:
    warnings = []
    if video_report.get("status") != "ok":
        warnings.append(f"Video-side 2D phase skipped: {video_report.get('reason')}")
    if (decision.get("walking_direction") or {}).get("status") != "ok":
        warnings.append(f"Walking direction unavailable: {(decision.get('walking_direction') or {}).get('reason')}")
    if plot_report.get("status") not in {"ok", "skipped"}:
        warnings.extend(str(w) for w in plot_report.get("warnings") or [])
    return warnings


def _summary(case_reports: list[dict[str, Any]], outputs: dict[str, str]) -> dict[str, Any]:
    valid = [case for case in case_reports if case.get("status") in {"ok", "warning"}]
    counts: dict[str, int] = {}
    for case in valid:
        rec = str(case.get("recommendation"))
        counts[rec] = counts.get(rec, 0) + 1
    return {
        "status": "failed" if not valid and case_reports else "warning" if any(case.get("status") == "warning" for case in valid) else "ok",
        "case_count": len(case_reports),
        "valid_case_count": len(valid),
        "recommendation_counts": counts,
        "outputs": outputs,
        "notes": [
            "Side-label recommendations are diagnostic only.",
            "MPJPE is reported for context but is not used to choose the label recommendation.",
        ],
    }


def _write_summary_md(path: Path, summary: dict[str, Any], case_reports: list[dict[str, Any]]) -> None:
    lines = [
        "# Direction-Aware Side Label Audit",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Cases: `{summary.get('case_count')}`",
        "- Recommendations are diagnostic-only and do not change artifacts or benchmark defaults.",
        "- MPJPE values are reported for context, not used to decide side labels.",
        "",
        "## Cases",
        "",
        "| Camera | Trial | Backend | Hz | Recommendation | Best Mapping | Axis | Offset | Warnings |",
        "|---|---|---|---:|---|---|---|---:|---|",
    ]
    for case in case_reports:
        best = _best_case_mapping(case)
        lines.append(
            f"| {case.get('camera')} | {case.get('trial')} | {case.get('backend')} | {_fmt(case.get('evaluation_hz'))} | "
            f"{case.get('recommendation')} | {best or ''} | `{case.get('axis')}` | {_fmt(case.get('time_offset_s'), 3)} | "
            f"{'; '.join(case.get('warnings') or [])} |"
        )
    lines.extend(["", "## Recommendation Counts", "", "| Recommendation | Count |", "|---|---:|"])
    for key, value in sorted((summary.get("recommendation_counts") or {}).items()):
        lines.append(f"| {key} | {value} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _best_case_mapping(case: dict[str, Any]) -> str | None:
    if case.get("best_mapping"):
        return str(case["best_mapping"])
    rows = case.get("rows") or []
    valid = [row for row in rows if row.get("evidence_score") not in {None, ""}]
    if not valid:
        return None
    return str(max(valid, key=lambda row: float(row["evidence_score"])).get("mapping"))


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SIDE_LABEL_ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in SIDE_LABEL_ROW_FIELDS})


def _load_best_rows(
    path: Path,
    camera: str | None,
    trial: str | None,
    backend: str | None,
    evaluation_hz: float | None,
) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    out = []
    for row in rows:
        if str(row.get("ranking")) != "best_pa":
            continue
        if camera and str(row.get("camera")) != camera:
            continue
        if trial and str(row.get("trial")) != trial:
            continue
        if backend and str(row.get("backend")) != backend:
            continue
        if evaluation_hz is not None and abs(float(row.get("evaluation_hz") or "nan") - float(evaluation_hz)) > 1e-9:
            continue
        out.append(row)
    return out


def _run_dir_for_row(benchmark_dir: Path, trial: str, backend: str) -> Path:
    summary = read_json(benchmark_dir / "level_a_summary.json")
    for row in summary.get("rows") or []:
        if row.get("trial") == trial and row.get("backend") == backend and row.get("status") == "valid":
            return Path(str(row["run_dir"]))
    raise ValueError(f"No valid run dir for {backend}/{trial} in {benchmark_dir}")


def _failed_case_report(row: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "camera": row.get("camera"),
        "trial": row.get("trial"),
        "backend": row.get("backend"),
        "evaluation_hz": row.get("evaluation_hz"),
        "recommendation": "unresolved",
        "rows": [],
        "warnings": [],
        "error": str(exc),
    }


def _case_stem(row: dict[str, Any]) -> str:
    hz = f"{float(row['evaluation_hz']):g}hz"
    return f"side_label_{row['camera']}_{row['trial']}_{row['backend']}_{hz}"


def _case_json_name(row: dict[str, Any]) -> str:
    return f"{_case_stem(row)}.json"


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}f}"


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    return value
