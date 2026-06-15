from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from monocap_v2.core.geometry import find_joint
from monocap_v2.core.gt_video_overlay import project_opensim_fk_to_pixels
from monocap_v2.core.level_a_benchmark import (
    compare_pose_to_opensim_reference,
    load_cached_wham_timeline,
    load_opensim_reference,
    load_pose_artifact,
    load_run_config,
)
from monocap_v2.core.logging_utils import read_json, write_json
from monocap_v2.core.mocap_eval import resample_timeseries, resolve_prediction_timebase


COCO17 = {
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

SIDE_JOINTS = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
PHASE_JOINTS = ["left_knee", "right_knee", "left_ankle", "right_ankle"]


def run_wham_side_phase_audit(
    benchmark_dir: Path,
    trial: str,
    evaluation_hz: float | None = 60.0,
    convention_profile: str | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    benchmark_dir = Path(benchmark_dir)
    out_dir = Path(out_dir) if out_dir else benchmark_dir / "side_phase"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"wham_side_phase_{trial}.json"
    out_csv = out_dir / f"wham_side_phase_{trial}.csv"
    out_png = out_dir / f"wham_side_phase_{trial}.png"
    out_mp4 = out_dir / f"wham_side_phase_{trial}_projection_overlay.mp4"

    row = _load_wham_summary_row(benchmark_dir, trial)
    run_dir = Path(str(row["run_dir"]))
    pose = load_pose_artifact(run_dir)
    run_config = load_run_config(run_dir)
    timeline = load_cached_wham_timeline(run_dir)
    reference = load_opensim_reference(
        benchmark_dir / "reference" / f"opensim_fk_{trial}.npz",
        benchmark_dir / "reference" / f"opensim_fk_{trial}.json",
    )

    level_a_report, level_a_series = compare_pose_to_opensim_reference(
        pose,
        reference,
        run_config=run_config,
        timeline_report=timeline,
        convention_profile=convention_profile,
        evaluation_hz=evaluation_hz,
    )
    time_s = np.asarray(level_a_series["time_s"], dtype=float)
    joint_names = [str(name) for name in level_a_series["joint_names"].tolist()]
    wham_3d = np.asarray(level_a_series["prediction_root_centered_m"], dtype=float)
    gt_3d = np.asarray(level_a_series["reference_root_centered_m"], dtype=float)

    wham_2d, wham_2d_report = wham_2d_series_for_level_a_times(pose, run_config, timeline, level_a_report, time_s)
    video_signals = side_motion_signals_2d(wham_2d, time_s)
    wham_signals = side_motion_signals_3d(wham_3d, joint_names, time_s)
    gt_signals = side_motion_signals_3d(gt_3d, joint_names, time_s)

    side_map_rows = side_mapping_rows(video_signals, wham_signals, gt_signals)
    projection_report, projection_series = gt_projection_side_check(
        run_config,
        reference,
        wham_2d,
        time_s,
        out_mp4,
        pose=pose,
        level_a_report=level_a_report,
    )
    plot_report = write_side_phase_plot(out_png, time_s, video_signals, wham_signals, gt_signals)
    write_side_phase_csv(out_csv, side_map_rows)

    warnings: list[str] = []
    warnings.extend(str(w) for w in wham_2d_report.get("warnings", []))
    warnings.extend(str(w) for w in projection_report.get("warnings", []))
    if level_a_report.get("left_right_swap"):
        warnings.append(
            "The selected convention profile already applies a WHAM left/right relabel; "
            "side-swap rows are relative to that profile."
        )

    wham_agreement = compare_side_signal_sets(video_signals, wham_signals)
    gt_agreement = compare_side_signal_sets(video_signals, gt_signals)
    result = {
        "status": "warning" if warnings else "ok",
        "trial": trial,
        "benchmark_dir": str(benchmark_dir),
        "run_dir": str(run_dir),
        "evaluation_hz": float(evaluation_hz) if evaluation_hz is not None else None,
        "convention_profile": level_a_report.get("convention_profile"),
        "convention_left_right_swap": level_a_report.get("left_right_swap"),
        "time_offset_s": level_a_report.get("time_offset_s"),
        "frames": int(time_s.shape[0]),
        "time_start_s": float(time_s[0]) if time_s.size else None,
        "time_end_s": float(time_s[-1]) if time_s.size else None,
        "wham_2d": wham_2d_report,
        "start_side_estimates": {
            "wham_2d": lead_side_estimate(video_signals, time_s),
            "wham_3d": lead_side_estimate(wham_signals, time_s),
            "gt_opensim_fk": lead_side_estimate(gt_signals, time_s),
        },
        "side_agreement": {
            "wham_3d_vs_wham_2d": wham_agreement,
            "gt_vs_wham_2d": gt_agreement,
            "mapping_candidates": side_map_rows,
            "best_mapping": best_side_mapping(side_map_rows),
            "note": "Scores are signal-correlation side checks, not MPJPE. They are diagnostic only.",
        },
        "projection_check": projection_report,
        "outputs": {
            "json": str(out_json),
            "csv": str(out_csv),
            "plot": str(out_png),
            "projection_overlay": str(out_mp4) if out_mp4.exists() else None,
        },
        "warnings": warnings,
    }
    if projection_series:
        result["projection_check"]["series_summary"] = projection_series
    if plot_report.get("status") != "ok":
        result.setdefault("warnings", []).extend(str(w) for w in plot_report.get("warnings", []))
        result["status"] = "warning"
    write_json(out_json, result)
    return result


def wham_2d_series_for_level_a_times(
    pose: dict[str, Any],
    run_config: dict[str, Any],
    timeline: dict[str, Any] | None,
    level_a_report: dict[str, Any],
    time_s: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    pose2d = pose.get("pose2d") or {}
    xy = np.asarray(pose2d.get("xy"), dtype=float)
    conf = np.asarray(pose2d.get("confidence"), dtype=float)
    if xy.ndim != 3 or xy.shape[-1] != 2 or xy.shape[1] < 17:
        raise ValueError("WHAM pose artifact does not contain COCO17-style 2D keypoints.")
    if conf.ndim != 2 or conf.shape[:2] != xy.shape[:2]:
        conf = np.ones(xy.shape[:2], dtype=float)
    timebase = resolve_prediction_timebase(pose, cfg=run_config, timeline_report=timeline)
    pose_indices = np.asarray(timebase["pose_indices"], dtype=int)
    native_time = np.asarray(timebase["prediction_timestamps_s"], dtype=float)
    n = min(native_time.shape[0], pose_indices.shape[0])
    native_time = native_time[:n]
    pose_indices = pose_indices[:n]
    offset = float(level_a_report.get("time_offset_s") or 0.0)
    sample_time = np.asarray(time_s, dtype=float) + offset
    selected_xy = np.stack([xy[pose_indices, COCO17[name], :] for name in SIDE_JOINTS], axis=1)
    selected_conf = np.stack([conf[pose_indices, COCO17[name]] for name in SIDE_JOINTS], axis=1)
    sampled_xy = _resample_vector_timeseries(native_time, selected_xy, sample_time)
    sampled_conf = _resample_vector_timeseries(native_time, selected_conf[..., None], sample_time)[..., 0]
    sampled_xy[~np.isfinite(sampled_conf) | (sampled_conf < 0.2)] = np.nan
    return sampled_xy, {
        "status": "ok",
        "source": "pose3d_initial.pose2d",
        "joint_names": SIDE_JOINTS,
        "native_frames": int(xy.shape[0]),
        "sampled_frames": int(sampled_xy.shape[0]),
        "sample_time_offset_s": offset,
        "finite_ratio": float(np.isfinite(sampled_xy).all(axis=2).mean()) if sampled_xy.size else 0.0,
        "warnings": [],
    }


def side_motion_signals_2d(values: np.ndarray, time_s: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 3 or arr.shape[1] < len(SIDE_JOINTS) or arr.shape[-1] != 2:
        raise ValueError("2D side values must have shape [T, 6, 2].")
    left_hip, right_hip = arr[:, 0], arr[:, 1]
    hip_mid = 0.5 * (left_hip + right_hip)
    scale = np.linalg.norm(left_hip - right_hip, axis=1)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, np.nan)
    norm = (arr - hip_mid[:, None, :]) / scale[:, None, None]
    return _motion_signals(norm, SIDE_JOINTS, time_s, dimensions=2, source="wham_2d")


def side_motion_signals_3d(values: np.ndarray, joint_names: list[str], time_s: np.ndarray) -> dict[str, Any]:
    selected = []
    for name in SIDE_JOINTS:
        idx = find_joint(joint_names, (name,))
        if idx is None:
            raise ValueError(f"Missing {name} in 3D side phase input.")
        selected.append(np.asarray(values, dtype=float)[:, idx, :])
    arr = np.stack(selected, axis=1)
    return _motion_signals(arr, SIDE_JOINTS, time_s, dimensions=3, source="3d")


def compare_side_signal_sets(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ref_l = np.asarray(reference["left_signal"], dtype=float)
    ref_r = np.asarray(reference["right_signal"], dtype=float)
    cand_l = np.asarray(candidate["left_signal"], dtype=float)
    cand_r = np.asarray(candidate["right_signal"], dtype=float)
    same = _mean_finite([_corr(ref_l, cand_l), _corr(ref_r, cand_r)])
    swapped = _mean_finite([_corr(ref_l, cand_r), _corr(ref_r, cand_l)])
    margin = None if same is None or swapped is None else float(same - swapped)
    if margin is None or abs(margin) < 0.05:
        preferred = "uncertain"
    else:
        preferred = "same_labels" if margin > 0.0 else "opposite_labels"
    return {
        "same_label_correlation": same,
        "opposite_label_correlation": swapped,
        "same_minus_opposite": margin,
        "preferred": preferred,
    }


def side_mapping_rows(video: dict[str, Any], wham_3d: dict[str, Any], gt_3d: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for swap_wham in (False, True):
        for swap_gt in (False, True):
            wham_cmp = compare_side_signal_sets(video, _maybe_swap_signal_sides(wham_3d, swap_wham))
            gt_cmp = compare_side_signal_sets(video, _maybe_swap_signal_sides(gt_3d, swap_gt))
            components = [wham_cmp.get("same_minus_opposite"), gt_cmp.get("same_minus_opposite")]
            finite = [float(v) for v in components if v is not None and np.isfinite(v)]
            rows.append(
                {
                    "mapping": _mapping_name(swap_wham, swap_gt),
                    "swap_wham_3d": swap_wham,
                    "swap_gt": swap_gt,
                    "combined_margin": float(np.mean(finite)) if finite else None,
                    "wham_same_minus_opposite": wham_cmp.get("same_minus_opposite"),
                    "gt_same_minus_opposite": gt_cmp.get("same_minus_opposite"),
                    "wham_preferred": wham_cmp.get("preferred"),
                    "gt_preferred": gt_cmp.get("preferred"),
                }
            )
    rows.sort(key=lambda row: -float(row["combined_margin"]) if row.get("combined_margin") is not None else float("inf"))
    return rows


def best_side_mapping(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get("combined_margin") is not None]
    if not valid:
        return None
    return dict(max(valid, key=lambda row: float(row["combined_margin"])))


def lead_side_estimate(signals: dict[str, Any], time_s: np.ndarray) -> dict[str, Any]:
    time = np.asarray(time_s, dtype=float)
    left = np.asarray(signals["left_signal"], dtype=float)
    right = np.asarray(signals["right_signal"], dtype=float)
    n = min(len(time), len(left), len(right))
    if n < 3:
        return {"side": "uncertain", "reason": "fewer than three frames"}
    limit = max(3, int(np.ceil(n * 0.7)))
    left_idx = _first_peak_index(left[:limit])
    right_idx = _first_peak_index(right[:limit])
    if left_idx is None or right_idx is None:
        return {"side": "uncertain", "reason": "insufficient finite motion signal"}
    dt = float(time[left_idx] - time[right_idx])
    frame_step = float(np.nanmedian(np.diff(time))) if n > 1 else 0.0
    if abs(dt) <= max(1.5 * frame_step, 1e-6):
        side = "uncertain"
    else:
        side = "left" if dt < 0.0 else "right"
    return {
        "side": side,
        "left_peak_time_s": float(time[left_idx]),
        "right_peak_time_s": float(time[right_idx]),
        "left_minus_right_peak_time_s": dt,
        "method": "earlier strongest activity peak in first half of sequence",
    }


def gt_projection_side_check(
    run_config: dict[str, Any],
    reference: dict[str, Any],
    wham_2d: np.ndarray,
    time_s: np.ndarray,
    out_mp4: Path | None = None,
    pose: dict[str, Any] | None = None,
    level_a_report: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    warnings: list[str] = []
    camera_path = _camera_path(run_config)
    if camera_path is None or not camera_path.exists():
        return (
            {
                "status": "skipped",
                "reason": "OpenCap camera calibration is unavailable.",
                "warnings": ["Projection side check requires manifest calibration."],
            },
            {},
        )
    ref_names = [str(name) for name in reference["joint_names"]]
    ref_time = np.asarray(reference["time_s"], dtype=float)
    ref_values = np.asarray(reference["joints_m"], dtype=float)
    selected = []
    selected_names = []
    for name in PHASE_JOINTS:
        idx = find_joint(ref_names, (name,))
        if idx is not None:
            selected.append(ref_values[:, idx, :])
            selected_names.append(name)
    if len(selected) < 4:
        return (
            {
                "status": "skipped",
                "reason": "OpenSim FK reference lacks knees/ankles for projection side check.",
                "warnings": ["Projection side check needs LKNE/RKNE/LANK/RANK."],
            },
            {},
        )
    ref_sample = resample_timeseries(ref_time, np.stack(selected, axis=1), time_s)
    uv, valid = project_opensim_fk_to_pixels(ref_sample, camera_path)
    wham_phase_xy = _phase_xy_from_six_joint_xy(wham_2d)
    side_check = projection_nearest_side_check(wham_phase_xy, uv, valid)
    report = {
        "status": "ok",
        "camera_calibration": str(camera_path),
        "joint_names": selected_names,
        **side_check,
        "warnings": warnings,
    }
    if out_mp4 is not None and pose is not None:
        try:
            overlay = write_projection_overlay(out_mp4, run_config, pose, time_s, wham_phase_xy, uv, valid, level_a_report or {})
            report["overlay"] = overlay
        except Exception as exc:
            warnings.append(f"Projection overlay was not written: {exc}")
            report["status"] = "warning"
    return report, {"same_side_closer_ratio": side_check.get("same_side_closer_ratio")}


def projection_nearest_side_check(wham_xy: np.ndarray, gt_uv: np.ndarray, gt_valid: np.ndarray) -> dict[str, Any]:
    wham = np.asarray(wham_xy, dtype=float)
    gt = np.asarray(gt_uv, dtype=float)
    valid = np.asarray(gt_valid, dtype=bool)
    pairs = [(0, 1, "knee"), (2, 3, "ankle")]
    same_better = 0
    opposite_better = 0
    ties = 0
    total = 0
    per_joint: dict[str, dict[str, Any]] = {}
    labels = ["left_knee", "right_knee", "left_ankle", "right_ankle"]
    for left, right, group in pairs:
        group_same = 0
        group_opp = 0
        group_total = 0
        for frame_idx in range(min(wham.shape[0], gt.shape[0])):
            if not (valid[frame_idx, left] and valid[frame_idx, right]):
                continue
            pts = [wham[frame_idx, left], wham[frame_idx, right], gt[frame_idx, left], gt[frame_idx, right]]
            if not all(np.isfinite(p).all() for p in pts):
                continue
            same = np.linalg.norm(gt[frame_idx, left] - wham[frame_idx, left]) + np.linalg.norm(gt[frame_idx, right] - wham[frame_idx, right])
            opposite = np.linalg.norm(gt[frame_idx, left] - wham[frame_idx, right]) + np.linalg.norm(gt[frame_idx, right] - wham[frame_idx, left])
            total += 1
            group_total += 1
            if abs(float(same - opposite)) < 1e-6:
                ties += 1
            elif same < opposite:
                same_better += 1
                group_same += 1
            else:
                opposite_better += 1
                group_opp += 1
        per_joint[group] = {
            "same_side_closer": int(group_same),
            "opposite_side_closer": int(group_opp),
            "frames": int(group_total),
        }
    ratio = float(same_better / total) if total else None
    if ratio is None:
        preference = "unavailable"
    elif ratio >= 0.6:
        preference = "same_side"
    elif ratio <= 0.4:
        preference = "opposite_side"
    else:
        preference = "uncertain"
    return {
        "frames_compared": int(total),
        "same_side_closer": int(same_better),
        "opposite_side_closer": int(opposite_better),
        "ties": int(ties),
        "same_side_closer_ratio": ratio,
        "preference": preference,
        "per_group": per_joint,
        "labels": labels,
    }


def write_side_phase_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "mapping",
        "swap_wham_3d",
        "swap_gt",
        "combined_margin",
        "wham_same_minus_opposite",
        "gt_same_minus_opposite",
        "wham_preferred",
        "gt_preferred",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_side_phase_plot(
    path: Path,
    time_s: np.ndarray,
    video: dict[str, Any],
    wham_3d: dict[str, Any],
    gt_3d: dict[str, Any],
) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for ax, title, signals in [
            (axes[0], "WHAM 2D video-side labels", video),
            (axes[1], "WHAM 3D SMPL joints", wham_3d),
            (axes[2], "OpenSim FK GT joints", gt_3d),
        ]:
            ax.plot(time_s, signals["left_signal"], color="#cc3333", label="left")
            ax.plot(time_s, signals["right_signal"], color="#2f6fdd", label="right")
            ax.set_title(title)
            ax.set_ylabel("activity")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right")
        axes[-1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return {"status": "ok", "output": str(path)}
    except Exception as exc:
        return {"status": "warning", "warnings": [f"Side phase plot was not written: {exc}"]}


def write_projection_overlay(
    out_mp4: Path,
    run_config: dict[str, Any],
    pose: dict[str, Any],
    time_s: np.ndarray,
    wham_xy: np.ndarray,
    gt_uv: np.ndarray,
    gt_valid: np.ndarray,
    level_a_report: dict[str, Any],
) -> dict[str, Any]:
    import cv2

    trial = run_config.get("trial") or {}
    sync_video = Path(str(trial.get("video_sync") or run_config.get("raw_video") or ""))
    if not sync_video.exists():
        raise FileNotFoundError(f"Synced video is unavailable: {sync_video}")
    cap = cv2.VideoCapture(str(sync_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open synced video: {sync_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or pose.get("fps") or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), min(fps, 30.0), (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open projection overlay writer: {out_mp4}")
    labels = ["LKNE", "RKNE", "LANK", "RANK"]
    written = 0
    try:
        for idx, timestamp in enumerate(np.asarray(time_s, dtype=float).tolist()):
            frame_idx = int(round(timestamp * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            _draw_phase_points(frame, wham_xy[idx], np.ones(4, dtype=bool), labels, prefix="2D", radius=5)
            _draw_phase_points(frame, gt_uv[idx], gt_valid[idx], labels, prefix="GT", radius=8)
            header = f"Side phase audit | sync frame {frame_idx} | t={timestamp:.3f}s | profile={level_a_report.get('convention_profile')}"
            _draw_text(frame, header, (18, 34), (255, 255, 255), 0.58)
            _draw_text(frame, "WHAM 2D small dots, GT projected rings | left=red right=blue", (18, 62), (235, 235, 235), 0.52)
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        cap.release()
    return {"status": "ok" if written else "warning", "source_video": str(sync_video), "frames_written": int(written), "output": str(out_mp4)}


def _motion_signals(values: np.ndarray, names: list[str], time_s: np.ndarray, dimensions: int, source: str) -> dict[str, Any]:
    lookup = {name: idx for idx, name in enumerate(names)}
    left = _one_side_signal(values, lookup, "left", time_s, dimensions)
    right = _one_side_signal(values, lookup, "right", time_s, dimensions)
    return {
        "source": source,
        "left_signal": left,
        "right_signal": right,
        "finite_ratio": {
            "left": float(np.isfinite(left).mean()) if left.size else 0.0,
            "right": float(np.isfinite(right).mean()) if right.size else 0.0,
        },
    }


def _one_side_signal(values: np.ndarray, lookup: dict[str, int], side: str, time_s: np.ndarray, dimensions: int) -> np.ndarray:
    hip = values[:, lookup[f"{side}_hip"], :dimensions]
    knee = values[:, lookup[f"{side}_knee"], :dimensions]
    ankle = values[:, lookup[f"{side}_ankle"], :dimensions]
    knee_speed = _point_speed(knee, time_s)
    ankle_speed = _point_speed(ankle, time_s)
    angle = _joint_angle(hip, knee, ankle)
    angular_speed = _abs_derivative(angle, time_s)
    return _zscore_nan(knee_speed) + _zscore_nan(ankle_speed) + 0.5 * _zscore_nan(angular_speed)


def _point_speed(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    time = np.asarray(time_s, dtype=float)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    if arr.shape[0] < 2:
        return out
    dt = np.diff(time)
    delta = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    valid = np.isfinite(delta) & np.isfinite(dt) & (dt > 0)
    out[1:][valid] = delta[valid] / dt[valid]
    return out


def _abs_derivative(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    time = np.asarray(time_s, dtype=float)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    if arr.shape[0] < 2:
        return out
    dt = np.diff(time)
    delta = np.abs(np.diff(arr))
    valid = np.isfinite(delta) & np.isfinite(dt) & (dt > 0)
    out[1:][valid] = delta[valid] / dt[valid]
    return out


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denom = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    dot = np.sum(ba * bc, axis=1)
    cosang = np.divide(dot, denom, out=np.full_like(dot, np.nan, dtype=float), where=denom > 1e-12)
    return np.arccos(np.clip(cosang, -1.0, 1.0))


def _zscore_nan(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    out = np.zeros(arr.shape, dtype=float)
    if finite.size < 2:
        return out
    std = float(np.nanstd(finite))
    if std <= 1e-12:
        return out
    out[np.isfinite(arr)] = (arr[np.isfinite(arr)] - float(np.nanmean(finite))) / std
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 3:
        return None
    x = x[valid]
    y = y[valid]
    if float(np.nanstd(x)) <= 1e-12 or float(np.nanstd(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _resample_vector_timeseries(t_src: np.ndarray, values: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 3:
        raise ValueError("Expected values with shape [T, J, C].")
    src_time = np.asarray(t_src, dtype=float)
    dst_time = np.asarray(t_dst, dtype=float)
    out = np.full((len(dst_time), arr.shape[1], arr.shape[2]), np.nan, dtype=float)
    for joint_idx in range(arr.shape[1]):
        for coord_idx in range(arr.shape[2]):
            series = arr[:, joint_idx, coord_idx]
            valid = np.isfinite(series) & np.isfinite(src_time)
            if np.count_nonzero(valid) < 2:
                continue
            ts = src_time[valid]
            xs = series[valid]
            interp = np.interp(dst_time, ts, xs)
            interp[(dst_time < ts.min()) | (dst_time > ts.max())] = np.nan
            out[:, joint_idx, coord_idx] = interp
    return out


def _mean_finite(values: list[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def _maybe_swap_signal_sides(signals: dict[str, Any], swap: bool) -> dict[str, Any]:
    if not swap:
        return signals
    out = dict(signals)
    out["left_signal"] = signals["right_signal"]
    out["right_signal"] = signals["left_signal"]
    return out


def _mapping_name(swap_wham: bool, swap_gt: bool) -> str:
    if swap_wham and swap_gt:
        return "swap_wham_3d_and_gt"
    if swap_wham:
        return "swap_wham_3d_only"
    if swap_gt:
        return "swap_gt_only"
    return "no_swap"


def _first_peak_index(values: np.ndarray) -> int | None:
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).any():
        return None
    return int(np.nanargmax(np.where(np.isfinite(arr), arr, -np.inf)))


def _phase_xy_from_six_joint_xy(wham_2d: np.ndarray) -> np.ndarray:
    arr = np.asarray(wham_2d, dtype=float)
    return arr[:, [2, 3, 4, 5], :]


def _camera_path(run_config: dict[str, Any]) -> Path | None:
    calibration = ((run_config or {}).get("manifest_summary") or {}).get("calibration") or {}
    source = calibration.get("intrinsics_extrinsics")
    return Path(str(source)) if source else None


def _load_wham_summary_row(benchmark_dir: Path, trial: str) -> dict[str, Any]:
    summary = read_json(benchmark_dir / "level_a_summary.json")
    for row in summary.get("rows", []):
        if row.get("backend") == "wham" and row.get("trial") == trial and row.get("status") == "valid":
            return row
    raise ValueError(f"No valid WHAM row found for trial {trial!r} in {benchmark_dir / 'level_a_summary.json'}")


def _draw_phase_points(frame: np.ndarray, xy: np.ndarray, valid: np.ndarray, labels: list[str], prefix: str, radius: int) -> None:
    import cv2

    for idx, label in enumerate(labels):
        if idx >= xy.shape[0] or not bool(valid[idx]) or not np.isfinite(xy[idx]).all():
            continue
        x, y = np.round(xy[idx]).astype(int).tolist()
        color = (70, 70, 255) if label.startswith("L") else (255, 120, 60)
        if prefix == "GT":
            cv2.circle(frame, (x, y), radius, color, 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, (x, y), radius, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), max(2, radius - 2), color, -1, cv2.LINE_AA)
        _draw_text(frame, f"{prefix}_{label}", (x + 8, y - 6), color, 0.42)


def _draw_text(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int], scale: float) -> None:
    import cv2

    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
