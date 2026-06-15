from __future__ import annotations

import csv
import copy
from pathlib import Path
from typing import Any

import numpy as np

from monocap_v2.core.geometry import apply_similarity, find_joint, kabsch_align
from monocap_v2.core.level_a_benchmark import (
    compare_pose_to_opensim_reference,
    load_cached_wham_timeline,
    load_opensim_reference,
    load_pose_artifact,
    load_run_config,
)
from monocap_v2.core.logging_utils import read_json, write_json
from monocap_v2.core.mocap_eval import mpjpe


ALIGNMENT_MODES = ["none", "translation", "rigid", "similarity"]
MAPPING_MODES = ["no_swap", "swap_wham_3d_only", "swap_gt_only", "swap_wham_3d_and_gt"]
EDGES = [
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_hip", "right_hip"),
]


def run_wham_gt_fit_transform_audit(
    benchmark_dir: Path,
    trial: str,
    evaluation_hz: float | None = 60.0,
    convention_profile: str | None = None,
    time_offset_s: float | None = None,
    fit_direction: str = "wham_to_gt",
    out_dir: Path | None = None,
) -> dict[str, Any]:
    fit_direction = _normalize_fit_direction(fit_direction)
    benchmark_dir = Path(benchmark_dir)
    out_dir = Path(out_dir) if out_dir else benchmark_dir / "fit_transform"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"wham_gt_fit_transform_{trial}.json"
    out_csv = out_dir / f"wham_gt_fit_transform_{trial}.csv"
    out_png = out_dir / f"wham_gt_fit_transform_{trial}.png"
    out_mp4 = out_dir / f"wham_gt_fit_transform_{trial}.mp4"

    row = _load_wham_summary_row(benchmark_dir, trial)
    run_dir = Path(str(row["run_dir"]))
    pose = load_pose_artifact(run_dir)
    run_config = load_run_config(run_dir)
    run_config_for_compare, resolved_profile = _run_config_with_time_offset(run_config, convention_profile, time_offset_s)
    reference = load_opensim_reference(
        benchmark_dir / "reference" / f"opensim_fk_{trial}.npz",
        benchmark_dir / "reference" / f"opensim_fk_{trial}.json",
    )
    report, series = compare_pose_to_opensim_reference(
        pose,
        reference,
        run_config=run_config_for_compare,
        timeline_report=load_cached_wham_timeline(run_dir),
        convention_profile=resolved_profile,
        evaluation_hz=evaluation_hz,
    )
    time_s = np.asarray(series["time_s"], dtype=float)
    names = [str(name) for name in series["joint_names"].tolist()]
    wham = np.asarray(series["prediction_eval_m"], dtype=float)
    gt = np.asarray(series["reference_eval_m"], dtype=float)

    rows: list[dict[str, Any]] = []
    fit_series: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    transforms: dict[tuple[str, str], dict[str, Any]] = {}
    for mapping in MAPPING_MODES:
        pred_map, ref_map = _apply_mapping(wham, gt, names, mapping)
        fit_source, fit_target = _fit_direction_arrays(pred_map, ref_map, fit_direction)
        for mode in ALIGNMENT_MODES:
            fit = fit_sequence_transform(fit_source, fit_target, names, mode=mode)
            fitted_source = apply_fit(fit_source, fit["transform"])
            holdout = temporal_holdout_fit(fit_source, fit_target, names, mode=mode)
            row_data = {
                "mapping": mapping,
                "alignment": mode,
                "diagnostic_only": True,
                "mpjpe_all_mm": fit["mpjpe_mm"],
                "per_joint_mpjpe_mm": fit["per_joint_mpjpe_mm"],
                "first_half_fit_mpjpe_mm": holdout.get("first_half_fit_mpjpe_mm"),
                "second_half_eval_mpjpe_mm": holdout.get("second_half_eval_mpjpe_mm"),
                "determinant": fit["transform"]["determinant"],
                "scale": fit["transform"]["scale"],
                "translation_norm_m": fit["transform"]["translation_norm_m"],
                "frames": int(time_s.shape[0]),
                "joint_count": int(len(names)),
            }
            rows.append(row_data)
            fit_series[(mapping, mode)] = _display_pair(fitted_source, pred_map, ref_map, fit_direction)
            transforms[(mapping, mode)] = fit["transform"]
    rows.sort(key=lambda item: float(item["mpjpe_all_mm"]) if item.get("mpjpe_all_mm") is not None else float("inf"))
    best_rigid = _best_row(rows, "rigid")
    best_similarity = _best_row(rows, "similarity")
    best_translation = _best_row(rows, "translation")
    write_fit_csv(out_csv, rows)
    plot_report = write_fit_plot(out_png, rows)
    video_report = write_fit_video(
        out_mp4,
        time_s,
        names,
        gt,
        wham,
        fit_series,
        best_rigid,
        best_similarity,
        fit_direction=fit_direction,
        preview_fps=min(float(evaluation_hz or report.get("prediction_fps") or 30.0), 30.0),
    )

    result = {
        "status": "ok",
        "trial": trial,
        "benchmark_dir": str(benchmark_dir),
        "run_dir": str(run_dir),
        "evaluation_hz": float(evaluation_hz) if evaluation_hz is not None else None,
        "convention_profile": report.get("convention_profile"),
        "diagnostic_time_offset_s": float(time_offset_s) if time_offset_s is not None else None,
        "fit_direction": fit_direction,
        "frames": int(time_s.shape[0]),
        "joint_names": names,
        "input_level_a_metrics": {
            "primary_root_centered_mpjpe_mm": report.get("primary_root_centered_mpjpe_mm"),
            "root_centered_rigid_mpjpe_mm": report.get("root_centered_rigid_mpjpe_mm"),
            "pa_mpjpe_mm": report.get("pa_mpjpe_mm"),
            "global_no_align_mpjpe_mm": report.get("global_no_align_mpjpe_mm"),
        },
        "best": {
            "translation": best_translation,
            "rigid": best_rigid,
            "similarity": best_similarity,
        },
        "transforms": {
            f"{mapping}:{mode}": transforms[(mapping, mode)]
            for mapping in MAPPING_MODES
            for mode in ALIGNMENT_MODES
        },
        "rows": rows,
        "outputs": {"json": str(out_json), "csv": str(out_csv), "plot": str(out_png), "video": str(out_mp4)},
        "notes": [
            "This is a GT-fit diagnostic shortcut. It must not be used to select or rewrite production outputs.",
            "Transforms map the selected fit source into the selected fit target.",
            "Rows with left/right swaps are diagnostic candidates, not label fixes.",
        ],
        "plot": plot_report,
        "video": video_report,
    }
    write_json(out_json, result)
    return result


def fit_sequence_transform(pred: np.ndarray, ref: np.ndarray, joint_names: list[str], mode: str = "rigid") -> dict[str, Any]:
    mode = str(mode)
    if mode not in ALIGNMENT_MODES:
        raise ValueError(f"Unsupported fit mode {mode!r}.")
    pred_arr = np.asarray(pred, dtype=float)
    ref_arr = np.asarray(ref, dtype=float)
    mask = np.isfinite(pred_arr).all(axis=2) & np.isfinite(ref_arr).all(axis=2)
    rot = np.eye(3, dtype=float)
    trans = np.zeros(3, dtype=float)
    scale = 1.0
    if mode == "translation" and np.any(mask):
        trans = np.nanmean(ref_arr[mask] - pred_arr[mask], axis=0)
    elif mode in {"rigid", "similarity"} and np.count_nonzero(mask) >= 3:
        rot, trans, scale = kabsch_align(pred_arr[mask], ref_arr[mask], mode=mode)
    fitted = apply_similarity(pred_arr, rot, trans, scale)
    metrics = mpjpe(fitted, ref_arr, joint_names)
    transform = _transform_dict(rot, trans, scale)
    transform["mode"] = mode
    transform["fit_point_count"] = int(np.count_nonzero(mask))
    transform["matrix_4x4_source_to_target"] = _matrix_4x4(rot, trans, scale).tolist()
    transform["matrix_4x4_target_to_source"] = np.linalg.inv(_matrix_4x4(rot, trans, scale)).tolist()
    return {"mpjpe_mm": metrics["mpjpe_mm"], "per_joint_mpjpe_mm": metrics["per_joint_mpjpe_mm"], "transform": transform}


def apply_fit(values: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    return apply_similarity(
        np.asarray(values, dtype=float),
        np.asarray(transform["rotation_matrix"], dtype=float),
        np.asarray(transform["translation"], dtype=float),
        float(transform["scale"]),
    )


def temporal_holdout_fit(pred: np.ndarray, ref: np.ndarray, joint_names: list[str], mode: str) -> dict[str, Any]:
    pred_arr = np.asarray(pred, dtype=float)
    ref_arr = np.asarray(ref, dtype=float)
    if pred_arr.shape[0] < 4:
        return {"status": "skipped", "reason": "fewer than four frames"}
    split = pred_arr.shape[0] // 2
    fit = fit_sequence_transform(pred_arr[:split], ref_arr[:split], joint_names, mode=mode)
    fitted_all = apply_fit(pred_arr, fit["transform"])
    first = mpjpe(fitted_all[:split], ref_arr[:split], joint_names)
    second = mpjpe(fitted_all[split:], ref_arr[split:], joint_names)
    return {
        "status": "ok",
        "split_frame": int(split),
        "first_half_fit_mpjpe_mm": first["mpjpe_mm"],
        "second_half_eval_mpjpe_mm": second["mpjpe_mm"],
    }


def _normalize_fit_direction(value: str) -> str:
    normalized = str(value or "wham_to_gt").strip().lower().replace("-", "_")
    aliases = {
        "wham_to_gt": "wham_to_gt",
        "wham2gt": "wham_to_gt",
        "gt_to_wham": "gt_to_wham",
        "gt2wham": "gt_to_wham",
    }
    if normalized not in aliases:
        raise ValueError("fit_direction must be 'wham_to_gt' or 'gt_to_wham'.")
    return aliases[normalized]


def _fit_direction_arrays(pred_map: np.ndarray, ref_map: np.ndarray, fit_direction: str) -> tuple[np.ndarray, np.ndarray]:
    if fit_direction == "gt_to_wham":
        return np.asarray(ref_map, dtype=float), np.asarray(pred_map, dtype=float)
    return np.asarray(pred_map, dtype=float), np.asarray(ref_map, dtype=float)


def _display_pair(fitted_source: np.ndarray, pred_map: np.ndarray, ref_map: np.ndarray, fit_direction: str) -> dict[str, np.ndarray]:
    if fit_direction == "gt_to_wham":
        return {"gt": np.asarray(fitted_source, dtype=float), "wham": np.asarray(pred_map, dtype=float)}
    return {"gt": np.asarray(ref_map, dtype=float), "wham": np.asarray(fitted_source, dtype=float)}


def write_fit_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "mapping",
        "alignment",
        "diagnostic_only",
        "mpjpe_all_mm",
        "first_half_fit_mpjpe_mm",
        "second_half_eval_mpjpe_mm",
        "determinant",
        "scale",
        "translation_norm_m",
        "frames",
        "joint_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_fit_plot(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = rows[: min(12, len(rows))]
        labels = [f"{row['mapping']}\n{row['alignment']}" for row in top]
        values = [float(row["mpjpe_all_mm"]) for row in top]
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(np.arange(len(values)), values, color="#4f83cc")
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("MPJPE after GT-fit transform (mm)")
        ax.set_title("Diagnostic WHAM-to-GT fit candidates")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return {"status": "ok", "output": str(path)}
    except Exception as exc:
        return {"status": "warning", "warnings": [f"Fit plot was not written: {exc}"]}


def write_fit_video(
    path: Path,
    time_s: np.ndarray,
    names: list[str],
    gt: np.ndarray,
    wham: np.ndarray,
    fit_series: dict[tuple[str, str], dict[str, np.ndarray]],
    best_rigid: dict[str, Any] | None,
    best_similarity: dict[str, Any] | None,
    fit_direction: str = "wham_to_gt",
    preview_fps: float = 30.0,
) -> dict[str, Any]:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    path.parent.mkdir(parents=True, exist_ok=True)
    rigid_key = (best_rigid["mapping"], "rigid") if best_rigid else ("no_swap", "rigid")
    similarity_key = (best_similarity["mapping"], "similarity") if best_similarity else ("no_swap", "similarity")
    rigid_pair = fit_series.get(rigid_key, {"gt": gt, "wham": wham})
    similarity_pair = fit_series.get(similarity_key, {"gt": gt, "wham": wham})
    panels = [
        ("No GT fit", gt, wham, "no_swap:none"),
        ("Best rigid GT fit", rigid_pair["gt"], rigid_pair["wham"], f"{rigid_key[0]}:rigid"),
        ("Best similarity GT fit", similarity_pair["gt"], similarity_pair["wham"], f"{similarity_key[0]}:similarity"),
    ]
    bounds = _axis_bounds([gt, wham, rigid_pair["gt"], rigid_pair["wham"], similarity_pair["gt"], similarity_pair["wham"]])
    edges = _edge_indices(names)
    fig = plt.figure(figsize=(18, 6), dpi=100)
    canvas = FigureCanvasAgg(fig)
    axes = [fig.add_subplot(131, projection="3d"), fig.add_subplot(132, projection="3d"), fig.add_subplot(133, projection="3d")]
    writer = None
    video_size = None
    frames = 0
    try:
        for frame_idx, timestamp in enumerate(np.asarray(time_s, dtype=float).tolist()):
            for ax, (title, gt_values, wham_values, label) in zip(axes, panels):
                _draw_panel(ax, gt_values[frame_idx], wham_values[frame_idx], names, edges, bounds, f"{title}\n{label}", timestamp)
            fig.tight_layout(pad=0.8)
            canvas.draw()
            rgb = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
            if writer is None:
                video_size = (int(rgb.shape[1]), int(rgb.shape[0]))
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(preview_fps), video_size)
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer: {path}")
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            frames += 1
    finally:
        if writer is not None:
            writer.release()
        plt.close(fig)
    return {"status": "ok" if frames else "warning", "output": str(path), "frames": int(frames), "preview_fps": float(preview_fps)}


def _apply_mapping(pred: np.ndarray, ref: np.ndarray, names: list[str], mapping: str) -> tuple[np.ndarray, np.ndarray]:
    pred_out = np.asarray(pred, dtype=float).copy()
    ref_out = np.asarray(ref, dtype=float).copy()
    if mapping in {"swap_wham_3d_only", "swap_wham_3d_and_gt"}:
        pred_out = _swap_lr_values(pred_out, names)
    if mapping in {"swap_gt_only", "swap_wham_3d_and_gt"}:
        ref_out = _swap_lr_values(ref_out, names)
    return pred_out, ref_out


def _swap_lr_values(values: np.ndarray, names: list[str]) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    for left, right in [("left_hip", "right_hip"), ("left_knee", "right_knee"), ("left_ankle", "right_ankle")]:
        li = find_joint(names, (left,))
        ri = find_joint(names, (right,))
        if li is not None and ri is not None:
            out[:, [li, ri], :] = out[:, [ri, li], :]
    return out


def _best_row(rows: list[dict[str, Any]], alignment: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("alignment") == alignment and row.get("mpjpe_all_mm") is not None]
    return dict(min(candidates, key=lambda row: float(row["mpjpe_all_mm"]))) if candidates else None


def _transform_dict(rot: np.ndarray, trans: np.ndarray, scale: float) -> dict[str, Any]:
    linear = float(scale) * np.asarray(rot, dtype=float)
    return {
        "rotation_matrix": np.asarray(rot, dtype=float).tolist(),
        "translation": np.asarray(trans, dtype=float).tolist(),
        "scale": float(scale),
        "determinant": float(np.linalg.det(linear)),
        "rotation_determinant": float(np.linalg.det(rot)),
        "translation_norm_m": float(np.linalg.norm(trans)),
    }


def _matrix_4x4(rot: np.ndarray, trans: np.ndarray, scale: float) -> np.ndarray:
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = float(scale) * np.asarray(rot, dtype=float)
    mat[:3, 3] = np.asarray(trans, dtype=float)
    return mat


def _load_wham_summary_row(benchmark_dir: Path, trial: str) -> dict[str, Any]:
    summary = read_json(benchmark_dir / "level_a_summary.json")
    for row in summary.get("rows", []):
        if row.get("backend") == "wham" and row.get("trial") == trial and row.get("status") == "valid":
            return row
    raise ValueError(f"No valid WHAM row found for trial {trial!r} in {benchmark_dir / 'level_a_summary.json'}")


def _run_config_with_time_offset(
    run_config: dict[str, Any],
    convention_profile: str | None,
    time_offset_s: float | None,
) -> tuple[dict[str, Any], str | None]:
    if time_offset_s is None:
        return run_config, convention_profile
    from monocap_v2.core.wham_conventions import LEGACY_PROFILE, normalize_wham_convention_profile, wham_convention_profile_specs

    cfg = copy.deepcopy(run_config)
    base_profile = normalize_wham_convention_profile(convention_profile or LEGACY_PROFILE, cfg)
    base_spec = dict(wham_convention_profile_specs(cfg)[base_profile])
    profile_name = f"gt_fit_{base_profile}_off_{_offset_label(float(time_offset_s))}"
    base_spec["time_offset_s"] = float(time_offset_s)
    base_spec["diagnostic_only"] = True
    base_spec["label"] = f"{base_spec.get('label') or base_profile} + GT-fit offset {float(time_offset_s):+.3f}s"
    base_spec["source"] = "gt_fit_diagnostic"
    config = cfg.setdefault("config", {})
    level_a = config.setdefault("level_a", {})
    profiles = level_a.setdefault("wham_convention_profiles", {})
    profiles[profile_name] = base_spec
    return cfg, profile_name


def _offset_label(value: float) -> str:
    ms = int(round(value * 1000.0))
    return f"p{abs(ms)}ms" if ms >= 0 else f"m{abs(ms)}ms"


def _draw_panel(ax, gt_values: np.ndarray, wham_values: np.ndarray, names: list[str], edges: list[tuple[int, int]], bounds: dict[str, tuple[float, float]], title: str, time_s: float) -> None:
    ax.clear()
    _draw_skeleton(ax, gt_values, edges, "#111111", "GT")
    _draw_skeleton(ax, wham_values, edges, "#1f9d55", "WHAM")
    ax.set_xlim(*bounds["x"])
    ax.set_ylim(*bounds["z"])
    ax.set_zlim(*bounds["up"])
    ax.set_xlabel("X")
    ax.set_ylabel("Z/display")
    ax.set_zlabel("Up/display")
    ax.view_init(elev=18, azim=-70)
    ax.set_title(f"{title} | t={time_s:.3f}s")
    for idx, name in enumerate(names):
        if np.isfinite(_to_display_coords(gt_values[idx])).all():
            p = _to_display_coords(gt_values[idx])
            ax.text(p[0], p[1], p[2], _short_label(name), color="#111111", fontsize=7)
    ax.legend(loc="upper left", fontsize=8)


def _draw_skeleton(ax, values: np.ndarray, edges: list[tuple[int, int]], color: str, label: str) -> None:
    display = _to_display_coords(values)
    finite = np.isfinite(display).all(axis=1)
    if np.any(finite):
        ax.scatter(display[finite, 0], display[finite, 1], display[finite, 2], s=22, color=color, alpha=0.9, depthshade=False, label=label)
    for a, b in edges:
        if finite[a] and finite[b]:
            ax.plot([display[a, 0], display[b, 0]], [display[a, 1], display[b, 1]], [display[a, 2], display[b, 2]], color=color, linewidth=2.0, alpha=0.85)


def _to_display_coords(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.empty_like(arr)
    out[..., 0] = arr[..., 0]
    out[..., 1] = arr[..., 2]
    out[..., 2] = arr[..., 1]
    return out


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
    out = []
    for a, b in EDGES:
        ai = find_joint(names, (a,))
        bi = find_joint(names, (b,))
        if ai is not None and bi is not None:
            out.append((ai, bi))
    return out


def _short_label(name: str) -> str:
    return {
        "left_hip": "LHIP",
        "right_hip": "RHIP",
        "left_knee": "LKNE",
        "right_knee": "RKNE",
        "left_ankle": "LANK",
        "right_ankle": "RANK",
    }.get(name, name.upper())
