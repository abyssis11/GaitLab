from __future__ import annotations

import copy
import csv
from itertools import permutations, product
from pathlib import Path
from typing import Any

import numpy as np

from monocap_v2.core.level_a_benchmark import (
    BACKEND_AXIS_MAP,
    compare_pose_to_opensim_reference,
    load_opensim_reference,
    load_pose_artifact,
    load_run_config,
)
from monocap_v2.core.logging_utils import read_json, write_json
from monocap_v2.core.video_io import probe_video_frames
from monocap_v2.core.wham_timeline import build_wham_timeline_report


DEFAULT_TIME_OFFSETS = [round(float(v), 3) for v in np.arange(-0.45, 0.4501, 0.025)]


def run_level_a_audit(
    benchmark_dir: Path,
    trials: list[str] | None = None,
    backends: list[str] | None = None,
    time_offsets: list[float] | None = None,
) -> dict[str, Any]:
    benchmark_dir = Path(benchmark_dir)
    summary = read_json(benchmark_dir / "level_a_summary.json")
    rows = [row for row in summary.get("rows", []) if row.get("status") == "valid"]
    if trials:
        trial_set = set(trials)
        rows = [row for row in rows if row.get("trial") in trial_set]
    if backends:
        backend_set = set(backends)
        rows = [row for row in rows if row.get("backend") in backend_set]

    audit_dir = benchmark_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    offsets = time_offsets or DEFAULT_TIME_OFFSETS
    axes = signed_axis_expressions()

    for row in rows:
        backend = str(row["backend"])
        trial = str(row["trial"])
        run_dir = Path(str(row["run_dir"]))
        case: dict[str, Any] = {"backend": backend, "trial": trial, "run_dir": str(run_dir)}
        try:
            pose = load_pose_artifact(run_dir)
            run_config = load_run_config(run_dir)
            reference = load_opensim_reference(
                benchmark_dir / "reference" / f"opensim_fk_{trial}.npz",
                benchmark_dir / "reference" / f"opensim_fk_{trial}.json",
            )
            timeline = build_wham_timeline_report(pose, run_config) if backend == "wham" else None
            frame_report = audit_frame_counts(run_dir, pose, run_config)
            default_report, _ = compare_pose_to_opensim_reference(pose, reference, run_config=run_config, timeline_report=timeline)
            axis_report = run_axis_sweep(pose, reference, run_config, backend, axes, timeline_report=timeline)
            timing_report = run_timing_sweep(pose, reference, run_config, backend, offsets, timeline_report=timeline)
            frame_row = {"backend": backend, "trial": trial, **frame_report}
            frame_rows.append(frame_row)
            for item in axis_report["top"]:
                sweep_rows.append({"backend": backend, "trial": trial, "sweep": "axis", **item})
            for item in timing_report["top"]:
                sweep_rows.append({"backend": backend, "trial": trial, "sweep": "timing", **item})
            case.update(
                {
                    "status": "ok",
                    "frames": frame_report,
                    "default_metrics": _metric_subset(default_report),
                    "best_axis": axis_report["best"],
                    "best_timing": timing_report["best"],
                    "diagnostic_notes": _diagnostic_notes(default_report, axis_report, timing_report),
                }
            )
        except Exception as exc:
            case.update({"status": "failed", "error": str(exc)})
        cases.append(case)

    aggregate = summarize_audit_cases(cases)
    report = {
        "status": aggregate["status"],
        "benchmark_dir": str(benchmark_dir),
        "audit_dir": str(audit_dir),
        "axis_policy": dict(BACKEND_AXIS_MAP),
        "time_offsets_s": offsets,
        "aggregate": aggregate,
        "cases": cases,
        "outputs": {
            "json": str(audit_dir / "level_a_audit.json"),
            "summary_md": str(audit_dir / "level_a_audit.md"),
            "frame_counts_csv": str(audit_dir / "frame_counts.csv"),
            "diagnostic_sweeps_csv": str(audit_dir / "diagnostic_sweeps.csv"),
        },
    }
    write_audit_outputs(audit_dir, report, frame_rows, sweep_rows)
    return report


def audit_frame_counts(run_dir: Path, pose: dict[str, Any], run_config: dict[str, Any]) -> dict[str, Any]:
    video_info_path = run_dir / "video" / "video_info.json"
    video_info = read_json(video_info_path) if video_info_path.exists() else {}
    raw_video = Path(str(video_info.get("raw_video") or run_config.get("raw_video") or ""))
    probe = {}
    if raw_video.exists():
        try:
            probe = probe_video_frames(raw_video)
        except Exception as exc:
            probe = {"status": "failed", "error": str(exc), "source": str(raw_video)}
    artifact_frames = int(np.asarray(pose.get("joints_3d")).shape[0]) if pose.get("joints_3d") is not None else None
    backend_meta = pose.get("backend_meta") or {}
    return {
        "raw_video": str(raw_video) if raw_video else None,
        "video_info_metadata_frame_count": video_info.get("metadata_frame_count"),
        "video_info_frame_count": video_info.get("frame_count"),
        "video_info_sequential_decoded_frame_count": video_info.get("sequential_decoded_frame_count", video_info.get("decoded_frame_count")),
        "video_info_usable_frame_count": video_info.get("usable_frame_count", video_info.get("frame_count")),
        "probe_metadata_frame_count": probe.get("metadata_frame_count"),
        "probe_sequential_decoded_frame_count": probe.get("sequential_decoded_frame_count"),
        "probe_random_access_recovered_count": probe.get("random_access_recovered_count"),
        "probe_usable_frame_count": probe.get("usable_frame_count"),
        "probe_missing_frame_indices": probe.get("missing_frame_indices"),
        "artifact_frame_count": artifact_frames,
        "backend_frame_indices_count": _len_or_none(backend_meta.get("frame_indices")),
        "backend_time_s_count": _len_or_none(backend_meta.get("time_s")),
    }


def run_axis_sweep(
    pose: dict[str, Any],
    reference: dict[str, Any],
    run_config: dict[str, Any],
    backend: str,
    axis_expressions: list[str] | None = None,
    timeline_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = []
    for swapped in (False, True):
        candidate_pose = swap_left_right_pose(pose) if swapped else pose
        for expr in axis_expressions or signed_axis_expressions():
            try:
                report, _ = compare_pose_to_opensim_reference(
                    candidate_pose,
                    reference,
                    run_config=run_config,
                    axis_map={backend: expr},
                    timeline_report=timeline_report,
                )
            except Exception as exc:
                candidates.append({"status": "failed", "axis": expr, "left_right_swapped": swapped, "error": str(exc)})
                continue
            candidates.append({"status": "ok", "axis": expr, "left_right_swapped": swapped, **_metric_subset(report)})
    valid = sorted([item for item in candidates if item["status"] == "ok"], key=lambda item: float(item["primary_root_centered_mpjpe_mm"]))
    return {"best": valid[0] if valid else None, "top": valid[:8], "candidate_count": len(candidates)}


def run_timing_sweep(
    pose: dict[str, Any],
    reference: dict[str, Any],
    run_config: dict[str, Any],
    backend: str,
    time_offsets: list[float] | None = None,
    timeline_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = []
    for swapped in (False, True):
        candidate_pose = swap_left_right_pose(pose) if swapped else pose
        for offset in time_offsets or DEFAULT_TIME_OFFSETS:
            shifted = dict(reference)
            shifted["time_s"] = np.asarray(reference["time_s"], dtype=float) + float(offset)
            try:
                report, _ = compare_pose_to_opensim_reference(
                    candidate_pose,
                    shifted,
                    run_config=run_config,
                    axis_map={backend: BACKEND_AXIS_MAP.get(backend, "identity")},
                    timeline_report=timeline_report,
                )
            except Exception as exc:
                candidates.append({"status": "failed", "time_offset_s": float(offset), "left_right_swapped": swapped, "error": str(exc)})
                continue
            candidates.append({"status": "ok", "time_offset_s": float(offset), "left_right_swapped": swapped, **_metric_subset(report)})
    valid = sorted([item for item in candidates if item["status"] == "ok"], key=lambda item: float(item["primary_root_centered_mpjpe_mm"]))
    return {"best": valid[0] if valid else None, "top": valid[:8], "candidate_count": len(candidates)}


def signed_axis_expressions() -> list[str]:
    axes = ["x", "y", "z"]
    out = []
    for perm in permutations(axes):
        for signs in product((1, -1), repeat=3):
            out.append(",".join(("-" if sign < 0 else "") + axis for sign, axis in zip(signs, perm)))
    return out


def swap_left_right_pose(pose: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(pose)
    out["joint_names"] = [_swap_name(str(name)) for name in out.get("joint_names", [])]
    pose2d = out.get("pose2d")
    if isinstance(pose2d, dict) and pose2d.get("names"):
        pose2d["names"] = [_swap_name(str(name)) for name in pose2d.get("names", [])]
    return out


def summarize_audit_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [case for case in cases if case.get("status") != "ok"]
    frame_mismatches = []
    axis_suspects = []
    lr_suspects = []
    timing_suspects = []
    for case in cases:
        if case.get("status") != "ok":
            continue
        frames = case.get("frames") or {}
        backend = str(case.get("backend") or "")
        if (
            backend != "wham"
            and frames.get("artifact_frame_count") != frames.get("probe_usable_frame_count")
        ):
            frame_mismatches.append({"backend": case.get("backend"), "trial": case.get("trial")})
        best_axis = case.get("best_axis") or {}
        default = case.get("default_metrics") or {}
        if best_axis and default.get("primary_root_centered_mpjpe_mm") is not None:
            improvement = float(default["primary_root_centered_mpjpe_mm"]) - float(best_axis["primary_root_centered_mpjpe_mm"])
            if improvement > 50.0:
                axis_suspects.append({"backend": case.get("backend"), "trial": case.get("trial"), "improvement_mm": improvement})
            if best_axis.get("left_right_swapped"):
                lr_suspects.append({"backend": case.get("backend"), "trial": case.get("trial")})
        best_timing = case.get("best_timing") or {}
        if best_timing and abs(float(best_timing.get("time_offset_s") or 0.0)) >= 0.05:
            timing_suspects.append({"backend": case.get("backend"), "trial": case.get("trial"), "time_offset_s": best_timing.get("time_offset_s")})
    return {
        "status": "failed" if failed else "warning" if (frame_mismatches or axis_suspects or lr_suspects or timing_suspects) else "ok",
        "case_count": len(cases),
        "failed_count": len(failed),
        "frame_mismatches": frame_mismatches,
        "axis_suspects": axis_suspects,
        "left_right_suspects": lr_suspects,
        "timing_suspects": timing_suspects,
    }


def write_audit_outputs(audit_dir: Path, report: dict[str, Any], frame_rows: list[dict[str, Any]], sweep_rows: list[dict[str, Any]]) -> None:
    write_json(audit_dir / "level_a_audit.json", report)
    _write_csv(audit_dir / "frame_counts.csv", frame_rows)
    _write_csv(audit_dir / "diagnostic_sweeps.csv", sweep_rows)
    _write_audit_md(audit_dir / "level_a_audit.md", report)


def _metric_subset(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "primary_root_centered_mpjpe_mm": report.get("primary_root_centered_mpjpe_mm"),
        "root_centered_rigid_mpjpe_mm": report.get("root_centered_rigid_mpjpe_mm"),
        "pa_mpjpe_mm": report.get("pa_mpjpe_mm"),
        "normal_minus_rigid_gap_mm": report.get("normal_minus_rigid_gap_mm"),
        "overlap_frames": report.get("overlap_frames"),
        "axis_mode": report.get("axis_mode"),
    }


def _diagnostic_notes(default_report: dict[str, Any], axis_report: dict[str, Any], timing_report: dict[str, Any]) -> list[str]:
    notes = []
    best_axis = axis_report.get("best")
    if best_axis and default_report.get("primary_root_centered_mpjpe_mm") is not None:
        improvement = float(default_report["primary_root_centered_mpjpe_mm"]) - float(best_axis["primary_root_centered_mpjpe_mm"])
        if improvement > 50.0:
            notes.append(f"Best diagnostic axis improves primary MPJPE by {improvement:.1f} mm; do not apply automatically.")
        if best_axis.get("left_right_swapped"):
            notes.append("Best diagnostic axis candidate uses left/right label swap; needs visual label confirmation.")
    best_timing = timing_report.get("best")
    if best_timing and abs(float(best_timing.get("time_offset_s") or 0.0)) >= 0.05:
        notes.append(f"Best diagnostic timing offset is {float(best_timing['time_offset_s']):.3f} s; do not tune from GT.")
    return notes


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _write_audit_md(path: Path, report: dict[str, Any]) -> None:
    aggregate = report.get("aggregate") or {}
    with path.open("w", encoding="utf-8") as f:
        f.write("# Level A Pipeline/Backend Audit\n\n")
        f.write(f"- Status: `{report.get('status')}`\n")
        f.write("- OpenSim FK reference is treated as trusted in this audit.\n")
        f.write(f"- Cases: `{aggregate.get('case_count')}`\n")
        f.write(f"- Frame mismatches: `{len(aggregate.get('frame_mismatches') or [])}`\n")
        f.write(f"- Axis suspects: `{len(aggregate.get('axis_suspects') or [])}`\n")
        f.write(f"- Left/right suspects: `{len(aggregate.get('left_right_suspects') or [])}`\n")
        f.write(f"- Timing suspects: `{len(aggregate.get('timing_suspects') or [])}`\n\n")
        f.write("## Cases\n\n")
        f.write("| Backend | Trial | Artifact Frames | Usable Video Frames | Default MPJPE | Best Axis MPJPE | Best Axis | LR Swap | Best Time Offset |\n")
        f.write("|---|---|---:|---:|---:|---:|---|---:|---:|\n")
        for case in report.get("cases") or []:
            frames = case.get("frames") or {}
            default = case.get("default_metrics") or {}
            axis = case.get("best_axis") or {}
            timing = case.get("best_timing") or {}
            f.write(
                f"| {case.get('backend')} | {case.get('trial')} | {frames.get('artifact_frame_count')} | "
                f"{frames.get('probe_usable_frame_count')} | {_fmt(default.get('primary_root_centered_mpjpe_mm'))} | "
                f"{_fmt(axis.get('primary_root_centered_mpjpe_mm'))} | `{axis.get('axis')}` | "
                f"{axis.get('left_right_swapped')} | {_fmt(timing.get('time_offset_s'))} |\n"
            )


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


def _len_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(len(value))
    except Exception:
        return None


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return str(value)
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)
