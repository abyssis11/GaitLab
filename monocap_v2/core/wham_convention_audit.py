from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

from monocap_v2.core.level_a_benchmark import compare_pose_to_opensim_reference, load_opensim_reference, load_pose_artifact, load_run_config
from monocap_v2.core.logging_utils import read_json, write_json
from monocap_v2.core.wham_conventions import (
    LEGACY_PROFILE,
    OPENCAP_CAMERA_FULL_LR_PROFILE,
    OPENCAP_CAMERA_FULL_PROFILE,
    OPENCAP_CAMERA_LR_PROFILE,
    OPENCAP_CAMERA_PROFILE,
    convention_profile_label,
    wham_convention_profile_names,
)
from monocap_v2.core.wham_timeline import build_wham_timeline_report


AUDIT_PROFILES = [
    LEGACY_PROFILE,
    OPENCAP_CAMERA_PROFILE,
    OPENCAP_CAMERA_LR_PROFILE,
    OPENCAP_CAMERA_FULL_PROFILE,
    OPENCAP_CAMERA_FULL_LR_PROFILE,
    "opencap_yaw180",
    "opencap_yaw180_lr",
    "opencap_yaw180_lr_offset120ms",
    "opencap_full_yaw180",
    "opencap_full_yaw180_lr",
    "opencap_full_yaw180_lr_offset120ms",
]


def run_wham_convention_audit(
    benchmark_dir: Path,
    trials: list[str] | None = None,
    profiles: list[str] | None = None,
    out_dir: Path | None = None,
    evaluation_hz: float | None = None,
) -> dict[str, Any]:
    benchmark_dir = Path(benchmark_dir)
    out_dir = Path(out_dir) if out_dir else benchmark_dir / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = read_json(benchmark_dir / "level_a_summary.json")
    trial_filter = set(trials or [])
    profiles = profiles or wham_convention_profile_names()
    rows = [
        row
        for row in summary.get("rows", [])
        if row.get("status") == "valid"
        and row.get("backend") == "wham"
        and (not trial_filter or row.get("trial") in trial_filter)
    ]

    audit_rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for row in rows:
        trial = str(row["trial"])
        run_dir = Path(str(row["run_dir"]))
        case: dict[str, Any] = {
            "trial": trial,
            "run_dir": str(run_dir),
            "status": "ok",
            "rows": [],
            "verification_artifacts": _verification_artifacts(run_dir),
        }
        try:
            pose = load_pose_artifact(run_dir)
            run_config = load_run_config(run_dir)
            reference = load_opensim_reference(
                benchmark_dir / "reference" / f"opensim_fk_{trial}.npz",
                benchmark_dir / "reference" / f"opensim_fk_{trial}.json",
            )
            timeline = _cached_or_build_timeline(run_dir, pose, run_config)
            for profile in profiles:
                audit_row = _score_profile(profile, pose, reference, run_config, timeline, trial, run_dir, evaluation_hz=evaluation_hz)
                audit_rows.append(audit_row)
                case["rows"].append(audit_row)
        except Exception as exc:
            case["status"] = "failed"
            case["error"] = str(exc)
            audit_rows.append({"trial": trial, "run_dir": str(run_dir), "status": "failed", "error": str(exc)})
        cases.append(case)

    aggregate = _aggregate(audit_rows)
    report = {
        "status": aggregate["status"],
        "benchmark_dir": str(benchmark_dir),
        "audit_dir": str(out_dir),
        "profiles": profiles,
        "evaluation_hz": float(evaluation_hz) if evaluation_hz is not None else None,
        "default_profile": LEGACY_PROFILE,
        "notes": [
            "This audit is diagnostic. It does not rewrite WHAM artifacts and does not change benchmark defaults.",
            "OpenCap camera profiles use row-vector math: apply_axis_expr(points, 'x,-y,z') @ camera_rotation.",
            "Full-extrinsic profiles use row-vector math: (apply_axis_expr(points, 'x,-y,z') - camera_translation_m) @ camera_rotation.",
            "The L/R profile is diagnostic-only until WHAM and GT labeled video overlays confirm anatomical side labels.",
        ],
        "aggregate": aggregate,
        "cases": cases,
        "outputs": {
            "json": str(out_dir / "wham_convention_audit.json"),
            "summary_md": str(out_dir / "wham_convention_audit.md"),
            "csv": str(out_dir / "wham_convention_audit.csv"),
        },
    }
    _write_outputs(out_dir, report, audit_rows)
    return report


def _score_profile(
    profile: str,
    pose: dict[str, Any],
    reference: dict[str, Any],
    run_config: dict[str, Any],
    timeline: dict[str, Any] | None,
    trial: str,
    run_dir: Path,
    evaluation_hz: float | None = None,
) -> dict[str, Any]:
    try:
        report, _series = compare_pose_to_opensim_reference(
            pose,
            reference,
            run_config=run_config,
            convention_profile=profile,
            timeline_report=timeline,
            evaluation_hz=evaluation_hz,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "trial": trial,
            "run_dir": str(run_dir),
            "convention_profile": profile,
            "profile_label": _profile_label(profile),
            "error": str(exc),
        }
    return {
        "status": "ok",
        "trial": trial,
        "run_dir": str(run_dir),
        "profile_label": _profile_label(profile),
        "convention_profile": report.get("convention_profile"),
        "convention_source": report.get("convention_source"),
        "pre_axis": report.get("pre_axis"),
        "axis_mode": report.get("axis_mode"),
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
        "resampling": report.get("resampling"),
        "evaluation_hz": report.get("evaluation_hz"),
        "primary_root_centered_mpjpe_mm": report.get("primary_root_centered_mpjpe_mm"),
        "root_centered_rigid_mpjpe_mm": report.get("root_centered_rigid_mpjpe_mm"),
        "root_centered_n_mpjpe_mm": report.get("root_centered_n_mpjpe_mm"),
        "root_centered_similarity_mpjpe_mm": report.get("root_centered_similarity_mpjpe_mm"),
        "pa_mpjpe_mm": report.get("pa_mpjpe_mm"),
        "global_no_align_mpjpe_mm": report.get("global_no_align_mpjpe_mm"),
        "global_sequence_similarity_mpjpe_mm": report.get("global_sequence_similarity_mpjpe_mm"),
        "normal_minus_rigid_gap_mm": report.get("normal_minus_rigid_gap_mm"),
        "overlap_frames": report.get("overlap_frames"),
        "warnings": "; ".join(str(w) for w in (report.get("warnings") or [])),
    }


def _cached_or_build_timeline(run_dir: Path, pose: dict[str, Any], run_config: dict[str, Any]) -> dict[str, Any] | None:
    for candidate in _run_dir_candidates(run_dir):
        for qc_path in [candidate / "reports" / "wham_timeline_qc.json", candidate / "pose3d_initial" / "pose3d_initial_qc.json"]:
            if not qc_path.exists():
                continue
            qc = read_json(qc_path)
            timeline = qc if "raw_sync_alignment" in qc else qc.get("wham_timeline") or qc.get("wham_timeline_report")
            if isinstance(timeline, dict) and isinstance(timeline.get("raw_sync_alignment"), dict):
                return timeline
    return build_wham_timeline_report(pose, run_config)


def _verification_artifacts(run_dir: Path) -> dict[str, Any]:
    wham = _first_artifact(run_dir, "wham_labeled_overlay.mp4")
    gt = _first_artifact(run_dir, "gt_labeled_overlay.mp4")
    return {
        "wham_labeled_overlay": str(wham) if wham.exists() else None,
        "gt_labeled_overlay": str(gt) if gt.exists() else None,
    }


def _first_artifact(run_dir: Path, filename: str) -> Path:
    candidates = [candidate / "reports" / filename for candidate in _run_dir_candidates(run_dir)]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _run_dir_candidates(run_dir: Path) -> list[Path]:
    candidates = [run_dir]
    name = run_dir.name
    if "__" in name:
        candidates.append(run_dir.with_name(name.split("__", 1)[0]))
    return candidates


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok" and row.get("primary_root_centered_mpjpe_mm") is not None]
    by_profile: dict[str, Any] = {}
    for profile in wham_convention_profile_names():
        profile_rows = [row for row in valid if row.get("convention_profile") == profile]
        values = [float(row["primary_root_centered_mpjpe_mm"]) for row in profile_rows]
        by_profile[profile] = {
            "profile_label": _profile_label(profile),
            "valid_trial_count": len(profile_rows),
            "median_primary_root_centered_mpjpe_mm": float(statistics.median(values)) if values else None,
            "diagnostic_only": any(bool(row.get("diagnostic_only")) for row in profile_rows) if profile_rows else profile in {OPENCAP_CAMERA_LR_PROFILE, OPENCAP_CAMERA_FULL_PROFILE, OPENCAP_CAMERA_FULL_LR_PROFILE},
        }
    failed = [row for row in rows if row.get("status") != "ok"]
    ranking = [
        {"convention_profile": profile, **stats}
        for profile, stats in by_profile.items()
        if stats.get("median_primary_root_centered_mpjpe_mm") is not None
    ]
    ranking.sort(key=lambda item: float(item["median_primary_root_centered_mpjpe_mm"]))
    return {
        "status": "failed" if not valid else "warning" if failed else "ok",
        "row_count": len(rows),
        "valid_row_count": len(valid),
        "failed_row_count": len(failed),
        "by_profile": by_profile,
        "diagnostic_ranking": ranking,
        "visual_candidate": _visual_candidate_summary(valid),
    }


def _write_outputs(out_dir: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    write_json(out_dir / "wham_convention_audit.json", report)
    _write_csv(out_dir / "wham_convention_audit.csv", rows)
    _write_md(out_dir / "wham_convention_audit.md", report, rows)


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


def _write_md(path: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    aggregate = report.get("aggregate") or {}
    with path.open("w", encoding="utf-8") as f:
        f.write("# WHAM Convention Audit\n\n")
        f.write(f"- Status: `{report.get('status')}`\n")
        f.write(f"- Default profile remains: `{report.get('default_profile')}`\n")
        f.write(f"- Evaluation Hz: `{report.get('evaluation_hz') or 'native prediction timestamps'}`\n")
        f.write("- Scope: evaluation-only; WHAM artifacts, SMPL vertices, markers, and TRC are unchanged.\n")
        f.write("- L/R profile is diagnostic-only until labeled video overlays confirm side labels.\n\n")
        visual = (aggregate.get("visual_candidate") or {})
        if visual:
            f.write("## Visual Candidate\n\n")
            f.write(
                f"- Profile: `{visual.get('convention_profile')}`; median primary MPJPE: `{_fmt(visual.get('median_primary_root_centered_mpjpe_mm'))} mm`; "
                f"median global raw: `{_fmt(visual.get('median_global_no_align_mpjpe_mm'))} mm`; time offset: `{_fmt(visual.get('time_offset_s'))} s`.\n\n"
            )
        f.write("## Diagnostic Ranking\n\n")
        f.write("| Profile | Diagnostic Only | Valid Trials | Median Primary MPJPE (mm) |\n")
        f.write("|---|---:|---:|---:|\n")
        for item in aggregate.get("diagnostic_ranking") or []:
            f.write(
                f"| `{item.get('convention_profile')}` | {item.get('diagnostic_only')} | "
                f"{item.get('valid_trial_count')} | {_fmt(item.get('median_primary_root_centered_mpjpe_mm'))} |\n"
            )
        f.write("\n## Per Trial\n\n")
        f.write("| Trial | Profile | Translation | LR Swap | Offset (s) | Post det | Linear det | Primary | Rigid | PA | Global Raw | Gap | Warnings/Error |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row.get('trial')} | `{row.get('convention_profile')}` | {row.get('uses_camera_translation')} | {row.get('left_right_swap')} | "
                f"{_fmt(row.get('time_offset_s'))} | {_fmt(row.get('post_transform_determinant'))} | {_fmt(row.get('linear_transform_determinant'))} | "
                f"{_fmt(row.get('primary_root_centered_mpjpe_mm'))} | {_fmt(row.get('root_centered_rigid_mpjpe_mm'))} | "
                f"{_fmt(row.get('pa_mpjpe_mm'))} | {_fmt(row.get('global_no_align_mpjpe_mm'))} | {_fmt(row.get('normal_minus_rigid_gap_mm'))} | "
                f"{row.get('warnings') or row.get('error') or ''} |\n"
            )
        f.write("\n## Verification Artifacts\n\n")
        for case in report.get("cases") or []:
            artifacts = case.get("verification_artifacts") or {}
            f.write(f"- `{case.get('trial')}` WHAM overlay: `{artifacts.get('wham_labeled_overlay')}`\n")
            f.write(f"- `{case.get('trial')}` GT overlay: `{artifacts.get('gt_labeled_overlay')}`\n")


def _profile_label(profile: str) -> str:
    try:
        return convention_profile_label(profile)
    except Exception:
        return str(profile)


def _visual_candidate_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    profile = "opencap_full_yaw180_lr_offset120ms"
    selected = [row for row in rows if row.get("convention_profile") == profile]
    if not selected:
        return None
    primary = [float(row["primary_root_centered_mpjpe_mm"]) for row in selected if row.get("primary_root_centered_mpjpe_mm") is not None]
    global_raw = [float(row["global_no_align_mpjpe_mm"]) for row in selected if row.get("global_no_align_mpjpe_mm") is not None]
    return {
        "convention_profile": profile,
        "valid_trial_count": len(selected),
        "median_primary_root_centered_mpjpe_mm": float(statistics.median(primary)) if primary else None,
        "median_global_no_align_mpjpe_mm": float(statistics.median(global_raw)) if global_raw else None,
        "time_offset_s": selected[0].get("time_offset_s"),
        "diagnostic_only": selected[0].get("diagnostic_only"),
    }


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
