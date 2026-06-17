#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monocap_v2.core.contact_utils import estimate_contacts_from_joints
from monocap_v2.core.foot_locking import apply_contact_foot_locking_to_pose, foot_locking_metrics
from monocap_v2.core.level_a_benchmark import (
    backend_run_name,
    compare_pose_to_opensim_reference,
    failure_row,
    load_cached_wham_timeline,
    load_opensim_reference,
    load_pose_artifact,
    load_run_config,
    resolve_mocap_opensim_paths,
    row_from_report,
)
from monocap_v2.core.logging_utils import read_json, read_yaml, write_json
from monocap_v2.core.manifest import find_trial, load_opencap_manifest
from monocap_v2.core.subject_scale import apply_subject_scale_to_pose
from monocap_v2.core.temporal_smoothing import apply_temporal_smoothing_to_pose


DEFAULT_TRIALS = "walking1"
DEFAULT_BACKENDS = "metrabs,rtmw3d"
DEFAULT_VARIANTS = "baseline,best_scale,ma9_bone,root_toe,best_scale_ma9_bone,best_scale_ma9_bone_root_toe"
DEFAULT_SCALE_MODES = "metrabs=static_bone,rtmw3d=height_bone"
DEFAULT_OPENSIM_PYTHON = "/home/denik/miniconda3/envs/gaitlab/bin/python"
PIPELINE_STAGES = "stage_00_validate_inputs,stage_01_preprocess_video,stage_02_assume_camera,stage_03_pose2d,stage_04_pose3d_initial"

CSV_FIELDS = [
    "backend",
    "trial",
    "variant",
    "status",
    "run_dir",
    "scale_mode",
    "scale_status",
    "static_trial",
    "target_source",
    "smoothing_status",
    "smoothing_window_frames",
    "contact_lock_status",
    "locked_segment_count",
    "primary_root_centered_mpjpe_mm",
    "root_centered_rigid_mpjpe_mm",
    "root_centered_n_mpjpe_mm",
    "root_centered_similarity_mpjpe_mm",
    "pa_mpjpe_mm",
    "global_no_align_mpjpe_mm",
    "global_sequence_similarity_mpjpe_mm",
    "normal_minus_rigid_gap_mm",
    "overlap_frames",
    "mean_second_diff_before_m",
    "mean_second_diff_after_m",
    "mean_contact_horizontal_speed_before_mps",
    "mean_contact_horizontal_speed_after_mps",
    "mean_contact_position_std_before_m",
    "mean_contact_position_std_after_m",
    "max_contact_correction_m",
    "segment_length_errors_after_mm",
    "warnings",
    "error",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Combine subject scale, temporal smoothing, and contact foot locking in diagnostic Level A ablations.")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--paths", required=True, type=Path)
    ap.add_argument("--trials", default=DEFAULT_TRIALS)
    ap.add_argument("--activity", default="walking")
    ap.add_argument("--backends", default=DEFAULT_BACKENDS)
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--scale-modes", default=DEFAULT_SCALE_MODES, help="Comma-separated backend=mode entries, e.g. metrabs=static_bone,rtmw3d=height_bone.")
    ap.add_argument("--static-trial", default="auto", help="Static trial id, 'auto', or 'none'.")
    ap.add_argument("--run-static", action="store_true")
    ap.add_argument("--force-static", action="store_true")
    ap.add_argument("--evaluation-hz", type=float, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--opensim-python", type=Path, default=Path(DEFAULT_OPENSIM_PYTHON))
    ap.add_argument("--repo-root", type=Path, default=None, help=argparse.SUPPRESS)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root or Path(__file__).resolve().parents[2]
    manifest = load_opencap_manifest(args.manifest, args.paths)
    trials = _csv_arg(args.trials)
    backends = _csv_arg(args.backends)
    variants = _csv_arg(args.variants)
    scale_modes = _scale_mode_arg(args.scale_modes)
    out_dir = args.out or (
        repo_root / "monocap_v2" / "benchmarks" / f"{manifest.get('subject_id', 'subject')}_{args.activity}_refinement_combo"
    )
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = out_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    static_pose_cache: dict[str, Any] = {}
    for trial in trials:
        try:
            reference_paths = _ensure_reference(repo_root, args, manifest, trial, reference_dir)
            reference = load_opensim_reference(reference_paths["npz"], reference_paths["json"])
        except Exception as exc:
            for backend in backends:
                for variant in variants:
                    row = failure_row(backend, trial, None, "reference_failed", str(exc))
                    row["variant"] = variant
                    rows.append(row)
            continue

        for backend in backends:
            run_dir = repo_root / "monocap_v2" / "runs" / backend_run_name(manifest, trial, backend)
            try:
                pose = load_pose_artifact(run_dir)
                run_config = load_run_config(run_dir)
                subject = _load_subject(run_dir, pose)
                contacts = _load_or_estimate_contacts(run_dir, pose, args.activity)
                timeline = load_cached_wham_timeline(run_dir) if pose.get("backend") == "wham" else None
            except Exception as exc:
                for variant in variants:
                    row = failure_row(backend, trial, run_dir, "artifact_failed", str(exc))
                    row["variant"] = variant
                    rows.append(row)
                continue

            for variant in variants:
                try:
                    corrected, correction_report = _apply_variant(
                        repo_root,
                        args,
                        manifest,
                        backend,
                        pose,
                        run_config,
                        subject,
                        contacts,
                        variant,
                        scale_modes,
                        static_pose_cache,
                    )
                    if correction_report.get("status") == "skipped":
                        row = failure_row(backend, trial, run_dir, "skipped", correction_report.get("reason", "Variant skipped."))
                        row.update(_combo_row_fields(variant, correction_report))
                        rows.append(row)
                        continue
                    report, _series = compare_pose_to_opensim_reference(
                        corrected,
                        reference,
                        run_config=run_config,
                        timeline_report=timeline,
                        evaluation_hz=args.evaluation_hz,
                    )
                    row = row_from_report(backend, trial, run_dir, report, reference.get("metadata"))
                    row.update(_combo_row_fields(variant, correction_report))
                    rows.append(row)
                except Exception as exc:
                    row = failure_row(backend, trial, run_dir, "comparison_failed", str(exc))
                    row["variant"] = variant
                    rows.append(row)

    outputs = write_outputs(out_dir, rows, args, manifest, scale_modes)
    _print_summary(outputs, rows)
    return 0 if any(row.get("status") == "valid" for row in rows) else 1


def write_outputs(out_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace, manifest: dict[str, Any], scale_modes: dict[str, str]) -> dict[str, str]:
    csv_path = out_dir / "refinement_combo_ablation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in CSV_FIELDS})

    summary = {
        "metadata": {
            "manifest": str(args.manifest),
            "paths": str(args.paths),
            "subject_id": manifest.get("subject_id"),
            "trials": _csv_arg(args.trials),
            "backends": _csv_arg(args.backends),
            "variants": _csv_arg(args.variants),
            "scale_modes": scale_modes,
            "evaluation_hz": args.evaluation_hz,
            "static_trial": args.static_trial,
            "run_static": bool(args.run_static),
            "mocap_used_for_target_selection": False,
        },
        "best_by_backend_trial": _best_by_backend_trial(rows),
        "best_contact_by_backend_trial": _best_contact_by_backend_trial(rows),
        "rows": rows,
    }
    json_path = out_dir / "refinement_combo_ablation.json"
    write_json(json_path, summary)
    md_path = out_dir / "refinement_combo_ablation.md"
    _write_markdown(md_path, summary)
    return {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)}


def _apply_variant(
    repo_root: Path,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    backend: str,
    pose: dict[str, Any],
    run_config: dict[str, Any],
    subject: dict[str, Any],
    contacts: dict[str, Any],
    variant: str,
    scale_modes: dict[str, str],
    static_pose_cache: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    name = str(variant).strip().lower()
    current = pose
    scale_report: dict[str, Any] = {"status": "baseline", "mocap_used_in_objective": False}
    smoothing_report: dict[str, Any] = {"status": "baseline", "mocap_used_in_objective": False}
    lock_report = _lock_metrics_report(pose, contacts, status="baseline")
    static_trial = None

    if name != "baseline" and ("scale" in name or name.startswith("best_")):
        scale_mode = scale_modes.get(backend)
        if not scale_mode:
            return current, {"status": "skipped", "reason": f"No scale mode configured for backend {backend!r}.", "variant": variant}
        static_pose = None
        if scale_mode.startswith("static_"):
            static_trial = _resolve_static_trial(manifest, args.static_trial)
            if static_trial is None:
                return current, {"status": "skipped", "reason": "No static trial is available.", "variant": variant, "scale_mode": scale_mode}
            static_pose = _load_or_run_static_pose(repo_root, args, manifest, backend, static_trial, static_pose_cache)
            if static_pose is None:
                return current, {
                    "status": "skipped",
                    "reason": "Static pose artifact is unavailable; pass --run-static to create it.",
                    "variant": variant,
                    "scale_mode": scale_mode,
                    "static_trial": static_trial,
                }
        scale_cfg = _subject_scale_cfg(run_config)
        scale_cfg["mode"] = scale_mode
        current, scale_report = apply_subject_scale_to_pose(current, subject, scale_cfg, static_pose=static_pose)
        if scale_report.get("status") != "ok":
            return current, {"status": "skipped", "reason": scale_report.get("reason", "Subject-scale skipped."), "variant": variant, "scale": scale_report}

    if "ma9_bone" in name:
        smoothing_cfg = _temporal_smoothing_cfg(run_config)
        smoothing_cfg.update({"enabled": True, "method": "moving_average", "window_frames": 9, "preserve_bones": True})
        current, smoothing_report = apply_temporal_smoothing_to_pose(current, smoothing_cfg)
        if smoothing_report.get("status") != "ok":
            return current, {"status": "skipped", "reason": smoothing_report.get("reason", "Temporal smoothing skipped."), "variant": variant, "smoothing": smoothing_report}

    if "root_toe" in name:
        lock_cfg = _contact_foot_locking_cfg(run_config)
        lock_cfg.update(
            {
                "enabled": True,
                "mode": "root_translation",
                "feet": "toes",
                "lock_vertical": False,
                "smooth_correction_window_frames": 1,
            }
        )
        current, lock_report = apply_contact_foot_locking_to_pose(current, contacts, lock_cfg)
        if lock_report.get("status") != "ok":
            return current, {"status": "skipped", "reason": lock_report.get("reason", "Contact foot locking skipped."), "variant": variant, "contact_foot_locking": lock_report}
    else:
        lock_report = _lock_metrics_report(current, contacts, status="metrics_only")

    return current, {
        "status": "ok",
        "variant": variant,
        "static_trial": static_trial,
        "scale": scale_report,
        "smoothing": smoothing_report,
        "contact_foot_locking": lock_report,
        "mocap_used_in_objective": False,
    }


def _ensure_reference(repo_root: Path, args: argparse.Namespace, manifest: dict[str, Any], trial: str, reference_dir: Path) -> dict[str, Path]:
    paths = resolve_mocap_opensim_paths(manifest, trial)
    npz_path = reference_dir / f"opensim_fk_{trial}.npz"
    json_path = reference_dir / f"opensim_fk_{trial}.json"
    if npz_path.exists() and json_path.exists():
        return {"npz": npz_path, "json": json_path, **paths}
    for key in ("model", "ik_mot"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing OpenSim reference {key}: {paths[key]}")
    cmd = [
        str(args.opensim_python),
        str(repo_root / "monocap_v2" / "scripts" / "export_opensim_fk_reference.py"),
        "--model",
        str(paths["model"]),
        "--ik-mot",
        str(paths["ik_mot"]),
        "--out-npz",
        str(npz_path),
        "--out-json",
        str(json_path),
    ]
    if paths.get("ik_marker_errors") and paths["ik_marker_errors"].exists():
        cmd.extend(["--marker-errors", str(paths["ik_marker_errors"])])
    completed = subprocess.run(cmd, cwd=str(repo_root))
    if completed.returncode != 0:
        raise RuntimeError(f"OpenSim FK export failed for {trial} with return code {completed.returncode}.")
    return {"npz": npz_path, "json": json_path, **paths}


def _load_or_estimate_contacts(run_dir: Path, pose: dict[str, Any], activity: str) -> dict[str, Any]:
    path = run_dir / "contacts" / "contacts.npz"
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return {
            "left_heel": data["left_heel"],
            "left_toe": data["left_toe"],
            "right_heel": data["right_heel"],
            "right_toe": data["right_toe"],
            "backend": str(data["backend"]),
            "activity": str(data["activity"]),
        }
    return estimate_contacts_from_joints(pose, activity)


def _load_subject(run_dir: Path, pose: dict[str, Any]) -> dict[str, Any]:
    path = run_dir / "input" / "subject.json"
    if path.exists():
        return read_json(path)
    return dict(pose.get("subject") or {})


def _subject_scale_cfg(run_config: dict[str, Any]) -> dict[str, Any]:
    return dict(((((run_config or {}).get("config") or {}).get("optimization") or {}).get("subject_scale") or {}))


def _temporal_smoothing_cfg(run_config: dict[str, Any]) -> dict[str, Any]:
    return dict(((((run_config or {}).get("config") or {}).get("optimization") or {}).get("temporal_smoothing") or {}))


def _contact_foot_locking_cfg(run_config: dict[str, Any]) -> dict[str, Any]:
    return dict(((((run_config or {}).get("config") or {}).get("optimization") or {}).get("contact_foot_locking") or {}))


def _resolve_static_trial(manifest: dict[str, Any], requested: str) -> str | None:
    requested = str(requested or "auto")
    if requested.lower() == "none":
        return None
    if requested.lower() != "auto":
        find_trial(manifest, requested)
        return requested
    camera = str(manifest.get("camera") or "")
    fallback = None
    for _subset, trials in (manifest.get("trials") or {}).items():
        for trial in trials or []:
            trial_id = str(trial.get("id") or "")
            if not trial_id.lower().startswith("static"):
                continue
            fallback = fallback or trial_id
            video = str(trial.get("video_sync") or trial.get("video_raw") or "")
            if camera and f"/{camera}/" in video:
                return trial_id
    return fallback


def _load_or_run_static_pose(
    repo_root: Path,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    backend: str,
    static_trial: str,
    cache: dict[str, Any],
) -> dict[str, Any] | None:
    key = f"{backend}:{static_trial}"
    if key in cache:
        return cache[key]
    run_dir = repo_root / "monocap_v2" / "runs" / backend_run_name(manifest, static_trial, backend)
    pose_path = run_dir / "pose3d_initial" / "pose3d_initial.pkl"
    if (not pose_path.exists() or args.force_static) and args.run_static:
        cmd = [
            str(sys.executable),
            str(repo_root / "monocap_v2" / "run_pipeline.py"),
            "--manifest",
            str(args.manifest),
            "--paths",
            str(args.paths),
            "--trial",
            static_trial,
            "--activity",
            "other",
            "--stages",
            PIPELINE_STAGES,
            "--out",
            str(run_dir),
        ]
        preset = _backend_preset(backend)
        if preset:
            cmd.extend(["--preset", preset])
        else:
            cmd.extend(["--backend-pose3d", backend])
        if args.force_static:
            cmd.append("--force")
        completed = subprocess.run(cmd, cwd=str(repo_root))
        if completed.returncode != 0:
            return None
    if not pose_path.exists():
        return None
    with pose_path.open("rb") as f:
        cache[key] = pickle.load(f)
    return cache[key]


def _backend_preset(backend: str) -> str | None:
    if backend == "rtmw3d":
        return "opencap_rtmw3d"
    if backend == "wham":
        return "opencap_wham"
    return None


def _lock_metrics_report(pose: dict[str, Any], contacts: dict[str, Any], status: str) -> dict[str, Any]:
    metrics = foot_locking_metrics(
        pose["joints_3d"],
        [str(name) for name in pose.get("joint_names", [])],
        contacts,
        fps=float(pose.get("fps") or 30.0),
        contact_threshold=0.85,
        min_segment_frames=3,
        foot_keys=["left_toe", "right_toe"],
    )
    return {"status": status, "metrics_before": metrics, "metrics_after": metrics, "mocap_used_in_objective": False}


def _combo_row_fields(variant: str, report: dict[str, Any]) -> dict[str, Any]:
    scale = report.get("scale") or {}
    smoothing = report.get("smoothing") or {}
    lock = report.get("contact_foot_locking") or {}
    smoothing_before = smoothing.get("metrics_before") or {}
    smoothing_after = smoothing.get("metrics_after") or {}
    lock_before = lock.get("metrics_before") or {}
    lock_after = lock.get("metrics_after") or {}
    correction = lock.get("correction_summary") or {}
    return {
        "variant": variant,
        "scale_mode": scale.get("mode") or report.get("scale_mode"),
        "scale_status": scale.get("status"),
        "static_trial": report.get("static_trial"),
        "target_source": scale.get("target_source"),
        "smoothing_status": smoothing.get("status"),
        "smoothing_window_frames": smoothing.get("window_frames"),
        "contact_lock_status": lock.get("status"),
        "locked_segment_count": lock.get("locked_segment_count"),
        "mean_second_diff_before_m": smoothing_before.get("mean_second_diff_m"),
        "mean_second_diff_after_m": smoothing_after.get("mean_second_diff_m"),
        "mean_contact_horizontal_speed_before_mps": lock_before.get("mean_contact_horizontal_speed_mps"),
        "mean_contact_horizontal_speed_after_mps": lock_after.get("mean_contact_horizontal_speed_mps"),
        "mean_contact_position_std_before_m": lock_before.get("mean_contact_horizontal_position_std_m"),
        "mean_contact_position_std_after_m": lock_after.get("mean_contact_horizontal_position_std_m"),
        "max_contact_correction_m": correction.get("max_m"),
        "segment_length_errors_after_mm": scale.get("segment_length_errors_after_mm"),
    }


def _best_by_backend_trial(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    keys = sorted({(row.get("backend"), row.get("trial")) for row in rows})
    for backend, trial in keys:
        valid = [
            row
            for row in rows
            if row.get("backend") == backend
            and row.get("trial") == trial
            and row.get("status") == "valid"
            and row.get("primary_root_centered_mpjpe_mm") is not None
        ]
        if not valid:
            continue
        best = min(valid, key=lambda row: float(row["primary_root_centered_mpjpe_mm"]))
        baseline = next((row for row in valid if row.get("variant") == "baseline"), None)
        improvement = None
        if baseline and baseline.get("primary_root_centered_mpjpe_mm") is not None:
            improvement = float(baseline["primary_root_centered_mpjpe_mm"]) - float(best["primary_root_centered_mpjpe_mm"])
        out.append(
            {
                "backend": backend,
                "trial": trial,
                "best_variant": best.get("variant"),
                "best_primary_mpjpe_mm": best.get("primary_root_centered_mpjpe_mm"),
                "baseline_primary_mpjpe_mm": baseline.get("primary_root_centered_mpjpe_mm") if baseline else None,
                "improvement_vs_baseline_mm": improvement,
            }
        )
    return out


def _best_contact_by_backend_trial(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    keys = sorted({(row.get("backend"), row.get("trial")) for row in rows})
    for backend, trial in keys:
        valid = [
            row
            for row in rows
            if row.get("backend") == backend
            and row.get("trial") == trial
            and row.get("status") == "valid"
            and row.get("mean_contact_horizontal_speed_after_mps") is not None
        ]
        if not valid:
            continue
        best = min(valid, key=lambda row: float(row["mean_contact_horizontal_speed_after_mps"]))
        baseline = next((row for row in valid if row.get("variant") == "baseline"), None)
        reduction = None
        if baseline and baseline.get("mean_contact_horizontal_speed_after_mps") is not None:
            reduction = float(baseline["mean_contact_horizontal_speed_after_mps"]) - float(best["mean_contact_horizontal_speed_after_mps"])
        out.append(
            {
                "backend": backend,
                "trial": trial,
                "best_variant": best.get("variant"),
                "baseline_contact_speed_mps": baseline.get("mean_contact_horizontal_speed_after_mps") if baseline else None,
                "best_contact_speed_mps": best.get("mean_contact_horizontal_speed_after_mps"),
                "contact_speed_reduction_mps": reduction,
            }
        )
    return out


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    rows = summary.get("rows") or []
    with path.open("w", encoding="utf-8") as f:
        f.write("# monocap_v2 Refinement Combo Ablation\n\n")
        f.write("- Corrections use subject metadata/static predicted pose, backend joints, and contact probabilities only.\n")
        f.write("- Mocap/OpenSim FK is reporting-only and is not used to choose correction targets, contact windows, or outputs.\n\n")
        f.write("## Best By Backend/Trial: MPJPE\n\n")
        f.write("| Backend | Trial | Best Variant | Baseline MPJPE | Best MPJPE | Improvement |\n")
        f.write("|---|---|---|---:|---:|---:|\n")
        for item in summary.get("best_by_backend_trial") or []:
            f.write(
                f"| {item.get('backend')} | {item.get('trial')} | {item.get('best_variant')} | "
                f"{_fmt(item.get('baseline_primary_mpjpe_mm'))} | {_fmt(item.get('best_primary_mpjpe_mm'))} | "
                f"{_fmt(item.get('improvement_vs_baseline_mm'))} |\n"
            )
        f.write("\n## Best By Backend/Trial: Contact Speed\n\n")
        f.write("| Backend | Trial | Best Variant | Baseline Contact Speed | Best Contact Speed | Reduction |\n")
        f.write("|---|---|---|---:|---:|---:|\n")
        for item in summary.get("best_contact_by_backend_trial") or []:
            f.write(
                f"| {item.get('backend')} | {item.get('trial')} | {item.get('best_variant')} | "
                f"{_fmt(item.get('baseline_contact_speed_mps'))} | {_fmt(item.get('best_contact_speed_mps'))} | "
                f"{_fmt(item.get('contact_speed_reduction_mps'))} |\n"
            )
        f.write("\n## Rows\n\n")
        f.write("| Backend | Trial | Variant | Status | Primary | Rigid | PA | Contact Speed | Smooth After | Error |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row.get('backend')} | {row.get('trial')} | {row.get('variant')} | {row.get('status')} | "
                f"{_fmt(row.get('primary_root_centered_mpjpe_mm'))} | {_fmt(row.get('root_centered_rigid_mpjpe_mm'))} | "
                f"{_fmt(row.get('pa_mpjpe_mm'))} | {_fmt(row.get('mean_contact_horizontal_speed_after_mps'))} | "
                f"{_fmt(row.get('mean_second_diff_after_m'))} | {row.get('error') or ''} |\n"
            )


def _csv_arg(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _scale_mode_arg(value: str) -> dict[str, str]:
    out = {}
    for item in _csv_arg(value):
        if "=" not in item:
            raise ValueError(f"Scale mode entry must be backend=mode, got {item!r}.")
        backend, mode = item.split("=", 1)
        out[backend.strip()] = mode.strip()
    return out


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _print_summary(outputs: dict[str, str], rows: list[dict[str, Any]]) -> None:
    valid = len([row for row in rows if row.get("status") == "valid"])
    print(f"[refinement-combo] Valid rows: {valid}/{len(rows)}", flush=True)
    print(f"[refinement-combo] Summary: {outputs['markdown']}", flush=True)
    print(f"[refinement-combo] CSV: {outputs['csv']}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
