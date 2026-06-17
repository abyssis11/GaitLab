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

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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


DEFAULT_TRIALS = "walking1"
DEFAULT_BACKENDS = "metrabs,rtmw3d"
DEFAULT_VARIANTS = "baseline,height_global,height_bone,static_global,static_bone"
DEFAULT_OPENSIM_PYTHON = "/home/denik/miniconda3/envs/gaitlab/bin/python"
PIPELINE_STAGES = "stage_00_validate_inputs,stage_01_preprocess_video,stage_02_assume_camera,stage_03_pose2d,stage_04_pose3d_initial"

CSV_FIELDS = [
    "backend",
    "trial",
    "variant",
    "status",
    "run_dir",
    "static_trial",
    "target_source",
    "subject_height_m",
    "global_scale",
    "primary_root_centered_mpjpe_mm",
    "root_centered_rigid_mpjpe_mm",
    "root_centered_n_mpjpe_mm",
    "root_centered_similarity_mpjpe_mm",
    "pa_mpjpe_mm",
    "global_no_align_mpjpe_mm",
    "global_sequence_similarity_mpjpe_mm",
    "normal_minus_rigid_gap_mm",
    "overlap_frames",
    "segment_length_errors_before_mm",
    "segment_length_errors_after_mm",
    "warnings",
    "error",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ablate subject-height/static bone-length correction against Level A reports.")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--paths", required=True, type=Path)
    ap.add_argument("--trials", default=DEFAULT_TRIALS)
    ap.add_argument("--activity", default="walking")
    ap.add_argument("--backends", default=DEFAULT_BACKENDS)
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--static-trial", default="auto", help="Static trial id, 'auto', or 'none'.")
    ap.add_argument("--run-static", action="store_true", help="Run/cache static stage 00-04 when static artifacts are missing.")
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
    out_dir = args.out or (
        repo_root / "monocap_v2" / "benchmarks" / f"{manifest.get('subject_id', 'subject')}_{args.activity}_subject_scale"
    )
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = out_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    references: dict[str, dict[str, Path]] = {}
    for trial in trials:
        try:
            references[trial] = _ensure_reference(repo_root, args, manifest, trial, reference_dir)
        except Exception as exc:
            for backend in backends:
                for variant in variants:
                    row = failure_row(backend, trial, None, "reference_failed", str(exc))
                    row["variant"] = variant
                    rows.append(row)
            continue

        reference = load_opensim_reference(references[trial]["npz"], references[trial]["json"])
        for backend in backends:
            run_dir = repo_root / "monocap_v2" / "runs" / backend_run_name(manifest, trial, backend)
            try:
                pose = load_pose_artifact(run_dir)
                run_config = load_run_config(run_dir)
                subject = _load_subject(run_dir, pose)
                scale_cfg = _subject_scale_cfg(run_config)
                timeline = load_cached_wham_timeline(run_dir) if pose.get("backend") == "wham" else None
            except Exception as exc:
                for variant in variants:
                    row = failure_row(backend, trial, run_dir, "artifact_failed", str(exc))
                    row["variant"] = variant
                    rows.append(row)
                continue

            static_pose_cache: dict[str, Any] = {}
            for variant in variants:
                try:
                    static_trial = None
                    corrected = pose
                    scale_report: dict[str, Any] = {"status": "baseline", "mode": "baseline", "mocap_used_in_objective": False}
                    if variant != "baseline":
                        static_pose = None
                        if variant.startswith("static_"):
                            static_trial = _resolve_static_trial(manifest, args.static_trial)
                            if static_trial is None:
                                rows.append(_skipped_row(backend, trial, variant, run_dir, "No static trial is available.", static_trial))
                                continue
                            static_pose = _load_or_run_static_pose(repo_root, args, manifest, backend, static_trial, static_pose_cache)
                            if static_pose is None:
                                rows.append(
                                    _skipped_row(
                                        backend,
                                        trial,
                                        variant,
                                        run_dir,
                                        "Static pose artifact is unavailable; pass --run-static to create it.",
                                        static_trial,
                                    )
                                )
                                continue
                        variant_cfg = dict(scale_cfg)
                        variant_cfg["mode"] = variant
                        corrected, scale_report = apply_subject_scale_to_pose(pose, subject, variant_cfg, static_pose=static_pose)
                        if scale_report.get("status") != "ok":
                            rows.append(_skipped_row(backend, trial, variant, run_dir, scale_report.get("reason", "Correction skipped."), static_trial, scale_report))
                            continue

                    report, _series = compare_pose_to_opensim_reference(
                        corrected,
                        reference,
                        run_config=run_config,
                        timeline_report=timeline,
                        evaluation_hz=args.evaluation_hz,
                    )
                    row = row_from_report(backend, trial, run_dir, report, reference.get("metadata"))
                    row.update(_scale_row_fields(variant, scale_report, static_trial))
                    rows.append(row)
                except Exception as exc:
                    row = failure_row(backend, trial, run_dir, "comparison_failed", str(exc))
                    row["variant"] = variant
                    rows.append(row)

    outputs = write_outputs(out_dir, rows, args, manifest)
    _print_summary(outputs, rows)
    return 0 if any(row.get("status") == "valid" for row in rows) else 1


def write_outputs(out_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, str]:
    csv_path = out_dir / "subject_scale_ablation.csv"
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
            "evaluation_hz": args.evaluation_hz,
            "static_trial": args.static_trial,
            "run_static": bool(args.run_static),
            "mocap_used_for_target_selection": False,
        },
        "best_by_backend_trial": _best_by_backend_trial(rows),
        "rows": rows,
    }
    json_path = out_dir / "subject_scale_ablation.json"
    write_json(json_path, summary)
    md_path = out_dir / "subject_scale_ablation.md"
    _write_markdown(md_path, summary)
    return {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)}


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


def _load_subject(run_dir: Path, pose: dict[str, Any]) -> dict[str, Any]:
    path = run_dir / "input" / "subject.json"
    if path.exists():
        return read_json(path)
    return dict(pose.get("subject") or {})


def _subject_scale_cfg(run_config: dict[str, Any]) -> dict[str, Any]:
    cfg = (((run_config or {}).get("config") or {}).get("optimization") or {}).get("subject_scale") or {}
    return dict(cfg)


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


def _scale_row_fields(variant: str, scale_report: dict[str, Any], static_trial: str | None) -> dict[str, Any]:
    global_scale = scale_report.get("global_scale") or {}
    return {
        "variant": variant,
        "static_trial": static_trial,
        "target_source": scale_report.get("target_source"),
        "subject_height_m": scale_report.get("subject_height_m"),
        "global_scale": global_scale.get("scale"),
        "segment_length_errors_before_mm": scale_report.get("segment_length_errors_before_mm"),
        "segment_length_errors_after_mm": scale_report.get("segment_length_errors_after_mm"),
    }


def _skipped_row(
    backend: str,
    trial: str,
    variant: str,
    run_dir: Path,
    reason: str,
    static_trial: str | None,
    scale_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = failure_row(backend, trial, run_dir, "skipped", reason)
    row.update(_scale_row_fields(variant, scale_report or {}, static_trial))
    return row


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


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    rows = summary.get("rows") or []
    with path.open("w", encoding="utf-8") as f:
        f.write("# monocap_v2 Subject-Scale Ablation\n\n")
        f.write("- Target selection uses `subject height` and optional `static predicted pose` only.\n")
        f.write("- Mocap/OpenSim FK is reporting-only and is not used to choose scale targets.\n\n")
        f.write("## Best By Backend/Trial\n\n")
        f.write("| Backend | Trial | Best Variant | Baseline MPJPE | Best MPJPE | Improvement |\n")
        f.write("|---|---|---|---:|---:|---:|\n")
        for item in summary.get("best_by_backend_trial") or []:
            f.write(
                f"| {item.get('backend')} | {item.get('trial')} | {item.get('best_variant')} | "
                f"{_fmt(item.get('baseline_primary_mpjpe_mm'))} | {_fmt(item.get('best_primary_mpjpe_mm'))} | "
                f"{_fmt(item.get('improvement_vs_baseline_mm'))} |\n"
            )
        f.write("\n## Rows\n\n")
        f.write("| Backend | Trial | Variant | Status | Primary | Rigid | PA | Scale | Target | Error |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            f.write(
                f"| {row.get('backend')} | {row.get('trial')} | {row.get('variant')} | {row.get('status')} | "
                f"{_fmt(row.get('primary_root_centered_mpjpe_mm'))} | {_fmt(row.get('root_centered_rigid_mpjpe_mm'))} | "
                f"{_fmt(row.get('pa_mpjpe_mm'))} | {_fmt(row.get('global_scale'))} | "
                f"{row.get('target_source') or ''} | {row.get('error') or ''} |\n"
            )


def _csv_arg(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


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
    print(f"[subject-scale] Valid rows: {valid}/{len(rows)}", flush=True)
    print(f"[subject-scale] Summary: {outputs['markdown']}", flush=True)
    print(f"[subject-scale] CSV: {outputs['csv']}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
