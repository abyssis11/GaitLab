#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
from monocap_v2.core.logging_utils import write_json
from monocap_v2.core.manifest import load_opencap_manifest
from monocap_v2.core.temporal_smoothing import apply_temporal_smoothing_to_pose


DEFAULT_TRIALS = "walking1"
DEFAULT_BACKENDS = "metrabs,rtmw3d"
DEFAULT_VARIANTS = "baseline,ma5,ma7_bone,ma9_bone,savgol7_bone,savgol9_bone"
DEFAULT_OPENSIM_PYTHON = "/home/denik/miniconda3/envs/gaitlab/bin/python"

CSV_FIELDS = [
    "backend",
    "trial",
    "variant",
    "status",
    "run_dir",
    "method",
    "window_frames",
    "preserve_bones",
    "smooth_root",
    "mean_second_diff_before_m",
    "mean_second_diff_after_m",
    "rms_second_diff_before_m",
    "rms_second_diff_after_m",
    "median_bone_length_error_after_mm",
    "max_bone_length_error_after_mm",
    "primary_root_centered_mpjpe_mm",
    "root_centered_rigid_mpjpe_mm",
    "root_centered_n_mpjpe_mm",
    "root_centered_similarity_mpjpe_mm",
    "pa_mpjpe_mm",
    "global_no_align_mpjpe_mm",
    "global_sequence_similarity_mpjpe_mm",
    "normal_minus_rigid_gap_mm",
    "overlap_frames",
    "warnings",
    "error",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ablate temporal smoothing variants against Level A reports.")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--paths", required=True, type=Path)
    ap.add_argument("--trials", default=DEFAULT_TRIALS)
    ap.add_argument("--activity", default="walking")
    ap.add_argument("--backends", default=DEFAULT_BACKENDS)
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
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
        repo_root / "monocap_v2" / "benchmarks" / f"{manifest.get('subject_id', 'subject')}_{args.activity}_temporal_smoothing"
    )
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = out_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
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
                timeline = load_cached_wham_timeline(run_dir) if pose.get("backend") == "wham" else None
            except Exception as exc:
                for variant in variants:
                    row = failure_row(backend, trial, run_dir, "artifact_failed", str(exc))
                    row["variant"] = variant
                    rows.append(row)
                continue

            for variant in variants:
                try:
                    corrected = pose
                    smoothing_report: dict[str, Any] = {"status": "baseline", "mocap_used_in_objective": False}
                    if variant != "baseline":
                        smoothing_cfg = _variant_cfg(variant, run_config)
                        corrected, smoothing_report = apply_temporal_smoothing_to_pose(pose, smoothing_cfg)
                        if smoothing_report.get("status") != "ok":
                            row = failure_row(
                                backend,
                                trial,
                                run_dir,
                                "skipped",
                                smoothing_report.get("reason", "Temporal smoothing skipped."),
                            )
                            row.update(_smoothing_row_fields(variant, smoothing_report))
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
                    row.update(_smoothing_row_fields(variant, smoothing_report))
                    rows.append(row)
                except Exception as exc:
                    row = failure_row(backend, trial, run_dir, "comparison_failed", str(exc))
                    row["variant"] = variant
                    rows.append(row)

    outputs = write_outputs(out_dir, rows, args, manifest)
    _print_summary(outputs, rows)
    return 0 if any(row.get("status") == "valid" for row in rows) else 1


def write_outputs(out_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, str]:
    csv_path = out_dir / "temporal_smoothing_ablation.csv"
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
            "mocap_used_for_target_selection": False,
        },
        "best_by_backend_trial": _best_by_backend_trial(rows),
        "rows": rows,
    }
    json_path = out_dir / "temporal_smoothing_ablation.json"
    write_json(json_path, summary)
    md_path = out_dir / "temporal_smoothing_ablation.md"
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


def _variant_cfg(variant: str, run_config: dict[str, Any] | None = None) -> dict[str, Any]:
    base = (
        (((run_config or {}).get("config") or {}).get("optimization") or {}).get("temporal_smoothing") or {}
    )
    cfg = dict(base)
    cfg["enabled"] = True
    name = variant.strip().lower()
    if name == "baseline":
        cfg["enabled"] = False
        return cfg
    preserve_bones = name.endswith("_bone")
    stem = name[:-5] if preserve_bones else name
    if stem.startswith("ma"):
        cfg["method"] = "moving_average"
        cfg["window_frames"] = int(stem[2:])
    elif stem.startswith("savgol"):
        cfg["method"] = "savgol"
        cfg["window_frames"] = int(stem[6:])
    else:
        raise ValueError(f"Unsupported temporal smoothing variant: {variant}")
    cfg["preserve_bones"] = preserve_bones
    cfg.setdefault("smooth_root", True)
    cfg.setdefault("passes", 1)
    cfg.setdefault("bone_projection_iterations", 2)
    cfg.setdefault("bone_preservation_blend", 1.0)
    cfg.setdefault("max_joint_displacement_m", 0.15)
    return cfg


def _smoothing_row_fields(variant: str, smoothing_report: dict[str, Any]) -> dict[str, Any]:
    before = smoothing_report.get("metrics_before") or {}
    after = smoothing_report.get("metrics_after") or {}
    return {
        "variant": variant,
        "method": smoothing_report.get("method"),
        "window_frames": smoothing_report.get("window_frames"),
        "preserve_bones": smoothing_report.get("preserve_bones"),
        "smooth_root": smoothing_report.get("smooth_root"),
        "mean_second_diff_before_m": before.get("mean_second_diff_m"),
        "mean_second_diff_after_m": after.get("mean_second_diff_m"),
        "rms_second_diff_before_m": before.get("rms_second_diff_m"),
        "rms_second_diff_after_m": after.get("rms_second_diff_m"),
        "median_bone_length_error_after_mm": after.get("median_bone_length_error_mm"),
        "max_bone_length_error_after_mm": after.get("max_bone_length_error_mm"),
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


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    rows = summary.get("rows") or []
    with path.open("w", encoding="utf-8") as f:
        f.write("# monocap_v2 Temporal Smoothing Ablation\n\n")
        f.write("- Smoothing uses only backend joint trajectories and original skeleton lengths.\n")
        f.write("- Mocap/OpenSim FK is reporting-only and is not used to choose smoothing parameters.\n\n")
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
        f.write("| Backend | Trial | Variant | Status | Primary | Rigid | PA | Smooth Before | Smooth After | Bone Error | Error |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row.get('backend')} | {row.get('trial')} | {row.get('variant')} | {row.get('status')} | "
                f"{_fmt(row.get('primary_root_centered_mpjpe_mm'))} | {_fmt(row.get('root_centered_rigid_mpjpe_mm'))} | "
                f"{_fmt(row.get('pa_mpjpe_mm'))} | {_fmt(row.get('mean_second_diff_before_m'))} | "
                f"{_fmt(row.get('mean_second_diff_after_m'))} | {_fmt(row.get('median_bone_length_error_after_mm'))} | "
                f"{row.get('error') or ''} |\n"
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
    print(f"[temporal-smoothing] Valid rows: {valid}/{len(rows)}", flush=True)
    print(f"[temporal-smoothing] Summary: {outputs['markdown']}", flush=True)
    print(f"[temporal-smoothing] CSV: {outputs['csv']}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
