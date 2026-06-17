#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

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
from monocap_v2.core.logging_utils import read_json, write_json
from monocap_v2.core.manifest import load_opencap_manifest
from monocap_v2.core.reprojection_consistency import (
    align_pose2d_to_joints,
    apply_reprojection_consistency_to_pose,
    reprojection_metrics,
)


DEFAULT_TRIALS = "walking1"
DEFAULT_BACKENDS = "metrabs,rtmw3d"
DEFAULT_VARIANTS = "baseline,reproj25,reproj50,reproj100,reproj50_bone,reproj100_bone,reproj50_bone_limited"
DEFAULT_OPENSIM_PYTHON = "/home/denik/miniconda3/envs/gaitlab/bin/python"

CSV_FIELDS = [
    "backend",
    "trial",
    "variant",
    "status",
    "run_dir",
    "method",
    "blend",
    "preserve_bones",
    "valid_observation_ratio",
    "mean_reprojection_error_before_px",
    "mean_reprojection_error_after_px",
    "median_reprojection_error_before_px",
    "median_reprojection_error_after_px",
    "mean_depth_preserving_displacement_before_m",
    "mean_depth_preserving_displacement_after_m",
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
    ap = argparse.ArgumentParser(description="Ablate depth-preserving 2D reprojection consistency against Level A reports.")
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
        repo_root / "monocap_v2" / "benchmarks" / f"{manifest.get('subject_id', 'subject')}_{args.activity}_reprojection_consistency"
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
                camera = _load_camera(run_dir, pose)
                pose2d = _load_pose2d(run_dir, pose)
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
                    reproj_report = _baseline_reprojection_report(pose, camera, pose2d)
                    if variant != "baseline":
                        reproj_cfg = _variant_cfg(variant, run_config)
                        corrected, reproj_report = apply_reprojection_consistency_to_pose(pose, camera, pose2d, reproj_cfg)
                        if reproj_report.get("status") != "ok":
                            row = failure_row(
                                backend,
                                trial,
                                run_dir,
                                "skipped",
                                reproj_report.get("reason", "Reprojection consistency skipped."),
                            )
                            row.update(_reprojection_row_fields(variant, reproj_report))
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
                    row.update(_reprojection_row_fields(variant, reproj_report))
                    rows.append(row)
                except Exception as exc:
                    row = failure_row(backend, trial, run_dir, "comparison_failed", str(exc))
                    row["variant"] = variant
                    rows.append(row)

    outputs = write_outputs(out_dir, rows, args, manifest)
    _print_summary(outputs, rows)
    return 0 if any(row.get("status") == "valid" for row in rows) else 1


def write_outputs(out_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, str]:
    csv_path = out_dir / "reprojection_consistency_ablation.csv"
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
        "best_reprojection_by_backend_trial": _best_reprojection_by_backend_trial(rows),
        "rows": rows,
    }
    json_path = out_dir / "reprojection_consistency_ablation.json"
    write_json(json_path, summary)
    md_path = out_dir / "reprojection_consistency_ablation.md"
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


def _load_camera(run_dir: Path, pose: dict[str, Any]) -> dict[str, Any]:
    path = run_dir / "input" / "camera_assumed.json"
    if path.exists():
        return read_json(path)
    camera = ((pose.get("camera") or {}).get("intrinsics") or {})
    if camera:
        return dict(camera)
    raise FileNotFoundError(f"No camera artifact exists for run: {run_dir}")


def _load_pose2d(run_dir: Path, pose: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(pose.get("pose2d"), dict):
        return pose["pose2d"]
    path = run_dir / "pose2d" / "keypoints_2d.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "xy": data["xy"],
        "confidence": data["confidence"],
        "names": data["names"].tolist(),
        "fps": float(data["fps"]),
        "backend": str(data["backend"]),
    }


def _variant_cfg(variant: str, run_config: dict[str, Any] | None = None) -> dict[str, Any]:
    base = (
        (((run_config or {}).get("config") or {}).get("optimization") or {}).get("reprojection_consistency") or {}
    )
    cfg = dict(base)
    cfg["enabled"] = True
    name = variant.strip().lower()
    if name == "baseline":
        cfg["enabled"] = False
        return cfg
    preserve_bones = name.endswith("_bone") or "_bone_" in name
    limited = name.endswith("_limited")
    stem = name.replace("_bone", "").replace("_limited", "")
    if not stem.startswith("reproj"):
        raise ValueError(f"Unsupported reprojection consistency variant: {variant}")
    blend_pct = float(stem.replace("reproj", ""))
    cfg["blend"] = blend_pct / 100.0
    cfg["preserve_bones"] = preserve_bones
    cfg["max_joint_displacement_m"] = 0.10 if limited else cfg.get("max_joint_displacement_m", 0.20)
    cfg.setdefault("confidence_threshold", 0.25)
    cfg.setdefault("min_depth_m", 0.1)
    cfg.setdefault("bone_projection_iterations", 2)
    return cfg


def _baseline_reprojection_report(pose: dict[str, Any], camera: dict[str, Any], pose2d: dict[str, Any] | None) -> dict[str, Any]:
    if not pose2d:
        return {"status": "baseline", "reason": "No 2D keypoints available.", "mocap_used_in_objective": False}
    xy, conf, align = align_pose2d_to_joints(pose2d, [str(name) for name in pose.get("joint_names", [])], np.asarray(pose["joints_3d"]).shape[:2])
    if xy is None or conf is None:
        return {"status": "baseline", "pose2d_alignment": align, "mocap_used_in_objective": False}
    metrics = reprojection_metrics(np.asarray(pose["joints_3d"], dtype=float), camera, xy, conf)
    return {
        "status": "baseline",
        "pose2d_alignment": align,
        "metrics_before": metrics,
        "metrics_after": metrics,
        "mocap_used_in_objective": False,
    }


def _reprojection_row_fields(variant: str, report: dict[str, Any]) -> dict[str, Any]:
    before = report.get("metrics_before") or {}
    after = report.get("metrics_after") or {}
    return {
        "variant": variant,
        "method": report.get("method"),
        "blend": report.get("blend"),
        "preserve_bones": report.get("preserve_bones"),
        "valid_observation_ratio": report.get("valid_observation_ratio") or after.get("valid_observation_ratio"),
        "mean_reprojection_error_before_px": before.get("mean_reprojection_error_px"),
        "mean_reprojection_error_after_px": after.get("mean_reprojection_error_px"),
        "median_reprojection_error_before_px": before.get("median_reprojection_error_px"),
        "median_reprojection_error_after_px": after.get("median_reprojection_error_px"),
        "mean_depth_preserving_displacement_before_m": before.get("mean_depth_preserving_displacement_m"),
        "mean_depth_preserving_displacement_after_m": after.get("mean_depth_preserving_displacement_m"),
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


def _best_reprojection_by_backend_trial(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    keys = sorted({(row.get("backend"), row.get("trial")) for row in rows})
    for backend, trial in keys:
        valid = [
            row
            for row in rows
            if row.get("backend") == backend
            and row.get("trial") == trial
            and row.get("status") == "valid"
            and row.get("mean_reprojection_error_after_px") is not None
        ]
        if not valid:
            continue
        best = min(valid, key=lambda row: float(row["mean_reprojection_error_after_px"]))
        baseline = next((row for row in valid if row.get("variant") == "baseline"), None)
        reduction = None
        if baseline and baseline.get("mean_reprojection_error_after_px") is not None:
            reduction = float(baseline["mean_reprojection_error_after_px"]) - float(best["mean_reprojection_error_after_px"])
        out.append(
            {
                "backend": backend,
                "trial": trial,
                "best_variant": best.get("variant"),
                "baseline_reprojection_error_px": baseline.get("mean_reprojection_error_after_px") if baseline else None,
                "best_reprojection_error_px": best.get("mean_reprojection_error_after_px"),
                "reprojection_error_reduction_px": reduction,
            }
        )
    return out


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    rows = summary.get("rows") or []
    with path.open("w", encoding="utf-8") as f:
        f.write("# monocap_v2 Reprojection Consistency Ablation\n\n")
        f.write("- Reprojection consistency uses backend 2D keypoints, resolved camera intrinsics, and predicted 3D depth only.\n")
        f.write("- Mocap/OpenSim FK is reporting-only and is not used to choose corrections.\n\n")
        f.write("## Best By Backend/Trial: MPJPE\n\n")
        f.write("| Backend | Trial | Best Variant | Baseline MPJPE | Best MPJPE | Improvement |\n")
        f.write("|---|---|---|---:|---:|---:|\n")
        for item in summary.get("best_by_backend_trial") or []:
            f.write(
                f"| {item.get('backend')} | {item.get('trial')} | {item.get('best_variant')} | "
                f"{_fmt(item.get('baseline_primary_mpjpe_mm'))} | {_fmt(item.get('best_primary_mpjpe_mm'))} | "
                f"{_fmt(item.get('improvement_vs_baseline_mm'))} |\n"
            )
        f.write("\n## Best By Backend/Trial: Reprojection Error\n\n")
        f.write("| Backend | Trial | Best Variant | Baseline Reproj | Best Reproj | Reduction |\n")
        f.write("|---|---|---|---:|---:|---:|\n")
        for item in summary.get("best_reprojection_by_backend_trial") or []:
            f.write(
                f"| {item.get('backend')} | {item.get('trial')} | {item.get('best_variant')} | "
                f"{_fmt(item.get('baseline_reprojection_error_px'))} | {_fmt(item.get('best_reprojection_error_px'))} | "
                f"{_fmt(item.get('reprojection_error_reduction_px'))} |\n"
            )
        f.write("\n## Rows\n\n")
        f.write("| Backend | Trial | Variant | Status | Primary | Rigid | PA | Reproj Before | Reproj After | Error |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row.get('backend')} | {row.get('trial')} | {row.get('variant')} | {row.get('status')} | "
                f"{_fmt(row.get('primary_root_centered_mpjpe_mm'))} | {_fmt(row.get('root_centered_rigid_mpjpe_mm'))} | "
                f"{_fmt(row.get('pa_mpjpe_mm'))} | {_fmt(row.get('mean_reprojection_error_before_px'))} | "
                f"{_fmt(row.get('mean_reprojection_error_after_px'))} | {row.get('error') or ''} |\n"
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
    print(f"[reprojection-consistency] Valid rows: {valid}/{len(rows)}", flush=True)
    print(f"[reprojection-consistency] Summary: {outputs['markdown']}", flush=True)
    print(f"[reprojection-consistency] CSV: {outputs['csv']}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
