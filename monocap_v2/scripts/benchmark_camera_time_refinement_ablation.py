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

from monocap_v2.core.camera_time_refinement import (
    apply_camera_time_refinement_to_pose,
    reprojection_time_metrics,
)
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
from monocap_v2.core.reprojection_consistency import align_pose2d_to_joints


DEFAULT_TRIALS = "walking1"
DEFAULT_BACKENDS = "metrabs,rtmw3d"
DEFAULT_VARIANTS = "baseline,time_only,camera_only,camera_tight,time_camera,time_camera_tight"
DEFAULT_OPENSIM_PYTHON = "/home/denik/miniconda3/envs/gaitlab/bin/python"

CSV_FIELDS = [
    "backend",
    "trial",
    "variant",
    "status",
    "run_dir",
    "method",
    "selected_time_offset_ms",
    "camera_delta_status",
    "camera_rotation_deg",
    "camera_translation_m",
    "mean_reprojection_error_before_px",
    "mean_reprojection_error_after_px",
    "median_reprojection_error_before_px",
    "median_reprojection_error_after_px",
    "motion_correlation_before",
    "motion_correlation_after",
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
    ap = argparse.ArgumentParser(description="Ablate non-GT camera/time refinement against Level A reports.")
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
        repo_root / "monocap_v2" / "benchmarks" / f"{manifest.get('subject_id', 'subject')}_{args.activity}_camera_time_refinement"
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
                    if variant == "baseline":
                        corrected = pose
                        camera_time_report = _baseline_report(pose, camera, pose2d)
                    else:
                        variant_cfg = _variant_cfg(variant, run_config)
                        corrected, camera_time_report = apply_camera_time_refinement_to_pose(pose, camera, pose2d, variant_cfg)
                        if camera_time_report.get("status") != "ok":
                            row = failure_row(
                                backend,
                                trial,
                                run_dir,
                                "skipped",
                                camera_time_report.get("reason", "Camera/time refinement skipped."),
                            )
                            row.update(_camera_time_row_fields(variant, camera_time_report))
                            rows.append(row)
                            continue
                    report, _series = compare_pose_to_opensim_reference(
                        corrected,
                        reference,
                        run_config=run_config,
                        timeline_report=timeline,
                        evaluation_hz=args.evaluation_hz,
                        diagnostic_time_offset_s=float(camera_time_report.get("time_offset_for_evaluation_s") or 0.0),
                    )
                    row = row_from_report(backend, trial, run_dir, report, reference.get("metadata"))
                    row.update(_camera_time_row_fields(variant, camera_time_report))
                    row["warnings"] = _merge_warnings(row.get("warnings"), camera_time_report.get("warnings"))
                    rows.append(row)
                except Exception as exc:
                    row = failure_row(backend, trial, run_dir, "comparison_failed", str(exc))
                    row["variant"] = variant
                    rows.append(row)

    outputs = write_outputs(out_dir, rows, args, manifest)
    _print_summary(outputs, rows)
    return 0 if any(row.get("status") == "valid" for row in rows) else 1


def write_outputs(out_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, str]:
    csv_path = out_dir / "camera_time_refinement_ablation.csv"
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
    json_path = out_dir / "camera_time_refinement_ablation.json"
    write_json(json_path, summary)
    md_path = out_dir / "camera_time_refinement_ablation.md"
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
        (((run_config or {}).get("config") or {}).get("optimization") or {}).get("camera_time_refinement") or {}
    )
    cfg = dict(base)
    cfg["enabled"] = True
    cfg.setdefault("confidence_threshold", 0.25)
    cfg.setdefault("min_depth_m", 0.1)
    name = variant.strip().lower()
    if name == "baseline":
        cfg["enabled"] = False
        return cfg
    time_cfg = dict(cfg.get("time_search") or {})
    camera_cfg = dict(cfg.get("camera_delta") or {})
    time_cfg.setdefault("min_offset_s", -0.25)
    time_cfg.setdefault("max_offset_s", 0.25)
    time_cfg.setdefault("step_s", 0.02)
    camera_cfg.setdefault("max_rotation_deg", 8.0)
    camera_cfg.setdefault("max_translation_m", 0.25)
    camera_cfg.setdefault("max_nfev", 40)
    if name == "time_only":
        time_cfg["enabled"] = True
        camera_cfg["enabled"] = False
    elif name == "camera_only":
        time_cfg["enabled"] = False
        cfg["time_offset_s"] = 0.0
        camera_cfg["enabled"] = True
    elif name == "camera_tight":
        time_cfg["enabled"] = False
        cfg["time_offset_s"] = 0.0
        camera_cfg["enabled"] = True
        camera_cfg["max_rotation_deg"] = 3.0
        camera_cfg["max_translation_m"] = 0.10
    elif name == "time_camera":
        time_cfg["enabled"] = True
        camera_cfg["enabled"] = True
    elif name == "time_camera_tight":
        time_cfg["enabled"] = True
        camera_cfg["enabled"] = True
        camera_cfg["max_rotation_deg"] = 3.0
        camera_cfg["max_translation_m"] = 0.10
    else:
        raise ValueError(f"Unsupported camera/time variant: {variant}")
    cfg["time_search"] = time_cfg
    cfg["camera_delta"] = camera_cfg
    return cfg


def _baseline_report(pose: dict[str, Any], camera: dict[str, Any], pose2d: dict[str, Any] | None) -> dict[str, Any]:
    if not pose2d:
        return {"status": "baseline", "reason": "No 2D keypoints available.", "mocap_used_in_objective": False}
    joints = np.asarray(pose["joints_3d"], dtype=float)
    xy, conf, align = align_pose2d_to_joints(pose2d, [str(name) for name in pose.get("joint_names", [])], joints.shape[:2])
    if xy is None or conf is None:
        return {"status": "baseline", "pose2d_alignment": align, "mocap_used_in_objective": False}
    fps = float(pose.get("fps") or pose2d.get("fps") or 30.0)
    pose2d_fps = float(pose2d.get("fps") or fps)
    metrics = reprojection_time_metrics(joints, camera, xy, conf, fps, pose2d_fps, 0.0)
    return {
        "status": "baseline",
        "method": "baseline",
        "time_offset_for_evaluation_s": 0.0,
        "selected_time_offset_s": 0.0,
        "selected_time_offset_ms": 0.0,
        "pose2d_alignment": align,
        "metrics_before": metrics,
        "metrics_after": metrics,
        "camera_delta": {"status": "skipped", "enabled": False},
        "mocap_used_in_objective": False,
    }


def _camera_time_row_fields(variant: str, report: dict[str, Any]) -> dict[str, Any]:
    before = report.get("metrics_before") or {}
    after = report.get("metrics_after") or {}
    camera_delta = report.get("camera_delta") or {}
    return {
        "variant": variant,
        "method": report.get("method"),
        "selected_time_offset_ms": report.get("selected_time_offset_ms"),
        "camera_delta_status": camera_delta.get("status"),
        "camera_rotation_deg": camera_delta.get("rotation_magnitude_deg"),
        "camera_translation_m": camera_delta.get("translation_magnitude_m"),
        "mean_reprojection_error_before_px": before.get("mean_reprojection_error_px"),
        "mean_reprojection_error_after_px": after.get("mean_reprojection_error_px"),
        "median_reprojection_error_before_px": before.get("median_reprojection_error_px"),
        "median_reprojection_error_after_px": after.get("median_reprojection_error_px"),
        "motion_correlation_before": before.get("motion_correlation"),
        "motion_correlation_after": after.get("motion_correlation"),
    }


def _merge_warnings(existing: Any, extra: Any) -> list[str]:
    out: list[str] = []
    if isinstance(existing, list):
        out.extend(str(item) for item in existing)
    elif existing:
        out.append(str(existing))
    if isinstance(extra, list):
        out.extend(str(item) for item in extra)
    elif extra:
        out.append(str(extra))
    return out


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
                "selected_time_offset_ms": best.get("selected_time_offset_ms"),
                "camera_rotation_deg": best.get("camera_rotation_deg"),
                "camera_translation_m": best.get("camera_translation_m"),
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
        f.write("# monocap_v2 Camera/Time Refinement Ablation\n\n")
        f.write("- Time offsets and camera deltas are estimated from 2D reprojection only.\n")
        f.write("- Mocap/OpenSim FK is reporting-only and is not used to choose corrections.\n")
        f.write("- Positive selected offsets use Level A semantics: prediction time is evaluated as native time minus offset.\n\n")
        f.write("## Best By Backend/Trial: MPJPE\n\n")
        f.write("| Backend | Trial | Best Variant | Baseline MPJPE | Best MPJPE | Improvement | Offset ms | Rot deg | Trans m |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|\n")
        for item in summary.get("best_by_backend_trial") or []:
            f.write(
                f"| {item.get('backend')} | {item.get('trial')} | {item.get('best_variant')} | "
                f"{_fmt(item.get('baseline_primary_mpjpe_mm'))} | {_fmt(item.get('best_primary_mpjpe_mm'))} | "
                f"{_fmt(item.get('improvement_vs_baseline_mm'))} | {_fmt(item.get('selected_time_offset_ms'))} | "
                f"{_fmt(item.get('camera_rotation_deg'))} | {_fmt(item.get('camera_translation_m'))} |\n"
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
        f.write("| Backend | Trial | Variant | Status | Primary | Rigid | PA | Offset ms | Rot deg | Trans m | Reproj Before | Reproj After | Warnings | Error |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            f.write(
                f"| {row.get('backend')} | {row.get('trial')} | {row.get('variant')} | {row.get('status')} | "
                f"{_fmt(row.get('primary_root_centered_mpjpe_mm'))} | {_fmt(row.get('root_centered_rigid_mpjpe_mm'))} | "
                f"{_fmt(row.get('pa_mpjpe_mm'))} | {_fmt(row.get('selected_time_offset_ms'))} | "
                f"{_fmt(row.get('camera_rotation_deg'))} | {_fmt(row.get('camera_translation_m'))} | "
                f"{_fmt(row.get('mean_reprojection_error_before_px'))} | {_fmt(row.get('mean_reprojection_error_after_px'))} | "
                f"{_warning_text(row.get('warnings'))} | {row.get('error') or ''} |\n"
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


def _warning_text(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        return "<br>".join(str(item) for item in value)
    return str(value)


def _print_summary(outputs: dict[str, str], rows: list[dict[str, Any]]) -> None:
    valid = len([row for row in rows if row.get("status") == "valid"])
    print(f"[camera-time] Valid rows: {valid}/{len(rows)}", flush=True)
    print(f"[camera-time] Summary: {outputs['markdown']}", flush=True)
    print(f"[camera-time] CSV: {outputs['csv']}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
