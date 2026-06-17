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
from monocap_v2.core.logging_utils import write_json
from monocap_v2.core.manifest import load_opencap_manifest


DEFAULT_TRIALS = "walking1"
DEFAULT_BACKENDS = "metrabs,rtmw3d"
DEFAULT_VARIANTS = "baseline,root_toe,root_toe_smooth,root_all_smooth,endpoint_toe"
DEFAULT_OPENSIM_PYTHON = "/home/denik/miniconda3/envs/gaitlab/bin/python"

CSV_FIELDS = [
    "backend",
    "trial",
    "variant",
    "status",
    "run_dir",
    "mode",
    "feet",
    "lock_vertical",
    "locked_segment_count",
    "mean_contact_horizontal_speed_before_mps",
    "mean_contact_horizontal_speed_after_mps",
    "mean_contact_position_std_before_m",
    "mean_contact_position_std_after_m",
    "max_correction_m",
    "active_correction_frames",
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
    ap = argparse.ArgumentParser(description="Ablate contact-aware foot locking against Level A reports.")
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
        repo_root / "monocap_v2" / "benchmarks" / f"{manifest.get('subject_id', 'subject')}_{args.activity}_contact_foot_locking"
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
                    corrected = pose
                    baseline_metrics = foot_locking_metrics(
                        pose["joints_3d"],
                        [str(name) for name in pose.get("joint_names", [])],
                        contacts,
                        fps=float(pose.get("fps") or 30.0),
                        contact_threshold=0.85,
                        min_segment_frames=3,
                        foot_keys=["left_toe", "right_toe"],
                    )
                    lock_report: dict[str, Any] = {
                        "status": "baseline",
                        "metrics_before": baseline_metrics,
                        "metrics_after": baseline_metrics,
                        "mocap_used_in_objective": False,
                    }
                    if variant != "baseline":
                        lock_cfg = _variant_cfg(variant, run_config)
                        corrected, lock_report = apply_contact_foot_locking_to_pose(pose, contacts, lock_cfg)
                        if lock_report.get("status") != "ok":
                            row = failure_row(
                                backend,
                                trial,
                                run_dir,
                                "skipped",
                                lock_report.get("reason", "Contact foot locking skipped."),
                            )
                            row.update(_lock_row_fields(variant, lock_report))
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
                    row.update(_lock_row_fields(variant, lock_report))
                    rows.append(row)
                except Exception as exc:
                    row = failure_row(backend, trial, run_dir, "comparison_failed", str(exc))
                    row["variant"] = variant
                    rows.append(row)

    outputs = write_outputs(out_dir, rows, args, manifest)
    _print_summary(outputs, rows)
    return 0 if any(row.get("status") == "valid" for row in rows) else 1


def write_outputs(out_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, str]:
    csv_path = out_dir / "contact_foot_locking_ablation.csv"
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
        "best_foot_lock_by_backend_trial": _best_foot_lock_by_backend_trial(rows),
        "rows": rows,
    }
    json_path = out_dir / "contact_foot_locking_ablation.json"
    write_json(json_path, summary)
    md_path = out_dir / "contact_foot_locking_ablation.md"
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


def _variant_cfg(variant: str, run_config: dict[str, Any] | None = None) -> dict[str, Any]:
    base = (
        (((run_config or {}).get("config") or {}).get("optimization") or {}).get("contact_foot_locking") or {}
    )
    cfg = dict(base)
    cfg["enabled"] = True
    name = variant.strip().lower()
    if name == "baseline":
        cfg["enabled"] = False
        return cfg
    if name == "root_toe":
        cfg.update({"mode": "root_translation", "feet": "toes", "lock_vertical": False, "smooth_correction_window_frames": 1})
    elif name == "root_toe_smooth":
        cfg.update({"mode": "root_translation", "feet": "toes", "lock_vertical": False, "smooth_correction_window_frames": 5})
    elif name == "root_all_smooth":
        cfg.update({"mode": "root_translation", "feet": "toes_and_heels", "lock_vertical": False, "smooth_correction_window_frames": 5})
    elif name == "root_toe_floor":
        cfg.update({"mode": "root_translation", "feet": "toes", "lock_vertical": True, "smooth_correction_window_frames": 5})
    elif name == "endpoint_toe":
        cfg.update({"mode": "endpoint", "feet": "toes", "lock_vertical": False, "smooth_correction_window_frames": 1})
    else:
        raise ValueError(f"Unsupported contact foot-locking variant: {variant}")
    cfg.setdefault("contact_threshold", 0.85)
    cfg.setdefault("min_segment_frames", 3)
    cfg.setdefault("blend", 1.0)
    cfg.setdefault("max_correction_m", 0.20)
    return cfg


def _lock_row_fields(variant: str, lock_report: dict[str, Any]) -> dict[str, Any]:
    before = lock_report.get("metrics_before") or {}
    after = lock_report.get("metrics_after") or {}
    correction = lock_report.get("correction_summary") or {}
    return {
        "variant": variant,
        "mode": lock_report.get("mode"),
        "feet": lock_report.get("feet"),
        "lock_vertical": lock_report.get("lock_vertical"),
        "locked_segment_count": lock_report.get("locked_segment_count"),
        "mean_contact_horizontal_speed_before_mps": before.get("mean_contact_horizontal_speed_mps"),
        "mean_contact_horizontal_speed_after_mps": after.get("mean_contact_horizontal_speed_mps"),
        "mean_contact_position_std_before_m": before.get("mean_contact_horizontal_position_std_m"),
        "mean_contact_position_std_after_m": after.get("mean_contact_horizontal_position_std_m"),
        "max_correction_m": correction.get("max_m"),
        "active_correction_frames": correction.get("active_frames"),
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


def _best_foot_lock_by_backend_trial(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        improvement = None
        if baseline and baseline.get("mean_contact_horizontal_speed_after_mps") is not None:
            improvement = float(baseline["mean_contact_horizontal_speed_after_mps"]) - float(best["mean_contact_horizontal_speed_after_mps"])
        out.append(
            {
                "backend": backend,
                "trial": trial,
                "best_variant": best.get("variant"),
                "baseline_contact_speed_mps": baseline.get("mean_contact_horizontal_speed_after_mps") if baseline else None,
                "best_contact_speed_mps": best.get("mean_contact_horizontal_speed_after_mps"),
                "contact_speed_reduction_mps": improvement,
            }
        )
    return out


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    rows = summary.get("rows") or []
    with path.open("w", encoding="utf-8") as f:
        f.write("# monocap_v2 Contact Foot-Locking Ablation\n\n")
        f.write("- Contact locking uses predicted joints and heuristic/backend contact probabilities only.\n")
        f.write("- Mocap/OpenSim FK is reporting-only and is not used to choose contact windows or anchors.\n\n")
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
        for item in summary.get("best_foot_lock_by_backend_trial") or []:
            f.write(
                f"| {item.get('backend')} | {item.get('trial')} | {item.get('best_variant')} | "
                f"{_fmt(item.get('baseline_contact_speed_mps'))} | {_fmt(item.get('best_contact_speed_mps'))} | "
                f"{_fmt(item.get('contact_speed_reduction_mps'))} |\n"
            )
        f.write("\n## Rows\n\n")
        f.write("| Backend | Trial | Variant | Status | Primary | Rigid | PA | Contact Speed Before | Contact Speed After | Max Correction | Error |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row.get('backend')} | {row.get('trial')} | {row.get('variant')} | {row.get('status')} | "
                f"{_fmt(row.get('primary_root_centered_mpjpe_mm'))} | {_fmt(row.get('root_centered_rigid_mpjpe_mm'))} | "
                f"{_fmt(row.get('pa_mpjpe_mm'))} | {_fmt(row.get('mean_contact_horizontal_speed_before_mps'))} | "
                f"{_fmt(row.get('mean_contact_horizontal_speed_after_mps'))} | {_fmt(row.get('max_correction_m'))} | "
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
    print(f"[contact-foot-locking] Valid rows: {valid}/{len(rows)}", flush=True)
    print(f"[contact-foot-locking] Summary: {outputs['markdown']}", flush=True)
    print(f"[contact-foot-locking] CSV: {outputs['csv']}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
