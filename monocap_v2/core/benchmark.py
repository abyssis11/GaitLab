from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

from monocap_v2.core.logging_utils import read_json, write_json


REGRESSION_THRESHOLD_MM = 2.0

CSV_FIELDS = [
    "trial",
    "run_dir",
    "report_source",
    "status",
    "stage_status",
    "pose3d_backend",
    "camera_source",
    "camera_mode",
    "camera_is_assumed",
    "camera_distortion_enabled",
    "initial_primary_mpjpe_mm",
    "refined_primary_mpjpe_mm",
    "primary_improvement_mm",
    "initial_normal_root_centered_mpjpe_mm",
    "refined_normal_root_centered_mpjpe_mm",
    "initial_root_centered_rigid_mpjpe_mm",
    "refined_root_centered_rigid_mpjpe_mm",
    "initial_pa_mpjpe_mm",
    "refined_pa_mpjpe_mm",
    "initial_sequence_similarity_mpjpe_mm",
    "refined_sequence_similarity_mpjpe_mm",
    "initial_pa_similarity_mpjpe_mm",
    "refined_pa_similarity_mpjpe_mm",
    "foot_speed_contact_before_mps",
    "foot_speed_contact_after_mps",
    "warnings",
    "error",
]


def parse_stage07_report(run_dir: Path, trial: str) -> dict[str, Any]:
    camera_info = _read_camera_info(run_dir)
    report_path = run_dir / "optimization" / "opt_stage2_report.json"
    if not report_path.exists():
        return {
            "trial": trial,
            "run_dir": str(run_dir),
            "report_source": "stage07",
            "status": "missing_report",
            "stage_status": None,
            **camera_info,
            "error": f"Missing report: {report_path}",
        }
    report = read_json(report_path)
    mocap = report.get("mocap_evaluation") or {}
    if mocap.get("status") != "ok":
        return {
            "trial": trial,
            "run_dir": str(run_dir),
            "report_source": "stage07",
            "status": "missing_mocap",
            "stage_status": report.get("status"),
            **camera_info,
            "warnings": _join(report.get("warnings", [])),
            "error": mocap.get("error") or mocap.get("reason") or "Mocap evaluation missing or not ok.",
        }
    initial = mocap.get("initial") or {}
    refined = mocap.get("refined") or {}
    before_metrics = report.get("metrics_before") or {}
    after_metrics = report.get("metrics_after") or {}
    return {
        "trial": trial,
        "run_dir": str(run_dir),
        "report_source": "stage07",
        "status": "valid",
        "stage_status": report.get("status"),
        **camera_info,
        "initial_primary_mpjpe_mm": _num(initial.get("primary_root_centered_rigid_mpjpe_mm")),
        "refined_primary_mpjpe_mm": _num(refined.get("primary_root_centered_rigid_mpjpe_mm")),
        "primary_improvement_mm": _num(mocap.get("primary_improvement_mm")),
        "initial_sequence_similarity_mpjpe_mm": _num(initial.get("sequence_similarity_mpjpe_mm")),
        "refined_sequence_similarity_mpjpe_mm": _num(refined.get("sequence_similarity_mpjpe_mm")),
        "initial_pa_similarity_mpjpe_mm": _num(initial.get("pa_similarity_mpjpe_mm")),
        "refined_pa_similarity_mpjpe_mm": _num(refined.get("pa_similarity_mpjpe_mm")),
        "foot_speed_contact_before_mps": _num(before_metrics.get("mean_foot_speed_during_contact_mps")),
        "foot_speed_contact_after_mps": _num(after_metrics.get("mean_foot_speed_during_contact_mps")),
        "warnings": _join(report.get("warnings", [])),
        "error": None,
    }


def parse_mocap_validation_report(run_dir: Path, trial: str, expected_backend: str | None = None) -> dict[str, Any]:
    camera_info = _read_camera_info(run_dir)
    report_path = run_dir / "reports" / "mocap_validation.json"
    if not report_path.exists():
        return {
            "trial": trial,
            "run_dir": str(run_dir),
            "report_source": "mocap_validation",
            "status": "missing_report",
            "stage_status": None,
            **camera_info,
            "error": f"Missing report: {report_path}",
        }
    report = read_json(report_path)
    backend = report.get("backend")
    if expected_backend and backend != expected_backend:
        return {
            "trial": trial,
            "run_dir": str(run_dir),
            "report_source": "mocap_validation",
            "status": "backend_mismatch",
            "stage_status": report.get("status"),
            "pose3d_backend": backend,
            **camera_info,
            "error": f"Expected pose3d backend {expected_backend!r}, found {backend!r}.",
        }
    initial = report.get("initial") or {}
    refined = report.get("refined") or {}
    if report.get("status") not in {"ok", "warning"} or refined.get("normal_root_centered_mpjpe_mm") is None:
        return {
            "trial": trial,
            "run_dir": str(run_dir),
            "report_source": "mocap_validation",
            "status": "missing_mocap",
            "stage_status": report.get("status"),
            **camera_info,
            "warnings": _join(report.get("warnings", [])),
            "error": report.get("reason") or report.get("error") or "Mocap validation metrics are unavailable.",
        }
    initial_normal = _num(initial.get("normal_root_centered_mpjpe_mm"))
    refined_normal = _num(refined.get("normal_root_centered_mpjpe_mm"))
    return {
        "trial": trial,
        "run_dir": str(run_dir),
        "report_source": "mocap_validation",
        "status": "valid",
        "stage_status": report.get("status"),
        "pose3d_backend": backend,
        **camera_info,
        "initial_primary_mpjpe_mm": initial_normal,
        "refined_primary_mpjpe_mm": refined_normal,
        "primary_improvement_mm": initial_normal - refined_normal if initial_normal is not None and refined_normal is not None else None,
        "initial_normal_root_centered_mpjpe_mm": initial_normal,
        "refined_normal_root_centered_mpjpe_mm": refined_normal,
        "initial_root_centered_rigid_mpjpe_mm": _num(initial.get("root_centered_rigid_mpjpe_mm")),
        "refined_root_centered_rigid_mpjpe_mm": _num(refined.get("root_centered_rigid_mpjpe_mm")),
        "initial_pa_mpjpe_mm": _num(initial.get("pa_mpjpe_mm")),
        "refined_pa_mpjpe_mm": _num(refined.get("pa_mpjpe_mm")),
        "initial_sequence_similarity_mpjpe_mm": _num(initial.get("global_sequence_similarity_mpjpe_mm")),
        "refined_sequence_similarity_mpjpe_mm": _num(refined.get("global_sequence_similarity_mpjpe_mm")),
        "initial_pa_similarity_mpjpe_mm": _num(initial.get("pa_mpjpe_mm")),
        "refined_pa_similarity_mpjpe_mm": _num(refined.get("pa_mpjpe_mm")),
        "foot_speed_contact_before_mps": None,
        "foot_speed_contact_after_mps": None,
        "warnings": _join(report.get("warnings", [])),
        "error": None,
    }


def parse_benchmark_report(
    run_dir: Path,
    trial: str,
    report_source: str = "stage07",
    expected_backend: str | None = None,
) -> dict[str, Any]:
    if report_source == "stage07":
        return parse_stage07_report(run_dir, trial)
    if report_source == "mocap_validation":
        return parse_mocap_validation_report(run_dir, trial, expected_backend=expected_backend)
    raise ValueError(f"Unknown benchmark report source: {report_source}")


def aggregate_trials(
    rows: list[dict[str, Any]],
    regression_threshold_mm: float = REGRESSION_THRESHOLD_MM,
    descriptive: bool = False,
) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "valid" and row.get("initial_primary_mpjpe_mm") is not None and row.get("refined_primary_mpjpe_mm") is not None]
    failures = [row for row in rows if row.get("status") != "valid"]
    if not valid:
        return {
            "status": "failed",
            "valid_trial_count": 0,
            "failure_count": len(failures),
            "failures": failures,
            "reason": "No valid trial reports were available.",
        }
    initial = [float(row["initial_primary_mpjpe_mm"]) for row in valid]
    refined = [float(row["refined_primary_mpjpe_mm"]) for row in valid]
    improvements = [float(row.get("primary_improvement_mm") or (i - r)) for row, i, r in zip(valid, initial, refined)]
    regressions = [row for row in valid if float(row["refined_primary_mpjpe_mm"]) - float(row["initial_primary_mpjpe_mm"]) > regression_threshold_mm]
    median_initial = float(statistics.median(initial))
    median_refined = float(statistics.median(refined))
    median_improvement = median_initial - median_refined
    if descriptive:
        status = "ok"
    elif median_refined > median_initial:
        status = "failed"
    elif regressions:
        status = "warning"
    else:
        status = "ok"
    return {
        "status": status,
        "aggregation_mode": "descriptive" if descriptive else "refinement_comparison",
        "valid_trial_count": len(valid),
        "failure_count": len(failures),
        "median_initial_primary_mpjpe_mm": median_initial,
        "median_refined_primary_mpjpe_mm": median_refined,
        "median_primary_improvement_mm": float(median_improvement),
        "mean_primary_improvement_mm": float(statistics.fmean(improvements)),
        "regression_threshold_mm": float(regression_threshold_mm),
        "large_regression_trials": [row["trial"] for row in regressions],
        "failures": failures,
    }


def write_benchmark_outputs(out_dir: Path, rows: list[dict[str, Any]], aggregate: dict[str, Any], metadata: dict[str, Any]) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_trial_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})

    summary = {"metadata": metadata, "aggregate": aggregate, "trials": rows}
    json_path = out_dir / "benchmark_summary.json"
    write_json(json_path, summary)

    md_path = out_dir / "benchmark_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# monocap_v2 Walking Benchmark\n\n")
        f.write(f"- Status: `{aggregate.get('status')}`\n")
        f.write(f"- Report source: `{metadata.get('report_source', 'stage07')}`\n")
        f.write(f"- Aggregation mode: `{aggregate.get('aggregation_mode', 'refinement_comparison')}`\n")
        f.write(f"- Valid trials: `{aggregate.get('valid_trial_count')}`\n")
        f.write(f"- Failures/missing reports: `{aggregate.get('failure_count')}`\n")
        if aggregate.get("median_initial_primary_mpjpe_mm") is not None:
            f.write(f"- Median initial MPJPE: `{aggregate['median_initial_primary_mpjpe_mm']:.3f} mm`\n")
            f.write(f"- Median refined MPJPE: `{aggregate['median_refined_primary_mpjpe_mm']:.3f} mm`\n")
            f.write(f"- Median improvement: `{aggregate['median_primary_improvement_mm']:.3f} mm`\n")
        if aggregate.get("large_regression_trials"):
            f.write(f"- Large regressions: `{', '.join(aggregate['large_regression_trials'])}`\n")
        f.write("\n| Trial | Status | Camera | Initial MPJPE (mm) | Refined MPJPE (mm) | Improvement (mm) | Foot Speed Before/After (m/s) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row.get('trial')} | {row.get('status')} | {row.get('camera_source') or ''} | {_fmt(row.get('initial_primary_mpjpe_mm'))} | "
                f"{_fmt(row.get('refined_primary_mpjpe_mm'))} | {_fmt(row.get('primary_improvement_mm'))} | "
                f"{_fmt(row.get('foot_speed_contact_before_mps'))} / {_fmt(row.get('foot_speed_contact_after_mps'))} |\n"
            )
    return {"summary_json": str(json_path), "summary_md": str(md_path), "per_trial_csv": str(csv_path)}


def _num(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _join(values) -> str:
    if not values:
        return ""
    if isinstance(values, str):
        return values
    return "; ".join(str(value) for value in values)


def _read_camera_info(run_dir: Path) -> dict[str, Any]:
    qc_path = run_dir / "input" / "camera_qc.json"
    camera_path = run_dir / "input" / "camera_assumed.json"
    qc = read_json(qc_path) if qc_path.exists() else {}
    camera = read_json(camera_path) if camera_path.exists() else {}
    distortion = camera.get("distortion") if isinstance(camera, dict) else {}
    return {
        "camera_source": qc.get("camera_source") or camera.get("mode"),
        "camera_mode": qc.get("mode") or camera.get("mode"),
        "camera_is_assumed": qc.get("is_assumed") if "is_assumed" in qc else camera.get("is_assumed"),
        "camera_distortion_enabled": qc.get("distortion_enabled")
        if "distortion_enabled" in qc
        else bool((distortion or {}).get("enabled", False)),
    }


def _fmt(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"
