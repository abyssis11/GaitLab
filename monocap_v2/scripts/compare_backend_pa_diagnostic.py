#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monocap_v2.core.level_a_benchmark import (
    compare_pose_to_opensim_reference,
    load_cached_wham_timeline,
    load_opensim_reference,
    load_pose_artifact,
    load_run_config,
)


DEFAULT_BACKEND_SETTINGS = {
    "wham": {"axis": "x,-y,z", "time_offset_s": -0.12},
    "metrabs": {"axis": "x,-y,z", "time_offset_s": -0.26},
    "rtmw3d": {"axis": "x,-y,-z", "time_offset_s": -0.26},
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare cached Level A camera/backend runs with explicit diagnostic PA-style settings."
    )
    ap.add_argument("--cam1-benchmark-dir", type=Path, default=Path("monocap_v2/benchmarks/subject7_walking_level_a"))
    ap.add_argument("--cam0-benchmark-dir", type=Path, default=Path("monocap_v2/benchmarks/subject7_walking_level_a_cam0"))
    ap.add_argument("--trials", default="walking1,walking2,walking3")
    ap.add_argument("--backends", default="wham,rtmw3d,metrabs")
    ap.add_argument("--evaluation-hz", default="60,100")
    ap.add_argument("--out-prefix", type=Path, default=Path("monocap_v2/benchmarks/cam0_cam1_backend_pa_diagnostic"))
    ap.add_argument("--no-swap-reference-lr", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    trials = _csv(args.trials)
    backends = _csv(args.backends)
    hz_values = [float(item) for item in _csv(args.evaluation_hz)]
    cameras = [("Cam1", args.cam1_benchmark_dir), ("Cam0", args.cam0_benchmark_dir)]
    rows = []
    for camera, benchmark_dir in cameras:
        rows.extend(
            _evaluate_camera(
                camera=camera,
                benchmark_dir=benchmark_dir,
                trials=trials,
                backends=backends,
                hz_values=hz_values,
                swap_reference_lr=not args.no_swap_reference_lr,
            )
        )

    out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    json_path = out_prefix.with_suffix(".json")
    md_path = out_prefix.with_suffix(".md")
    summary = _summary(rows, cameras=[camera for camera, _ in cameras], backends=backends, hz_values=hz_values)
    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    _write_md(md_path, rows, summary)
    print(f"[diagnostic-compare] csv -> {csv_path}", flush=True)
    print(f"[diagnostic-compare] md -> {md_path}", flush=True)
    _print_summary(summary)
    return 0


def _evaluate_camera(
    camera: str,
    benchmark_dir: Path,
    trials: list[str],
    backends: list[str],
    hz_values: list[float],
    swap_reference_lr: bool,
) -> list[dict[str, Any]]:
    summary_path = benchmark_dir / "level_a_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row_lookup = {(row.get("backend"), row.get("trial")): row for row in summary.get("rows", [])}
    rows: list[dict[str, Any]] = []
    for trial in trials:
        reference = load_opensim_reference(
            benchmark_dir / "reference" / f"opensim_fk_{trial}.npz",
            benchmark_dir / "reference" / f"opensim_fk_{trial}.json",
        )
        for backend in backends:
            setting = DEFAULT_BACKEND_SETTINGS[backend]
            source_row = row_lookup.get((backend, trial))
            if not source_row or source_row.get("status") != "valid":
                rows.append(_failed_row(camera, trial, backend, None, "missing_valid_row", "No valid Level A source row."))
                continue
            run_dir = Path(str(source_row["run_dir"]))
            try:
                pose = load_pose_artifact(run_dir)
                run_config = load_run_config(run_dir)
                timeline = load_cached_wham_timeline(run_dir)
            except Exception as exc:
                rows.append(_failed_row(camera, trial, backend, None, "artifact_failed", str(exc)))
                continue
            for hz in hz_values:
                print(f"[diagnostic-compare] {camera} {trial} {backend} {float(hz):g}Hz", flush=True)
                try:
                    report, _series = compare_pose_to_opensim_reference(
                        pose,
                        reference,
                        run_config=run_config,
                        axis_map={backend: str(setting["axis"])},
                        timeline_report=timeline,
                        evaluation_hz=float(hz),
                        diagnostic_time_offset_s=float(setting["time_offset_s"]),
                        swap_reference_lr=swap_reference_lr,
                    )
                    rows.append(
                        {
                            "camera": camera,
                            "trial": trial,
                            "backend": backend,
                            "evaluation_hz": float(hz),
                            "status": "valid",
                            "axis_mode": setting["axis"],
                            "time_offset_s": float(setting["time_offset_s"]),
                            "gt_left_right_swap": bool(swap_reference_lr),
                            "model_left_right_swap": False,
                            "raw_primary_mm": report.get("primary_root_centered_mpjpe_mm"),
                            "rigid_mm": report.get("root_centered_rigid_mpjpe_mm"),
                            "pa_mm": report.get("pa_mpjpe_mm"),
                            "normal_minus_rigid_gap_mm": report.get("normal_minus_rigid_gap_mm"),
                            "frames": report.get("frames"),
                            "overlap_frames": report.get("overlap_frames"),
                            "time_start_s": report.get("time_start_s"),
                            "time_end_s": report.get("time_end_s"),
                            "warnings": "; ".join(report.get("warnings") or []),
                            "error": "",
                        }
                    )
                except Exception as exc:
                    rows.append(_failed_row(camera, trial, backend, hz, "failed", str(exc), setting, swap_reference_lr))
    return rows


def _failed_row(
    camera: str,
    trial: str,
    backend: str,
    hz: float | None,
    status: str,
    error: str,
    setting: dict[str, Any] | None = None,
    swap_reference_lr: bool = True,
) -> dict[str, Any]:
    setting = setting or DEFAULT_BACKEND_SETTINGS.get(backend, {})
    return {
        "camera": camera,
        "trial": trial,
        "backend": backend,
        "evaluation_hz": hz,
        "status": status,
        "axis_mode": setting.get("axis"),
        "time_offset_s": setting.get("time_offset_s"),
        "gt_left_right_swap": bool(swap_reference_lr),
        "model_left_right_swap": False,
        "error": error,
    }


def _summary(rows: list[dict[str, Any]], cameras: list[str], backends: list[str], hz_values: list[float]) -> list[dict[str, Any]]:
    out = []
    for camera in cameras:
        for backend in backends:
            for hz in hz_values:
                selected = [
                    row
                    for row in rows
                    if row.get("status") == "valid"
                    and row.get("camera") == camera
                    and row.get("backend") == backend
                    and float(row.get("evaluation_hz")) == float(hz)
                ]
                if not selected:
                    continue
                out.append(
                    {
                        "camera": camera,
                        "backend": backend,
                        "evaluation_hz": float(hz),
                        "valid_trials": len(selected),
                        "median_raw_primary_mm": _median(selected, "raw_primary_mm"),
                        "median_rigid_mm": _median(selected, "rigid_mm"),
                        "median_pa_mm": _median(selected, "pa_mm"),
                    }
                )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "camera",
        "trial",
        "backend",
        "evaluation_hz",
        "status",
        "axis_mode",
        "time_offset_s",
        "gt_left_right_swap",
        "model_left_right_swap",
        "raw_primary_mm",
        "rigid_mm",
        "pa_mm",
        "normal_minus_rigid_gap_mm",
        "frames",
        "overlap_frames",
        "time_start_s",
        "time_end_s",
        "warnings",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    valid_trials = sorted({str(row.get("trial")) for row in rows if row.get("status") == "valid"})
    lines = [
        "# Cam0 vs Cam1 Backend PA-Style Diagnostic",
        "",
        "- Model artifacts are unchanged.",
        "- OpenSim FK reference L/R is relabeled for this diagnostic.",
        "- Backend settings match the earlier Cam1 `backend_pa_compare` convention style.",
        "- WHAM: `x,-y,z`, offset `-0.12s`.",
        "- MeTRAbs: `x,-y,z`, offset `-0.26s`.",
        "- RTMW3D: `x,-y,-z`, offset `-0.26s`.",
        "",
        "## Median Walking1-3",
        "",
        "| Camera | Hz | Backend | Raw/root | Rigid | PA |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['camera']} | {_fmt(row['evaluation_hz'])} | {row['backend']} | "
            f"{_fmt(row['median_raw_primary_mm'])} | {_fmt(row['median_rigid_mm'])} | {_fmt(row['median_pa_mm'])} |"
        )
    lines.extend(
        [
            "",
        ]
    )
    for trial in valid_trials:
        trial_rows = [row for row in rows if row.get("status") == "valid" and row.get("trial") == trial]
        lines.extend(
            [
                "",
                f"## {trial}",
                "",
                "| Camera | Hz | Backend | Raw/root | Rigid | PA |",
                "|---|---:|---|---:|---:|---:|",
            ]
        )
        for row in trial_rows:
            lines.append(
                f"| {row['camera']} | {_fmt(row['evaluation_hz'])} | {row['backend']} | "
                f"{_fmt(row.get('raw_primary_mm'))} | {_fmt(row.get('rigid_mm'))} | {_fmt(row.get('pa_mm'))} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(summary: list[dict[str, Any]]) -> None:
    print("| Camera | Hz | Backend | Raw/root | Rigid | PA |", flush=True)
    print("|---|---:|---|---:|---:|---:|", flush=True)
    for row in summary:
        print(
            f"| {row['camera']} | {_fmt(row['evaluation_hz'])} | {row['backend']} | "
            f"{_fmt(row['median_raw_primary_mm'])} | {_fmt(row['median_rigid_mm'])} | {_fmt(row['median_pa_mm'])} |",
            flush=True,
        )


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
    return float(statistics.median(values)) if values else None


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.2f}"


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
