#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monocap_v2.core.level_a_benchmark import (
    aggregate_level_a,
    backend_run_name,
    compare_pose_to_opensim_reference,
    failure_row,
    load_cached_wham_timeline,
    load_opensim_reference,
    load_pose_artifact,
    load_run_config,
    resolve_mocap_opensim_paths,
    row_from_report,
    write_level_a_outputs,
)
from monocap_v2.core.manifest import find_trial, load_opencap_manifest
from monocap_v2.core.mocap_eval import parse_trc


DEFAULT_TRIALS = "walking1,walking2,walking3"
DEFAULT_BACKENDS = "metrabs,rtmw3d,wham"
DEFAULT_OPENSIM_PYTHON = "/home/denik/miniconda3/envs/gaitlab/bin/python"
DEFAULT_WHAM_PYTHON = "/home/denik/miniconda3/envs/monocap-wham/bin/python"
PIPELINE_STAGES = "stage_00_validate_inputs,stage_01_preprocess_video,stage_02_assume_camera,stage_03_pose2d,stage_04_pose3d_initial"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare raw monocap_v2 backend initial poses against OpenSim FK mocap reference.")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--paths", required=True, type=Path)
    ap.add_argument("--trials", default=DEFAULT_TRIALS, help="Comma-separated trial IDs.")
    ap.add_argument("--activity", default="walking")
    ap.add_argument("--backends", default=DEFAULT_BACKENDS, help="Comma-separated backend IDs: metrabs,rtmw3d,wham.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-inference", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--evaluation-hz", type=float, default=None, help="Optional uniform evaluation rate; e.g. 100 resamples prediction and reference to 100 Hz before scoring.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--opensim-python", type=Path, default=Path(DEFAULT_OPENSIM_PYTHON))
    ap.add_argument("--wham-python", type=Path, default=Path(DEFAULT_WHAM_PYTHON), help="Python executable for WHAM pipeline subprocesses.")
    ap.add_argument("--repo-root", type=Path, default=None, help=argparse.SUPPRESS)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root or Path(__file__).resolve().parents[2]
    manifest = load_opencap_manifest(args.manifest, args.paths)
    trials = _csv_arg(args.trials)
    backends = _csv_arg(args.backends)
    out_dir = args.out or (repo_root / "monocap_v2" / "benchmarks" / f"{manifest.get('subject_id', 'subject')}_{args.activity}_level_a")
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    reference_dir = out_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    pipeline_failures: list[dict[str, Any]] = []
    reference_failures: list[dict[str, Any]] = []
    references: dict[str, dict[str, Any]] = {}
    raw_trc_sanity: dict[str, Any] = {}

    for trial in trials:
        raw_trc_sanity[trial] = _raw_trc_sanity(manifest, trial)
        try:
            references[trial] = _ensure_reference(repo_root, args, manifest, trial, reference_dir)
        except Exception as exc:
            reference_failures.append({"trial": trial, "error": str(exc)})
            for backend in backends:
                run_dir = repo_root / "monocap_v2" / "runs" / backend_run_name(manifest, trial, backend)
                rows.append(failure_row(backend, trial, run_dir, "reference_failed", str(exc)))
            continue

        for backend in backends:
            run_dir = repo_root / "monocap_v2" / "runs" / backend_run_name(manifest, trial, backend)
            if not args.skip_inference:
                rc = _run_pipeline(repo_root, args, backend, trial, run_dir)
                if rc != 0:
                    error = f"Pipeline exited with return code {rc}."
                    pipeline_failures.append({"backend": backend, "trial": trial, "run_dir": str(run_dir), "returncode": rc})
                    rows.append(failure_row(backend, trial, run_dir, "pipeline_failed", error))
                    continue
            try:
                pose = load_pose_artifact(run_dir)
                reference = load_opensim_reference(references[trial]["npz"], references[trial]["json"])
                run_config = load_run_config(run_dir)
                timeline = load_cached_wham_timeline(run_dir) if pose.get("backend") == "wham" else None
                report, _series = compare_pose_to_opensim_reference(
                    pose,
                    reference,
                    run_config=run_config,
                    timeline_report=timeline,
                    evaluation_hz=args.evaluation_hz,
                )
                rows.append(row_from_report(backend, trial, run_dir, report, reference.get("metadata")))
            except Exception as exc:
                rows.append(failure_row(backend, trial, run_dir, "comparison_failed", str(exc)))

    aggregate = aggregate_level_a(rows)
    metadata = {
        "manifest": str(args.manifest),
        "paths": str(args.paths),
        "trials": trials,
        "backends": backends,
        "activity": args.activity,
        "skip_inference": bool(args.skip_inference),
        "force": bool(args.force),
        "max_frames": args.max_frames,
        "evaluation_hz": args.evaluation_hz,
        "opensim_python": str(args.opensim_python),
        "wham_python": str(args.wham_python),
        "pipeline_stages": PIPELINE_STAGES.split(","),
        "reference_failures": reference_failures,
        "pipeline_failures": pipeline_failures,
        "raw_trc_sanity": raw_trc_sanity,
    }
    outputs = write_level_a_outputs(out_dir, rows, aggregate, metadata)
    _print_summary(aggregate, outputs)
    return 0 if aggregate.get("status") == "ok" else 1


def _ensure_reference(repo_root: Path, args: argparse.Namespace, manifest: dict[str, Any], trial: str, reference_dir: Path) -> dict[str, Path]:
    paths = resolve_mocap_opensim_paths(manifest, trial)
    npz_path = reference_dir / f"opensim_fk_{trial}.npz"
    json_path = reference_dir / f"opensim_fk_{trial}.json"
    if npz_path.exists() and json_path.exists() and not args.force:
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
    print(f"[level-a] Exporting OpenSim FK reference for {trial}: {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, cwd=str(repo_root))
    if completed.returncode != 0:
        raise RuntimeError(f"OpenSim FK export failed for {trial} with return code {completed.returncode}.")
    if not npz_path.exists() or not json_path.exists():
        raise RuntimeError(f"OpenSim FK export did not create expected files for {trial}.")
    return {"npz": npz_path, "json": json_path, **paths}


def _run_pipeline(repo_root: Path, args: argparse.Namespace, backend: str, trial: str, run_dir: Path) -> int:
    cmd = [
        str(_pipeline_python(args, backend)),
        str(repo_root / "monocap_v2" / "run_pipeline.py"),
        "--manifest",
        str(args.manifest),
        "--paths",
        str(args.paths),
        "--trial",
        trial,
        "--activity",
        args.activity,
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
    if args.force:
        cmd.append("--force")
    if args.max_frames is not None:
        cmd.extend(["--max-frames", str(int(args.max_frames))])
    print(f"[level-a] Running {backend} {trial}: {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, cwd=str(repo_root))
    return int(completed.returncode)


def _raw_trc_sanity(manifest: dict[str, Any], trial_id: str) -> dict[str, Any]:
    try:
        _subset, trial = find_trial(manifest, trial_id)
        trc_path = Path(str(trial.get("mocap_trc") or ""))
        if not trc_path.exists():
            return {"status": "missing", "source": str(trc_path)}
        trc = parse_trc(trc_path)
        values = np.stack(list(trc.markers.values()), axis=1) if trc.markers else np.empty((0, 0, 3))
        return {
            "status": "ok",
            "source": str(trc_path),
            "frames": int(trc.time.shape[0]),
            "marker_count": len(trc.marker_names),
            "data_rate_hz": trc.data_rate,
            "units": trc.units,
            "finite_ratio": float(np.isfinite(values).mean()) if values.size else 0.0,
        }
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


def _backend_preset(backend: str) -> str | None:
    if backend == "rtmw3d":
        return "opencap_rtmw3d"
    if backend == "wham":
        return "opencap_wham"
    return None


def _pipeline_python(args: argparse.Namespace, backend: str) -> Path:
    if backend == "wham":
        return Path(args.wham_python)
    return Path(sys.executable)


def _csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _print_summary(aggregate: dict[str, Any], outputs: dict[str, str]) -> None:
    print("[level-a] Status:", aggregate.get("status"), flush=True)
    for item in aggregate.get("ranking") or []:
        print(
            "[level-a] Rank",
            item.get("rank"),
            item.get("backend"),
            f"median={float(item.get('median_mm')):.3f} mm",
            flush=True,
        )
    print("[level-a] Summary:", outputs["summary_md"], flush=True)
    print("[level-a] CSV:", outputs["per_backend_trial_csv"], flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
