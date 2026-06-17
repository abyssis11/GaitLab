#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monocap_v2.core.level_a_benchmark import (
    aggregate_level_a,
    compare_pose_to_opensim_reference,
    failure_row,
    load_cached_wham_timeline,
    load_opensim_reference,
    load_run_config,
    resolve_mocap_opensim_paths,
    row_from_report,
    write_level_a_outputs,
)
from monocap_v2.core.logging_utils import read_yaml
from monocap_v2.core.manifest import load_opencap_manifest


DEFAULT_OPENSIM_PYTHON = "/home/denik/miniconda3/envs/gaitlab/bin/python"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Level A comparison for one existing monocap_v2 run.")
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--pose-source", choices=["initial", "refined", "both"], default="refined")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--paths", type=Path, default=None)
    ap.add_argument("--trial", default=None)
    ap.add_argument("--evaluation-hz", type=float, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force-reference", action="store_true")
    ap.add_argument("--opensim-python", type=Path, default=Path(DEFAULT_OPENSIM_PYTHON))
    ap.add_argument("--repo-root", type=Path, default=None, help=argparse.SUPPRESS)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root or Path(__file__).resolve().parents[2]
    run_dir = args.run if args.run.is_absolute() else repo_root / args.run
    run_config = load_run_config(run_dir)
    manifest_path = args.manifest or Path(str(run_config.get("manifest_path") or ""))
    paths_path = args.paths or Path(str(run_config.get("paths_path") or ""))
    trial = args.trial or str(run_config.get("trial_id") or "")
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest path: {manifest_path}")
    if not paths_path.exists():
        raise SystemExit(f"Missing paths path: {paths_path}")
    if not trial:
        raise SystemExit("Trial id is unavailable; pass --trial.")

    manifest = load_opencap_manifest(manifest_path, paths_path)
    out_dir = args.out or (run_dir / "reports" / f"level_a_{args.pose_source}")
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    reference_dir = out_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    try:
        reference_paths = _ensure_reference(repo_root, args, manifest, trial, reference_dir)
        reference = load_opensim_reference(reference_paths["npz"], reference_paths["json"])
    except Exception as exc:
        for source in _pose_sources(args.pose_source):
            rows.append(failure_row(_backend_label(run_dir, source, None), trial, run_dir, "reference_failed", str(exc)))
    else:
        timeline = load_cached_wham_timeline(run_dir)
        for source in _pose_sources(args.pose_source):
            try:
                pose = _load_pose(run_dir, source)
                backend = _backend_label(run_dir, source, pose)
                report, _series = compare_pose_to_opensim_reference(
                    pose,
                    reference,
                    run_config=run_config,
                    timeline_report=timeline if pose.get("backend") == "wham" else None,
                    evaluation_hz=args.evaluation_hz,
                )
                row = row_from_report(backend, trial, run_dir, report, reference.get("metadata"))
                row["warnings"] = _join_warnings(row.get("warnings"), [f"pose_source={source}"])
                rows.append(row)
            except Exception as exc:
                rows.append(failure_row(_backend_label(run_dir, source, None), trial, run_dir, "comparison_failed", str(exc)))

    aggregate = aggregate_level_a(rows)
    metadata = {
        "manifest": str(manifest_path),
        "paths": str(paths_path),
        "trial": trial,
        "trials": [trial],
        "run_dir": str(run_dir),
        "pose_source": args.pose_source,
        "evaluation_hz": args.evaluation_hz,
        "opensim_python": str(args.opensim_python),
        "force_reference": bool(args.force_reference),
    }
    outputs = write_level_a_outputs(out_dir, rows, aggregate, metadata)
    _print_summary(aggregate, outputs)
    return 0 if aggregate.get("status") in {"ok", "warning"} else 1


def _ensure_reference(repo_root: Path, args: argparse.Namespace, manifest: dict[str, Any], trial: str, reference_dir: Path) -> dict[str, Path]:
    paths = resolve_mocap_opensim_paths(manifest, trial)
    npz_path = reference_dir / f"opensim_fk_{trial}.npz"
    json_path = reference_dir / f"opensim_fk_{trial}.json"
    if npz_path.exists() and json_path.exists() and not args.force_reference:
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


def _load_pose(run_dir: Path, source: str) -> dict[str, Any]:
    rel = "pose3d_initial/pose3d_initial.pkl" if source == "initial" else "optimization/pose3d_refined.pkl"
    path = run_dir / rel
    if not path.exists():
        raise FileNotFoundError(f"Missing {source} pose artifact: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def _pose_sources(value: str) -> list[str]:
    return ["initial", "refined"] if value == "both" else [value]


def _backend_label(run_dir: Path, source: str, pose: dict[str, Any] | None) -> str:
    backend = str((pose or {}).get("backend") or _backend_from_run_name(run_dir) or "unknown")
    return f"{backend}_{source}"


def _backend_from_run_name(run_dir: Path) -> str | None:
    name = run_dir.name
    if "__" not in name:
        return None
    return name.rsplit("__", 1)[-1].split("_", 1)[0]


def _join_warnings(existing: Any, extra: list[str]) -> str:
    parts: list[str] = []
    if isinstance(existing, list):
        parts.extend(str(item) for item in existing if item)
    elif existing:
        parts.append(str(existing))
    parts.extend(extra)
    return "; ".join(parts)


def _print_summary(aggregate: dict[str, Any], outputs: dict[str, str]) -> None:
    print("[level-a-run] Status:", aggregate.get("status"), flush=True)
    for item in aggregate.get("ranking") or []:
        print(
            "[level-a-run] Rank",
            item.get("rank"),
            item.get("backend"),
            f"median={float(item.get('median_mm')):.3f} mm",
            flush=True,
        )
    print("[level-a-run] Summary:", outputs["summary_md"], flush=True)
    print("[level-a-run] CSV:", outputs["per_backend_trial_csv"], flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
