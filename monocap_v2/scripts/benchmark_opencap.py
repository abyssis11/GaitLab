#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monocap_v2.core.benchmark import aggregate_trials, parse_benchmark_report, write_benchmark_outputs
from monocap_v2.core.manifest import default_run_name, load_opencap_manifest


DEFAULT_TRIALS = "walking1,walking2,walking3"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Benchmark monocap_v2 on OpenCapDataset walking trials.")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--paths", required=True, type=Path)
    ap.add_argument("--trials", default=DEFAULT_TRIALS, help="Comma-separated trial IDs.")
    ap.add_argument("--activity", default="walking")
    ap.add_argument("--force", action="store_true", help="Force pipeline recomputation.")
    ap.add_argument("--skip-inference", action="store_true", help="Only summarize existing run folders.")
    ap.add_argument("--preset", default=None, help="Optional monocap_v2 pipeline preset.")
    ap.add_argument("--report-source", choices=["stage07", "mocap_validation"], default="stage07")
    ap.add_argument("--out", type=Path, default=None, help="Benchmark output directory.")
    ap.add_argument("--repo-root", type=Path, default=None, help=argparse.SUPPRESS)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root or Path(__file__).resolve().parents[2]
    manifest = load_opencap_manifest(args.manifest, args.paths)
    trials = [trial.strip() for trial in args.trials.split(",") if trial.strip()]
    out_dir = args.out or (repo_root / "monocap_v2" / "benchmarks" / f"{manifest.get('subject_id', 'subject')}_{args.activity}")
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    rows = []
    pipeline_failures = []
    for trial in trials:
        run_dir = repo_root / "monocap_v2" / "runs" / default_run_name(manifest, trial)
        if not args.skip_inference:
            rc = _run_pipeline(repo_root, args, trial)
            if rc != 0:
                pipeline_failures.append({"trial": trial, "run_dir": str(run_dir), "returncode": rc})
                rows.append(
                    {
                        "trial": trial,
                        "run_dir": str(run_dir),
                        "report_source": args.report_source,
                        "status": "pipeline_failed",
                        "stage_status": None,
                        "error": f"Pipeline exited with return code {rc}.",
                    }
                )
                continue
        expected_backend = "wham" if args.preset == "opencap_wham" else None
        rows.append(parse_benchmark_report(run_dir, trial, report_source=args.report_source, expected_backend=expected_backend))

    aggregate = aggregate_trials(rows, descriptive=args.report_source == "mocap_validation")
    metadata: dict[str, Any] = {
        "manifest": str(args.manifest),
        "paths": str(args.paths),
        "trials": trials,
        "activity": args.activity,
        "skip_inference": bool(args.skip_inference),
        "force": bool(args.force),
        "preset": args.preset,
        "report_source": args.report_source,
        "pipeline_failures": pipeline_failures,
    }
    outputs = write_benchmark_outputs(out_dir, rows, aggregate, metadata)
    _print_summary(aggregate, outputs)
    return 0 if aggregate["status"] in {"ok", "warning"} else 1


def _run_pipeline(repo_root: Path, args: argparse.Namespace, trial: str) -> int:
    cmd = [
        sys.executable,
        str(repo_root / "monocap_v2" / "run_pipeline.py"),
        "--manifest",
        str(args.manifest),
        "--paths",
        str(args.paths),
        "--trial",
        trial,
        "--activity",
        args.activity,
    ]
    if args.force:
        cmd.append("--force")
    if args.preset:
        cmd.extend(["--preset", args.preset])
    print(f"[benchmark] Running trial {trial}: {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, cwd=str(repo_root))
    return int(completed.returncode)


def _print_summary(aggregate: dict[str, Any], outputs: dict[str, str]) -> None:
    print("[benchmark] Status:", aggregate.get("status"), flush=True)
    if aggregate.get("valid_trial_count"):
        print(
            "[benchmark] Median MPJPE initial/refined/improvement:",
            f"{aggregate['median_initial_primary_mpjpe_mm']:.3f} /",
            f"{aggregate['median_refined_primary_mpjpe_mm']:.3f} /",
            f"{aggregate['median_primary_improvement_mm']:.3f} mm",
            flush=True,
        )
    print("[benchmark] Summary:", outputs["summary_md"], flush=True)
    print("[benchmark] CSV:", outputs["per_trial_csv"], flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
