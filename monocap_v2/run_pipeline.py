#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.logging_utils import (
    read_yaml,
    setup_logger,
    update_stage_state,
    write_yaml,
)
from monocap_v2.core.manifest import default_run_name, find_trial, load_opencap_manifest


STAGES = [
    "stage_00_validate_inputs",
    "stage_01_preprocess_video",
    "stage_02_assume_camera",
    "stage_03_pose2d",
    "stage_04_pose3d_initial",
    "stage_05_activity_and_contacts",
    "stage_06_optimize_extrinsics",
    "stage_07_optimize_pose",
    "stage_08_extract_virtual_markers",
    "stage_09_export_trc",
    "stage_10_opensim_scale_ik",
    "stage_11_kinetics_optional",
    "stage_12_visualize",
    "stage_12_mocap_validation",
    "stage_13_qc_report",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the isolated monocap_v2 OpenCap pipeline.")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--paths", required=True, type=Path)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--activity", required=True)
    ap.add_argument("--preset", default=None, help="Optional config preset name from monocap_v2/configs/presets.")
    ap.add_argument("--video-field", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stages", default=None, help="Comma-separated subset of stage module names.")
    ap.add_argument("--backend-pose2d", default=None)
    ap.add_argument("--backend-pose3d", default=None)
    ap.add_argument("--camera", type=Path, default=None)
    ap.add_argument("--max-frames", type=int, default=None, help="Limit frames for backend smoke tests.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    cfg = _load_config(args.preset)
    manifest = load_opencap_manifest(args.manifest, args.paths)
    subset, trial = find_trial(manifest, args.trial)

    if args.backend_pose2d:
        cfg.setdefault("backends", {})["pose2d"] = args.backend_pose2d
    if args.backend_pose3d:
        cfg.setdefault("backends", {})["pose3d"] = args.backend_pose3d
    if args.video_field:
        cfg.setdefault("pipeline", {})["default_video_field"] = args.video_field
    if args.max_frames is not None:
        cfg.setdefault("metrabs", {})["max_frames"] = int(args.max_frames)
        cfg.setdefault("wham", {})["max_frames"] = int(args.max_frames)
        cfg.setdefault("rtmw3d", {})["max_frames"] = int(args.max_frames)

    video_field = cfg.get("pipeline", {}).get("default_video_field", "video_sync")
    run_dir = args.out or (repo_root / "monocap_v2" / "runs" / default_run_name(manifest, args.trial))
    run_dir.mkdir(parents=True, exist_ok=True)
    registry = ArtifactRegistry(run_dir)
    registry.ensure_standard_dirs()
    logger = setup_logger(run_dir)

    selected_stages = _select_stages(args.stages)
    run_config: dict[str, Any] = {
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
        "manifest_path": str(args.manifest),
        "paths_path": str(args.paths),
        "trial_id": args.trial,
        "trial_subset": subset,
        "trial": trial,
        "activity": args.activity,
        "video_field": video_field,
        "raw_video": trial.get(video_field),
        "mocap_trc": trial.get("mocap_trc"),
        "camera_override": str(args.camera) if args.camera else None,
        "preset": args.preset,
        "dry_run": bool(args.dry_run),
        "force": bool(args.force),
        "selected_stages": selected_stages,
        "config": cfg,
        "manifest_summary": {
            "subject_id": manifest.get("subject_id"),
            "session": manifest.get("session"),
            "camera": manifest.get("camera"),
            "fps_video": manifest.get("fps_video"),
            "fps_mocap": manifest.get("fps_mocap"),
            "units": manifest.get("units"),
            "calibration": manifest.get("calibration"),
        },
    }

    write_yaml(registry.ensure_parent("manifest_resolved"), manifest)
    write_yaml(registry.ensure_parent("run_config"), run_config)
    logger.info("Run directory: %s", run_dir)
    logger.info("Selected backends: pose2d=%s pose3d=%s", cfg["backends"]["pose2d"], cfg["backends"]["pose3d"])
    logger.info("Planned stages: %s", ", ".join(selected_stages))

    force_downstream = bool(args.force)
    for stage_name in selected_stages:
        if args.dry_run and stage_name != "stage_00_validate_inputs":
            result = {"stage": stage_name, "status": "planned", "dry_run": True}
            update_stage_state(run_dir, stage_name, result)
            logger.info("[DRY-RUN] Planned %s", stage_name)
            continue
        try:
            module = importlib.import_module(f"monocap_v2.pipeline.{stage_name}")
            result = module.run(run_dir, run_config, force=force_downstream)
            update_stage_state(run_dir, stage_name, result)
            logger.info("%s -> %s", stage_name, result.get("status"))
            if result.get("status") not in {"cached", "skipped", "planned"}:
                force_downstream = True
        except Exception as exc:
            result = {"stage": stage_name, "status": "failed", "error": str(exc)}
            update_stage_state(run_dir, stage_name, result)
            logger.exception("%s failed", stage_name)
            if cfg.get("pipeline", {}).get("stop_on_stage_error", True):
                return 1

    return 0


def _select_stages(stages_arg: str | None) -> list[str]:
    if not stages_arg:
        return list(STAGES)
    requested = [s.strip() for s in stages_arg.split(",") if s.strip()]
    unknown = [s for s in requested if s not in STAGES]
    if unknown:
        raise SystemExit(f"Unknown stage(s): {', '.join(unknown)}")
    return requested


def _load_config(preset: str | None) -> dict[str, Any]:
    config_dir = Path(__file__).resolve().parent / "configs"
    cfg = read_yaml(config_dir / "default.yaml")
    if not preset:
        return cfg
    preset_name = preset[:-5] if preset.endswith(".yaml") else preset
    preset_path = config_dir / "presets" / f"{preset_name}.yaml"
    if not preset_path.exists():
        raise SystemExit(f"Unknown preset '{preset}'. Expected YAML under {preset_path.parent}")
    return _deep_merge(cfg, read_yaml(preset_path))


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


if __name__ == "__main__":
    raise SystemExit(main())
