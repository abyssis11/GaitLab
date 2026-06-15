from __future__ import annotations

from pathlib import Path

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.logging_utils import write_json
from monocap_v2.core.manifest import extract_subject_info
from monocap_v2.core.stage_utils import cached, stage_result


STAGE = "stage_00_validate_inputs"


def run(run_dir: Path, cfg: dict, force: bool = False) -> dict:
    registry = ArtifactRegistry(run_dir)
    subject_path = registry.ensure_parent("subject_info")
    if cached(subject_path, force):
        return stage_result(STAGE, "cached", output=str(subject_path))

    trial = cfg["trial"]
    manifest_summary = cfg.get("manifest_summary", {})
    manifest = {
        **manifest_summary,
        "subject_id": manifest_summary.get("subject_id"),
        "session": manifest_summary.get("session"),
        "camera": manifest_summary.get("camera"),
        "session_metadata": manifest_summary.get("session_metadata"),
    }
    # Use resolved manifest file for full subject extraction if available.
    try:
        import yaml

        with registry.get("manifest_resolved").open("r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f) or manifest
    except Exception:
        pass

    subject = extract_subject_info(manifest, trial)
    checks = {
        "video_field": cfg["video_field"],
        "raw_video": cfg.get("raw_video"),
        "raw_video_exists": Path(str(cfg.get("raw_video", ""))).exists(),
        "mocap_trc": cfg.get("mocap_trc"),
        "mocap_trc_exists": Path(str(cfg.get("mocap_trc", ""))).exists() if cfg.get("mocap_trc") else False,
        "session_metadata": subject.get("session_metadata"),
        "session_metadata_exists": Path(str(subject.get("session_metadata", ""))).exists()
        if subject.get("session_metadata")
        else False,
    }
    errors = []
    if not checks["raw_video"]:
        errors.append(f"Trial is missing video field '{cfg['video_field']}'")
    elif not checks["raw_video_exists"] and not cfg.get("dry_run"):
        errors.append(f"Video does not exist: {checks['raw_video']}")

    subject["validation"] = checks
    write_json(subject_path, subject)
    write_json(registry.ensure_parent("raw_video_ref"), {"path": checks["raw_video"], "exists": checks["raw_video_exists"]})

    if errors:
        return stage_result(STAGE, "failed", errors=errors)
    status = "ok_with_warnings" if not checks["raw_video_exists"] else "ok"
    return stage_result(STAGE, status, output=str(subject_path), checks=checks)

