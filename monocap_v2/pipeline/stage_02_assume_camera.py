from __future__ import annotations

from pathlib import Path

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.camera_model import (
    ASSUMED_CAMERA_WARNING,
    assumed_pinhole_camera,
    load_manual_camera,
    load_opencap_calibration,
)
from monocap_v2.core.logging_utils import read_json, write_json
from monocap_v2.core.stage_utils import cached, stage_result


STAGE = "stage_02_assume_camera"


def run(run_dir: Path, cfg: dict, force: bool = False) -> dict:
    registry = ArtifactRegistry(run_dir)
    camera_path = registry.ensure_parent("camera_assumed")
    if cached(camera_path, force):
        return stage_result(STAGE, "cached", output=str(camera_path))

    video_info = read_json(registry.get("video_info"))
    width = int(video_info["width"])
    height = int(video_info["height"])
    warnings = []
    override = cfg.get("camera_override")
    if override:
        camera = load_manual_camera(Path(override), width, height)
        source_kind = "manual"
    else:
        calibration_path = _manifest_calibration_path(cfg)
        if calibration_path:
            try:
                camera = load_opencap_calibration(calibration_path, width, height)
                source_kind = "opencap_manifest"
            except Exception as exc:
                warnings.append(f"Could not use manifest camera calibration at {calibration_path}: {exc}")
                camera = _assumed_camera(cfg, width, height)
                source_kind = "assumed_static"
        else:
            camera = _assumed_camera(cfg, width, height)
            source_kind = "assumed_static"

    if camera.get("is_assumed"):
        warnings.append(ASSUMED_CAMERA_WARNING)
    qc = {
        "stage": STAGE,
        "status": "ok",
        "camera_source": source_kind,
        "mode": camera.get("mode"),
        "source": camera.get("source"),
        "is_assumed": bool(camera["is_assumed"]),
        "is_calibrated": not bool(camera["is_assumed"]),
        "fx_positive": camera["fx"] > 0,
        "fy_positive": camera["fy"] > 0,
        "principal_point_inside_image": 0 <= camera["cx"] <= width and 0 <= camera["cy"] <= height,
        "distortion_enabled": bool((camera.get("distortion") or {}).get("enabled", False)),
        "calibration_image_size": camera.get("calibration_image_size"),
        "warnings": warnings,
        "warning": "; ".join(warnings) if warnings else None,
    }
    write_json(camera_path, camera)
    write_json(registry.ensure_parent("camera_qc"), qc)
    return stage_result(
        STAGE,
        "ok",
        output=str(camera_path),
        camera_source=source_kind,
        is_assumed=bool(camera["is_assumed"]),
        is_calibrated=not bool(camera["is_assumed"]),
    )


def _manifest_calibration_path(cfg: dict) -> Path | None:
    calibration = (cfg.get("manifest_summary") or {}).get("calibration") or {}
    raw = calibration.get("intrinsics_extrinsics") if isinstance(calibration, dict) else None
    if not raw:
        return None
    return Path(raw)


def _assumed_camera(cfg: dict, width: int, height: int) -> dict:
    focal_scale = float(cfg.get("config", {}).get("camera", {}).get("focal_scale", 1.2))
    return assumed_pinhole_camera(width, height, focal_scale=focal_scale)
