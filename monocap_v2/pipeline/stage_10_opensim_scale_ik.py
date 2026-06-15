from __future__ import annotations

from pathlib import Path

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.logging_utils import write_json
from monocap_v2.core.stage_utils import cached, stage_result


STAGE = "stage_10_opensim_scale_ik"


def run(run_dir: Path, cfg: dict, force: bool = False) -> dict:
    registry = ArtifactRegistry(run_dir)
    out_path = registry.ensure_parent("opensim_ik_report")
    if cached(out_path, force):
        return stage_result(STAGE, "cached", output=str(out_path))
    if not cfg.get("config", {}).get("opensim", {}).get("enabled", False):
        result = stage_result(STAGE, "skipped", reason="OpenSim disabled in config.")
        write_json(out_path, result)
        return result
    result = stage_result(STAGE, "skipped", reason="OpenSim stage is not implemented in MVP.")
    write_json(out_path, result)
    return result

