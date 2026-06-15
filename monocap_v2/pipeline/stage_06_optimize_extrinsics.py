from __future__ import annotations

from pathlib import Path

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.logging_utils import write_json
from monocap_v2.core.stage_utils import cached, stage_result


STAGE = "stage_06_optimize_extrinsics"


def run(run_dir: Path, cfg: dict, force: bool = False) -> dict:
    registry = ArtifactRegistry(run_dir)
    out_path = registry.ensure_parent("opt_stage1_report")
    if cached(out_path, force):
        return stage_result(STAGE, "cached", output=str(out_path))
    report = stage_result(
        STAGE,
        "skipped",
        reason="Extrinsics optimization contract exists, but optimizer is not implemented in MVP.",
    )
    write_json(out_path, report)
    return report

