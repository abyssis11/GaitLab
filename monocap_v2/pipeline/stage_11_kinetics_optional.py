from __future__ import annotations

from pathlib import Path

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.logging_utils import write_json
from monocap_v2.core.stage_utils import stage_result


STAGE = "stage_11_kinetics_optional"


def run(run_dir: Path, cfg: dict, force: bool = False) -> dict:
    registry = ArtifactRegistry(run_dir)
    result = stage_result(STAGE, "skipped", reason="Kinetics are intentionally out of MVP scope.")
    write_json(registry.run_dir / "reports" / "kinetics_optional.json", result)
    return result

