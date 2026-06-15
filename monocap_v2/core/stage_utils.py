from __future__ import annotations

from pathlib import Path
from typing import Any

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.logging_utils import utc_now_iso, write_json


def cached(path: Path, force: bool) -> bool:
    return path.exists() and not force


def stage_result(stage: str, status: str, **extra: Any) -> dict[str, Any]:
    return {"stage": stage, "status": status, "timestamp": utc_now_iso(), **extra}


def write_stage_qc(registry: ArtifactRegistry, key: str, result: dict[str, Any]) -> None:
    write_json(registry.ensure_parent(key), result)

