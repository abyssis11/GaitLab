from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object in {path}")
    return data


def write_yaml(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(to_jsonable(data), f, sort_keys=False)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2)
        f.write("\n")


def to_jsonable(obj: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is expected but optional here
        np = None

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
    return obj


def setup_logger(run_dir: Path | None = None, name: str = "monocap_v2") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if run_dir is not None:
        log_dir = ensure_dir(run_dir / "logs")
        file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_state(run_dir: Path) -> dict[str, Any]:
    state_path = run_dir / "pipeline_state.json"
    if not state_path.exists():
        return {"created_at": utc_now_iso(), "stages": {}}
    return read_json(state_path)


def update_stage_state(run_dir: Path, stage_name: str, result: dict[str, Any]) -> None:
    state = load_state(run_dir)
    state.setdefault("stages", {})[stage_name] = {
        "updated_at": utc_now_iso(),
        **to_jsonable(result),
    }
    state["updated_at"] = utc_now_iso()
    write_json(run_dir / "pipeline_state.json", state)

