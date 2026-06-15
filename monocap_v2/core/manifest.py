from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from monocap_v2.core.logging_utils import read_yaml


_PLACEHOLDER = re.compile(r"\$\{([^}]+)\}")


def _resolve_placeholder(value: str, variables: dict[str, Any]) -> str:
    def replacer(match: re.Match[str]) -> str:
        key = match.group(1)
        current: Any = variables
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                raise KeyError(f"Placeholder '{key}' could not be resolved")
            current = current[part]
        return str(current)

    return _PLACEHOLDER.sub(replacer, value)


def _recursive_resolve(obj: Any, variables: dict[str, Any]) -> Any:
    if isinstance(obj, str):
        return _resolve_placeholder(obj, variables)
    if isinstance(obj, dict):
        return {k: _recursive_resolve(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_recursive_resolve(v, variables) for v in obj]
    return obj


def load_opencap_manifest(manifest_path: Path, paths_path: Path) -> dict[str, Any]:
    paths_config = read_yaml(paths_path)
    raw = read_yaml(manifest_path)
    context = {
        "datasets": {
            "opencap_root": paths_config.get("datasets", {}).get("opencap_root", ""),
            "gpjatk_root": paths_config.get("datasets", {}).get("gpjatk_root", ""),
        },
        "outputs_root": paths_config.get("outputs_root"),
    }
    if "paths" not in raw or "root" not in raw["paths"]:
        raise ValueError("Manifest must contain paths.root")
    root = _resolve_placeholder(str(raw["paths"]["root"]), context)
    context["paths"] = {"root": root}
    resolved = _recursive_resolve(raw, context)
    resolved["_source_manifest"] = str(manifest_path)
    resolved["_source_paths"] = str(paths_path)
    return resolved


def find_trial(manifest: dict[str, Any], trial_id: str) -> tuple[str, dict[str, Any]]:
    for subset, trials in (manifest.get("trials") or {}).items():
        for trial in trials or []:
            if trial.get("id") == trial_id:
                return str(subset), dict(trial)
    raise KeyError(f"Trial '{trial_id}' not found in manifest")


def default_run_name(manifest: dict[str, Any], trial_id: str) -> str:
    parts = [
        manifest.get("subject_id", "subject"),
        manifest.get("session", "Session"),
        manifest.get("camera", "Cam"),
        trial_id,
    ]
    return "_".join(_safe_part(str(p)) for p in parts)


def _safe_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "unknown"


def extract_subject_info(manifest: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    info = {
        "id": manifest.get("subject_id"),
        "session": manifest.get("session"),
        "camera": manifest.get("camera"),
        "height_m": None,
        "mass_kg": None,
        "sex": None,
        "session_metadata": manifest.get("session_metadata"),
        "trial_id": trial.get("id"),
    }
    meta_path = manifest.get("session_metadata")
    if isinstance(meta_path, str) and Path(meta_path).exists():
        try:
            meta = read_yaml(Path(meta_path))
            info.update(_extract_height_mass_sex(meta))
        except Exception as exc:
            info["session_metadata_warning"] = str(exc)
    return info


def _extract_height_mass_sex(meta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"height_m": None, "mass_kg": None, "sex": None}
    flat = _flatten(meta)

    sex = _first(flat, ["sex", "gender", "biologicalsex"])
    if sex is not None:
        s = str(sex).strip().lower()
        out["sex"] = "male" if s in {"m", "male"} else "female" if s in {"f", "female"} else str(sex)

    height = _first(flat, ["height", "heightm", "stature", "bodyheight", "bodyheightm"])
    if height is not None:
        out["height_m"] = _parse_metric_value(height, default_unit="m", plausible_m=True)

    mass = _first(flat, ["mass", "masskg", "weight", "weightkg", "bodymass", "bodyweight"])
    if mass is not None:
        out["mass_kg"] = _parse_metric_value(mass, default_unit="kg", plausible_m=False)
    return out


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            norm = re.sub(r"[^a-z0-9]", "", str(key).lower())
            out[norm] = value
            out.update(_flatten(value, f"{prefix}{norm}."))
    return out


def _first(flat: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        norm = re.sub(r"[^a-z0-9]", "", key.lower())
        if norm in flat:
            return flat[norm]
    return None


def _parse_metric_value(value: Any, default_unit: str, plausible_m: bool) -> float | None:
    if isinstance(value, (int, float)):
        number = float(value)
        unit = default_unit
    else:
        match = re.match(r"\s*([-+]?\d*\.?\d+)\s*([A-Za-z]*)", str(value))
        if not match:
            return None
        number = float(match.group(1))
        unit = match.group(2).lower() or default_unit

    if plausible_m:
        if unit == "cm":
            return number / 100.0
        if unit == "mm":
            return number / 1000.0
        if 50.0 <= number <= 260.0:
            return number / 100.0
        if 500.0 <= number <= 2600.0:
            return number / 1000.0
        return number

    if unit == "g":
        return number / 1000.0
    if unit == "kg":
        return number
    if number > 1000.0:
        return number / 1000.0
    return number

