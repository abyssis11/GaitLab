from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from monocap_v2.core.geometry import apply_axis_expr, camera_to_eval_coords


LEGACY_PROFILE = "legacy_x_yup_z"
OPENCAP_CAMERA_PROFILE = "opencap_camera_yflip_to_lab"
OPENCAP_CAMERA_LR_PROFILE = "opencap_camera_yflip_to_lab_lr"
OPENCAP_CAMERA_FULL_PROFILE = "opencap_camera_yflip_full_to_lab"
OPENCAP_CAMERA_FULL_LR_PROFILE = "opencap_camera_yflip_full_to_lab_lr"
CUSTOM_AXIS_PROFILE = "custom_axis"
BACKEND_AXIS_PROFILE = "backend_axis_map"

WHAM_CONVENTION_PROFILES = [
    LEGACY_PROFILE,
    OPENCAP_CAMERA_PROFILE,
    OPENCAP_CAMERA_LR_PROFILE,
    OPENCAP_CAMERA_FULL_PROFILE,
    OPENCAP_CAMERA_FULL_LR_PROFILE,
    "opencap_yaw180",
    "opencap_yaw180_lr",
    "opencap_yaw180_lr_offset120ms",
    "opencap_full_yaw180",
    "opencap_full_yaw180_lr",
    "opencap_full_yaw180_lr_offset120ms",
]


def apply_level_a_convention(
    backend: str,
    source_names: list[str],
    joints: np.ndarray,
    axis_mode: str,
    run_config: dict[str, Any] | None = None,
    convention_profile: str | None = None,
    axis_explicit: bool = False,
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    """Apply the evaluation-only coordinate convention used by Level A.

    This intentionally transforms only the arrays used for scoring/plots. It
    never mutates the original WHAM artifact or SMPL vertex/parameter payloads.
    """
    backend = str(backend or "unknown")
    values = np.asarray(joints, dtype=float)
    if backend != "wham":
        transformed = apply_axis_mode(values, axis_mode)
        return list(source_names), transformed, _axis_meta(axis_mode, backend_specific=not axis_explicit)

    requested = convention_profile or _config_wham_profile(run_config)
    if axis_explicit and convention_profile is None:
        transformed = apply_axis_mode(values, axis_mode)
        return list(source_names), transformed, _axis_meta(axis_mode, backend_specific=False)

    profile = normalize_wham_convention_profile(requested, run_config=run_config)
    spec = wham_convention_profile_specs(run_config)[profile]
    return _apply_profile_spec(profile, spec, source_names, values, run_config)


def normalize_wham_convention_profile(profile: str | None, run_config: dict[str, Any] | None = None) -> str:
    raw = str(profile or LEGACY_PROFILE).strip().lower().replace("-", "_")
    aliases = {
        "legacy": LEGACY_PROFILE,
        "legacy_x_yup_z": LEGACY_PROFILE,
        "x_yup_z": LEGACY_PROFILE,
        "opencap": OPENCAP_CAMERA_PROFILE,
        "opencap_camera": OPENCAP_CAMERA_PROFILE,
        "opencap_camera_yflip_to_lab": OPENCAP_CAMERA_PROFILE,
        "opencap_camera_yflip_to_lab_lr": OPENCAP_CAMERA_LR_PROFILE,
        "opencap_camera_yflip_full_to_lab": OPENCAP_CAMERA_FULL_PROFILE,
        "opencap_camera_yflip_full_to_lab_lr": OPENCAP_CAMERA_FULL_LR_PROFILE,
        "opencap_camera_lr": OPENCAP_CAMERA_LR_PROFILE,
        "opencap_lr": OPENCAP_CAMERA_LR_PROFILE,
        "opencap_full": OPENCAP_CAMERA_FULL_PROFILE,
        "opencap_full_lr": OPENCAP_CAMERA_FULL_LR_PROFILE,
        "opencap_yaw180": "opencap_yaw180",
        "opencap_yaw180_lr": "opencap_yaw180_lr",
        "opencap_yaw180_lr_offset120ms": "opencap_yaw180_lr_offset120ms",
        "opencap_full_yaw180": "opencap_full_yaw180",
        "opencap_full_yaw180_lr": "opencap_full_yaw180_lr",
        "opencap_full_yaw180_lr_offset120ms": "opencap_full_yaw180_lr_offset120ms",
        "visual_candidate": "opencap_full_yaw180_lr_offset120ms",
    }
    profile_name = aliases.get(raw, raw)
    specs = wham_convention_profile_specs(run_config)
    if profile_name not in specs:
        allowed = ", ".join(specs)
        raise ValueError(f"Unsupported WHAM convention profile {profile!r}; expected one of: {allowed}")
    return profile_name


def wham_convention_profile_names(run_config: dict[str, Any] | None = None, include_diagnostic: bool = True) -> list[str]:
    specs = wham_convention_profile_specs(run_config)
    if include_diagnostic:
        return list(specs)
    return [name for name, spec in specs.items() if not bool(spec.get("diagnostic_only"))]


def wham_convention_profile_specs(run_config: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    config_specs = (((run_config or {}).get("config") or {}).get("level_a") or {}).get("wham_convention_profiles")
    if isinstance(config_specs, dict):
        specs.update(_normalize_profile_specs(config_specs))
    default_specs = _default_profile_specs()
    for name, spec in default_specs.items():
        specs.setdefault(name, spec)
    return specs


def apply_axis_mode(values: np.ndarray, axis_mode: str) -> np.ndarray:
    if axis_mode == "identity":
        return np.asarray(values, dtype=float)
    if axis_mode == "camera_to_eval":
        return camera_to_eval_coords(values)
    if axis_mode.startswith("axis:"):
        return apply_axis_expr(values, axis_mode.split(":", 1)[1])
    return apply_axis_expr(values, axis_mode)


def axis_expr_matrix(axis_mode: str) -> np.ndarray:
    if axis_mode in {None, "", "identity"}:
        return np.eye(3, dtype=float)
    if axis_mode == "camera_to_eval":
        return np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=float)
    if str(axis_mode).startswith("axis:"):
        axis_mode = str(axis_mode).split(":", 1)[1]
    return apply_axis_expr(np.eye(3, dtype=float), str(axis_mode))


def axis_expr_determinant(axis_mode: str) -> float:
    return float(np.linalg.det(axis_expr_matrix(axis_mode)))


def load_opencap_camera_rotation(run_config: dict[str, Any] | None) -> tuple[np.ndarray, Path]:
    calibration = load_opencap_camera_extrinsics(run_config)
    return calibration["rotation"], calibration["source"]


def load_opencap_camera_extrinsics(run_config: dict[str, Any] | None, require_translation: bool = False) -> dict[str, Any]:
    calibration = ((run_config or {}).get("manifest_summary") or {}).get("calibration") or {}
    source = calibration.get("intrinsics_extrinsics")
    if not source:
        raise ValueError("OpenCap camera calibration is unavailable in run_config manifest_summary.calibration.intrinsics_extrinsics.")
    path = Path(str(source))
    if not path.exists():
        raise ValueError(f"OpenCap camera calibration does not exist: {path}")
    with path.open("rb") as f:
        data = pickle.load(f)
    rotation = np.asarray(data.get("rotation"), dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"OpenCap camera calibration rotation must be 3x3: {path}")
    raw_translation = data.get("translation")
    translation = np.asarray(raw_translation, dtype=float).reshape(-1) if raw_translation is not None else np.asarray([], dtype=float)
    if translation.size < 3:
        if require_translation:
            raise ValueError(f"OpenCap camera calibration translation must contain at least three values: {path}")
        translation_m = np.zeros(3, dtype=float)
        translation_units = "missing"
    else:
        translation_m, translation_units = _translation_to_meters(translation[:3])
    return {
        "rotation": rotation,
        "translation_m": translation_m,
        "translation_units": translation_units,
        "source": path,
    }


def swap_left_right_names(names: list[str] | tuple[str, ...]) -> list[str]:
    return [_swap_name(str(name)) for name in names]


def convention_profile_label(profile: str) -> str:
    profile = normalize_wham_convention_profile(profile)
    return str(wham_convention_profile_specs()[profile].get("label") or profile)


def _config_wham_profile(run_config: dict[str, Any] | None) -> str:
    cfg = ((run_config or {}).get("config") or {}).get("level_a") or {}
    return str(cfg.get("wham_convention_profile") or LEGACY_PROFILE)


def _axis_meta(axis_mode: str, backend_specific: bool) -> dict[str, Any]:
    det = axis_expr_determinant(axis_mode)
    return {
        "convention_profile": BACKEND_AXIS_PROFILE if backend_specific else CUSTOM_AXIS_PROFILE,
        "convention_source": "backend_axis_map" if backend_specific else "explicit_axis_map",
        "pre_axis": axis_mode,
        "post_axis": "identity",
        "axis_mode": axis_mode,
        "camera_rotation_source": None,
        "camera_translation_source": None,
        "camera_translation_units": None,
        "camera_translation_m": None,
        "uses_camera_translation": False,
        "time_offset_s": 0.0,
        "pre_axis_determinant": det,
        "post_transform_determinant": 1.0,
        "linear_transform_determinant": det,
        "proper_post_transform": True,
        "proper_linear_transform": _is_proper_determinant(det),
        "default_eligible": _is_proper_determinant(det),
        "left_right_swap": False,
        "diagnostic_only": False,
    }


def _apply_profile_spec(
    profile: str,
    spec: dict[str, Any],
    source_names: list[str],
    values: np.ndarray,
    run_config: dict[str, Any] | None,
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    pre_axis = str(spec.get("pre_axis") or "identity")
    post_axis = str(spec.get("post_axis") or spec.get("post_axis_or_rotation") or "identity")
    use_camera_rotation = bool(spec.get("use_camera_rotation", False))
    use_camera_translation = bool(spec.get("use_camera_translation", False))
    left_right_swap = bool(spec.get("left_right_swap", False))
    time_offset_s = float(spec.get("time_offset_s") or 0.0)
    calibration: dict[str, Any] | None = None
    transformed = apply_axis_mode(values, pre_axis)
    if use_camera_rotation or use_camera_translation:
        calibration = load_opencap_camera_extrinsics(run_config, require_translation=use_camera_translation)
    if use_camera_translation:
        transformed = transformed - calibration["translation_m"][None, None, :]
    if use_camera_rotation:
        transformed = transformed @ calibration["rotation"]
    if post_axis != "identity":
        transformed = apply_axis_mode(transformed, post_axis)
    names = swap_left_right_names(source_names) if left_right_swap else list(source_names)

    pre_det = axis_expr_determinant(pre_axis)
    post_det = axis_expr_determinant(post_axis)
    camera_det = float(np.linalg.det(calibration["rotation"])) if calibration is not None and use_camera_rotation else 1.0
    linear_det = float(pre_det * camera_det * post_det)
    proper_post = _is_proper_determinant(post_det)
    proper_linear = _is_proper_determinant(linear_det)
    diagnostic_only = bool(spec.get("diagnostic_only", False)) or not proper_post
    default_eligible = (
        bool(spec.get("default_eligible", not diagnostic_only))
        and proper_post
        and proper_linear
        and not left_right_swap
        and abs(time_offset_s) < 1e-12
    )
    return names, transformed, {
        "convention_profile": profile,
        "convention_source": _convention_source(profile, spec, use_camera_rotation, use_camera_translation),
        "profile_source": str(spec.get("source") or "config"),
        "label": str(spec.get("label") or profile),
        "pre_axis": pre_axis,
        "post_axis": post_axis,
        "axis_mode": _axis_mode_label(pre_axis, post_axis, use_camera_rotation, use_camera_translation),
        "camera_rotation_source": str(calibration["source"]) if calibration is not None and use_camera_rotation else None,
        "camera_translation_source": str(calibration["source"]) if calibration is not None and use_camera_translation else None,
        "camera_translation_units": calibration["translation_units"] if calibration is not None and use_camera_translation else None,
        "camera_translation_m": calibration["translation_m"].tolist() if calibration is not None and use_camera_translation else None,
        "uses_camera_translation": use_camera_translation,
        "time_offset_s": time_offset_s,
        "pre_axis_determinant": pre_det,
        "post_transform_determinant": post_det,
        "camera_rotation_determinant": camera_det,
        "linear_transform_determinant": linear_det,
        "proper_post_transform": proper_post,
        "proper_linear_transform": proper_linear,
        "default_eligible": default_eligible,
        "left_right_swap": left_right_swap,
        "diagnostic_only": diagnostic_only,
    }


def _axis_mode_label(pre_axis: str, post_axis: str, use_camera_rotation: bool, use_camera_translation: bool) -> str:
    parts = [pre_axis]
    if use_camera_translation:
        parts.append("-camera_translation")
    if use_camera_rotation:
        parts.append("@opencap_camera_rotation")
    if post_axis != "identity":
        parts.append(f"@post_axis({post_axis})")
    return "".join(parts)


def _convention_source(profile: str, spec: dict[str, Any], use_camera_rotation: bool, use_camera_translation: bool) -> str:
    if use_camera_rotation or use_camera_translation:
        return "opencap_camera_calibration"
    if profile == LEGACY_PROFILE:
        return "level_a_default"
    return str(spec.get("source") or "config")


def _normalize_profile_specs(raw_specs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, raw in raw_specs.items():
        if not isinstance(raw, dict):
            continue
        key = str(name).strip().lower().replace("-", "_")
        spec = dict(raw)
        spec.setdefault("source", "config")
        spec.setdefault("post_axis", spec.pop("post_axis_or_rotation", "identity"))
        spec.setdefault("pre_axis", "identity")
        spec.setdefault("use_camera_rotation", False)
        spec.setdefault("use_camera_translation", False)
        spec.setdefault("left_right_swap", False)
        spec.setdefault("time_offset_s", 0.0)
        spec.setdefault("diagnostic_only", False)
        out[key] = spec
    return out


def _default_profile_specs() -> dict[str, dict[str, Any]]:
    path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        specs = (((cfg.get("level_a") or {}).get("wham_convention_profiles")) or {})
        if isinstance(specs, dict):
            return _normalize_profile_specs(specs)
    except Exception:
        pass
    return _fallback_profile_specs()


def _fallback_profile_specs() -> dict[str, dict[str, Any]]:
    return _normalize_profile_specs(
        {
            LEGACY_PROFILE: {"label": "Legacy x,-y,z", "pre_axis": "x,-y,z", "diagnostic_only": False},
            OPENCAP_CAMERA_PROFILE: {
                "label": "OpenCap camera y-flip to lab",
                "pre_axis": "x,-y,z",
                "use_camera_rotation": True,
                "diagnostic_only": False,
            },
            OPENCAP_CAMERA_LR_PROFILE: {
                "label": "OpenCap camera y-flip to lab + L/R",
                "pre_axis": "x,-y,z",
                "use_camera_rotation": True,
                "left_right_swap": True,
                "diagnostic_only": True,
            },
            OPENCAP_CAMERA_FULL_PROFILE: {
                "label": "OpenCap camera full extrinsic to lab",
                "pre_axis": "x,-y,z",
                "use_camera_rotation": True,
                "use_camera_translation": True,
                "diagnostic_only": True,
            },
            OPENCAP_CAMERA_FULL_LR_PROFILE: {
                "label": "OpenCap camera full extrinsic to lab + L/R",
                "pre_axis": "x,-y,z",
                "use_camera_rotation": True,
                "use_camera_translation": True,
                "left_right_swap": True,
                "diagnostic_only": True,
            },
        }
    )


def _is_proper_determinant(value: float) -> bool:
    return bool(np.isfinite(value) and abs(float(value) - 1.0) < 1e-6)


def _translation_to_meters(translation: np.ndarray) -> tuple[np.ndarray, str]:
    translation = np.asarray(translation, dtype=float).reshape(3)
    if np.linalg.norm(translation) > 20.0:
        return translation / 1000.0, "mm_to_m_inferred"
    return translation, "m_inferred"


def _swap_name(name: str) -> str:
    lower = name.lower()
    if lower.startswith("left"):
        return "right" + name[4:]
    if lower.startswith("right"):
        return "left" + name[5:]
    if lower.startswith("l") and any(key in lower for key in ["hip", "kne", "ank", "toe", "mtp", "heel"]):
        return "r" + name[1:]
    if lower.startswith("r") and any(key in lower for key in ["hip", "kne", "ank", "toe", "mtp", "heel"]):
        return "l" + name[1:]
    return name
