from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from monocap_v2.core.logging_utils import read_yaml


ASSUMED_CAMERA_WARNING = (
    "Camera intrinsics are assumed, not calibrated. Reprojection-based "
    "optimization and absolute 3D scale may be sensitive to this value."
)


def assumed_pinhole_camera(width: int, height: int, focal_scale: float = 1.2) -> dict[str, Any]:
    focal = float(focal_scale) * float(max(width, height))
    return {
        "mode": "assumed_static",
        "model": "pinhole",
        "width": int(width),
        "height": int(height),
        "fx": focal,
        "fy": focal,
        "cx": float(width) / 2.0,
        "cy": float(height) / 2.0,
        "focal_scale": float(focal_scale),
        "distortion": {"enabled": False},
        "is_assumed": True,
        "source": None,
        "warning": ASSUMED_CAMERA_WARNING,
    }


def load_manual_camera(camera_path: Path, width: int, height: int) -> dict[str, Any]:
    data = read_yaml(camera_path)
    camera = data.get("camera", data)
    required = ["fx", "fy", "cx", "cy"]
    missing = [k for k in required if camera.get(k) is None]
    if missing:
        raise ValueError(f"Manual camera is missing required fields: {missing}")
    out = {
        "mode": "manual",
        "model": camera.get("model", "pinhole"),
        "width": int(camera.get("width", width)),
        "height": int(camera.get("height", height)),
        "fx": float(camera["fx"]),
        "fy": float(camera["fy"]),
        "cx": float(camera["cx"]),
        "cy": float(camera["cy"]),
        "distortion": camera.get("distortion", {"enabled": False}),
        "is_assumed": False,
        "source": str(camera_path),
    }
    validate_camera(out, width, height)
    return out


def load_opencap_calibration(camera_path: Path, width: int, height: int) -> dict[str, Any]:
    with camera_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"OpenCap camera calibration must be a dict: {camera_path}")

    K = np.asarray(data.get("intrinsicMat"), dtype=float)
    if K.shape != (3, 3):
        raise ValueError("OpenCap calibration missing 3x3 intrinsicMat")

    image_size_info = _describe_opencap_image_size(data.get("imageSize"), width, height, K)
    distortion_coeffs = _normalize_distortion(data.get("distortion"))
    out = {
        "mode": "opencap_manifest",
        "model": "pinhole",
        "width": int(width),
        "height": int(height),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "distortion": {"enabled": bool(np.any(np.abs(distortion_coeffs) > 0)), "coeffs": distortion_coeffs.tolist()},
        "is_assumed": False,
        "source": str(camera_path),
        "calibration_image_size": image_size_info,
    }

    extrinsics = _extract_dataset_extrinsics(data)
    if extrinsics:
        out["extrinsics_dataset"] = extrinsics

    validate_camera(out, width, height)
    if not image_size_info["principal_point_inside_video"]:
        raise ValueError("OpenCap calibration principal point is outside the preprocessed video bounds")
    return out


def _normalize_distortion(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((5,), dtype=float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros((5,), dtype=float)
    out = np.zeros((5,), dtype=float)
    out[: min(5, arr.size)] = arr[:5]
    return out


def _describe_opencap_image_size(value: Any, width: int, height: int, K: np.ndarray) -> dict[str, Any]:
    raw_values: list[float] = []
    if value is not None:
        arr = np.asarray(value, dtype=float).reshape(-1)
        raw_values = [float(v) for v in arr[:2]]

    direct_match = False
    swapped_match = False
    if len(raw_values) >= 2:
        a, b = int(round(raw_values[0])), int(round(raw_values[1]))
        direct_match = a == int(width) and b == int(height)
        swapped_match = a == int(height) and b == int(width)

    principal_inside = 0 <= float(K[0, 2]) <= float(width) and 0 <= float(K[1, 2]) <= float(height)
    if direct_match:
        handling = "direct_match"
    elif swapped_match:
        handling = "swapped_to_match_video"
    elif principal_inside:
        handling = "image_size_mismatch_principal_point_valid"
    else:
        handling = "invalid_for_video"

    return {
        "raw": raw_values,
        "video_width": int(width),
        "video_height": int(height),
        "direct_match": direct_match,
        "swapped_match": swapped_match,
        "principal_point_inside_video": principal_inside,
        "handling": handling,
    }


def _extract_dataset_extrinsics(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if data.get("rotation") is not None:
        rotation = np.asarray(data["rotation"], dtype=float)
        if rotation.shape == (3, 3):
            out["rotation_matrix"] = rotation.tolist()
    if data.get("translation") is not None:
        translation = np.asarray(data["translation"], dtype=float).reshape(-1)
        if translation.size >= 3:
            out["translation"] = translation[:3].tolist()
            out["translation_units"] = "dataset"
    if data.get("rotation_EulerAngles") is not None:
        euler = np.asarray(data["rotation_EulerAngles"], dtype=float).reshape(-1)
        if euler.size >= 3:
            out["rotation_euler_angles"] = euler[:3].tolist()
    return out


def validate_camera(camera: dict[str, Any], width: int, height: int) -> list[str]:
    errors = []
    if camera["fx"] <= 0 or camera["fy"] <= 0:
        errors.append("fx/fy must be positive")
    if not (0 <= camera["cx"] <= width):
        errors.append("cx must be inside image bounds")
    if not (0 <= camera["cy"] <= height):
        errors.append("cy must be inside image bounds")
    if int(camera["width"]) != int(width) or int(camera["height"]) != int(height):
        errors.append("camera resolution does not match preprocessed video")
    if errors:
        raise ValueError("; ".join(errors))
    return errors
