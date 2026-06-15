from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.camera_model import assumed_pinhole_camera, load_opencap_calibration, validate_camera
from monocap_v2.core.logging_utils import read_json, write_json
from monocap_v2.pipeline import stage_02_assume_camera


def test_assumed_camera_math() -> None:
    camera = assumed_pinhole_camera(1920, 1080, focal_scale=1.2)
    assert camera["cx"] == 960
    assert camera["cy"] == 540
    assert camera["fx"] == 2304
    assert camera["fy"] == 2304
    assert camera["is_assumed"] is True
    assert validate_camera(camera, 1920, 1080) == []


def test_load_opencap_calibration_direct_image_size(tmp_path: Path) -> None:
    path = _write_opencap_pickle(tmp_path / "camera.pickle", image_size=(720, 1280))
    camera = load_opencap_calibration(path, width=720, height=1280)
    assert camera["mode"] == "opencap_manifest"
    assert camera["is_assumed"] is False
    assert camera["fx"] == 913.0
    assert camera["fy"] == 914.0
    assert camera["distortion"]["enabled"] is True
    assert camera["distortion"]["coeffs"][:2] == [0.1, -0.2]
    assert camera["calibration_image_size"]["handling"] == "direct_match"
    assert "extrinsics_dataset" in camera


def test_load_opencap_calibration_swapped_image_size(tmp_path: Path) -> None:
    path = _write_opencap_pickle(tmp_path / "camera.pickle", image_size=(1280, 720))
    camera = load_opencap_calibration(path, width=720, height=1280)
    assert camera["calibration_image_size"]["handling"] == "swapped_to_match_video"
    assert camera["cx"] == 366.0
    assert camera["cy"] == 638.0


def test_load_opencap_calibration_rejects_invalid_principal_point(tmp_path: Path) -> None:
    path = _write_opencap_pickle(tmp_path / "camera.pickle", cx=900.0, cy=1400.0)
    with pytest.raises(ValueError, match="inside image bounds"):
        load_opencap_calibration(path, width=720, height=1280)


def test_stage_02_uses_manifest_calibration_by_default(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_standard_dirs()
    write_json(registry.ensure_parent("video_info"), {"width": 720, "height": 1280})
    calibration = _write_opencap_pickle(tmp_path / "camera.pickle", image_size=(1280, 720))

    result = stage_02_assume_camera.run(
        tmp_path,
        {"manifest_summary": {"calibration": {"intrinsics_extrinsics": str(calibration)}}},
        force=False,
    )

    camera = read_json(registry.get("camera_assumed"))
    qc = read_json(registry.get("camera_qc"))
    assert result["camera_source"] == "opencap_manifest"
    assert camera["mode"] == "opencap_manifest"
    assert camera["is_assumed"] is False
    assert qc["distortion_enabled"] is True


def test_stage_02_manual_camera_overrides_manifest_calibration(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_standard_dirs()
    write_json(registry.ensure_parent("video_info"), {"width": 720, "height": 1280})
    calibration = _write_opencap_pickle(tmp_path / "camera.pickle", image_size=(1280, 720))
    manual = tmp_path / "manual_camera.yaml"
    manual.write_text(
        yaml.safe_dump({"camera": {"fx": 700.0, "fy": 701.0, "cx": 360.0, "cy": 640.0, "width": 720, "height": 1280}}),
        encoding="utf-8",
    )

    result = stage_02_assume_camera.run(
        tmp_path,
        {
            "camera_override": str(manual),
            "manifest_summary": {"calibration": {"intrinsics_extrinsics": str(calibration)}},
        },
        force=False,
    )

    camera = read_json(registry.get("camera_assumed"))
    assert result["camera_source"] == "manual"
    assert camera["mode"] == "manual"
    assert camera["fx"] == 700.0


def _write_opencap_pickle(path: Path, image_size: tuple[int, int] = (720, 1280), cx: float = 366.0, cy: float = 638.0) -> Path:
    data = {
        "intrinsicMat": np.array([[913.0, 0.0, cx], [0.0, 914.0, cy], [0.0, 0.0, 1.0]], dtype=float),
        "distortion": np.array([[0.1, -0.2, 0.001, 0.002, 0.3]], dtype=float),
        "imageSize": np.array([[float(image_size[0])], [float(image_size[1])]], dtype=float),
        "rotation": np.eye(3, dtype=float),
        "translation": np.array([[1.0], [2.0], [3.0]], dtype=float),
        "rotation_EulerAngles": np.array([[0.1], [0.2], [0.3]], dtype=float),
    }
    with path.open("wb") as f:
        pickle.dump(data, f)
    return path
