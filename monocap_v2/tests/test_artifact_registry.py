from __future__ import annotations

from pathlib import Path

import pytest

from monocap_v2.core.artifact_registry import ArtifactRegistry


def test_artifact_registry_paths(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    assert registry.get("video_info") == tmp_path / "video" / "video_info.json"
    assert registry.get("pose3d_initial") == tmp_path / "pose3d_initial" / "pose3d_initial.pkl"
    assert registry.get("mocap_validation") == tmp_path / "reports" / "mocap_validation.json"
    registry.ensure_standard_dirs()
    assert (tmp_path / "video").is_dir()
    assert (tmp_path / "reports").is_dir()


def test_artifact_registry_unknown_key(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    with pytest.raises(KeyError):
        registry.get("nope")
