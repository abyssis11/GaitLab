from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from monocap_v2.core.manifest import default_run_name, find_trial, load_opencap_manifest


def test_opencap_manifest_resolution(tmp_path: Path) -> None:
    paths = tmp_path / "paths.yaml"
    manifest = tmp_path / "manifest.yaml"
    paths.write_text(
        yaml.safe_dump(
            {
                "datasets": {"opencap_root": str(tmp_path / "OpenCapDataset"), "gpjatk_root": str(tmp_path / "GPJATK")},
                "outputs_root": str(tmp_path / "outputs"),
            }
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        yaml.safe_dump(
            {
                "subject_id": "subject7",
                "session": "Session1",
                "camera": "Cam1",
                "session_metadata": "${paths.root}/sessionMetadata.yaml",
                "fps_video": "auto",
                "paths": {"root": "${datasets.opencap_root}/subject7"},
                "trials": {
                    "healthy": [{"id": "walking1", "video_sync": "${paths.root}/video.avi"}],
                    "pathological": [{"id": "walkingTS1", "video_sync": "${paths.root}/ts.avi"}],
                },
            }
        ),
        encoding="utf-8",
    )

    resolved = load_opencap_manifest(manifest, paths)
    assert resolved["paths"]["root"] == str(tmp_path / "OpenCapDataset" / "subject7")
    assert resolved["trials"]["healthy"][0]["video_sync"].endswith("subject7/video.avi")
    subset, trial = find_trial(resolved, "walkingTS1")
    assert subset == "pathological"
    assert trial["id"] == "walkingTS1"
    assert default_run_name(resolved, "walking1") == "subject7_Session1_Cam1_walking1"


def test_missing_trial_error(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        find_trial({"trials": {"healthy": [{"id": "walking1"}]}}, "missing")

