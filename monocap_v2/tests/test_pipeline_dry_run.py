from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def test_pipeline_dry_run_creates_run_metadata(tmp_path: Path) -> None:
    paths = tmp_path / "paths.yaml"
    manifest = tmp_path / "manifest.yaml"
    data_root = tmp_path / "OpenCapDataset" / "subject7"
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
                "session_metadata": str(data_root / "sessionMetadata.yaml"),
                "fps_video": "auto",
                "paths": {"root": "${datasets.opencap_root}/subject7"},
                "calibration": {"intrinsics_extrinsics": "${paths.root}/camera.pickle"},
                "trials": {"healthy": [{"id": "walking1", "video_sync": "${paths.root}/missing.avi"}]},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "run"
    cmd = [
        sys.executable,
        "monocap_v2/run_pipeline.py",
        "--manifest",
        str(manifest),
        "--paths",
        str(paths),
        "--trial",
        "walking1",
        "--activity",
        "walking",
        "--out",
        str(out),
        "--dry-run",
    ]
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assert (out / "manifest_resolved.yaml").exists()
    assert (out / "run_config.yaml").exists()
    assert (out / "pipeline_state.json").exists()
    assert (out / "logs" / "pipeline.log").exists()


def test_pipeline_preset_overlay_cli_override_and_wham_max_frames(tmp_path: Path) -> None:
    paths, manifest = _write_minimal_manifest(tmp_path)
    out = tmp_path / "run"
    cmd = [
        sys.executable,
        "monocap_v2/run_pipeline.py",
        "--manifest",
        str(manifest),
        "--paths",
        str(paths),
        "--trial",
        "walking1",
        "--activity",
        "walking",
        "--out",
        str(out),
        "--dry-run",
        "--preset",
        "opencap_wham",
        "--backend-pose3d",
        "dummy",
        "--max-frames",
        "12",
    ]
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    run_config = yaml.safe_load((out / "run_config.yaml").read_text(encoding="utf-8"))
    assert run_config["preset"] == "opencap_wham"
    assert run_config["config"]["backends"]["pose3d"] == "dummy"
    assert run_config["config"]["wham"]["video_field"] == "video_raw"
    assert run_config["config"]["wham"]["max_frames"] == 12
    assert run_config["config"]["metrabs"]["max_frames"] == 12
    assert run_config["config"]["rtmw3d"]["max_frames"] == 12


def _write_minimal_manifest(tmp_path: Path) -> tuple[Path, Path]:
    paths = tmp_path / "paths.yaml"
    manifest = tmp_path / "manifest.yaml"
    data_root = tmp_path / "OpenCapDataset" / "subject7"
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
                "session_metadata": str(data_root / "sessionMetadata.yaml"),
                "fps_video": "auto",
                "paths": {"root": "${datasets.opencap_root}/subject7"},
                "calibration": {"intrinsics_extrinsics": "${paths.root}/camera.pickle"},
                "trials": {
                    "healthy": [
                        {
                            "id": "walking1",
                            "video_sync": "${paths.root}/sync.avi",
                            "video_raw": "${paths.root}/raw.avi",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return paths, manifest
