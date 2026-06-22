from __future__ import annotations

import json
import pickle
import subprocess
from pathlib import Path
from typing import Any

from monocap_v2.core.logging_utils import read_json


def run_pose3d(video_path: Path, pose2d: dict | None, camera: dict, subject: dict, cfg: dict) -> dict:
    sam_cfg = cfg.get("config", {}).get("sam3d_body", {})
    repo_root = Path(str(cfg.get("repo_root") or Path.cwd())).resolve()
    run_dir = Path(str(cfg.get("run_dir") or Path.cwd())).resolve()
    work_root = run_dir / "pose3d_initial" / str(sam_cfg.get("output_dir_name", "sam3d_body_work"))
    work_root.mkdir(parents=True, exist_ok=True)

    source_video = _select_source_video(video_path, cfg, sam_cfg, repo_root)
    output_pose = work_root / "sam3d_body_pose.pkl"
    output_report = work_root / "sam3d_body_report.json"
    settings = {
        **sam_cfg,
        "repo_path": str(_resolve_path(sam_cfg.get("repo_path", "external/sam-3d-body"), repo_root)),
        "checkpoint_path": str(_resolve_path(sam_cfg.get("checkpoint_path", "models/sam3d_body/sam-3d-body-dinov3/model.ckpt"), repo_root)),
        "mhr_model_path": str(
            _resolve_path(sam_cfg.get("mhr_model_path", "models/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt"), repo_root)
        ),
        "source_video": str(source_video),
        "inference_video": str(source_video),
    }

    cmd = [
        str(_sam3d_python(sam_cfg)),
        str(Path(__file__).resolve().parents[1] / "scripts" / "sam3d_body_worker.py"),
        "--video",
        str(source_video),
        "--output-pose",
        str(output_pose),
        "--report",
        str(output_report),
        "--settings-json",
        json.dumps(settings),
        "--camera-json",
        json.dumps(camera),
        "--subject-json",
        json.dumps(subject),
        "--cfg-json",
        json.dumps(_json_safe_cfg(cfg)),
    ]
    if bool(cfg.get("force")) or not output_pose.exists():
        completed = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
        if completed.returncode != 0:
            report = read_json(output_report) if output_report.exists() else {}
            reason = report.get("error") or completed.stderr.strip() or f"SAM3D worker exited with code {completed.returncode}"
            raise RuntimeError(f"SAM3D Body backend failed: {reason}")
    if not output_pose.exists():
        raise RuntimeError(f"SAM3D Body worker did not write pose artifact: {output_pose}")
    with output_pose.open("rb") as f:
        artifact = pickle.load(f)
    return artifact


def _select_source_video(video_path: Path, cfg: dict, sam_cfg: dict, repo_root: Path) -> Path:
    video_field = str(sam_cfg.get("video_field") or cfg.get("video_field") or "video_sync")
    trial = cfg.get("trial") or {}
    if trial.get(video_field):
        candidate = _resolve_path(trial[video_field], repo_root)
        if candidate.exists():
            return candidate
    return Path(video_path).resolve()


def _sam3d_python(sam_cfg: dict) -> Path:
    return Path(str(sam_cfg.get("python") or "/home/denik/miniconda3/envs/monocap-sam3d-body/bin/python"))


def _resolve_path(value: Any, repo_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _json_safe_cfg(cfg: dict) -> dict:
    def convert(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value

    return convert(cfg)
