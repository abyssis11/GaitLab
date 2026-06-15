from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from monocap_v2.core.logging_utils import read_json, read_yaml, write_json, write_yaml
from monocap_v2.core.video_io import read_video_frames_bgr, write_video_frames_bgr


def run_pose3d(video_path: Path, pose2d: dict | None, camera: dict, subject: dict, cfg: dict) -> dict:
    """Run the existing RTMW3D scripts and adapt their metric JSONL output.

    RTMW3D currently lives in the original GaitLab scripts and requires the
    `gaitlab` conda environment. This adapter keeps monocap_v2 backend-neutral
    by shelling out to that environment, writing all generated RTMW3D files into
    the run directory, and converting `preds_metric.jsonl` to `pose3d_initial`.
    """
    repo_root = Path(str(cfg.get("repo_root") or Path.cwd())).resolve()
    run_dir = Path(str(cfg.get("run_dir") or Path.cwd())).resolve()
    rtmw_cfg = cfg.get("config", {}).get("rtmw3d", {})
    trial_id = str(cfg.get("trial_id") or (cfg.get("trial") or {}).get("id") or "trial")
    work_root = run_dir / "pose3d_initial" / str(rtmw_cfg.get("output_dir_name", "rtmw3d_work"))
    if bool(cfg.get("force")) and bool(rtmw_cfg.get("clear_cache_on_force", True)) and work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    video_field = str(rtmw_cfg.get("video_field") or cfg.get("video_field") or "video_sync")
    source_video = _select_source_video(video_path, cfg, video_field, repo_root)
    prepared_video = _prepare_inference_video(source_video, work_root, rtmw_cfg)
    inference_video = Path(str(prepared_video["path"]))
    manifest_path, paths_path = _write_local_manifest(run_dir, work_root, cfg, trial_id, video_field, inference_video)
    paths = _rtmw3d_paths(work_root, trial_id)

    _ensure_rtmw3d_outputs(repo_root, manifest_path, paths_path, paths, trial_id, video_field, subject, rtmw_cfg, force=bool(cfg.get("force")))
    meta = read_json(paths["meta"])
    predictions = read_rtmw3d_metric_predictions(paths["preds_metric"], meta)
    fps = float(meta.get("fps") or cfg.get("video_info", {}).get("fps") or 30.0)
    joints_m = predictions["joints_3d_m"].astype(np.float32)
    xy = predictions["xy"].astype(np.float32)
    confidence = predictions["confidence"].astype(np.float32)
    joint_names = [str(name) for name in predictions["joint_names"]]

    return {
        "representation": "joints",
        "backend": "rtmw3d",
        "fps": fps,
        "units": "m",
        "joint_names": joint_names,
        "joints_3d": joints_m,
        "pose2d": {
            "xy": xy,
            "confidence": confidence,
            "names": joint_names,
            "fps": fps,
            "backend": "rtmw3d",
        },
        "camera": {
            "intrinsics": camera,
            "extrinsics": None,
            "is_assumed": bool(camera.get("is_assumed", True)),
        },
        "source_video": str(source_video),
        "subject": {"id": subject.get("id"), "height_m": subject.get("height_m"), "mass_kg": subject.get("mass_kg")},
        "backend_meta": {
            "coordinate_space": "rtmw3d_height_scaled_root_relative",
            "scale_note": "RTMW3D keypoints are scaled to metric units from subject height; global camera/world translation is not estimated.",
            "source_video": str(source_video),
            "inference_video": str(inference_video),
            "inference_video_was_materialized": bool(prepared_video.get("materialized")),
            "video_read_report": prepared_video.get("read_report"),
            "video_field": video_field,
            "work_root": str(work_root),
            "manifest": str(manifest_path),
            "paths": str(paths_path),
            "preds": str(paths["preds"]),
            "preds_metric": str(paths["preds_metric"]),
            "meta": str(paths["meta"]),
            "frame_indices": predictions["frame_indices"].astype(int),
            "time_s": predictions["time_s"].astype(np.float32),
            "valid_person_frames": int(predictions["valid_person_frames"]),
            "mean_confidence": float(np.nanmean(confidence)) if confidence.size else 0.0,
            "python": str(_rtmw3d_python(rtmw_cfg)),
            "config": str(_resolve_path(rtmw_cfg.get("config_path", "models/rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288.py"), repo_root)),
            "checkpoint": str(_resolve_path(rtmw_cfg.get("checkpoint_path", "models/rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth"), repo_root)),
            "max_frames": rtmw_cfg.get("max_frames"),
            "stride": int(rtmw_cfg.get("stride", 1)),
            "refine_pass": bool(rtmw_cfg.get("refine_pass", True)),
        },
    }


def read_rtmw3d_metric_predictions(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    names = [str(name) for name in meta.get("keypoint_names") or []]
    if not names:
        raise RuntimeError(f"RTMW3D meta has no keypoint_names: {path}")
    rows = list(_iter_jsonl(path))
    if not rows:
        raise RuntimeError(f"RTMW3D metric predictions are empty: {path}")
    joint_count = len(names)
    joints_mm = np.full((len(rows), joint_count, 3), np.nan, dtype=np.float32)
    xy = np.full((len(rows), joint_count, 2), np.nan, dtype=np.float32)
    confidence = np.zeros((len(rows), joint_count), dtype=np.float32)
    frame_indices = np.full((len(rows),), -1, dtype=np.int64)
    time_s = np.full((len(rows),), np.nan, dtype=np.float32)
    valid_person_frames = 0
    for idx, row in enumerate(rows):
        frame_indices[idx] = int(row.get("frame_index", idx))
        time_s[idx] = float(row.get("time_sec", idx / float(meta.get("fps") or 30.0)))
        person = _select_person(row.get("persons") or [])
        if not person:
            continue
        kps_mm = np.asarray(person.get("keypoints_xyz_mm"), dtype=np.float32)
        if kps_mm.ndim == 2 and kps_mm.shape[1] == 3:
            n = min(joint_count, kps_mm.shape[0])
            joints_mm[idx, :n] = kps_mm[:n]
            valid_person_frames += 1
        px = np.asarray(person.get("keypoints_px"), dtype=np.float32)
        if px.ndim == 2 and px.shape[1] == 2:
            n = min(joint_count, px.shape[0])
            xy[idx, :n] = px[:n]
        scores = person.get("keypoint_scores")
        if scores is not None:
            scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
            n = min(joint_count, scores_arr.shape[0])
            confidence[idx, :n] = scores_arr[:n]
        else:
            confidence[idx, :] = float(person.get("mean_score", 1.0))
    confidence[~np.isfinite(xy).all(axis=2)] = 0.0
    return {
        "joint_names": names,
        "joints_3d_m": joints_mm / 1000.0,
        "xy": xy,
        "confidence": confidence,
        "frame_indices": frame_indices,
        "time_s": time_s,
        "valid_person_frames": valid_person_frames,
    }


def _ensure_rtmw3d_outputs(
    repo_root: Path,
    manifest_path: Path,
    paths_path: Path,
    paths: dict[str, Path],
    trial_id: str,
    video_field: str,
    subject: dict,
    rtmw_cfg: dict,
    force: bool,
) -> None:
    if force or not paths["preds"].exists() or not paths["meta"].exists():
        cmd = [
            str(_rtmw3d_python(rtmw_cfg)),
            str(repo_root / "src" / "pose" / "rtmw3d_pose_estimation.py"),
            "--manifest",
            str(manifest_path),
            "--paths",
            str(paths_path),
            "--trials",
            trial_id,
            "--video-field",
            video_field,
            "--device",
            str(rtmw_cfg.get("device", "cuda:0")),
            "--config",
            str(_resolve_path(rtmw_cfg.get("config_path", "models/rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288.py"), repo_root)),
            "--checkpoint",
            str(_resolve_path(rtmw_cfg.get("checkpoint_path", "models/rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth"), repo_root)),
            "--metainfo-from-file",
            str(_resolve_path(rtmw_cfg.get("metainfo_from_file", "external/datasets_config/h3wb.py"), repo_root)),
            "--stride",
            str(int(rtmw_cfg.get("stride", 1))),
            "--print-every",
            str(int(rtmw_cfg.get("print_every", 50))),
        ]
        if bool(rtmw_cfg.get("refine_pass", True)):
            cmd.append("--refine-pass")
        if bool(rtmw_cfg.get("amp", False)):
            cmd.append("--amp")
        _run_command(cmd, repo_root, _rtmw3d_env(repo_root))
    if force or not paths["preds_metric"].exists():
        cmd = [
            str(_rtmw3d_python(rtmw_cfg)),
            str(repo_root / "src" / "pose" / "rtmw3d_scale_from_height.py"),
            "--manifest",
            str(manifest_path),
            "--paths",
            str(paths_path),
            "--trial",
            trial_id,
            "--preds",
            str(paths["preds"]),
            "--meta",
            str(paths["meta"]),
            "--out",
            str(paths["preds_metric"]),
        ]
        height_m = subject.get("height_m")
        if height_m:
            cmd.extend(["--height-mm", f"{float(height_m) * 1000.0:.6f}"])
        if bool(rtmw_cfg.get("export_trc", False)):
            cmd.append("--trc")
            cmd.extend(["--trc-out", str(paths["trc"])])
        _run_command(cmd, repo_root, _rtmw3d_env(repo_root))
    for key in ("preds_metric", "meta"):
        if not paths[key].exists():
            raise RuntimeError(f"RTMW3D output is missing after subprocess run: {paths[key]}")


def _write_local_manifest(run_dir: Path, work_root: Path, cfg: dict, trial_id: str, video_field: str, inference_video: Path) -> tuple[Path, Path]:
    source_manifest = run_dir / "manifest_resolved.yaml"
    manifest = read_yaml(source_manifest) if source_manifest.exists() else _minimal_manifest(cfg)
    manifest = dict(manifest)
    manifest["output_dir"] = str(work_root)
    trial_subset = str(cfg.get("trial_subset") or "healthy")
    trial = dict(cfg.get("trial") or {"id": trial_id})
    trial["id"] = trial_id
    trial[video_field] = str(inference_video)
    manifest.setdefault("paths", {"root": str(run_dir)})
    if "root" not in manifest["paths"]:
        manifest["paths"]["root"] = str(run_dir)
    trials = manifest.setdefault("trials", {})
    trials[trial_subset] = [trial]
    for subset in list(trials):
        if subset != trial_subset:
            trials[subset] = []
    manifest_path = work_root / "manifest_rtmw3d.yaml"
    write_yaml(manifest_path, manifest)

    source_paths = Path(str(cfg.get("paths_path") or ""))
    paths_cfg = read_yaml(source_paths) if source_paths.exists() else {"datasets": {}}
    paths_cfg.setdefault("datasets", {})
    paths_cfg["datasets"].setdefault("opencap_root", str(run_dir))
    paths_cfg["datasets"].setdefault("gpjatk_root", str(run_dir))
    paths_path = work_root / "paths_rtmw3d.yaml"
    write_yaml(paths_path, paths_cfg)
    return manifest_path, paths_path


def _minimal_manifest(cfg: dict) -> dict:
    trial = dict(cfg.get("trial") or {})
    return {
        "subject_id": (cfg.get("manifest_summary") or {}).get("subject_id", "subject"),
        "session": (cfg.get("manifest_summary") or {}).get("session", "Session"),
        "camera": (cfg.get("manifest_summary") or {}).get("camera", "Cam"),
        "session_metadata": None,
        "fps_video": "auto",
        "paths": {"root": str(Path(str(cfg.get("run_dir") or ".")).resolve())},
        "trials": {str(cfg.get("trial_subset") or "healthy"): [trial]},
    }


def _rtmw3d_paths(work_root: Path, trial_id: str) -> dict[str, Path]:
    trial_root = work_root / trial_id
    return {
        "trial_root": trial_root,
        "rtmw3d_dir": trial_root / "rtmw3d",
        "meta": trial_root / "meta.json",
        "preds": trial_root / "rtmw3d" / "preds.jsonl",
        "preds_metric": trial_root / "rtmw3d" / "preds_metric.jsonl",
        "trc": trial_root / "rtmw3d" / "rtmw3d_metric.trc",
    }


def _select_source_video(video_path: Path, cfg: dict, video_field: str, repo_root: Path) -> Path:
    trial = cfg.get("trial") or {}
    if trial.get(video_field):
        candidate = _resolve_path(trial[video_field], repo_root)
        if candidate.exists():
            return candidate
    return Path(video_path).resolve()


def _prepare_inference_video(source_video: Path, work_root: Path, rtmw_cfg: dict) -> dict[str, Any]:
    max_frames = rtmw_cfg.get("max_frames")
    max_frames_int = int(max_frames) if max_frames is not None else None
    max_frames_arg = max_frames_int if max_frames_int is not None and max_frames_int > 0 else None
    frames, report = read_video_frames_bgr(source_video, max_frames=max_frames_arg)
    needs_materialized = bool(max_frames_arg) or bool(report.get("random_access_recovered_count")) or bool(report.get("missing_frame_indices"))
    report_path = work_root / "input" / "video_read_report.json"
    write_json(report_path, report)
    if not needs_materialized:
        return {"path": str(source_video), "materialized": False, "read_report": report, "read_report_path": str(report_path)}
    suffix = f"_frames{max_frames_arg:04d}" if max_frames_arg else "_robust"
    target = work_root / "input" / f"{source_video.stem}{suffix}.mp4"
    if target.exists():
        return {"path": str(target), "materialized": True, "read_report": report, "read_report_path": str(report_path)}
    width = int(report.get("width") or 0)
    height = int(report.get("height") or 0)
    size = (width, height) if width > 0 and height > 0 else None
    write_video_frames_bgr(frames, target, float(report.get("fps") or 30.0), size=size)
    return {"path": str(target), "materialized": True, "read_report": report, "read_report_path": str(report_path)}


def _select_person(persons: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not persons:
        return None
    return max(persons, key=lambda p: float(p.get("mean_score", 0.0)))


def _iter_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "RTMW3D subprocess failed with exit code "
            f"{proc.returncode}: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _rtmw3d_env(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    parts = [str(repo_root / "src"), str(repo_root / "external")]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-monocap-v2")
    return env


def _rtmw3d_python(rtmw_cfg: dict) -> Path:
    return Path(str(rtmw_cfg.get("python", "/home/denik/miniconda3/envs/gaitlab/bin/python")))


def _resolve_path(value: Any, repo_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path
