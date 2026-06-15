from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from monocap_v2.core.smpl_model import SMPL_FOOT_VERTEX_NAMES, regress_smpl_24_joints

FEET_JOINT_NAMES = list(SMPL_FOOT_VERTEX_NAMES)


def run_pose3d(video_path: Path, pose2d: dict | None, camera: dict, subject: dict, cfg: dict) -> dict:
    wham_cfg = cfg.get("config", {}).get("wham", {})
    repo_root = Path(str(cfg.get("repo_root") or Path.cwd())).resolve()
    wham_repo = _resolve_path(wham_cfg.get("repo_path", "external/WHAM"), repo_root)
    if not (wham_repo / "wham_api.py").exists():
        raise RuntimeError(f"WHAM API not found under {wham_repo}. Run the WHAM prerequisite setup first.")

    output_dir = _wham_output_dir(cfg, wham_cfg)
    if bool(cfg.get("force")) and bool(wham_cfg.get("clear_cache_on_force", True)) and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_video = _select_source_video(video_path, cfg, wham_cfg, repo_root)
    inference_video = _prepare_inference_video(source_video, output_dir, wham_cfg)
    _, source_frame_count, _, _ = _video_metadata(source_video, cfg)
    fps, inference_frame_count, width, height = _video_metadata(inference_video, cfg)

    local_only = bool(wham_cfg.get("local_only", wham_cfg.get("estimate_local_only", True)))
    run_global_requested = not local_only
    results, tracking_results, _slam_results, dpvo_available = _run_wham_api(
        wham_repo,
        inference_video,
        output_dir,
        run_global=run_global_requested,
        visualize=bool(wham_cfg.get("visualize", False)),
    )

    track_id, subject_result = _select_primary_track(results)
    frame_ids = _frame_ids(subject_result)
    raw_frame_ids = frame_ids + int(wham_cfg.get("start_frame") or 0)
    vertices = _required_array(subject_result, "verts_cam").astype(np.float32)
    joints, joint_names, joint_regression_meta = _regress_joints_from_vertices(vertices, wham_repo, cfg)
    selected_pose2d = _pose2d_from_tracking(tracking_results, track_id, frame_ids, fps)

    artifact = {
        "representation": "hybrid",
        "backend": "wham",
        "fps": fps,
        "units": "m",
        "joint_names": joint_names,
        "joints_3d": joints.astype(np.float32),
        "pose2d": selected_pose2d,
        "smpl": {
            "model_type": "smpl",
            "gender": _subject_gender(subject),
            "betas": _optional_array(subject_result, "betas"),
            "body_pose": _rotmats_to_axis_angle(_required_array(subject_result, "poses_body")),
            "global_orient": _rotmats_to_axis_angle(_required_array(subject_result, "poses_root_cam")).reshape(-1, 3),
            "transl": _optional_array(subject_result, "trans_world"),
            "vertices": vertices,
            "rotation_representation": "axis_angle",
            "vertices_coordinate_space": "wham_camera_local",
        },
        "camera": {
            "intrinsics": camera,
            "extrinsics": None,
            "is_assumed": bool(camera.get("is_assumed", True)),
        },
        "source_video": str(source_video),
        "subject": {"id": subject.get("id"), "height_m": subject.get("height_m"), "mass_kg": subject.get("mass_kg")},
        "backend_meta": {
            "repo_path": str(wham_repo),
            "output_dir": str(output_dir),
            "source_video": str(source_video),
            "inference_video": str(inference_video),
            "video_field": wham_cfg.get("video_field"),
            "source_frame_count": int(source_frame_count),
            "inference_frame_count": int(inference_frame_count),
            "fps": float(fps),
            "width": int(width),
            "height": int(height),
            "selected_track_id": str(track_id),
            "frame_ids": frame_ids.astype(int),
            "raw_frame_ids": raw_frame_ids.astype(int),
            "raw_video_time_s": (raw_frame_ids.astype(float) / float(fps)).astype(np.float32),
            "wham_relative_time_s": ((raw_frame_ids - raw_frame_ids[0]).astype(float) / float(fps)).astype(np.float32),
            "run_global_requested": bool(run_global_requested),
            "dpvo_available": bool(dpvo_available),
            "local_only": bool(local_only or not dpvo_available),
            "coordinate_space": "WHAM local/camera coordinates; DPVO global trajectory is not required for this profile.",
            "max_frames": wham_cfg.get("max_frames"),
            "start_frame": int(wham_cfg.get("start_frame") or 0),
            "joint_regression": joint_regression_meta,
        },
    }
    if artifact["smpl"]["transl"] is None:
        artifact["smpl"]["transl"] = np.zeros((vertices.shape[0], 3), dtype=np.float32)
    if selected_pose2d is None:
        artifact.pop("pose2d")
    return artifact


def _select_source_video(video_path: Path, cfg: dict, wham_cfg: dict, repo_root: Path) -> Path:
    video_field = wham_cfg.get("video_field")
    trial = cfg.get("trial") or {}
    if video_field and trial.get(video_field):
        candidate = _resolve_path(trial[video_field], repo_root)
        if candidate.exists():
            return candidate
    return Path(video_path).resolve()


def _wham_output_dir(cfg: dict, wham_cfg: dict) -> Path:
    run_dir = Path(str(cfg.get("run_dir") or Path.cwd())).resolve()
    return run_dir / "pose3d_initial" / str(wham_cfg.get("output_dir_name", "wham_work"))


def _prepare_inference_video(source_video: Path, output_dir: Path, wham_cfg: dict) -> Path:
    max_frames = wham_cfg.get("max_frames")
    max_frames_int = int(max_frames) if max_frames is not None else None
    start_frame = int(wham_cfg.get("start_frame") or 0)
    if (max_frames_int is None or max_frames_int <= 0) and start_frame <= 0:
        return source_video
    frames_label = "full" if max_frames_int is None or max_frames_int <= 0 else f"{max_frames_int:04d}"
    target = output_dir / f"{source_video.stem}_start{start_frame:06d}_frames{frames_label}.mp4"
    if target.exists():
        return target
    _trim_video(source_video, target, start_frame=start_frame, max_frames=max_frames_int)
    return target


def _trim_video(source: Path, target: Path, start_frame: int, max_frames: int | None) -> None:
    import cv2

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open WHAM source video: {source}")
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create WHAM trim video: {target}")
    frames = 0
    try:
        while True:
            if max_frames is not None and max_frames > 0 and frames >= max_frames:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            writer.write(frame)
            frames += 1
    finally:
        cap.release()
        writer.release()
    if frames == 0:
        raise RuntimeError(f"No frames were read from WHAM source video: {source}")


def _video_metadata(video: Path, cfg: dict) -> tuple[float, int, int, int]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            info = cfg.get("video_info", {})
            return float(info.get("fps") or 30.0), int(info.get("frame_count") or 0), int(info.get("width") or 0), int(info.get("height") or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or cfg.get("video_info", {}).get("fps") or 30.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return fps, frame_count, width, height
    finally:
        cap.release()


def _run_wham_api(
    wham_repo: Path,
    video: Path,
    output_dir: Path,
    run_global: bool,
    visualize: bool,
) -> tuple[dict, dict, Any, bool]:
    old_cwd = Path.cwd()
    inserted = False
    if str(wham_repo) not in sys.path:
        sys.path.insert(0, str(wham_repo))
        inserted = True
    os.chdir(wham_repo)
    try:
        from wham_api import WHAM_API  # type: ignore
        import wham_api as wham_api_module  # type: ignore
        from lib.data.datasets import CustomDataset  # type: ignore

        if not hasattr(CustomDataset, "prefix"):
            CustomDataset.prefix = ""

        model = WHAM_API()
        results, tracking_results, slam_results = model(
            str(video),
            output_dir=str(output_dir),
            calib=None,
            run_global=bool(run_global),
            visualize=bool(visualize),
        )
        dpvo_available = bool(getattr(wham_api_module, "_run_global", False))
        return dict(results), dict(tracking_results), slam_results, dpvo_available
    finally:
        os.chdir(old_cwd)
        if inserted:
            try:
                sys.path.remove(str(wham_repo))
            except ValueError:
                pass


def _select_primary_track(results: dict) -> tuple[Any, dict]:
    if not results:
        raise RuntimeError("WHAM returned no subject tracks.")

    def sort_key(item: tuple[Any, Any]) -> tuple[int, str]:
        track_id, values = item
        frame_count = len(_frame_ids(values)) if isinstance(values, dict) else 0
        return (-frame_count, str(track_id))

    track_id, values = sorted(results.items(), key=sort_key)[0]
    if not isinstance(values, dict):
        raise RuntimeError(f"WHAM subject track {track_id} is not a mapping.")
    return track_id, values


def _frame_ids(subject_result: dict) -> np.ndarray:
    frame_ids = _to_numpy(subject_result.get("frame_id"))
    if frame_ids is None:
        verts = _to_numpy(subject_result.get("verts_cam"))
        count = int(verts.shape[0]) if verts is not None and verts.ndim >= 1 else 0
        return np.arange(count, dtype=int)
    return np.asarray(frame_ids, dtype=int).reshape(-1)


def _required_array(mapping: dict, key: str) -> np.ndarray:
    arr = _optional_array(mapping, key)
    if arr is None:
        raise RuntimeError(f"WHAM result missing required key: {key}")
    return arr


def _optional_array(mapping: dict, key: str) -> np.ndarray | None:
    value = mapping.get(key)
    if value is None:
        return None
    return np.asarray(_to_numpy(value), dtype=np.float32)


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _regress_joints_from_vertices(vertices: np.ndarray, wham_repo: Path, cfg: dict | None = None) -> tuple[np.ndarray, list[str], dict]:
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise RuntimeError(f"WHAM vertices must have shape [T, V, 3], got {vertices.shape}")
    names: list[str] = []
    chunks: list[np.ndarray] = []
    sources: list[dict] = []

    try:
        smpl_joints = regress_smpl_24_joints(vertices, cfg or {})
        chunks.append(smpl_joints["joints"])
        names.extend(smpl_joints["joint_names"])
        sources.append(
            {
                "type": "smpl_24_J_regressor",
                "source": smpl_joints["source"],
                "source_type": smpl_joints["source_type"],
                "joint_count": len(smpl_joints["joint_names"]),
            }
        )
    except Exception as exc:
        wham_regressor = _load_regressor(wham_repo / "dataset" / "body_models" / "J_regressor_wham.npy")
        if wham_regressor is not None:
            chunks.append(np.einsum("jv,tvc->tjc", wham_regressor, vertices, optimize=True))
            names.extend(_fallback_wham_joint_names(wham_regressor.shape[0]))
            sources.append(
                {
                    "type": "wham_regressor_generic",
                    "source": str(wham_repo / "dataset" / "body_models" / "J_regressor_wham.npy"),
                    "joint_count": int(wham_regressor.shape[0]),
                    "warning": f"SMPL 24-joint regressor unavailable; WHAM rows are intentionally generic: {exc}",
                }
            )

    feet_regressor = _load_regressor(wham_repo / "dataset" / "body_models" / "J_regressor_feet.npy")
    if feet_regressor is not None:
        chunks.append(np.einsum("jv,tvc->tjc", feet_regressor, vertices, optimize=True))
        names.extend(FEET_JOINT_NAMES[: feet_regressor.shape[0]])
        sources.append(
            {
                "type": "smpl_vertex_foot_landmarks",
                "source": str(wham_repo / "dataset" / "body_models" / "J_regressor_feet.npy"),
                "joint_count": int(feet_regressor.shape[0]),
                "names": FEET_JOINT_NAMES[: feet_regressor.shape[0]],
            }
        )

    if not chunks:
        raise RuntimeError(f"No WHAM joint regressors found under {wham_repo / 'dataset' / 'body_models'}")
    joints = np.concatenate(chunks, axis=1)
    if len(names) != joints.shape[1]:
        names = [f"wham_joint_{idx:02d}" for idx in range(joints.shape[1])]
    return joints.astype(np.float32), names, {"sources": sources}


def _load_regressor(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    regressor = np.asarray(np.load(path), dtype=np.float32)
    if regressor.ndim != 2:
        raise RuntimeError(f"Invalid WHAM joint regressor shape in {path}: {regressor.shape}")
    return regressor


def _fallback_wham_joint_names(count: int) -> list[str]:
    return [f"wham_joint_{idx:02d}" for idx in range(count)]


def _pose2d_from_tracking(tracking_results: dict, track_id: Any, frame_ids: np.ndarray, fps: float) -> dict | None:
    tracking = _lookup_track(tracking_results, track_id)
    if not isinstance(tracking, dict):
        return None
    keypoints = _to_numpy(tracking.get("keypoints"))
    tracking_frame_ids = _to_numpy(tracking.get("frame_id"))
    if keypoints is None or keypoints.ndim != 3 or keypoints.shape[-1] < 2:
        return None
    if tracking_frame_ids is None:
        tracking_frame_ids = np.arange(keypoints.shape[0], dtype=int)
    tracking_frame_ids = np.asarray(tracking_frame_ids, dtype=int).reshape(-1)
    by_frame = {int(fid): idx for idx, fid in enumerate(tracking_frame_ids[: keypoints.shape[0]])}

    xy = np.full((len(frame_ids), keypoints.shape[1], 2), np.nan, dtype=np.float32)
    confidence = np.zeros((len(frame_ids), keypoints.shape[1]), dtype=np.float32)
    for out_idx, frame_id in enumerate(frame_ids.astype(int).tolist()):
        kp_idx = by_frame.get(int(frame_id))
        if kp_idx is None:
            continue
        frame_kps = np.asarray(keypoints[kp_idx], dtype=np.float32)
        xy[out_idx] = frame_kps[:, :2]
        if frame_kps.shape[1] >= 3:
            confidence[out_idx] = frame_kps[:, 2]
        else:
            confidence[out_idx] = np.isfinite(frame_kps[:, :2]).all(axis=1).astype(np.float32)
    return {
        "xy": xy,
        "confidence": confidence,
        "names": [f"wham_kp_{idx:02d}" for idx in range(xy.shape[1])],
        "fps": float(fps),
        "backend": "wham",
    }


def _lookup_track(mapping: dict, track_id: Any) -> Any:
    if track_id in mapping:
        return mapping[track_id]
    str_id = str(track_id)
    if str_id in mapping:
        return mapping[str_id]
    try:
        int_id = int(track_id)
    except Exception:
        int_id = None
    if int_id is not None and int_id in mapping:
        return mapping[int_id]
    return None


def _rotmats_to_axis_angle(rotmats: np.ndarray) -> np.ndarray:
    arr = np.asarray(rotmats, dtype=np.float64)
    if arr.shape[-2:] != (3, 3):
        # Already axis-angle-like. Keep vector groups explicit when possible.
        if arr.shape[-1] == 3:
            return arr.astype(np.float32)
        if arr.shape[-1] % 3 == 0:
            return arr.reshape(*arr.shape[:-1], -1, 3).astype(np.float32)
        return arr.astype(np.float32)
    flat = arr.reshape(-1, 3, 3)
    out = np.zeros((flat.shape[0], 3), dtype=np.float64)
    for i, rot in enumerate(flat):
        trace = float(np.trace(rot))
        cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        angle = float(np.arccos(cos_angle))
        if angle < 1e-8:
            continue
        if np.pi - angle < 1e-5:
            axis = np.sqrt(np.maximum(np.diag(rot) + 1.0, 0.0) / 2.0)
            axis[0] = np.copysign(axis[0], rot[2, 1] - rot[1, 2])
            axis[1] = np.copysign(axis[1], rot[0, 2] - rot[2, 0])
            axis[2] = np.copysign(axis[2], rot[1, 0] - rot[0, 1])
            norm = np.linalg.norm(axis)
            out[i] = angle * axis / norm if norm > 1e-8 else 0.0
            continue
        axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]], dtype=np.float64)
        axis /= 2.0 * np.sin(angle)
        out[i] = axis * angle
    return out.reshape(*arr.shape[:-2], 3).astype(np.float32)


def _subject_gender(subject: dict) -> str:
    value = str(subject.get("sex") or subject.get("gender") or "neutral").lower()
    if value.startswith("m"):
        return "male"
    if value.startswith("f"):
        return "female"
    return "neutral"


def _resolve_path(value: Any, repo_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()
