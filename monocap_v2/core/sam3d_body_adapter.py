from __future__ import annotations

from typing import Any

import numpy as np


MHR70_JOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe_tip",
    "left_small_toe_tip",
    "left_heel",
    "right_big_toe_tip",
    "right_small_toe_tip",
    "right_heel",
    "right_thumb_tip",
    "right_thumb_first_joint",
    "right_thumb_second_joint",
    "right_thumb_third_joint",
    "right_index_tip",
    "right_index_first_joint",
    "right_index_second_joint",
    "right_index_third_joint",
    "right_middle_tip",
    "right_middle_first_joint",
    "right_middle_second_joint",
    "right_middle_third_joint",
    "right_ring_tip",
    "right_ring_first_joint",
    "right_ring_second_joint",
    "right_ring_third_joint",
    "right_pinky_tip",
    "right_pinky_first_joint",
    "right_pinky_second_joint",
    "right_pinky_third_joint",
    "right_wrist",
    "left_thumb_tip",
    "left_thumb_first_joint",
    "left_thumb_second_joint",
    "left_thumb_third_joint",
    "left_index_tip",
    "left_index_first_joint",
    "left_index_second_joint",
    "left_index_third_joint",
    "left_middle_tip",
    "left_middle_first_joint",
    "left_middle_second_joint",
    "left_middle_third_joint",
    "left_ring_tip",
    "left_ring_first_joint",
    "left_ring_second_joint",
    "left_ring_third_joint",
    "left_pinky_tip",
    "left_pinky_first_joint",
    "left_pinky_second_joint",
    "left_pinky_third_joint",
    "left_wrist",
    "left_olecranon",
    "right_olecranon",
    "left_cubital_fossa",
    "right_cubital_fossa",
    "left_acromion",
    "right_acromion",
    "neck",
]

DEFAULT_CANONICAL_JOINT_INDICES = {
    "left_hip": 9,
    "right_hip": 10,
    "left_knee": 11,
    "right_knee": 12,
    "left_ankle": 13,
    "right_ankle": 14,
    "left_mtp": 15,
    "right_mtp": 18,
}

DEFAULT_MAPPING_NOTE = "Official SAM3D MHR70 lower-limb keypoint mapping; pelvis is derived downstream from the hip midpoint."
DEFAULT_MAPPING_SOURCE = "sam3d_mhr70_official"


def select_primary_detection(detections: list[dict[str, Any]], previous_center: np.ndarray | None = None) -> int | None:
    candidates: list[tuple[int, float, np.ndarray]] = []
    for idx, detection in enumerate(detections):
        bbox = _bbox_array(detection)
        if bbox is None:
            continue
        area = max(float(bbox[2] - bbox[0]), 0.0) * max(float(bbox[3] - bbox[1]), 0.0)
        center = np.asarray([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=float)
        candidates.append((idx, area, center))
    if not candidates:
        return 0 if detections else None
    if previous_center is None or not np.isfinite(previous_center).all():
        return sorted(candidates, key=lambda item: (-item[1], item[0]))[0][0]
    return sorted(candidates, key=lambda item: (float(np.linalg.norm(item[2] - previous_center)), -item[1], item[0]))[0][0]


def build_sam3d_pose_artifact(
    frame_results: list[dict[str, Any] | None],
    *,
    fps: float,
    camera: dict[str, Any],
    subject: dict[str, Any],
    cfg: dict[str, Any],
    source_video: str,
    inference_video: str,
    video_report: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sam_cfg = cfg.get("config", {}).get("sam3d_body", cfg.get("sam3d_body", {}))
    mapping_cfg = sam_cfg.get("canonical_joint_indices")
    if mapping_cfg is None:
        mapping_cfg = DEFAULT_CANONICAL_JOINT_INDICES
    mapping = {str(key): int(value) for key, value in dict(mapping_cfg).items()}
    if mapping == DEFAULT_CANONICAL_JOINT_INDICES:
        mapping_source = DEFAULT_MAPPING_SOURCE
        mapping_note = str(sam_cfg.get("mapping_note") or DEFAULT_MAPPING_NOTE)
    else:
        mapping_source = "config"
        mapping_note = str(sam_cfg.get("mapping_note") or "Configured SAM3D canonical joint index mapping.")

    selected, selection_report = _select_detections_over_time(frame_results)
    joint_count = _infer_joint_count(selected, mapping)
    mesh_vertex_count = _infer_vertex_count(selected)
    frame_count = len(frame_results)
    joints_raw = np.full((frame_count, joint_count, 3), np.nan, dtype=np.float32)
    pose2d_raw = np.full((frame_count, joint_count, 2), np.nan, dtype=np.float32)
    raw_names = _mhr_raw_names(joint_count)
    vertices = np.full((frame_count, mesh_vertex_count, 3), np.nan, dtype=np.float32) if mesh_vertex_count else None
    pred_cam_t = np.full((frame_count, 3), np.nan, dtype=np.float32)
    raw_metadata: list[dict[str, Any]] = []

    for frame_idx, detection in enumerate(selected):
        if detection is None:
            raw_metadata.append({"frame_index": frame_idx, "status": "missing"})
            continue
        joints3d = _first_array(detection, ("pred_keypoints_3d", "keypoints_3d", "pred_joint_coords", "joints_3d"))
        joints2d = _first_array(detection, ("pred_keypoints_2d", "keypoints_2d", "joints_2d"))
        if joints3d is not None:
            n = min(joints_raw.shape[1], joints3d.shape[0])
            joints_raw[frame_idx, :n, :] = joints3d[:n, :3]
        if joints2d is not None:
            n = min(pose2d_raw.shape[1], joints2d.shape[0])
            pose2d_raw[frame_idx, :n, :] = joints2d[:n, :2]
        frame_vertices = _first_array(detection, ("pred_vertices", "vertices", "verts"))
        if vertices is not None and frame_vertices is not None:
            n = min(vertices.shape[1], frame_vertices.shape[0])
            vertices[frame_idx, :n, :] = frame_vertices[:n, :3]
        cam_t = _first_array(detection, ("pred_cam_t", "cam_t", "camera_translation"))
        if cam_t is not None:
            flat = cam_t.reshape(-1)
            pred_cam_t[frame_idx, : min(3, flat.shape[0])] = flat[:3]
        raw_metadata.append(_metadata_for_detection(frame_idx, detection))

    joint_names, joints, pose2d = _canonical_first(joints_raw, pose2d_raw, raw_names, mapping)
    confidence = np.isfinite(pose2d).all(axis=2).astype(np.float32)
    faces = _faces_from_selected(selected)
    artifact: dict[str, Any] = {
        "representation": "joints",
        "backend": "sam3d_body",
        "fps": float(fps or 30.0),
        "units": "m",
        "joint_names": joint_names,
        "joints_3d": joints.astype(np.float32),
        "pose2d": {
            "xy": pose2d.astype(np.float32),
            "confidence": confidence,
            "names": joint_names,
            "fps": float(fps or 30.0),
            "backend": "sam3d_body",
        },
        "camera": {
            "intrinsics": camera,
            "extrinsics": None,
            "is_assumed": bool(camera.get("is_assumed", True)),
        },
        "source_video": source_video,
        "subject": {"id": subject.get("id"), "height_m": subject.get("height_m"), "mass_kg": subject.get("mass_kg")},
        "backend_meta": {
            "repo_path": sam_cfg.get("repo_path"),
            "checkpoint_path": sam_cfg.get("checkpoint_path"),
            "mhr_model_path": sam_cfg.get("mhr_model_path"),
            "source_video": source_video,
            "inference_video": inference_video,
            "video_field": sam_cfg.get("video_field"),
            "coordinate_space": "sam3d_body_mhr_camera_local",
            "mapping_source": mapping_source,
            "mapping_note": mapping_note,
            "canonical_joint_indices": mapping,
            "canonical_joint_mapping_source": mapping_source,
            "canonical_joint_mapping_note": mapping_note,
            "frame_indices": np.arange(frame_count, dtype=np.int64),
            "time_s": (np.arange(frame_count, dtype=np.float32) / float(fps or 30.0)).astype(np.float32),
            "selection": selection_report,
            "raw_frame_metadata": raw_metadata,
            "video_read_report": video_report or {},
            "max_frames": sam_cfg.get("max_frames"),
            "bbox_strategy": sam_cfg.get("bbox_strategy", "detector"),
            "detector_name": sam_cfg.get("detector_name", "vitdet"),
            "fov_name": sam_cfg.get("fov_name", "moge2"),
            "segmentor_name": sam_cfg.get("segmentor_name"),
        },
    }
    if vertices is not None:
        artifact["mesh"] = {
            "model_type": "mhr",
            "vertices": vertices.astype(np.float32),
            "faces": faces.astype(np.int32) if faces is not None else None,
            "pred_cam_t": pred_cam_t,
            "mhr_model_params": _collect_mhr_params(selected),
            "coordinate_space": "sam3d_body_mhr_camera_local",
            "metadata": {"mapping_note": mapping_note},
        }
    qc = sam3d_pose_qc(artifact)
    return artifact, qc


def sam3d_pose_qc(artifact: dict[str, Any]) -> dict[str, Any]:
    joints = np.asarray(artifact.get("joints_3d"), dtype=float)
    pose2d = artifact.get("pose2d") or {}
    confidence = np.asarray(pose2d.get("confidence"), dtype=float)
    mesh = artifact.get("mesh") or {}
    vertices = np.asarray(mesh.get("vertices"), dtype=float)
    meta = artifact.get("backend_meta") or {}
    selection = meta.get("selection") or {}
    canonical = list((meta.get("canonical_joint_indices") or {}).keys())
    missing_canonical = [name for name in canonical if name not in artifact.get("joint_names", [])]
    warnings = []
    if missing_canonical:
        warnings.append(f"Missing canonical SAM3D joints after mapping: {', '.join(missing_canonical)}")
    if selection.get("missing_frame_count"):
        warnings.append(f"SAM3D returned no person for {selection.get('missing_frame_count')} frame(s).")
    return {
        "stage": "pose3d_sam3d_body",
        "status": "warning" if warnings else "ok",
        "backend": "sam3d_body",
        "frames": int(joints.shape[0]) if joints.ndim == 3 else 0,
        "joints": int(joints.shape[1]) if joints.ndim == 3 else 0,
        "finite_joint_ratio": float(np.isfinite(joints).mean()) if joints.size else 0.0,
        "pose2d_mean_confidence": float(np.nanmean(confidence)) if confidence.size else 0.0,
        "detected_frame_count": int(selection.get("detected_frame_count") or 0),
        "missing_frame_count": int(selection.get("missing_frame_count") or 0),
        "mesh_available": vertices.ndim == 3 and vertices.shape[-1] == 3 and bool(np.isfinite(vertices).any()),
        "mesh_vertices": int(vertices.shape[1]) if vertices.ndim == 3 else 0,
        "mesh_faces": int(np.asarray(mesh.get("faces")).shape[0]) if mesh.get("faces") is not None else 0,
        "canonical_joint_coverage": {
            "configured": canonical,
            "available": [name for name in canonical if name in artifact.get("joint_names", [])],
            "missing": missing_canonical,
            "mapping_source": meta.get("mapping_source"),
            "mapping_note": meta.get("mapping_note"),
            "selected_indices": meta.get("canonical_joint_indices") or {},
        },
        "bbox_continuity": selection.get("bbox_continuity", {}),
        "checkpoint_path": meta.get("checkpoint_path"),
        "mhr_model_path": meta.get("mhr_model_path"),
        "coordinate_space": meta.get("coordinate_space"),
        "warnings": warnings,
    }


def _select_detections_over_time(frame_results: list[dict[str, Any] | None]) -> tuple[list[dict[str, Any] | None], dict[str, Any]]:
    selected: list[dict[str, Any] | None] = []
    previous_center: np.ndarray | None = None
    selected_indices: list[int | None] = []
    distances: list[float] = []
    for result in frame_results:
        detections = _detections(result)
        idx = select_primary_detection(detections, previous_center)
        selected_indices.append(idx)
        if idx is None:
            selected.append(None)
            distances.append(float("nan"))
            continue
        detection = detections[idx]
        selected.append(detection)
        center = _bbox_center(detection)
        distances.append(float(np.linalg.norm(center - previous_center)) if previous_center is not None and center is not None else float("nan"))
        previous_center = center if center is not None else previous_center
    missing = [idx for idx, detection in enumerate(selected) if detection is None]
    finite_distances = [value for value in distances if np.isfinite(value)]
    return selected, {
        "selected_detection_indices": selected_indices,
        "detected_frame_count": len(selected) - len(missing),
        "missing_frame_count": len(missing),
        "missing_frame_indices": missing,
        "bbox_continuity": {
            "mean_center_step_px": float(np.mean(finite_distances)) if finite_distances else None,
            "max_center_step_px": float(np.max(finite_distances)) if finite_distances else None,
        },
    }


def _detections(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not result:
        return []
    detections = result.get("detections")
    if isinstance(detections, list):
        return [item for item in detections if isinstance(item, dict)]
    people = result.get("people")
    if isinstance(people, list):
        return [item for item in people if isinstance(item, dict)]
    return [result]


def _canonical_first(
    joints_raw: np.ndarray,
    pose2d_raw: np.ndarray,
    raw_names: list[str],
    mapping: dict[str, int],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = []
    joints = []
    pose2d = []
    used = set()
    for canonical, idx in mapping.items():
        if 0 <= idx < joints_raw.shape[1]:
            names.append(canonical)
            joints.append(joints_raw[:, idx, :])
            pose2d.append(pose2d_raw[:, idx, :])
            used.add(idx)
    for idx, raw_name in enumerate(raw_names):
        if idx in used:
            continue
        names.append(raw_name)
        joints.append(joints_raw[:, idx, :])
        pose2d.append(pose2d_raw[:, idx, :])
    if not joints:
        raise RuntimeError("SAM3D Body produced no joints.")
    return names, np.stack(joints, axis=1), np.stack(pose2d, axis=1)


def _mhr_raw_names(joint_count: int) -> list[str]:
    names = []
    for idx in range(joint_count):
        if idx < len(MHR70_JOINT_NAMES):
            names.append(f"mhr70_{MHR70_JOINT_NAMES[idx]}")
        else:
            names.append(f"mhr_{idx}")
    return names


def _infer_joint_count(selected: list[dict[str, Any] | None], mapping: dict[str, int]) -> int:
    count = max(mapping.values(), default=-1) + 1
    for detection in selected:
        if detection is None:
            continue
        arr = _first_array(detection, ("pred_keypoints_3d", "keypoints_3d", "pred_joint_coords", "joints_3d"))
        if arr is not None:
            count = max(count, int(arr.shape[0]))
    return count


def _infer_vertex_count(selected: list[dict[str, Any] | None]) -> int:
    count = 0
    for detection in selected:
        if detection is None:
            continue
        arr = _first_array(detection, ("pred_vertices", "vertices", "verts"))
        if arr is not None:
            count = max(count, int(arr.shape[0]))
    return count


def _first_array(mapping: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key not in mapping or mapping[key] is None:
            continue
        arr = _to_numpy(mapping[key]).astype(np.float32)
        if arr.ndim >= 2:
            return arr
    return None


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _bbox_array(detection: dict[str, Any]) -> np.ndarray | None:
    for key in ("bbox", "boxes", "pred_boxes"):
        if key not in detection or detection[key] is None:
            continue
        arr = _to_numpy(detection[key]).reshape(-1)
        if arr.shape[0] >= 4 and np.isfinite(arr[:4]).all():
            return arr[:4].astype(float)
    return None


def _bbox_center(detection: dict[str, Any]) -> np.ndarray | None:
    bbox = _bbox_array(detection)
    if bbox is None:
        return None
    return np.asarray([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=float)


def _faces_from_selected(selected: list[dict[str, Any] | None]) -> np.ndarray | None:
    for detection in selected:
        if detection is None:
            continue
        for key in ("faces", "mesh_faces", "pred_faces"):
            if detection.get(key) is None:
                continue
            faces = _to_numpy(detection[key]).astype(np.int32)
            if faces.ndim == 2 and faces.shape[1] == 3:
                return faces
    return None


def _collect_mhr_params(selected: list[dict[str, Any] | None]) -> dict[str, Any]:
    params: dict[str, list[Any]] = {}
    for detection in selected:
        values = detection.get("mhr_model_params") if detection else None
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            params.setdefault(str(key), []).append(_to_numpy(value).astype(np.float32))
    out: dict[str, Any] = {}
    for key, values in params.items():
        try:
            out[key] = np.stack(values, axis=0)
        except Exception:
            out[key] = values
    return out


def _metadata_for_detection(frame_idx: int, detection: dict[str, Any]) -> dict[str, Any]:
    bbox = _bbox_array(detection)
    return {
        "frame_index": int(frame_idx),
        "status": "ok",
        "bbox": bbox.tolist() if bbox is not None else None,
        "has_vertices": _first_array(detection, ("pred_vertices", "vertices", "verts")) is not None,
        "has_joints_3d": _first_array(detection, ("pred_keypoints_3d", "keypoints_3d", "pred_joint_coords", "joints_3d")) is not None,
        "has_joints_2d": _first_array(detection, ("pred_keypoints_2d", "keypoints_2d", "joints_2d")) is not None,
    }
