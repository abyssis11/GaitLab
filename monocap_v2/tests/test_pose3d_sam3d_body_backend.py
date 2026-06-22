from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.backend_registry import load_backend
from monocap_v2.core.level_a_benchmark import _canonical_prediction_joints
from monocap_v2.core.sam3d_body_adapter import build_sam3d_pose_artifact, select_primary_detection
from monocap_v2.core.schemas import has_mesh_vertices, has_smpl_vertices, validate_pose3d_artifact
from monocap_v2.pipeline import stage_08_extract_virtual_markers
from monocap_v2.pipeline import stage_12_visualize
from monocap_v2.scripts.benchmark_backends_level_a import _backend_preset


def test_sam3d_backend_is_registered() -> None:
    backend = load_backend("pose3d", "sam3d_body")
    assert backend.__name__ == "monocap_v2.backends.pose3d_sam3d_body"


def test_sam3d_primary_detection_uses_largest_then_nearest_center() -> None:
    detections = [
        {"bbox": np.asarray([0, 0, 20, 20], dtype=np.float32)},
        {"bbox": np.asarray([100, 100, 160, 160], dtype=np.float32)},
    ]
    assert select_primary_detection(detections) == 1
    assert select_primary_detection(detections, previous_center=np.asarray([10.0, 10.0])) == 0


def test_sam3d_adapter_converts_synthetic_output_to_pose_artifact() -> None:
    frame_results = [
        {
            "detections": [
                _detection(0.0, bbox=[0, 0, 100, 200]),
                _detection(1.0, bbox=[0, 0, 20, 20]),
            ]
        },
        {"detections": [_detection(0.2, bbox=[4, 0, 104, 200])]},
        {"detections": []},
    ]
    cfg = {
        "config": {
            "sam3d_body": {
                "canonical_joint_indices": {
                    "pelvis": 0,
                    "left_hip": 1,
                    "right_hip": 2,
                    "left_knee": 3,
                    "right_knee": 4,
                    "left_ankle": 5,
                    "right_ankle": 6,
                },
                "repo_path": "external/sam-3d-body",
                "checkpoint_path": "models/sam3d_body/model.ckpt",
                "mhr_model_path": "models/sam3d_body/assets/mhr_model.pt",
            }
        }
    }

    artifact, qc = build_sam3d_pose_artifact(
        frame_results,
        fps=60.0,
        camera={"is_assumed": False},
        subject={"id": "subject7", "height_m": 1.8},
        cfg=cfg,
        source_video="walking1_syncdWithMocap.avi",
        inference_video="walking1_syncdWithMocap.avi",
    )

    validate_pose3d_artifact(artifact)
    assert artifact["representation"] == "joints"
    assert artifact["backend"] == "sam3d_body"
    assert artifact["joint_names"][:7] == ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    assert artifact["joints_3d"].shape == (3, 12, 3)
    assert np.isnan(artifact["joints_3d"][2]).all()
    assert has_mesh_vertices(artifact)
    assert not has_smpl_vertices(artifact)
    assert qc["missing_frame_count"] == 1
    assert qc["mesh_available"] is True


def test_sam3d_default_mapping_uses_official_mhr70_lower_limb_indices() -> None:
    artifact, qc = build_sam3d_pose_artifact(
        [{"detections": [_detection_with_joint_count(70, bbox=[0, 0, 100, 200])]}],
        fps=60.0,
        camera={"is_assumed": False},
        subject={"id": "subject7", "height_m": 1.8},
        cfg={
            "config": {
                "sam3d_body": {
                    "repo_path": "external/sam-3d-body",
                    "checkpoint_path": "models/sam3d_body/model.ckpt",
                    "mhr_model_path": "models/sam3d_body/assets/mhr_model.pt",
                }
            }
        },
        source_video="walking1_syncdWithMocap.avi",
        inference_video="walking1_syncdWithMocap.avi",
    )

    assert artifact["joint_names"][:8] == [
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_mtp",
        "right_mtp",
    ]
    assert "pelvis" not in artifact["joint_names"]
    assert artifact["backend_meta"]["mapping_source"] == "sam3d_mhr70_official"
    assert qc["canonical_joint_coverage"]["selected_indices"] == {
        "left_hip": 9,
        "right_hip": 10,
        "left_knee": 11,
        "right_knee": 12,
        "left_ankle": 13,
        "right_ankle": 14,
        "left_mtp": 15,
        "right_mtp": 18,
    }
    assert np.allclose(artifact["joints_3d"][0, 0], [9.0, 9.1, 9.2])
    assert np.allclose(artifact["joints_3d"][0, 1], [10.0, 10.1, 10.2])
    assert "mhr70_nose" in artifact["joint_names"]

    selected_names, selected_joints, _convention = _canonical_prediction_joints(artifact, "identity")
    pelvis_idx = selected_names.index("pelvis")
    assert np.allclose(selected_joints[0, pelvis_idx], [9.5, 9.6, 9.7])


def test_sam3d_mesh_does_not_trigger_smpl_marker_stage(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_standard_dirs()
    artifact, _qc = build_sam3d_pose_artifact(
        [{"detections": [_detection(0.0, bbox=[0, 0, 100, 200])]}],
        fps=30.0,
        camera={"is_assumed": True},
        subject={"id": "subject"},
        cfg={"config": {"sam3d_body": {"canonical_joint_indices": {"pelvis": 0, "left_hip": 1, "right_hip": 2}}}},
        source_video="video.mp4",
        inference_video="video.mp4",
    )
    with registry.ensure_parent("pose3d_refined").open("wb") as f:
        pickle.dump(artifact, f)

    result = stage_08_extract_virtual_markers.run(tmp_path, {"config": {}}, force=True)

    assert result["status"] == "skipped"
    assert "No SMPL vertices" in result["reason"]


def test_sam3d_generic_mesh_preview_smoke(tmp_path: Path) -> None:
    artifact, _qc = build_sam3d_pose_artifact(
        [{"detections": [_detection(0.0, bbox=[0, 0, 100, 200])]}, {"detections": [_detection(0.1, bbox=[2, 0, 102, 200])]}],
        fps=5.0,
        camera={"is_assumed": True},
        subject={"id": "subject"},
        cfg={"config": {"sam3d_body": {"canonical_joint_indices": {"pelvis": 0, "left_hip": 1, "right_hip": 2}}}},
        source_video="video.mp4",
        inference_video="video.mp4",
    )
    out = tmp_path / "mesh.mp4"

    report = stage_12_visualize._write_generic_mesh_preview(
        artifact,
        out,
        {"config": {"visualization": {"preview_width": 320, "preview_height": 240, "preview_fps": 5, "mesh": {"max_faces": 2}}}},
    )

    assert out.exists()
    assert out.stat().st_size > 0
    assert report["render_mode"] == "mesh"
    assert report["model_type"] == "mhr"


def test_level_a_benchmark_uses_sam3d_preset() -> None:
    assert _backend_preset("sam3d_body") == "opencap_sam3d_body_vith_detector_intrinsics"
    assert _backend_preset("sam3d_body", sam3d_preset="opencap_sam3d_body_vith_nodetector") == "opencap_sam3d_body_vith_nodetector"


def _detection(offset: float, bbox: list[float]) -> dict:
    joints = np.arange(12 * 3, dtype=np.float32).reshape(12, 3) / 100.0 + np.float32(offset)
    xy = np.arange(12 * 2, dtype=np.float32).reshape(12, 2) + np.float32(offset)
    vertices = np.arange(5 * 3, dtype=np.float32).reshape(5, 3) / 10.0 + np.float32(offset)
    return {
        "bbox": np.asarray(bbox, dtype=np.float32),
        "pred_keypoints_3d": joints,
        "pred_keypoints_2d": xy,
        "pred_vertices": vertices,
        "pred_cam_t": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        "faces": np.asarray([[0, 1, 2], [2, 3, 4]], dtype=np.int32),
        "mhr_model_params": {"shape": np.zeros((2,), dtype=np.float32)},
    }


def _detection_with_joint_count(count: int, bbox: list[float]) -> dict:
    base = np.arange(count, dtype=np.float32)
    joints = np.stack([base, base + 0.1, base + 0.2], axis=1)
    xy = np.stack([base + 10.0, base + 20.0], axis=1)
    vertices = np.arange(5 * 3, dtype=np.float32).reshape(5, 3) / 10.0
    return {
        "bbox": np.asarray(bbox, dtype=np.float32),
        "pred_keypoints_3d": joints,
        "pred_keypoints_2d": xy,
        "pred_vertices": vertices,
        "pred_cam_t": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        "faces": np.asarray([[0, 1, 2], [2, 3, 4]], dtype=np.int32),
        "mhr_model_params": {"shape": np.zeros((2,), dtype=np.float32)},
    }
