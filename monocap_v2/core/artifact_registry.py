from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ARTIFACTS: dict[str, str] = {
    "manifest_resolved": "manifest_resolved.yaml",
    "run_config": "run_config.yaml",
    "pipeline_state": "pipeline_state.json",
    "pipeline_log": "logs/pipeline.log",
    "subject_info": "input/subject.json",
    "camera_assumed": "input/camera_assumed.json",
    "camera_qc": "input/camera_qc.json",
    "raw_video_ref": "video/raw_video_ref.json",
    "preprocessed_video": "video/preprocessed_video.mp4",
    "video_info": "video/video_info.json",
    "video_qc": "video/video_qc.json",
    "sample_frame_000": "video/sample_frames/frame_000.png",
    "sample_frame_mid": "video/sample_frames/frame_mid.png",
    "sample_frame_last": "video/sample_frames/frame_last.png",
    "keypoints_2d": "pose2d/keypoints_2d.npz",
    "pose2d_qc": "pose2d/pose2d_qc.json",
    "pose3d_initial": "pose3d_initial/pose3d_initial.pkl",
    "pose3d_initial_qc": "pose3d_initial/pose3d_initial_qc.json",
    "activity": "contacts/activity.json",
    "contacts": "contacts/contacts.npz",
    "contacts_qc": "contacts/contacts_qc.json",
    "opt_stage1_report": "optimization/opt_stage1_report.json",
    "opt_stage2_report": "optimization/opt_stage2_report.json",
    "pose3d_refined": "optimization/pose3d_refined.pkl",
    "virtual_markers": "markers/virtual_markers.pkl",
    "virtual_markers_qc": "markers/virtual_markers.json",
    "smpl_markers_trc": "markers/smpl_markers.trc",
    "smpl_markers_qc": "markers/smpl_markers.json",
    "opensim_ik_report": "opensim/opensim_ik_report.json",
    "visualize_qc": "reports/visualize.json",
    "pose3d_initial_preview": "reports/pose3d_initial_preview.mp4",
    "pose3d_refined_preview": "reports/pose3d_refined_preview.mp4",
    "pelvis_drift_plot": "reports/pelvis_drift.png",
    "foot_contact_plot": "reports/foot_contact.png",
    "smpl_vertices_preview": "reports/smpl_vertices_preview.mp4",
    "smpl_vertices_qc": "reports/smpl_vertices_qc.json",
    "smpl_mesh_preview": "reports/smpl_mesh_preview.mp4",
    "smpl_mesh_qc": "reports/smpl_mesh_qc.json",
    "smpl_marker_placement_qc": "reports/smpl_marker_placement_qc.json",
    "smpl_marker_placement_plot": "reports/smpl_marker_placement_lower_limb.png",
    "smpl_marker_placement_preview": "reports/smpl_marker_placement_preview.mp4",
    "marker_map_proposal": "reports/marker_map_proposal_wham_smpl_debug_v2.yaml",
    "marker_trajectory_plot": "reports/marker_trajectories.png",
    "marker_jump_qc": "reports/marker_jump_diagnostics.json",
    "marker_jump_plot": "reports/marker_jump_diagnostics.png",
    "marker_jump_preview": "reports/marker_jump_preview.mp4",
    "wham_raw_overlay": "reports/wham_raw_overlay.mp4",
    "wham_labeled_overlay": "reports/wham_labeled_overlay.mp4",
    "wham_timeline_plot": "reports/wham_timeline.png",
    "wham_timeline_qc": "reports/wham_timeline_qc.json",
    "mocap_validation": "reports/mocap_validation.json",
    "mocap_validation_series": "reports/mocap_validation_series.npz",
    "mocap_joint_errors_plot": "reports/mocap_joint_errors.png",
    "mocap_lower_limb_overlay": "reports/mocap_lower_limb_overlay.mp4",
    "mocap_foot_trajectories_plot": "reports/mocap_foot_trajectories.png",
    "mocap_segment_lengths_plot": "reports/mocap_segment_lengths.png",
    "mocap_smpl_marker_qc": "reports/mocap_smpl_marker_qc.json",
    "mocap_smpl_marker_series": "reports/mocap_smpl_marker_series.npz",
    "mocap_smpl_marker_overlay": "reports/mocap_smpl_marker_overlay.mp4",
    "mocap_smpl_marker_errors_plot": "reports/mocap_smpl_marker_errors.png",
    "mocap_smpl_marker_representative_frame": "reports/mocap_smpl_marker_representative_frame.png",
    "mocap_only_native_overlay": "reports/mocap_only_native_overlay.mp4",
    "mocap_only_native_representative_frame": "reports/mocap_only_native_representative_frame.png",
    "mocap_only_native_qc": "reports/mocap_only_native_qc.json",
    "qc_report": "reports/qc_report.json",
    "summary_md": "reports/summary.md",
}


@dataclass(frozen=True)
class ArtifactRegistry:
    run_dir: Path

    def get(self, key: str) -> Path:
        if key not in ARTIFACTS:
            known = ", ".join(sorted(ARTIFACTS))
            raise KeyError(f"Unknown artifact key '{key}'. Known keys: {known}")
        return self.run_dir / ARTIFACTS[key]

    def exists(self, key: str) -> bool:
        return self.get(key).exists()

    def ensure_parent(self, key: str) -> Path:
        path = self.get(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_standard_dirs(self) -> None:
        for rel in ARTIFACTS.values():
            (self.run_dir / rel).parent.mkdir(parents=True, exist_ok=True)
