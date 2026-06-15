from __future__ import annotations

from pathlib import Path

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.logging_utils import load_state, read_json, write_json
from monocap_v2.core.stage_utils import cached, stage_result


STAGE = "stage_13_qc_report"


def run(run_dir: Path, cfg: dict, force: bool = False) -> dict:
    registry = ArtifactRegistry(run_dir)
    out_path = registry.ensure_parent("qc_report")
    if cached(out_path, force):
        return stage_result(STAGE, "cached", output=str(out_path))
    state = load_state(run_dir)
    artifacts = _artifact_summaries(registry)
    status = "ok"
    for key in ["wham_timeline", "smpl_mesh", "marker_placement", "markers", "marker_jump", "trc", "mocap_validation"]:
        if (artifacts.get(key) or {}).get("status") == "warning":
            status = "warning"
    result = stage_result(STAGE, status, output=str(out_path), summary=str(registry.get("summary_md")))
    stages = dict(state.get("stages", {}))
    stages[STAGE] = result
    report = {
        "run_dir": str(run_dir),
        "activity": cfg.get("activity"),
        "backends": cfg.get("config", {}).get("backends", {}),
        "artifacts": artifacts,
        "stages": stages,
    }
    write_json(out_path, report)
    summary_path = registry.ensure_parent("summary_md")
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# monocap_v2 QC Summary\n\n")
        f.write(f"- Run: `{run_dir}`\n")
        f.write(f"- Activity: `{cfg.get('activity')}`\n")
        pose3d = report["artifacts"].get("pose3d_initial") or {}
        if pose3d:
            f.write(f"- Pose3D backend: `{pose3d.get('backend')}`\n")
            f.write(f"- Pose3D representation: `{pose3d.get('representation')}`\n")
            f.write(f"- SMPL available: `{pose3d.get('has_smpl')}`\n")
            if pose3d.get("source_video"):
                f.write(f"- Pose3D source video: `{pose3d.get('source_video')}`\n")
        smpl_vis = report["artifacts"].get("smpl_visualization") or {}
        if smpl_vis:
            f.write(f"- SMPL preview: `{smpl_vis.get('status')}` ({smpl_vis.get('frames_rendered')}/{smpl_vis.get('frames_available')} frames)\n")
            time_window = smpl_vis.get("time_window") or {}
            if time_window.get("status") == "applied":
                f.write(f"- SMPL preview raw frames: `{time_window.get('raw_frame_start')}`-`{time_window.get('raw_frame_end')}`\n")
        smpl_mesh = report["artifacts"].get("smpl_mesh") or {}
        if smpl_mesh:
            f.write(
                f"- SMPL mesh preview: `{smpl_mesh.get('status')}` "
                f"`{smpl_mesh.get('render_mode')}` ({smpl_mesh.get('frames_rendered')}/{smpl_mesh.get('frames_available')} frames)\n"
            )
            f.write(f"- SMPL mesh faces: `{smpl_mesh.get('faces_rendered')}` from `{smpl_mesh.get('face_source_type')}`\n")
            f.write(f"- SMPL mesh marker overlay: `{smpl_mesh.get('marker_overlay_status')}`\n")
            for warning in smpl_mesh.get("warnings") or []:
                f.write(f"- Warning: {warning}\n")
        wham = report["artifacts"].get("wham_timeline") or {}
        if wham:
            timebase = wham.get("timebase") or {}
            alignment = wham.get("raw_sync_alignment") or {}
            overlap = wham.get("overlap") or {}
            f.write("\n## WHAM Timeline\n\n")
            f.write(f"- Timeline status: `{wham.get('status')}`\n")
            f.write(f"- Track: `{timebase.get('selected_track_id')}`\n")
            f.write(f"- Raw frames: `{timebase.get('frame_start')}`-`{timebase.get('frame_end')}`\n")
            if alignment.get("best_raw_offset") is not None:
                f.write(f"- Estimated sync offset: `{alignment.get('best_raw_offset')}`\n")
                f.write(f"- Raw/sync match score: `{alignment.get('best_match_score')}`\n")
            if overlap.get("overlap_frames") is not None:
                f.write(f"- Sync/WHAM overlap frames: `{overlap.get('overlap_frames')}`\n")
            for warning in wham.get("warnings") or []:
                f.write(f"- Warning: {warning}\n")
        markers = report["artifacts"].get("markers") or {}
        marker_placement = report["artifacts"].get("marker_placement") or {}
        marker_jump = report["artifacts"].get("marker_jump") or {}
        trc = report["artifacts"].get("trc") or {}
        if markers or marker_placement or marker_jump or trc:
            f.write("\n## Markers\n\n")
            if markers:
                f.write(f"- Marker status: `{markers.get('status')}`\n")
                f.write(f"- Marker set: `{markers.get('marker_set')}`\n")
                f.write(f"- Marker count: `{markers.get('markers')}`\n")
                f.write(f"- Marker finite ratio: `{markers.get('finite_ratio')}`\n")
                time_window = markers.get("time_window") or {}
                if time_window:
                    f.write(f"- Marker time window: `{time_window.get('status')}` via `{time_window.get('source') or time_window.get('mode')}`\n")
                    if time_window.get("raw_frame_start") is not None:
                        f.write(
                            f"- Marker raw frames: `{time_window.get('raw_frame_start')}`-`{time_window.get('raw_frame_end')}` "
                            f"({time_window.get('kept_frames')}/{time_window.get('original_frames')} kept)\n"
                        )
                if markers.get("debug_warning"):
                    f.write(f"- Warning: {markers.get('debug_warning')}\n")
                for warning in markers.get("warnings") or []:
                    f.write(f"- Warning: {warning}\n")
            if marker_placement:
                f.write(f"- Marker placement status: `{marker_placement.get('status')}`\n")
                f.write(f"- Marker placement group: `{marker_placement.get('group')}`\n")
                f.write(f"- Marker placement worst marker: `{marker_placement.get('worst_marker')}`\n")
                if marker_placement.get("proposal_output"):
                    f.write(f"- Marker map proposal: `{marker_placement.get('proposal_output')}`\n")
                for warning in (marker_placement.get("warnings") or [])[:8]:
                    f.write(f"- Warning: {warning}\n")
            if marker_jump:
                worst = (marker_jump.get("top_jumps") or [{}])[0]
                f.write(f"- Marker jump status: `{marker_jump.get('status')}`\n")
                f.write(f"- Marker jump classification: `{marker_jump.get('classification')}`\n")
                f.write(f"- Max marker jump: `{marker_jump.get('max_jump_m')}` m\n")
                if worst:
                    f.write(
                        f"- Worst marker jump: `{worst.get('marker')}` frames "
                        f"`{worst.get('from_frame')}`-`{worst.get('to_frame')}`\n"
                    )
            if trc:
                f.write(f"- TRC status: `{trc.get('status')}`\n")
                f.write(f"- TRC units: `{trc.get('units_trc')}`\n")
                f.write(f"- TRC output: `{trc.get('output')}`\n")
                if trc.get("debug_warning"):
                    f.write(f"- Warning: {trc.get('debug_warning')}\n")
        mocap = report["artifacts"].get("mocap_validation") or {}
        if mocap:
            refined = mocap.get("refined") or {}
            timing = mocap.get("timing") or {}
            transform = mocap.get("transform_qc") or {}
            outputs = mocap.get("outputs") or {}
            f.write("\n## Mocap Validation\n\n")
            f.write(f"- Validation status: `{mocap.get('status')}`\n")
            f.write(f"- Timing mode: `{timing.get('mode')}`\n")
            f.write(
                f"- Frames: `{timing.get('frames_after_mocap_overlap')}`; "
                f"prediction FPS: `{refined.get('prediction_fps')}`; mocap FPS: `{refined.get('mocap_data_rate_hz')}`\n"
            )
            f.write(f"- Compared joints: `{mocap.get('joint_count')}`\n")
            f.write(f"- Normal pelvis-centered MPJPE: `{_fmt_mm(refined.get('normal_root_centered_mpjpe_mm'))}`\n")
            f.write(f"- Root-centered rigid MPJPE: `{_fmt_mm(refined.get('root_centered_rigid_mpjpe_mm'))}`\n")
            f.write(f"- PA-MPJPE: `{_fmt_mm(refined.get('pa_mpjpe_mm'))}`\n")
            f.write(f"- Absolute transform QC: `{transform.get('status')}`\n")
            for label in ["mocap_joint_errors_plot", "mocap_lower_limb_overlay", "mocap_foot_trajectories_plot", "mocap_segment_lengths_plot"]:
                if outputs.get(label):
                    f.write(f"- {label}: `{outputs[label]}`\n")
            for warning in mocap.get("warnings") or []:
                f.write(f"- Warning: {warning}\n")
            mocap_only_native = mocap.get("mocap_only_native") or {}
            if mocap_only_native:
                native_outputs = mocap_only_native.get("outputs") or {}
                axes = mocap_only_native.get("axis_description") or {}
                f.write("\n### Mocap Only Native View\n\n")
                f.write(f"- Native mocap view status: `{mocap_only_native.get('status')}`\n")
                f.write(
                    f"- Raw TRC markers: `{mocap_only_native.get('marker_count')}` across "
                    f"`{mocap_only_native.get('frames')}` frames at `{mocap_only_native.get('data_rate_hz')}` Hz\n"
                )
                f.write(f"- Raw TRC finite ratio: `{mocap_only_native.get('finite_ratio')}`\n")
                f.write(f"- Native axes: `X={axes.get('x')}, Y={axes.get('y')}, Z={axes.get('z')}`\n")
                f.write(f"- Alignment: `{mocap_only_native.get('alignment')}`; resampling: `{mocap_only_native.get('resampling')}`\n")
                if native_outputs.get("mocap_only_native_overlay"):
                    f.write(f"- Mocap-only native overlay: `{native_outputs['mocap_only_native_overlay']}`\n")
                if native_outputs.get("mocap_only_native_representative_frame"):
                    f.write(f"- Mocap-only representative frame: `{native_outputs['mocap_only_native_representative_frame']}`\n")
            marker_comparison = mocap.get("marker_comparison") or {}
            if marker_comparison:
                comparison_outputs = marker_comparison.get("outputs") or {}
                comparison_timebase = marker_comparison.get("timebase") or {}
                f.write("\n### Mocap Vs SMPL Debug Markers\n\n")
                f.write(f"- Marker comparison status: `{marker_comparison.get('status')}`\n")
                f.write(f"- Marker comparison mode: `{marker_comparison.get('comparison_mode')}`\n")
                f.write(
                    f"- Marker comparison timeline: `{comparison_timebase.get('mode')}` "
                    f"at `{comparison_timebase.get('comparison_rate_hz')}` Hz "
                    f"({comparison_timebase.get('comparison_frames')} frames)\n"
                )
                f.write(f"- Compared markers: `{marker_comparison.get('marker_count')}`\n")
                f.write(f"- Median debug-marker residual: `{_fmt_mm(marker_comparison.get('overall_median_residual_mm'))}`\n")
                for limitation in marker_comparison.get("limitations") or []:
                    f.write(f"- Limitation: {limitation}\n")
                if comparison_outputs.get("mocap_smpl_marker_overlay"):
                    f.write(f"- Mocap/SMPL marker overlay: `{comparison_outputs['mocap_smpl_marker_overlay']}`\n")
                if comparison_outputs.get("mocap_smpl_marker_errors_plot"):
                    f.write(f"- Mocap/SMPL marker errors: `{comparison_outputs['mocap_smpl_marker_errors_plot']}`\n")
                for warning in marker_comparison.get("warnings") or []:
                    f.write(f"- Warning: {warning}\n")
        if wham or markers or trc or mocap:
            f.write("\n## Stages\n\n")
        for name, entry in sorted(stages.items()):
            f.write(f"- {name}: `{entry.get('status')}`\n")
    return result


def _artifact_summaries(registry: ArtifactRegistry) -> dict:
    keys = {
        "camera": "camera_qc",
        "pose3d_initial": "pose3d_initial_qc",
        "optimization_stage2": "opt_stage2_report",
        "visualization": "visualize_qc",
        "smpl_visualization": "smpl_vertices_qc",
        "smpl_mesh": "smpl_mesh_qc",
        "marker_placement": "smpl_marker_placement_qc",
        "wham_timeline": "wham_timeline_qc",
        "markers": "virtual_markers_qc",
        "marker_jump": "marker_jump_qc",
        "trc": "smpl_markers_qc",
        "mocap_validation": "mocap_validation",
        "opensim": "opensim_ik_report",
    }
    summaries = {}
    for label, key in keys.items():
        path = registry.get(key)
        if not path.exists() or path.suffix != ".json":
            continue
        try:
            summaries[label] = read_json(path)
        except Exception as exc:
            summaries[label] = {"status": "unreadable", "path": str(path), "error": str(exc)}
    return summaries


def _fmt_mm(value) -> str:
    return "unavailable" if value is None else f"{float(value):.3f} mm"
