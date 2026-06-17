from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.camera_time_refinement import apply_camera_time_refinement_to_pose
from monocap_v2.core.foot_locking import apply_contact_foot_locking_to_pose
from monocap_v2.core.kinematic_chain import apply_kinematic_chain_to_pose
from monocap_v2.core.logging_utils import read_json, read_yaml, write_json
from monocap_v2.core.mocap_eval import evaluate_pose_against_mocap
from monocap_v2.core.pose_prior import apply_pose_prior_to_pose
from monocap_v2.core.refinement import compute_refinement_metrics, optimize_joints_only
from monocap_v2.core.reprojection_consistency import apply_reprojection_consistency_to_pose
from monocap_v2.core.stage_utils import cached, stage_result
from monocap_v2.core.subject_scale import apply_subject_scale_to_pose
from monocap_v2.core.temporal_smoothing import apply_temporal_smoothing_to_pose


STAGE = "stage_07_optimize_pose"


def run(run_dir: Path, cfg: dict, force: bool = False) -> dict:
    registry = ArtifactRegistry(run_dir)
    refined_path = registry.ensure_parent("pose3d_refined")
    report_path = registry.ensure_parent("opt_stage2_report")
    if cached(refined_path, force) and cached(report_path, force):
        return stage_result(STAGE, "cached", output=str(refined_path), report=str(report_path))

    with registry.get("pose3d_initial").open("rb") as f:
        pose3d = pickle.load(f)

    optimization_cfg, refinement_profile_report = _resolve_refinement_profile(cfg.get("config", {}).get("optimization", {}))
    opt_cfg = optimization_cfg.get("joints_only", {})
    camera_time_cfg = optimization_cfg.get("camera_time_refinement", {})
    scale_cfg = optimization_cfg.get("subject_scale", {})
    reproj_cfg = optimization_cfg.get("reprojection_consistency", {})
    pose_prior_cfg = optimization_cfg.get("pose_prior", {})
    chain_cfg = optimization_cfg.get("kinematic_chain", {})
    smoothing_cfg = optimization_cfg.get("temporal_smoothing", {})
    foot_lock_cfg = optimization_cfg.get("contact_foot_locking", {})
    representation = pose3d.get("representation")
    contacts = _read_contacts(registry)
    pose2d = _read_pose2d(registry)

    if not opt_cfg.get("enabled", True):
        refined_pose = _passthrough_pose(pose3d, "Joints-only optimization disabled in config.")
        camera_time_refinement_report = None
        subject_scale_report = None
        reprojection_consistency_report = None
        pose_prior_report = None
        kinematic_chain_report = None
        temporal_smoothing_report = None
        contact_foot_locking_report = None
        if camera_time_cfg.get("enabled", False):
            camera = read_json(registry.get("camera_assumed"))
            refined_pose, camera_time_refinement_report = _maybe_apply_camera_time_refinement(refined_pose, camera, pose2d, camera_time_cfg)
            refined_pose.setdefault("refinement", {})["camera_time_refinement"] = camera_time_refinement_report
            if camera_time_refinement_report.get("status") == "ok":
                refined_pose["refinement"]["status"] = _combined_status(camera_time_refinement_report)
                refined_pose["refinement"]["method"] = _combined_method(camera_time_refinement_report)
        if scale_cfg.get("enabled", False):
            refined_pose, subject_scale_report = _maybe_apply_subject_scale(registry, refined_pose, scale_cfg, cfg)
            refined_pose.setdefault("refinement", {})["subject_scale"] = subject_scale_report
            if subject_scale_report.get("status") == "ok":
                refined_pose["refinement"]["status"] = _combined_status(camera_time_refinement_report, subject_scale_report)
                refined_pose["refinement"]["method"] = _combined_method(camera_time_refinement_report, subject_scale_report)
        if reproj_cfg.get("enabled", False):
            camera = read_json(registry.get("camera_assumed"))
            refined_pose, reprojection_consistency_report = _maybe_apply_reprojection_consistency(refined_pose, camera, pose2d, reproj_cfg)
            refined_pose.setdefault("refinement", {})["reprojection_consistency"] = reprojection_consistency_report
            if reprojection_consistency_report.get("status") == "ok":
                refined_pose["refinement"]["status"] = _combined_status(
                    camera_time_refinement_report,
                    subject_scale_report,
                    reprojection_consistency_report,
                )
                refined_pose["refinement"]["method"] = _combined_method(
                    camera_time_refinement_report,
                    subject_scale_report,
                    reprojection_consistency_report,
                )
        if pose_prior_cfg.get("enabled", False):
            refined_pose, pose_prior_report = _maybe_apply_pose_prior(refined_pose, pose_prior_cfg)
            refined_pose.setdefault("refinement", {})["pose_prior"] = pose_prior_report
            if pose_prior_report.get("status") == "ok":
                refined_pose["refinement"]["status"] = _combined_status(
                    camera_time_refinement_report,
                    subject_scale_report,
                    reprojection_consistency_report,
                    pose_prior_report,
                )
                refined_pose["refinement"]["method"] = _combined_method(
                    camera_time_refinement_report,
                    subject_scale_report,
                    reprojection_consistency_report,
                    pose_prior_report,
                )
        if chain_cfg.get("enabled", False):
            refined_pose, kinematic_chain_report = _maybe_apply_kinematic_chain(refined_pose, chain_cfg)
            refined_pose.setdefault("refinement", {})["kinematic_chain"] = kinematic_chain_report
            if kinematic_chain_report.get("status") == "ok":
                refined_pose["refinement"]["status"] = _combined_status(
                    camera_time_refinement_report,
                    subject_scale_report,
                    reprojection_consistency_report,
                    pose_prior_report,
                    kinematic_chain_report,
                )
                refined_pose["refinement"]["method"] = _combined_method(
                    camera_time_refinement_report,
                    subject_scale_report,
                    reprojection_consistency_report,
                    pose_prior_report,
                    kinematic_chain_report,
                )
        if smoothing_cfg.get("enabled", False):
            refined_pose, temporal_smoothing_report = _maybe_apply_temporal_smoothing(refined_pose, smoothing_cfg)
            refined_pose.setdefault("refinement", {})["temporal_smoothing"] = temporal_smoothing_report
            if temporal_smoothing_report.get("status") == "ok":
                refined_pose["refinement"]["status"] = _combined_status(
                    camera_time_refinement_report,
                    subject_scale_report,
                    reprojection_consistency_report,
                    pose_prior_report,
                    kinematic_chain_report,
                    temporal_smoothing_report,
                )
                refined_pose["refinement"]["method"] = _combined_method(
                    camera_time_refinement_report,
                    subject_scale_report,
                    reprojection_consistency_report,
                    pose_prior_report,
                    kinematic_chain_report,
                    temporal_smoothing_report,
                )
        if foot_lock_cfg.get("enabled", False):
            refined_pose, contact_foot_locking_report = _maybe_apply_contact_foot_locking(refined_pose, contacts, foot_lock_cfg)
            refined_pose.setdefault("refinement", {})["contact_foot_locking"] = contact_foot_locking_report
            if contact_foot_locking_report.get("status") == "ok":
                refined_pose["refinement"]["status"] = _combined_status(
                    subject_scale_report,
                    camera_time_refinement_report,
                    reprojection_consistency_report,
                    pose_prior_report,
                    kinematic_chain_report,
                    temporal_smoothing_report,
                    contact_foot_locking_report,
                )
                refined_pose["refinement"]["method"] = _combined_method(
                    subject_scale_report,
                    camera_time_refinement_report,
                    reprojection_consistency_report,
                    pose_prior_report,
                    kinematic_chain_report,
                    temporal_smoothing_report,
                    contact_foot_locking_report,
                )
        _attach_refinement_profile(refined_pose, refinement_profile_report)
        _write_pose(refined_path, refined_pose)
        status = (
            "ok"
            if _any_ok(
                camera_time_refinement_report,
                subject_scale_report,
                reprojection_consistency_report,
                pose_prior_report,
                kinematic_chain_report,
                temporal_smoothing_report,
                contact_foot_locking_report,
            )
            else "skipped"
        )
        report = stage_result(
            STAGE,
            status,
            output=str(refined_path),
            representation=representation,
            method=refined_pose.get("refinement", {}).get("method", "passthrough"),
            refinement_profile=refinement_profile_report,
            reason=refined_pose["refinement"]["reason"],
            camera_time_refinement=camera_time_refinement_report,
            subject_scale=subject_scale_report,
            reprojection_consistency=reprojection_consistency_report,
            pose_prior=pose_prior_report,
            kinematic_chain=kinematic_chain_report,
            temporal_smoothing=temporal_smoothing_report,
            contact_foot_locking=contact_foot_locking_report,
        )
        write_json(report_path, report)
        return report

    if representation in {"hybrid", "smpl"} and not opt_cfg.get("allow_hybrid_or_smpl", False):
        reason = (
            f"Joints-only optimization skipped for {representation} artifacts because it would make "
            "joints inconsistent with SMPL parameters/vertices."
        )
        refined_pose = _passthrough_pose(pose3d, reason)
        subject_scale_report = None
        camera_time_refinement_report = None
        reprojection_consistency_report = None
        pose_prior_report = None
        kinematic_chain_report = None
        temporal_smoothing_report = None
        contact_foot_locking_report = None
        if camera_time_cfg.get("enabled", False):
            camera_time_refinement_report = {
                "status": "skipped",
                "reason": "Camera/time refinement skipped for SMPL/hybrid artifacts to keep joints and vertices consistent.",
                "representation": representation,
                "mocap_used_in_objective": False,
            }
            refined_pose.setdefault("refinement", {})["camera_time_refinement"] = camera_time_refinement_report
        if scale_cfg.get("enabled", False):
            subject_scale_report = {
                "status": "skipped",
                "reason": "Subject-scale correction skipped for SMPL/hybrid artifacts to keep joints and vertices consistent.",
                "representation": representation,
                "mocap_used_in_objective": False,
            }
            refined_pose.setdefault("refinement", {})["subject_scale"] = subject_scale_report
        if reproj_cfg.get("enabled", False):
            reprojection_consistency_report = {
                "status": "skipped",
                "reason": "Reprojection consistency skipped for SMPL/hybrid artifacts to keep joints and vertices consistent.",
                "representation": representation,
                "mocap_used_in_objective": False,
            }
            refined_pose.setdefault("refinement", {})["reprojection_consistency"] = reprojection_consistency_report
        if pose_prior_cfg.get("enabled", False):
            pose_prior_report = {
                "status": "skipped",
                "reason": "Pose prior skipped for SMPL/hybrid artifacts to keep joints and vertices consistent.",
                "representation": representation,
                "mocap_used_in_objective": False,
            }
            refined_pose.setdefault("refinement", {})["pose_prior"] = pose_prior_report
        if chain_cfg.get("enabled", False):
            kinematic_chain_report = {
                "status": "skipped",
                "reason": "Kinematic-chain refinement skipped for SMPL/hybrid artifacts to keep joints and vertices consistent.",
                "representation": representation,
                "mocap_used_in_objective": False,
            }
            refined_pose.setdefault("refinement", {})["kinematic_chain"] = kinematic_chain_report
        if smoothing_cfg.get("enabled", False):
            temporal_smoothing_report = {
                "status": "skipped",
                "reason": "Temporal smoothing skipped for SMPL/hybrid artifacts to keep joints and vertices consistent.",
                "representation": representation,
                "mocap_used_in_objective": False,
            }
            refined_pose.setdefault("refinement", {})["temporal_smoothing"] = temporal_smoothing_report
        if foot_lock_cfg.get("enabled", False):
            contact_foot_locking_report = {
                "status": "skipped",
                "reason": "Contact foot locking skipped for SMPL/hybrid artifacts to keep joints and vertices consistent.",
                "representation": representation,
                "mocap_used_in_objective": False,
            }
            refined_pose.setdefault("refinement", {})["contact_foot_locking"] = contact_foot_locking_report
        _attach_refinement_profile(refined_pose, refinement_profile_report)
        _write_pose(refined_path, refined_pose)
        report = stage_result(
            STAGE,
            "skipped",
            output=str(refined_path),
            representation=representation,
            refinement_profile=refinement_profile_report,
            reason=reason,
            camera_time_refinement=camera_time_refinement_report,
            subject_scale=subject_scale_report,
            reprojection_consistency=reprojection_consistency_report,
            pose_prior=pose_prior_report,
            kinematic_chain=kinematic_chain_report,
            temporal_smoothing=temporal_smoothing_report,
            contact_foot_locking=contact_foot_locking_report,
        )
        write_json(report_path, report)
        return report

    if representation not in {"joints", "hybrid", "smpl"}:
        refined_pose = _passthrough_pose(pose3d, f"Unsupported representation: {pose3d.get('representation')}")
        _attach_refinement_profile(refined_pose, refinement_profile_report)
        _write_pose(refined_path, refined_pose)
        report = stage_result(
            STAGE,
            "warning",
            output=str(refined_path),
            refinement_profile=refinement_profile_report,
            reason=refined_pose["refinement"]["reason"],
        )
        write_json(report_path, report)
        return report

    camera = read_json(registry.get("camera_assumed"))
    activity_weights = _activity_weights(cfg.get("activity", "other"))

    working_pose = pose3d
    camera_time_refinement_report = None
    subject_scale_report = None
    reprojection_consistency_report = None
    pose_prior_report = None
    kinematic_chain_report = None
    temporal_smoothing_report = None
    contact_foot_locking_report = None
    if camera_time_cfg.get("enabled", False):
        working_pose, camera_time_refinement_report = _maybe_apply_camera_time_refinement(working_pose, camera, pose2d, camera_time_cfg)
    if scale_cfg.get("enabled", False):
        working_pose, subject_scale_report = _maybe_apply_subject_scale(registry, working_pose, scale_cfg, cfg)
    if reproj_cfg.get("enabled", False):
        working_pose, reprojection_consistency_report = _maybe_apply_reprojection_consistency(working_pose, camera, pose2d, reproj_cfg)
    if pose_prior_cfg.get("enabled", False):
        working_pose, pose_prior_report = _maybe_apply_pose_prior(working_pose, pose_prior_cfg)
    if chain_cfg.get("enabled", False):
        working_pose, kinematic_chain_report = _maybe_apply_kinematic_chain(working_pose, chain_cfg)
    if smoothing_cfg.get("enabled", False):
        working_pose, temporal_smoothing_report = _maybe_apply_temporal_smoothing(working_pose, smoothing_cfg)
    if foot_lock_cfg.get("enabled", False):
        working_pose, contact_foot_locking_report = _maybe_apply_contact_foot_locking(working_pose, contacts, foot_lock_cfg)

    initial_joints = np.asarray(working_pose["joints_3d"], dtype=float)
    joint_names = [str(name) for name in working_pose.get("joint_names", [])]
    fps = float(working_pose.get("fps") or 30.0)
    before_metrics = compute_refinement_metrics(
        initial_joints,
        joint_names,
        fps,
        contacts,
        contact_threshold=float(opt_cfg.get("contact_threshold", 0.75)),
    )

    refined_joints, optimizer_report = optimize_joints_only(working_pose, camera, pose2d, contacts, activity_weights, opt_cfg)
    after_metrics = compute_refinement_metrics(
        refined_joints,
        joint_names,
        fps,
        contacts,
        contact_threshold=float(opt_cfg.get("contact_threshold", 0.75)),
    )

    refined_pose = copy.deepcopy(working_pose)
    refined_pose["joints_3d"] = refined_joints.astype(np.float32)
    refined_pose["refinement"] = {
        "status": optimizer_report.get("status", "ok"),
        "method": "scipy_least_squares_joints_only",
        "refinement_profile": refinement_profile_report,
        "mocap_used_in_objective": False,
        "source_stage": "stage_04_pose3d_initial",
        "optimizer": optimizer_report,
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
    }
    if subject_scale_report:
        refined_pose["refinement"]["subject_scale"] = subject_scale_report
    if camera_time_refinement_report:
        refined_pose["refinement"]["camera_time_refinement"] = camera_time_refinement_report
    if reprojection_consistency_report:
        refined_pose["refinement"]["reprojection_consistency"] = reprojection_consistency_report
    if pose_prior_report:
        refined_pose["refinement"]["pose_prior"] = pose_prior_report
    if kinematic_chain_report:
        refined_pose["refinement"]["kinematic_chain"] = kinematic_chain_report
    if temporal_smoothing_report:
        refined_pose["refinement"]["temporal_smoothing"] = temporal_smoothing_report
    if contact_foot_locking_report:
        refined_pose["refinement"]["contact_foot_locking"] = contact_foot_locking_report

    mocap_report = _mocap_before_after(cfg, pose3d, refined_pose, opt_cfg)
    if mocap_report:
        refined_pose["refinement"]["mocap_evaluation"] = mocap_report

    status, warnings = _stage_status(optimizer_report, mocap_report, opt_cfg)
    _write_pose(refined_path, refined_pose)

    report = stage_result(
        STAGE,
        status,
        output=str(refined_path),
        representation=refined_pose.get("representation"),
        method="scipy_least_squares_joints_only",
        refinement_profile=refinement_profile_report,
        optimizer=optimizer_report,
        camera_time_refinement=camera_time_refinement_report,
        subject_scale=subject_scale_report,
        reprojection_consistency=reprojection_consistency_report,
        pose_prior=pose_prior_report,
        kinematic_chain=kinematic_chain_report,
        temporal_smoothing=temporal_smoothing_report,
        contact_foot_locking=contact_foot_locking_report,
        metrics_before=before_metrics,
        metrics_after=after_metrics,
        mocap_evaluation=mocap_report,
        warnings=warnings,
    )
    write_json(report_path, report)
    return report


def _resolve_refinement_profile(optimization_cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    base = copy.deepcopy(optimization_cfg or {})
    selected = base.get("refinement_profile")
    if isinstance(selected, dict):
        enabled = bool(selected.get("enabled", True))
        name = str(selected.get("name") or selected.get("profile") or "").strip()
    else:
        enabled = selected not in {None, "", False}
        name = str(selected or "").strip()

    if not enabled or not name or name.lower() in {"none", "null", "false"}:
        return base, {"status": "disabled", "name": None, "mocap_used_in_objective": False}

    profiles = base.get("refinement_profiles") or {}
    profile = profiles.get(name)
    if not isinstance(profile, dict):
        available = ", ".join(sorted(str(key) for key in profiles)) or "<none>"
        raise ValueError(f"Unknown refinement profile {name!r}. Available profiles: {available}")

    merged = _deep_merge(base, profile.get("optimization") or {})
    merged["refinement_profile"] = name
    return merged, {
        "status": "selected",
        "name": name,
        "label": profile.get("label"),
        "version": profile.get("version"),
        "description": profile.get("description"),
        "representation": profile.get("representation"),
        "stage_order": profile.get("stage_order") or [],
        "mocap_used_in_objective": bool(profile.get("mocap_used_in_objective", False)),
        "diagnostic": bool(profile.get("diagnostic", False)),
    }


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _attach_refinement_profile(pose: dict[str, Any], profile_report: dict[str, Any]) -> None:
    pose.setdefault("refinement", {})["refinement_profile"] = profile_report


def _read_pose2d(registry: ArtifactRegistry) -> dict | None:
    path = registry.get("keypoints_2d")
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "xy": data["xy"],
        "confidence": data["confidence"],
        "names": data["names"].tolist(),
        "fps": float(data["fps"]),
        "backend": str(data["backend"]),
    }


def _read_contacts(registry: ArtifactRegistry) -> dict | None:
    path = registry.get("contacts")
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "left_heel": data["left_heel"],
        "left_toe": data["left_toe"],
        "right_heel": data["right_heel"],
        "right_toe": data["right_toe"],
        "backend": str(data["backend"]),
        "activity": str(data["activity"]),
    }


def _maybe_apply_subject_scale(registry: ArtifactRegistry, pose3d: dict, scale_cfg: dict, run_cfg: dict | None = None) -> tuple[dict, dict]:
    subject_path = registry.get("subject_info")
    subject = read_json(subject_path) if subject_path.exists() else {}
    scale_cfg = copy.deepcopy(scale_cfg or {})
    static_pose = None
    static_resolution = None
    static_path_value = scale_cfg.get("static_pose3d_path")
    if not static_path_value and str(scale_cfg.get("mode") or "").startswith("static_"):
        static_path, static_resolution = _resolve_static_pose_path(registry, pose3d, scale_cfg, run_cfg or {})
        if static_path:
            static_path_value = str(static_path)
            scale_cfg["static_pose3d_path"] = str(static_path)
    if static_path_value:
        static_path = Path(str(static_path_value))
        if not static_path.exists():
            refined = copy.deepcopy(pose3d)
            report = {
                "status": "skipped",
                "mode": str(scale_cfg.get("mode") or "height_global"),
                "reason": f"Configured static_pose3d_path does not exist: {static_path}",
                "mocap_used_in_objective": False,
            }
            if static_resolution:
                report["static_resolution"] = static_resolution
            refined["subject_scale"] = report
            return refined, report
        with static_path.open("rb") as f:
            static_pose = pickle.load(f)
    refined, report = apply_subject_scale_to_pose(pose3d, subject, scale_cfg, static_pose=static_pose)
    if static_resolution:
        report["static_resolution"] = static_resolution
        refined["subject_scale"] = report
    return refined, report


def _resolve_static_pose_path(
    registry: ArtifactRegistry,
    pose3d: dict[str, Any],
    scale_cfg: dict[str, Any],
    run_cfg: dict[str, Any],
) -> tuple[Path | None, dict[str, Any]]:
    requested = str(scale_cfg.get("static_trial") or "auto")
    backend = str(pose3d.get("backend") or run_cfg.get("config", {}).get("backends", {}).get("pose3d") or "")
    trials = _static_trial_candidates(requested, scale_cfg)
    run_root = registry.run_dir.parent
    subject, session, camera = _run_identity(registry, run_cfg)
    checked: list[str] = []
    for trial_id in trials:
        if not subject or not session or not camera or not backend:
            continue
        path = run_root / f"{subject}_{session}_{camera}_{trial_id}__{backend}" / "pose3d_initial" / "pose3d_initial.pkl"
        checked.append(str(path))
        if path.exists():
            return path, {
                "status": "ok",
                "requested_static_trial": requested,
                "selected_static_trial": trial_id,
                "backend": backend,
                "source": str(path),
                "checked": checked,
                "mocap_used_in_objective": False,
            }
    return None, {
        "status": "missing",
        "requested_static_trial": requested,
        "candidate_static_trials": trials,
        "backend": backend,
        "checked": checked,
        "reason": "No cached same-subject/camera/backend static pose artifact was found.",
        "mocap_used_in_objective": False,
    }


def _static_trial_candidates(requested: str, scale_cfg: dict[str, Any]) -> list[str]:
    if requested.lower() == "none":
        return []
    if requested.lower() != "auto":
        return [requested]
    priority = scale_cfg.get("static_trial_priority") or ["static2", "static1", "static3"]
    return [str(item) for item in priority if str(item).strip()]


def _run_identity(registry: ArtifactRegistry, run_cfg: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    summary = run_cfg.get("manifest_summary") or {}
    subject = summary.get("subject_id")
    session = summary.get("session")
    camera = summary.get("camera")
    if subject and session and camera:
        return str(subject), str(session), str(camera)
    base = registry.run_dir.name.split("__", 1)[0]
    parts = base.split("_")
    if len(parts) >= 4:
        return parts[0], parts[1], parts[2]
    return None, None, None


def _maybe_apply_camera_time_refinement(pose3d: dict, camera: dict, pose2d: dict | None, camera_time_cfg: dict) -> tuple[dict, dict]:
    return apply_camera_time_refinement_to_pose(pose3d, camera, pose2d, camera_time_cfg)


def _maybe_apply_reprojection_consistency(pose3d: dict, camera: dict, pose2d: dict | None, reproj_cfg: dict) -> tuple[dict, dict]:
    return apply_reprojection_consistency_to_pose(pose3d, camera, pose2d, reproj_cfg)


def _maybe_apply_pose_prior(pose3d: dict, pose_prior_cfg: dict) -> tuple[dict, dict]:
    return apply_pose_prior_to_pose(pose3d, pose_prior_cfg)


def _maybe_apply_kinematic_chain(pose3d: dict, chain_cfg: dict) -> tuple[dict, dict]:
    return apply_kinematic_chain_to_pose(pose3d, chain_cfg)


def _maybe_apply_temporal_smoothing(pose3d: dict, smoothing_cfg: dict) -> tuple[dict, dict]:
    return apply_temporal_smoothing_to_pose(pose3d, smoothing_cfg)


def _maybe_apply_contact_foot_locking(pose3d: dict, contacts: dict | None, foot_lock_cfg: dict) -> tuple[dict, dict]:
    return apply_contact_foot_locking_to_pose(pose3d, contacts, foot_lock_cfg)


def _activity_weights(activity: str) -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "configs" / "activities.yaml"
    data = read_yaml(path)
    return data.get(activity) or data.get("other", {})


def _mocap_before_after(cfg: dict, initial_pose: dict, refined_pose: dict, opt_cfg: dict) -> dict | None:
    mocap_path = cfg.get("mocap_trc")
    if not mocap_path:
        return None
    path = Path(str(mocap_path))
    if not path.exists():
        return {"status": "skipped", "reason": f"mocap_trc does not exist: {path}"}
    try:
        before = evaluate_pose_against_mocap(initial_pose, path, mocap_axis=str(opt_cfg.get("mocap_axis", "-y,x,z")))
        after = evaluate_pose_against_mocap(refined_pose, path, mocap_axis=str(opt_cfg.get("mocap_axis", "-y,x,z")))
    except Exception as exc:
        return {"status": "failed", "error": str(exc), "mocap_trc": str(path)}
    before_primary = before.get("primary_root_centered_rigid_mpjpe_m")
    after_primary = after.get("primary_root_centered_rigid_mpjpe_m")
    improvement = None
    if before_primary is not None and after_primary is not None:
        improvement = float(before_primary - after_primary)
    return {
        "status": "ok",
        "mocap_used_in_objective": False,
        "primary_metric": "root_centered_rigid_mpjpe_mm",
        "primary_metric_internal": "root_centered_rigid_mpjpe_m",
        "initial": before,
        "refined": after,
        "primary_improvement_m": improvement,
        "primary_improvement_mm": improvement * 1000.0 if improvement is not None else None,
    }


def _stage_status(optimizer_report: dict, mocap_report: dict | None, opt_cfg: dict) -> tuple[str, list[str]]:
    warnings = []
    status = "ok"
    if optimizer_report.get("status") == "warning" or not optimizer_report.get("success", True):
        warnings.append("Optimizer did not reduce the objective.")
        status = "warning"
    if mocap_report:
        if mocap_report.get("status") == "failed":
            warnings.append(f"Mocap evaluation failed: {mocap_report.get('error')}")
            status = "warning"
        improvement = mocap_report.get("primary_improvement_m")
        eps = float(opt_cfg.get("primary_improvement_epsilon_m", 0.0005))
        if improvement is not None and improvement < eps:
            warnings.append(
                f"Primary mocap MPJPE did not improve by at least {eps * 1000.0:.3f} mm; "
                f"improvement={improvement * 1000.0:.3f} mm."
            )
            status = "warning"
    return status, warnings


def _passthrough_pose(pose3d: dict, reason: str) -> dict:
    refined_pose = copy.deepcopy(pose3d)
    refined_pose["refinement"] = {
        "status": "passthrough",
        "reason": reason,
        "mocap_used_in_objective": False,
    }
    return refined_pose


def _any_ok(*reports: dict | None) -> bool:
    return any((report or {}).get("status") == "ok" for report in reports)


def _combined_status(*reports: dict | None) -> str:
    parts = _ok_report_names(*reports)
    if parts:
        return "_".join(parts) + "_only"
    return "passthrough"


def _combined_method(*reports: dict | None) -> str:
    parts = _ok_report_names(*reports)
    return "_plus_".join(parts) + "_only" if parts else "passthrough"


def _ok_report_names(*reports: dict | None) -> list[str]:
    parts = []
    for report in reports:
        if (report or {}).get("status") != "ok":
            continue
        if report.get("method") == "contact_foot_locking":
            parts.append("contact_foot_locking")
        elif report.get("method") == "camera_time_reprojection":
            parts.append("camera_time_refinement")
        elif str(report.get("method") or "").startswith("kinematic_chain"):
            parts.append("kinematic_chain")
        elif report.get("method") == "geometric_joint_angle_limits":
            parts.append("pose_prior")
        elif str(report.get("method") or "").startswith("depth_preserving_reprojection"):
            parts.append("reprojection_consistency")
        elif "window_frames" in report and "preserve_bones" in report:
            parts.append("temporal_smoothing")
        elif "mode" in report:
            parts.append("subject_scale")
        else:
            parts.append(str(report.get("method") or "correction"))
    return parts


def _write_pose(path: Path, pose: dict) -> None:
    with path.open("wb") as f:
        pickle.dump(pose, f)
