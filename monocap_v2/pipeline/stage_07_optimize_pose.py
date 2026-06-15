from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.logging_utils import read_json, read_yaml, write_json
from monocap_v2.core.mocap_eval import evaluate_pose_against_mocap
from monocap_v2.core.refinement import compute_refinement_metrics, optimize_joints_only
from monocap_v2.core.stage_utils import cached, stage_result


STAGE = "stage_07_optimize_pose"


def run(run_dir: Path, cfg: dict, force: bool = False) -> dict:
    registry = ArtifactRegistry(run_dir)
    refined_path = registry.ensure_parent("pose3d_refined")
    report_path = registry.ensure_parent("opt_stage2_report")
    if cached(refined_path, force) and cached(report_path, force):
        return stage_result(STAGE, "cached", output=str(refined_path), report=str(report_path))

    with registry.get("pose3d_initial").open("rb") as f:
        pose3d = pickle.load(f)

    opt_cfg = cfg.get("config", {}).get("optimization", {}).get("joints_only", {})
    if not opt_cfg.get("enabled", True):
        refined_pose = _passthrough_pose(pose3d, "Joints-only optimization disabled in config.")
        _write_pose(refined_path, refined_pose)
        report = stage_result(
            STAGE,
            "skipped",
            output=str(refined_path),
            representation=pose3d.get("representation"),
            method="passthrough",
            reason=refined_pose["refinement"]["reason"],
        )
        write_json(report_path, report)
        return report

    representation = pose3d.get("representation")
    if representation in {"hybrid", "smpl"} and not opt_cfg.get("allow_hybrid_or_smpl", False):
        reason = (
            f"Joints-only optimization skipped for {representation} artifacts because it would make "
            "joints inconsistent with SMPL parameters/vertices."
        )
        refined_pose = _passthrough_pose(pose3d, reason)
        _write_pose(refined_path, refined_pose)
        report = stage_result(STAGE, "skipped", output=str(refined_path), representation=representation, reason=reason)
        write_json(report_path, report)
        return report

    if representation not in {"joints", "hybrid", "smpl"}:
        refined_pose = _passthrough_pose(pose3d, f"Unsupported representation: {pose3d.get('representation')}")
        _write_pose(refined_path, refined_pose)
        report = stage_result(STAGE, "warning", output=str(refined_path), reason=refined_pose["refinement"]["reason"])
        write_json(report_path, report)
        return report

    camera = read_json(registry.get("camera_assumed"))
    pose2d = _read_pose2d(registry)
    contacts = _read_contacts(registry)
    activity_weights = _activity_weights(cfg.get("activity", "other"))

    initial_joints = np.asarray(pose3d["joints_3d"], dtype=float)
    joint_names = [str(name) for name in pose3d.get("joint_names", [])]
    fps = float(pose3d.get("fps") or 30.0)
    before_metrics = compute_refinement_metrics(
        initial_joints,
        joint_names,
        fps,
        contacts,
        contact_threshold=float(opt_cfg.get("contact_threshold", 0.75)),
    )

    refined_joints, optimizer_report = optimize_joints_only(pose3d, camera, pose2d, contacts, activity_weights, opt_cfg)
    after_metrics = compute_refinement_metrics(
        refined_joints,
        joint_names,
        fps,
        contacts,
        contact_threshold=float(opt_cfg.get("contact_threshold", 0.75)),
    )

    refined_pose = copy.deepcopy(pose3d)
    refined_pose["joints_3d"] = refined_joints.astype(np.float32)
    refined_pose["refinement"] = {
        "status": optimizer_report.get("status", "ok"),
        "method": "scipy_least_squares_joints_only",
        "mocap_used_in_objective": False,
        "source_stage": "stage_04_pose3d_initial",
        "optimizer": optimizer_report,
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
    }

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
        optimizer=optimizer_report,
        metrics_before=before_metrics,
        metrics_after=after_metrics,
        mocap_evaluation=mocap_report,
        warnings=warnings,
    )
    write_json(report_path, report)
    return report


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


def _write_pose(path: Path, pose: dict) -> None:
    with path.open("wb") as f:
        pickle.dump(pose, f)
