from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from monocap_v2.core.geometry import find_joint


DEFAULT_TARGET_RATIOS = {
    "pelvis_width": 0.191,
    "thigh": 0.245,
    "shank": 0.246,
    "foot": 0.152,
}

DEFAULT_GLOBAL_SEGMENTS = ("left_thigh", "right_thigh", "left_shank", "right_shank")

JOINT_ALIASES = {
    "pelvis": ("pelvis", "pelv", "root"),
    "left_hip": ("left_hip", "lhip"),
    "right_hip": ("right_hip", "rhip"),
    "left_knee": ("left_knee", "lkne"),
    "right_knee": ("right_knee", "rkne"),
    "left_ankle": ("left_ankle", "lank"),
    "right_ankle": ("right_ankle", "rank"),
    "left_toe": ("left_toe", "ltoe", "left_big_toe", "left_mtp", "lmtp"),
    "right_toe": ("right_toe", "rtoe", "right_big_toe", "right_mtp", "rmtp"),
}


@dataclass(frozen=True)
class SegmentDef:
    name: str
    proximal: str
    distal: str
    target_group: str


LOWER_LIMB_SEGMENTS = (
    SegmentDef("pelvis_width", "left_hip", "right_hip", "pelvis_width"),
    SegmentDef("left_thigh", "left_hip", "left_knee", "thigh"),
    SegmentDef("right_thigh", "right_hip", "right_knee", "thigh"),
    SegmentDef("left_shank", "left_knee", "left_ankle", "shank"),
    SegmentDef("right_shank", "right_knee", "right_ankle", "shank"),
    SegmentDef("left_foot", "left_ankle", "left_toe", "foot"),
    SegmentDef("right_foot", "right_ankle", "right_toe", "foot"),
)


def apply_subject_scale_to_pose(
    pose: dict[str, Any],
    subject: dict[str, Any] | None,
    cfg: dict[str, Any] | None,
    mode: str | None = None,
    static_pose: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a pose copy plus a JSON-friendly correction report.

    This helper never uses mocap walking data. Static poses are optional and
    should be backend predictions from a static trial, not marker/IK ground truth.
    """

    out_pose = copy.deepcopy(pose)
    cfg = cfg or {}
    representation = str(pose.get("representation") or "")
    selected_mode = str(mode or cfg.get("mode") or "height_global")
    report: dict[str, Any] = {
        "status": "skipped",
        "mode": selected_mode,
        "representation": representation,
        "mocap_used_in_objective": False,
        "target_source": None,
    }
    if representation != "joints":
        report["reason"] = f"Subject-scale correction only applies to joints artifacts, got {representation!r}."
        return out_pose, report

    joints = np.asarray(pose.get("joints_3d"), dtype=float)
    names = [str(name) for name in pose.get("joint_names", [])]
    if joints.ndim != 3 or joints.shape[-1] != 3 or not names:
        report["reason"] = "Pose artifact has no valid [T, J, 3] joints and joint names."
        return out_pose, report

    height_m = _subject_height_m(subject)
    if height_m is None:
        report["reason"] = "Subject height is unavailable."
        return out_pose, report

    measured_before = measure_segment_lengths(joints, names)
    try:
        targets, target_report = build_targets(selected_mode, height_m, cfg, static_pose=static_pose)
    except ValueError as exc:
        report["reason"] = str(exc)
        report["segment_lengths_before"] = measured_before
        return out_pose, report

    min_scale = float(cfg.get("min_scale", 0.75))
    max_scale = float(cfg.get("max_scale", 1.35))
    blend = float(cfg.get("retarget_blend", 1.0))
    if selected_mode.endswith("_global"):
        scale_report = estimate_global_scale(
            measured_before,
            targets,
            include_segments=tuple(cfg.get("global_segments") or DEFAULT_GLOBAL_SEGMENTS),
            min_scale=min_scale,
            max_scale=max_scale,
        )
        if scale_report["status"] != "ok":
            report.update(target_report)
            report["reason"] = scale_report.get("reason")
            report["global_scale"] = scale_report
            report["segment_lengths_before"] = measured_before
            return out_pose, report
        corrected = apply_global_scale(joints, names, float(scale_report["scale"]))
        method = "global_scale"
    elif selected_mode.endswith("_bone"):
        corrected = retarget_lower_limb_bones(joints, names, targets, blend=blend)
        scale_report = {"status": "not_applicable", "method": "bone_retarget", "retarget_blend": blend}
        method = "bone_retarget"
    else:
        report["reason"] = f"Unsupported subject-scale mode: {selected_mode}"
        return out_pose, report

    measured_after = measure_segment_lengths(corrected, names)
    out_pose["joints_3d"] = corrected.astype(np.float32)
    scale_metadata = {
        "status": "ok",
        "mode": selected_mode,
        "method": method,
        "subject_height_m": height_m,
        "target_source": target_report["target_source"],
        "target_lengths_m": targets,
        "target_report": target_report,
        "global_scale": scale_report,
        "segment_lengths_before": measured_before,
        "segment_lengths_after": measured_after,
        "segment_length_errors_before_mm": segment_length_errors_mm(measured_before, targets),
        "segment_length_errors_after_mm": segment_length_errors_mm(measured_after, targets),
        "mocap_used_in_objective": False,
    }
    out_pose["subject_scale"] = scale_metadata
    return out_pose, scale_metadata


def build_targets(
    mode: str,
    subject_height_m: float,
    cfg: dict[str, Any] | None = None,
    static_pose: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    cfg = cfg or {}
    ratios = _target_ratios(cfg)
    if mode.startswith("height_"):
        targets = height_based_targets(subject_height_m, ratios)
        return targets, {"target_source": "height", "subject_height_m": float(subject_height_m), "target_ratios": ratios}
    if mode.startswith("static_"):
        if static_pose is None:
            raise ValueError("Static-trial target requested but no static pose artifact was provided.")
        static_joints = np.asarray(static_pose.get("joints_3d"), dtype=float)
        static_names = [str(name) for name in static_pose.get("joint_names", [])]
        targets, static_report = static_based_targets(static_joints, static_names, subject_height_m, ratios)
        return targets, {"target_source": "static", "subject_height_m": float(subject_height_m), **static_report}
    raise ValueError(f"Unsupported subject-scale mode: {mode}")


def height_based_targets(subject_height_m: float, ratios: dict[str, float] | None = None) -> dict[str, float]:
    height = _finite_positive(subject_height_m)
    if height is None:
        raise ValueError("Subject height must be finite and positive.")
    ratios = ratios or DEFAULT_TARGET_RATIOS
    return {segment.name: float(height * ratios[segment.target_group]) for segment in LOWER_LIMB_SEGMENTS if segment.target_group in ratios}


def static_based_targets(
    static_joints: np.ndarray,
    static_names: list[str],
    subject_height_m: float,
    ratios: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    ratios = ratios or DEFAULT_TARGET_RATIOS
    measured = measure_segment_lengths(static_joints, static_names)
    group_lengths = _group_median_lengths(measured)
    static_leg = _finite_positive(_sum_present(group_lengths, ("thigh", "shank")))
    target_leg = _finite_positive(float(subject_height_m) * float(ratios["thigh"] + ratios["shank"]))
    if static_leg is None or target_leg is None:
        raise ValueError("Static pose does not contain finite thigh and shank lengths for normalization.")
    normalization_scale = float(target_leg / static_leg)
    targets: dict[str, float] = {}
    for segment in LOWER_LIMB_SEGMENTS:
        source = measured.get(segment.name, {}).get("median_m")
        source = _finite_positive(source)
        if source is not None:
            targets[segment.name] = float(source * normalization_scale)
    if not any(name in targets for name in DEFAULT_GLOBAL_SEGMENTS):
        raise ValueError("Static pose did not produce any usable lower-limb target lengths.")
    return targets, {
        "static_segment_lengths_m": measured,
        "static_group_lengths_m": group_lengths,
        "static_leg_length_m": static_leg,
        "target_leg_length_m": target_leg,
        "static_normalization_scale": normalization_scale,
        "target_ratios": ratios,
    }


def measure_segment_lengths(joints: np.ndarray, joint_names: list[str]) -> dict[str, dict[str, Any]]:
    values = np.asarray(joints, dtype=float)
    out: dict[str, dict[str, Any]] = {}
    for segment in LOWER_LIMB_SEGMENTS:
        a = _joint_index(joint_names, segment.proximal)
        b = _joint_index(joint_names, segment.distal)
        entry: dict[str, Any] = {
            "proximal": segment.proximal,
            "distal": segment.distal,
            "target_group": segment.target_group,
            "status": "ok",
            "median_m": None,
            "mean_m": None,
            "std_m": None,
            "finite_ratio": 0.0,
            "valid_frames": 0,
        }
        if a is None or b is None:
            entry["status"] = "missing_joints"
            out[segment.name] = entry
            continue
        lengths = np.linalg.norm(values[:, a, :] - values[:, b, :], axis=1)
        finite = np.isfinite(lengths) & (lengths > 0)
        entry["finite_ratio"] = float(finite.mean()) if finite.size else 0.0
        entry["valid_frames"] = int(np.count_nonzero(finite))
        if not np.any(finite):
            entry["status"] = "no_finite_lengths"
            out[segment.name] = entry
            continue
        valid_lengths = lengths[finite]
        entry["median_m"] = float(np.nanmedian(valid_lengths))
        entry["mean_m"] = float(np.nanmean(valid_lengths))
        entry["std_m"] = float(np.nanstd(valid_lengths))
        out[segment.name] = entry
    return out


def estimate_global_scale(
    measured: dict[str, dict[str, Any]],
    targets: dict[str, float],
    include_segments: tuple[str, ...] = DEFAULT_GLOBAL_SEGMENTS,
    min_scale: float = 0.75,
    max_scale: float = 1.35,
) -> dict[str, Any]:
    ratios = []
    used = []
    for name in include_segments:
        source = _finite_positive((measured.get(name) or {}).get("median_m"))
        target = _finite_positive(targets.get(name))
        if source is None or target is None:
            continue
        ratios.append(target / source)
        used.append(name)
    if not ratios:
        return {"status": "skipped", "reason": "No finite measured/target segment pairs for global scale.", "used_segments": []}
    raw = float(np.nanmedian(np.asarray(ratios, dtype=float)))
    clipped = float(np.clip(raw, float(min_scale), float(max_scale)))
    return {
        "status": "ok",
        "scale": clipped,
        "raw_scale": raw,
        "clipped": bool(abs(clipped - raw) > 1e-12),
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
        "used_segments": used,
        "segment_scale_ratios": {name: float(ratio) for name, ratio in zip(used, ratios)},
    }


def apply_global_scale(joints: np.ndarray, joint_names: list[str], scale: float) -> np.ndarray:
    values = np.asarray(joints, dtype=float)
    out = values.copy()
    roots = _root_points(values, joint_names)
    for t in range(values.shape[0]):
        root = roots[t]
        if not np.isfinite(root).all():
            continue
        out[t] = root + float(scale) * (values[t] - root)
    return out


def retarget_lower_limb_bones(joints: np.ndarray, joint_names: list[str], targets: dict[str, float], blend: float = 1.0) -> np.ndarray:
    values = np.asarray(joints, dtype=float)
    out = values.copy()
    blend = float(np.clip(blend, 0.0, 1.0))
    for side in ("left", "right"):
        hip = _joint_index(joint_names, f"{side}_hip")
        knee = _joint_index(joint_names, f"{side}_knee")
        ankle = _joint_index(joint_names, f"{side}_ankle")
        toe = _joint_index(joint_names, f"{side}_toe")
        chain = [(hip, knee, f"{side}_thigh"), (knee, ankle, f"{side}_shank"), (ankle, toe, f"{side}_foot")]
        for t in range(values.shape[0]):
            frame_targets: dict[int, np.ndarray] = {}
            for parent_idx, child_idx, segment_name in chain:
                target = _finite_positive(targets.get(segment_name))
                if parent_idx is None or child_idx is None or target is None:
                    continue
                parent = frame_targets.get(parent_idx, out[t, parent_idx])
                child = values[t, child_idx]
                direction = _unit(child - values[t, parent_idx])
                if direction is None or not np.isfinite(parent).all():
                    continue
                frame_targets[child_idx] = parent + target * direction
            for idx, target_point in frame_targets.items():
                out[t, idx] = values[t, idx] + blend * (target_point - values[t, idx])
    return out


def segment_length_errors_mm(measured: dict[str, dict[str, Any]], targets: dict[str, float]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name, target in targets.items():
        source = _finite_positive((measured.get(name) or {}).get("median_m"))
        target_value = _finite_positive(target)
        out[name] = None if source is None or target_value is None else float((source - target_value) * 1000.0)
    return out


def _target_ratios(cfg: dict[str, Any]) -> dict[str, float]:
    out = dict(DEFAULT_TARGET_RATIOS)
    for key, value in (cfg.get("target_ratios") or {}).items():
        parsed = _finite_positive(value)
        if parsed is not None:
            out[str(key)] = parsed
    return out


def _subject_height_m(subject: dict[str, Any] | None) -> float | None:
    if not subject:
        return None
    return _finite_positive(subject.get("height_m") or subject.get("height"))


def _joint_index(names: list[str], canonical: str) -> int | None:
    return find_joint(names, JOINT_ALIASES.get(canonical, (canonical,)))


def _root_points(joints: np.ndarray, joint_names: list[str]) -> np.ndarray:
    left = _joint_index(joint_names, "left_hip")
    right = _joint_index(joint_names, "right_hip")
    pelvis = _joint_index(joint_names, "pelvis")
    roots = np.full((joints.shape[0], 3), np.nan, dtype=float)
    if left is not None and right is not None:
        roots = 0.5 * (joints[:, left, :] + joints[:, right, :])
    elif pelvis is not None:
        roots = joints[:, pelvis, :]
    return roots


def _group_median_lengths(measured: dict[str, dict[str, Any]]) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for entry in measured.values():
        value = _finite_positive(entry.get("median_m"))
        group = str(entry.get("target_group") or "")
        if value is not None and group:
            groups.setdefault(group, []).append(value)
    return {group: float(np.nanmedian(values)) for group, values in groups.items() if values}


def _sum_present(values: dict[str, float], names: tuple[str, ...]) -> float | None:
    selected = [_finite_positive(values.get(name)) for name in names]
    if any(value is None for value in selected):
        return None
    return float(sum(value for value in selected if value is not None))


def _finite_positive(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) and parsed > 0 else None


def _unit(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    return vector / norm
