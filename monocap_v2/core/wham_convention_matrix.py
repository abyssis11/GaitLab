from __future__ import annotations

import copy
import csv
import math
import os
import statistics
from itertools import permutations, product
from pathlib import Path
from typing import Any

import numpy as np

from monocap_v2.core.level_a_benchmark import compare_pose_to_opensim_reference, load_opensim_reference, load_pose_artifact, load_run_config
from monocap_v2.core.logging_utils import read_json, write_json
from monocap_v2.core.wham_conventions import axis_expr_determinant
from monocap_v2.core.wham_timeline import build_wham_timeline_report


VISUAL_CANDIDATE_PROFILE = "opencap_full_yaw180_lr_offset120ms"
DEFAULT_TRIALS = ["walking1", "walking2", "walking3"]
DEFAULT_EVALUATION_HZ = [None, 100.0]
RANKING_METRICS = {
    "normal": "primary_root_centered_mpjpe_mm",
    "rigid": "root_centered_rigid_mpjpe_mm",
    "pa": "pa_mpjpe_mm",
}


def run_wham_convention_matrix_audit(
    benchmark_dir: Path,
    trials: list[str] | None = None,
    profile_set: str = "physical",
    evaluation_hz_values: list[float | None] | None = None,
    time_offset_min: float = -0.30,
    time_offset_max: float = 0.30,
    time_offset_step: float = 0.02,
    out_dir: Path | None = None,
    max_viewer_candidates: int = 12,
) -> dict[str, Any]:
    benchmark_dir = Path(benchmark_dir)
    out_dir = Path(out_dir) if out_dir else benchmark_dir / "wham_convention_matrix_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    summary = read_json(benchmark_dir / "level_a_summary.json")
    requested_trials = trials or DEFAULT_TRIALS
    rows = [
        row
        for row in summary.get("rows", [])
        if row.get("status") == "valid"
        and row.get("backend") == "wham"
        and row.get("trial") in set(requested_trials)
    ]
    offsets = time_offsets(time_offset_min, time_offset_max, time_offset_step)
    candidates = generate_matrix_candidates(profile_set=profile_set, time_offsets=offsets)
    evaluation_hz_values = evaluation_hz_values if evaluation_hz_values is not None else DEFAULT_EVALUATION_HZ

    audit_rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for row in rows:
        trial = str(row["trial"])
        run_dir = Path(str(row["run_dir"]))
        case: dict[str, Any] = {"trial": trial, "run_dir": str(run_dir), "status": "ok", "row_count": 0}
        try:
            pose = load_pose_artifact(run_dir)
            base_run_config = load_run_config(run_dir)
            reference = load_opensim_reference(
                benchmark_dir / "reference" / f"opensim_fk_{trial}.npz",
                benchmark_dir / "reference" / f"opensim_fk_{trial}.json",
            )
            timeline = _cached_or_build_timeline(run_dir, pose, base_run_config)
            for eval_hz in evaluation_hz_values:
                for candidate in candidates:
                    audit_row = _score_candidate(candidate, pose, reference, base_run_config, timeline, trial, run_dir, eval_hz)
                    audit_rows.append(audit_row)
                    case["row_count"] += 1
        except Exception as exc:
            case.update({"status": "failed", "error": str(exc)})
            audit_rows.append({"status": "failed", "trial": trial, "run_dir": str(run_dir), "error": str(exc)})
        cases.append(case)

    aggregate = aggregate_matrix_rows(audit_rows, expected_trials=requested_trials)
    viewer_candidates = select_viewer_candidates(aggregate, candidates, max_count=max_viewer_candidates)
    report = {
        "status": "failed" if not audit_rows else aggregate["status"],
        "benchmark_dir": str(benchmark_dir),
        "audit_dir": str(out_dir),
        "profile_set": profile_set,
        "trials": requested_trials,
        "evaluation_hz_values": [_evaluation_label(value) for value in evaluation_hz_values],
        "time_offsets_s": offsets,
        "candidate_count": len(candidates),
        "row_count": len(audit_rows),
        "notes": [
            "This matrix audit is evaluation-only. It does not rewrite WHAM artifacts, SMPL vertices, markers, TRC files, or benchmark defaults.",
            "Positive time_offset_s follows the interactive viewer convention: compare WHAM at t + offset against GT at t.",
            "Candidates with left/right relabeling, non-zero timing offset, or reflection/improper linear transforms are diagnostic-only.",
        ],
        "aggregate": aggregate,
        "viewer_candidates": viewer_candidates,
        "cases": cases,
        "outputs": {
            "json": str(out_dir / "wham_convention_matrix_audit.json"),
            "summary_md": str(out_dir / "wham_convention_matrix_audit.md"),
            "csv": str(out_dir / "wham_convention_matrix_audit.csv"),
            "top_normal_csv": str(out_dir / "top_candidates_normal.csv"),
            "top_rigid_csv": str(out_dir / "top_candidates_rigid.csv"),
            "top_pa_csv": str(out_dir / "top_candidates_pa.csv"),
            "top_stable_csv": str(out_dir / "top_candidates_stable.csv"),
            "plot": str(out_dir / "plots" / "top_candidate_metrics.png"),
        },
    }
    _write_outputs(out_dir, report, audit_rows)
    return report


def generate_matrix_candidates(profile_set: str, time_offsets: list[float]) -> list[dict[str, Any]]:
    profile_set = str(profile_set or "physical").strip().lower()
    if profile_set not in {"physical", "full"}:
        raise ValueError(f"profile_set must be 'physical' or 'full', got {profile_set!r}.")
    bases = _physical_bases()
    if profile_set == "full":
        bases = bases + _signed_axis_bases()

    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for base in bases:
        for left_right_swap in (False, True):
            for offset in time_offsets:
                candidate = _candidate_from_base(base, left_right_swap=left_right_swap, time_offset_s=offset, profile_set=profile_set)
                if candidate["profile"] in seen:
                    continue
                seen.add(candidate["profile"])
                candidates.append(candidate)
    return candidates


def time_offsets(min_s: float, max_s: float, step_s: float) -> list[float]:
    min_s = float(min_s)
    max_s = float(max_s)
    step_s = float(step_s)
    if not np.isfinite(step_s) or step_s <= 0:
        raise ValueError("time_offset_step must be positive.")
    if min_s > max_s:
        raise ValueError("time_offset_min must be <= time_offset_max.")
    count = int(math.floor((max_s - min_s) / step_s + 0.5)) + 1
    values = [round(min_s + idx * step_s, 6) for idx in range(count)]
    if values and values[-1] < max_s - 1e-9:
        values.append(round(max_s, 6))
    if min_s <= 0.0 <= max_s and not any(abs(value) < 1e-12 for value in values):
        values.append(0.0)
    return sorted(set(round(float(value), 6) for value in values))


def signed_axis_expressions() -> list[str]:
    axes = ["x", "y", "z"]
    out = []
    for perm in permutations(axes):
        for signs in product((1, -1), repeat=3):
            out.append(",".join(("-" if sign < 0 else "") + axis for sign, axis in zip(signs, perm)))
    return out


def aggregate_matrix_rows(rows: list[dict[str, Any]], expected_trials: list[str] | None = None) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") != "ok"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in valid:
        key = (str(row.get("profile")), str(row.get("evaluation_label")))
        grouped.setdefault(key, []).append(row)

    summaries = []
    for (profile, evaluation_label), group in grouped.items():
        summaries.append(_summarize_candidate(profile, evaluation_label, group, expected_trials=expected_trials))

    rankings = {
        "normal": _rank_candidates(summaries, "median_primary_root_centered_mpjpe_mm"),
        "rigid": _rank_candidates(summaries, "median_root_centered_rigid_mpjpe_mm"),
        "pa": _rank_candidates(summaries, "median_pa_mpjpe_mm"),
        "stable": _rank_candidates(summaries, "stable_candidate_score_mm"),
    }
    return {
        "status": "failed" if not valid else "warning" if failed else "ok",
        "valid_row_count": len(valid),
        "failed_row_count": len(failed),
        "candidate_summary_count": len(summaries),
        "rankings": rankings,
        "recommendation": _recommendation(rankings, summaries),
    }


def select_viewer_candidates(aggregate: dict[str, Any], candidates: list[dict[str, Any]], max_count: int = 12) -> dict[str, Any]:
    lookup = {candidate["profile"]: candidate for candidate in candidates}
    selected: dict[str, dict[str, Any]] = {}
    presets: dict[str, str | None] = {}
    ranking_map = {
        "best_normal": "normal",
        "best_rigid": "rigid",
        "best_pa": "pa",
        "best_stable": "stable",
    }
    for preset_name, ranking_name in ranking_map.items():
        ranking = ((aggregate.get("rankings") or {}).get(ranking_name) or [])
        profile = str(ranking[0]["profile"]) if ranking else None
        presets[preset_name] = profile
        if profile and profile in lookup:
            selected[profile] = {**lookup[profile], "viewer_source": preset_name, "summary": ranking[0]}
    if VISUAL_CANDIDATE_PROFILE in lookup:
        visual_summary = _find_summary(aggregate, VISUAL_CANDIDATE_PROFILE)
        selected[VISUAL_CANDIDATE_PROFILE] = {
            **lookup[VISUAL_CANDIDATE_PROFILE],
            "viewer_source": "visual_candidate",
            "summary": visual_summary,
        }
        presets["visual_candidate"] = VISUAL_CANDIDATE_PROFILE
    else:
        presets["visual_candidate"] = None

    for ranking_name in ["normal", "rigid", "pa", "stable"]:
        for item in ((aggregate.get("rankings") or {}).get(ranking_name) or [])[: max(1, max_count)]:
            profile = str(item["profile"])
            if profile in lookup:
                selected.setdefault(profile, {**lookup[profile], "viewer_source": f"top_{ranking_name}", "summary": item})
            if len(selected) >= max_count:
                break
        if len(selected) >= max_count:
            break
    return {"presets": presets, "profiles": list(selected.values())}


def matrix_profiles_for_viewer(audit_path: Path, evaluation_hz: float | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, str | None]]:
    report = read_json(Path(audit_path))
    viewer = report.get("viewer_candidates") or {}
    profiles: dict[str, dict[str, Any]] = {}
    for item in viewer.get("profiles") or []:
        spec = dict(item.get("spec") or {})
        if not spec:
            continue
        spec.setdefault("source", "matrix_audit")
        spec.setdefault("diagnostic_only", bool(item.get("diagnostic_only", True)))
        spec.setdefault("label", item.get("label") or item.get("profile"))
        profiles[str(item["profile"])] = spec
    return profiles, dict(viewer.get("presets") or {})


def _score_candidate(
    candidate: dict[str, Any],
    pose: dict[str, Any],
    reference: dict[str, Any],
    base_run_config: dict[str, Any],
    timeline: dict[str, Any] | None,
    trial: str,
    run_dir: Path,
    evaluation_hz: float | None,
) -> dict[str, Any]:
    run_config = _run_config_with_candidate(base_run_config, candidate)
    try:
        report, _series = compare_pose_to_opensim_reference(
            pose,
            reference,
            run_config=run_config,
            convention_profile=candidate["profile"],
            timeline_report=timeline,
            evaluation_hz=evaluation_hz,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "trial": trial,
            "run_dir": str(run_dir),
            "profile": candidate["profile"],
            "label": candidate["label"],
            "evaluation_label": _evaluation_label(evaluation_hz),
            "evaluation_hz": evaluation_hz,
            "time_offset_s": candidate["time_offset_s"],
            "left_right_swap": candidate["left_right_swap"],
            "diagnostic_only": candidate["diagnostic_only"],
            "error": str(exc),
        }
    return {
        "status": "ok",
        "trial": trial,
        "run_dir": str(run_dir),
        "profile": candidate["profile"],
        "label": candidate["label"],
        "profile_set": candidate["profile_set"],
        "family": candidate["family"],
        "evaluation_label": _evaluation_label(evaluation_hz),
        "evaluation_hz": evaluation_hz,
        "convention_profile": report.get("convention_profile"),
        "pre_axis": report.get("pre_axis"),
        "post_axis": candidate["spec"].get("post_axis"),
        "uses_camera_rotation": candidate["spec"].get("use_camera_rotation"),
        "uses_camera_translation": report.get("uses_camera_translation"),
        "camera_rotation_source": report.get("camera_rotation_source"),
        "camera_translation_source": report.get("camera_translation_source"),
        "camera_translation_units": report.get("camera_translation_units"),
        "left_right_swap": report.get("left_right_swap"),
        "time_offset_s": report.get("time_offset_s"),
        "pre_axis_determinant": report.get("pre_axis_determinant"),
        "post_transform_determinant": report.get("post_transform_determinant"),
        "linear_transform_determinant": report.get("linear_transform_determinant"),
        "proper_post_transform": report.get("proper_post_transform"),
        "proper_linear_transform": report.get("proper_linear_transform"),
        "diagnostic_only": candidate["diagnostic_only"] or bool(report.get("diagnostic_only")),
        "promotable": _row_promotable(candidate, report),
        "overlap_frames": report.get("overlap_frames"),
        "primary_root_centered_mpjpe_mm": report.get("primary_root_centered_mpjpe_mm"),
        "root_centered_rigid_mpjpe_mm": report.get("root_centered_rigid_mpjpe_mm"),
        "root_centered_n_mpjpe_mm": report.get("root_centered_n_mpjpe_mm"),
        "root_centered_similarity_mpjpe_mm": report.get("root_centered_similarity_mpjpe_mm"),
        "pa_mpjpe_mm": report.get("pa_mpjpe_mm"),
        "global_no_align_mpjpe_mm": report.get("global_no_align_mpjpe_mm"),
        "global_sequence_similarity_mpjpe_mm": report.get("global_sequence_similarity_mpjpe_mm"),
        "normal_minus_rigid_gap_mm": report.get("normal_minus_rigid_gap_mm"),
        "warnings": "; ".join(str(w) for w in (report.get("warnings") or [])),
    }


def _run_config_with_candidate(run_config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(run_config or {})
    cfg = out.setdefault("config", {})
    level_a = cfg.setdefault("level_a", {})
    profiles = level_a.setdefault("wham_convention_profiles", {})
    profiles[candidate["profile"]] = dict(candidate["spec"])
    return out


def _candidate_from_base(base: dict[str, Any], left_right_swap: bool, time_offset_s: float, profile_set: str) -> dict[str, Any]:
    offset = round(float(time_offset_s), 6)
    profile = _profile_name(base["key"], left_right_swap, offset)
    pre_det = axis_expr_determinant(base["pre_axis"])
    post_axis = str(base.get("post_axis") or "identity")
    post_det = axis_expr_determinant(post_axis)
    linear_det_without_camera = float(pre_det * post_det)
    proper_post = _is_proper(post_det)
    proper_linear = _is_proper(linear_det_without_camera)
    diagnostic_only = bool(left_right_swap or abs(offset) > 1e-12 or not proper_post or not proper_linear or base.get("diagnostic_only", False))
    spec = {
        "label": _profile_label(base["label"], left_right_swap, offset),
        "pre_axis": base["pre_axis"],
        "use_camera_rotation": bool(base.get("use_camera_rotation", False)),
        "use_camera_translation": bool(base.get("use_camera_translation", False)),
        "post_axis": post_axis,
        "left_right_swap": bool(left_right_swap),
        "time_offset_s": offset,
        "diagnostic_only": diagnostic_only,
        "source": "matrix_audit",
    }
    return {
        "profile": profile,
        "label": spec["label"],
        "profile_set": profile_set,
        "family": base["family"],
        "spec": spec,
        "pre_axis": base["pre_axis"],
        "post_axis": post_axis,
        "time_offset_s": offset,
        "left_right_swap": bool(left_right_swap),
        "uses_camera_rotation": spec["use_camera_rotation"],
        "uses_camera_translation": spec["use_camera_translation"],
        "pre_axis_determinant": pre_det,
        "post_transform_determinant": post_det,
        "linear_transform_determinant_without_camera": linear_det_without_camera,
        "proper_post_transform": proper_post,
        "proper_linear_without_camera": proper_linear,
        "diagnostic_only": diagnostic_only,
        "promotable_by_definition": not diagnostic_only,
    }


def _physical_bases() -> list[dict[str, Any]]:
    return [
        {
            "key": "legacy",
            "label": "Legacy x,-y,z",
            "family": "physical",
            "pre_axis": "x,-y,z",
            "use_camera_rotation": False,
            "use_camera_translation": False,
            "post_axis": "identity",
        },
        {
            "key": "opencap_rot",
            "label": "OpenCap camera rotation to lab",
            "family": "physical",
            "pre_axis": "x,-y,z",
            "use_camera_rotation": True,
            "use_camera_translation": False,
            "post_axis": "identity",
        },
        {
            "key": "opencap_full",
            "label": "OpenCap full extrinsic to lab",
            "family": "physical",
            "pre_axis": "x,-y,z",
            "use_camera_rotation": True,
            "use_camera_translation": True,
            "post_axis": "identity",
            "diagnostic_only": True,
        },
        {
            "key": "opencap_yaw180",
            "label": "OpenCap camera rotation to lab + yaw 180",
            "family": "physical",
            "pre_axis": "x,-y,z",
            "use_camera_rotation": True,
            "use_camera_translation": False,
            "post_axis": "-x,y,-z",
            "diagnostic_only": True,
        },
        {
            "key": "opencap_full_yaw180",
            "label": "OpenCap full extrinsic to lab + yaw 180",
            "family": "physical",
            "pre_axis": "x,-y,z",
            "use_camera_rotation": True,
            "use_camera_translation": True,
            "post_axis": "-x,y,-z",
            "diagnostic_only": True,
        },
    ]


def _signed_axis_bases() -> list[dict[str, Any]]:
    bases = []
    for expr in signed_axis_expressions():
        suffix = _axis_slug(expr)
        bases.extend(
            [
                {
                    "key": f"axis_{suffix}",
                    "label": f"Axis {expr}",
                    "family": "signed_axis",
                    "pre_axis": expr,
                    "use_camera_rotation": False,
                    "use_camera_translation": False,
                    "post_axis": "identity",
                },
                {
                    "key": f"axis_rot_{suffix}",
                    "label": f"Axis {expr} + OpenCap rotation",
                    "family": "signed_axis_camera",
                    "pre_axis": expr,
                    "use_camera_rotation": True,
                    "use_camera_translation": False,
                    "post_axis": "identity",
                },
                {
                    "key": f"axis_full_{suffix}",
                    "label": f"Axis {expr} + OpenCap full extrinsic",
                    "family": "signed_axis_full",
                    "pre_axis": expr,
                    "use_camera_rotation": True,
                    "use_camera_translation": True,
                    "post_axis": "identity",
                    "diagnostic_only": True,
                },
            ]
        )
    return bases


def _profile_name(base_key: str, left_right_swap: bool, offset: float) -> str:
    if abs(offset) < 1e-12:
        static = {
            ("legacy", False): "legacy_x_yup_z",
            ("opencap_rot", False): "opencap_camera_yflip_to_lab",
            ("opencap_rot", True): "opencap_camera_yflip_to_lab_lr",
            ("opencap_full", False): "opencap_camera_yflip_full_to_lab",
            ("opencap_full", True): "opencap_camera_yflip_full_to_lab_lr",
            ("opencap_yaw180", False): "opencap_yaw180",
            ("opencap_yaw180", True): "opencap_yaw180_lr",
            ("opencap_full_yaw180", False): "opencap_full_yaw180",
            ("opencap_full_yaw180", True): "opencap_full_yaw180_lr",
        }
        if (base_key, left_right_swap) in static:
            return static[(base_key, left_right_swap)]
    if abs(offset - 0.12) < 1e-9 and left_right_swap:
        static_offset = {
            "opencap_yaw180": "opencap_yaw180_lr_offset120ms",
            "opencap_full_yaw180": VISUAL_CANDIDATE_PROFILE,
        }
        if base_key in static_offset:
            return static_offset[base_key]
    parts = [base_key]
    if left_right_swap:
        parts.append("lr")
    if abs(offset) >= 1e-12:
        parts.append(f"off_{_offset_slug(offset)}")
    return "_".join(parts)


def _profile_label(base_label: str, left_right_swap: bool, offset: float) -> str:
    label = base_label
    if left_right_swap:
        label += " + L/R"
    if abs(offset) >= 1e-12:
        label += f" + {offset:+.3f} s"
    return label


def _summarize_candidate(profile: str, evaluation_label: str, rows: list[dict[str, Any]], expected_trials: list[str] | None = None) -> dict[str, Any]:
    first = rows[0]
    primary = _values(rows, "primary_root_centered_mpjpe_mm")
    rigid = _values(rows, "root_centered_rigid_mpjpe_mm")
    pa = _values(rows, "pa_mpjpe_mm")
    gap = [abs(value) for value in _values(rows, "normal_minus_rigid_gap_mm")]
    valid_trials = sorted({str(row.get("trial")) for row in rows})
    expected_count = len(expected_trials or valid_trials)
    median_primary = _median(primary)
    iqr_primary = _iqr(primary)
    median_gap = _median(gap)
    stable_score = None
    if median_primary is not None:
        stable_score = float(median_primary + (iqr_primary or 0.0) + (median_gap or 0.0))
    promotable = (
        bool(first.get("promotable"))
        and len(valid_trials) == expected_count
        and all(bool(row.get("promotable")) for row in rows)
        and all(not row.get("warnings") for row in rows)
    )
    return {
        "profile": profile,
        "label": first.get("label"),
        "evaluation_label": evaluation_label,
        "evaluation_hz": first.get("evaluation_hz"),
        "valid_trial_count": len(valid_trials),
        "expected_trial_count": expected_count,
        "valid_trials": valid_trials,
        "median_primary_root_centered_mpjpe_mm": median_primary,
        "iqr_primary_root_centered_mpjpe_mm": iqr_primary,
        "median_root_centered_rigid_mpjpe_mm": _median(rigid),
        "median_pa_mpjpe_mm": _median(pa),
        "median_global_no_align_mpjpe_mm": _median(_values(rows, "global_no_align_mpjpe_mm")),
        "median_abs_normal_minus_rigid_gap_mm": median_gap,
        "stable_candidate_score_mm": stable_score,
        "left_right_swap": bool(first.get("left_right_swap")),
        "time_offset_s": first.get("time_offset_s"),
        "uses_camera_translation": bool(first.get("uses_camera_translation")),
        "diagnostic_only": bool(first.get("diagnostic_only")),
        "promotable": promotable,
        "proper_post_transform": bool(first.get("proper_post_transform")),
        "proper_linear_transform": bool(first.get("proper_linear_transform")),
        "linear_transform_determinant": first.get("linear_transform_determinant"),
    }


def _rank_candidates(summaries: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    valid = [item for item in summaries if item.get(metric) is not None]
    valid.sort(key=lambda item: float(item[metric]))
    return valid


def _recommendation(rankings: dict[str, list[dict[str, Any]]], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    promotable = [item for item in summaries if item.get("promotable")]
    promotable.sort(key=lambda item: float(item.get("stable_candidate_score_mm") or float("inf")))
    best_pa = (rankings.get("pa") or [None])[0]
    best_rigid = (rankings.get("rigid") or [None])[0]
    visual = _find_summary({"rankings": rankings}, VISUAL_CANDIDATE_PROFILE)
    notes = [
        "Do not promote candidates selected only by GT-scored timing offset, L/R relabel, or reflection/improper transforms.",
        "Low rigid/PA MPJPE indicates good local pose shape after fitted alignment; normal MPJPE remains the convention-sensitive metric.",
    ]
    if not promotable:
        notes.append("No candidate satisfies the promotable criteria in this audit.")
    return {
        "promotable_count": len(promotable),
        "top_promotable": promotable[0] if promotable else None,
        "best_pa": best_pa,
        "best_rigid": best_rigid,
        "visual_candidate": visual,
        "notes": notes,
    }


def _row_promotable(candidate: dict[str, Any], report: dict[str, Any]) -> bool:
    return bool(
        not candidate.get("diagnostic_only")
        and not candidate.get("left_right_swap")
        and abs(float(candidate.get("time_offset_s") or 0.0)) < 1e-12
        and report.get("proper_linear_transform") is True
        and report.get("proper_post_transform") is True
    )


def _write_outputs(out_dir: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    write_json(out_dir / "wham_convention_matrix_audit.json", report)
    _write_csv(out_dir / "wham_convention_matrix_audit.csv", rows)
    for name, ranking_name in [("normal", "normal"), ("rigid", "rigid"), ("pa", "pa"), ("stable", "stable")]:
        _write_csv(out_dir / f"top_candidates_{name}.csv", ((report.get("aggregate") or {}).get("rankings") or {}).get(ranking_name, [])[:20])
    _write_md(out_dir / "wham_convention_matrix_audit.md", report)
    _write_plot(out_dir / "plots" / "top_candidate_metrics.png", report)


def _write_md(path: Path, report: dict[str, Any]) -> None:
    aggregate = report.get("aggregate") or {}
    rankings = aggregate.get("rankings") or {}
    recommendation = aggregate.get("recommendation") or {}
    with path.open("w", encoding="utf-8") as f:
        f.write("# WHAM Convention Matrix Audit\n\n")
        f.write(f"- Status: `{report.get('status')}`\n")
        f.write(f"- Profile set: `{report.get('profile_set')}`\n")
        f.write(f"- Trials: `{', '.join(report.get('trials') or [])}`\n")
        f.write(f"- Evaluation Hz: `{', '.join(report.get('evaluation_hz_values') or [])}`\n")
        f.write(f"- Candidates: `{report.get('candidate_count')}`\n")
        f.write("- Scope: evaluation-only; no WHAM/SMPL/marker/TRC artifacts or defaults are changed.\n\n")
        f.write("## Recommendation\n\n")
        f.write(f"- Promotable candidates: `{recommendation.get('promotable_count')}`\n")
        top_promotable = recommendation.get("top_promotable")
        if top_promotable:
            f.write(f"- Top promotable: `{top_promotable.get('profile')}` with stable score `{_fmt(top_promotable.get('stable_candidate_score_mm'))}`\n")
        else:
            f.write("- Top promotable: `None`\n")
        visual = recommendation.get("visual_candidate") or {}
        if visual:
            f.write(
                f"- Visual candidate `{VISUAL_CANDIDATE_PROFILE}`: normal `{_fmt(visual.get('median_primary_root_centered_mpjpe_mm'))} mm`, "
                f"rigid `{_fmt(visual.get('median_root_centered_rigid_mpjpe_mm'))} mm`, PA `{_fmt(visual.get('median_pa_mpjpe_mm'))} mm`, "
                f"offset `{_fmt(visual.get('time_offset_s'))} s`.\n"
            )
        for note in recommendation.get("notes") or []:
            f.write(f"- {note}\n")
        f.write("\n## Rankings\n\n")
        for title, ranking_name, metric in [
            ("Normal", "normal", "median_primary_root_centered_mpjpe_mm"),
            ("Rigid", "rigid", "median_root_centered_rigid_mpjpe_mm"),
            ("PA", "pa", "median_pa_mpjpe_mm"),
            ("Stable", "stable", "stable_candidate_score_mm"),
        ]:
            f.write(f"### {title}\n\n")
            f.write("| Rank | Profile | Eval | Metric | Rigid | PA | Offset | LR | Diagnostic | Promotable |\n")
            f.write("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for idx, item in enumerate((rankings.get(ranking_name) or [])[:12], start=1):
                f.write(
                    f"| {idx} | `{item.get('profile')}` | {item.get('evaluation_label')} | {_fmt(item.get(metric))} | "
                    f"{_fmt(item.get('median_root_centered_rigid_mpjpe_mm'))} | {_fmt(item.get('median_pa_mpjpe_mm'))} | "
                    f"{_fmt(item.get('time_offset_s'))} | {item.get('left_right_swap')} | {item.get('diagnostic_only')} | {item.get('promotable')} |\n"
                )
            f.write("\n")


def _write_plot(path: Path, report: dict[str, Any]) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-monocap-v2")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    rankings = ((report.get("aggregate") or {}).get("rankings") or {})
    rows = (rankings.get("pa") or [])[:8]
    if not rows:
        return
    labels = [str(row.get("profile"))[:34] for row in rows]
    x = np.arange(len(rows))
    normal = [float(row.get("median_primary_root_centered_mpjpe_mm") or np.nan) for row in rows]
    rigid = [float(row.get("median_root_centered_rigid_mpjpe_mm") or np.nan) for row in rows]
    pa = [float(row.get("median_pa_mpjpe_mm") or np.nan) for row in rows]
    fig, ax = plt.subplots(figsize=(11, 5), dpi=140)
    ax.plot(x, normal, marker="o", label="normal")
    ax.plot(x, rigid, marker="o", label="rigid")
    ax.plot(x, pa, marker="o", label="PA")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Median MPJPE (mm)")
    ax.set_title("Top PA-ranked WHAM convention candidates")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _cached_or_build_timeline(run_dir: Path, pose: dict[str, Any], run_config: dict[str, Any]) -> dict[str, Any] | None:
    for candidate in _run_dir_candidates(run_dir):
        for qc_path in [candidate / "reports" / "wham_timeline_qc.json", candidate / "pose3d_initial" / "pose3d_initial_qc.json"]:
            if not qc_path.exists():
                continue
            qc = read_json(qc_path)
            timeline = qc if "raw_sync_alignment" in qc else qc.get("wham_timeline") or qc.get("wham_timeline_report")
            if isinstance(timeline, dict) and isinstance(timeline.get("raw_sync_alignment"), dict):
                return timeline
    return build_wham_timeline_report(pose, run_config)


def _run_dir_candidates(run_dir: Path) -> list[Path]:
    candidates = [run_dir]
    name = run_dir.name
    if "__" in name:
        candidates.append(run_dir.with_name(name.split("__", 1)[0]))
    return candidates


def _find_summary(aggregate: dict[str, Any], profile: str) -> dict[str, Any] | None:
    for ranking in ((aggregate.get("rankings") or {}).values()):
        for item in ranking or []:
            if item.get("profile") == profile:
                return item
    return None


def _values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            value_f = float(value)
        except Exception:
            continue
        if np.isfinite(value_f):
            out.append(value_f)
    return out


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _iqr(values: list[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return float(np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25))


def _evaluation_label(evaluation_hz: float | None) -> str:
    if evaluation_hz is None:
        return "native"
    value = float(evaluation_hz)
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}hz"
    return f"{value:g}hz"


def _axis_slug(expr: str) -> str:
    return str(expr).replace("-", "m").replace(",", "_").replace("+", "")


def _offset_slug(offset: float) -> str:
    ms = int(round(abs(float(offset)) * 1000.0))
    return ("p" if offset > 0 else "n") + f"{ms}ms"


def _is_proper(value: float) -> bool:
    return bool(np.isfinite(value) and abs(float(value) - 1.0) < 1e-6)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return str(value)
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)
