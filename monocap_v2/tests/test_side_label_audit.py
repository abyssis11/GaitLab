from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from monocap_v2.core import side_label_audit as audit


def test_side_label_audit_prefers_no_swap_when_video_model_and_gt_agree() -> None:
    time, names, pred, ref, video = _synthetic_case("no_swap")

    decision = audit.evaluate_side_label_mappings(time, names, pred, ref, video_signals=video)

    assert decision["recommendation"] == "both_or_neither_equivalent"
    assert decision["best_mapping"] == "no_swap"


def test_side_label_audit_detects_model_side_swap() -> None:
    time, names, pred, ref, video = _synthetic_case("model_swapped")

    decision = audit.evaluate_side_label_mappings(time, names, pred, ref, video_signals=video)

    assert decision["recommendation"] == "model_swap_likely"
    assert decision["best_mapping"] == "swap_model_only"


def test_side_label_audit_detects_gt_side_swap() -> None:
    time, names, pred, ref, video = _synthetic_case("gt_swapped")

    decision = audit.evaluate_side_label_mappings(time, names, pred, ref, video_signals=video)

    assert decision["recommendation"] == "gt_swap_likely"
    assert decision["best_mapping"] == "swap_gt_only"


def test_side_label_audit_prefers_no_swap_when_both_and_neither_are_equivalent() -> None:
    rows = [
        {"mapping": "no_swap", "evidence_score": 0.2},
        {"mapping": "swap_model_only", "evidence_score": -1.0},
        {"mapping": "swap_gt_only", "evidence_score": -1.0},
        {"mapping": "swap_both", "evidence_score": 3.0},
    ]

    recommendation = audit.recommend_side_label_mapping(rows)

    assert recommendation["recommendation"] == "both_or_neither_equivalent"
    assert recommendation["highest_scoring_mapping"] == "swap_both"
    assert recommendation["best_mapping"] == "no_swap"


def test_walking_direction_handedness_detects_inverted_hip_labels() -> None:
    time, names, _pred, ref, _video = _synthetic_case("no_swap")
    direction = audit.walking_direction(ref, names)

    good = audit.hip_lateral_geometry(ref, names, np.asarray(direction["forward"], dtype=float))
    bad = audit.hip_lateral_geometry(audit.swap_side_values(ref, names), names, np.asarray(direction["forward"], dtype=float))

    assert direction["status"] == "ok"
    assert good["preference"] == "labels_consistent"
    assert good["right_axis_dot"] > 0.9
    assert bad["preference"] == "labels_inverted"
    assert bad["right_axis_dot"] < -0.9


def test_video_side_signals_skip_cleanly_when_pose2d_missing() -> None:
    signals, report = audit.video_side_signals_for_level_a_times(
        {"backend": "metrabs", "joints_3d": np.zeros((5, 6, 3)), "joint_names": []},
        {},
        None,
        np.arange(5) / 60.0,
        0.0,
    )

    assert signals is None
    assert report["status"] == "skipped"
    assert "pose2d" in report["reason"]


def test_side_label_runner_writes_outputs_from_fake_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = tmp_path / "bench"
    bench.mkdir()
    run_dir = tmp_path / "run__metrabs"
    _write_level_a_summary(bench, run_dir)
    summary_csv = tmp_path / "best_pa.csv"
    _write_best_pa(summary_csv)
    time, names, pred, ref, video = _synthetic_case("model_swapped")

    monkeypatch.setattr(audit, "load_pose_artifact", lambda _path: {"backend": "metrabs", "joint_names": names, "joints_3d": pred})
    monkeypatch.setattr(audit, "load_run_config", lambda _path: {})
    monkeypatch.setattr(audit, "load_cached_wham_timeline", lambda _path: None)
    monkeypatch.setattr(audit, "load_opensim_reference", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(audit, "video_side_signals_for_level_a_times", lambda *_args, **_kwargs: (video, {"status": "ok"}))

    def fake_compare(
        _pose,
        _reference,
        _run_config,
        _timeline,
        _backend,
        _axis,
        _hz,
        _offset,
        swap_model: bool,
        swap_reference: bool,
    ):
        pred_item = audit.swap_side_values(pred, names) if swap_model else pred
        ref_item = audit.swap_side_values(ref, names) if swap_reference else ref
        report = {
            "status": "ok",
            "primary_root_centered_mpjpe_mm": 100.0,
            "root_centered_rigid_mpjpe_mm": 50.0,
            "pa_mpjpe_mm": 25.0,
            "overlap_frames": len(time),
            "warnings": [],
        }
        series = {
            "time_s": time,
            "joint_names": np.asarray(names, dtype=object),
            "prediction_eval_m": pred_item,
            "reference_eval_m": ref_item,
        }
        return report, series

    monkeypatch.setattr(audit, "_compare_mapping", fake_compare)

    report = audit.run_side_label_audit(
        summary_csv=summary_csv,
        benchmark_dirs={"Cam0": bench},
        out_dir=tmp_path / "side_labels",
        camera="Cam0",
        trial="walking2",
        backend="metrabs",
        evaluation_hz=100.0,
        write_plots=False,
    )

    out_dir = tmp_path / "side_labels"
    assert report["case_count"] == 1
    assert report["recommendation_counts"]["model_swap_likely"] == 1
    assert (out_dir / "side_label_rows.csv").exists()
    assert (out_dir / "side_label_summary.md").exists()
    assert (out_dir / "side_label_Cam0_walking2_metrabs_100hz.json").exists()
    rows = list(csv.DictReader((out_dir / "side_label_rows.csv").open()))
    assert len(rows) == 4
    assert {row["mapping"] for row in rows} == {"no_swap", "swap_model_only", "swap_gt_only", "swap_both"}
    case = json.loads((out_dir / "side_label_Cam0_walking2_metrabs_100hz.json").read_text())
    assert case["recommendation"] == "model_swap_likely"


def _synthetic_case(kind: str) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, dict]:
    time = np.linspace(0.0, 1.0, 41)
    names = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    ref = _walking_values(time)
    pred = ref.copy()
    if kind == "model_swapped":
        pred = audit.swap_side_values(pred, names)
    elif kind == "gt_swapped":
        ref = audit.swap_side_values(ref, names)
    elif kind != "no_swap":
        raise ValueError(kind)
    direction = audit.walking_direction(_walking_values(time), names)
    video = audit.forward_motion_signals(_walking_values(time), names, time, np.asarray(direction["forward"], dtype=float), source="synthetic_video")
    return time, names, pred, ref, video


def _walking_values(time: np.ndarray) -> np.ndarray:
    out = np.zeros((len(time), 6, 3), dtype=float)
    root = np.stack([time, np.zeros_like(time), np.zeros_like(time)], axis=1)
    lateral_left = np.asarray([0.0, 0.0, 0.12])
    lateral_right = np.asarray([0.0, 0.0, -0.12])
    phase = 0.18 * np.sin(2 * np.pi * time)
    out[:, 0, :] = root + lateral_left
    out[:, 1, :] = root + lateral_right
    out[:, 2, :] = root + lateral_left + np.stack([phase, -0.45 * np.ones_like(time), np.zeros_like(time)], axis=1)
    out[:, 3, :] = root + lateral_right + np.stack([-phase, -0.45 * np.ones_like(time), np.zeros_like(time)], axis=1)
    out[:, 4, :] = root + lateral_left + np.stack([phase * 1.2, -0.9 * np.ones_like(time), np.zeros_like(time)], axis=1)
    out[:, 5, :] = root + lateral_right + np.stack([-phase * 1.2, -0.9 * np.ones_like(time), np.zeros_like(time)], axis=1)
    return out


def _write_level_a_summary(bench: Path, run_dir: Path) -> None:
    payload = {"rows": [{"backend": "metrabs", "trial": "walking2", "status": "valid", "run_dir": str(run_dir)}]}
    (bench / "level_a_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_best_pa(path: Path) -> None:
    fields = [
        "camera",
        "trial",
        "backend",
        "evaluation_hz",
        "ranking",
        "axis",
        "time_offset_s",
        "reference_left_right_swap",
        "model_left_right_swap",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "camera": "Cam0",
                "trial": "walking2",
                "backend": "metrabs",
                "evaluation_hz": "100.0",
                "ranking": "best_pa",
                "axis": "x,-y,-z",
                "time_offset_s": "-0.22",
                "reference_left_right_swap": "False",
                "model_left_right_swap": "False",
            }
        )
