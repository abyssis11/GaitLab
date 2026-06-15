from __future__ import annotations

import numpy as np

from monocap_v2.core.side_phase_audit import (
    best_side_mapping,
    compare_side_signal_sets,
    lead_side_estimate,
    projection_nearest_side_check,
    side_mapping_rows,
)


def test_side_phase_agreement_prefers_no_swap_when_wham_and_gt_match_video() -> None:
    video = _signals([0, 3, 1, 0, 1], [0, 0, 1, 3, 1])
    wham = _signals([0, 2.9, 1.1, 0, 1], [0, 0.1, 0.9, 2.8, 1])
    gt = _signals([0, 3.1, 1.0, 0, 1], [0, 0.0, 1.0, 3.0, 1])

    rows = side_mapping_rows(video, wham, gt)
    best = best_side_mapping(rows)

    assert best is not None
    assert best["mapping"] == "no_swap"
    assert compare_side_signal_sets(video, wham)["preferred"] == "same_labels"


def test_side_phase_agreement_detects_wham_3d_reversal() -> None:
    video = _signals([0, 3, 1, 0, 1], [0, 0, 1, 3, 1])
    wham_reversed = _signals([0, 0, 1, 3, 1], [0, 3, 1, 0, 1])
    gt = _signals([0, 3, 1, 0, 1], [0, 0, 1, 3, 1])

    best = best_side_mapping(side_mapping_rows(video, wham_reversed, gt))

    assert best is not None
    assert best["mapping"] == "swap_wham_3d_only"


def test_side_phase_agreement_detects_gt_reversal() -> None:
    video = _signals([0, 3, 1, 0, 1], [0, 0, 1, 3, 1])
    wham = _signals([0, 3, 1, 0, 1], [0, 0, 1, 3, 1])
    gt_reversed = _signals([0, 0, 1, 3, 1], [0, 3, 1, 0, 1])

    best = best_side_mapping(side_mapping_rows(video, wham, gt_reversed))

    assert best is not None
    assert best["mapping"] == "swap_gt_only"


def test_projection_nearest_side_check_detects_same_side_and_opposite_side() -> None:
    wham = np.asarray(
        [
            [[10, 20], [30, 20], [11, 40], [31, 40]],
            [[12, 22], [32, 22], [13, 42], [33, 42]],
        ],
        dtype=float,
    )
    same_gt = wham + np.asarray([1.0, 0.0])
    valid = np.ones((2, 4), dtype=bool)

    same = projection_nearest_side_check(wham, same_gt, valid)
    opposite = projection_nearest_side_check(wham, same_gt[:, [1, 0, 3, 2], :], valid)

    assert same["preference"] == "same_side"
    assert same["same_side_closer_ratio"] == 1.0
    assert opposite["preference"] == "opposite_side"
    assert opposite["same_side_closer_ratio"] == 0.0


def test_lead_side_estimate_reports_earlier_motion_peak() -> None:
    time = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)
    signals = _signals([0, 4, 1, 0, 0], [0, 0, 1, 4, 0])

    estimate = lead_side_estimate(signals, time)

    assert estimate["side"] == "left"
    assert estimate["left_minus_right_peak_time_s"] < 0.0


def _signals(left: list[float], right: list[float]) -> dict:
    return {
        "left_signal": np.asarray(left, dtype=float),
        "right_signal": np.asarray(right, dtype=float),
        "source": "synthetic",
    }
