from __future__ import annotations

from monocap_v2.core.wham_gt_interactive import build_interactive_html


def test_build_interactive_html_contains_controls_and_labels() -> None:
    payload = {
        "title": "WHAM vs OpenSim FK GT | walking1",
        "trial": "walking1",
        "backend": "wham",
        "time_s": [0.0, 0.01667],
        "joint_names": ["hip_midpoint", "left_hip", "left_knee"],
        "short_labels": ["PELV", "LHIP", "LKNE"],
        "edges": [[0, 1], [1, 2]],
        "lr_swap_pairs": [],
        "gt": [
            [[0.0, 0.0, 0.0], [0.1, -0.1, 0.0], [0.1, -0.5, 0.0]],
            [[0.0, 0.0, 0.0], [0.1, -0.1, 0.0], [0.1, -0.5, 0.0]],
        ],
        "wham": {
            "raw": [
                [[0.0, 0.0, 0.0], [0.1, -0.1, 0.0], [0.1, -0.5, 0.0]],
                [[0.0, 0.0, 0.0], [0.1, -0.1, 0.0], [0.1, -0.5, 0.0]],
            ],
            "rigid": [
                [[0.0, 0.0, 0.0], [0.1, -0.1, 0.0], [0.1, -0.5, 0.0]],
                [[0.0, 0.0, 0.0], [0.1, -0.1, 0.0], [0.1, -0.5, 0.0]],
            ],
            "pa": [
                [[0.0, 0.0, 0.0], [0.1, -0.1, 0.0], [0.1, -0.5, 0.0]],
                [[0.0, 0.0, 0.0], [0.1, -0.1, 0.0], [0.1, -0.5, 0.0]],
            ],
        },
        "wham_conventions": {
            "legacy_x_yup_z": {
                "raw": [
                    [[0.0, 0.0, 0.0], [0.1, -0.1, 0.0], [0.1, -0.5, 0.0]],
                    [[0.0, 0.0, 0.0], [0.1, -0.1, 0.0], [0.1, -0.5, 0.0]],
                ],
                "rigid": [
                    [[0.0, 0.0, 0.0], [0.1, -0.1, 0.0], [0.1, -0.5, 0.0]],
                    [[0.0, 0.0, 0.0], [0.1, -0.1, 0.0], [0.1, -0.5, 0.0]],
                ],
                "pa": [
                    [[0.0, 0.0, 0.0], [0.1, -0.1, 0.0], [0.1, -0.5, 0.0]],
                    [[0.0, 0.0, 0.0], [0.1, -0.1, 0.0], [0.1, -0.5, 0.0]],
                ],
            }
        },
        "convention_profiles": [
            {"profile": "legacy_x_yup_z", "label": "Legacy x,-y,z", "diagnostic_only": False, "status": "ok"}
        ],
        "default_convention_profile": "legacy_x_yup_z",
        "metrics_by_convention": {
            "legacy_x_yup_z": {
                "primary_root_centered_mpjpe_mm": 0.0,
                "root_centered_rigid_mpjpe_mm": 0.0,
                "pa_mpjpe_mm": 0.0,
                "normal_minus_rigid_gap_mm": 0.0,
                "time_offset_s": 0.0,
                "post_transform_determinant": 1.0,
                "warnings": ["synthetic warning"],
            }
        },
        "matrix_presets": {
            "best_normal": "legacy_x_yup_z",
            "best_rigid": "legacy_x_yup_z",
            "best_pa": "legacy_x_yup_z",
            "visual_candidate": "legacy_x_yup_z",
        },
        "bounds": {"min": [-1, -1, -1], "max": [1, 1, 1]},
        "default_speed": 0.25,
        "lr_swap_diagnostics": {
            "raw_default_mpjpe_mm": 0.0,
            "raw_with_one_side_lr_swapped_mpjpe_mm": 0.0,
        },
        "mirror_diagnostics": [
            {"label": "none", "raw_mpjpe_mm": 0.0},
            {"label": "mirror X", "raw_mpjpe_mm": 1.0},
        ],
        "yaw_diagnostics": {
            "axis": "OpenSim_Y_up",
            "default_raw_mpjpe_mm": 0.0,
            "best_yaw_deg": 0.0,
            "best_yaw_raw_mpjpe_mm": 0.0,
            "improvement_mm": 0.0,
            "top": [],
        },
        "phase_diagnostics": {
            "best_frame_offset": 0,
            "best_raw_mpjpe_mm": 0.0,
            "top": [],
        },
        "metrics": {
            "primary_root_centered_mpjpe_mm": 0.0,
            "root_centered_rigid_mpjpe_mm": 0.0,
            "pa_mpjpe_mm": 0.0,
            "normal_minus_rigid_gap_mm": 0.0,
            "warnings": ["synthetic warning"],
        },
        "timebase": {"mode": "synthetic"},
    }

    html = build_interactive_html(payload)

    assert "WHAM vs OpenSim FK GT" in html
    assert "Play" in html
    assert "0.25x" in html
    assert "Sequence rigid" in html
    assert "PA per-frame" in html
    assert "Convention" in html
    assert "legacy_x_yup_z" in html
    assert "Orbit" in html
    assert "Reset View" in html
    assert "Preset Rigid XZ" in html
    assert "Preset Raw Yaw" in html
    assert "Best Normal" in html
    assert "Best Rigid" in html
    assert "Best PA" in html
    assert "Visual Candidate" in html
    assert "Reset Fixes" in html
    assert "pointerdown" in html
    assert "wheel" in html
    assert "Swap WHAM L/R" in html
    assert "Swap GT L/R" in html
    assert "Raw L/R diagnostic" in html
    assert "Mirror WHAM X" in html
    assert "Mirror WHAM Y" in html
    assert "Mirror WHAM Z" in html
    assert "Reverse WHAM time" in html
    assert "Raw mirror diagnostic" in html
    assert "WHAM yaw" in html
    assert "Raw yaw diagnostic" in html
    assert "Best yaw" in html
    assert "WHAM offset" in html
    assert "Offset" in html
    assert "Post det" in html
    assert "Raw phase diagnostic" in html
    assert "Side Phase" in html
    assert "Labels" in html
    assert "LHIP" in html
    assert "synthetic warning" in html
