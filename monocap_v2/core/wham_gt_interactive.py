from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from monocap_v2.core.level_a_benchmark import compare_pose_to_opensim_reference, load_opensim_reference, load_pose_artifact, load_run_config
from monocap_v2.core.logging_utils import read_json, write_json
from monocap_v2.core.wham_convention_matrix import matrix_profiles_for_viewer
from monocap_v2.core.wham_conventions import LEGACY_PROFILE, wham_convention_profile_names, wham_convention_profile_specs


EDGES = [
    ("hip_midpoint", "left_hip"),
    ("hip_midpoint", "right_hip"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_hip", "right_hip"),
]


def render_wham_gt_interactive(
    benchmark_dir: Path,
    trial: str,
    out_html: Path,
    out_json: Path,
    default_speed: float = 0.25,
    evaluation_hz: float | None = None,
    matrix_audit_path: Path | None = None,
) -> dict[str, Any]:
    payload = build_wham_gt_payload(
        benchmark_dir,
        trial,
        default_speed=default_speed,
        evaluation_hz=evaluation_hz,
        matrix_audit_path=matrix_audit_path,
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(build_interactive_html(payload), encoding="utf-8")
    report = {
        "status": "ok",
        "trial": trial,
        "backend": "wham",
        "frames": len(payload["time_s"]),
        "joint_names": payload["joint_names"],
        "default_speed": default_speed,
        "evaluation_hz": evaluation_hz,
        "matrix_audit_path": str(matrix_audit_path) if matrix_audit_path else None,
        "outputs": {"html": str(out_html), "qc": str(out_json)},
        "metrics": payload["metrics"],
        "lr_swap_diagnostics": payload.get("lr_swap_diagnostics"),
        "mirror_diagnostics": payload.get("mirror_diagnostics"),
        "yaw_diagnostics": payload.get("yaw_diagnostics"),
        "phase_diagnostics": payload.get("phase_diagnostics"),
        "side_phase_audit": payload.get("side_phase_audit"),
        "notes": [
            "Interactive HTML is self-contained and can be opened directly in a browser.",
            "Labels refer to Level A lower-limb joint-center markers, not raw optical marker labels.",
            "Yaw, swap, mirror, and reverse-time controls are visual diagnostics only and do not change saved artifacts or benchmark rankings.",
        ],
    }
    write_json(out_json, report)
    return report


def build_wham_gt_payload(
    benchmark_dir: Path,
    trial: str,
    default_speed: float = 0.25,
    evaluation_hz: float | None = None,
    matrix_audit_path: Path | None = None,
) -> dict[str, Any]:
    benchmark_dir = Path(benchmark_dir)
    summary = read_json(benchmark_dir / "level_a_summary.json")
    row_lookup = {(row.get("backend"), row.get("trial")): row for row in summary.get("rows", [])}
    row = row_lookup.get(("wham", trial))
    if not row or row.get("status") != "valid":
        raise ValueError(f"No valid WHAM Level A row found for trial {trial!r}.")
    run_dir = Path(str(row["run_dir"]))
    pose = load_pose_artifact(run_dir)
    reference = load_opensim_reference(
        benchmark_dir / "reference" / f"opensim_fk_{trial}.npz",
        benchmark_dir / "reference" / f"opensim_fk_{trial}.json",
    )
    run_config = load_run_config(run_dir)
    matrix_profiles, matrix_presets = _load_matrix_profiles(matrix_audit_path, evaluation_hz)
    viewer_run_config = _run_config_with_matrix_profiles(run_config, matrix_profiles)
    timeline = _cached_wham_timeline(run_dir)
    report, series = compare_pose_to_opensim_reference(
        pose,
        reference,
        run_config=run_config,
        timeline_report=timeline,
        evaluation_hz=evaluation_hz,
    )
    root_name = str(series.get("root_name") or "pelvis")
    names = [root_name] + [str(name) for name in series["joint_names"].tolist()]
    gt = _add_pelvis(np.asarray(series["reference_root_centered_m"], dtype=float))
    modes = {
        "raw": _add_pelvis(np.asarray(series["prediction_root_centered_m"], dtype=float)),
        "rigid": _add_pelvis(np.asarray(series["prediction_root_centered_rigid_m"], dtype=float)),
        "pa": _add_pelvis(np.asarray(series["prediction_pa_root_centered_m"], dtype=float)),
    }
    wham_conventions, metrics_by_convention, convention_options = _convention_payloads(
        pose,
        reference,
        viewer_run_config,
        timeline,
        report,
        modes,
        evaluation_hz=evaluation_hz,
    )
    wham_conventions_json = {
        profile: {key: _round_array(value, 5).tolist() for key, value in profile_modes.items()}
        for profile, profile_modes in wham_conventions.items()
    }
    lr_swap_pairs = _lr_swap_pairs(names)
    bounds = _bounds([gt, *[value for profile_modes in wham_conventions.values() for value in profile_modes.values()]])
    return {
        "title": f"WHAM vs OpenSim FK GT | {trial}",
        "trial": trial,
        "backend": "wham",
        "time_s": _round_array(series["time_s"], 5).tolist(),
        "joint_names": names,
        "short_labels": [_short_label(name) for name in names],
        "edges": _edge_indices(names),
        "lr_swap_pairs": lr_swap_pairs,
        "gt": _round_array(gt, 5).tolist(),
        "wham": {key: _round_array(value, 5).tolist() for key, value in modes.items()},
        "wham_conventions": wham_conventions_json,
        "convention_profiles": convention_options,
        "default_convention_profile": str(report.get("convention_profile") or LEGACY_PROFILE),
        "metrics_by_convention": metrics_by_convention,
        "matrix_presets": matrix_presets,
        "matrix_audit_path": str(matrix_audit_path) if matrix_audit_path else None,
        "bounds": bounds,
        "default_speed": float(default_speed),
        "evaluation_hz": float(evaluation_hz) if evaluation_hz is not None else None,
        "lr_swap_diagnostics": {
            "raw_default_mpjpe_mm": _raw_mpjpe_mm(modes["raw"], gt),
            "raw_with_one_side_lr_swapped_mpjpe_mm": _raw_mpjpe_mm(_swap_lr_values(modes["raw"], lr_swap_pairs), gt),
            "note": "Swapping WHAM or GT left/right gives the same pairwise correspondence test; use OpenSim trust and visual context to decide which mapping to fix.",
        },
        "mirror_diagnostics": _mirror_diagnostics(modes["raw"], gt),
        "yaw_diagnostics": _yaw_diagnostics(modes["raw"], gt),
        "phase_diagnostics": _phase_diagnostics(modes["raw"], gt),
        "side_phase_audit": _load_side_phase_audit(benchmark_dir, trial),
        "metrics": {
            "primary_root_centered_mpjpe_mm": report.get("primary_root_centered_mpjpe_mm"),
            "root_centered_rigid_mpjpe_mm": report.get("root_centered_rigid_mpjpe_mm"),
            "pa_mpjpe_mm": report.get("pa_mpjpe_mm"),
            "global_no_align_mpjpe_mm": report.get("global_no_align_mpjpe_mm"),
            "normal_minus_rigid_gap_mm": report.get("normal_minus_rigid_gap_mm"),
            "time_offset_s": report.get("time_offset_s"),
            "post_transform_determinant": report.get("post_transform_determinant"),
            "linear_transform_determinant": report.get("linear_transform_determinant"),
            "convention_profile": report.get("convention_profile"),
            "warnings": report.get("warnings"),
        },
        "timebase": report.get("timebase"),
    }


def _convention_payloads(
    pose: dict[str, Any],
    reference: dict[str, Any],
    run_config: dict[str, Any],
    timeline: dict[str, Any] | None,
    default_report: dict[str, Any],
    default_modes: dict[str, np.ndarray],
    evaluation_hz: float | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    wham_sets: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    options: list[dict[str, Any]] = []
    specs = wham_convention_profile_specs(run_config)
    for profile in wham_convention_profile_names(run_config):
        label = str((specs.get(profile) or {}).get("label") or profile)
        try:
            if profile == default_report.get("convention_profile"):
                report = default_report
                modes = default_modes
            else:
                report, series = compare_pose_to_opensim_reference(
                    pose,
                    reference,
                    run_config=run_config,
                    timeline_report=timeline,
                    convention_profile=profile,
                    evaluation_hz=evaluation_hz,
                )
                modes = {
                    "raw": _add_pelvis(np.asarray(series["prediction_root_centered_m"], dtype=float)),
                    "rigid": _add_pelvis(np.asarray(series["prediction_root_centered_rigid_m"], dtype=float)),
                    "pa": _add_pelvis(np.asarray(series["prediction_pa_root_centered_m"], dtype=float)),
                }
            profile_name = str(report.get("convention_profile") or profile)
            wham_sets[profile_name] = modes
            metrics[profile_name] = {
                "primary_root_centered_mpjpe_mm": report.get("primary_root_centered_mpjpe_mm"),
                "root_centered_rigid_mpjpe_mm": report.get("root_centered_rigid_mpjpe_mm"),
                "pa_mpjpe_mm": report.get("pa_mpjpe_mm"),
                "global_no_align_mpjpe_mm": report.get("global_no_align_mpjpe_mm"),
                "normal_minus_rigid_gap_mm": report.get("normal_minus_rigid_gap_mm"),
                "time_offset_s": report.get("time_offset_s"),
                "post_transform_determinant": report.get("post_transform_determinant"),
                "linear_transform_determinant": report.get("linear_transform_determinant"),
                "warnings": report.get("warnings"),
                "diagnostic_only": report.get("diagnostic_only"),
                "evaluation_hz": report.get("evaluation_hz"),
            }
            options.append(
                {
                    "profile": profile_name,
                    "label": str((specs.get(profile_name) or {}).get("label") or label),
                    "diagnostic_only": bool(report.get("diagnostic_only")),
                    "status": "ok",
                }
            )
        except Exception as exc:
            options.append(
                {
                    "profile": profile,
                    "label": label,
                    "diagnostic_only": profile.endswith("_lr"),
                    "status": "failed",
                    "error": str(exc),
                }
            )
    if not wham_sets:
        wham_sets[LEGACY_PROFILE] = default_modes
        metrics[LEGACY_PROFILE] = {
            "primary_root_centered_mpjpe_mm": default_report.get("primary_root_centered_mpjpe_mm"),
            "root_centered_rigid_mpjpe_mm": default_report.get("root_centered_rigid_mpjpe_mm"),
            "pa_mpjpe_mm": default_report.get("pa_mpjpe_mm"),
            "global_no_align_mpjpe_mm": default_report.get("global_no_align_mpjpe_mm"),
            "normal_minus_rigid_gap_mm": default_report.get("normal_minus_rigid_gap_mm"),
            "time_offset_s": default_report.get("time_offset_s"),
            "post_transform_determinant": default_report.get("post_transform_determinant"),
            "linear_transform_determinant": default_report.get("linear_transform_determinant"),
            "warnings": default_report.get("warnings"),
            "diagnostic_only": default_report.get("diagnostic_only"),
        }
    return wham_sets, metrics, options


def build_interactive_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html_escape(str(payload["title"]))}</title>
<style>
  :root {{
    color-scheme: light;
    --gt: #111111;
    --wham: #1f9d55;
    --muted: #5a6372;
    --line: #d8dde6;
    --panel: #ffffff;
    --bg: #f4f6f8;
  }}
  body {{
    margin: 0;
    background: var(--bg);
    color: #1d2430;
    font: 14px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  header {{
    padding: 14px 18px 8px;
    border-bottom: 1px solid var(--line);
    background: var(--panel);
  }}
  h1 {{
    margin: 0 0 8px;
    font-size: 18px;
    letter-spacing: 0;
  }}
  .metrics {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    color: var(--muted);
    font-size: 13px;
  }}
  .controls {{
    display: grid;
    grid-template-columns: auto minmax(220px, 1fr) repeat(12, auto);
    gap: 10px;
    align-items: center;
    padding: 12px 18px;
    background: #fff;
    border-bottom: 1px solid var(--line);
  }}
  button, select {{
    height: 34px;
    border: 1px solid #b9c1ce;
    background: #fff;
    color: #1d2430;
    border-radius: 6px;
    padding: 0 10px;
    font: inherit;
  }}
  button {{
    min-width: 74px;
    cursor: pointer;
  }}
  label {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--muted);
    white-space: nowrap;
  }}
  input[type="range"] {{ width: 100%; }}
  .yaw-control input[type="range"] {{ width: 150px; }}
  .offset-control input[type="range"] {{ width: 120px; }}
  main {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) 280px;
    gap: 12px;
    padding: 12px 18px 18px;
  }}
  .canvas-wrap {{
    min-height: 640px;
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 8px;
    overflow: hidden;
  }}
  canvas {{
    width: 100%;
    height: 100%;
    display: block;
    cursor: grab;
    touch-action: none;
  }}
  canvas.dragging {{ cursor: grabbing; }}
  aside {{
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 12px;
  }}
  .legend-row {{
    display: grid;
    grid-template-columns: 14px 1fr;
    gap: 8px;
    align-items: center;
    margin-bottom: 8px;
  }}
  .swatch {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
  }}
  .note {{
    color: var(--muted);
    font-size: 12px;
    margin-top: 12px;
  }}
  .warning {{
    margin-top: 12px;
    padding: 8px;
    border-radius: 6px;
    background: #fff7e6;
    border: 1px solid #f2d49b;
    color: #6a4b00;
    font-size: 12px;
  }}
  @media (max-width: 960px) {{
    .controls {{ grid-template-columns: 1fr 1fr; }}
    main {{ grid-template-columns: 1fr; }}
    .canvas-wrap {{ min-height: 520px; }}
  }}
</style>
</head>
<body>
<header>
  <h1>{_html_escape(str(payload["title"]))}</h1>
  <div class="metrics" id="metrics"></div>
</header>
<section class="controls">
  <button id="play">Play</button>
  <input id="frame" type="range" min="0" max="0" step="1" value="0">
  <label>Frame <span id="frameText">0</span></label>
  <label>Speed
    <select id="speed">
      <option value="0.1">0.10x</option>
      <option value="0.25" selected>0.25x</option>
      <option value="0.5">0.50x</option>
      <option value="1">1.00x</option>
      <option value="2">2.00x</option>
    </select>
  </label>
    <label>Alignment
    <select id="mode">
      <option value="raw">Raw root-centered</option>
      <option value="rigid">Sequence rigid</option>
      <option value="pa">PA per-frame</option>
    </select>
  </label>
  <label>Convention
    <select id="convention"></select>
  </label>
  <label>View
    <select id="view">
      <option value="orbit">Orbit</option>
      <option value="iso">Iso</option>
      <option value="front">Front X-Up</option>
      <option value="side">Side Z-Up</option>
      <option value="top">Top X-Z</option>
    </select>
  </label>
  <button id="resetView">Reset View</button>
  <button id="presetRigid">Preset Rigid XZ</button>
  <button id="presetRawYaw">Preset Raw Yaw</button>
  <button id="presetBestNormal">Best Normal</button>
  <button id="presetBestRigid">Best Rigid</button>
  <button id="presetBestPa">Best PA</button>
  <button id="presetVisualCandidate">Visual Candidate</button>
  <button id="resetFixes">Reset Fixes</button>
  <label class="yaw-control">WHAM yaw
    <input id="whamYaw" type="range" min="-180" max="180" step="1" value="0">
    <span id="whamYawText">0 deg</span>
  </label>
  <label class="offset-control">WHAM offset
    <input id="whamOffset" type="range" min="-12" max="12" step="1" value="0">
    <span id="whamOffsetText">0 fr</span>
  </label>
  <label><input id="swapWham" type="checkbox"> Swap WHAM L/R</label>
  <label><input id="swapGt" type="checkbox"> Swap GT L/R</label>
  <label><input id="mirrorX" type="checkbox"> Mirror WHAM X</label>
  <label><input id="mirrorY" type="checkbox"> Mirror WHAM Y</label>
  <label><input id="mirrorZ" type="checkbox"> Mirror WHAM Z</label>
  <label><input id="reverseWham" type="checkbox"> Reverse WHAM time</label>
  <label><input id="labels" type="checkbox" checked> Labels</label>
</section>
<main>
  <div class="canvas-wrap"><canvas id="viewer"></canvas></div>
  <aside>
    <div class="legend-row"><span class="swatch" style="background:var(--gt)"></span><strong>GT OpenSim FK</strong></div>
    <div class="legend-row"><span class="swatch" style="background:var(--wham)"></span><strong>WHAM</strong></div>
    <div id="lrDiag" class="note"></div>
    <div id="mirrorDiag" class="note"></div>
    <div id="yawDiag" class="note"></div>
    <div id="phaseDiag" class="note"></div>
    <div id="sidePhaseDiag" class="note"></div>
    <div class="note">Labels are lower-limb joint-center markers used by Level A: pelvis, hips, knees, ankles.</div>
    <div class="note">Use Raw to inspect convention/timing issues. Use Sequence rigid and PA to separate orientation from local pose shape.</div>
    <div id="evaluationNote" class="note"></div>
    <div class="note">The yaw, swap, mirror, and reverse-time controls are visual diagnostics only. They do not change saved artifacts or benchmark rankings.</div>
    <div class="note">Drag the canvas to rotate the orbit view. Use the mouse wheel or trackpad scroll to zoom.</div>
    <div id="warningBox"></div>
  </aside>
</main>
<script>
const DATA = {data_json};
const canvas = document.getElementById('viewer');
const ctx = canvas.getContext('2d');
const frame = document.getElementById('frame');
const frameText = document.getElementById('frameText');
const playBtn = document.getElementById('play');
const speedSel = document.getElementById('speed');
const modeSel = document.getElementById('mode');
const conventionSel = document.getElementById('convention');
const viewSel = document.getElementById('view');
const resetViewBtn = document.getElementById('resetView');
const presetRigidBtn = document.getElementById('presetRigid');
const presetRawYawBtn = document.getElementById('presetRawYaw');
const presetBestNormalBtn = document.getElementById('presetBestNormal');
const presetBestRigidBtn = document.getElementById('presetBestRigid');
const presetBestPaBtn = document.getElementById('presetBestPa');
const presetVisualCandidateBtn = document.getElementById('presetVisualCandidate');
const resetFixesBtn = document.getElementById('resetFixes');
const whamYaw = document.getElementById('whamYaw');
const whamYawText = document.getElementById('whamYawText');
const whamOffset = document.getElementById('whamOffset');
const whamOffsetText = document.getElementById('whamOffsetText');
const swapWhamToggle = document.getElementById('swapWham');
const swapGtToggle = document.getElementById('swapGt');
const mirrorXTog = document.getElementById('mirrorX');
const mirrorYTog = document.getElementById('mirrorY');
const mirrorZTog = document.getElementById('mirrorZ');
const reverseWhamToggle = document.getElementById('reverseWham');
const labelsToggle = document.getElementById('labels');
let idx = 0;
let playing = false;
let lastT = null;
let simTime = DATA.time_s[0] || 0;
let dragging = false;
let dragLast = null;
let camera = {{yaw: -0.75, pitch: 0.35, zoom: 1.0}};

const VIEW_PRESETS = {{
  orbit: {{yaw: -0.75, pitch: 0.35, zoom: 1.0}},
  iso: {{yaw: -0.75, pitch: 0.35, zoom: 1.0}},
  front: {{yaw: 0.0, pitch: 0.0, zoom: 1.0}},
  side: {{yaw: Math.PI / 2, pitch: 0.0, zoom: 1.0}},
  top: {{yaw: 0.0, pitch: -Math.PI / 2, zoom: 1.0}},
}};

frame.max = String(DATA.time_s.length - 1);
speedSel.value = String(DATA.default_speed || 0.25);
for (const item of DATA.convention_profiles || []) {{
  const opt = document.createElement('option');
  opt.value = item.profile;
  opt.textContent = item.label + (item.diagnostic_only ? ' (diagnostic)' : '') + (item.status && item.status !== 'ok' ? ' (unavailable)' : '');
  if (item.status && item.status !== 'ok') opt.disabled = true;
  if (item.profile === DATA.default_convention_profile) opt.selected = true;
  conventionSel.appendChild(opt);
}}
renderMetrics();
document.getElementById('evaluationNote').textContent = DATA.evaluation_hz
  ? `Prediction and reference are resampled to ${{DATA.evaluation_hz}} Hz for this viewer; one offset frame is ${{(1000 / DATA.evaluation_hz).toFixed(1)}} ms.`
  : 'This viewer uses native prediction timestamps; for WHAM walking1 that is the synced 60 Hz video timeline.';
document.getElementById('lrDiag').innerHTML = [
  `<strong>Raw L/R diagnostic</strong>`,
  `Default: ${{fmt(DATA.lr_swap_diagnostics.raw_default_mpjpe_mm)}} mm`,
  `One side swapped: ${{fmt(DATA.lr_swap_diagnostics.raw_with_one_side_lr_swapped_mpjpe_mm)}} mm`
].join('<br>');
const mirrorRows = (DATA.mirror_diagnostics || []).slice(0, 4).map((item) => `${{item.label}}: ${{fmt(item.raw_mpjpe_mm)}} mm`);
document.getElementById('mirrorDiag').innerHTML = [`<strong>Raw mirror diagnostic</strong>`, ...mirrorRows].join('<br>');
document.getElementById('yawDiag').innerHTML = [
  `<strong>Raw yaw diagnostic</strong>`,
  `Best yaw: ${{fmt(DATA.yaw_diagnostics.best_yaw_deg)}} deg`,
  `Best MPJPE: ${{fmt(DATA.yaw_diagnostics.best_yaw_raw_mpjpe_mm)}} mm`,
  `Improvement: ${{fmt(DATA.yaw_diagnostics.improvement_mm)}} mm`
].join('<br>');
document.getElementById('phaseDiag').innerHTML = [
  `<strong>Raw phase diagnostic</strong>`,
  `Best offset: ${{DATA.phase_diagnostics.best_frame_offset}} fr`,
  `Best MPJPE: ${{fmt(DATA.phase_diagnostics.best_raw_mpjpe_mm)}} mm`
].join('<br>');
document.getElementById('sidePhaseDiag').innerHTML = sidePhaseHtml(DATA.side_phase_audit);

function fmt(v) {{
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '';
  return Number(v).toFixed(1);
}}

function sidePhaseHtml(audit) {{
  if (!audit || audit.status === 'missing') {{
    return '<strong>Side Phase</strong><br>Audit not run yet.';
  }}
  const starts = audit.start_side_estimates || {{}};
  const agreement = audit.side_agreement || {{}};
  const best = agreement.best_mapping || null;
  const proj = audit.projection_check || {{}};
  const lines = [
    '<strong>Side Phase</strong>',
    `WHAM 2D starts: ${{sideLabel(starts.wham_2d)}}`,
    `WHAM 3D starts: ${{sideLabel(starts.wham_3d)}}`,
    `GT starts: ${{sideLabel(starts.gt_opensim_fk)}}`,
    `Best mapping: ${{best ? best.mapping : 'unavailable'}}`,
    `GT projection: ${{proj.preference || proj.status || 'unavailable'}}`
  ];
  return lines.join('<br>');
}}

function sideLabel(item) {{
  if (!item) return 'unavailable';
  const side = item.side || 'uncertain';
  const dt = item.left_minus_right_peak_time_s;
  return Number.isFinite(Number(dt)) ? `${{side}} (${{Number(dt).toFixed(3)}}s L-R)` : side;
}}

function resize() {{
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(400, Math.floor(rect.width * scale));
  canvas.height = Math.max(360, Math.floor(rect.height * scale));
  draw();
}}

function project(p, view) {{
  const q = rotatePoint(p);
  return [q[0], q[1]];
}}

function rotatePoint(p) {{
  const x = p[0], y = p[1], z = p[2];
  const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
  const cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
  const x1 = cy * x + sy * z;
  const z1 = -sy * x + cy * z;
  const y1 = y;
  const y2 = cp * y1 - sp * z1;
  const z2 = sp * y1 + cp * z1;
  return [x1, y2, z2];
}}

function applyViewPreset(view) {{
  const preset = VIEW_PRESETS[view] || VIEW_PRESETS.orbit;
  camera = {{yaw: preset.yaw, pitch: preset.pitch, zoom: preset.zoom}};
}}

function bounds2d(view) {{
  const pts = [];
  for (const arr of [DATA.gt, currentWhamSet().raw, currentWhamSet().rigid, currentWhamSet().pa]) {{
    for (const f of arr) for (const p of f) pts.push(project(p, view));
  }}
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) {{
    if (!Number.isFinite(p[0]) || !Number.isFinite(p[1])) continue;
    minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
    minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
  }}
  if (!Number.isFinite(minX)) return {{minX:-1,maxX:1,minY:-1,maxY:1}};
  const pad = Math.max(maxX - minX, maxY - minY, 0.5) * 0.18;
  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const halfX = Math.max((maxX - minX) * 0.5 + pad, 0.25) / camera.zoom;
  const halfY = Math.max((maxY - minY) * 0.5 + pad, 0.25) / camera.zoom;
  return {{minX:cx-halfX,maxX:cx+halfX,minY:cy-halfY,maxY:cy+halfY}};
}}

function toScreen(p, b, view) {{
  const q = project(p, view);
  const w = canvas.width, h = canvas.height;
  const plotW = w - 90, plotH = h - 90;
  const sx = 45 + (q[0] - b.minX) / (b.maxX - b.minX) * plotW;
  const sy = 35 + (b.maxY - q[1]) / (b.maxY - b.minY) * plotH;
  return [sx, sy];
}}

function drawSkeleton(values, color, labelPrefix, labelDx, labelDy) {{
  const view = viewSel.value;
  const b = bounds2d(view);
  ctx.lineWidth = 3;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  for (const e of DATA.edges) {{
    const a = values[e[0]], c = values[e[1]];
    if (!finite3(a) || !finite3(c)) continue;
    const pa = toScreen(a, b, view), pc = toScreen(c, b, view);
    ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pc[0], pc[1]); ctx.stroke();
  }}
  ctx.font = `${{Math.max(22, canvas.width * 0.015)}}px system-ui, sans-serif`;
  for (let i = 0; i < values.length; i++) {{
    const p = values[i];
    if (!finite3(p)) continue;
    const s = toScreen(p, b, view);
    ctx.beginPath();
    ctx.arc(s[0], s[1], 6 * (window.devicePixelRatio || 1), 0, Math.PI * 2);
    ctx.fill();
    if (labelsToggle.checked) {{
      const text = `${{labelPrefix}} ${{DATA.short_labels[i]}}`;
      ctx.save();
      ctx.lineWidth = 4;
      ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.strokeText(text, s[0] + labelDx, s[1] + labelDy);
      ctx.fillStyle = color;
      ctx.fillText(text, s[0] + labelDx, s[1] + labelDy);
      ctx.restore();
    }}
  }}
}}

function maybeSwap(values, enabled) {{
  if (!enabled) return values;
  const out = values.map((p) => p);
  for (const pair of DATA.lr_swap_pairs || []) {{
    const a = pair[0], b = pair[1];
    const tmp = out[a];
    out[a] = out[b];
    out[b] = tmp;
  }}
  return out;
}}

function maybeMirror(values) {{
  const sx = mirrorXTog.checked ? -1 : 1;
  const sy = mirrorYTog.checked ? -1 : 1;
  const sz = mirrorZTog.checked ? -1 : 1;
  if (sx === 1 && sy === 1 && sz === 1) return values;
  return values.map((p) => finite3(p) ? [sx * p[0], sy * p[1], sz * p[2]] : p);
}}

function maybeYaw(values) {{
  const degrees = Number(whamYaw.value || 0);
  whamYawText.textContent = `${{degrees}} deg`;
  if (degrees === 0) return values;
  const rad = degrees * Math.PI / 180;
  const c = Math.cos(rad), s = Math.sin(rad);
  return values.map((p) => finite3(p) ? [c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2]] : p);
}}

function whamValuesForFrame() {{
  const offset = Number(whamOffset.value || 0);
  whamOffsetText.textContent = `${{offset}} fr`;
  let whamIdx = idx + offset;
  whamIdx = Math.max(0, Math.min(DATA.time_s.length - 1, whamIdx));
  if (reverseWhamToggle.checked) whamIdx = DATA.time_s.length - 1 - whamIdx;
  return maybeYaw(maybeMirror(maybeSwap(currentWhamSet()[modeSel.value][whamIdx], swapWhamToggle.checked)));
}}

function currentConvention() {{
  return conventionSel.value || DATA.default_convention_profile;
}}

function currentWhamSet() {{
  return (DATA.wham_conventions && DATA.wham_conventions[currentConvention()]) || DATA.wham;
}}

function renderMetrics() {{
  const profile = currentConvention();
  const metrics = (DATA.metrics_by_convention && DATA.metrics_by_convention[profile]) || DATA.metrics;
  const evalHz = metrics.evaluation_hz || DATA.evaluation_hz;
  const postDet = metrics.post_transform_determinant;
  document.getElementById('metrics').innerHTML = [
    ['Convention', profile],
    ['Eval Hz', evalHz ? `${{evalHz}} Hz` : 'native'],
    ['Raw MPJPE', metrics.primary_root_centered_mpjpe_mm],
    ['Rigid', metrics.root_centered_rigid_mpjpe_mm],
    ['PA', metrics.pa_mpjpe_mm],
    ['Global Raw', metrics.global_no_align_mpjpe_mm],
    ['Offset', metrics.time_offset_s ? `${{Number(metrics.time_offset_s).toFixed(3)}} s` : '0 s'],
    ['Post det', postDet === null || postDet === undefined ? '' : Number(postDet).toFixed(3)],
    ['Gap', metrics.normal_minus_rigid_gap_mm]
  ].map(([k,v]) => `<span><strong>${{k}}:</strong> ${{typeof v === 'number' ? fmt(v) + ' mm' : v || ''}}</span>`).join('');
  if (metrics.warnings && metrics.warnings.length) {{
    document.getElementById('warningBox').innerHTML = `<div class="warning">${{metrics.warnings.join('<br>')}}</div>`;
  }} else {{
    document.getElementById('warningBox').innerHTML = '';
  }}
}}

function resetManualDiagnostics() {{
  whamYaw.value = '0';
  whamOffset.value = '0';
  swapWhamToggle.checked = false;
  swapGtToggle.checked = false;
  mirrorXTog.checked = false;
  mirrorYTog.checked = false;
  mirrorZTog.checked = false;
  reverseWhamToggle.checked = false;
}}

function applyMatrixPreset(presetName, modeName) {{
  const profile = DATA.matrix_presets && DATA.matrix_presets[presetName];
  if (profile && DATA.wham_conventions && DATA.wham_conventions[profile]) {{
    conventionSel.value = profile;
  }}
  modeSel.value = modeName;
  resetManualDiagnostics();
  draw();
}}

function finite3(p) {{
  return p && Number.isFinite(p[0]) && Number.isFinite(p[1]) && Number.isFinite(p[2]);
}}

function draw() {{
  renderMetrics();
  const dpr = window.devicePixelRatio || 1;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.fillStyle = '#1d2430';
  ctx.font = '15px system-ui, sans-serif';
  const modeName = modeSel.options[modeSel.selectedIndex].text;
  ctx.fillText(`${{modeName}} | ${{viewSel.options[viewSel.selectedIndex].text}} | zoom=${{camera.zoom.toFixed(2)}}x | t=${{DATA.time_s[idx].toFixed(3)}}s`, 18, 24);
  ctx.restore();
  drawSkeleton(maybeSwap(DATA.gt[idx], swapGtToggle.checked), '#111111', 'GT', 12 * dpr, -10 * dpr);
  drawSkeleton(whamValuesForFrame(), '#1f9d55', 'WHAM', 12 * dpr, 18 * dpr);
  frame.value = String(idx);
  frameText.textContent = `${{idx}} / ${{DATA.time_s.length - 1}}`;
}}

function nearestIndex(t) {{
  let best = 0, dist = Infinity;
  for (let i = 0; i < DATA.time_s.length; i++) {{
    const d = Math.abs(DATA.time_s[i] - t);
    if (d < dist) {{ best = i; dist = d; }}
  }}
  return best;
}}

function tick(ts) {{
  if (!playing) return;
  if (lastT === null) lastT = ts;
  const dt = (ts - lastT) / 1000;
  lastT = ts;
  simTime += dt * Number(speedSel.value || 0.25);
  const start = DATA.time_s[0], end = DATA.time_s[DATA.time_s.length - 1];
  if (simTime > end) simTime = start;
  idx = nearestIndex(simTime);
  draw();
  requestAnimationFrame(tick);
}}

playBtn.addEventListener('click', () => {{
  playing = !playing;
  playBtn.textContent = playing ? 'Pause' : 'Play';
  lastT = null;
  simTime = DATA.time_s[idx];
  if (playing) requestAnimationFrame(tick);
}});
frame.addEventListener('input', () => {{
  idx = Number(frame.value);
  simTime = DATA.time_s[idx];
  draw();
}});
viewSel.addEventListener('change', () => {{
  applyViewPreset(viewSel.value);
  draw();
}});
resetViewBtn.addEventListener('click', () => {{
  applyViewPreset(viewSel.value);
  draw();
}});
presetRigidBtn.addEventListener('click', () => {{
  modeSel.value = 'rigid';
  whamYaw.value = '0';
  whamOffset.value = '6';
  swapWhamToggle.checked = true;
  swapGtToggle.checked = false;
  mirrorXTog.checked = true;
  mirrorYTog.checked = false;
  mirrorZTog.checked = true;
  reverseWhamToggle.checked = false;
  draw();
}});
presetRawYawBtn.addEventListener('click', () => {{
  modeSel.value = 'raw';
  whamYaw.value = '-32';
  whamOffset.value = '12';
  swapWhamToggle.checked = true;
  swapGtToggle.checked = false;
  mirrorXTog.checked = false;
  mirrorYTog.checked = false;
  mirrorZTog.checked = false;
  reverseWhamToggle.checked = false;
  draw();
}});
presetVisualCandidateBtn.addEventListener('click', () => {{
  const profile = (DATA.matrix_presets && DATA.matrix_presets.visual_candidate) || 'opencap_full_yaw180_lr_offset120ms';
  if (DATA.wham_conventions && DATA.wham_conventions[profile]) {{
    conventionSel.value = profile;
  }}
  modeSel.value = 'rigid';
  resetManualDiagnostics();
  draw();
}});
presetBestNormalBtn.addEventListener('click', () => applyMatrixPreset('best_normal', 'raw'));
presetBestRigidBtn.addEventListener('click', () => applyMatrixPreset('best_rigid', 'rigid'));
presetBestPaBtn.addEventListener('click', () => applyMatrixPreset('best_pa', 'pa'));
resetFixesBtn.addEventListener('click', () => {{
  resetManualDiagnostics();
  draw();
}});
canvas.addEventListener('pointerdown', (ev) => {{
  dragging = true;
  dragLast = [ev.clientX, ev.clientY];
  canvas.classList.add('dragging');
  canvas.setPointerCapture(ev.pointerId);
}});
canvas.addEventListener('pointermove', (ev) => {{
  if (!dragging || !dragLast) return;
  const dx = ev.clientX - dragLast[0];
  const dy = ev.clientY - dragLast[1];
  dragLast = [ev.clientX, ev.clientY];
  camera.yaw += dx * 0.008;
  camera.pitch = clamp(camera.pitch + dy * 0.008, -1.48, 1.48);
  viewSel.value = 'orbit';
  draw();
}});
canvas.addEventListener('pointerup', (ev) => {{
  dragging = false;
  dragLast = null;
  canvas.classList.remove('dragging');
  try {{ canvas.releasePointerCapture(ev.pointerId); }} catch (err) {{}}
}});
canvas.addEventListener('pointercancel', () => {{
  dragging = false;
  dragLast = null;
  canvas.classList.remove('dragging');
}});
canvas.addEventListener('wheel', (ev) => {{
  ev.preventDefault();
  camera.zoom = clamp(camera.zoom * Math.exp(-ev.deltaY * 0.001), 0.35, 6.0);
  draw();
}}, {{passive: false}});
canvas.addEventListener('dblclick', () => {{
  applyViewPreset(viewSel.value);
  draw();
}});
function clamp(value, lo, hi) {{
  return Math.max(lo, Math.min(hi, value));
}}
whamYaw.addEventListener('input', draw);
whamOffset.addEventListener('input', draw);
for (const el of [speedSel, modeSel, conventionSel, swapWhamToggle, swapGtToggle, mirrorXTog, mirrorYTog, mirrorZTog, reverseWhamToggle, labelsToggle]) el.addEventListener('change', draw);
window.addEventListener('resize', resize);
applyViewPreset(viewSel.value);
resize();
</script>
</body>
</html>
"""


def _cached_wham_timeline(run_dir: Path) -> dict[str, Any] | None:
    for candidate in _run_dir_candidates(run_dir):
        for qc_path in [candidate / "reports" / "wham_timeline_qc.json", candidate / "pose3d_initial" / "pose3d_initial_qc.json"]:
            if not qc_path.exists():
                continue
            qc = read_json(qc_path)
            timeline = qc if "raw_sync_alignment" in qc else qc.get("wham_timeline") or qc.get("wham_timeline_report")
            if (
                isinstance(timeline, dict)
                and isinstance(timeline.get("raw_sync_alignment"), dict)
                and isinstance(timeline.get("overlap"), dict)
            ):
                return timeline
    return None


def _load_matrix_profiles(matrix_audit_path: Path | None, evaluation_hz: float | None) -> tuple[dict[str, dict[str, Any]], dict[str, str | None]]:
    if not matrix_audit_path:
        return {}, {}
    return matrix_profiles_for_viewer(Path(matrix_audit_path), evaluation_hz=evaluation_hz)


def _run_config_with_matrix_profiles(run_config: dict[str, Any], matrix_profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not matrix_profiles:
        return run_config
    out = json.loads(json.dumps(run_config or {}))
    cfg = out.setdefault("config", {})
    level_a = cfg.setdefault("level_a", {})
    profiles = level_a.setdefault("wham_convention_profiles", {})
    for name, spec in matrix_profiles.items():
        profiles[str(name)] = dict(spec)
    return out


def _run_dir_candidates(run_dir: Path) -> list[Path]:
    candidates = [run_dir]
    name = run_dir.name
    if "__" in name:
        candidates.append(run_dir.with_name(name.split("__", 1)[0]))
    return candidates


def _add_pelvis(values: np.ndarray) -> np.ndarray:
    pelvis = np.zeros((values.shape[0], 1, 3), dtype=float)
    return np.concatenate([pelvis, np.asarray(values, dtype=float)], axis=1)


def _round_array(values: np.ndarray, decimals: int) -> np.ndarray:
    return np.round(np.asarray(values, dtype=float), decimals=decimals)


def _bounds(arrays: list[np.ndarray]) -> dict[str, list[float]]:
    data = np.concatenate([arr.reshape(-1, 3) for arr in arrays], axis=0)
    finite = data[np.isfinite(data).all(axis=1)]
    if finite.size == 0:
        return {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}
    return {"min": np.nanmin(finite, axis=0).tolist(), "max": np.nanmax(finite, axis=0).tolist()}


def _edge_indices(names: list[str]) -> list[list[int]]:
    lookup = {_canon(name): idx for idx, name in enumerate(names)}
    out = []
    for a, b in EDGES:
        ai = lookup.get(_canon(a))
        bi = lookup.get(_canon(b))
        if ai is not None and bi is not None:
            out.append([ai, bi])
    return out


def _lr_swap_pairs(names: list[str]) -> list[list[int]]:
    lookup = {_canon(name): idx for idx, name in enumerate(names)}
    pairs = [
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
        ("left_mtp", "right_mtp"),
        ("left_toe", "right_toe"),
    ]
    out = []
    for left, right in pairs:
        li = lookup.get(_canon(left))
        ri = lookup.get(_canon(right))
        if li is not None and ri is not None:
            out.append([li, ri])
    return out


def _swap_lr_values(values: np.ndarray, pairs: list[list[int]]) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    for left, right in pairs:
        out[:, [left, right], :] = out[:, [right, left], :]
    return out


def _raw_mpjpe_mm(pred: np.ndarray, ref: np.ndarray) -> float | None:
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if pred.shape != ref.shape or pred.ndim != 3:
        return None
    # Exclude the synthetic pelvis root inserted at index 0; it is always zero.
    err = np.linalg.norm(pred[:, 1:, :] - ref[:, 1:, :], axis=-1)
    finite = err[np.isfinite(err)]
    if finite.size == 0:
        return None
    return float(np.mean(finite) * 1000.0)


def _mirror_diagnostics(pred: np.ndarray, ref: np.ndarray) -> list[dict[str, Any]]:
    candidates = [
        ("none", (1.0, 1.0, 1.0)),
        ("mirror X", (-1.0, 1.0, 1.0)),
        ("mirror Y", (1.0, -1.0, 1.0)),
        ("mirror Z", (1.0, 1.0, -1.0)),
        ("mirror X+Y", (-1.0, -1.0, 1.0)),
        ("mirror X+Z", (-1.0, 1.0, -1.0)),
        ("mirror Y+Z", (1.0, -1.0, -1.0)),
        ("mirror X+Y+Z", (-1.0, -1.0, -1.0)),
    ]
    rows = []
    pred_arr = np.asarray(pred, dtype=float)
    for label, signs in candidates:
        mirrored = pred_arr * np.asarray(signs, dtype=float)[None, None, :]
        rows.append({"label": label, "signs": list(signs), "raw_mpjpe_mm": _raw_mpjpe_mm(mirrored, ref)})
    return sorted(rows, key=lambda item: float("inf") if item["raw_mpjpe_mm"] is None else float(item["raw_mpjpe_mm"]))


def _yaw_diagnostics(pred: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    candidates = []
    pred_arr = np.asarray(pred, dtype=float)
    for degrees in range(-180, 181):
        rotated = _rotate_yaw_y(pred_arr, float(degrees))
        candidates.append({"yaw_deg": float(degrees), "raw_mpjpe_mm": _raw_mpjpe_mm(rotated, ref)})
    valid = [item for item in candidates if item["raw_mpjpe_mm"] is not None]
    valid.sort(key=lambda item: float(item["raw_mpjpe_mm"]))
    best = valid[0] if valid else {"yaw_deg": None, "raw_mpjpe_mm": None}
    default = _raw_mpjpe_mm(pred_arr, ref)
    return {
        "axis": "OpenSim_Y_up",
        "default_raw_mpjpe_mm": default,
        "best_yaw_deg": best.get("yaw_deg"),
        "best_yaw_raw_mpjpe_mm": best.get("raw_mpjpe_mm"),
        "improvement_mm": None
        if default is None or best.get("raw_mpjpe_mm") is None
        else float(default) - float(best["raw_mpjpe_mm"]),
        "top": valid[:8],
        "note": "Diagnostic-only search over fixed proper rotations around the OpenSim vertical Y axis.",
    }


def _phase_diagnostics(pred: np.ndarray, ref: np.ndarray, max_offset_frames: int = 12) -> dict[str, Any]:
    rows = []
    for offset in range(-int(max_offset_frames), int(max_offset_frames) + 1):
        rows.append({"frame_offset": int(offset), "raw_mpjpe_mm": _offset_mpjpe_mm(pred, ref, offset)})
    valid = [row for row in rows if row["raw_mpjpe_mm"] is not None]
    valid.sort(key=lambda item: float(item["raw_mpjpe_mm"]))
    best = valid[0] if valid else {"frame_offset": None, "raw_mpjpe_mm": None}
    return {
        "definition": "WHAM frame index = GT frame index + frame_offset; diagnostic only.",
        "max_offset_frames": int(max_offset_frames),
        "best_frame_offset": best.get("frame_offset"),
        "best_raw_mpjpe_mm": best.get("raw_mpjpe_mm"),
        "top": valid[:8],
    }


def _offset_mpjpe_mm(pred: np.ndarray, ref: np.ndarray, frame_offset: int) -> float | None:
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if pred.shape != ref.shape or pred.ndim != 3:
        return None
    offset = int(frame_offset)
    if offset > 0:
        pred_view = pred[offset:]
        ref_view = ref[:-offset]
    elif offset < 0:
        pred_view = pred[:offset]
        ref_view = ref[-offset:]
    else:
        pred_view = pred
        ref_view = ref
    if pred_view.shape[0] < 2:
        return None
    return _raw_mpjpe_mm(pred_view, ref_view)


def _load_side_phase_audit(benchmark_dir: Path, trial: str) -> dict[str, Any]:
    path = Path(benchmark_dir) / "side_phase" / f"wham_side_phase_{trial}.json"
    if not path.exists():
        return {"status": "missing", "expected_path": str(path)}
    try:
        return read_json(path)
    except Exception as exc:
        return {"status": "failed", "source": str(path), "error": str(exc)}


def _rotate_yaw_y(values: np.ndarray, degrees: float) -> np.ndarray:
    radians = np.deg2rad(float(degrees))
    c = float(np.cos(radians))
    s = float(np.sin(radians))
    rot = np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)
    return np.asarray(values, dtype=float) @ rot.T


def _short_label(name: str) -> str:
    mapping = {
        "pelvis": "PELV",
        "hip_midpoint": "HIPMID",
        "left_hip": "LHIP",
        "right_hip": "RHIP",
        "left_knee": "LKNE",
        "right_knee": "RKNE",
        "left_ankle": "LANK",
        "right_ankle": "RANK",
    }
    return mapping.get(name, name.upper())


def _canon(name: str) -> str:
    return name.strip().lower().replace("_", "").replace("-", "").replace(".", "")


def _html_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
