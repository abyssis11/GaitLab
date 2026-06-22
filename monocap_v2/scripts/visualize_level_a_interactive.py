#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monocap_v2.core.level_a_benchmark import (
    compare_pose_to_opensim_reference,
    load_cached_wham_timeline,
    load_opensim_reference,
    load_pose_artifact,
    load_run_config,
)
from monocap_v2.core.level_a_visualization import _add_pelvis
from monocap_v2.core.logging_utils import read_json, write_json


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
BACKEND_COLORS = {
    "metrabs": "#1f77b4",
    "rtmw3d": "#ff7f0e",
    "wham": "#2ca02c",
    "sam3d_body": "#9467bd",
}


def main() -> int:
    args = parse_args()
    out_dir = args.benchmark_dir / "visualizations"
    out_html = args.out_html or out_dir / f"level_a_{args.trial}_{args.backend}_interactive.html"
    out_json = args.out_json or out_dir / f"level_a_{args.trial}_{args.backend}_interactive_qc.json"
    payload, report = build_payload(args)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(build_html(payload), encoding="utf-8")
    report["outputs"] = {"html": str(out_html), "qc": str(out_json)}
    write_json(out_json, report)
    print(f"[level-a-interactive] ok -> {out_html}", flush=True)
    print(f"[level-a-interactive] qc -> {out_json}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create an interactive Level A GT-vs-backend 3D viewer.")
    ap.add_argument("--benchmark-dir", required=True, type=Path)
    ap.add_argument("--trial", default="walking1")
    ap.add_argument("--backend", default="sam3d_body")
    ap.add_argument("--evaluation-hz", type=float, default=None)
    ap.add_argument("--axis", default=None, help="Optional backend axis override, e.g. x,-y,z.")
    ap.add_argument("--swap-reference-lr", action="store_true")
    ap.add_argument("--out-html", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    return ap.parse_args()


def build_payload(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = read_json(args.benchmark_dir / "level_a_summary.json")
    row = _find_row(summary, args.backend, args.trial)
    run_dir = Path(str(row["run_dir"]))
    reference = load_opensim_reference(
        args.benchmark_dir / "reference" / f"opensim_fk_{args.trial}.npz",
        args.benchmark_dir / "reference" / f"opensim_fk_{args.trial}.json",
    )
    pose = load_pose_artifact(run_dir)
    axis_map = {args.backend: args.axis} if args.axis else None
    report, series = compare_pose_to_opensim_reference(
        pose,
        reference,
        run_config=load_run_config(run_dir),
        axis_map=axis_map,
        evaluation_hz=args.evaluation_hz,
        swap_reference_lr=bool(args.swap_reference_lr),
        timeline_report=load_cached_wham_timeline(run_dir) if args.backend == "wham" else None,
    )
    root_name = str(series.get("root_name") or "pelvis")
    joint_names = [root_name] + [str(name) for name in series["joint_names"].tolist()]
    gt = _add_pelvis(np.asarray(series["reference_root_centered_m"], dtype=float))
    raw = _add_pelvis(np.asarray(series["prediction_root_centered_m"], dtype=float))
    rigid = _add_pelvis(np.asarray(series["prediction_root_centered_rigid_m"], dtype=float))
    pa = _add_pelvis(np.asarray(series["prediction_pa_root_centered_m"], dtype=float))
    payload = {
        "title": f"Level A interactive | {args.trial} | GT vs {args.backend}",
        "trial": args.trial,
        "backend": args.backend,
        "evaluation_hz": report.get("evaluation_hz"),
        "time_s": _round(np.asarray(series["time_s"], dtype=float), 5).tolist(),
        "joint_names": joint_names,
        "short_labels": [_short_label(name) for name in joint_names],
        "edges": _edge_indices(joint_names),
        "gt": _round(gt, 5).tolist(),
        "prediction": {
            "raw": _round(raw, 5).tolist(),
            "rigid": _round(rigid, 5).tolist(),
            "pa": _round(pa, 5).tolist(),
        },
        "bounds": _bounds([gt, raw, rigid, pa]),
        "color": BACKEND_COLORS.get(args.backend, "#9467bd"),
        "metrics": {
            "raw_root_mpjpe_mm": report.get("primary_root_centered_mpjpe_mm"),
            "rigid_mpjpe_mm": report.get("root_centered_rigid_mpjpe_mm"),
            "pa_mpjpe_mm": report.get("pa_mpjpe_mm"),
            "axis_mode": report.get("axis_mode"),
            "resampling": report.get("resampling"),
            "time_offset_s": report.get("time_offset_s"),
            "reference_left_right_swap": report.get("reference_left_right_swap"),
            "warnings": report.get("warnings") or [],
        },
    }
    qc = {
        "status": "ok",
        "benchmark_dir": str(args.benchmark_dir),
        "trial": args.trial,
        "backend": args.backend,
        "run_dir": str(run_dir),
        "frames": len(payload["time_s"]),
        "joint_names": joint_names,
        "level_a_report": report,
        "notes": [
            "Interactive HTML is self-contained and can be opened directly in a browser.",
            "Raw shows hip/root-centered Level A coordinates with the configured backend axis map.",
            "Rigid and PA are diagnostic fitted alignments; they do not change saved artifacts.",
        ],
    }
    return payload, qc


def build_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    title = _escape(str(payload["title"]))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{
    margin: 0;
    background: #f5f6f8;
    color: #1f2530;
    font: 14px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  header {{ background: #fff; border-bottom: 1px solid #d9dee7; padding: 14px 18px 10px; }}
  h1 {{ margin: 0 0 8px; font-size: 18px; letter-spacing: 0; }}
  .metrics {{ display: flex; flex-wrap: wrap; gap: 12px; color: #5e6675; font-size: 13px; }}
  .controls {{
    display: grid;
    grid-template-columns: auto minmax(240px, 1fr) repeat(7, auto);
    gap: 10px;
    align-items: center;
    padding: 12px 18px;
    background: #fff;
    border-bottom: 1px solid #d9dee7;
  }}
  button, select {{
    height: 34px;
    border: 1px solid #b8c0cd;
    border-radius: 6px;
    background: #fff;
    padding: 0 10px;
    font: inherit;
  }}
  button {{ min-width: 72px; cursor: pointer; }}
  input[type="range"] {{ width: 100%; }}
  label {{ display: inline-flex; align-items: center; gap: 6px; color: #5e6675; white-space: nowrap; }}
  main {{ display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 12px; padding: 12px 18px 18px; }}
  .canvas-wrap {{ min-height: 720px; background: #fff; border: 1px solid #d9dee7; border-radius: 8px; overflow: hidden; }}
  canvas {{ width: 100%; height: 100%; display: block; cursor: grab; touch-action: none; }}
  canvas.dragging {{ cursor: grabbing; }}
  aside {{ background: #fff; border: 1px solid #d9dee7; border-radius: 8px; padding: 12px; }}
  .legend-row {{ display: grid; grid-template-columns: 14px 1fr; gap: 8px; align-items: center; margin-bottom: 8px; }}
  .swatch {{ width: 12px; height: 12px; border-radius: 50%; }}
  .note {{ color: #5e6675; font-size: 12px; margin-top: 12px; }}
  .warn {{ padding: 8px; border: 1px solid #f0cf8a; border-radius: 6px; background: #fff8e8; color: #684a00; }}
  @media (max-width: 900px) {{
    .controls {{ grid-template-columns: 1fr 1fr; }}
    main {{ grid-template-columns: 1fr; }}
    .canvas-wrap {{ min-height: 560px; }}
  }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="metrics" id="metrics"></div>
</header>
<section class="controls">
  <button id="play">Play</button>
  <input id="frame" type="range" min="0" max="0" step="1" value="0">
  <label>Frame <span id="frameText">0</span></label>
  <label>Mode
    <select id="mode">
      <option value="raw">Raw root-centered</option>
      <option value="rigid">Sequence rigid</option>
      <option value="pa" selected>PA per-frame</option>
    </select>
  </label>
  <label>Speed
    <select id="speed">
      <option value="0.1">0.10x</option>
      <option value="0.25" selected>0.25x</option>
      <option value="0.5">0.50x</option>
      <option value="1">1.00x</option>
      <option value="2">2.00x</option>
    </select>
  </label>
  <label>View
    <select id="view">
      <option value="orbit">Orbit</option>
      <option value="front">Front</option>
      <option value="side">Side</option>
      <option value="top">Top</option>
    </select>
  </label>
  <button id="reset">Reset View</button>
  <label><input id="labels" type="checkbox" checked> Labels</label>
</section>
<main>
  <div class="canvas-wrap"><canvas id="viewer"></canvas></div>
  <aside>
    <div class="legend-row"><span class="swatch" style="background:#111"></span><strong>OpenSim FK GT</strong></div>
    <div class="legend-row"><span class="swatch" id="predSwatch"></span><strong id="predName"></strong></div>
    <div class="note warn">Rigid and PA modes are diagnostic fitted alignments. They show pose-shape agreement, not a production coordinate convention.</div>
    <div class="note">Drag to rotate. Scroll to zoom. Double-click to reset.</div>
    <div class="note">Skeletons are hip/root-centered lower-limb joint centers used by Level A.</div>
    <div class="note" id="settings"></div>
    <div class="note" id="warnings"></div>
  </aside>
</main>
<script>
const DATA = {data};
const canvas = document.getElementById('viewer');
const ctx = canvas.getContext('2d');
const frame = document.getElementById('frame');
const frameText = document.getElementById('frameText');
const playBtn = document.getElementById('play');
const speedSel = document.getElementById('speed');
const viewSel = document.getElementById('view');
const modeSel = document.getElementById('mode');
const resetBtn = document.getElementById('reset');
const labelsToggle = document.getElementById('labels');
let idx = 0, playing = false, lastT = null;
let camera = {{yaw: -0.75, pitch: 0.32, zoom: 1.0}};
let dragging = false, lastDrag = null;
const views = {{
  orbit: {{yaw: -0.75, pitch: 0.32, zoom: 1.0}},
  front: {{yaw: 0.0, pitch: 0.0, zoom: 1.0}},
  side: {{yaw: Math.PI / 2, pitch: 0.0, zoom: 1.0}},
  top: {{yaw: 0.0, pitch: -Math.PI / 2, zoom: 1.0}},
}};

frame.max = String(DATA.time_s.length - 1);
document.getElementById('predSwatch').style.background = DATA.color;
document.getElementById('predName').textContent = DATA.backend;
document.getElementById('metrics').innerHTML = [
  `Raw ${{fmt(DATA.metrics.raw_root_mpjpe_mm)}} mm`,
  `Rigid ${{fmt(DATA.metrics.rigid_mpjpe_mm)}} mm`,
  `PA ${{fmt(DATA.metrics.pa_mpjpe_mm)}} mm`,
  `Axis ${{DATA.metrics.axis_mode || ''}}`,
  `Eval Hz ${{DATA.evaluation_hz || 'native'}}`,
].map(v => `<span>${{v}}</span>`).join('');
document.getElementById('settings').innerHTML = [
  `<strong>Level A settings</strong>`,
  `Resampling: ${{DATA.metrics.resampling || ''}}`,
  `Time offset: ${{Number(DATA.metrics.time_offset_s || 0).toFixed(3)}} s`,
  `Reference L/R swap: ${{DATA.metrics.reference_left_right_swap}}`,
].join('<br>');
document.getElementById('warnings').innerHTML = (DATA.metrics.warnings || []).length
  ? `<strong>Warnings</strong><br>${{DATA.metrics.warnings.map(escapeHtml).join('<br>')}}`
  : '';

function fmt(v) {{
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '';
  return Number(v).toFixed(2);
}}
function escapeHtml(v) {{
  return String(v).replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
}}
function resize() {{
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(500, Math.floor(rect.width * scale));
  canvas.height = Math.max(420, Math.floor(rect.height * scale));
  draw();
}}
function setView(name) {{ camera = {{...views[name]}}; draw(); }}
function rotatePoint(p) {{
  const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
  const cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
  let x = p[0], y = p[1], z = p[2];
  const x1 = cy * x + sy * z;
  const z1 = -sy * x + cy * z;
  const y2 = cp * y - sp * z1;
  const z2 = sp * y + cp * z1;
  return [x1, y2, z2];
}}
function project(p) {{
  const r = rotatePoint(p);
  const w = canvas.width, h = canvas.height;
  const span = Math.max(DATA.bounds.radius || 0.8, 0.4);
  const s = Math.min(w, h) * 0.42 * camera.zoom / span;
  return [w * 0.5 + r[0] * s, h * 0.54 - r[1] * s, r[2]];
}}
function drawSkeleton(values, color, label, lineWidth) {{
  const pts = values.map(project);
  ctx.lineCap = 'round';
  for (const [a, b] of DATA.edges) {{
    const pa = pts[a], pb = pts[b];
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.9;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(pa[0], pa[1]);
    ctx.lineTo(pb[0], pb[1]);
    ctx.stroke();
  }}
  for (let i = 0; i < pts.length; i++) {{
    const p = pts[i];
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.95;
    ctx.beginPath();
    ctx.arc(p[0], p[1], label === 'GT' ? 4.2 : 5.2, 0, Math.PI * 2);
    ctx.fill();
    if (labelsToggle.checked) {{
      ctx.font = `${{Math.max(11, canvas.width * 0.008)}}px system-ui, sans-serif`;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.9;
      ctx.fillText(DATA.short_labels[i], p[0] + 6, p[1] - 5);
    }}
  }}
  ctx.globalAlpha = 1;
}}
function drawGrid() {{
  const w = canvas.width, h = canvas.height;
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = '#e6e9ef';
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 50) {{ ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }}
  for (let y = 0; y < h; y += 50) {{ ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }}
}}
function modeLabel() {{
  return modeSel.options[modeSel.selectedIndex].textContent;
}}
function draw() {{
  drawGrid();
  const gt = DATA.gt[idx];
  const pred = DATA.prediction[modeSel.value][idx];
  drawSkeleton(gt, '#111111', 'GT', 3.0);
  drawSkeleton(pred, DATA.color, DATA.backend, 3.4);
  ctx.fillStyle = '#1f2530';
  ctx.font = `${{Math.max(14, canvas.width * 0.011)}}px system-ui, sans-serif`;
  ctx.fillText(`Frame ${{idx + 1}}/${{DATA.time_s.length}} | t=${{DATA.time_s[idx].toFixed(3)}}s | ${{modeLabel()}}`, 18, 28);
  frame.value = String(idx);
  frameText.textContent = `${{idx}} / ${{DATA.time_s.length - 1}}`;
}}
function animate(t) {{
  if (!playing) return;
  if (lastT === null) lastT = t;
  const dt = (t - lastT) / 1000;
  lastT = t;
  const step = Math.max(1, Math.floor(dt * 30 * Number(speedSel.value)));
  if (step > 0) idx = (idx + step) % DATA.time_s.length;
  draw();
  requestAnimationFrame(animate);
}}
playBtn.addEventListener('click', () => {{
  playing = !playing;
  playBtn.textContent = playing ? 'Pause' : 'Play';
  lastT = null;
  if (playing) requestAnimationFrame(animate);
}});
frame.addEventListener('input', () => {{ idx = Number(frame.value); draw(); }});
viewSel.addEventListener('change', () => setView(viewSel.value));
modeSel.addEventListener('change', draw);
resetBtn.addEventListener('click', () => setView('orbit'));
labelsToggle.addEventListener('change', draw);
canvas.addEventListener('pointerdown', (ev) => {{
  dragging = true; lastDrag = [ev.clientX, ev.clientY]; canvas.classList.add('dragging'); canvas.setPointerCapture(ev.pointerId);
}});
canvas.addEventListener('pointermove', (ev) => {{
  if (!dragging || !lastDrag) return;
  const dx = ev.clientX - lastDrag[0], dy = ev.clientY - lastDrag[1];
  lastDrag = [ev.clientX, ev.clientY];
  camera.yaw += dx * 0.01;
  camera.pitch = Math.max(-Math.PI/2 + 0.02, Math.min(Math.PI/2 - 0.02, camera.pitch + dy * 0.01));
  draw();
}});
canvas.addEventListener('pointerup', (ev) => {{
  dragging = false; lastDrag = null; canvas.classList.remove('dragging');
  try {{ canvas.releasePointerCapture(ev.pointerId); }} catch (err) {{}}
}});
canvas.addEventListener('pointercancel', () => {{ dragging = false; lastDrag = null; canvas.classList.remove('dragging'); }});
canvas.addEventListener('wheel', (ev) => {{
  ev.preventDefault();
  camera.zoom = Math.max(0.35, Math.min(4.0, camera.zoom * (ev.deltaY < 0 ? 1.08 : 0.92)));
  draw();
}}, {{passive: false}});
canvas.addEventListener('dblclick', () => setView('orbit'));
window.addEventListener('resize', resize);
resize();
</script>
</body>
</html>
"""


def _find_row(summary: dict[str, Any], backend: str, trial: str) -> dict[str, Any]:
    for row in summary.get("rows") or []:
        if row.get("backend") == backend and row.get("trial") == trial and row.get("status") == "valid":
            return row
    raise ValueError(f"No valid Level A row for backend={backend!r}, trial={trial!r}.")


def _edge_indices(names: list[str]) -> list[list[int]]:
    lookup = {_canon(name): idx for idx, name in enumerate(names)}
    out = []
    for a, b in EDGES:
        ai = lookup.get(_canon(a))
        bi = lookup.get(_canon(b))
        if ai is not None and bi is not None:
            out.append([ai, bi])
    return out


def _bounds(arrays: list[np.ndarray]) -> dict[str, float]:
    values = np.concatenate([np.asarray(arr, dtype=float).reshape(-1, 3) for arr in arrays], axis=0)
    finite = values[np.isfinite(values).all(axis=1)]
    if finite.size == 0:
        return {"radius": 1.0}
    mins = np.nanpercentile(finite, 1, axis=0)
    maxs = np.nanpercentile(finite, 99, axis=0)
    return {"radius": float(max(np.nanmax(maxs - mins) / 2.0, 0.5))}


def _short_label(name: str) -> str:
    aliases = {
        "hip_midpoint": "HIPMID",
        "pelvis": "PELV",
        "left_hip": "LHIP",
        "right_hip": "RHIP",
        "left_knee": "LKNE",
        "right_knee": "RKNE",
        "left_ankle": "LANK",
        "right_ankle": "RANK",
        "left_mtp": "LMTP",
        "right_mtp": "RMTP",
    }
    return aliases.get(str(name), str(name).upper())


def _round(values: np.ndarray, decimals: int) -> np.ndarray:
    return np.round(np.asarray(values, dtype=float), int(decimals))


def _canon(name: str) -> str:
    return str(name).strip().lower().replace("_", "").replace("-", "").replace(".", "")


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


if __name__ == "__main__":
    raise SystemExit(main())
