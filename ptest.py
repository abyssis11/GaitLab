#!/usr/bin/env python3
"""
TRC MoCap Viewer (matplotlib) — robust header & headless-safe
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple, Optional

# Try to ensure an interactive backend before importing pyplot
import matplotlib

def _try_enable_gui_backend():
    current = matplotlib.get_backend().lower()
    non_gui = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
    if current in non_gui:
        for cand in ("QtAgg", "Qt5Agg", "TkAgg", "MacOSX"):
            try:
                matplotlib.use(cand, force=True)
                return
            except Exception:
                continue
_try_enable_gui_backend()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import re

def split_ws(s: str):
    # Robust split on tabs/whitespace (including NBSP) and strip weird BOMs
    s = s.replace("\ufeff", "").replace("\u00A0", " ")
    return [tok for tok in re.split(r"[\t\s]+", s.strip()) if tok]

def _find_time_col(header_tokens):
    # Case-insensitive + common variants; fallback to column 1 after Frame#
    candidates = {"time", "time(s)", "timesec", "timestamp", "seconds", "sec"}
    lowered = [t.lower() for t in header_tokens]
    for i, tok in enumerate(lowered):
        if tok in candidates:
            return i
    norm = [re.sub(r"[^a-z()]", "", t) for t in lowered]
    for i, tok in enumerate(norm):
        if tok in {"time", "times", "timestamp", "seconds", "sec"}:
            return i
    if lowered and lowered[0].startswith("frame"):
        return 1
    raise ValueError(f"Could not find a time column in header tokens: {header_tokens}")

def _first_data_line_index(lines, start_idx):
    # Find the first line whose first token is an integer (Frame#)
    for i in range(start_idx, len(lines)):
        raw = lines[i]
        if raw.strip() == "":
            continue
        toks = split_ws(raw)
        if not toks:
            continue
        try:
            int(toks[0])  # Frame number
            return i
        except Exception:
            continue
    return None

def parse_trc(path: str) -> dict:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    if len(lines) < 6:
        raise ValueError("File too short to be a valid TRC.")

    # Find header line
    header_idx = None
    for i, ln in enumerate(lines[:200]):
        s = ln.strip()
        if s.startswith("Frame#") or s.lower().startswith("frame"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find the 'Frame#' header line in TRC.")

    # Parse meta from previous two lines if present
    rate = n_frames = n_markers = None
    units = None
    try:
        keys = split_ws(lines[header_idx - 2])
        vals = split_ws(lines[header_idx - 1])
        kv = {keys[i]: vals[i] for i in range(min(len(keys), len(vals)))}
        if "DataRate" in kv: rate = float(kv["DataRate"])
        if "NumFrames" in kv: n_frames = int(float(kv["NumFrames"]))
        if "NumMarkers" in kv: n_markers = int(float(kv["NumMarkers"]))
        if "Units" in kv: units = kv["Units"]
    except Exception:
        pass

    # Header tokens and time column
    header_tokens = split_ws(lines[header_idx])
    if len(header_tokens) < 3:
        raise ValueError(f"Malformed header line near: {lines[header_idx]!r}")
    time_col_idx = _find_time_col(header_tokens)
    names_all = header_tokens[time_col_idx + 1 :]

    # XYZ hint line (may exist)
    hint_tokens = split_ws(lines[header_idx + 1]) if header_idx + 1 < len(lines) else []
    has_xyz_hint = len(hint_tokens) >= 3 and all(t[0].upper() in "XYZ" for t in hint_tokens[:3])

    # Marker count from hint, else header names, else NumMarkers
    M_hint = 0
    if has_xyz_hint:
        # Prefer numeric suffix (e.g., Z51 -> 51), else count X tokens
        nums = []
        for t in hint_tokens:
            m = re.search(r"[XYZxyz](\d+)$", t)
            if m:
                nums.append(int(m.group(1)))
        if nums:
            M_hint = max(nums)
        else:
            M_hint = len([t for t in hint_tokens if t.upper().startswith("X")])
    M = M_hint or len(names_all) or (n_markers or 0)
    if M <= 0:
        raise ValueError("Could not infer number of markers.")

    # Find first numeric data row (don’t assume header+2)
    scan_start = header_idx + 1
    data_start_idx = _first_data_line_index(lines, scan_start)
    if data_start_idx is None:
        raise ValueError("No data-like lines found after the header.")

    # Trim names to M; then verify against the first data row and shrink if needed
    names = names_all[:M]
    first_tokens = split_ws(lines[data_start_idx])
    M_infer = max(0, (len(first_tokens) - 2) // 3)
    if M_infer and M_infer < M:
        M = M_infer
        names = names[:M]

    # Read all numeric rows
    need_cols = 2 + 3 * M
    raw = []
    for ln in lines[data_start_idx:]:
        if ln.strip() == "":
            continue
        toks = split_ws(ln)
        if len(toks) < need_cols:
            continue
        # Require the first token to be an integer frame index
        try:
            int(toks[0])
        except Exception:
            continue
        raw.append(toks[:need_cols])
    if not raw:
        raise ValueError(
            f"No data rows found (needed at least {need_cols} columns per row; "
            f"header had M={M} markers, names={len(names)})."
        )

    data = np.array(raw, dtype=float)
    time = data[:, 1].astype(float)
    coords = data[:, 2:]
    T = coords.shape[0]
    frames = coords.reshape(T, M, 3)

    # Fill metadata
    if rate is None:
        if T >= 2:
            dt = float(np.median(np.diff(time)))
            rate = 1.0 / dt if dt > 0 else 100.0
        else:
            rate = 100.0
    if n_frames is None: n_frames = T
    if n_markers is None: n_markers = M
    if units is None: units = "mm"

    return {
        "rate": float(rate),
        "n_frames": int(n_frames),
        "n_markers": int(n_markers),
        "units": str(units),
        "frames": frames,
        "time": time,
        "names": names,
    }

def _exists(names: List[str], key: str) -> bool:
    return key in names

def default_connections(names: List[str]) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    def add(a, b):
        if _exists(names, a) and _exists(names, b):
            edges.append((a, b))

    for side in ["r", "L", "R", "l"]:
        ankle = f"{side}_ankle"
        calc = f"{side}_calc"
        meta = f"{side}_5meta"
        toe = f"{side}_toe"
        shank = f"{side}_shank_antsup"
        knee  = f"{side}_knee"
        add(toe, meta); add(meta, calc); add(calc, ankle); add(ankle, shank); add(shank, knee)

    for side in ["r", "L", "R", "l"]:
        knee = f"{side}_knee"
        thigh_candidates = [f"{side}_thigh{i}" for i in range(1, 6)]
        present = [t for t in thigh_candidates if _exists(names, t)]
        if present:
            add(knee, present[0])
            for a, b in zip(present, present[1:]):
                add(a, b)

    for side in ["r", "L", "R", "l"]:
        hjc = f"{side}_HJC"; hjc_reg = f"{side}_HJC_reg"; asis = f"{side}.ASIS"; psis = f"{side}.PSIS"
        add(asis, psis); add(hjc, asis); add(hjc, psis); add(hjc_reg, hjc)

    for sh in ["R_Shoulder", "L_Shoulder"]:
        add("C7", sh)
    for side in ["R", "L"]:
        hum = f"{side}_humerus"; elbow_m = f"{side}_elbow_med"; elbow_l = f"{side}_elbow_lat"
        fore = f"{side}_forearm"; wr1 = f"{side}_wrist_radius"; wr2 = f"{side}_wrist_ulna"; stern = f"{side}_Sternum"
        add(stern, f"{side}_Shoulder"); add(f"{side}_Shoulder", hum); add(hum, elbow_m); add(elbow_m, elbow_l)
        add(elbow_l, fore); add(fore, wr1); add(wr1, wr2)

    add("R_Sternum", "L_Sternum")
    add("R_Shoulder", "L_Shoulder")
    for asis in ["r.ASIS", "L.ASIS", "R.ASIS", "l.ASIS"]:
        add(asis, "R_Sternum"); add(asis, "L_Sternum")
    return edges

def _axis_equal_3d(ax, X: np.ndarray):
    x_max, y_max, z_max = X.max(axis=0)
    x_min, y_min, z_min = X.min(axis=0)
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    if max_range <= 0: max_range = 1.0
    mid_x = 0.5 * (x_max + x_min); mid_y = 0.5 * (y_max + y_min); mid_z = 0.5 * (z_max + z_min)
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

def plot_static(frame_xyz: np.ndarray, names: List[str], units: str = "mm", connections: Optional[List[Tuple[str,str]]] = None, title: str = "", show: bool = True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    xs, ys, zs = frame_xyz[:,0], frame_xyz[:,1], frame_xyz[:,2]
    ax.scatter(xs, ys, zs)
    if connections:
        name_to_idx = {n:i for i,n in enumerate(names)}
        for a, b in connections:
            i, j = name_to_idx[a], name_to_idx[b]
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]])
    _axis_equal_3d(ax, frame_xyz)
    ax.set_xlabel(f"X ({units})"); ax.set_ylabel(f"Y ({units})"); ax.set_zlabel(f"Z ({units})")
    ax.set_title(title or "TRC frame")
    if show: plt.show()
    return fig

def animate(frames_xyz: np.ndarray, names: List[str], time: np.ndarray, units: str = "mm", connections: Optional[List[Tuple[str,str]]] = None, fps: Optional[int] = None, interval_ms: Optional[float] = None, save_path: Optional[str] = None):
    T, M, _ = frames_xyz.shape
    if fps is None and interval_ms is None and T >= 2:
        dt = np.median(np.diff(time)); interval_ms = max(1.0, 1000.0 * dt)
    if fps is not None: interval_ms = 1000.0 / fps

    fig = plt.figure(); ax = fig.add_subplot(111, projection="3d")
    scat = ax.scatter([], [], []); lines = []
    name_to_idx = {n:i for i,n in enumerate(names)}
    if connections:
        for _ in connections:
            line, = ax.plot([], [], []); lines.append(line)

    all_pts = frames_xyz.reshape(-1, 3); _axis_equal_3d(ax, all_pts)
    ax.set_xlabel(f"X ({units})"); ax.set_ylabel(f"Y ({units})"); ax.set_zlabel(f"Z ({units})"); ax.set_title("TRC animation")

    def init():
        scat._offsets3d = ([], [], [])
        for line in lines: line.set_data([], []); line.set_3d_properties([])
        return [scat, *lines]

    def update(t):
        pts = frames_xyz[t]; xs, ys, zs = pts[:,0], pts[:,1], pts[:,2]
        scat._offsets3d = (xs, ys, zs)
        if connections:
            for k, (a, b) in enumerate(connections):
                i, j = name_to_idx[a], name_to_idx[b]
                lines[k].set_data([xs[i], xs[j]], [ys[i], ys[j]]); lines[k].set_3d_properties([zs[i], zs[j]])
        ax.set_title(f"TRC animation  t={time[t]:.3f}s")
        return [scat, *lines]

    anim = FuncAnimation(fig, update, init_func=init, frames=T, interval=interval_ms, blit=False)

    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        if ext in [".mp4", ".m4v", ".mov"]:
            anim.save(save_path, writer="ffmpeg")
        elif ext in [".gif"]:
            anim.save(save_path, writer="imagemagick")
        else:
            print(f"Unrecognized extension '{ext}'. Try .mp4 or .gif")
    else:
        non_gui = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
        if matplotlib.get_backend().lower() in non_gui:
            print("No GUI backend available. Use --save out.mp4 to save the animation instead.")
        else:
            plt.show()

def _backend_is_gui() -> bool:
    non_gui = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
    return matplotlib.get_backend().lower() not in non_gui

def main():
    parser = argparse.ArgumentParser(description="Visualize TRC Motion Capture data (3D).")
    parser.add_argument("--file", required=True, help="Path to .trc file")
    parser.add_argument("--static", action="store_true", help="Show a single frame snapshot")
    parser.add_argument("--frame", type=int, default=0, help="Frame index for --static")
    parser.add_argument("--animate", action="store_true", help="Animate the trial")
    parser.add_argument("--start", type=int, default=0, help="Start frame (inclusive)")
    parser.add_argument("--end", type=int, default=-1, help="End frame (exclusive, -1 for end)")
    parser.add_argument("--downsample", type=int, default=1, help="Use every Nth frame")
    parser.add_argument("--fps", type=int, default=None, help="Animation FPS (overrides TRC timing)")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save animation (.mp4/.gif)")
    parser.add_argument("--no-connect", action="store_true", help="Do not draw skeleton connections")
    parser.add_argument("--out", type=str, default=None, help="When using --static, save the figure to this path instead of showing a window")
    args = parser.parse_args()

    d = parse_trc(args.file)
    names: List[str] = d["names"]  # type: ignore
    frames: np.ndarray = d["frames"]  # (T, M, 3)
    time: np.ndarray = d["time"]
    units: str = d["units"]

    # Subselect frames
    T = frames.shape[0]
    end = T if args.end < 0 else min(args.end, T)
    start = max(0, args.start)
    idx = np.arange(start, end, dtype=int)
    if args.downsample > 1:
        idx = idx[::args.downsample]

    frames = frames[idx]
    time = time[idx]

    conns = [] if args.no_connect else default_connections(names)

    if args.static:
        fi = max(0, min(args.frame, frames.shape[0]-1))
        want_show = _backend_is_gui() and (args.out is None)
        fig = plot_static(frames[fi], names, units=units, connections=conns, title=f"Frame {fi}", show=want_show)
        if args.out is not None or not _backend_is_gui():
            out_path = args.out or "trc_frame.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved static frame to {out_path}")
    elif args.animate:
        animate(frames, names, time, units=units, connections=conns, fps=args.fps, save_path=args.save)
    else:
        animate(frames, names, time, units=units, connections=conns, fps=args.fps, save_path=args.save)

if __name__ == "__main__":
    main()
