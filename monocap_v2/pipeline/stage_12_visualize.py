from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.logging_utils import write_json, write_yaml
from monocap_v2.core.smpl_marker_placement import LOWER_LIMB_MARKERS, build_lower_limb_marker_map_proposal, lower_limb_marker_placement_qc
from monocap_v2.core.smpl_mesh import load_smpl_faces
from monocap_v2.core.stage_utils import cached, stage_result
from monocap_v2.core.wham_timeline import build_wham_timeline_report


STAGE = "stage_12_visualize"

SKELETON_EDGES = [
    ("pelv", "lhip"),
    ("lhip", "lkne"),
    ("lkne", "lank"),
    ("lank", "ltoe"),
    ("pelv", "rhip"),
    ("rhip", "rkne"),
    ("rkne", "rank"),
    ("rank", "rtoe"),
    ("pelv", "spi1"),
    ("spi1", "spi2"),
    ("spi2", "spi3"),
    ("spi3", "neck"),
    ("neck", "head"),
    ("neck", "lcla"),
    ("lcla", "lsho"),
    ("lsho", "lelb"),
    ("lelb", "lwri"),
    ("lwri", "lhan"),
    ("neck", "rcla"),
    ("rcla", "rsho"),
    ("rsho", "relb"),
    ("relb", "rwri"),
    ("rwri", "rhan"),
]


def run(run_dir: Path, cfg: dict, force: bool = False) -> dict:
    registry = ArtifactRegistry(run_dir)
    out_path = registry.ensure_parent("visualize_qc")
    if cached(out_path, force):
        return stage_result(STAGE, "cached", output=str(out_path))

    vis_cfg = cfg.get("config", {}).get("visualization", {})
    if not vis_cfg.get("enabled", True):
        result = stage_result(STAGE, "skipped", reason="Visualization disabled in config.")
        write_json(out_path, result)
        return result

    outputs: dict[str, str] = {}
    warnings: list[str] = []
    smpl_mesh_report = None
    marker_placement = None

    try:
        initial = _read_pose(registry.get("pose3d_initial"))
    except FileNotFoundError:
        result = stage_result(STAGE, "skipped", reason="pose3d_initial artifact is missing.")
        write_json(out_path, result)
        return result

    if _has_finite_joints(initial):
        initial_path = registry.ensure_parent("pose3d_initial_preview")
        _write_pose_preview(initial, initial_path, vis_cfg)
        outputs["pose3d_initial_preview"] = str(initial_path)

        pelvis_path = registry.ensure_parent("pelvis_drift_plot")
        _write_pelvis_drift_plot(initial, pelvis_path)
        outputs["pelvis_drift_plot"] = str(pelvis_path)
    else:
        warnings.append("Initial pose has no finite joints to visualize.")

    marker_path = registry.get("virtual_markers")
    refined_path = registry.get("pose3d_refined")
    if refined_path.exists():
        refined = _read_pose(refined_path)
        if _has_finite_joints(refined):
            preview_path = registry.ensure_parent("pose3d_refined_preview")
            _write_pose_preview(refined, preview_path, vis_cfg)
            outputs["pose3d_refined_preview"] = str(preview_path)
        else:
            warnings.append("Refined pose has no finite joints to visualize.")
        if _has_smpl_vertices(refined):
            try:
                smpl_preview = registry.ensure_parent("smpl_vertices_preview")
                smpl_report = _write_smpl_vertices_preview(refined, smpl_preview, vis_cfg, marker_path if marker_path.exists() else None)
                smpl_qc = registry.ensure_parent("smpl_vertices_qc")
                write_json(smpl_qc, smpl_report)
                outputs["smpl_vertices_preview"] = str(smpl_preview)
                outputs["smpl_vertices_qc"] = str(smpl_qc)
                warnings.extend(str(w) for w in smpl_report.get("warnings", []))
            except Exception as exc:
                warnings.append(f"SMPL vertices preview was not written: {exc}")
            if (vis_cfg.get("smpl_mesh") or {}).get("enabled", True):
                try:
                    mesh_preview = registry.ensure_parent("smpl_mesh_preview")
                    smpl_mesh_report = _write_smpl_mesh_preview(refined, mesh_preview, cfg, marker_path if marker_path.exists() else None)
                    mesh_qc = registry.ensure_parent("smpl_mesh_qc")
                    write_json(mesh_qc, smpl_mesh_report)
                    outputs["smpl_mesh_preview"] = str(mesh_preview)
                    outputs["smpl_mesh_qc"] = str(mesh_qc)
                    warnings.extend(str(w) for w in smpl_mesh_report.get("warnings", []))
                except Exception as exc:
                    warnings.append(f"SMPL mesh preview was not written: {exc}")
            placement_cfg = cfg.get("config", {}).get("markers", {}).get("placement_qc", {})
            if placement_cfg.get("enabled", True) and marker_path.exists():
                try:
                    marker_payload = _load_marker_payload(marker_path)
                    marker_placement = lower_limb_marker_placement_qc(refined, marker_payload, placement_cfg)
                    proposal = build_lower_limb_marker_map_proposal(marker_payload)

                    placement_qc_path = registry.ensure_parent("smpl_marker_placement_qc")
                    proposal_path = registry.ensure_parent("marker_map_proposal")
                    placement_plot = registry.ensure_parent("smpl_marker_placement_plot")
                    placement_preview = registry.ensure_parent("smpl_marker_placement_preview")

                    marker_placement["proposal_output"] = str(proposal_path)
                    marker_placement["plot_output"] = str(placement_plot)
                    marker_placement["preview_output"] = str(placement_preview)
                    write_json(placement_qc_path, marker_placement)
                    write_yaml(proposal_path, proposal)
                    _write_marker_placement_plot(marker_placement, placement_plot)
                    _write_smpl_mesh_preview(refined, placement_preview, cfg, marker_path, set(LOWER_LIMB_MARKERS))

                    outputs["smpl_marker_placement_qc"] = str(placement_qc_path)
                    outputs["smpl_marker_placement_plot"] = str(placement_plot)
                    outputs["smpl_marker_placement_preview"] = str(placement_preview)
                    outputs["marker_map_proposal"] = str(proposal_path)
                    warnings.extend(str(w) for w in marker_placement.get("warnings", []))
                except Exception as exc:
                    warnings.append(f"SMPL marker placement QC was not written: {exc}")
    else:
        warnings.append("Refined pose artifact is missing; only initial pose was visualized.")

    contacts_path = registry.get("contacts")
    if contacts_path.exists():
        contact_plot = registry.ensure_parent("foot_contact_plot")
        _write_contact_plot(contacts_path, contact_plot)
        outputs["foot_contact_plot"] = str(contact_plot)
    else:
        warnings.append("Contacts artifact is missing; contact plot was not written.")

    marker_jump = None
    if marker_path.exists():
        try:
            marker_plot = registry.ensure_parent("marker_trajectory_plot")
            _write_marker_trajectory_plot(marker_path, marker_plot)
            outputs["marker_trajectory_plot"] = str(marker_plot)

            marker_jump = _marker_jump_diagnostics(marker_path)
            marker_jump_qc = registry.ensure_parent("marker_jump_qc")
            write_json(marker_jump_qc, marker_jump)
            outputs["marker_jump_qc"] = str(marker_jump_qc)

            marker_jump_plot = registry.ensure_parent("marker_jump_plot")
            _write_marker_jump_plot(marker_path, marker_jump, marker_jump_plot)
            outputs["marker_jump_plot"] = str(marker_jump_plot)

            marker_jump_preview = registry.ensure_parent("marker_jump_preview")
            _write_marker_jump_preview(marker_path, marker_jump, marker_jump_preview, vis_cfg)
            outputs["marker_jump_preview"] = str(marker_jump_preview)
            warnings.extend(str(w) for w in marker_jump.get("warnings", []))
        except Exception as exc:
            warnings.append(f"Marker visualization was not written: {exc}")

    wham_timeline = None
    if initial.get("backend") == "wham":
        wham_timeline = build_wham_timeline_report(initial, cfg)
        write_json(registry.ensure_parent("wham_timeline_qc"), wham_timeline)
        if wham_timeline.get("status") != "skipped":
            timeline_plot = registry.ensure_parent("wham_timeline_plot")
            _write_wham_timeline_plot(wham_timeline, timeline_plot)
            outputs["wham_timeline_plot"] = str(timeline_plot)
            try:
                overlay_path = registry.ensure_parent("wham_raw_overlay")
                _write_wham_raw_overlay(initial, overlay_path, vis_cfg)
                outputs["wham_raw_overlay"] = str(overlay_path)
            except Exception as exc:
                warnings.append(f"WHAM raw overlay was not written: {exc}")
            try:
                labeled_overlay = registry.ensure_parent("wham_labeled_overlay")
                _write_wham_labeled_overlay(initial, labeled_overlay, vis_cfg)
                outputs["wham_labeled_overlay"] = str(labeled_overlay)
            except Exception as exc:
                warnings.append(f"WHAM labeled overlay was not written: {exc}")
        warnings.extend(str(w) for w in wham_timeline.get("warnings", []))

    status = "ok" if outputs else "skipped"
    if outputs and wham_timeline and wham_timeline.get("status") == "warning":
        status = "warning"
    if outputs and marker_jump and marker_jump.get("status") == "warning":
        status = "warning"
    if outputs and smpl_mesh_report and smpl_mesh_report.get("status") == "warning":
        status = "warning"
    if outputs and marker_placement and marker_placement.get("status") == "warning":
        status = "warning"
    result = stage_result(
        STAGE,
        status,
        output=str(out_path),
        outputs=outputs,
        warnings=warnings,
        representation=initial.get("representation"),
        backend=initial.get("backend"),
        wham_timeline=wham_timeline,
        marker_jump=marker_jump,
        smpl_mesh=smpl_mesh_report,
        marker_placement=marker_placement,
    )
    write_json(out_path, result)
    return result


def _read_pose(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def _has_finite_joints(pose: dict) -> bool:
    joints = np.asarray(pose.get("joints_3d"), dtype=float)
    return joints.ndim == 3 and joints.shape[-1] == 3 and bool(np.isfinite(joints).any())


def _has_smpl_vertices(pose: dict) -> bool:
    vertices = np.asarray((pose.get("smpl") or {}).get("vertices"), dtype=float)
    return vertices.ndim == 3 and vertices.shape[-1] == 3 and bool(np.isfinite(vertices).any())


def _write_pose_preview(pose: dict, out_path: Path, vis_cfg: dict) -> None:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    joints = np.asarray(pose["joints_3d"], dtype=float)
    names = [str(name).lower() for name in pose.get("joint_names", [])]
    edges = _edge_indices(names)
    width = int(vis_cfg.get("preview_width", 960))
    height = int(vis_cfg.get("preview_height", 720))
    fps = float(vis_cfg.get("preview_fps") or min(float(pose.get("fps") or 30.0), 30.0))
    bounds = _axis_bounds(joints)
    pelvis_idx = _find_joint(names, ("pelv", "pelvis", "root"))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out_path}")

    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    try:
        for frame_idx, frame in enumerate(joints):
            ax.clear()
            _draw_frame(ax, frame, edges, bounds, pelvis_idx, frame_idx, pose)
            fig.tight_layout(pad=0.2)
            canvas.draw()
            rgba = np.asarray(canvas.buffer_rgba())
            bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()
        plt.close(fig)


def _draw_frame(ax, frame: np.ndarray, edges: list[tuple[int, int]], bounds: dict[str, tuple[float, float]], pelvis_idx: int | None, frame_idx: int, pose: dict) -> None:
    finite = np.isfinite(frame).all(axis=1)
    plot_frame = _to_plot_coords(frame)
    points = plot_frame[finite]
    if points.size:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=18, c="#222222", depthshade=False)
    for a, b in edges:
        if finite[a] and finite[b]:
            color = "#1f77b4" if _is_left(a, b, pose) else "#d62728" if _is_right(a, b, pose) else "#333333"
            ax.plot(
                [plot_frame[a, 0], plot_frame[b, 0]],
                [plot_frame[a, 1], plot_frame[b, 1]],
                [plot_frame[a, 2], plot_frame[b, 2]],
                color=color,
                linewidth=2,
            )
    if pelvis_idx is not None and finite[pelvis_idx]:
        pelvis = plot_frame[pelvis_idx]
        ax.scatter([pelvis[0]], [pelvis[1]], [pelvis[2]], s=42, c="#2ca02c", depthshade=False)
    ax.set_xlim(*bounds["x"])
    ax.set_ylim(*bounds["z"])
    ax.set_zlim(*bounds["up"])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_zlabel("Up (m)")
    ax.view_init(elev=18, azim=-70)
    ax.set_title(f"{pose.get('backend', 'pose3d')} {pose.get('representation', '')} | frame {frame_idx:04d}")


def _is_left(a: int, b: int, pose: dict) -> bool:
    names = [str(name).lower() for name in pose.get("joint_names", [])]
    return names[a].startswith("l") or names[b].startswith("l") or "left" in names[a] or "left" in names[b]


def _is_right(a: int, b: int, pose: dict) -> bool:
    names = [str(name).lower() for name in pose.get("joint_names", [])]
    return names[a].startswith("r") or names[b].startswith("r") or "right" in names[a] or "right" in names[b]


def _edge_indices(names: list[str]) -> list[tuple[int, int]]:
    aliases = _aliases(names)
    out: list[tuple[int, int]] = []
    for a_name, b_name in SKELETON_EDGES:
        a = aliases.get(a_name)
        b = aliases.get(b_name)
        if a is not None and b is not None:
            out.append((a, b))
    return out


def _aliases(names: list[str]) -> dict[str, int]:
    aliases: dict[str, int] = {}
    for i, name in enumerate(names):
        norm = name.replace("_", "").replace("-", "")
        aliases[norm] = i
        if norm.startswith("left"):
            aliases["l" + norm[4:]] = i
        if norm.startswith("right"):
            aliases["r" + norm[5:]] = i
    aliases.setdefault("pelv", _find_joint(names, ("pelv", "pelvis", "root")) or 0)
    return aliases


def _find_joint(names: list[str], candidates: Iterable[str]) -> int | None:
    canon = [name.replace("_", "").replace("-", "") for name in names]
    for candidate in candidates:
        c = candidate.replace("_", "").replace("-", "")
        for idx, name in enumerate(canon):
            if name == c:
                return idx
    for candidate in candidates:
        c = candidate.replace("_", "").replace("-", "")
        for idx, name in enumerate(canon):
            if c in name:
                return idx
    return None


def _axis_bounds(joints: np.ndarray) -> dict[str, tuple[float, float]]:
    finite = _to_plot_coords(joints)[np.isfinite(joints).all(axis=2)]
    if finite.size == 0:
        return {"x": (-1.0, 1.0), "z": (-1.0, 1.0), "up": (-1.0, 1.0)}
    mins = np.nanpercentile(finite, 2, axis=0)
    maxs = np.nanpercentile(finite, 98, axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.nanmax(maxs - mins) / 2.0)
    radius = max(radius, 0.75)
    return {
        "x": (float(center[0] - radius), float(center[0] + radius)),
        "z": (float(center[1] - radius), float(center[1] + radius)),
        "up": (float(center[2] - radius), float(center[2] + radius)),
    }


def _to_plot_coords(joints: np.ndarray) -> np.ndarray:
    """Convert backend camera coordinates to preview coordinates.

    MeTRAbs uses an image/camera-style convention where +Y points down. The
    preview uses +Up, so raw Y is negated for display only. Artifacts remain in
    backend-neutral metres and are not modified here.
    """
    coords = np.asarray(joints, dtype=float).copy()
    if coords.shape[-1] >= 3:
        x = coords[..., 0].copy()
        y_down = coords[..., 1].copy()
        z = coords[..., 2].copy()
        coords[..., 0] = x
        coords[..., 1] = z
        coords[..., 2] = -y_down
    return coords


def _write_pelvis_drift_plot(pose: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    joints = np.asarray(pose["joints_3d"], dtype=float)
    names = [str(name).lower() for name in pose.get("joint_names", [])]
    pelvis_idx = _find_joint(names, ("pelv", "pelvis", "root"))
    if pelvis_idx is None:
        pelvis_idx = 0
    pelvis = joints[:, pelvis_idx, :]
    finite = np.isfinite(pelvis).all(axis=1)
    drift = np.full((pelvis.shape[0],), np.nan, dtype=float)
    if np.any(finite):
        first = pelvis[np.flatnonzero(finite)[0], :]
        drift[finite] = np.linalg.norm(pelvis[finite][:, [0, 2]] - first[[0, 2]], axis=1)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    ax.plot(drift, color="#2ca02c", linewidth=2)
    ax.set_title("Pelvis Horizontal Drift")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Drift (m)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_contact_plot(contacts_path: Path, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(contacts_path, allow_pickle=True)
    keys = ["left_heel", "left_toe", "right_heel", "right_toe"]
    fig, ax = plt.subplots(figsize=(9, 4), dpi=140)
    for key in keys:
        if key in data:
            ax.plot(np.asarray(data[key], dtype=float), label=key)
    ax.set_title("Foot Contact Probability")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_smpl_vertices_preview(pose: dict, out_path: Path, vis_cfg: dict, marker_path: Path | None = None) -> dict:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    smpl = pose.get("smpl") or {}
    vertices = np.asarray(smpl.get("vertices"), dtype=float)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise RuntimeError("SMPL vertices must have shape [T, V, 3].")
    frame_indices, window_report = _smpl_preview_frame_indices(pose, vertices.shape[0], marker_path)
    if frame_indices.size == 0:
        raise RuntimeError("No SMPL frames selected for preview.")

    max_frames = int(vis_cfg.get("smpl_preview_max_frames", 120))
    if max_frames > 0 and frame_indices.size > max_frames:
        pick = np.linspace(0, frame_indices.size - 1, max_frames).round().astype(int)
        frame_indices = frame_indices[pick]

    max_vertices = int(vis_cfg.get("smpl_preview_max_vertices", 1600))
    vertex_indices = _sample_vertex_indices(vertices.shape[1], max_vertices)
    joints = np.asarray(pose.get("joints_3d"), dtype=float)
    has_joints = joints.ndim == 3 and joints.shape[-1] == 3 and joints.shape[0] == vertices.shape[0]
    names = [str(name).lower() for name in pose.get("joint_names", [])]
    edges = _edge_indices(names) if has_joints else []
    raw_frame_ids = _pose_raw_frame_ids(pose, vertices.shape[0])
    bounds = _axis_bounds(vertices[frame_indices][:, vertex_indices, :])
    width = int(vis_cfg.get("preview_width", 960))
    height = int(vis_cfg.get("preview_height", 720))
    fps = float(vis_cfg.get("smpl_preview_fps") or min(float(vis_cfg.get("preview_fps") or 30.0), 30.0))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open SMPL preview writer for {out_path}")

    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    try:
        for frame_idx in frame_indices.tolist():
            frame_vertices = vertices[int(frame_idx), vertex_indices, :]
            plot_vertices = _to_plot_coords(frame_vertices)
            finite = np.isfinite(frame_vertices).all(axis=1)
            ax.clear()
            if np.any(finite):
                colors = _smpl_vertex_colors(plot_vertices[finite])
                ax.scatter(
                    plot_vertices[finite, 0],
                    plot_vertices[finite, 1],
                    plot_vertices[finite, 2],
                    s=3.0,
                    c=colors,
                    alpha=0.72,
                    depthshade=False,
                    linewidths=0,
                )
            if has_joints:
                _draw_smpl_joint_overlay(ax, joints[int(frame_idx)], edges, names)
            raw_label = ""
            if raw_frame_ids.size > frame_idx:
                raw_label = f" | raw frame {int(raw_frame_ids[int(frame_idx)])}"
            ax.set_xlim(*bounds["x"])
            ax.set_ylim(*bounds["z"])
            ax.set_zlim(*bounds["up"])
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z (m)")
            ax.set_zlabel("Up (m)")
            ax.view_init(elev=18, azim=-70)
            ax.set_title(f"SMPL vertices | frame {int(frame_idx):04d}{raw_label} | {vertices.shape[1]} verts")
            fig.tight_layout(pad=0.2)
            canvas.draw()
            rgba = np.asarray(canvas.buffer_rgba())
            bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()
        plt.close(fig)

    warnings = []
    if vertex_indices.size < vertices.shape[1]:
        warnings.append(f"SMPL preview downsampled vertices from {vertices.shape[1]} to {vertex_indices.size}.")
    return {
        "stage": STAGE,
        "status": "ok",
        "output": str(out_path),
        "render_mode": "vertex_cloud",
        "backend": pose.get("backend"),
        "representation": pose.get("representation"),
        "coordinate_space": smpl.get("vertices_coordinate_space") or "unknown",
        "units": pose.get("units", "m"),
        "frames_available": int(vertices.shape[0]),
        "frames_rendered": int(frame_indices.size),
        "frame_start": int(frame_indices[0]),
        "frame_end": int(frame_indices[-1]),
        "vertices_available": int(vertices.shape[1]),
        "vertices_rendered": int(vertex_indices.size),
        "time_window": window_report,
        "warnings": warnings,
    }


def _write_smpl_mesh_preview(pose: dict, out_path: Path, cfg: dict, marker_path: Path | None = None, marker_filter: set[str] | None = None) -> dict:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    vis_cfg = cfg.get("config", {}).get("visualization", {})
    mesh_cfg = vis_cfg.get("smpl_mesh") or {}
    smpl = pose.get("smpl") or {}
    vertices = np.asarray(smpl.get("vertices"), dtype=float)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise RuntimeError("SMPL vertices must have shape [T, V, 3].")

    frame_indices, window_report = _smpl_preview_frame_indices(pose, vertices.shape[0], marker_path)
    if frame_indices.size == 0:
        raise RuntimeError("No SMPL frames selected for mesh preview.")
    max_frames = int(mesh_cfg.get("max_frames") or vis_cfg.get("smpl_preview_max_frames", 120))
    if max_frames > 0 and frame_indices.size > max_frames:
        pick = np.linspace(0, frame_indices.size - 1, max_frames).round().astype(int)
        frame_indices = frame_indices[pick]

    warnings: list[str] = []
    faces = None
    face_info: dict[str, object] = {}
    render_mode = "mesh"
    try:
        face_info = load_smpl_faces(cfg, vertex_count=vertices.shape[1])
        faces = np.asarray(face_info["faces"], dtype=np.int32)
    except Exception as exc:
        render_mode = "vertex_cloud_fallback"
        warnings.append(f"Could not load SMPL mesh faces; rendered vertex cloud fallback: {exc}")

    original_face_count = int(faces.shape[0]) if faces is not None else 0
    if faces is not None:
        faces = _decimate_faces(faces, int(mesh_cfg.get("max_faces", 6000)))
        vertex_indices = np.arange(vertices.shape[1], dtype=int)
    else:
        vertex_indices = _sample_vertex_indices(vertices.shape[1], int(vis_cfg.get("smpl_preview_max_vertices", 1600)))

    marker_overlay = _load_marker_overlay(marker_path, marker_filter=marker_filter) if mesh_cfg.get("show_markers", True) else None
    raw_frame_ids = _pose_raw_frame_ids(pose, vertices.shape[0])
    bounds_vertices = vertices[frame_indices][:, vertex_indices, :]
    bounds = _axis_bounds(bounds_vertices)
    width = int(vis_cfg.get("preview_width", 960))
    height = int(vis_cfg.get("preview_height", 720))
    fps = float(mesh_cfg.get("preview_fps") or vis_cfg.get("preview_fps") or 30.0)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open SMPL mesh preview writer for {out_path}")

    marker_frames_used = 0
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    try:
        for frame_idx in frame_indices.tolist():
            frame_vertices = vertices[int(frame_idx)]
            plot_vertices = _to_plot_coords(frame_vertices)
            ax.clear()
            if faces is not None:
                ax.plot_trisurf(
                    plot_vertices[:, 0],
                    plot_vertices[:, 1],
                    plot_vertices[:, 2],
                    triangles=faces,
                    color="#89a8d6",
                    alpha=0.62,
                    linewidth=0.03,
                    edgecolor="#5b6f91",
                    shade=True,
                )
            else:
                sampled = plot_vertices[vertex_indices]
                finite = np.isfinite(sampled).all(axis=1)
                if np.any(finite):
                    ax.scatter(
                        sampled[finite, 0],
                        sampled[finite, 1],
                        sampled[finite, 2],
                        s=3.0,
                        c=_smpl_vertex_colors(sampled[finite]),
                        alpha=0.72,
                        depthshade=False,
                        linewidths=0,
                    )
            raw_frame = int(raw_frame_ids[int(frame_idx)]) if raw_frame_ids.size > frame_idx else None
            if marker_overlay is not None:
                used = _draw_marker_overlay(ax, marker_overlay, raw_frame, int(frame_idx), bool(mesh_cfg.get("show_marker_labels", True)))
                marker_frames_used += int(used)
            raw_label = f" | raw frame {raw_frame}" if raw_frame is not None else ""
            ax.set_xlim(*bounds["x"])
            ax.set_ylim(*bounds["z"])
            ax.set_zlim(*bounds["up"])
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z (m)")
            ax.set_zlabel("Up (m)")
            ax.view_init(elev=18, azim=-70)
            face_label = f"{faces.shape[0]} faces" if faces is not None else f"{vertex_indices.size} vertices"
            ax.set_title(f"SMPL {render_mode} + markers | frame {int(frame_idx):04d}{raw_label} | {face_label}")
            fig.tight_layout(pad=0.2)
            canvas.draw()
            rgba = np.asarray(canvas.buffer_rgba())
            bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()
        plt.close(fig)

    marker_status = "disabled"
    marker_count = 0
    if marker_overlay is not None:
        marker_count = len(marker_overlay.get("names") or [])
        marker_status = "ok" if marker_frames_used else "no_aligned_marker_frames"
        if not marker_frames_used:
            warnings.append("No marker frames aligned with SMPL mesh preview frames.")

    return {
        "stage": STAGE,
        "status": "warning" if warnings else "ok",
        "output": str(out_path),
        "render_mode": render_mode,
        "backend": pose.get("backend"),
        "representation": pose.get("representation"),
        "coordinate_space": smpl.get("vertices_coordinate_space") or "unknown",
        "units": pose.get("units", "m"),
        "frames_available": int(vertices.shape[0]),
        "frames_rendered": int(frame_indices.size),
        "frame_start": int(frame_indices[0]),
        "frame_end": int(frame_indices[-1]),
        "vertices_available": int(vertices.shape[1]),
        "faces_available": original_face_count,
        "faces_rendered": int(faces.shape[0]) if faces is not None else 0,
        "face_source": face_info.get("source"),
        "face_source_type": face_info.get("source_type"),
        "marker_overlay_status": marker_status,
        "marker_frames_aligned": int(marker_frames_used),
        "marker_count": int(marker_count),
        "time_window": window_report,
        "warnings": warnings,
    }


def _write_marker_placement_plot(report: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_marker = report.get("per_marker") or {}
    names = [name for name in LOWER_LIMB_MARKERS if name in per_marker]
    distances = [
        float(((per_marker.get(name) or {}).get("distance_to_source_joint_m") or {}).get("median") or 0.0)
        for name in names
    ]
    jumps = [
        float(((per_marker.get(name) or {}).get("trajectory") or {}).get("max_jump_m") or 0.0)
        for name in names
    ]
    thresholds = report.get("thresholds") or {}
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), dpi=140, sharex=True)
    x = np.arange(len(names))
    axes[0].bar(x, distances, color="#4c78a8")
    axes[0].axhline(float(thresholds.get("max_joint_distance_m", 0.20)), color="#d62728", linestyle="--", linewidth=1.2)
    axes[0].set_ylabel("Median distance (m)")
    axes[0].set_title(f"Lower-Limb Marker Placement | status={report.get('status')}")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, jumps, color="#72b7b2")
    axes[1].axhline(float(thresholds.get("max_marker_jump_m", 0.20)), color="#d62728", linestyle="--", linewidth=1.2)
    axes[1].set_ylabel("Max jump (m)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([name.replace("DBG_", "") for name in names], rotation=45, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)

    worst = report.get("worst_marker")
    if worst in names:
        idx = names.index(worst)
        for ax in axes:
            ax.axvline(idx, color="#f58518", linestyle=":", linewidth=1.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _draw_smpl_joint_overlay(ax, frame: np.ndarray, edges: list[tuple[int, int]], names: list[str]) -> None:
    plot_frame = _to_plot_coords(frame)
    finite = np.isfinite(frame).all(axis=1)
    for a, b in edges:
        if a < len(finite) and b < len(finite) and finite[a] and finite[b]:
            color = "#1f77b4" if names[a].startswith("l") or names[b].startswith("l") else "#d62728" if names[a].startswith("r") or names[b].startswith("r") else "#111111"
            ax.plot(
                [plot_frame[a, 0], plot_frame[b, 0]],
                [plot_frame[a, 1], plot_frame[b, 1]],
                [plot_frame[a, 2], plot_frame[b, 2]],
                color=color,
                linewidth=1.6,
                alpha=0.9,
            )


def _smpl_vertex_colors(plot_vertices: np.ndarray) -> np.ndarray:
    up = plot_vertices[:, 2]
    if not np.isfinite(up).any() or float(np.nanmax(up) - np.nanmin(up)) < 1e-9:
        values = np.full((plot_vertices.shape[0],), 0.5, dtype=float)
    else:
        values = (up - np.nanmin(up)) / (np.nanmax(up) - np.nanmin(up))
    base = np.zeros((plot_vertices.shape[0], 4), dtype=float)
    base[:, 0] = 0.12 + 0.48 * values
    base[:, 1] = 0.24 + 0.34 * (1.0 - values)
    base[:, 2] = 0.56 + 0.18 * values
    base[:, 3] = 0.72
    return base


def _sample_vertex_indices(vertex_count: int, max_vertices: int) -> np.ndarray:
    if max_vertices <= 0 or vertex_count <= max_vertices:
        return np.arange(vertex_count, dtype=int)
    return np.linspace(0, vertex_count - 1, max_vertices).round().astype(int)


def _decimate_faces(faces: np.ndarray, max_faces: int) -> np.ndarray:
    if max_faces <= 0 or faces.shape[0] <= max_faces:
        return faces
    indices = np.linspace(0, faces.shape[0] - 1, max_faces).round().astype(int)
    return faces[indices]


def _load_marker_overlay(marker_path: Path | None, marker_filter: set[str] | None = None) -> dict | None:
    if marker_path is None or not marker_path.exists():
        return None
    payload = _load_marker_payload(marker_path)
    markers = np.asarray(payload.get("markers_m"), dtype=float)
    names = [str(name) for name in payload.get("marker_names", [])]
    if markers.ndim != 3 or markers.shape[-1] != 3 or not names:
        return None
    if marker_filter is not None:
        keep = [idx for idx, name in enumerate(names) if name in marker_filter]
        if not keep:
            return None
        markers = markers[:, keep, :]
        names = [names[idx] for idx in keep]
    raw_frame_ids = _optional_1d_array(payload.get("raw_frame_ids")).astype(int)
    raw_lookup = {int(raw): idx for idx, raw in enumerate(raw_frame_ids.tolist())} if raw_frame_ids.size == markers.shape[0] else {}
    return {"markers": markers, "names": names, "raw_frame_ids": raw_frame_ids, "raw_lookup": raw_lookup}


def _draw_marker_overlay(ax, overlay: dict, raw_frame: int | None, frame_idx: int, show_labels: bool) -> bool:
    markers = np.asarray(overlay.get("markers"), dtype=float)
    names = [str(name) for name in overlay.get("names", [])]
    marker_idx = None
    raw_lookup = overlay.get("raw_lookup") or {}
    if raw_frame is not None and raw_frame in raw_lookup:
        marker_idx = int(raw_lookup[raw_frame])
    elif markers.shape[0] > frame_idx:
        marker_idx = int(frame_idx)
    if marker_idx is None or marker_idx >= markers.shape[0]:
        return False
    frame = markers[marker_idx]
    plot_frame = _to_plot_coords(frame)
    finite = np.isfinite(frame).all(axis=1)
    for idx, point in enumerate(plot_frame):
        if not finite[idx]:
            continue
        name = names[idx] if idx < len(names) else str(idx)
        color = _marker_color(name)
        ax.scatter([point[0]], [point[1]], [point[2]], s=36, c=color, edgecolors="#111111", linewidths=0.5, depthshade=False)
        if show_labels:
            ax.text(point[0], point[1], point[2], _compact_marker_label(name), fontsize=6, color=color)
    return True


def _marker_color(name: str) -> str:
    if name.startswith("DBG_L"):
        return "#1f77b4"
    if name.startswith("DBG_R"):
        return "#d62728"
    return "#2ca02c"


def _compact_marker_label(name: str) -> str:
    return name[4:] if name.startswith("DBG_") else name


def _smpl_preview_frame_indices(pose: dict, frame_count: int, marker_path: Path | None) -> tuple[np.ndarray, dict]:
    all_indices = np.arange(frame_count, dtype=int)
    if marker_path is None or not marker_path.exists():
        return all_indices, {"status": "full_sequence", "reason": "No trimmed marker artifact was available."}
    try:
        payload = _load_marker_payload(marker_path)
    except Exception as exc:
        return all_indices, {"status": "full_sequence", "reason": f"Could not read marker time window: {exc}"}
    time_window = payload.get("time_window") or {}
    raw_start = time_window.get("raw_frame_start")
    raw_end = time_window.get("raw_frame_end")
    raw_frame_ids = _pose_raw_frame_ids(pose, frame_count)
    if raw_start is None or raw_end is None or raw_frame_ids.size != frame_count:
        return all_indices, {"status": "full_sequence", "reason": "Marker time window does not define raw-frame bounds."}
    mask = (raw_frame_ids >= int(raw_start)) & (raw_frame_ids <= int(raw_end))
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return all_indices, {"status": "full_sequence", "reason": "Marker time window did not overlap SMPL frame ids."}
    return selected.astype(int), {
        "status": "applied",
        "source": "marker_time_window",
        "raw_frame_start": int(raw_start),
        "raw_frame_end": int(raw_end),
        "kept_frames": int(selected.size),
        "original_frames": int(frame_count),
        "local_frame_start": int(selected[0]),
        "local_frame_end": int(selected[-1]),
    }


def _pose_raw_frame_ids(pose: dict, frame_count: int) -> np.ndarray:
    meta = pose.get("backend_meta") or {}
    raw = _optional_1d_array(meta.get("raw_frame_ids"))
    if raw.size == frame_count:
        return raw.astype(int)
    frame_ids = _optional_1d_array(meta.get("frame_ids"))
    if frame_ids.size == frame_count:
        start_frame = int(meta.get("start_frame") or 0)
        return (frame_ids.astype(int) + start_frame).astype(int)
    return np.asarray([], dtype=int)


def _write_marker_trajectory_plot(marker_path: Path, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with marker_path.open("rb") as f:
        payload = pickle.load(f)
    markers = np.asarray(payload.get("markers_m"), dtype=float)
    names = [str(name) for name in payload.get("marker_names", [])]
    if markers.ndim != 3 or markers.shape[-1] != 3:
        raise RuntimeError("Invalid marker payload shape.")
    pelvis_idx = names.index("DBG_PELV") if "DBG_PELV" in names else 0
    ref = markers[:, pelvis_idx : pelvis_idx + 1, :]
    rel = markers - ref
    distance = np.linalg.norm(rel, axis=2)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    for idx, name in enumerate(names[: min(len(names), 24)]):
        ax.plot(distance[:, idx], linewidth=1.2, label=name)
    ax.set_title(f"Debug Marker Distance From {names[pelvis_idx]}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Distance (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _load_marker_payload(marker_path: Path) -> dict:
    with marker_path.open("rb") as f:
        return pickle.load(f)


def _marker_jump_diagnostics(marker_path: Path, threshold_m: float = 0.5, top_n: int = 10) -> dict:
    payload = _load_marker_payload(marker_path)
    markers = np.asarray(payload.get("markers_m"), dtype=float)
    names = [str(name) for name in payload.get("marker_names", [])]
    if markers.ndim != 3 or markers.shape[-1] != 3:
        raise RuntimeError("Invalid marker payload shape.")

    warnings: list[str] = []
    frame_count = int(markers.shape[0])
    marker_count = int(markers.shape[1])
    if frame_count < 2 or marker_count == 0:
        warnings.append("Marker jump diagnostics require at least two frames and one marker.")
        return {
            "status": "warning",
            "marker_set": payload.get("marker_set"),
            "debug": bool(payload.get("debug", False)),
            "coordinate_space": payload.get("coordinate_space"),
            "units": payload.get("units", "m"),
            "frames": frame_count,
            "markers": marker_count,
            "threshold_m": threshold_m,
            "max_jump_m": 0.0,
            "top_jumps": [],
            "classification": "not_enough_data",
            "warnings": warnings,
        }

    jumps = np.linalg.norm(np.diff(markers, axis=0), axis=2)
    finite_jumps = np.isfinite(jumps)
    if not finite_jumps.any():
        warnings.append("No finite frame-to-frame marker jumps were found.")
        max_jump = 0.0
        top_entries: list[dict] = []
    else:
        scores = np.where(finite_jumps, jumps, -np.inf).reshape(-1)
        order = np.argsort(scores)[::-1]
        top_entries = []
        for flat_idx in order[:top_n]:
            value = float(scores[int(flat_idx)])
            if not np.isfinite(value):
                continue
            transition_idx, marker_idx = divmod(int(flat_idx), marker_count)
            entry = {
                "from_frame": int(transition_idx),
                "to_frame": int(transition_idx + 1),
                "marker_index": int(marker_idx),
                "marker": names[marker_idx] if marker_idx < len(names) else str(marker_idx),
                "jump_m": value,
            }
            _add_marker_timebase(entry, payload, transition_idx)
            top_entries.append(entry)
        max_jump = float(top_entries[0]["jump_m"]) if top_entries else 0.0

    classification, transition_stats = _classify_marker_jump(markers, top_entries[0] if top_entries else None)
    median_jump = float(np.nanmedian(jumps[finite_jumps])) if finite_jumps.any() else 0.0
    non_contiguous = _non_contiguous_frame_ids(payload)
    if non_contiguous:
        warnings.append(non_contiguous)
    if max_jump > threshold_m:
        warning = f"Maximum frame-to-frame marker jump is large: {max_jump:.3f} m."
        if top_entries:
            warning += f" Worst marker: {top_entries[0]['marker']} frames {top_entries[0]['from_frame']}->{top_entries[0]['to_frame']}."
        warnings.append(warning)

    return {
        "status": "warning" if warnings else "ok",
        "marker_set": payload.get("marker_set"),
        "debug": bool(payload.get("debug", False)),
        "coordinate_space": payload.get("coordinate_space"),
        "units": payload.get("units", "m"),
        "fps": payload.get("fps"),
        "frames": frame_count,
        "markers": marker_count,
        "threshold_m": threshold_m,
        "max_jump_m": max_jump,
        "median_jump_m": median_jump,
        "top_jumps": top_entries,
        "classification": classification,
        "transition_stats": transition_stats,
        "warnings": warnings,
    }


def _add_marker_timebase(entry: dict, payload: dict, transition_idx: int) -> None:
    raw_frame_ids = _optional_1d_array(payload.get("raw_frame_ids"))
    frame_ids = _optional_1d_array(payload.get("frame_ids"))
    raw_time_s = _optional_1d_array(payload.get("raw_video_time_s"))
    rel_time_s = _optional_1d_array(payload.get("wham_relative_time_s"))
    for name, values in [
        ("raw_frame", raw_frame_ids),
        ("frame_id", frame_ids),
        ("raw_time_s", raw_time_s),
        ("relative_time_s", rel_time_s),
    ]:
        if values.size > transition_idx + 1:
            entry[f"{name}_from"] = _json_scalar(values[transition_idx])
            entry[f"{name}_to"] = _json_scalar(values[transition_idx + 1])


def _optional_1d_array(value) -> np.ndarray:
    if value is None:
        return np.asarray([])
    try:
        return np.asarray(value).reshape(-1)
    except Exception:
        return np.asarray([])


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return float(value) if isinstance(value, float) else value


def _non_contiguous_frame_ids(payload: dict) -> str | None:
    values = _optional_1d_array(payload.get("raw_frame_ids"))
    label = "raw_frame_ids"
    if values.size == 0:
        values = _optional_1d_array(payload.get("frame_ids"))
        label = "frame_ids"
    if values.size < 2:
        return None
    diffs = np.diff(values.astype(float))
    if np.isfinite(diffs).all() and np.any(np.abs(diffs - 1.0) > 1e-6):
        return f"Marker {label} are non-contiguous; preview uses marker sequence frame indices."
    return None


def _classify_marker_jump(markers: np.ndarray, max_entry: dict | None) -> tuple[str, dict]:
    if not max_entry:
        return "not_available", {}
    transition_idx = int(max_entry["from_frame"])
    if transition_idx < 0 or transition_idx + 1 >= markers.shape[0]:
        return "not_available", {}
    vectors = markers[transition_idx + 1] - markers[transition_idx]
    valid = np.isfinite(vectors).all(axis=1)
    if not np.any(valid):
        return "not_available", {}
    valid_vectors = vectors[valid]
    norms = np.linalg.norm(valid_vectors, axis=1)
    median_vector = np.nanmedian(valid_vectors, axis=0)
    median_norm = float(np.linalg.norm(median_vector))
    residuals = np.linalg.norm(valid_vectors - median_vector[None, :], axis=1)
    coherence_threshold = max(0.05, 0.2 * max(median_norm, 1e-9))
    coherent_fraction = float(np.mean(residuals <= coherence_threshold))
    median_marker_jump = float(np.nanmedian(norms))
    max_jump = float(max_entry.get("jump_m") or 0.0)
    if median_norm > 0.25 and coherent_fraction >= 0.6:
        classification = "whole_body_translation_or_root_jump"
    elif max_jump > max(0.10, 3.0 * max(median_marker_jump, 1e-9)) and coherent_fraction < 0.5:
        classification = "isolated_marker_or_local_marker_jump"
    else:
        classification = "mixed_marker_motion"
    return classification, {
        "transition_from_frame": transition_idx,
        "valid_marker_count": int(np.sum(valid)),
        "median_translation_vector_m": [float(v) for v in median_vector.tolist()],
        "median_translation_norm_m": median_norm,
        "median_marker_jump_m": median_marker_jump,
        "coherent_fraction": coherent_fraction,
        "coherence_threshold_m": coherence_threshold,
    }


def _write_marker_jump_plot(marker_path: Path, diagnostics: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = _load_marker_payload(marker_path)
    markers = np.asarray(payload.get("markers_m"), dtype=float)
    if markers.ndim != 3 or markers.shape[0] < 2:
        raise RuntimeError("Marker jump plot requires at least two frames.")
    jumps = np.linalg.norm(np.diff(markers, axis=0), axis=2)
    max_per_transition = np.nanmax(jumps, axis=1)
    median_per_transition = np.nanmedian(jumps, axis=1)
    x = np.arange(jumps.shape[0])
    worst = (diagnostics.get("top_jumps") or [{}])[0]
    worst_frame = worst.get("from_frame")

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    ax.plot(x, max_per_transition, color="#d62728", linewidth=2, label="max marker jump")
    ax.plot(x, median_per_transition, color="#1f77b4", linewidth=1.6, label="median marker jump")
    ax.axhline(float(diagnostics.get("threshold_m") or 0.5), color="#777777", linestyle="--", linewidth=1.2, label="warning threshold")
    if worst_frame is not None:
        ax.axvline(int(worst_frame), color="#d62728", linestyle=":", linewidth=1.5)
        ax.text(
            int(worst_frame),
            float(diagnostics.get("max_jump_m") or 0.0),
            f" {worst.get('marker', '?')} {float(worst.get('jump_m') or 0.0):.2f} m",
            va="bottom",
            fontsize=8,
            color="#7f1d1d",
        )
    ax.set_title(f"Marker Jump Diagnostics | {diagnostics.get('classification', 'unknown')}")
    ax.set_xlabel("Transition frame")
    ax.set_ylabel("Frame-to-frame distance (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_marker_jump_preview(marker_path: Path, diagnostics: dict, out_path: Path, vis_cfg: dict) -> None:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    payload = _load_marker_payload(marker_path)
    markers = np.asarray(payload.get("markers_m"), dtype=float)
    names = [str(name) for name in payload.get("marker_names", [])]
    if markers.ndim != 3 or markers.shape[-1] != 3 or markers.shape[0] == 0:
        raise RuntimeError("Invalid marker payload shape.")

    worst = (diagnostics.get("top_jumps") or [{}])[0]
    center = int(worst.get("from_frame") or 0)
    radius = int(vis_cfg.get("marker_jump_preview_radius", 5))
    start = max(0, center - radius)
    stop = min(markers.shape[0], center + radius + 2)
    window = markers[start:stop]
    bounds = _axis_bounds(window)
    edges = _debug_marker_edges(names)
    width = int(vis_cfg.get("preview_width", 960))
    height = int(vis_cfg.get("preview_height", 720))
    fps = float(vis_cfg.get("marker_jump_preview_fps") or min(float(vis_cfg.get("preview_fps") or 6.0), 12.0))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open marker jump preview writer for {out_path}")

    highlight_idx = int(worst.get("marker_index")) if worst.get("marker_index") is not None else None
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    try:
        for frame_idx in range(start, stop):
            frame = markers[frame_idx]
            plot_frame = _to_plot_coords(frame)
            finite = np.isfinite(frame).all(axis=1)
            ax.clear()
            if np.any(finite):
                ax.scatter(plot_frame[finite, 0], plot_frame[finite, 1], plot_frame[finite, 2], s=24, c="#222222", depthshade=False)
            for a, b in edges:
                if a < len(finite) and b < len(finite) and finite[a] and finite[b]:
                    color = "#1f77b4" if names[a].startswith("DBG_L") or names[b].startswith("DBG_L") else "#d62728" if names[a].startswith("DBG_R") or names[b].startswith("DBG_R") else "#333333"
                    ax.plot(
                        [plot_frame[a, 0], plot_frame[b, 0]],
                        [plot_frame[a, 1], plot_frame[b, 1]],
                        [plot_frame[a, 2], plot_frame[b, 2]],
                        color=color,
                        linewidth=2,
                    )
            if highlight_idx is not None and highlight_idx < len(finite) and finite[highlight_idx]:
                p = plot_frame[highlight_idx]
                ax.scatter([p[0]], [p[1]], [p[2]], s=90, c="#ffcc00", edgecolors="#111111", depthshade=False)
            raw_label = _frame_label(payload, frame_idx)
            ax.set_xlim(*bounds["x"])
            ax.set_ylim(*bounds["z"])
            ax.set_zlim(*bounds["up"])
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z (m)")
            ax.set_zlabel("Up (m)")
            ax.view_init(elev=18, azim=-70)
            ax.set_title(
                f"Marker jump focus | frame {frame_idx} {raw_label}\n"
                f"{worst.get('marker', '?')} {float(worst.get('jump_m') or 0.0):.3f} m | {diagnostics.get('classification', 'unknown')}"
            )
            fig.tight_layout(pad=0.2)
            canvas.draw()
            rgba = np.asarray(canvas.buffer_rgba())
            bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()
        plt.close(fig)


def _frame_label(payload: dict, frame_idx: int) -> str:
    raw_frame_ids = _optional_1d_array(payload.get("raw_frame_ids"))
    raw_time_s = _optional_1d_array(payload.get("raw_video_time_s"))
    parts = []
    if raw_frame_ids.size > frame_idx:
        parts.append(f"raw {int(raw_frame_ids[frame_idx])}")
    if raw_time_s.size > frame_idx:
        parts.append(f"t={float(raw_time_s[frame_idx]):.3f}s")
    return f"({' | '.join(parts)})" if parts else ""


def _debug_marker_edges(names: list[str]) -> list[tuple[int, int]]:
    lookup = {name: idx for idx, name in enumerate(names)}
    pairs = [
        ("DBG_PELV", "DBG_LHIP"),
        ("DBG_PELV", "DBG_RHIP"),
        ("DBG_PELV", "DBG_SP1"),
        ("DBG_SP1", "DBG_SP3"),
        ("DBG_SP3", "DBG_NECK"),
        ("DBG_NECK", "DBG_HEAD"),
        ("DBG_LHIP", "DBG_LKNE"),
        ("DBG_LKNE", "DBG_LANK"),
        ("DBG_LANK", "DBG_LTOE"),
        ("DBG_LANK", "DBG_LHEE"),
        ("DBG_RHIP", "DBG_RKNE"),
        ("DBG_RKNE", "DBG_RANK"),
        ("DBG_RANK", "DBG_RTOE"),
        ("DBG_RANK", "DBG_RHEE"),
        ("DBG_NECK", "DBG_LSHO"),
        ("DBG_LSHO", "DBG_LELB"),
        ("DBG_LELB", "DBG_LWRI"),
        ("DBG_NECK", "DBG_RSHO"),
        ("DBG_RSHO", "DBG_RELB"),
        ("DBG_RELB", "DBG_RWRI"),
    ]
    return [(lookup[a], lookup[b]) for a, b in pairs if a in lookup and b in lookup]


def _write_wham_raw_overlay(pose: dict, out_path: Path, vis_cfg: dict) -> None:
    import cv2

    meta = pose.get("backend_meta") or {}
    source_video = meta.get("source_video") or pose.get("source_video")
    if not source_video:
        raise RuntimeError("WHAM source video is missing.")
    pose2d = pose.get("pose2d") or {}
    xy = np.asarray(pose2d.get("xy"), dtype=float)
    conf = np.asarray(pose2d.get("confidence"), dtype=float)
    frame_ids = np.asarray(meta.get("raw_frame_ids", meta.get("frame_ids", [])), dtype=int).reshape(-1)
    if xy.ndim != 3 or xy.shape[-1] != 2 or frame_ids.size == 0:
        raise RuntimeError("WHAM pose2d or frame_ids are missing.")
    n = min(xy.shape[0], frame_ids.size)
    frame_ids = frame_ids[:n]
    xy = xy[:n]
    conf = conf[:n] if conf.ndim == 2 else np.ones(xy.shape[:2], dtype=float)

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open WHAM source video: {source_video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(vis_cfg.get("preview_fps") or min(float(pose.get("fps") or 30.0), 30.0))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open WHAM overlay writer for {out_path}")

    min_conf = float(vis_cfg.get("wham_overlay_min_conf", 0.2))
    edges = _coco17_edges(xy.shape[1])
    try:
        for out_idx, frame_id in enumerate(frame_ids.tolist()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            _draw_2d_keypoints(frame, xy[out_idx], conf[out_idx], edges, min_conf)
            label = f"WHAM track {meta.get('selected_track_id', '?')} | raw frame {int(frame_id)} | t={frame_id / float(pose.get('fps') or 30.0):.3f}s"
            cv2.putText(frame, label, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, label, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
        cap.release()


def _write_wham_labeled_overlay(pose: dict, out_path: Path, vis_cfg: dict) -> None:
    import cv2

    meta = pose.get("backend_meta") or {}
    source_video = meta.get("source_video") or pose.get("source_video")
    if not source_video:
        raise RuntimeError("WHAM source video is missing.")
    pose2d = pose.get("pose2d") or {}
    xy = np.asarray(pose2d.get("xy"), dtype=float)
    conf = np.asarray(pose2d.get("confidence"), dtype=float)
    frame_ids = np.asarray(meta.get("raw_frame_ids", meta.get("frame_ids", [])), dtype=int).reshape(-1)
    if xy.ndim != 3 or xy.shape[-1] != 2 or frame_ids.size == 0:
        raise RuntimeError("WHAM pose2d or frame_ids are missing.")

    n = min(xy.shape[0], frame_ids.size)
    xy = xy[:n]
    frame_ids = frame_ids[:n]
    conf = conf[:n] if conf.ndim == 2 else np.ones(xy.shape[:2], dtype=float)
    labels = _wham_2d_labels(pose2d.get("names"), xy.shape[1])

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open WHAM source video: {source_video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(vis_cfg.get("preview_fps") or min(float(pose.get("fps") or 30.0), 30.0))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open WHAM labeled overlay writer for {out_path}")

    min_conf = float(vis_cfg.get("wham_overlay_min_conf", 0.2))
    try:
        for out_idx, frame_id in enumerate(frame_ids.tolist()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            _draw_labeled_2d_markers(frame, xy[out_idx], conf[out_idx], labels, min_conf)
            header = (
                f"WHAM 2D labels only | track {meta.get('selected_track_id', '?')} | "
                f"raw frame {int(frame_id)} | t={frame_id / float(pose.get('fps') or 30.0):.3f}s"
            )
            _draw_text_with_outline(frame, header, (20, 36), 0.75, (255, 255, 255), thickness=2)
            _draw_text_with_outline(frame, "left=red  right=blue  midline=white", (20, 68), 0.62, (230, 230, 230), thickness=2)
            writer.write(frame)
    finally:
        writer.release()
        cap.release()


def _draw_2d_keypoints(frame: np.ndarray, xy: np.ndarray, conf: np.ndarray, edges: list[tuple[int, int]], min_conf: float) -> None:
    import cv2

    valid = np.isfinite(xy).all(axis=1) & np.isfinite(conf) & (conf >= min_conf)
    for a, b in edges:
        if a < len(valid) and b < len(valid) and valid[a] and valid[b]:
            cv2.line(frame, tuple(np.round(xy[a]).astype(int)), tuple(np.round(xy[b]).astype(int)), (50, 220, 255), 2, cv2.LINE_AA)
    for idx, point in enumerate(xy):
        if valid[idx]:
            cv2.circle(frame, tuple(np.round(point).astype(int)), 4, (0, 255, 80), -1, cv2.LINE_AA)


def _draw_labeled_2d_markers(frame: np.ndarray, xy: np.ndarray, conf: np.ndarray, labels: list[str], min_conf: float) -> None:
    import cv2

    valid = np.isfinite(xy).all(axis=1) & np.isfinite(conf) & (conf >= min_conf)
    height, width = frame.shape[:2]
    for idx, point in enumerate(xy):
        if idx >= len(labels) or not valid[idx]:
            continue
        x, y = np.round(point).astype(int).tolist()
        label = labels[idx]
        color = _label_color_bgr(label)
        cv2.circle(frame, (x, y), 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
        dx = -78 if label.startswith("R") else 9
        dy = -9 if "ANK" not in label else 18
        tx = int(np.clip(x + dx, 4, max(4, width - 82)))
        ty = int(np.clip(y + dy, 18, max(18, height - 8)))
        cv2.line(frame, (x, y), (tx, ty), color, 1, cv2.LINE_AA)
        _draw_text_with_outline(frame, label, (tx, ty), 0.52, color, thickness=1)


def _draw_text_with_outline(frame: np.ndarray, text: str, origin: tuple[int, int], scale: float, color: tuple[int, int, int], thickness: int = 1) -> None:
    import cv2

    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _wham_2d_labels(names: Iterable | None, joint_count: int) -> list[str]:
    coco17 = [
        "NOSE",
        "LEYE",
        "REYE",
        "LEAR",
        "REAR",
        "LSHO",
        "RSHO",
        "LELB",
        "RELB",
        "LWRI",
        "RWRI",
        "LHIP",
        "RHIP",
        "LKNE",
        "RKNE",
        "LANK",
        "RANK",
    ]
    if int(joint_count) == len(coco17):
        return coco17
    raw = [str(name) for name in (names or [])]
    if len(raw) == joint_count:
        return [_short_2d_label(name, idx) for idx, name in enumerate(raw)]
    return [f"KP{idx:02d}" for idx in range(int(joint_count))]


def _short_2d_label(name: str, idx: int) -> str:
    normalized = name.strip().upper().replace("WHAM_KP_", "KP")
    if normalized and normalized != name:
        return normalized
    return normalized or f"KP{idx:02d}"


def _label_color_bgr(label: str) -> tuple[int, int, int]:
    if label.startswith("L"):
        return (70, 70, 255)
    if label.startswith("R"):
        return (255, 120, 60)
    return (255, 255, 255)


def _coco17_edges(joint_count: int) -> list[tuple[int, int]]:
    edges = [
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
    ]
    return [(a, b) for a, b in edges if a < joint_count and b < joint_count]


def _write_wham_timeline_plot(report: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timebase = report.get("timebase") or {}
    alignment = report.get("raw_sync_alignment") or {}
    overlap = report.get("overlap") or {}
    raw_count = int(timebase.get("source_frame_count") or alignment.get("raw_frame_count") or 0)
    wham_start = timebase.get("frame_start")
    wham_end = timebase.get("frame_end")
    sync_start = alignment.get("best_raw_offset")
    sync_count = alignment.get("sync_frame_count")

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=150)
    if raw_count > 0:
        ax.broken_barh([(0, raw_count)], (30, 8), facecolors="#dddddd", label="raw video")
    if sync_start is not None and sync_count:
        ax.broken_barh([(int(sync_start), int(sync_count))], (18, 8), facecolors="#4c78a8", label="estimated synced clip")
    if wham_start is not None and wham_end is not None:
        ax.broken_barh([(int(wham_start), int(wham_end) - int(wham_start) + 1)], (6, 8), facecolors="#f58518", label="WHAM selected frames")
    if overlap.get("overlap_frames"):
        ax.broken_barh(
            [(int(overlap["overlap_frame_start"]), int(overlap["overlap_frames"]))],
            (0, 4),
            facecolors="#54a24b",
            label="overlap",
        )
    ax.set_yticks([34, 22, 10, 2])
    ax.set_yticklabels(["raw", "sync", "WHAM", "overlap"])
    ax.set_xlabel("Raw video frame")
    score = alignment.get("best_match_score")
    title = "WHAM Timeline"
    if score is not None:
        title += f" | raw/sync score={float(score):.3f}"
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
