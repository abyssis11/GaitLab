#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monocap_v2.core.logging_utils import read_yaml, write_json
from monocap_v2.core.smpl_model import SMPL_24_JOINT_NAMES, SMPL_FOOT_VERTEX_NAMES, resolve_smpl_model_path


CONTACT_JOINTS = {
    "left_heel": ("left_heel", "lheel"),
    "left_toe": ("left_big_toe", "left_toe", "ltoe"),
    "right_heel": ("right_heel", "rheel"),
    "right_toe": ("right_big_toe", "right_toe", "rtoe"),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Refine WHAM SMPL root parameters in the WHAM/Torch environment.")
    ap.add_argument("--input-pose", required=True, type=Path)
    ap.add_argument("--run-config", required=True, type=Path)
    ap.add_argument("--contacts", type=Path, default=None)
    ap.add_argument("--output-pose", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--settings-json", default="{}")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    try:
        settings = json.loads(args.settings_json or "{}")
        run_cfg = read_yaml(args.run_config)
        with args.input_pose.open("rb") as f:
            pose = pickle.load(f)
        contacts = _read_contacts(args.contacts) if args.contacts else None
        refined, report = refine_wham_smpl_root(pose, run_cfg, settings, contacts, started)
        args.output_pose.parent.mkdir(parents=True, exist_ok=True)
        with args.output_pose.open("wb") as f:
            pickle.dump(refined, f)
        write_json(args.report, report)
        return 0
    except Exception as exc:
        report = {
            "status": "failed",
            "method": "wham_smpl_root_refinement",
            "error": str(exc),
            "runtime_sec": time.perf_counter() - started,
            "mocap_used_in_objective": False,
        }
        try:
            write_json(args.report, report)
        except Exception:
            pass
        print(f"[wham-smpl-root-worker] failed: {exc}", file=sys.stderr, flush=True)
        return 1


def refine_wham_smpl_root(
    pose: dict[str, Any],
    run_cfg: dict[str, Any],
    settings: dict[str, Any],
    contacts: dict[str, Any] | None,
    started: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from smplx import SMPL

    started = time.perf_counter() if started is None else started
    device = _device(settings)
    smpl_payload = pose.get("smpl") or {}
    vertices_initial = np.asarray(smpl_payload.get("vertices"), dtype=np.float32)
    if vertices_initial.ndim != 3 or vertices_initial.shape[-1] != 3:
        raise ValueError(f"Expected smpl.vertices with shape [T, V, 3], got {vertices_initial.shape}")
    frames = int(vertices_initial.shape[0])
    betas_np = _expand_betas(np.asarray(smpl_payload.get("betas"), dtype=np.float32), frames)
    body_pose_np = np.asarray(smpl_payload.get("body_pose"), dtype=np.float32).reshape(frames, -1)
    global_orient_np = np.asarray(smpl_payload.get("global_orient"), dtype=np.float32).reshape(frames, 3)
    if body_pose_np.shape[1] != 69:
        raise ValueError(f"Expected SMPL body_pose to flatten to [T, 69], got {body_pose_np.shape}")

    smpl_model_path, smpl_gender = resolve_smpl_model_path(run_cfg)
    model = SMPL(model_path=str(smpl_model_path), gender=smpl_gender, batch_size=frames).to(device)
    model.eval()

    betas = torch.as_tensor(betas_np, dtype=torch.float32, device=device)
    body_pose_initial = torch.as_tensor(body_pose_np, dtype=torch.float32, device=device)
    global_orient_initial = torch.as_tensor(global_orient_np, dtype=torch.float32, device=device)
    vertices_target = torch.as_tensor(vertices_initial, dtype=torch.float32, device=device)
    zero_transl = torch.zeros((frames, 3), dtype=torch.float32, device=device)

    with torch.no_grad():
        verts_zero = _smpl_forward(model, betas, body_pose_initial, global_orient_initial, zero_transl)
        initial_camera_transl = torch.mean(vertices_target - verts_zero, dim=1)
        verts_recon = _smpl_forward(model, betas, body_pose_initial, global_orient_initial, initial_camera_transl)
        recon_diff = torch.linalg.norm(verts_recon - vertices_target, dim=-1)
        finite_recon = recon_diff[torch.isfinite(recon_diff)]
        if finite_recon.numel() == 0:
            initial_reconstruction_median_m = float("nan")
            initial_reconstruction_max_m = float("nan")
        else:
            initial_reconstruction_median_m = float(torch.median(finite_recon).detach().cpu())
            initial_reconstruction_max_m = float(torch.max(finite_recon).detach().cpu())

    joint_regressors, joint_names, regressor_report = _load_joint_regressors(model, pose, run_cfg, device)
    with torch.no_grad():
        initial_joints = _regress_joints(verts_recon, joint_regressors)
    contact_specs = _build_contact_specs(
        initial_joints.detach().cpu().numpy(),
        joint_names,
        contacts,
        settings,
    )

    transl_delta = torch.zeros_like(initial_camera_transl, requires_grad=True)
    orient_delta = torch.zeros_like(global_orient_initial, requires_grad=True)
    optimize_body_pose = bool(settings.get("optimize_body_pose", False))
    body_pose_delta = torch.zeros_like(body_pose_initial, requires_grad=optimize_body_pose)
    params = [transl_delta, orient_delta]
    if optimize_body_pose:
        params.append(body_pose_delta)
    opt = torch.optim.Adam(params, lr=float(settings.get("learning_rate", 0.03)))
    max_iterations = int(settings.get("max_iterations", 80))
    weights = settings.get("weights") or {}
    loss_history: list[dict[str, float]] = []

    for iteration in range(max_iterations):
        opt.zero_grad(set_to_none=True)
        transl = initial_camera_transl + transl_delta
        global_orient = global_orient_initial + orient_delta
        body_pose = body_pose_initial + body_pose_delta
        vertices = _smpl_forward(model, betas, body_pose, global_orient, transl)
        joints = _regress_joints(vertices, joint_regressors)
        terms = _loss_terms(
            transl,
            global_orient,
            body_pose,
            transl_delta,
            orient_delta,
            body_pose_delta,
            joints,
            contact_specs,
            float(pose.get("fps") or 30.0),
            settings,
            weights,
        )
        loss = terms["total"]
        loss.backward()
        opt.step()
        _clamp_delta_(transl_delta, float(settings.get("max_translation_delta_m", 0.35)))
        _clamp_delta_(orient_delta, np.deg2rad(float(settings.get("max_orientation_delta_deg", 15.0))))
        if optimize_body_pose:
            _clamp_pose_delta_(body_pose_delta, np.deg2rad(float(settings.get("max_body_pose_delta_deg", 8.0))))
        loss_history.append({name: float(value.detach().cpu()) for name, value in terms.items()})

    with torch.no_grad():
        transl_final = initial_camera_transl + transl_delta
        global_orient_final = global_orient_initial + orient_delta
        body_pose_final = body_pose_initial + body_pose_delta
        vertices_final = _smpl_forward(model, betas, body_pose_final, global_orient_final, transl_final)
        joints_final = _regress_joints(vertices_final, joint_regressors)

    refined = copy.deepcopy(pose)
    refined["joints_3d"] = joints_final.detach().cpu().numpy().astype(np.float32)
    refined["joint_names"] = joint_names
    refined_smpl = refined.setdefault("smpl", {})
    refined_smpl["vertices"] = vertices_final.detach().cpu().numpy().astype(np.float32)
    refined_smpl["global_orient"] = global_orient_final.detach().cpu().numpy().astype(np.float32)
    refined_smpl["body_pose"] = body_pose_final.detach().cpu().numpy().reshape(frames, 23, 3).astype(np.float32)
    refined_smpl["transl"] = transl_final.detach().cpu().numpy().astype(np.float32)
    refined_smpl["transl_coordinate_space"] = "wham_camera_local_reconstructed_from_vertices"
    refined_smpl["vertices_coordinate_space"] = smpl_payload.get("vertices_coordinate_space", "wham_camera_local")
    refined_smpl["rotation_representation"] = smpl_payload.get("rotation_representation", "axis_angle")

    translation_delta_np = transl_delta.detach().cpu().numpy()
    orientation_delta_np = orient_delta.detach().cpu().numpy()
    body_pose_delta_np = body_pose_delta.detach().cpu().numpy().reshape(frames, -1, 3)
    optimized_parameters = ["smpl.transl", "smpl.global_orient"]
    fixed_parameters = ["smpl.betas", "backend_meta.frame_ids"]
    if optimize_body_pose:
        optimized_parameters.append("smpl.body_pose")
    else:
        fixed_parameters.append("smpl.body_pose")
    before_metrics = _contact_metrics(initial_joints.detach().cpu().numpy(), joint_names, contacts, settings, float(pose.get("fps") or 30.0))
    after_metrics = _contact_metrics(refined["joints_3d"], joint_names, contacts, settings, float(pose.get("fps") or 30.0))
    report = {
        "status": _status(initial_reconstruction_median_m, settings),
        "method": "wham_smpl_root_refinement",
        "optimized_parameters": optimized_parameters,
        "fixed_parameters": fixed_parameters,
        "optimize_body_pose": optimize_body_pose,
        "mocap_used_in_objective": False,
        "device": str(device),
        "iterations": max_iterations,
        "learning_rate": float(settings.get("learning_rate", 0.03)),
        "initial_reconstruction_error_m": {
            "median": initial_reconstruction_median_m,
            "max": initial_reconstruction_max_m,
            "mode": "camera_local_translation_reconstructed_from_vertices",
        },
        "loss_initial": loss_history[0] if loss_history else {},
        "loss_final": loss_history[-1] if loss_history else {},
        "loss_history": loss_history,
        "contact_specs": _public_contact_specs(contact_specs),
        "contact_metrics_before": before_metrics,
        "contact_metrics_after": after_metrics,
        "root_delta": {
            "translation_mean_m": _safe_float(np.nanmean(np.linalg.norm(translation_delta_np, axis=1))),
            "translation_max_m": _safe_float(np.nanmax(np.linalg.norm(translation_delta_np, axis=1))),
            "orientation_mean_deg": _safe_float(np.rad2deg(np.nanmean(np.linalg.norm(orientation_delta_np, axis=1)))),
            "orientation_max_deg": _safe_float(np.rad2deg(np.nanmax(np.linalg.norm(orientation_delta_np, axis=1)))),
        },
        "body_pose_delta": {
            "mean_deg": _safe_float(np.rad2deg(np.nanmean(np.linalg.norm(body_pose_delta_np, axis=2)))),
            "max_deg": _safe_float(np.rad2deg(np.nanmax(np.linalg.norm(body_pose_delta_np, axis=2)))),
        },
        "smpl_consistency": {
            "status": "ok",
            "mode": "regenerated_from_smpl_forward",
            "joint_regressors": regressor_report,
            "body_pose_unchanged": bool(np.allclose(np.asarray(refined_smpl["body_pose"]), np.asarray(smpl_payload["body_pose"]))),
            "betas_unchanged": bool(np.allclose(np.asarray(refined_smpl["betas"]), np.asarray(smpl_payload["betas"]))),
            "transl_source": "WHAM camera-local translation reconstructed by aligning SMPL zero-translation vertices to saved WHAM vertices.",
        },
        "runtime_sec": time.perf_counter() - started,
    }
    refined["refinement"] = {
        "status": report["status"],
        "method": "wham_smpl_root_refinement",
        "mocap_used_in_objective": False,
        "source_stage": "stage_04_pose3d_initial",
        "wham_smpl_root_refinement": report,
    }
    return refined, report


def _smpl_forward(model: Any, betas: Any, body_pose: Any, global_orient: Any, transl: Any) -> Any:
    output = model(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        return_verts=True,
    )
    return output.vertices


def _loss_terms(
    transl: Any,
    global_orient: Any,
    body_pose: Any,
    transl_delta: Any,
    orient_delta: Any,
    body_pose_delta: Any,
    joints: Any,
    contact_specs: list[dict[str, Any]],
    fps: float,
    settings: dict[str, Any],
    weights: dict[str, Any],
) -> dict[str, Any]:
    import torch

    terms: dict[str, Any] = {}
    terms["translation_prior"] = float(weights.get("translation_prior", 20.0)) * torch.mean(transl_delta**2)
    terms["orientation_prior"] = float(weights.get("orientation_prior", 5.0)) * torch.mean(orient_delta**2)
    terms["translation_smoothness"] = float(weights.get("translation_smoothness", 2.0)) * _second_diff_loss(transl)
    terms["orientation_smoothness"] = float(weights.get("orientation_smoothness", 0.5)) * _second_diff_loss(global_orient)
    if bool(settings.get("optimize_body_pose", False)):
        terms["body_pose_prior"] = float(weights.get("body_pose_prior", 15.0)) * torch.mean(body_pose_delta**2)
        terms["body_pose_smoothness"] = float(weights.get("body_pose_smoothness", 0.5)) * _second_diff_loss(body_pose)
    contact_velocity = torch.zeros((), dtype=joints.dtype, device=joints.device)
    contact_position = torch.zeros((), dtype=joints.dtype, device=joints.device)
    flat_floor = torch.zeros((), dtype=joints.dtype, device=joints.device)
    for spec in contact_specs:
        frames = torch.as_tensor(spec["frames"], dtype=torch.long, device=joints.device)
        foot = joints.index_select(0, frames)[:, int(spec["joint_index"]), :]
        if foot.shape[0] >= 2:
            contact_velocity = contact_velocity + torch.mean((torch.diff(foot[:, [0, 2]], dim=0) * fps) ** 2)
        anchor_h = torch.as_tensor(spec["horizontal_anchor"], dtype=joints.dtype, device=joints.device)
        contact_position = contact_position + torch.mean((foot[:, [0, 2]] - anchor_h) ** 2)
        floor_y = torch.as_tensor(float(spec["floor_y"]), dtype=joints.dtype, device=joints.device)
        flat_floor = flat_floor + torch.mean((foot[:, 1] - floor_y) ** 2)
    count = max(len(contact_specs), 1)
    terms["contact_velocity"] = float(weights.get("contact_velocity", 0.5)) * contact_velocity / count
    terms["contact_position"] = float(weights.get("contact_position", 20.0)) * contact_position / count
    terms["flat_floor"] = float(weights.get("flat_floor", 5.0)) * flat_floor / count
    terms["translation_bound"] = 10.0 * _bound_loss(transl_delta, float(settings.get("max_translation_delta_m", 0.35)))
    terms["orientation_bound"] = 10.0 * _bound_loss(orient_delta, np.deg2rad(float(settings.get("max_orientation_delta_deg", 15.0))))
    if bool(settings.get("optimize_body_pose", False)):
        terms["body_pose_bound"] = 10.0 * _pose_bound_loss(body_pose_delta, np.deg2rad(float(settings.get("max_body_pose_delta_deg", 8.0))))
    total = torch.zeros((), dtype=joints.dtype, device=joints.device)
    for value in terms.values():
        total = total + value
    terms["total"] = total
    return terms


def _second_diff_loss(values: Any) -> Any:
    import torch

    if values.shape[0] < 3:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    diff = values[2:] - 2.0 * values[1:-1] + values[:-2]
    return torch.mean(diff**2)


def _bound_loss(values: Any, threshold: float) -> Any:
    import torch

    if threshold <= 0:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    norm = torch.linalg.norm(values, dim=1)
    return torch.mean(torch.relu(norm - threshold) ** 2)


def _pose_bound_loss(values: Any, threshold: float) -> Any:
    import torch

    if threshold <= 0:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    pose = values.reshape(values.shape[0], -1, 3)
    norm = torch.linalg.norm(pose, dim=2)
    return torch.mean(torch.relu(norm - threshold) ** 2)


def _clamp_delta_(values: Any, threshold: float) -> None:
    if threshold <= 0:
        return
    import torch

    with torch.no_grad():
        norm = torch.linalg.norm(values, dim=1, keepdim=True)
        scale = torch.clamp(threshold / torch.clamp(norm, min=1e-8), max=1.0)
        values.mul_(scale)


def _clamp_pose_delta_(values: Any, threshold: float) -> None:
    if threshold <= 0:
        return
    import torch

    with torch.no_grad():
        pose = values.reshape(values.shape[0], -1, 3)
        norm = torch.linalg.norm(pose, dim=2, keepdim=True)
        scale = torch.clamp(threshold / torch.clamp(norm, min=1e-8), max=1.0)
        pose.mul_(scale)


def _load_joint_regressors(model: Any, pose: dict[str, Any], run_cfg: dict[str, Any], device: Any) -> tuple[list[Any], list[str], list[dict[str, Any]]]:
    import torch

    regressors = [model.J_regressor.to(device=device, dtype=torch.float32)]
    names = list(SMPL_24_JOINT_NAMES)
    report = [{"type": "smpl_24_J_regressor", "joint_count": len(SMPL_24_JOINT_NAMES)}]
    repo_root = Path(str(run_cfg.get("repo_root") or ".")).resolve()
    wham_repo = Path(str(((run_cfg.get("config") or {}).get("wham") or {}).get("repo_path") or "external/WHAM"))
    if not wham_repo.is_absolute():
        wham_repo = repo_root / wham_repo
    feet_path = wham_repo / "dataset" / "body_models" / "J_regressor_feet.npy"
    if feet_path.exists():
        feet = np.asarray(np.load(feet_path), dtype=np.float32)
        if feet.ndim == 2 and feet.shape[1] == int(model.J_regressor.shape[1]):
            regressors.append(torch.as_tensor(feet, dtype=torch.float32, device=device))
            foot_names = SMPL_FOOT_VERTEX_NAMES[: feet.shape[0]]
            names.extend(foot_names)
            report.append({"type": "smpl_vertex_foot_landmarks", "source": str(feet_path), "joint_count": len(foot_names), "names": foot_names})
    artifact_names = [str(name) for name in pose.get("joint_names", [])]
    if len(artifact_names) == len(names):
        names = artifact_names
    return regressors, names, report


def _regress_joints(vertices: Any, regressors: list[Any]) -> Any:
    import torch

    chunks = [torch.einsum("jv,tvc->tjc", regressor, vertices) for regressor in regressors]
    return torch.cat(chunks, dim=1)


def _build_contact_specs(
    joints: np.ndarray,
    joint_names: list[str],
    contacts: dict[str, Any] | None,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    if not contacts:
        return []
    threshold = float(settings.get("contact_threshold", 0.65))
    min_frames = int(settings.get("min_segment_frames", 3))
    keys = _selected_foot_keys(str(settings.get("feet") or "toes_and_heels"))
    specs: list[dict[str, Any]] = []
    for key in keys:
        idx = _find_joint(joint_names, CONTACT_JOINTS[key])
        if idx is None:
            continue
        prob = _contact_prob(contacts, key, joints.shape[0])
        for frames in _segments(prob >= threshold, min_frames):
            foot = joints[frames, idx, :]
            finite = np.isfinite(foot).all(axis=1)
            if np.count_nonzero(finite) < min_frames:
                continue
            valid = foot[finite]
            specs.append(
                {
                    "contact_key": key,
                    "joint_index": int(idx),
                    "joint_name": joint_names[idx],
                    "frames": frames.astype(int),
                    "horizontal_anchor": np.nanmedian(valid[:, [0, 2]], axis=0).astype(np.float32),
                    "floor_y": float(np.nanmedian(valid[:, 1])),
                    "mean_probability": float(np.nanmean(prob[frames])),
                }
            )
    return specs


def _contact_metrics(joints: np.ndarray, joint_names: list[str], contacts: dict[str, Any] | None, settings: dict[str, Any], fps: float) -> dict[str, Any]:
    specs = _build_contact_specs(joints, joint_names, contacts, settings)
    speeds = []
    position_stds = []
    vertical_stds = []
    for spec in specs:
        frames = np.asarray(spec["frames"], dtype=int)
        if frames.size < 2:
            continue
        foot = joints[frames, int(spec["joint_index"]), :]
        speed = np.linalg.norm(np.diff(foot[:, [0, 2]], axis=0) * fps, axis=1)
        speeds.extend(speed[np.isfinite(speed)].tolist())
        h_std = np.nanmean(np.nanstd(foot[:, [0, 2]], axis=0))
        y_std = np.nanstd(foot[:, 1])
        if np.isfinite(h_std):
            position_stds.append(float(h_std))
        if np.isfinite(y_std):
            vertical_stds.append(float(y_std))
    return {
        "contact_segment_count": len(specs),
        "mean_contact_horizontal_speed_mps": _safe_float(np.nanmean(speeds)) if speeds else None,
        "mean_contact_horizontal_position_std_m": _safe_float(np.nanmean(position_stds)) if position_stds else None,
        "mean_contact_vertical_std_m": _safe_float(np.nanmean(vertical_stds)) if vertical_stds else None,
    }


def _read_contacts(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _expand_betas(betas: np.ndarray, frames: int) -> np.ndarray:
    arr = np.asarray(betas, dtype=np.float32)
    if arr.ndim == 1:
        return np.repeat(arr.reshape(1, -1), frames, axis=0)
    if arr.ndim == 2 and arr.shape[0] == 1:
        return np.repeat(arr, frames, axis=0)
    if arr.ndim == 2 and arr.shape[0] == frames:
        return arr
    raise ValueError(f"Could not expand SMPL betas with shape {arr.shape} to {frames} frames.")


def _device(settings: dict[str, Any]) -> Any:
    import torch

    requested = str(settings.get("device") or "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _selected_foot_keys(mode: str) -> list[str]:
    if mode == "toes":
        return ["left_toe", "right_toe"]
    if mode == "heels":
        return ["left_heel", "right_heel"]
    return ["left_heel", "left_toe", "right_heel", "right_toe"]


def _find_joint(names: list[str], aliases: tuple[str, ...]) -> int | None:
    lower = {name.lower(): idx for idx, name in enumerate(names)}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def _contact_prob(contacts: dict[str, Any], key: str, frames: int) -> np.ndarray:
    value = contacts.get(key)
    if value is None:
        return np.zeros(frames, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size >= frames:
        return arr[:frames]
    out = np.zeros(frames, dtype=np.float32)
    out[: arr.size] = arr
    return out


def _segments(mask: np.ndarray, min_frames: int) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    start = None
    for idx, active in enumerate(np.asarray(mask, dtype=bool).tolist()):
        if active and start is None:
            start = idx
        if (not active or idx == len(mask) - 1) and start is not None:
            end = idx + 1 if active and idx == len(mask) - 1 else idx
            if end - start >= min_frames:
                segments.append(np.arange(start, end, dtype=int))
            start = None
    return segments


def _public_contact_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "contact_key": spec["contact_key"],
            "joint_name": spec["joint_name"],
            "start_frame": int(np.asarray(spec["frames"])[0]),
            "end_frame": int(np.asarray(spec["frames"])[-1]),
            "frame_count": int(len(spec["frames"])),
            "mean_probability": float(spec["mean_probability"]),
        }
        for spec in specs
    ]


def _status(initial_reconstruction_median_m: float, settings: dict[str, Any]) -> str:
    threshold = float(settings.get("reconstruction_error_warning_m", 0.02))
    return "warning" if initial_reconstruction_median_m > threshold else "ok"


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


if __name__ == "__main__":
    raise SystemExit(main())
