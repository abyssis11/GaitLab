from __future__ import annotations

import copy
import json
import pickle
import subprocess
from pathlib import Path
from typing import Any

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.logging_utils import read_json, write_json


def apply_wham_smpl_root_refinement(
    registry: ArtifactRegistry,
    pose3d: dict[str, Any],
    run_cfg: dict[str, Any],
    refine_cfg: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the Torch/SMPL worker for WHAM root-only refinement.

    The normal monocap-v2 environment intentionally does not import Torch. This
    function validates cheap schema requirements, launches the configured WHAM
    environment as a subprocess, and then loads the worker output.
    """

    refine_cfg = copy.deepcopy(refine_cfg or {})
    out_pose = copy.deepcopy(pose3d)
    validation = _validate_wham_smpl_root_inputs(pose3d)
    if validation is not None:
        out_pose.setdefault("refinement", {})["wham_smpl_root_refinement"] = validation
        return out_pose, validation

    worker_python = Path(str(refine_cfg.get("python") or "/home/denik/miniconda3/envs/monocap-wham/bin/python"))
    worker_script = Path(__file__).resolve().parents[1] / "scripts" / "wham_smpl_root_refine_worker.py"
    if not worker_python.exists():
        return _missing_runtime_passthrough(out_pose, refine_cfg, f"Configured WHAM/SMPL Python does not exist: {worker_python}")
    if not worker_script.exists():
        return _missing_runtime_passthrough(out_pose, refine_cfg, f"WHAM/SMPL worker script does not exist: {worker_script}")

    output_pose = registry.ensure_parent("wham_smpl_root_refine_pose")
    output_report = registry.ensure_parent("wham_smpl_root_refine_report")
    input_pose = registry.get("pose3d_initial")
    contacts = registry.get("contacts")
    run_config = registry.get("run_config")
    if not input_pose.exists():
        raise FileNotFoundError(f"Missing pose3d initial artifact: {input_pose}")
    if not run_config.exists():
        raise FileNotFoundError(f"Missing run config: {run_config}")

    cmd = [
        str(worker_python),
        str(worker_script),
        "--input-pose",
        str(input_pose),
        "--run-config",
        str(run_config),
        "--output-pose",
        str(output_pose),
        "--report",
        str(output_report),
        "--settings-json",
        json.dumps(refine_cfg),
    ]
    if contacts.exists():
        cmd.extend(["--contacts", str(contacts)])

    proc = subprocess.run(cmd, cwd=Path(str(run_cfg.get("repo_root") or Path.cwd())), text=True, capture_output=True)
    if proc.returncode != 0:
        report = _read_worker_report(output_report)
        reason = report.get("error") or report.get("reason") or proc.stderr.strip() or f"Worker exited with code {proc.returncode}"
        if bool(refine_cfg.get("allow_passthrough_on_missing_runtime", False)):
            passthrough_report = {
                "status": "warning",
                "method": "wham_smpl_root_refinement",
                "reason": reason,
                "worker_returncode": proc.returncode,
                "worker_stdout": proc.stdout.strip(),
                "worker_stderr": proc.stderr.strip(),
                "passthrough": True,
                "mocap_used_in_objective": False,
            }
            out_pose.setdefault("refinement", {})["wham_smpl_root_refinement"] = passthrough_report
            write_json(output_report, passthrough_report)
            return out_pose, passthrough_report
        raise RuntimeError(f"WHAM SMPL root refinement worker failed: {reason}")

    if not output_pose.exists():
        raise RuntimeError(f"WHAM SMPL root refinement worker did not write output pose: {output_pose}")
    with output_pose.open("rb") as f:
        refined_pose = pickle.load(f)
    report = _read_worker_report(output_report)
    report.setdefault("status", "ok")
    report.setdefault("method", "wham_smpl_root_refinement")
    report.setdefault("mocap_used_in_objective", False)
    refined_pose.setdefault("refinement", {})["wham_smpl_root_refinement"] = report
    return refined_pose, report


def _validate_wham_smpl_root_inputs(pose3d: dict[str, Any]) -> dict[str, Any] | None:
    representation = str(pose3d.get("representation") or "")
    backend = str(pose3d.get("backend") or "")
    if backend != "wham":
        return {
            "status": "skipped",
            "method": "wham_smpl_root_refinement",
            "reason": f"WHAM SMPL root refinement requires backend='wham', got {backend!r}.",
            "mocap_used_in_objective": False,
        }
    if representation not in {"hybrid", "smpl"}:
        return {
            "status": "skipped",
            "method": "wham_smpl_root_refinement",
            "reason": f"WHAM SMPL root refinement requires hybrid/SMPL representation, got {representation!r}.",
            "mocap_used_in_objective": False,
        }
    smpl = pose3d.get("smpl")
    if not isinstance(smpl, dict):
        return {
            "status": "skipped",
            "method": "wham_smpl_root_refinement",
            "reason": "WHAM SMPL root refinement requires an SMPL payload.",
            "mocap_used_in_objective": False,
        }
    required = ["vertices", "betas", "body_pose", "global_orient"]
    missing = [key for key in required if smpl.get(key) is None]
    if missing:
        return {
            "status": "skipped",
            "method": "wham_smpl_root_refinement",
            "reason": f"WHAM SMPL root refinement missing SMPL fields: {', '.join(missing)}.",
            "mocap_used_in_objective": False,
        }
    return None


def _missing_runtime_passthrough(pose: dict[str, Any], refine_cfg: dict[str, Any], reason: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not bool(refine_cfg.get("allow_passthrough_on_missing_runtime", False)):
        raise RuntimeError(reason)
    report = {
        "status": "warning",
        "method": "wham_smpl_root_refinement",
        "reason": reason,
        "passthrough": True,
        "mocap_used_in_objective": False,
    }
    pose.setdefault("refinement", {})["wham_smpl_root_refinement"] = report
    return pose, report


def _read_worker_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = read_json(path)
    except Exception as exc:
        return {"status": "warning", "error": f"Could not read worker report {path}: {exc}"}
    return data if isinstance(data, dict) else {"status": "warning", "report": data}
