#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_ENV_PATH = Path("/home/denik/miniconda3/envs/monocap-wham")
SMPL_CANDIDATES = {
    "male": ["SMPL_MALE.pkl", "basicmodel_m_lbs_10_207_0_v1.1.0.pkl"],
    "female": ["SMPL_FEMALE.pkl", "basicmodel_f_lbs_10_207_0_v1.1.0.pkl"],
    "neutral": ["SMPL_NEUTRAL.pkl", "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"],
}
WHAM_REQUIRED_ASSETS = {
    "wham_checkpoint": "checkpoints/wham_vit_bedlam_w_3dpw.pth.tar",
    "hmr2a_checkpoint": "checkpoints/hmr2a.ckpt",
    "yolo_checkpoint": "checkpoints/yolov8x.pt",
    "vitpose_checkpoint": "checkpoints/vitpose-h-multi-coco.pth",
    "smpl_mean_params": "dataset/body_models/smpl_mean_params.npz",
    "smplx2smpl": "dataset/body_models/smplx2smpl.pkl",
    "wham_joint_regressor": "dataset/body_models/J_regressor_wham.npy",
    "coco_joint_regressor": "dataset/body_models/J_regressor_coco.npy",
    "h36m_joint_regressor": "dataset/body_models/J_regressor_h36m.npy",
    "feet_joint_regressor": "dataset/body_models/J_regressor_feet.npy",
}
WHAM_OPTIONAL_ASSETS = {
    "dpvo_checkpoint": "checkpoints/dpvo.pth",
    "wham_3dpw_checkpoint": "checkpoints/wham_vit_w_3dpw.pth.tar",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Check monocap_v2 WHAM/SMPL prerequisite status.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--env-path", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--wham-repo", type=Path, default=repo_root / "external" / "WHAM")
    parser.add_argument("--smpl-dir", type=Path, default=repo_root / "models" / "smpl")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero unless all required checks pass.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_human(report)
    return 0 if report["status"] == "ok" or not args.strict else 1


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    env_python = args.env_path / "bin" / "python"
    checks: dict[str, Any] = {
        "env": {
            "path": str(args.env_path),
            "exists": args.env_path.exists(),
            "python": str(env_python),
            "python_exists": env_python.exists(),
        },
        "smpl": check_smpl(args.smpl_dir),
        "wham_repo": check_wham_repo(args.wham_repo),
        "wham_assets": check_wham_assets(args.wham_repo),
        "git_ignore": check_git_ignore(args.repo_root),
    }
    checks["python_imports"] = check_python_imports(env_python, args.wham_repo) if env_python.exists() else {"status": "missing_env_python"}
    checks["dpvo_native"] = check_dpvo_native(env_python) if env_python.exists() else {"status": "missing_env_python"}
    required_ok = [
        checks["env"]["python_exists"],
        checks["smpl"]["status"] == "ok",
        checks["wham_repo"]["status"] == "ok",
        checks["wham_assets"]["status"] in {"ok", "partial"},
        checks["git_ignore"]["status"] == "ok",
        checks["python_imports"]["status"] == "ok",
        checks["dpvo_native"]["status"] == "ok",
    ]
    core_ok = all(required_ok[:-1])
    if all(required_ok):
        status = "ok"
    elif core_ok:
        status = "partial"
    else:
        status = "missing"
    return {"status": status, "checks": checks}


def check_smpl(smpl_dir: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    missing = []
    for gender, names in SMPL_CANDIDATES.items():
        found = next((smpl_dir / name for name in names if (smpl_dir / name).exists()), None)
        files[gender] = str(found) if found else None
        if found is None:
            missing.append(gender)
    return {"status": "ok" if not missing else "missing", "dir": str(smpl_dir), "files": files, "missing": missing}


def check_wham_repo(wham_repo: Path) -> dict[str, Any]:
    expected = {
        "repo": wham_repo.exists(),
        "wham_api": (wham_repo / "wham_api.py").exists(),
        "vitpose": (wham_repo / "third-party" / "ViTPose").exists(),
        "dpvo": (wham_repo / "third-party" / "DPVO").exists(),
    }
    return {"status": "ok" if all(expected.values()) else "missing", "path": str(wham_repo), **expected}


def check_wham_assets(wham_repo: Path) -> dict[str, Any]:
    required = {name: _file_status(wham_repo / rel) for name, rel in WHAM_REQUIRED_ASSETS.items()}
    optional = {name: _file_status(wham_repo / rel) for name, rel in WHAM_OPTIONAL_ASSETS.items()}
    required_ok = all(item["exists"] for item in required.values())
    optional_ok = all(item["exists"] for item in optional.values())
    if required_ok and optional_ok:
        status = "ok"
    elif required_ok:
        status = "partial"
    else:
        status = "missing"
    return {"status": status, "required": required, "optional": optional}


def _file_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def check_python_imports(env_python: Path, wham_repo: Path) -> dict[str, Any]:
    code = (
        "import json, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(wham_repo)!r})\n"
        "out = {}\n"
        "try:\n"
        "    import torch\n"
        "    out['torch'] = {'ok': True, 'version': torch.__version__, 'cuda_available': bool(torch.cuda.is_available())}\n"
        "except Exception as exc:\n"
        "    out['torch'] = {'ok': False, 'error': str(exc)}\n"
        "try:\n"
        "    import smplx\n"
        "    out['smplx'] = {'ok': True, 'version': getattr(smplx, '__version__', None)}\n"
        "except Exception as exc:\n"
        "    out['smplx'] = {'ok': False, 'error': str(exc)}\n"
        "try:\n"
        "    from wham_api import WHAM_API\n"
        "    out['wham_api'] = {'ok': True, 'class': WHAM_API.__name__}\n"
        "except Exception as exc:\n"
        "    out['wham_api'] = {'ok': False, 'error': str(exc)}\n"
        "print(json.dumps(out))\n"
    )
    completed = subprocess.run([str(env_python), "-c", code], text=True, capture_output=True)
    if completed.returncode != 0:
        return {"status": "failed", "returncode": completed.returncode, "stderr": completed.stderr.strip()}
    try:
        imports = _loads_last_json_object(completed.stdout)
    except json.JSONDecodeError:
        return {"status": "failed", "stdout": completed.stdout, "stderr": completed.stderr}
    ok = all(imports.get(name, {}).get("ok") for name in ["torch", "smplx", "wham_api"])
    return {"status": "ok" if ok else "missing", "imports": imports, "stderr": completed.stderr.strip()}


def check_dpvo_native(env_python: Path) -> dict[str, Any]:
    code = (
        "import json\n"
        "out = {}\n"
        "for name in ['dpvo', 'cuda_corr', 'cuda_ba', 'lietorch_backends']:\n"
        "    try:\n"
        "        __import__(name)\n"
        "        out[name] = {'ok': True}\n"
        "    except Exception as exc:\n"
        "        out[name] = {'ok': False, 'error': str(exc)}\n"
        "print(json.dumps(out))\n"
    )
    completed = subprocess.run([str(env_python), "-c", code], text=True, capture_output=True)
    if completed.returncode != 0:
        return {"status": "failed", "returncode": completed.returncode, "stderr": completed.stderr.strip()}
    try:
        imports = _loads_last_json_object(completed.stdout)
    except json.JSONDecodeError:
        return {"status": "failed", "stdout": completed.stdout, "stderr": completed.stderr}
    ok = all(imports.get(name, {}).get("ok") for name in ["dpvo", "cuda_corr", "cuda_ba", "lietorch_backends"])
    reason = None if ok else "DPVO native CUDA extensions are not installed; WHAM can import but will fall back to local-coordinate estimation."
    return {"status": "ok" if ok else "missing", "imports": imports, "reason": reason}


def _loads_last_json_object(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return json.loads(stripped)
    raise json.JSONDecodeError("No JSON object found in stdout", stdout, 0)


def check_git_ignore(repo_root: Path) -> dict[str, Any]:
    paths = ["external/WHAM/README.md", "models/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"]
    results = {}
    for rel in paths:
        completed = subprocess.run(["git", "check-ignore", "-q", rel], cwd=repo_root)
        results[rel] = completed.returncode == 0
    return {"status": "ok" if all(results.values()) else "missing", "paths": results}


def print_human(report: dict[str, Any]) -> None:
    print(f"WHAM prereq status: {report['status']}")
    for name, value in report["checks"].items():
        print(f"\n[{name}]")
        print(json.dumps(value, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
