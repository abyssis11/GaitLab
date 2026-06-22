#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_ENV_PATH = Path("/home/denik/miniconda3/envs/monocap-sam3d-body")
REQUIRED_ASSETS = {
    "model_config": "sam-3d-body-dinov3/model_config.yaml",
    "checkpoint": "sam-3d-body-dinov3/model.ckpt",
    "mhr_model": "sam-3d-body-dinov3/assets/mhr_model.pt",
}
OPTIONAL_ASSETS = {
    "vith_model_config": "sam-3d-body-vith/model_config.yaml",
    "vith_checkpoint": "sam-3d-body-vith/model.ckpt",
    "vith_mhr_model": "sam-3d-body-vith/assets/mhr_model.pt",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Check monocap_v2 SAM3D Body prerequisite status.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--env-path", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--sam3d-repo", type=Path, default=repo_root / "external" / "sam-3d-body")
    parser.add_argument("--checkpoint-dir", type=Path, default=repo_root / "models" / "sam3d_body")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
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
    checks = {
        "env": {
            "path": str(args.env_path),
            "exists": args.env_path.exists(),
            "python": str(env_python),
            "python_exists": env_python.exists(),
        },
        "sam3d_repo": check_repo(args.sam3d_repo),
        "assets": check_assets(args.checkpoint_dir),
        "git_ignore": check_git_ignore(args.repo_root),
    }
    checks["python_imports"] = check_python_imports(env_python, args.sam3d_repo) if env_python.exists() else {"status": "missing_env_python"}
    required_ok = [
        checks["env"]["python_exists"],
        checks["sam3d_repo"]["status"] == "ok",
        checks["assets"]["status"] == "ok",
        checks["git_ignore"]["status"] == "ok",
        checks["python_imports"]["status"] == "ok",
    ]
    return {"status": "ok" if all(required_ok) else "missing", "checks": checks}


def check_repo(repo: Path) -> dict[str, Any]:
    expected = {
        "repo": repo.exists(),
        "demo": (repo / "demo.py").exists(),
        "package": (repo / "sam_3d_body").exists(),
        "estimator": (repo / "sam_3d_body" / "sam_3d_body_estimator.py").exists(),
    }
    return {"status": "ok" if all(expected.values()) else "missing", "path": str(repo), **expected}


def check_assets(root: Path) -> dict[str, Any]:
    required = {name: _file_status(root / rel) for name, rel in REQUIRED_ASSETS.items()}
    optional = {name: _file_status(root / rel) for name, rel in OPTIONAL_ASSETS.items()}
    required_ok = all(item["exists"] and item.get("valid", True) for item in required.values())
    return {
        "status": "ok" if required_ok else "missing",
        "root": str(root),
        "required": required,
        "optional": optional,
        "default_checkpoint": "sam-3d-body-dinov3",
        "fallback_checkpoint": "sam-3d-body-vith",
    }


def check_python_imports(env_python: Path, sam3d_repo: Path) -> dict[str, Any]:
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(sam3d_repo)!r})\n"
        "out = {}\n"
        "for name in ['torch', 'cv2', 'detectron2', 'moge']:\n"
        "    try:\n"
        "        module = __import__(name)\n"
        "        item = {'ok': True, 'version': getattr(module, '__version__', None)}\n"
        "        if name == 'torch':\n"
        "            item['cuda_available'] = bool(module.cuda.is_available())\n"
        "        out[name] = item\n"
        "    except Exception as exc:\n"
        "        out[name] = {'ok': False, 'error': str(exc)}\n"
        "try:\n"
        "    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body\n"
        "    out['sam_3d_body'] = {'ok': True, 'estimator': SAM3DBodyEstimator.__name__}\n"
        "except Exception as exc:\n"
        "    out['sam_3d_body'] = {'ok': False, 'error': str(exc)}\n"
        "print(json.dumps(out))\n"
    )
    completed = subprocess.run([str(env_python), "-c", code], text=True, capture_output=True)
    if completed.returncode != 0:
        return {"status": "failed", "returncode": completed.returncode, "stderr": completed.stderr.strip()}
    try:
        imports = _loads_last_json_object(completed.stdout)
    except json.JSONDecodeError:
        return {"status": "failed", "stdout": completed.stdout, "stderr": completed.stderr}
    ok = all(imports.get(name, {}).get("ok") for name in ["torch", "cv2", "detectron2", "moge", "sam_3d_body"])
    ok = ok and bool((imports.get("torch") or {}).get("cuda_available"))
    return {"status": "ok" if ok else "missing", "imports": imports, "stderr": completed.stderr.strip()}


def check_git_ignore(repo_root: Path) -> dict[str, Any]:
    paths = ["external/sam-3d-body/README.md", "models/sam3d_body/sam-3d-body-dinov3/model.ckpt"]
    results = {}
    for rel in paths:
        completed = subprocess.run(["git", "check-ignore", "-q", rel], cwd=repo_root)
        results[rel] = completed.returncode == 0
    return {"status": "ok" if all(results.values()) else "missing", "paths": results}


def _file_status(path: Path) -> dict[str, Any]:
    status = {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None}
    if path.name == "model_config.yaml" and path.exists():
        text = path.read_text(errors="replace")
        invalid_markers = ["Access to model", "restricted", "Please log in"]
        status["valid"] = not any(marker in text for marker in invalid_markers)
        status["preview"] = text[:120].replace("\n", "\\n")
    return status


def _loads_last_json_object(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return json.loads(stripped)
    raise json.JSONDecodeError("No JSON object found in stdout", stdout, 0)


def print_human(report: dict[str, Any]) -> None:
    print(f"SAM3D Body prereq status: {report['status']}")
    for name, value in report["checks"].items():
        print(f"\n[{name}]")
        print(json.dumps(value, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
