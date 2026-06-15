from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from monocap_v2.core.smpl_mesh import _temporary_chumpy_unpickle_shim


SMPL_24_JOINT_NAMES = [
    "pelv",
    "lhip",
    "rhip",
    "spi1",
    "lkne",
    "rkne",
    "spi2",
    "lank",
    "rank",
    "spi3",
    "left_foot",
    "right_foot",
    "neck",
    "lcla",
    "rcla",
    "head",
    "lsho",
    "rsho",
    "lelb",
    "relb",
    "lwri",
    "rwri",
    "lhan",
    "rhan",
]


SMPL_FOOT_VERTEX_NAMES = ["left_big_toe", "left_heel", "right_big_toe", "right_heel"]


def regress_smpl_24_joints(vertices: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    regressor_info = load_smpl_joint_regressor(cfg)
    regressor = regressor_info["regressor"]
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"SMPL vertices must have shape [T, V, 3], got {vertices.shape}.")
    if regressor.shape[1] != vertices.shape[1]:
        raise ValueError(
            f"SMPL joint regressor expects {regressor.shape[1]} vertices, "
            f"but artifact has {vertices.shape[1]} vertices."
        )
    joints = np.einsum("jv,tvc->tjc", regressor, vertices, optimize=True)
    return {
        "joints": joints.astype(np.float32),
        "joint_names": list(SMPL_24_JOINT_NAMES),
        "source": regressor_info["source"],
        "source_type": regressor_info["source_type"],
        "gender": regressor_info.get("gender"),
    }


def load_smpl_joint_regressor(cfg: dict[str, Any]) -> dict[str, Any]:
    path, gender = resolve_smpl_model_path(cfg)
    with _temporary_chumpy_unpickle_shim():
        with path.open("rb") as f:
            data = pickle.load(f, encoding="latin1")
    if not isinstance(data, dict) or "J_regressor" not in data:
        raise ValueError(f"SMPL pickle {path} does not contain J_regressor.")
    regressor = data["J_regressor"]
    if hasattr(regressor, "toarray"):
        regressor = regressor.toarray()
    regressor = np.asarray(regressor, dtype=np.float32)
    if regressor.ndim != 2 or regressor.shape[0] != len(SMPL_24_JOINT_NAMES):
        raise ValueError(f"Invalid SMPL J_regressor shape in {path}: {regressor.shape}.")
    return {
        "regressor": regressor,
        "source": str(path),
        "source_type": "smpl_pickle_J_regressor",
        "gender": gender,
    }


def resolve_smpl_model_path(cfg: dict[str, Any]) -> tuple[Path, str]:
    config = cfg.get("config", cfg)
    repo_root = Path(cfg.get("repo_root") or ".").resolve()
    smpl_cfg = config.get("smpl", {}) or {}
    model_dir = _resolve_path(smpl_cfg.get("model_dir", "models/smpl"), repo_root)
    gender = str(smpl_cfg.get("gender") or "neutral").lower()
    file_map = smpl_cfg.get("file_map") or {}
    filename = file_map.get(gender) or file_map.get("neutral") or _default_smpl_filename(gender)
    path = _resolve_path(filename, model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Could not find SMPL model pickle: {path}")
    return path, gender


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base / path


def _default_smpl_filename(gender: str) -> str:
    mapping = {
        "male": "basicmodel_m_lbs_10_207_0_v1.1.0.pkl",
        "female": "basicmodel_f_lbs_10_207_0_v1.1.0.pkl",
        "neutral": "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl",
    }
    return mapping.get(gender, mapping["neutral"])
