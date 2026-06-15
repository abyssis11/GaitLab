from __future__ import annotations

import pickle
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


def load_smpl_faces(cfg: dict[str, Any], vertex_count: int | None = None) -> dict[str, Any]:
    config = cfg.get("config", cfg)
    repo_root = Path(cfg.get("repo_root") or ".").resolve()
    smpl_cfg = config.get("smpl", {}) or {}

    faces_path = smpl_cfg.get("faces_path")
    if faces_path:
        path = _resolve_path(faces_path, repo_root)
        faces = validate_smpl_faces(np.load(path), vertex_count=vertex_count)
        return {"faces": faces, "source": str(path), "source_type": "faces_path"}

    model_dir = _resolve_path(smpl_cfg.get("model_dir", "models/smpl"), repo_root)
    npy_path = model_dir / "smpl_faces.npy"
    if npy_path.exists():
        faces = validate_smpl_faces(np.load(npy_path), vertex_count=vertex_count)
        return {"faces": faces, "source": str(npy_path), "source_type": "model_dir_npy"}

    gender = str(smpl_cfg.get("gender") or "neutral").lower()
    file_map = smpl_cfg.get("file_map") or {}
    filename = file_map.get(gender) or file_map.get("neutral") or _default_smpl_filename(gender)
    pkl_path = _resolve_path(filename, model_dir)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Could not find SMPL face source. Checked {npy_path} and {pkl_path}.")
    faces = validate_smpl_faces(_load_faces_from_smpl_pickle(pkl_path), vertex_count=vertex_count)
    return {"faces": faces, "source": str(pkl_path), "source_type": "smpl_pickle_f", "gender": gender}


def validate_smpl_faces(faces: np.ndarray, vertex_count: int | None = None) -> np.ndarray:
    arr = np.asarray(faces)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise ValueError(f"SMPL faces must have shape [F, 3], got {arr.shape}.")
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.isfinite(arr)) or not np.allclose(arr, np.round(arr)):
            raise ValueError("SMPL faces must contain integer vertex indices.")
        arr = np.round(arr)
    arr = arr.astype(np.int32, copy=False)
    if int(np.min(arr)) < 0:
        raise ValueError("SMPL faces contain negative vertex indices.")
    if vertex_count is not None and int(np.max(arr)) >= int(vertex_count):
        raise ValueError(f"SMPL faces reference vertex {int(np.max(arr))}, but only {int(vertex_count)} vertices are available.")
    return arr


def _load_faces_from_smpl_pickle(path: Path) -> np.ndarray:
    with _temporary_chumpy_unpickle_shim():
        with path.open("rb") as f:
            data = pickle.load(f, encoding="latin1")
    if not isinstance(data, dict) or "f" not in data:
        raise ValueError(f"SMPL pickle {path} does not contain face key 'f'.")
    return np.asarray(data["f"])


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


@contextmanager
def _temporary_chumpy_unpickle_shim():
    if "chumpy" in sys.modules and "chumpy.ch" in sys.modules:
        yield
        return

    saved_chumpy = sys.modules.get("chumpy")
    saved_ch = sys.modules.get("chumpy.ch")

    class _FakeCh:
        pass

    chumpy = types.ModuleType("chumpy")
    ch = types.ModuleType("chumpy.ch")
    ch.Ch = _FakeCh
    chumpy.ch = ch
    sys.modules["chumpy"] = chumpy
    sys.modules["chumpy.ch"] = ch
    try:
        yield
    finally:
        if saved_chumpy is None:
            sys.modules.pop("chumpy", None)
        else:
            sys.modules["chumpy"] = saved_chumpy
        if saved_ch is None:
            sys.modules.pop("chumpy.ch", None)
        else:
            sys.modules["chumpy.ch"] = saved_ch
