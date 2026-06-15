from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.logging_utils import write_json
from monocap_v2.core.opensim_io import write_trc
from monocap_v2.core.stage_utils import cached, stage_result


STAGE = "stage_09_export_trc"


def run(run_dir: Path, cfg: dict, force: bool = False) -> dict:
    registry = ArtifactRegistry(run_dir)
    out_path = registry.ensure_parent("smpl_markers_trc")
    qc_path = registry.ensure_parent("smpl_markers_qc")
    if cached(out_path, force):
        return stage_result(STAGE, "cached", output=str(out_path))
    markers_path = registry.get("virtual_markers")
    if not markers_path.exists():
        result = stage_result(STAGE, "skipped", reason="TRC export requires virtual_markers.pkl.")
        write_json(qc_path, result)
        return result
    with markers_path.open("rb") as f:
        payload = pickle.load(f)
    marker_names = [str(name) for name in payload.get("marker_names", [])]
    markers_m = np.asarray(payload.get("markers_m"), dtype=float)
    if markers_m.ndim != 3 or markers_m.shape[-1] != 3 or not marker_names:
        result = stage_result(STAGE, "failed", reason="Invalid virtual marker payload.")
        write_json(qc_path, result)
        return result
    fps = float(payload.get("fps") or 30.0)
    write_trc(out_path, marker_names, markers_m, fps=fps)
    qc = {
        "stage": STAGE,
        "status": "ok",
        "output": str(out_path),
        "source": str(markers_path),
        "marker_set": payload.get("marker_set"),
        "debug": bool(payload.get("debug", False)),
        "debug_warning": "Debug SMPL marker TRC; not final anatomical OpenSim markers." if payload.get("debug") else None,
        "units_internal": "m",
        "units_trc": "mm",
        "fps": fps,
        "frames": int(markers_m.shape[0]),
        "markers": int(markers_m.shape[1]),
        "marker_names": marker_names,
        "coordinate_space": payload.get("coordinate_space"),
        "time_origin": "wham_marker_sequence_zero",
        "raw_frame_start": int(np.asarray(payload.get("raw_frame_ids")).reshape(-1)[0]) if payload.get("raw_frame_ids") is not None else None,
        "raw_frame_end": int(np.asarray(payload.get("raw_frame_ids")).reshape(-1)[-1]) if payload.get("raw_frame_ids") is not None else None,
        "time_window": payload.get("time_window"),
    }
    write_json(qc_path, qc)
    result = stage_result(
        STAGE,
        "ok",
        output=str(out_path),
        marker_set=payload.get("marker_set"),
        markers=len(marker_names),
        units_trc="mm",
        debug=bool(payload.get("debug", False)),
    )
    return result
