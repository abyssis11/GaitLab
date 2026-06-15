from __future__ import annotations

import pickle
from pathlib import Path

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.logging_utils import write_json
from monocap_v2.core.schemas import has_smpl_vertices
from monocap_v2.core.smpl_markers import apply_marker_time_window, extract_virtual_markers, load_marker_set, marker_qc
from monocap_v2.core.stage_utils import cached, stage_result


STAGE = "stage_08_extract_virtual_markers"


def run(run_dir: Path, cfg: dict, force: bool = False) -> dict:
    registry = ArtifactRegistry(run_dir)
    out_path = registry.ensure_parent("virtual_markers")
    qc_path = registry.ensure_parent("virtual_markers_qc")
    if cached(out_path, force):
        return stage_result(STAGE, "cached", output=str(out_path))
    with registry.get("pose3d_refined").open("rb") as f:
        pose3d = pickle.load(f)
    if not has_smpl_vertices(pose3d):
        result = stage_result(STAGE, "skipped", reason="No SMPL vertices available in pose3d artifact.")
        write_json(qc_path, result)
        return result
    marker_map_path = Path(__file__).resolve().parents[1] / "configs" / "marker_map_smpl_to_opensim.yaml"
    marker_cfg = cfg.get("config", {}).get("markers", {})
    marker_set = load_marker_set(marker_map_path, marker_cfg.get("marker_set"))
    payload = extract_virtual_markers(pose3d, marker_set)
    payload, window_report = apply_marker_time_window(payload, pose3d, cfg)
    with out_path.open("wb") as f:
        pickle.dump(payload, f)
    qc = marker_qc(payload)
    qc["stage"] = STAGE
    qc["status"] = qc.get("status", "ok")
    qc["output"] = str(out_path)
    write_json(qc_path, qc)
    result = stage_result(
        STAGE,
        qc["status"],
        output=str(out_path),
        marker_set=payload["marker_set"],
        markers=len(payload["marker_names"]),
        frames=int(payload["markers_m"].shape[0]),
        debug=bool(payload.get("debug", False)),
        time_window=window_report,
        warnings=qc.get("warnings", []),
    )
    return result
