from __future__ import annotations

from pathlib import Path

import numpy as np


def write_trc(path: Path, marker_names: list[str], frames_xyz_m, fps: float) -> None:
    """Write TRC in millimeters from internal meter coordinates."""
    data_mm = np.asarray(frames_xyz_m, dtype=float) * 1000.0
    path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = data_mm.shape[0]
    with path.open("w", encoding="utf-8") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{path.name}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps:.2f}\t{fps:.2f}\t{num_frames}\t{len(marker_names)}\tmm\t{fps:.2f}\t1\t{num_frames}\n")
        f.write("Frame#\tTime\t" + "\t\t\t".join(marker_names) + "\t\t\t\n")
        cols = []
        for i in range(1, len(marker_names) + 1):
            cols.extend([f"X{i}", f"Y{i}", f"Z{i}"])
        f.write("\t" + "\t".join(cols) + "\n\n")
        for frame_idx, frame in enumerate(data_mm):
            row = [str(frame_idx + 1), f"{frame_idx / float(fps):.8f}"]
            for xyz in frame:
                row.extend(f"{v:.5f}" if np.isfinite(v) else "" for v in xyz)
            f.write("\t".join(row) + "\n")

