from __future__ import annotations

import numpy as np


def finite_ratio(array) -> float:
    arr = np.asarray(array)
    if arr.size == 0:
        return 0.0
    return float(np.isfinite(arr).mean())


def mean_speed(joints_3d, fps: float) -> float:
    arr = np.asarray(joints_3d, dtype=float)
    if arr.shape[0] < 2:
        return 0.0
    vel = np.diff(arr, axis=0) * float(fps)
    return float(np.nanmean(np.linalg.norm(vel, axis=-1)))

