# ptest.py
# Read selected keypoints from a C3D, optionally filter & downsample, and save to TRC.
'''
python src/utils/gpjatk_mocap_to_trc.py -m ./manifests/GPJATK/subject1.yaml -p ./config/paths.yaml --trial walking1 --resampling_rate 50
'''

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"No analog data found in file\.",
    category=UserWarning,
    module="c3d.c3d",   # <- module name, not your filename
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas")

import os
import c3d
import numpy as np
import pandas as pd
from IO.load_manifest import load_manifest
import argparse
from pathlib import Path


# ---------- Logging ----------
def log_step(msg): print(f"[STEP] {msg}")
def log_info(msg): print(f"[INFO] {msg}")
def log_warn(msg): print(f"[WARN] {msg}")
def log_err (msg): print(f"[ERROR] {msg}")
def log_done(msg): print(f"[DONE] {msg}")

# ---------- IO helpers ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ================== USER SETTINGS ==================
C3D_PATH = None
OUTPUT_TRC_PATH = None
UNITS = "mm"                 # TRC "Units" field
CONVERT_M_TO_MM = False      # Set True if your C3D is in meters and you want TRC in mm

# Downsample & filtering
TARGET_RATE_HZ = None         # e.g., 100; set None to keep original rate
LOWPASS_CUTOFF_HZ = 6        # e.g., 6 Hz for gait kinematics; set None to skip
FILTER_ORDER = 4             # Used if SciPy is available; otherwise first-order IIR
# ====================================================

# Your requested keypoints
keypoints = [
    'RSHO', 'LSHO', 'RASI', 'LASI', 'RKNE', 'LKNE', 'RANK', 'LANK',
    'RHEE', 'LHEE', 'RTOE', 'LTOE', 'RELB', 'LELB', 'RWRB', 'LWRB'
]

def _norm(s: str) -> str:
    return s.strip().upper()

def _first_order_lowpass(x, fs, fc):
    """Simple first-order IIR low-pass (exponential smoothing).
    Zero-phase is better (filtfilt), but this works without SciPy."""
    if fc is None or fc <= 0:
        return x
    dt = 1.0 / fs
    rc = 1.0 / (2.0 * np.pi * fc)
    alpha = dt / (rc + dt)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i-1] + alpha * (x[i] - y[i-1])
    return y

def lowpass_filter_df(df, fs, fc, order=4):
    """Low-pass each numeric column. Uses SciPy filtfilt if available, else first-order IIR."""
    if fc is None or fc <= 0:
        return df.copy()
    try:
        from scipy.signal import butter, filtfilt
        wn = min(0.9999, fc / (fs * 0.5))
        b, a = butter(order, wn, btype="lowpass")
        arr = df.to_numpy(dtype=float)
        arr = filtfilt(b, a, arr, axis=0)
        out = pd.DataFrame(arr, index=df.index, columns=df.columns)
        return out
    except Exception:
        # Fallback: apply first-order filter forward-only per column
        out = df.copy()
        for col in out.columns:
            out[col] = _first_order_lowpass(out[col].to_numpy(dtype=float), fs, fc)
        return out

def resample_time_index(orig_len, orig_rate, target_rate):
    """Build new time vector for resampling (inclusive of t=0; exclusive of >last time)."""
    duration = (orig_len - 1) / float(orig_rate)
    n_new = int(np.floor(duration * target_rate)) + 1
    t_new = np.arange(n_new, dtype=float) / float(target_rate)
    return t_new

def interpolate_df_to_times(df, orig_rate, t_new):
    """Column-wise linear interpolation onto new time stamps."""
    # Source times; assume df index starts at 0 s (it does below)
    t_old = np.arange(len(df), dtype=float) / float(orig_rate)
    out = {}
    for col in df.columns:
        s = pd.Series(df[col].to_numpy(dtype=float))
        # Handle NaNs before interpolation
        s = s.interpolate("linear", limit_direction="both")
        out[col] = np.interp(t_new, t_old, s.to_numpy())
    res = pd.DataFrame(out, index=t_new, columns=df.columns)
    res.index.name = "time"
    return res

def write_trc(df_xyz, filepath, data_rate, units="mm", start_frame=1, filename_for_header=None):
    """
    Write a TRC file from DataFrame with MultiIndex columns: (marker, axis) axis in ['X','Y','Z'].
    df index should be time in seconds starting at 0.
    """
    assert isinstance(df_xyz.columns, pd.MultiIndex) and df_xyz.columns.nlevels == 2
    markers = list(df_xyz.columns.levels[0])
    # Preserve original order as they appear left-to-right in df
    seen, ordered_markers = set(), []
    for m, ax in df_xyz.columns:
        if m not in seen:
            seen.add(m)
            ordered_markers.append(m)

    num_frames = len(df_xyz)
    num_markers = len(ordered_markers)
    camera_rate = data_rate
    orig_data_rate = getattr(df_xyz, "_orig_rate", data_rate)
    orig_start_frame = start_frame
    orig_num_frames = getattr(df_xyz, "_orig_num_frames", num_frames)
    fname = filename_for_header or os.path.basename(filepath)

    with open(filepath, "w", newline="") as f:
        # Header block
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{fname}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{data_rate:.2f}\t{camera_rate:.2f}\t{num_frames}\t{num_markers}\t{units}\t{orig_data_rate:.2f}\t{orig_start_frame}\t{orig_num_frames}\n")

        # Column headers (two rows)
        # Row 1: marker names spanning three columns each
        hdr1 = ["Frame#", "Time"]
        for m in ordered_markers:
            hdr1 += [m, "", ""]
        f.write("\t".join(hdr1) + "\n")

        # Row 2: per-axis labels
        hdr2 = ["", ""]
        for i in range(num_markers):
            hdr2 += ["X{}".format(i+1), "Y{}".format(i+1), "Z{}".format(i+1)]
        f.write("\t".join(hdr2) + "\n")

        # Data rows
        # Ensure columns are in the correct order (marker-major, X Y Z)
        cols = []
        for m in ordered_markers:
            cols += [(m, "X"), (m, "Y"), (m, "Z")]
        data = df_xyz[cols].to_numpy(dtype=float)

        # Write lines
        for i in range(num_frames):
            frame_num = start_frame + i
            t = df_xyz.index[i]
            row = [str(frame_num), f"{t:.5f}"] + ["{:.5f}".format(v) for v in data[i]]
            f.write("\t".join(row) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Prepare OCAP20 from RTMW3D (metric) and run Marker-Enhancer.")
    ap.add_argument("-m", "--manifest", required=True)
    ap.add_argument("-p", "--paths", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--resampling-rate", default=100)
    ap.add_argument("--lp-filter", default=6)
    args = ap.parse_args()   

    log_step("Loading and resolving manifest")
    manifest = load_manifest(args.manifest, args.paths)

    TARGET_RATE_HZ = int(args.resampling_rate)
    LOWPASS_CUTOFF_HZ = int(args.lp_filter)

    # Find trial
    log_step(f"Finding trial '{args.trial}'")
    trial = None
    for subset, trials in manifest.get("trials", {}).items():
        for t in trials:
            if t.get("id") == args.trial:
                trial = t
                break
        if trial: break
    if trial is None:
        raise SystemExit(f"[ERROR] Trial '{args.trial}' not found in manifest.")

    # Paths
    base = manifest.get('output_dir')
    C3D_PATH = trial['mocap_c3d']
    OUTPUT_TRC_PATH = trial['mocap_trc']
    ensure_dir(Path(base))
    log_info(f"Trial root : {base}")



    # ---------- Load & filter selection from C3D ----------
    with open(C3D_PATH, "rb") as h:
        r = c3d.Reader(h)

        # Labels from file
        file_labels = [s.strip() for s in r.point_labels[:r.point_used]]
        norm_to_file = {}
        for i, lab in enumerate(file_labels):
            norm_to_file.setdefault(_norm(lab), (i, lab))  # keep first if duplicates

        wanted, wanted_idx, missing = [], [], []
        for kp in keypoints:
            nk = _norm(kp)
            if nk in norm_to_file:
                idx, real = norm_to_file[nk]
                wanted.append(real)
                wanted_idx.append(idx)
            else:
                missing.append(kp)

        print(f"Found {len(wanted)} / {len(keypoints)} requested markers.")
        if missing:
            print("Missing:", missing)
        if not wanted_idx:
            raise SystemExit("None of the requested keypoints were found.")

        # Gather frames (X,Y,Z) only for the desired markers
        frames_xyz = []
        for frame_idx, points, _ in r.read_frames():
            frames_xyz.append(points[wanted_idx, :3])  # (K, 3)

        data = np.stack(frames_xyz, axis=0)  # (F, K, 3)
        F, K, _ = data.shape

        # Optional unit conversion
        if CONVERT_M_TO_MM:
            data = data * 1000.0

        # DataFrame with (marker, axis)
        cols = pd.MultiIndex.from_product([wanted, ["X", "Y", "Z"]],
                                          names=["marker", "axis"])
        df = pd.DataFrame(data.reshape(F, K * 3), columns=cols)

        # Original rate/time index
        orig_rate = getattr(r, "point_rate", None)
        if not orig_rate or orig_rate <= 0:
            raise SystemExit("Could not read original sampling rate from C3D (point_rate).")
        df.index = np.arange(F) / float(orig_rate)
        df.index.name = "time"
        df._orig_rate = orig_rate          # stash for TRC header
        df._orig_num_frames = F

    # ---------- Filtering (anti-alias) ----------
    # If downsampling, ensure cutoff <= Nyquist of target (with safety margin)
    if TARGET_RATE_HZ and TARGET_RATE_HZ > 0 and TARGET_RATE_HZ < orig_rate:
        fc = LOWPASS_CUTOFF_HZ if LOWPASS_CUTOFF_HZ else 0.45 * TARGET_RATE_HZ
        fc = min(fc, 0.49 * orig_rate)
        df_filtered = lowpass_filter_df(df, orig_rate, fc, order=FILTER_ORDER)
        effective_rate = TARGET_RATE_HZ
    else:
        # If only filtering without downsampling, use provided cutoff
        if LOWPASS_CUTOFF_HZ and LOWPASS_CUTOFF_HZ > 0:
            df_filtered = lowpass_filter_df(df, orig_rate, LOWPASS_CUTOFF_HZ, order=FILTER_ORDER)
        else:
            df_filtered = df.copy()
        effective_rate = orig_rate

    # ---------- Resampling ----------
    if TARGET_RATE_HZ and TARGET_RATE_HZ > 0 and TARGET_RATE_HZ != orig_rate:
        if TARGET_RATE_HZ > orig_rate:
            log_info("Resampling")
            # No upsampling here; if you really want it, remove this guard.
            print(f"TARGET_RATE_HZ ({TARGET_RATE_HZ}) > original rate ({orig_rate}); keeping original rate.")
            target_rate = orig_rate
            t_new = np.arange(len(df_filtered)) / float(orig_rate)
            df_rs = df_filtered.copy()
        else:
            target_rate = TARGET_RATE_HZ
            t_new = resample_time_index(len(df_filtered), orig_rate, target_rate)
            df_rs = interpolate_df_to_times(df_filtered, orig_rate, t_new)
    else:
        target_rate = orig_rate
        df_rs = df_filtered.copy()

    # ---------- Write TRC ----------
    write_trc(df_rs, OUTPUT_TRC_PATH, data_rate=target_rate, units=UNITS, filename_for_header=os.path.basename(OUTPUT_TRC_PATH))
    print(f"\nSaved TRC to: {OUTPUT_TRC_PATH}")

if __name__ == "__main__":
    main()
