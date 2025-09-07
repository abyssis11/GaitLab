#!/usr/bin/env python3
import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from typing import Dict, Any, Optional

def _rational_to_float(s: str) -> Optional[float]:
    if not s or s in ("0/0", "N/A"):
        return None
    try:
        if "/" in s:
            num, den = s.split("/", 1)
            num = float(num)
            den = float(den)
            if den == 0:
                return None
            return num / den
        return float(s)
    except Exception:
        return None

def probe_ffprobe(path: str, ffprobe_bin: str = "ffprobe", timeout: float = 15.0) -> Dict[str, Any]:
    """Use ffprobe to get precise metadata (preferred)."""
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-print_format", "json",
        path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=True)
        data = json.loads(res.stdout.decode("utf-8", errors="ignore") or "{}")
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "error": f"ffprobe failed: {e}", "cmd": " ".join(shlex.quote(c) for c in cmd)}
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    out: Dict[str, Any] = {"ok": True, "source": "ffprobe"}
    if streams:
        s0 = streams[0]
        out["codec"] = s0.get("codec_name")
        out["width"] = int(s0.get("width") or 0) or None
        out["height"] = int(s0.get("height") or 0) or None
        out["avg_frame_rate"] = s0.get("avg_frame_rate")
        out["r_frame_rate"] = s0.get("r_frame_rate")
        out["fps_avg"] = _rational_to_float(s0.get("avg_frame_rate"))
        out["fps_r"] = _rational_to_float(s0.get("r_frame_rate"))
        nb = s0.get("nb_frames")
        try:
            out["nb_frames"] = int(nb) if nb is not None and str(nb).isdigit() else None
        except Exception:
            out["nb_frames"] = None
        # Prefer stream duration; fall back to container duration
        dur = None
        try:
            if s0.get("duration") not in (None, "N/A"):
                dur = float(s0.get("duration"))
        except Exception:
            dur = None
        if dur is None:
            try:
                if fmt.get("duration") not in (None, "N/A"):
                    dur = float(fmt.get("duration"))
            except Exception:
                dur = None
        out["duration"] = dur
        # Estimate frame count if missing but have fps and duration
        if out.get("nb_frames") in (None, 0) and dur is not None:
            fps = out.get("fps_avg") or out.get("fps_r")
            if fps:
                # Using round(duration * fps) + 1 approximates inclusive frame endpoints in some pipelines
                est = int(round(dur * fps)) + 1
                out["nb_frames_est"] = est
    return out

def probe_opencv(path: str, force_decode: bool = False) -> Dict[str, Any]:
    """Use OpenCV as a fallback. CAP_PROP_FRAME_COUNT is fast; force_decode iterates to count accurately."""
    out: Dict[str, Any] = {"ok": False, "source": "opencv"}
    try:
        import cv2  # type: ignore
    except Exception as e:
        out["error"] = f"OpenCV not available: {e}"
        return out
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            out["error"] = "cv2.VideoCapture failed to open"
            return out
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        out.update({
            "ok": True,
            "fps_cv": float(fps) if fps > 0 else None,
            "width": w or None,
            "height": h or None,
            "nb_frames_prop": n if n > 0 else None
        })
        if force_decode:
            # Iterate through frames to count reliably (slow).
            count = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                count += 1
            out["nb_frames_decoded"] = count
    finally:
        if cap is not None:
            cap.release()
    return out

def summarize(path: str, ffprobe_bin: str, force_decode: bool) -> None:
    print(f"\n=== {path} ===")
    ff = probe_ffprobe(path, ffprobe_bin=ffprobe_bin)
    cv = probe_opencv(path, force_decode=force_decode)

    # Consolidate
    fps = None
    fps_src = None
    nb = None
    nb_src = None
    dur = None

    if ff.get("ok"):
        fps = ff.get("fps_avg") or ff.get("fps_r") or fps
        fps_src = "ffprobe(avg)" if ff.get("fps_avg") else ("ffprobe(r)" if ff.get("fps_r") else None)
        nb = ff.get("nb_frames") or ff.get("nb_frames_est") or nb
        nb_src = "ffprobe(nb_frames)" if ff.get("nb_frames") else ("ffprobe(est)" if ff.get("nb_frames_est") else None)
        dur = ff.get("duration") or dur

    if (fps is None) and cv.get("ok"):
        fps = cv.get("fps_cv") or fps
        fps_src = "opencv(FPS)" if cv.get("fps_cv") else fps_src

    if (nb is None) and cv.get("ok"):
        nb = cv.get("nb_frames_prop") or cv.get("nb_frames_decoded") or nb
        if cv.get("nb_frames_prop"):
            nb_src = nb_src or "opencv(prop)"
        if cv.get("nb_frames_decoded"):
            nb_src = nb_src or "opencv(decoded)"

    # Derived duration
    derived_dur = None
    if fps and nb and fps > 0 and nb > 0:
        derived_dur = (nb - 1) / fps

    # Print a neat summary
    def fmt(x, nd=3):
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "—"
        if isinstance(x, float):
            return f"{x:.{nd}f}"
        return str(x)

    # Basic line
    print(f"fps: {fmt(fps)} [{fps_src or '—'}]")
    if ff.get("ok"):
        print(f"ffprobe: codec={ff.get('codec') or '—'} size={ff.get('width') or '—'}x{ff.get('height') or '—'} "
              f"avg={ff.get('avg_frame_rate') or '—'} r={ff.get('r_frame_rate') or '—'} "
              f"nb={ff.get('nb_frames') or '—'} dur={fmt(ff.get('duration'))}")
    if cv.get("ok"):
        print(f"opencv: fps_prop={fmt(cv.get('fps_cv'))} nb_prop={cv.get('nb_frames_prop') or '—'} "
              f"nb_decoded={cv.get('nb_frames_decoded') or '—'} size={cv.get('width') or '—'}x{cv.get('height') or '—'}")
    if nb is not None:
        print(f"chosen frames: {nb} [{nb_src or '—'}]  |  derived_duration: {fmt(derived_dur)} s")
    if dur is not None and derived_dur is not None:
        delta = abs(dur - derived_dur)
        print(f"container_duration: {fmt(dur)} s  |  Δ(derived - container)={fmt(delta, nd=4)} s")


def frame_iter(capture):
    while capture.grab():
        yield capture.retrieve()[1]

def main():
    import cv2
    import numpy as np
    capture = cv2.VideoCapture('walking1_syncdWithMocap.avi')
    frames = np.stack([x for x in frame_iter(capture)])
    print(len(frames))
    '''
    retcode = 0
    for p in args.paths:
        if not os.path.exists(p):
            print(f"\n=== {p} ===\nERROR: file not found")
            retcode = 2
            continue
        summarize(p, ffprobe_bin=args.ffprobe, force_decode=args.force_decode)
    sys.exit(retcode)
    '''

if __name__ == "__main__":
    main()
