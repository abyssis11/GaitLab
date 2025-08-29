#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import re
from typing import Dict, List, Tuple, Optional, Iterable
import numpy as np
import pandas as pd
from pathlib import Path

def _tokenize_words(line: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_.\-#]+", line)

def parse_trc_header(lines: List[str]) -> Tuple[int, List[str], Optional[str]]:
    frame_row_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Frame#"):
            frame_row_idx = i
            break
    if frame_row_idx is None:
        raise ValueError("Could not find 'Frame#' header line in TRC.")
    tokens = _tokenize_words(lines[frame_row_idx])
    marker_names = tokens[2:]  # after Frame# and Time
    axis_row_idx = frame_row_idx + 1
    units = None
    try:
        for i, line in enumerate(lines):
            if "DataRate" in line and "Units" in line:
                values = _tokenize_words(lines[i + 1])
                if len(values) >= 5:
                    units = values[4]
                break
    except Exception:
        pass
    return axis_row_idx, marker_names, units

def read_trc(path: Path) -> Tuple[pd.DataFrame, List[str], Optional[str]]:
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    axis_row_idx, marker_names, units = parse_trc_header(text)
    cols = ["Frame", "Time"]
    for m in marker_names:
        cols.extend([f"{m}_X", f"{m}_Y", f"{m}_Z"])
    skip = axis_row_idx + 1
    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        skiprows=skip,
        names=cols,
        na_values=["NaN", "nan", ""],
    )
    df = df.dropna(how="all")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, marker_names, units

def extract_marker_arrays(df: pd.DataFrame, markers: Iterable[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for m in markers:
        cols = [f"{m}_X", f"{m}_Y", f"{m}_Z"]
        if all(c in df.columns for c in cols):
            out[m] = df[cols].to_numpy(dtype=float)
    return out

def choose_b_markers(
    names_a: List[str],
    names_b: List[str],
    suffixes_b: List[str],
) -> Dict[str, str]:
    """
    For each marker 'm' from A, try to find 'm + suffix' in B (first match wins).
    Returns mapping {m_in_A : corresponding_name_in_B}. Only includes found pairs.
    """
    set_b = set(names_b)
    mapping = {}
    for m in names_a:
        # try exact name first only if "" is present in suffix list
        for sfx in suffixes_b:
            candidate = m + sfx
            if candidate in set_b:
                mapping[m] = candidate
                break
    return mapping

def mpjpe_between(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    mapping_a_to_b: Dict[str, str],
    frame_range: Optional[Tuple[int, int]] = None,
) -> Tuple[pd.DataFrame, float]:
    # Align by frame number if possible
    if "Frame" in df_a.columns and "Frame" in df_b.columns:
        a = df_a.set_index("Frame")
        b = df_b.set_index("Frame")
        if frame_range:
            start, end = frame_range
            a = a.loc[start:end]
            b = b.loc[start:end]
        common = a.index.intersection(b.index)
        if len(common) == 0:
            raise ValueError("No overlapping frames between files.")
        a = a.loc[common]
        b = b.loc[common]
    else:
        n = min(len(df_a), len(df_b))
        a = df_a.iloc[:n].copy()
        b = df_b.iloc[:n].copy()
        if frame_range:
            start, end = frame_range
            a = a.iloc[max(0, start - 1): min(n, end)]
            b = b.iloc[max(0, start - 1): min(n, end)]

    rows = []
    total_err = 0.0
    total_count = 0
    all_equal = True  # to warn if we compared identical series

    for m_a, m_b in sorted(mapping_a_to_b.items()):
        cols_a = [f"{m_a}_X", f"{m_a}_Y", f"{m_a}_Z"]
        cols_b = [f"{m_b}_X", f"{m_b}_Y", f"{m_b}_Z"]
        if not all(c in a.columns for c in cols_a): 
            continue
        if not all(c in b.columns for c in cols_b): 
            continue

        pa = a[cols_a].to_numpy(dtype=float)
        pb = b[cols_b].to_numpy(dtype=float)
        n = min(len(pa), len(pb))
        pa = pa[:n]
        pb = pb[:n]
        valid = ~np.isnan(pa).any(axis=1) & ~np.isnan(pb).any(axis=1)

        if not np.any(valid):
            rows.append({"marker": m_a, "mpjpe_mm": np.nan, "n_valid": 0})
            continue

        # equality check (exact float match) for warning purposes
        if not np.array_equal(pa[valid], pb[valid]):
            all_equal = False

        d = np.linalg.norm(pa[valid] - pb[valid], axis=1)
        mpjpe_mm = float(np.mean(d))
        rows.append({"marker": m_a, "mpjpe_mm": mpjpe_mm, "n_valid": int(valid.sum())})

        total_err += float(np.sum(d))
        total_count += int(valid.sum())

    per_marker_df = pd.DataFrame(rows).sort_values("marker").reset_index(drop=True)
    overall_mpjpe_mm = float(total_err / total_count) if total_count > 0 else float("nan")
    return per_marker_df, overall_mpjpe_mm

def main():
    ap = argparse.ArgumentParser(description="Compute MPJPE between two TRC files.")
    ap.add_argument("trc_a", type=Path, help="Path to TRC A (reference, e.g., Mocap)")
    ap.add_argument("trc_b", type=Path, help="Path to TRC B (e.g., OpenPose)")
    ap.add_argument("--markers", type=Path, default=None,
                    help="Optional text file with one marker name per line (names from file A).")
    ap.add_argument("--include", type=str, default=None,
                    help="Regex to include markers (applied to A's names).")
    ap.add_argument("--exclude", type=str, default=None,
                    help="Regex to exclude markers (applied to A's names).")
    ap.add_argument("--frame-range", type=str, default=None,
                    help="1-based inclusive frame range, e.g., '1:158'.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional CSV path to save per-joint MPJPE results.")
    ap.add_argument("--b-suffixes", type=str, default="_study_offsetRemoved,_offsetRemoved,_study,",
                    help=("Comma-separated list of suffixes to try on B for each A marker, "
                          "in priority order. Include a bare empty suffix by ending with a comma. "
                          "Default: '_study_offsetRemoved,_offsetRemoved,_study,'"))
    args = ap.parse_args()

    df_a, names_a, units_a = read_trc(args.trc_a)
    df_b, names_b, units_b = read_trc(args.trc_b)

    print(f"Loaded A: {args.trc_a} ({len(df_a)} rows)  units={units_a}")
    print(f"Loaded B: {args.trc_b} ({len(df_b)} rows)  units={units_b}")

    # Select A markers
    if args.markers:
        markers_a = [ln.strip() for ln in args.markers.read_text().splitlines() if ln.strip()]
    else:
        markers_a = list(names_a)

    if args.include:
        inc = re.compile(args.include)
        markers_a = [m for m in markers_a if inc.search(m)]
    if args.exclude:
        exc = re.compile(args.exclude)
        markers_a = [m for m in markers_a if not exc.search(m)]

    if not markers_a:
        raise SystemExit("No markers selected from file A.")

    # Build A->B mapping using suffix strategy
    suffixes = [s for s in (s.strip() for s in args.b_suffixes.split(",")) if s is not None]
    mapping = choose_b_markers(markers_a, names_b, suffixes)

    if not mapping:
        raise SystemExit(
            "Could not map any A markers to B. "
            "Try adjusting --b-suffixes (e.g., '_study_offsetRemoved,_offsetRemoved,_study,')."
        )

    # Optional frame range
    frame_range = None
    if args.frame_range:
        m = re.match(r"^\s*(\d+)\s*:\s*(\d+)\s*$", args.frame_range)
        if not m:
            raise SystemExit("--frame-range must look like 'start:end' with integers.")
        start, end = int(m.group(1)), int(m.group(2))
        if start > end:
            start, end = end, start
        frame_range = (start, end)

    per_marker, overall = mpjpe_between(df_a, df_b, mapping, frame_range)

    # Detect suspiciously perfect equality
    if np.isfinite(overall) and overall == 0.0:
        print("\nWARNING: Overall MPJPE is 0. "
              "This likely means you're comparing identical coordinates.")
        print("Tip: Use --b-suffixes to point to OpenPose-specific columns, e.g.:")
        print("     --b-suffixes _study_offsetRemoved,_offsetRemoved,_study,")

    print("\nMapped markers (A -> B):")
    for k in sorted(mapping):
        print(f"  {k} -> {mapping[k]}")

    print("\nPer-joint MPJPE (mm):")
    for _, r in per_marker.iterrows():
        val = r['mpjpe_mm']
        print(f"  {r['marker']:<24s}  MPJPE={(f'{val:.3f}' if pd.notna(val) else 'NaN'):>8}   n_valid={int(r['n_valid'])}")

    print(f"\nOVERALL MPJPE (mm) across all mapped joints/frames: {overall:.3f}")

    if args.out:
        per_marker.to_csv(args.out, index=False)
        print(f"Saved per-joint results to: {args.out}")

if __name__ == "__main__":
    main()
