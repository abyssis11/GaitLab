#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monocap_v2.core.opensim_fk import export_opensim_fk_reference


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export OpenSim FK joint-center reference from a mocap IK .mot file.")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--ik-mot", required=True, type=Path)
    ap.add_argument("--out-npz", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--marker-errors", type=Path, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    meta = export_opensim_fk_reference(
        model_path=args.model,
        ik_mot_path=args.ik_mot,
        out_npz=args.out_npz,
        out_json=args.out_json,
        marker_errors_path=args.marker_errors,
    )
    print(f"[opensim-fk] {meta.get('status')} -> {args.out_npz}", flush=True)
    return 0 if meta.get("status") in {"ok", "warning"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
