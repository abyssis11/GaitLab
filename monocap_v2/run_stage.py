#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monocap_v2.core.logging_utils import read_yaml, setup_logger, update_stage_state
from monocap_v2.run_pipeline import STAGES


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run one monocap_v2 stage from an existing run directory.")
    ap.add_argument("stage_name", choices=STAGES)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run
    cfg_path = run_dir / "run_config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"Missing run config: {cfg_path}")
    cfg = read_yaml(cfg_path)
    logger = setup_logger(run_dir)
    module = importlib.import_module(f"monocap_v2.pipeline.{args.stage_name}")
    result = module.run(run_dir, cfg, force=args.force)
    update_stage_state(run_dir, args.stage_name, result)
    logger.info("%s -> %s", args.stage_name, result.get("status"))
    return 0 if result.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

