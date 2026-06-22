from __future__ import annotations

import subprocess
from argparse import Namespace
from pathlib import Path

from monocap_v2.scripts import check_sam3d_prereqs


def test_sam3d_prereq_checker_reports_ok_with_mocked_imports(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    sam_repo = repo / "external" / "sam-3d-body"
    checkpoint_dir = repo / "models" / "sam3d_body"
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").write_text("#!/usr/bin/env python\n", encoding="utf-8")
    (sam_repo / "sam_3d_body").mkdir(parents=True)
    (sam_repo / "demo.py").write_text("", encoding="utf-8")
    (sam_repo / "sam_3d_body" / "sam_3d_body_estimator.py").write_text("", encoding="utf-8")
    (checkpoint_dir / "sam-3d-body-dinov3" / "assets").mkdir(parents=True)
    (checkpoint_dir / "sam-3d-body-dinov3" / "model_config.yaml").write_text("MODEL:\n  NAME: SAM3DBody\n", encoding="utf-8")
    (checkpoint_dir / "sam-3d-body-dinov3" / "model.ckpt").write_bytes(b"ckpt")
    (checkpoint_dir / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt").write_bytes(b"mhr")

    def fake_run(cmd, **kwargs):
        if cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(
            cmd,
            0,
            '{"torch": {"ok": true, "cuda_available": true}, "cv2": {"ok": true}, '
            '"detectron2": {"ok": true}, "moge": {"ok": true}, "sam_3d_body": {"ok": true}}\n',
            "",
        )

    monkeypatch.setattr(check_sam3d_prereqs.subprocess, "run", fake_run)
    report = check_sam3d_prereqs.build_report(
        Namespace(repo_root=repo, env_path=env, sam3d_repo=sam_repo, checkpoint_dir=checkpoint_dir)
    )

    assert report["status"] == "ok"
    assert report["checks"]["assets"]["status"] == "ok"
    assert report["checks"]["python_imports"]["status"] == "ok"
