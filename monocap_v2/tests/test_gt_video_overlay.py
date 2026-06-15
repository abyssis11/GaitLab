from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np

from monocap_v2.core.gt_video_overlay import project_opensim_fk_to_pixels, render_gt_labeled_video_overlay


def test_project_opensim_fk_to_pixels_with_synthetic_camera(tmp_path: Path) -> None:
    camera = _camera_pickle(tmp_path / "camera.pickle")
    joints = np.asarray([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]], dtype=float)

    uv, valid = project_opensim_fk_to_pixels(joints, camera)

    assert valid.tolist() == [[True, True]]
    assert np.allclose(uv[0, 0], [50.0, 50.0], atol=1e-5)
    assert uv[0, 1, 0] > uv[0, 0, 0]


def test_gt_labeled_video_overlay_writes_readable_mp4(tmp_path: Path) -> None:
    raw = _write_video(tmp_path / "raw.mp4", [_frame(i) for i in range(6)])
    sync = _write_video(tmp_path / "sync.mp4", [_frame(i) for i in range(2, 5)])
    camera = _camera_pickle(tmp_path / "camera.pickle")
    run_dir = tmp_path / "run"
    (run_dir / "reports").mkdir(parents=True)
    (run_dir / "run_config.yaml").write_text(
        "\n".join(
            [
                f"repo_root: {tmp_path}",
                "trial_id: walking1",
                "trial:",
                f"  video_sync: {sync}",
                f"  video_raw: {raw}",
                "manifest_summary:",
                "  calibration:",
                f"    intrinsics_extrinsics: {camera}",
            ]
        ),
        encoding="utf-8",
    )
    ref = tmp_path / "opensim_fk_walking1.npz"
    time = np.asarray([0.0, 0.1, 0.2], dtype=float)
    joints = np.asarray(
        [
            [[0.0, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(ref, time_s=time, joints_m=joints, joint_names=np.asarray(["pelvis", "left_hip", "right_hip"]))

    out_mp4 = run_dir / "reports" / "gt_labeled_overlay.mp4"
    out_json = run_dir / "reports" / "gt_labeled_overlay_qc.json"
    report = render_gt_labeled_video_overlay(run_dir, ref, out_mp4, out_json, video_space="raw", preview_fps=10.0)

    assert report["status"] == "ok"
    assert report["raw_sync_alignment"]["best_raw_offset"] == 2
    assert out_mp4.exists() and out_mp4.stat().st_size > 0
    cap = cv2.VideoCapture(str(out_mp4))
    try:
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) == 3
    finally:
        cap.release()


def _camera_pickle(path: Path) -> Path:
    with path.open("wb") as f:
        pickle.dump(
            {
                "intrinsicMat": np.asarray([[500.0, 0.0, 50.0], [0.0, 500.0, 50.0], [0.0, 0.0, 1.0]], dtype=float),
                "distortion": np.zeros((1, 5), dtype=float),
                "imageSize": np.asarray([[100.0], [100.0]], dtype=float),
                "rotation": np.eye(3, dtype=float),
                "translation": np.asarray([[0.0], [0.0], [3000.0]], dtype=float),
            },
            f,
        )
    return path


def _write_video(path: Path, frames: list[np.ndarray]) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (100, 100))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def _frame(idx: int) -> np.ndarray:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :, 0] = np.uint8((idx * 30) % 255)
    frame[20:80, 20:80, 1] = np.uint8((80 + idx * 20) % 255)
    frame[40:60, :, 2] = np.uint8((120 + idx * 10) % 255)
    return frame
