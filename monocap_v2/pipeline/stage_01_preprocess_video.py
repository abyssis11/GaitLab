from __future__ import annotations

import shutil
from pathlib import Path

from monocap_v2.core.artifact_registry import ArtifactRegistry
from monocap_v2.core.logging_utils import ensure_dir, write_json
from monocap_v2.core.stage_utils import cached, stage_result
from monocap_v2.core.video_io import read_video_frames_bgr


STAGE = "stage_01_preprocess_video"


def run(run_dir: Path, cfg: dict, force: bool = False) -> dict:
    registry = ArtifactRegistry(run_dir)
    info_path = registry.ensure_parent("video_info")
    if cached(info_path, force):
        return stage_result(STAGE, "cached", output=str(info_path))

    raw_video = Path(str(cfg.get("raw_video") or ""))
    if not raw_video.exists():
        result = stage_result(STAGE, "failed", error=f"Video does not exist: {raw_video}")
        write_json(registry.ensure_parent("video_qc"), result)
        return result

    try:
        import cv2
    except Exception as exc:
        result = stage_result(STAGE, "failed", error=f"OpenCV import failed: {exc}")
        write_json(registry.ensure_parent("video_qc"), result)
        return result

    try:
        frames, read_report = read_video_frames_bgr(raw_video)
    except Exception as exc:
        result = stage_result(STAGE, "failed", error=f"Could not open/read video: {raw_video}: {exc}")
        write_json(registry.ensure_parent("video_qc"), result)
        return result

    width = int(read_report.get("width") or 0)
    height = int(read_report.get("height") or 0)
    fps = float(read_report.get("fps") or 0.0)
    metadata_frame_count = int(read_report.get("metadata_frame_count") or 0)
    metadata_duration = float(metadata_frame_count / fps) if fps > 0 else 0.0
    sequential_count = int(read_report.get("sequential_decoded_frame_count") or 0)
    usable_count = int(read_report.get("usable_frame_count") or 0)
    frame_count = usable_count if usable_count > 0 else metadata_frame_count
    duration = float(frame_count / fps) if fps > 0 else 0.0
    sample_dir = ensure_dir(registry.get("sample_frame_000").parent)
    sample_paths = _write_sample_frames(frames, sample_dir)

    preprocessed = registry.ensure_parent("preprocessed_video")
    copied = False
    if raw_video.suffix.lower() == ".mp4":
        shutil.copy2(raw_video, preprocessed)
        copied = True

    info = {
        "raw_video": str(raw_video),
        "preprocessed_video": str(preprocessed) if copied else None,
        "preprocessed_is_copy": copied,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "metadata_frame_count": metadata_frame_count,
        "decoded_frame_count": sequential_count,
        "sequential_decoded_frame_count": sequential_count,
        "random_access_frame_count": read_report.get("random_access_frame_count"),
        "random_access_recovered_count": read_report.get("random_access_recovered_count"),
        "usable_frame_count": usable_count,
        "missing_frame_indices": read_report.get("missing_frame_indices"),
        "duration_sec": duration,
        "metadata_duration_sec": metadata_duration,
        "rotation_degrees": 0,
        "codec": None,
        "sample_frames": sample_paths,
    }
    warnings = []
    if not copied:
        warnings.append("Original video is referenced instead of transcoded to mp4.")
    if metadata_frame_count > 0 and sequential_count > 0 and metadata_frame_count != sequential_count:
        warnings.append(
            f"Video metadata reports {metadata_frame_count} frames, but sequential OpenCV decoded {sequential_count} frames."
        )
    if read_report.get("random_access_recovered_count"):
        warnings.append(f"Recovered {read_report['random_access_recovered_count']} tail frames with random access.")
    if read_report.get("missing_frame_indices"):
        warnings.append(f"Missing frame indices after recovery: {read_report['missing_frame_indices']}.")
    qc = {
        "stage": STAGE,
        "status": "ok",
        "frame_count_gt_zero": frame_count > 0,
        "metadata_frame_count": metadata_frame_count,
        "decoded_frame_count": sequential_count,
        "sequential_decoded_frame_count": sequential_count,
        "random_access_frame_count": read_report.get("random_access_frame_count"),
        "random_access_recovered_count": read_report.get("random_access_recovered_count"),
        "usable_frame_count": usable_count,
        "missing_frame_indices": read_report.get("missing_frame_indices"),
        "frame_count_mismatch": metadata_frame_count > 0 and usable_count > 0 and metadata_frame_count != usable_count,
        "fps_detected": fps > 0,
        "sample_frames_saved": len(sample_paths),
        "warning": " ".join(warnings) if warnings else None,
    }
    write_json(info_path, info)
    write_json(registry.ensure_parent("video_qc"), qc)
    return stage_result(STAGE, "ok", output=str(info_path), frame_count=frame_count, fps=fps)


def _write_sample_frames(frames: list, sample_dir: Path) -> list[str]:
    import cv2

    if not frames:
        return []
    frame_count = len(frames)
    targets = [
        ("frame_000.png", 0),
        ("frame_mid.png", max(0, frame_count // 2)),
        ("frame_last.png", max(0, frame_count - 1)),
    ]
    saved = []
    for name, idx in targets:
        frame = frames[idx].bgr
        out = sample_dir / name
        cv2.imwrite(str(out), frame)
        saved.append(str(out))
    return saved
