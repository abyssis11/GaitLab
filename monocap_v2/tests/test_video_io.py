from __future__ import annotations

import numpy as np

from monocap_v2.core.video_io import read_video_frames_bgr


def test_robust_reader_normal_sequential_decode() -> None:
    frames, report = read_video_frames_bgr("fake.avi", capture_factory=_factory(metadata_count=4, sequential_limit=4))
    assert [frame.index for frame in frames] == [0, 1, 2, 3]
    assert report["sequential_decoded_frame_count"] == 4
    assert report["random_access_recovered_count"] == 0
    assert report["usable_frame_count"] == 4
    assert report["missing_frame_indices"] == []


def test_robust_reader_recovers_random_access_tail() -> None:
    frames, report = read_video_frames_bgr("fake.avi", capture_factory=_factory(metadata_count=5, sequential_limit=3))
    assert [frame.index for frame in frames] == [0, 1, 2, 3, 4]
    assert report["sequential_decoded_frame_count"] == 3
    assert report["random_access_recovered_count"] == 2
    assert report["random_access_recovered_indices"] == [3, 4]
    assert report["missing_frame_indices"] == []


def test_robust_reader_reports_unrecoverable_missing_frames() -> None:
    frames, report = read_video_frames_bgr(
        "fake.avi",
        capture_factory=_factory(metadata_count=5, sequential_limit=3, accessible_random={0, 1, 2, 4}),
    )
    assert [frame.index for frame in frames] == [0, 1, 2, 4]
    assert report["random_access_recovered_indices"] == [4]
    assert report["missing_frame_indices"] == [3]
    assert report["complete"] is False


class _FakeCapture:
    def __init__(self, metadata_count: int, sequential_limit: int, accessible_random: set[int] | None = None):
        self.metadata_count = metadata_count
        self.sequential_limit = sequential_limit
        self.accessible_random = accessible_random or set(range(metadata_count))
        self.pos = 0
        self.random_mode = False

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        pass

    def get(self, prop: int) -> float:
        if prop == 3:
            return 8
        if prop == 4:
            return 6
        if prop == 5:
            return 30.0
        if prop == 7:
            return float(self.metadata_count)
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        if prop == 1:
            self.pos = int(value)
            self.random_mode = True
            return True
        return False

    def read(self):
        idx = self.pos
        if self.random_mode:
            if idx not in self.accessible_random:
                return False, None
        elif idx >= self.sequential_limit:
            return False, None
        if idx >= self.metadata_count:
            return False, None
        self.pos += 1
        return True, np.full((6, 8, 3), idx, dtype=np.uint8)


def _factory(metadata_count: int, sequential_limit: int, accessible_random: set[int] | None = None):
    def make(_path: str) -> _FakeCapture:
        return _FakeCapture(metadata_count, sequential_limit, accessible_random)

    return make
