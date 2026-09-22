"""Regression tests for local clipping (WinError 32 masking + output naming).

Repro: OpenCV 5.0.0 removed cv2.CascadeClassifier (the whole objdetect
module). _reframe_vertical raised AttributeError right after opening the
temp cut file; with no try/finally the capture handle never released, so
crop_clip_local's cleanup `os.remove` blew up with WinError 32 that masked
the real error.

Run directly (no pytest required):
    python tests/test_clipper.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shorts_generator.local.clipper import (  # noqa: E402
    _load_face_cascade,
    _remove_quietly,
    _short_output_path,
)


class _NoCascadeClassifier:
    """Mirrors OpenCV 5: objdetect / CascadeClassifier do not exist."""

    def __getattr__(self, name):
        raise AttributeError(name)


class _FakeCascade:
    def empty(self):
        return False


class _FakeCv2:
    """Mirrors OpenCV 4: cascade loading works."""

    class data:
        haarcascades = "/fake/haarcascades/"

    @staticmethod
    def CascadeClassifier(path):
        return _FakeCascade()


def test_face_cascade_falls_back_when_cascade_missing():
    # OpenCV 5 build: CascadeClassifier is gone -> loader must return None,
    # not crash the clip.
    assert _load_face_cascade(_NoCascadeClassifier()) is None


def test_face_cascade_loaded_when_available():
    # OpenCV 4 build: loader returns a usable cascade.
    assert _load_face_cascade(_FakeCv2()) is not None


def test_short_output_path_names_from_source_stem():
    out = _short_output_path(r"output\Test Video_source_abc123.mp4", r"output\Test Video", 1)
    assert out == os.path.join(r"output\Test Video", "Test Video_source_abc123_short_01.mp4")
    out = _short_output_path("my_local_clip.mp4", "out", 12)
    assert out == os.path.join("out", "my_local_clip_short_12.mp4")


def test_remove_quietly_never_raises():
    with tempfile.TemporaryDirectory() as td:
        missing = os.path.join(td, "does_not_exist.mp4")
        _remove_quietly(missing)  # must not raise

        locked = os.path.join(td, "locked")
        os.mkdir(locked)  # os.remove on a directory -> OSError, swallowed
        _remove_quietly(locked, attempts=2, delay=0.0)
        assert os.path.isdir(locked)  # left in place, run not failed


def main() -> None:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {e!r}")
    if failures:
        sys.exit(f"{failures} test(s) failed")


if __name__ == "__main__":
    main()