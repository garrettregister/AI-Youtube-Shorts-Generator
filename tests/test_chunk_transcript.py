"""Regression tests for long-video highlight chunking.

Repro: on videos >= LONG_VIDEO_THRESHOLD, chunk 2+ segments kept their
absolute video timestamps in the transcript text while highlights were
sanitized against the chunk length, so every returned time >= chunk length
was clamped to end == start and dropped -> "invalid model output" retries.

Run directly (no pytest required):
    python tests/test_chunk_transcript.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shorts_generator.highlights import (  # noqa: E402
    LONG_VIDEO_THRESHOLD,
    chunk_transcript,
    get_highlights,
)


def make_transcript(duration: float = 2512.0, step: float = 5.0) -> dict:
    segments = []
    t = 0.0
    while t < duration:
        text = "A spoken sentence here."
        segments.append({"start": round(t, 2), "end": round(t + 3.5, 2), "text": text})
        t += step
    return {"segments": segments, "duration": duration}


def test_transcript_is_long_enough_to_chunk():
    assert make_transcript()["duration"] >= LONG_VIDEO_THRESHOLD


def test_chunks_are_rebased_to_local_origin():
    chunks = chunk_transcript(make_transcript())
    assert len(chunks) >= 2
    for i, chunk in enumerate(chunks):
        local_starts = [s["start"] for s in chunk["segments"]]
        # A chunk past the first must not carry absolute video timestamps.
        assert min(local_starts) >= 0, f"chunk {i}: negative local start"
        # No visible timestamp may exceed the sanitizer's clamp window,
        # otherwise every highlight the model returns there gets dropped.
        max_end = max(s["end"] for s in chunk["segments"])
        assert max_end <= chunk["duration"], (
            f"chunk {i}: segment end {max_end} > chunk duration {chunk['duration']}"
        )


def _stub_llm(prompt: str) -> str:
    """Return a 45s highlight that ends at the LAST timestamp the model sees
    in the prompt. On the old buggy code this time is absolute and gets
    clamped to end == start (dropped); on the fixed code it is local and
    survives, proving the chunk pipeline now produces highlights."""
    times = [float(m) for m in re.findall(r"\[(\d+\.?\d*)s\]", prompt)]
    assert times, "no timestamps found in prompt"
    hi = max(times)
    lo = hi - 45.0
    return json.dumps(
        {
            "highlights": [
                {
                    "title": "test",
                    "start_time": lo,
                    "end_time": hi,
                    "score": 80,
                    "hook_sentence": "hook",
                    "virality_reason": "reason",
                }
            ]
        }
    )


def test_long_video_returns_chunk_highlights_with_absolute_times():
    transcript = make_transcript()
    result = get_highlights(transcript, num_clips=2, llm_fn=_stub_llm)
    highlights = result["highlights"]
    assert highlights, "chunk pipeline produced zero highlights"
    # Sanity: at least one highlight must land past the first 20-min chunk,
    # proving chunk 2+ now survives sanitization with its offset applied.
    assert any(float(h["start_time"]) > 1200 for h in highlights), highlights
    for h in highlights:
        start, end = float(h["start_time"]), float(h["end_time"])
        assert 0 <= start < end <= transcript["duration"], (start, end)


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