"""Pure-logic tests for local ASS caption generation.

No whisperx / torch / opencv imports here: these tests only exercise the
caption slicing, grouping, and ASS-string builders in
shorts_generator/local/captions.py, so they run anywhere Python runs.

Run directly (no pytest required):
    python tests/test_captions.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shorts_generator.local.captions import (  # noqa: E402
    _ass_color,
    _ass_time,
    _clean_word,
    build_ass_for_clip,
    group_words,
    resolve_preset,
    slice_words_for_clip,
)


def _transcript_with_words():
    return {
        "duration": 10.0,
        "segments": [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "hello world again",
                "words": [
                    {"word": "hello", "start": 0.20, "end": 0.80},
                    {"word": " world", "start": 0.85, "end": 1.40},
                    {"word": "again", "start": 1.45, "end": 2.10},
                ],
            },
            {
                "start": 4.0,
                "end": 6.0,
                "text": "other words",
                "words": [
                    {"word": "other", "start": 4.10, "end": 4.60},
                    {"word": "words", "start": 4.70, "end": 5.30},
                ],
            },
        ],
    }


def test_slice_words_rezeros_relative_to_clip_start():
    words = slice_words_for_clip(_transcript_with_words(), start=0.0, end=3.0)
    assert [w["word"] for w in words] == ["hello", "world", "again"]
    # Times re-zeroed relative to clip start (round for float tolerance).
    assert round(words[0]["start"], 2) == 0.20
    assert round(words[0]["end"], 2) == 0.80


def test_slice_words_rezeros_with_offset():
    words = slice_words_for_clip(_transcript_with_words(), start=4.0, end=6.0)
    assert [w["word"] for w in words] == ["other", "words"]
    assert round(words[0]["start"], 2) == 0.10
    assert round(words[1]["start"], 2) == 0.70


def test_slice_words_excludes_out_of_window():
    words = slice_words_for_clip(_transcript_with_words(), start=0.2, end=5.0)
    # hello is cut mid-word (end > start window) and everything >= 5 excluded.
    assert words[0]["word"] == "hello"
    assert all(w["end"] <= 5.0 for w in words)


def test_slice_words_dedupes():
    t = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "x",
                "words": [
                    {"word": "dup", "start": 0.5, "end": 1.0},
                    {"word": "dup", "start": 0.5, "end": 1.0},
                ],
            }
        ]
    }
    assert len(slice_words_for_clip(t, 0.0, 2.0)) == 1


def test_group_words_respects_group_size():
    words = [{"word": f"w{i}", "start": float(i), "end": float(i) + 0.4} for i in range(9)]
    groups = group_words(words, words_per_group=4)
    assert [len(g) for g in groups] == [4, 4, 1]
    groups = group_words(words, words_per_group=1)
    assert len(groups) == 9
    groups = group_words(words, words_per_group=0)  # invalid -> config default (4)
    assert [len(g) for g in groups] == [4, 4, 1]


def test_clean_word_strips_ass_markup():
    assert _clean_word("{b}hey{/b}") == "bhey/b"
    assert _clean_word("a\\b") == "ab"
    assert _clean_word("multi\nline") == "multi line"
    assert _clean_word("   ") == ""


def test_ass_color_is_bbggrr():
    assert _ass_color("#FFD60A") == "0AD6FF"  # R=FF G=D6 B=0A -> BB GG RR
    assert _ass_color("#5BFF6B") == "6BFF5B"
    assert _ass_color("#000000") == "000000"
    assert _ass_color("garbage") == "FFFFFF"  # invalid -> white


def test_ass_time_format():
    assert _ass_time(0.0) == "0:00:00.00"
    assert _ass_time(1.5) == "0:00:01.50"
    assert _ass_time(65.05) == "0:01:05.05"
    assert _ass_time(3600.99) == "1:00:00.99"


def test_build_ass_empty_when_no_words():
    assert build_ass_for_clip([]) == ""


def test_build_ass_contains_dialogue_lines():
    words = slice_words_for_clip(_transcript_with_words(), 0.0, 3.0)
    ass = build_ass_for_clip(words, style="clean", words_per_group=2)
    assert ass.startswith("[Script Info]")
    assert "PlayResX: 720" in ass
    assert "PlayResY: 1280" in ass
    assert "Dialogue: 0,0:00:00.20,0:00:01.40" in ass  # first group start/end
    assert ass.count("Dialogue: 0,") == 2  # 3 words / 2 per group
    assert "\\t(" in ass  # per-word overrides present


def test_resolve_preset_unknown_raises():
    try:
        resolve_preset("nope")
        assert False, "expected ValueError"
    except ValueError:
        pass


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