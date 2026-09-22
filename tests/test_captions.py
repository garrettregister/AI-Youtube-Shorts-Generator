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


def _event_texts(ass: str):
    return [ln for ln in ass.splitlines() if ln.startswith("Dialogue: 0,")]


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


def test_sticky_styles_carry_active_colour_in_primary():
    words = slice_words_for_clip(_transcript_with_words(), 0.0, 3.0)
    # libass draws the played part of a karaoke line in the PrimaryColour and
    # the upcoming part in the SecondaryColour (reverse of the ASS spec), so
    # sticky styles must put the active colour in *primary* and the plain text
    # in secondary, or the highlight renders inside-out.
    ass = build_ass_for_clip(words, style="hype")
    # hype active #FFD60A -> BBGGRR 0AD6FF as PrimaryColour, white secondary.
    assert "&H000AD6FF,&H00FFFFFF" in ass
    ass = build_ass_for_clip(words, style="karaoke")
    # karaoke active #5BFF6B -> BBGGRR 6BFF5B as PrimaryColour, white secondary.
    assert "&H006BFF5B,&H00FFFFFF" in ass


def _karaoke_tokens(event):
    """The `\\k` centisecond values in the order they appear in a line."""
    parts = event.split("{\\k")[1:]
    return [int(p.split("}")[0]) for p in parts]


def test_sticky_karaoke_uses_karaoke_tokens_keyed_to_word_starts():
    words = slice_words_for_clip(_transcript_with_words(), 0.0, 3.0)
    ass = build_ass_for_clip(words, style="karaoke", words_per_group=3)
    assert "Dialogue: 0,0:00:00.20,0:00:02.10" in ass
    event = _event_texts(ass)[0]
    # Line starts at hello 0.20. Karaoke \k sweeps the PrimaryColour (set to
    # the active colour) forward and never resets rightward text.
    #   hello 0.20-0.80, world 0.85-1.40, again 1.45-2.10 (line-relative).
    # \k before each word = start of next word - start of this word (last
    # word = its own spoken span, 190 - 125 = 65cs).
    assert _karaoke_tokens(event) == [65, 60, 65]
    assert "{\\fad(60,60)}" in event
    assert "{\\k65}hello" in event
    assert "{\\k60}world" in event
    assert "{\\k65}again" in event
    # No per-word \1c colour overrides or \t flips: highlight is pure \k.
    assert "\\1c" not in event
    assert "\\t(" not in event
    # Cumulative karaoke time reaches the end of the line at the last word's
    # end (65 + 60 + 65 = 190cs = 2.10 - 0.20).
    assert sum(_karaoke_tokens(event)) == 190


def test_sticky_hype_preserves_karaoke_tokens_with_pop():
    words = slice_words_for_clip(_transcript_with_words(), 0.0, 3.0)
    ass = build_ass_for_clip(words, style="hype", words_per_group=3)
    event = _event_texts(ass)[0]
    # hype is sticky: same \k word-start keys, pop \fscx tags interleaved.
    assert _karaoke_tokens(event) == [65, 60, 65]
    assert "\\1c" not in event
    assert "\\fscx126" in event
    assert event.index("{\\k60}") < event.index("world")


def test_clean_time_windows_are_line_relative():
    words = slice_words_for_clip(_transcript_with_words(), 0.0, 3.0)
    ass = build_ass_for_clip(words, style="clean", words_per_group=3)
    # First word of the line animates from 0, not from its absolute 20cs.
    assert "{\\t(0,60,\\1c" in ass
    assert "\\t(20,80" not in ass


def test_later_group_windows_are_line_relative():
    words = slice_words_for_clip(_transcript_with_words(), 4.0, 6.0)
    ass = build_ass_for_clip(words, style="clean", words_per_group=2)
    # Group starts at clip 4.10 -> line at 0.10; windows relative to the line.
    assert "Dialogue: 0,0:00:00.10,0:00:01.30" in ass
    assert "{\\t(0,50,\\1c" in ass
    assert "\\t(410,460" not in ass


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