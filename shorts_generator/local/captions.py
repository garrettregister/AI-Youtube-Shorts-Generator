"""Burned-in ASS subtitle generation for local short clips.

Turns word-level transcript data (produced by `transcriber.transcribe_local`)
into a static `.ass` file, rendered by ffmpeg's `ass` filter. The whole point
is word-for-word sync: each caption chunk highlights words as they are spoken,
styled to match the CapCut / Opus Clip / Submagic look.

Per-word effects are implemented with explicit `\\t(...)` override tags driven
from each word's own start/end times, rather than libass `\\k` karaoke tokens,
so behaviour is consistent across renderers.
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from ..config import (
    LOCAL_CAPTION_FONT,
    LOCAL_CAPTION_FONTSIZE,
    LOCAL_CAPTION_STYLE,
    LOCAL_CAPTION_WORDS_PER_GROUP,
)

DEFAULT_WIDTH = 720
DEFAULT_HEIGHT = 1280


class StylePreset:
    __slots__ = (
        "name", "text_color", "active_color", "outline_color", "outline",
        "shadow", "bold", "pop", "sticky", "font_scale", "margin_h_ratio",
        "margin_v_ratio",
    )

    def __init__(
        self,
        name: str,
        text_color: str,
        active_color: str,
        outline_color: str,
        outline: float,
        shadow: float,
        bold: bool,
        pop: bool,
        sticky: bool,
        font_scale: float,
        margin_h_ratio: float = 0.04,
        margin_v_ratio: float = 0.07,
    ) -> None:
        self.name = name
        self.text_color = text_color
        self.active_color = active_color
        self.outline_color = outline_color
        self.outline = outline
        self.shadow = shadow
        self.bold = bold
        self.pop = pop
        self.sticky = sticky
        self.font_scale = font_scale
        self.margin_h_ratio = margin_h_ratio
        self.margin_v_ratio = margin_v_ratio


STYLE_PRESETS: Dict[str, StylePreset] = {
    "hype": StylePreset(
        name="hype",
        text_color="#FFFFFF",
        active_color="#FFD60A",
        outline_color="#101010",
        outline=5,
        shadow=1,
        bold=True,
        pop=True,
        sticky=True,
        font_scale=0.055,
    ),
    "clean": StylePreset(
        name="clean",
        text_color="#FFFFFF",
        active_color="#FFE55C",
        outline_color="#000000",
        outline=3,
        shadow=0,
        bold=True,
        pop=True,
        sticky=False,
        font_scale=0.048,
    ),
    "karaoke": StylePreset(
        name="karaoke",
        text_color="#FFFFFF",
        active_color="#5BFF6B",
        outline_color="#101010",
        outline=4,
        shadow=1,
        bold=False,
        pop=False,
        sticky=True,
        font_scale=0.052,
    ),
}

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{fontname},{fontsize},&H00FFFFFF,&H000000FF,&H00{outline_hex},&H64000000,{bold},0,0,0,100,100,0,0,1,{outline},{shadow},2,{margin_l},{margin_r},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def resolve_preset(style: Optional[str] = None) -> StylePreset:
    style_name = (style or LOCAL_CAPTION_STYLE or "hype").strip().lower()
    if style_name not in STYLE_PRESETS:
        raise ValueError(
            f"Unknown caption style {style!r}. Choose from: {', '.join(sorted(STYLE_PRESETS))}"
        )
    return STYLE_PRESETS[style_name]


def _resolve_words_per_group(words_per_group: Optional[int]) -> int:
    if words_per_group is not None and words_per_group > 0:
        return int(words_per_group)
    try:
        value = int(LOCAL_CAPTION_WORDS_PER_GROUP)
    except (TypeError, ValueError):
        value = 4
    return max(1, value)


def _resolve_font_size(preset: StylePreset, height: int) -> int:
    if LOCAL_CAPTION_FONTSIZE:
        try:
            return max(12, int(float(LOCAL_CAPTION_FONTSIZE)))
        except (TypeError, ValueError):
            pass
    return max(12, int(round(height * preset.font_scale)))


def _ass_color(color: str) -> str:
    """#RRGGBB -> &H00BBGGRR (ASS colour order)."""
    if not color or len(color) < 7 or color[0] != "#":
        return "FFFFFF"
    try:
        rr = int(color[1:3], 16)
        gg = int(color[3:5], 16)
        bb = int(color[5:7], 16)
    except ValueError:
        return "FFFFFF"
    return f"{bb:02X}{gg:02X}{rr:02X}"


def _ass_time(seconds: float) -> str:
    """Float seconds -> H:MM:SS.cc"""
    ms = max(0, int(round(seconds * 100)))
    return f"{ms // 360000}:{(ms // 6000) % 60:02d}:{(ms // 100) % 60:02d}.{ms % 100:02d}"


def _clean_word(word: str) -> str:
    # Strip ASS braces/tags/newlines so a transcript can't inject or corrupt
    # subtitle markup.
    word = word.replace("{", "").replace("}", "").replace("\\", "").replace("\n", " ").replace("\r", " ")
    return word.strip()


def slice_words_for_clip(transcript: Dict, start: float, end: float) -> List[Dict]:
    """Flatten the transcript's words, keep those inside [start, end), and
    re-zero their timestamps relative to the clip start, clamped so no word
    starts before 0 or outlives the clip."""
    clip_words = []
    seen = set()
    clip_duration = max(0.0, end - start)
    for segment in transcript.get("segments", []):
        for w in segment.get("words", []):
            try:
                w_start, w_end = float(w["start"]), float(w["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if w_start >= end:
                continue
            if w_end <= start:
                continue
            word = _clean_word(str(w.get("word", "")))
            if not word:
                continue
            key = (round(w_start, 3), word)
            if key in seen:
                continue
            seen.add(key)
            clip_words.append({
                "word": word,
                "start": max(0.0, min(w_start - start, clip_duration)),
                "end": max(0.0, min(w_end - start, clip_duration)),
            })
    clip_words.sort(key=lambda w: w["start"])
    return clip_words


def group_words(words: List[Dict], words_per_group: Optional[int] = None) -> List[List[Dict]]:
    """Split flat words into display groups of `words_per_group` words."""
    count = _resolve_words_per_group(words_per_group)
    groups = [words[i : i + count] for i in range(0, len(words), count)]
    return [g for g in groups if g]


def _group_event_text(
    group: List[Dict], preset: StylePreset, active_hex: str, dim_hex: str
) -> str:
    """ASS markup for one caption group.

    Each word gets an explicit `\\t(ws,we,\\1c<active>)` colour override; the
    words before the active word stay dim (inactive colour). With sticky
    karaoke the colour is never reset, so all prior words stay highlighted.
    """
    text_parts = []
    prev_end = 0.0
    for i, w in enumerate(group):
        ws_cs = max(0, int(round(w["start"] * 100)))
        we_cs = max(ws_cs + 1, int(round(w["end"] * 100)))

        if preset.sticky:
            color_tags = f"{{\\t({ws_cs},{we_cs},\\1c{active_hex})}}"
        else:
            reset_cs = we_cs + 4
            color_tags = (
                f"{{\\t({ws_cs},{we_cs},\\1c{active_hex})}}"
                f"{{\\t({reset_cs},{reset_cs + 20},\\1c{dim_hex})}}"
            )

        if preset.pop:
            up_cs = ws_cs
            flash_cs = max(we_cs, up_cs + 12)
            settle_cs = flash_cs + max(20, int(round((w["end"] - prev_end if i else 0.0) * 100)))
            pop_tags = (
                f"{{\\t({up_cs},{flash_cs},\\fscx126\\fscy126)}}"
                f"{{\\t({flash_cs},{settle_cs},\\fscx100\\fscy100)}}"
            )
        else:
            pop_tags = ""
        prev_end = w["end"]

        text_parts.append(f"{color_tags}{pop_tags}{w['word']}")

    fad = "{\\fad(60,60)}"
    return f"{fad}{' '.join(text_parts)}"


def build_ass_for_clip(
    words: List[Dict],
    style: Optional[str] = None,
    words_per_group: Optional[int] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> str:
    """Render a complete ASS document for one clip. Returns "" if there are no
    words to caption (callers treat falsy as "no captions")."""
    if not words:
        return ""

    preset = resolve_preset(style)
    fontname = LOCAL_CAPTION_FONT or "Arial"
    fontsize = _resolve_font_size(preset, height)

    active_hex = _ass_color(preset.active_color)
    dim_hex = _ass_color(preset.text_color)
    outline_hex = _ass_color(preset.outline_color)

    margin_l = max(10, int(width * preset.margin_h_ratio))
    margin_r = margin_l
    margin_v = max(10, int(height * preset.margin_v_ratio))
    bold = 1 if preset.bold else 0

    groups = group_words(words, words_per_group)

    lines = [ASS_HEADER.format(
        width=width,
        height=height,
        fontname=fontname,
        fontsize=fontsize,
        outline_hex=outline_hex,
        bold=bold,
        outline=int(preset.outline),
        shadow=int(preset.shadow),
        margin_l=margin_l,
        margin_r=margin_r,
        margin_v=margin_v,
    )]

    for group in groups:
        if not group:
            continue
        start = _ass_time(group[0]["start"])
        end = _ass_time(group[-1]["end"])
        text = _group_event_text(group, preset, active_hex, dim_hex)
        lines.append(f"Dialogue: 0,{start},{end},Caption,,0,0,0,,{text}")

    return "\n".join(lines) + "\n"


def write_ass_file(content: str, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8-sig")
    return path


def ass_filename_for(media_path, purpose: str = "captions") -> Path:
    """`<video>.captions.ass` beside the media file."""
    media = Path(media_path)
    return media.with_name(f"{media.stem}.{purpose}.ass")


def _delete_ass(media_path) -> None:
    try:
        os.remove(ass_filename_for(media_path))
    except OSError:
        pass