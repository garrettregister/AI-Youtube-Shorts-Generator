"""Local transcription via WhisperX (faster-whisper + wav2vec2 word alignment).

Reads a local media file and returns the same shape the highlight generator
expects: {duration, segments[start, end, text]}. Segments additionally carry
a `words` list [{word, start, end}] when word-level timestamps are available,
which the caption burn-in uses for word-for-word synced captions.

Transcription is cached twice, both beside the source file:
  * <stem>.srt        — segment-level cache (same format as before).
  * <stem>.words.json — segment + word-level cache; a fresh cache is reused
                        without loading a model or touching the audio again.
"""
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from ..config import LOCAL_WHISPER_DEVICE, LOCAL_WHISPER_MODEL


def _transcript_cache_path(media_path: str) -> Path:
    """Return the .srt cache path for a media file (kept beside the source)."""
    media = Path(media_path).resolve()
    cache_dir = media.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / (media.stem + ".srt")


def _words_cache_path(media_path: str) -> Path:
    """Return the word-level .words.json cache path (kept beside the .srt)."""
    media = Path(media_path).resolve()
    cache_dir = media.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / (media.stem + ".words.json")


def _cache_fresh(cache_path: Path, media_path: str) -> bool:
    if not cache_path.exists():
        return False
    return cache_path.stat().st_mtime >= os.path.getmtime(media_path)


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _parse_srt_timestamp(value: str) -> float:
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    hours, minutes, seconds, millis = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + (millis / 1000.0)


def _write_srt_cache(media_path: str, transcript: Dict) -> Path:
    cache_path = _transcript_cache_path(media_path)
    lines = []
    for idx, segment in enumerate(transcript.get("segments", []), start=1):
        start = _format_srt_timestamp(float(segment["start"]))
        end = _format_srt_timestamp(float(segment["end"]))
        text = str(segment.get("text", "")).strip().replace("\r", "").replace("\n", " ")
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    cache_path.write_text("\n".join(lines), encoding="utf-8")
    return cache_path


def _load_srt_cache(cache_path: Path) -> Dict:
    content = cache_path.read_text(encoding="utf-8-sig").strip()
    if not content:
        return {"duration": 0.0, "segments": []}

    segments = []
    for block in re.split(r"\n\s*\n", content):
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if "-->" not in lines[0] and len(lines) > 1 and "-->" in lines[1]:
            lines = lines[1:]
        if not lines or "-->" not in lines[0]:
            continue
        start_raw, end_raw = [part.strip() for part in lines[0].split("-->", 1)]
        text = "\n".join(lines[1:]).strip()
        segments.append(
            {
                "start": _parse_srt_timestamp(start_raw),
                "end": _parse_srt_timestamp(end_raw),
                "text": text,
            }
        )

    duration = segments[-1]["end"] if segments else 0.0
    return {"duration": duration, "segments": segments}


def _normalize_words(raw_words: List[Dict]) -> List[Dict]:
    out = []
    for w in raw_words or []:
        if not isinstance(w, dict):
            continue
        word = str(w.get("word", "")).strip()
        start, end = w.get("start"), w.get("end")
        if not word or start is None or end is None:
            continue
        try:
            out.append({"word": word, "start": float(start), "end": float(end)})
        except (TypeError, ValueError):
            continue
    return out


def _normalize_segments(raw_segments: List[Dict]) -> List[Dict]:
    out = []
    for s in raw_segments or []:
        if not isinstance(s, dict):
            continue
        try:
            segment = {
                "start": float(s["start"]),
                "end": float(s["end"]),
                "text": str(s.get("text", "")).strip(),
            }
        except (KeyError, TypeError, ValueError):
            continue
        words = _normalize_words(s.get("words", []) or [])
        if words:
            segment["words"] = words
        out.append(segment)
    return out


def _segments_have_words(segments: List[Dict]) -> bool:
    return any(bool(s.get("words")) for s in segments)


def _write_words_cache(media_path: str, transcript: Dict) -> Path:
    cache_path = _words_cache_path(media_path)
    segments = []
    for s in transcript.get("segments", []):
        segments.append(
            {
                "start": float(s["start"]),
                "end": float(s["end"]),
                "text": str(s.get("text", "")),
                "words": _normalize_words(s.get("words", [])),
            }
        )
    payload = {"duration": float(transcript.get("duration", 0.0)), "segments": segments}
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache_path


def _load_words_cache(cache_path: Path) -> Dict:
    data = json.loads(cache_path.read_text(encoding="utf-8-sig"))
    return {
        "duration": float(data.get("duration", 0.0)),
        "segments": _normalize_segments(data.get("segments", [])),
    }


def _resolve_device() -> str:
    if LOCAL_WHISPER_DEVICE != "auto":
        return LOCAL_WHISPER_DEVICE
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            # Test that CUDA actually works (catches missing cuBLAS/cuDNN libs)
            torch.zeros(1, device="cuda")
            return "cuda"
    except (ImportError, OSError, RuntimeError):
        pass
    return "cpu"


def _import_whisperx():
    try:
        import whisperx  # noqa: F401
        return whisperx
    except ImportError:
        return None


def _vad_options_params() -> Dict:
    """Map LOCAL_WHISPER_VAD_PARAMETERS onto whisperx's silero VAD options."""
    from ..config import LOCAL_WHISPER_VAD_PARAMETERS
    p = LOCAL_WHISPER_VAD_PARAMETERS
    onset = float(p.get("threshold", 0.5))
    return {
        "vad_onset": onset,
        "vad_offset": max(0.0, onset - 0.1),
        "vad_min_speech_len_ms": int(p.get("min_speech_duration_ms", 250)),
        "vad_min_silence_len_ms": int(p.get("min_silence_duration_ms", 2000)),
        "vad_max_speech_len_s": float(p.get("max_speech_duration_s", float("inf"))),
        "vad_speech_pad_ms": int(p.get("speech_pad_ms", 400)),
    }


def _is_segment_like(item) -> bool:
    return isinstance(item, dict) and any(k in item for k in ("start", "text", "words", "word_segments"))


def _extract_aligned_segments(out) -> List[Dict]:
    """Pull the refined segment list out of whisperx.align, whatever shape the
    installed version returns (list of segments, tuple, or dict)."""
    if out is None:
        return []
    if isinstance(out, (list, tuple)):
        for item in out:
            if isinstance(item, dict) and isinstance(item.get("segments"), list):
                return item["segments"]
        for item in out:
            if isinstance(item, list):
                if not item:
                    return []
                if _is_segment_like(item[0]):
                    return item
        return []
    if isinstance(out, dict):
        for key in ("segments", "word_segments"):
            value = out.get(key)
            if isinstance(value, list):
                return value
        if _is_segment_like(out):
            return [out]
    return []


def _try_whisperx(media_path: str, language: Optional[str] = None) -> Optional[Dict]:
    """Transcribe + word-align with WhisperX. Returns None on any failure so
    the caller can fall back to faster-whisper."""
    whisperx = _import_whisperx()
    if whisperx is None:
        return None

    device = _resolve_device()
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[transcribe/local] whisperx model={LOCAL_WHISPER_MODEL} device={device}", flush=True)

    from ..config import LOCAL_WHISPER_VAD_FILTER

    try:
        load_kwargs = {"device": device, "compute_type": compute_type}
        if LOCAL_WHISPER_VAD_FILTER:
            load_kwargs["vad_options"] = _vad_options_params()
        model = whisperx.load_model(LOCAL_WHISPER_MODEL, **load_kwargs)
        audio = whisperx.load_audio(media_path)

        result = model.transcribe(audio, batch_size=8, language=language)
        raw_segments = result.get("segments") or []
        if not raw_segments:
            print("[transcribe/local] whisperx produced no segments", flush=True)
            return None

        segments = _normalize_segments(raw_segments)
        language_code = str(result.get("language") or language or "en")

        if not _segments_have_words(segments):
            try:
                align_model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
                aligned = whisperx.align(
                    raw_segments, align_model, metadata, audio, language_code, device,
                    return_char_alignments=False,
                )
                aligned_segments = _normalize_segments(_extract_aligned_segments(aligned))
                if _segments_have_words(aligned_segments):
                    segments = aligned_segments
            except Exception as e:  # noqa: BLE001
                print(
                    f"[transcribe/local] whisperx word alignment failed ({e}); keeping segment timings",
                    flush=True,
                )

        duration = float(result.get("duration") or 0.0) or (segments[-1]["end"] if segments else 0.0)
        transcript = {"duration": duration, "segments": segments}
        print(f"[transcribe/local] {len(segments)} segments, {duration:.0f}s of audio", flush=True)
        return transcript
    except Exception as e:  # noqa: BLE001
        print(f"[transcribe/local] whisperx failed: {e}", flush=True)
        return None


def _transcribe_with_faster_whisper(
    media_path: str,
    language: Optional[str] = None,
    word_timestamps: bool = False,
) -> Dict:
    """faster-whisper transcribe (the same behaviour as before, plus optional
    word timestamps)."""
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "WhisperX / faster-whisper is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    device = _resolve_device()
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[transcribe/local] faster-whisper model={LOCAL_WHISPER_MODEL} device={device}", flush=True)

    from ..config import LOCAL_WHISPER_VAD_FILTER, LOCAL_WHISPER_VAD_PARAMETERS

    model = WhisperModel(LOCAL_WHISPER_MODEL, device=device, compute_type=compute_type)

    transcribe_kwargs = {
        "audio": media_path,
        "language": language,
        "beam_size": 5,
        "condition_on_previous_text": False,
        "word_timestamps": word_timestamps,
    }
    if LOCAL_WHISPER_VAD_FILTER:
        transcribe_kwargs["vad_filter"] = True
        transcribe_kwargs["vad_parameters"] = LOCAL_WHISPER_VAD_PARAMETERS
    else:
        transcribe_kwargs["vad_filter"] = False

    segments_iter, info = model.transcribe(**transcribe_kwargs)

    segments = []
    for s in segments_iter:
        segment = {
            "start": float(s.start),
            "end": float(s.end),
            "text": (s.text or "").strip(),
        }
        if word_timestamps:
            words = _normalize_words(
                [{"word": w.word, "start": w.start, "end": w.end} for w in (s.words or [])]
            )
            if words:
                segment["words"] = words
        segments.append(segment)

    duration = float(getattr(info, "duration", 0.0)) or (segments[-1]["end"] if segments else 0.0)
    print(f"[transcribe/local] {len(segments)} segments, {duration:.0f}s of audio", flush=True)
    return {"duration": duration, "segments": segments}


def transcribe_local(media_path: str, language: Optional[str] = None) -> Dict:
    """Run WhisperX (falling back to faster-whisper) on a local file path.

    Cache reuse:
      * a fresh `<stem>.words.json` skips the models entirely;
      * a fresh `<stem>.srt` is reused as-is when WhisperX is unavailable
        (old installs keep working), otherwise upgraded to word level once;
      * no cache -> WhisperX transcribe + wav2vec2 alignment, with a
        faster-whisper word-timestamps fallback.
    """
    cache_path = _transcript_cache_path(media_path)
    words_cache_path = _words_cache_path(media_path)

    if _cache_fresh(words_cache_path, media_path):
        cached = _load_words_cache(words_cache_path)
        if cached.get("segments"):
            print(f"[transcribe/local] reusing cached word-level transcript: {words_cache_path}", flush=True)
            return cached
        print(f"[transcribe/local] word cache is empty/invalid, deleting: {words_cache_path}", flush=True)
        words_cache_path.unlink(missing_ok=True)

    base: Optional[Dict] = None
    if _cache_fresh(cache_path, media_path):
        base = _load_srt_cache(cache_path)
        if base.get("segments") and base.get("duration", 0.0) > 0.0:
            print(f"[transcribe/local] reusing cached transcript: {cache_path}", flush=True)
            print("[transcribe/local] upgrading cached transcript with word timestamps (one-time)", flush=True)
        else:
            print(f"[transcribe/local] cache is empty/invalid, deleting: {cache_path}", flush=True)
            cache_path.unlink(missing_ok=True)
            base = None

    if base is not None and _import_whisperx() is None:
        # Old install: keep the previous faster-whisper-only behaviour and
        # reuse the .srt as-is; captions simply fall back to no word data.
        print("[transcribe/local] whisperx is not installed - skipping word-level upgrade", flush=True)
        return base

    transcript = _try_whisperx(media_path, language)
    if transcript is None:
        transcript = _transcribe_with_faster_whisper(media_path, language, word_timestamps=True)

    if not _segments_have_words(transcript.get("segments", [])):
        fallback = _transcribe_with_faster_whisper(media_path, language, word_timestamps=True)
        if fallback is not None and _segments_have_words(fallback.get("segments", [])):
            transcript = fallback

    cache_path = _write_srt_cache(media_path, transcript)
    print(f"[transcribe/local] wrote cache: {cache_path}", flush=True)

    if _segments_have_words(transcript.get("segments", [])):
        words_cache_path = _write_words_cache(media_path, transcript)
        print(f"[transcribe/local] wrote word-level cache: {words_cache_path}", flush=True)

    return transcript