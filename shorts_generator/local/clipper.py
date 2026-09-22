"""Local clipping: ffmpeg subclip + OpenCV face-aware vertical crop.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
  2. Reframe the cut to the target aspect ratio. For 9:16 we slide a vertical
     window horizontally across the frame to keep faces centred (Haar
     cascade — same approach as the original repo, no external models).
"""
import os
import subprocess
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from ..config import LOCAL_OUTPUT_DIR


def _load_face_cascade(cv2) -> Optional[object]:
    """Return a loaded Haar cascade, or None when the build lacks it.

    OpenCV 5 removed the objdetect module, so `cv2.CascadeClassifier` and
    `cv2.data.haarcascades` no longer exist. When unavailable we leave the
    crop centred (no face tracking) instead of crashing the clip.
    """
    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception:
        return None
    if cascade is None or cascade.empty():
        return None
    return cascade


def _remove_quietly(path: str, attempts: int = 5, delay: float = 0.2) -> None:
    """Delete a temp file without ever raising or masking a real error.

    An outer process can transiently hold the file (a still-open OpenCV
    capture, an AV scan lock), which on Windows turns a cosmetic cleanup into
    `WinError 32` that masks the actual clip failure. Retry briefly, then give
    up with a warning so only the real result decides pass/fail.
    """
    last_error: Optional[OSError] = None
    for _ in range(attempts):
        try:
            os.remove(path)
            return
        except FileNotFoundError:
            return
        except OSError as e:
            last_error = e
            time.sleep(delay)
    if last_error is not None:
        print(f"[clip/local] warn: could not remove temp file: {path}: {last_error}", flush=True)


def _ratio(aspect_ratio: str) -> float:
    """Parse '9:16' → 9/16, '1:1' → 1.0."""
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    """ffmpeg -ss start -to end → re-encoded mp4 with audio."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", source_path,
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _reframe_vertical(
    in_path: str,
    out_path: str,
    aspect_ratio: str,
    silent_path: Optional[str] = None,
) -> str:
    """Crop the cut clip to the target aspect ratio, tracking faces if possible."""
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "opencv-python is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    target_ratio = _ratio(aspect_ratio)
    silent_path = silent_path or (out_path + ".silent.mp4")

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    writer = None
    try:
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Compute the largest crop that fits inside the frame at the target ratio.
        if target_ratio < src_w / src_h:
            crop_h = src_h
            crop_w = int(crop_h * target_ratio)
        else:
            crop_w = src_w
            crop_h = int(crop_w / target_ratio)
        crop_w = max(2, crop_w - (crop_w % 2))
        crop_h = max(2, crop_h - (crop_h % 2))

        face_cascade = _load_face_cascade(cv2)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(silent_path, fourcc, fps, (crop_w, crop_h))
        if writer is None or not writer.isOpened():
            raise RuntimeError(
                f"OpenCV could not create a video writer for {silent_path} "
                "(mp4v encoder unavailable?)"
            )

        last_center: Optional[Tuple[int, int]] = None
        smoothing = 0.15  # how aggressively to chase a new face position
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if face_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
                if len(faces) > 0:
                    # Pick the largest face — usually the speaker.
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    cx = x + w // 2
                    cy = y + h // 2
                    if last_center is None:
                        last_center = (cx, cy)
                    else:
                        lx, ly = last_center
                        last_center = (
                            int(lx + (cx - lx) * smoothing),
                            int(ly + (cy - ly) * smoothing),
                        )
            if last_center is None:
                last_center = (src_w // 2, src_h // 2)

            cx, cy = last_center
            x0 = max(0, min(src_w - crop_w, cx - crop_w // 2))
            y0 = max(0, min(src_h - crop_h, cy - crop_h // 2))
            cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
            writer.write(cropped)
    finally:
        # Always release both handles: a leaked VideoCapture keeps its input
        # file locked on Windows, which later turns an unrelated exception
        # into a misleading "file is being used by another process".
        cap.release()
        if writer is not None:
            writer.release()

    # Mux audio from the cut clip back onto the silent reframed video.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", silent_path,
        "-i", in_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _probe_size(media_path: str) -> Optional[Tuple[int, int]]:
    """Return (width, height) of a media file via ffprobe, or None on failure."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=s=x:p=0",
                media_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        parts = [int(p) for p in out.stdout.strip().split("x") if p.strip().isdigit()]
        if len(parts) == 2:
            return (parts[0], parts[1])
    except (ValueError, subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None


@lru_cache(maxsize=1)
def _ffmpeg_has_libass() -> bool:
    """True when ffmpeg was built with the `ass` subtitle filter (libass)."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", "filter=ass"],
            capture_output=True,
            text=True,
            check=False,
        )
        return "Filter ass" in (out.stdout + out.stderr)
    except (FileNotFoundError, OSError):
        return False


def _ass_filter_arg(ass_path: str) -> str:
    """Build a filtergraph-safe `ass=filename=...` argument for a Windows path.

    ffmpeg escapes in two passes: a generic pass turns ``\\x`` into ``x``, then
    the option pass splits ``=``/``:``/``,``/``;`` and honours remaining
    backslash escapes. A single ``\\:`` is therefore consumed by the generic
    pass and the drive letter's ``:`` still splits the option — the escape
    backslash itself must be doubled so the option pass sees ``\\:``.
    """
    safe = ass_path.replace("\\", "/")
    for special in ("\\", ":", ",", ";", "'", "[", "]", "#"):
        safe = safe.replace(special, "\\" + special)
    safe = safe.replace("\\", "\\\\")
    return f"ass=filename={safe}"


def _burn_captions(in_path: str, out_path: str, ass_path: str, silent_path: str) -> Optional[str]:
    """Re-encode `in_path` with the ASS captions burned in. Returns the output
    path on success, or None (without touching `out_path`) when ffmpeg lacks
    libass so the plain clip can still be produced."""
    if not _ffmpeg_has_libass():
        print(
            "[clip/local] warn: ffmpeg was not built with libass - skipping captions "
            "(clip still produced)",
            flush=True,
        )
        return None

    _ = silent_path  # reserved for a future sidecar pass
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", in_path,
        "-vf", _ass_filter_arg(ass_path),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "copy",
        "-map", "0:v:0", "-map", "0:a:0?",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[clip/local] caption burn failed: {e} (keeping plain clip)", flush=True)
        return None
    return out_path


def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: Optional[str],
    out_path: str,
    *,
    words: Optional[List[Dict]] = None,
    caption_style: Optional[str] = None,
    captions: bool = True,
) -> str:
    """Cut (and optionally reframe) one highlight, returning the local mp4 path.

    When `aspect_ratio` is None the sliced clip is kept as-is, preserving the
    source video's native resolution and aspect ratio. When a ratio is given
    (e.g. "9:16") the cut is reframed to it with face tracking.

    When `captions` is True and `words` are supplied, an ASS subtitle is built
    from the word timestamps and burned in with ffmpeg's `ass` filter as a
    second pass. If ffmpeg lacks libass the clip is still produced — just
    without captions.

    Intermediates live in a `.tmp` subfolder beside the output so the
    deliverable directory never shows `short_NN.mp4.cut.mp4` leftovers, and
    cleanup is best-effort — it retries and warns instead of ever masking the
    real clip result with a file-lock error.
    """
    out_dir = os.path.dirname(os.path.abspath(out_path))
    tmp_dir = os.path.join(out_dir, ".tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(out_path))[0]
    cut_path = os.path.join(tmp_dir, base + ".cut.mp4")
    reframed_path = os.path.join(tmp_dir, base + ".reframed.mp4")
    silent_path = os.path.join(tmp_dir, base + ".silent.mp4")
    ass_path = os.path.join(tmp_dir, base + ".captions.ass")

    caption_words = words if captions else None

    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        if aspect_ratio is None:
            staged = cut_path
        else:
            staged = reframed_path
            _reframe_vertical(cut_path, staged, aspect_ratio, silent_path=silent_path)

        if caption_words:
            from .captions import build_ass_for_clip, write_ass_file

            size = _probe_size(staged)
            width, height = size if size else (720, 1280)
            ass_content = build_ass_for_clip(
                caption_words, style=caption_style, width=width, height=height
            )
            if ass_content:
                write_ass_file(ass_content, ass_path)
                burned = _burn_captions(staged, out_path, ass_path, silent_path)
                if burned is not None:
                    return burned
                # burn failed (no libass or ffmpeg error) -> fall through to
                # the plain staged clip so a short is still produced
            else:
                print("[clip/local] no word timestamps for clip - no captions to burn", flush=True)

        os.replace(staged, out_path)
    finally:
        for p in (cut_path, reframed_path, silent_path, ass_path):
            _remove_quietly(p)
    return out_path


def _short_output_path(source_path: str, out_dir: str, index: int) -> str:
    """Name a rendered short after its source video, e.g.
    `<title>_source_<id>_short_01.mp4`."""
    stem = os.path.splitext(os.path.basename(source_path))[0]
    return os.path.join(out_dir, f"{stem}_short_{index:02d}.mp4")


def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: Optional[str] = None,
    out_dir: Optional[str] = None,
    *,
    transcript: Optional[Dict] = None,
    caption_style: Optional[str] = None,
    captions: bool = True,
) -> List[Dict]:
    # Default to the source video's own folder so a YouTube download renders
    # each video's shorts beside it, under `output/<video title>/`.
    out_dir = os.path.abspath(
        out_dir or os.path.dirname(os.path.abspath(source_path)) or LOCAL_OUTPUT_DIR
    )
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
    for i, h in enumerate(highlights, 1):
        out_path = _short_output_path(source_path, out_dir, i)
        print(f"[clip/local] {i}/{len(highlights)}: {h.get('title', '(untitled)')} -> {out_path}", flush=True)
        try:
            words = None
            if captions and transcript:
                from .captions import slice_words_for_clip

                words = slice_words_for_clip(
                    transcript,
                    float(h["start_time"]),
                    float(h["end_time"]),
                )
            crop_clip_local(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio,
                out_path,
                words=words,
                caption_style=caption_style,
                captions=captions,
            )
            results.append({**h, "clip_url": out_path})
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
