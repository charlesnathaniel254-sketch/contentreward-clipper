"""
processor.py — Video clipping pipeline for ContentReward Clipper.

Pipeline
--------
1. Download   : ``yt-dlp`` grabs only the requested section of a video
                (YouTube or any other yt-dlp-supported site).
2. Transcribe : OpenAI ``whisper`` produces *word-level* timestamps
                (``word_timestamps=True``).
3. Captions   : Word timestamps are turned into a dynamic, TikTok-style
                ASS subtitle file — words are grouped into short chunks and
                the word currently being spoken is highlighted / popped,
                karaoke-style.
4. Render     : ``FFmpeg`` crops/pads the clip to vertical 9:16
                (1080x1920) and burns the captions into the video.

Heavy dependencies (``yt_dlp``, ``whisper``/torch) are imported lazily so
this module — and therefore the FastAPI app — can start and be inspected
even before those packages are installed. A clear error is raised only
when a stage that actually needs the missing package is executed.

Standalone usage::

    python processor.py "https://youtube.com/watch?v=..." \
        --start 00:01:00 --end 00:01:45 --style tiktok --mode blur
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

# ---------------------------------------------------------------------------
# Types & constants
# ---------------------------------------------------------------------------

TimeLike = Union[str, int, float, None]
ProgressCallback = Callable[[str, int, str], None]
"""(stage, percent 0-100, human-readable message)"""

OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920

# Caption style presets. Colours are ASS format: &HAABBGGRR (alpha, blue,
# green, red) — so pure yellow is &H0000FFFF, white is &H00FFFFFF.
CAPTION_STYLES: dict[str, dict] = {
    "tiktok": {
        "label": "TikTok (yellow highlight)",
        "font": "Arial",
        "font_size": 84,
        "primary": "&H00FFFFFF",      # upcoming / normal words: white
        "highlight": "&H0000FFFF",    # spoken word: yellow
        "outline": "&H00000000",
        "back": "&H64000000",
        "bold": True,
        "outline_width": 6,
        "max_words": 3,
        "pop": True,
    },
    "karaoke": {
        "label": "Karaoke (green highlight)",
        "font": "Arial",
        "font_size": 84,
        "primary": "&H00FFFFFF",
        "highlight": "&H0000FF00",    # spoken word: green
        "outline": "&H00000000",
        "back": "&H64000000",
        "bold": True,
        "outline_width": 6,
        "max_words": 3,
        "pop": True,
    },
    "boxed": {
        "label": "Boxed (black bar, white text)",
        "font": "Arial",
        "font_size": 76,
        "primary": "&H00FFFFFF",
        "highlight": "&H0000FFFF",
        "outline": "&H00000000",
        "back": "&HCC000000",         # near-opaque black box behind text
        "bold": True,
        "outline_width": 4,
        "max_words": 4,
        "pop": False,
    },
    "minimal": {
        "label": "Minimal (clean white, no highlight)",
        "font": "Arial",
        "font_size": 68,
        "primary": "&H00FFFFFF",
        "highlight": "&H00FFFFFF",
        "outline": "&H00000000",
        "back": "&H50000000",
        "bold": False,
        "outline_width": 4,
        "max_words": 4,
        "pop": False,
    },
}

FILL_MODES = ("crop", "blur")


class ProcessingError(RuntimeError):
    """Raised when any pipeline stage fails."""


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class ClipResult:
    job_id: str
    source_url: str
    output_path: Path
    subtitle_path: Optional[Path]
    words: list[Word] = field(default_factory=list)
    duration: float = 0.0


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def parse_timestamp(value: TimeLike, *, field_name: str = "time") -> Optional[float]:
    """Parse ``SS``, ``MM:SS`` or ``HH:MM:SS(.ms)`` (or a number) to seconds."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        if value < 0:
            raise ProcessingError(f"{field_name} cannot be negative.")
        return float(value)

    text = str(value).strip()
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)

    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise ProcessingError(
            f"Invalid {field_name} {value!r}. Use seconds (90), MM:SS or HH:MM:SS."
        )
    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        raise ProcessingError(
            f"Invalid {field_name} {value!r}. Use seconds (90), MM:SS or HH:MM:SS."
        ) from None
    if any(n < 0 for n in numbers):
        raise ProcessingError(f"{field_name} cannot be negative.")
    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes * 60 + seconds
    hours, minutes, seconds = numbers
    return hours * 3600 + minutes * 60 + seconds


def _ass_time(seconds: float) -> str:
    """Format seconds as an ASS timestamp: H:MM:SS.cc"""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis == 100:  # rounding edge case
        centis = 0
        secs += 1
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _ffmpeg_time(seconds: float) -> str:
    return f"{seconds:.3f}"


# ---------------------------------------------------------------------------
# ASS (dynamic TikTok-style) caption generation
# ---------------------------------------------------------------------------

def _escape_ass_text(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
        .strip()
    )


def chunk_words(words: Iterable[Word], max_words: int) -> Iterable[list[Word]]:
    """Group consecutive words into short caption chunks.

    A chunk is also broken when there is a long pause (> 0.8s) between
    words, so captions don't linger through silences.
    """
    chunk: list[Word] = []
    for word in words:
        if chunk and (
            len(chunk) >= max_words or word.start - chunk[-1].end > 0.8
        ):
            yield chunk
            chunk = []
        chunk.append(word)
    if chunk:
        yield chunk


def generate_ass_subtitles(
    words: list[Word],
    output_path: Path,
    style: str = "tiktok",
) -> Path:
    """Write a dynamic ASS file: one event per word, active word highlighted.

    For every spoken word we emit a dialogue event lasting exactly that
    word, showing its whole chunk with only the active word coloured (and
    optionally scale-"popped"). That is what gives the familiar TikTok /
    karaoke word-by-word effect once burned in with FFmpeg's ``ass`` filter.
    """
    preset = CAPTION_STYLES.get(style, CAPTION_STYLES["tiktok"])

    header = f"""[Script Info]
Title: ContentReward Clipper captions
ScriptType: v4.00+
PlayResX: {OUTPUT_WIDTH}
PlayResY: {OUTPUT_HEIGHT}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{preset['font']},{preset['font_size']},{preset['primary']},{preset['highlight']},{preset['outline']},{preset['back']},{-1 if preset['bold'] else 0},0,0,0,100,100,0,0,1,{preset['outline_width']},2,2,60,60,420,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines: list[str] = []
    for chunk in chunk_words(words, preset["max_words"]):
        for index, active in enumerate(chunk):
            rendered: list[str] = []
            for position, word in enumerate(chunk):
                text = _escape_ass_text(word.text)
                if position == index:
                    overrides = f"\\c{preset['highlight']}"
                    if preset["pop"]:
                        # Quick 110% -> 100% scale pop on the spoken word.
                        overrides += r"\fscx112\fscy112\t(0,120,\fscx100\fscy100)"
                    rendered.append("{" + overrides + "}" + text)
                else:
                    rendered.append(
                        "{\\c" + preset["primary"] + "}" + text
                    )
            dialogue = (
                f"Dialogue: 0,{_ass_time(active.start)},{_ass_time(active.end)},"
                f"Caption,,0,0,0,,{' '.join(rendered)}"
            )
            lines.append(dialogue)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class ClipProcessor:
    """End-to-end clip pipeline: download -> transcribe -> caption -> render."""

    def __init__(
        self,
        output_dir: Union[str, Path] = "outputs",
        work_dir: Union[str, Path] = "downloads",
        whisper_model: str = "base",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.work_dir = Path(work_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.whisper_model = whisper_model
        self._model = None  # lazily loaded whisper model (expensive)

    # -- infrastructure -----------------------------------------------------

    @staticmethod
    def _report(
        callback: Optional[ProgressCallback], stage: str, percent: int, msg: str
    ) -> None:
        if callback is not None:
            callback(stage, max(0, min(100, percent)), msg)

    @staticmethod
    def _require_ffmpeg() -> str:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise ProcessingError(
                "FFmpeg was not found on PATH. Install it first "
                "(macOS: `brew install ffmpeg`, Ubuntu: `sudo apt install ffmpeg`, "
                "Windows: https://ffmpeg.org/download.html) and try again."
            )
        return ffmpeg

    # -- stage 1: download ----------------------------------------------------

    def download_clip(
        self,
        url: str,
        start: Optional[float],
        end: Optional[float],
        job_id: str,
        callback: Optional[ProgressCallback] = None,
    ) -> Path:
        """Download (only the requested section of) a video with yt-dlp."""
        try:
            import yt_dlp
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ProcessingError(
                "yt-dlp is not installed. Run: pip install -r requirements.txt"
            ) from exc

        if not re.match(r"^https?://", url or ""):
            raise ProcessingError("Please provide a valid http(s) video URL.")
        if start is not None and end is not None and end <= start:
            raise ProcessingError("End time must be after the start time.")

        self._report(callback, "downloading", 5, "Starting download with yt-dlp...")

        def progress_hook(status: dict) -> None:
            if status.get("status") != "downloading":
                return
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            downloaded = status.get("downloaded_bytes", 0)
            if total:
                # Download occupies 5% -> 35% of overall progress.
                percent = 5 + int(30 * downloaded / total)
                self._report(
                    callback, "downloading", percent,
                    f"Downloading... {downloaded / total:.0%}",
                )

        options: dict = {
            # Cap at 1080p to keep transcription/render fast; still plenty
            # for a 1080x1920 vertical crop.
            "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
            "merge_output_format": "mp4",
            "outtmpl": str(self.work_dir / f"{job_id}_source.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "progress_hooks": [progress_hook],
        }
        if start is not None or end is not None:
            # Only fetch the section we need — much faster for long videos.
            section_start = start or 0.0
            section_end = end if end is not None else float("inf")
            options["download_ranges"] = (
                lambda info, ydl, s=section_start, e=section_end: [
                    {"start_time": s, "end_time": e}
                ]
            )
            options["force_keyframes_at_cuts"] = True

        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
        except Exception as exc:  # yt-dlp raises many exception types
            raise ProcessingError(f"Download failed: {exc}") from exc

        path = Path(filename)
        if not path.exists():
            # merge_output_format may change the extension to .mp4
            candidates = sorted(self.work_dir.glob(f"{job_id}_source.*"))
            if not candidates:
                raise ProcessingError("Download finished but no file was produced.")
            path = candidates[0]

        self._report(callback, "downloading", 35, "Download complete.")
        return path

    # -- stage 2: transcribe --------------------------------------------------

    def transcribe(
        self, video_path: Path, callback: Optional[ProgressCallback] = None
    ) -> list[Word]:
        """Transcribe with Whisper and return flat word-level timestamps."""
        try:
            import whisper
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ProcessingError(
                "openai-whisper is not installed. Run: pip install -r requirements.txt"
                " (note: it also requires PyTorch and FFmpeg)."
            ) from exc

        self._report(
            callback, "transcribing", 40,
            f"Loading Whisper model '{self.whisper_model}'...",
        )
        if self._model is None:
            self._model = whisper.load_model(self.whisper_model)

        self._report(callback, "transcribing", 50, "Transcribing with Whisper...")
        result = self._model.transcribe(
            str(video_path),
            word_timestamps=True,   # <-- word-level timestamps
            fp16=False,             # safe on CPU-only machines
            verbose=False,
        )

        words: list[Word] = []
        for segment in result.get("segments", []):
            for raw in segment.get("words", []) or []:
                text = (raw.get("word") or "").strip()
                if not text:
                    continue
                words.append(
                    Word(
                        text=text,
                        start=float(raw["start"]),
                        end=max(float(raw["end"]), float(raw["start"]) + 0.05),
                    )
                )
        words.sort(key=lambda w: w.start)
        self._report(
            callback, "transcribing", 65, f"Transcribed {len(words)} words."
        )
        return words

    # -- stage 3+4: captions + render -----------------------------------------

    def render(
        self,
        source_path: Path,
        words: list[Word],
        job_id: str,
        *,
        caption_style: str = "tiktok",
        fill_mode: str = "blur",
        burn_captions: bool = True,
        callback: Optional[ProgressCallback] = None,
    ) -> tuple[Path, Optional[Path]]:
        """Crop/pad to 9:16 (1080x1920) and burn captions in with FFmpeg."""
        ffmpeg = self._require_ffmpeg()
        if fill_mode not in FILL_MODES:
            raise ProcessingError(
                f"Unknown fill mode {fill_mode!r}; choose one of {FILL_MODES}."
            )

        subtitle_path: Optional[Path] = None
        if burn_captions and words:
            subtitle_path = self.output_dir / f"{job_id}.ass"
            generate_ass_subtitles(words, subtitle_path, style=caption_style)

        # Absolute paths: FFmpeg may run with cwd=output_dir (for the ass
        # filter), which would otherwise break relative input/output paths.
        source_path = Path(source_path).resolve()
        output_path = (self.output_dir / f"{job_id}.mp4").resolve()

        if fill_mode == "crop":
            # Centre-crop to 9:16, then scale to exactly 1080x1920.
            video_filter = (
                f"crop=w='min(iw,ih*9/16)':h='min(ih,iw*16/9)'"
                f":x='(iw-ow)/2':y='(ih-oh)/2',"
                f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},setsar=1"
            )
        else:
            # Blurred-background ("blur pad"): the full frame sits centred
            # over a blurred, darkened, 9:16-filled copy of itself. Nothing
            # is cropped away — the classic look for landscape -> Shorts.
            video_filter = (
                f"split=2[bg][fg];"
                f"[bg]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}"
                f":force_original_aspect_ratio=increase,"
                f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},"
                f"boxblur=20:5,eq=brightness=-0.08[bgb];"
                f"[fg]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}"
                f":force_original_aspect_ratio=decrease[fgs];"
                f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1"
            )

        # Burn captions in. Use the bare filename + cwd=output_dir so the
        # `ass` filter needs minimal path escaping (absolute paths with
        # colons break FFmpeg filter parsing).
        ass_filter = ""
        if subtitle_path is not None:
            ass_name = subtitle_path.name.replace("'", r"\'")
            ass_filter = f"ass='{ass_name}'"

        self._report(callback, "rendering", 75, "Rendering 9:16 video with FFmpeg...")
        command = [ffmpeg, "-y", "-i", str(source_path)]
        if fill_mode == "crop":
            if ass_filter:
                video_filter += f",{ass_filter}"
            command += ["-vf", video_filter]
        else:
            # filter_complex graph, terminated with an explicit [vout] label.
            graph = video_filter
            if ass_filter:
                graph += f"[v];[v]{ass_filter}[vout]"
            else:
                graph += "[vout]"
            command += [
                "-filter_complex", graph,
                "-map", "[vout]", "-map", "0:a?",
            ]
        command += [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart",
            str(output_path),
        ]

        completed = subprocess.run(
            command,
            cwd=str(self.output_dir) if subtitle_path is not None else None,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not output_path.exists():
            tail = (completed.stderr or "").strip().splitlines()[-8:]
            raise ProcessingError(
                "FFmpeg render failed:\n" + "\n".join(tail)
            )

        self._report(callback, "rendering", 95, "Render complete.")
        return output_path, subtitle_path

    # -- full pipeline ----------------------------------------------------------

    def process(
        self,
        url: str,
        start_time: TimeLike = 0,
        end_time: TimeLike = None,
        duration: TimeLike = None,
        *,
        caption_style: str = "tiktok",
        fill_mode: str = "blur",
        burn_captions: bool = True,
        job_id: Optional[str] = None,
        callback: Optional[ProgressCallback] = None,
    ) -> ClipResult:
        """Run the whole pipeline and return a :class:`ClipResult`."""
        job_id = job_id or uuid.uuid4().hex[:12]
        start = parse_timestamp(start_time, field_name="start time") or 0.0
        end = parse_timestamp(end_time, field_name="end time")
        if end is None and duration is not None:
            dur = parse_timestamp(duration, field_name="duration")
            if dur is not None:
                end = start + dur
        if end is not None and end <= start:
            raise ProcessingError("End time must be after the start time.")
        if caption_style not in CAPTION_STYLES:
            raise ProcessingError(
                f"Unknown caption style {caption_style!r}; "
                f"choose one of {sorted(CAPTION_STYLES)}."
            )

        source_path = self.download_clip(url, start, end, job_id, callback)
        try:
            words = self.transcribe(source_path, callback) if burn_captions else []
            if burn_captions and not words:
                self._report(
                    callback, "transcribing", 65,
                    "No speech detected — rendering without captions.",
                )
            output_path, subtitle_path = self.render(
                source_path,
                words,
                job_id,
                caption_style=caption_style,
                fill_mode=fill_mode,
                burn_captions=burn_captions,
                callback=callback,
            )
        finally:
            # Keep the downloaded source out of the outputs folder; remove
            # the (potentially large) intermediate file once rendered.
            try:
                source_path.unlink(missing_ok=True)
            except OSError:
                pass

        clip_duration = (end - start) if end is not None else 0.0
        self._report(callback, "done", 100, "Clip ready.")
        return ClipResult(
            job_id=job_id,
            source_url=url,
            output_path=output_path,
            subtitle_path=subtitle_path,
            words=words,
            duration=clip_duration,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a captioned 9:16 clip from a video URL."
    )
    parser.add_argument("url", help="Video URL (YouTube or any yt-dlp site)")
    parser.add_argument("--start", default="0", help="Start time: SS, MM:SS or HH:MM:SS")
    parser.add_argument("--end", default=None, help="End time: SS, MM:SS or HH:MM:SS")
    parser.add_argument("--duration", default=None, help="Clip length in seconds (if no --end)")
    parser.add_argument("--model", default="base", help="Whisper model: tiny|base|small|medium|large")
    parser.add_argument("--style", default="tiktok", choices=sorted(CAPTION_STYLES))
    parser.add_argument("--mode", default="blur", choices=list(FILL_MODES), dest="fill_mode",
                        help="crop = centre-crop to 9:16, blur = blurred background pad")
    parser.add_argument("--no-captions", action="store_true", help="Skip transcription/captions")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--work-dir", default="downloads")
    args = parser.parse_args(argv)

    def report(stage: str, percent: int, message: str) -> None:
        print(f"[{percent:3d}%] {stage}: {message}", flush=True)

    processor = ClipProcessor(
        output_dir=args.output_dir, work_dir=args.work_dir, whisper_model=args.model
    )
    try:
        result = processor.process(
            args.url,
            start_time=args.start,
            end_time=args.end,
            duration=args.duration,
            caption_style=args.style,
            fill_mode=args.fill_mode,
            burn_captions=not args.no_captions,
            callback=report,
        )
    except ProcessingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nDone! Clip saved to: {result.output_path}")
    if result.subtitle_path:
        print(f"Captions saved to: {result.subtitle_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
