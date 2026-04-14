"""
Post-processing step: burn karaoke-style synced lyrics onto a finished video.

Uses OpenAI Whisper (local) to detect when each word is sung,
then generates an ASS subtitle file with karaoke timing where
the active word turns red, then ffmpeg burns it onto the video.

Usage:
    python -m modules.captioner                        # caption today's output
    python -m modules.captioner --date 2026-04-12      # caption a specific date
    python -m modules.captioner --input video.mp4 --audio song.mp3 --lyrics lyrics.txt
"""

from __future__ import annotations
import asyncio
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

_FONT_ABS = Path(__file__).parent.parent / "assets" / "fonts" / "Roboto-Bold.ttf"

FONT_SIZE = 44
WATERMARK_TEXT = "@currentnoise"


@dataclass
class TimedWord:
    word: str
    start: float
    end: float


@dataclass
class TimedLine:
    text: str
    start: float
    end: float
    words: list[TimedWord] = field(default_factory=list)


def transcribe_audio(audio_path: Path, model_name: str = "base") -> list[dict]:
    """Run Whisper on the audio and return word-level segments."""
    import whisper

    log.info(f"[caption] Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)

    log.info(f"[caption] Transcribing {audio_path}...")
    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language="en",
    )

    words = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            words.append({
                "word": word["word"].strip(),
                "start": word["start"],
                "end": word["end"],
            })

    log.info(f"[caption] Transcribed {len(words)} words")
    return words


def parse_lyrics_lines(lyrics_path: Path) -> list[str]:
    """Extract clean lyric lines from lyrics.txt (skip tags, title, caption)."""
    text = lyrics_path.read_text(encoding="utf-8")
    body = text.split("---")[0] if "---" in text else text

    lines = []
    for line in body.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("TITLE:"):
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        lines.append(line)

    return lines


def match_lyrics_to_timestamps(
    lyric_lines: list[str],
    whisper_words: list[dict],
) -> list[TimedLine]:
    """Match each lyric line to word-level timestamps from Whisper.

    Approach: find each line's most distinctive word(s) in the whisper transcript,
    use that as an anchor, then build word timing around it. Lines are processed
    in order and each whisper word can only be used once.
    """
    if not whisper_words or not lyric_lines:
        return []

    total_duration = whisper_words[-1]["end"]
    n = len(lyric_lines)
    avg_duration = total_duration / n

    # Build a simple lookup: for each cleaned whisper word, store its indices
    whisper_lookup: dict[str, list[int]] = {}
    for wi, ww in enumerate(whisper_words):
        clean = ww["word"].lower().strip(".,!?;:'\"")
        whisper_lookup.setdefault(clean, []).append(wi)

    used_indices: set[int] = set()
    timed_lines = []

    for i, lyric_line in enumerate(lyric_lines):
        estimated_start = i * avg_duration
        lyric_words = lyric_line.split()
        if not lyric_words:
            continue

        word_count = len(lyric_words)

        # Find the best anchor: a distinctive lyric word that appears in whisper
        # Prefer longer/rarer words as anchors (more reliable match)
        best_anchor_wi = None
        best_anchor_dist = float("inf")

        # Try each lyric word as a potential anchor, prefer longer words
        candidates = sorted(range(word_count), key=lambda j: -len(lyric_words[j]))
        for j in candidates:
            lw_clean = lyric_words[j].lower().strip(".,!?;:'\"&")
            if len(lw_clean) < 3:
                continue

            # Look for this word in whisper transcript
            for match_key, indices in whisper_lookup.items():
                if match_key == lw_clean or (len(lw_clean) >= 4 and (lw_clean in match_key or match_key in lw_clean)):
                    for wi in indices:
                        if wi in used_indices:
                            continue
                        # The anchor whisper word should be near the estimated position
                        # Adjust estimated position for this word's offset in the line
                        est_word_time = estimated_start + (j / max(word_count, 1)) * avg_duration
                        dist = abs(whisper_words[wi]["start"] - est_word_time)
                        if dist < best_anchor_dist and dist < avg_duration * 1.5:
                            best_anchor_dist = dist
                            # The line starts at whisper index (wi - j)
                            best_anchor_wi = max(0, wi - j)

            if best_anchor_wi is not None and best_anchor_dist < avg_duration * 0.5:
                break  # good enough match

        # Build word-level timing
        timed_words = []
        if best_anchor_wi is not None:
            start_time = whisper_words[best_anchor_wi]["start"]
            for j, lw in enumerate(lyric_words):
                wi = best_anchor_wi + j
                if wi < len(whisper_words) and wi not in used_indices:
                    timed_words.append(TimedWord(
                        word=lw,
                        start=whisper_words[wi]["start"],
                        end=whisper_words[wi]["end"],
                    ))
                    used_indices.add(wi)
                else:
                    prev_end = timed_words[-1].end if timed_words else start_time
                    remaining = word_count - j
                    est_per_word = max(0.3, (avg_duration - (prev_end - start_time)) / max(remaining, 1))
                    timed_words.append(TimedWord(
                        word=lw,
                        start=prev_end + 0.05,
                        end=prev_end + 0.05 + est_per_word,
                    ))
        else:
            # No match — distribute evenly
            start_time = estimated_start
            word_dur = avg_duration / word_count
            for j, lw in enumerate(lyric_words):
                ws = start_time + j * word_dur
                timed_words.append(TimedWord(word=lw, start=ws, end=ws + word_dur))

        end_time = start_time + avg_duration

        timed_lines.append(TimedLine(
            text=lyric_line,
            start=start_time,
            end=end_time,
            words=timed_words,
        ))

    # Sort by start time (in case anchor matching produced out-of-order lines)
    timed_lines.sort(key=lambda tl: tl.start)

    # Enforce strictly sequential word timing across all lines
    # Each word must start after the previous word ends
    all_words = []
    for tl in timed_lines:
        all_words.extend(tl.words)

    for i in range(1, len(all_words)):
        prev = all_words[i - 1]
        curr = all_words[i]
        # Current word must start after previous word ends
        if curr.start < prev.end:
            curr.start = prev.end + 0.01
        # Ensure word has some duration
        if curr.end <= curr.start:
            curr.end = curr.start + 0.2

    # Line start = first word's start, line end = last word's end
    for tl in timed_lines:
        if tl.words:
            tl.start = tl.words[0].start
            tl.end = tl.words[-1].end

    # Ensure minimum display time
    for tl in timed_lines:
        if tl.end - tl.start < 1.0:
            tl.end = tl.start + 1.0

    return timed_lines


def _ass_timestamp(seconds: float) -> str:
    """Convert seconds to ASS timestamp format: H:MM:SS.cc"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _clean_word(word: str) -> str:
    """Clean a word for ASS subtitle display."""
    w = word.encode("ascii", "ignore").decode("ascii")
    # Escape ASS special chars
    w = w.replace("\\", "\\\\")
    w = w.replace("{", "\\{").replace("}", "\\}")
    return w


def generate_ass_subtitles(
    timed_lines: list[TimedLine],
    output_path: Path,
    video_width: int = 1080,
    video_height: int = 1920,
) -> Path:
    """Generate an ASS subtitle file with karaoke word highlighting.

    Each word starts white and turns red when sung.
    """
    font_name = "Roboto Bold"
    margin_v = video_height // 2 - FONT_SIZE

    # ASS header — two lyric styles: white base, red highlight
    header = f"""[Script Info]
Title: Karaoke Lyrics
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: LyricsWhite,{font_name},{FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,5,40,40,{margin_v},1
Style: LyricsRed,{font_name},{FONT_SIZE},&H000000FF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,5,40,40,{margin_v},1
Style: Watermark,{font_name},36,&H99FFFFFF,&H99FFFFFF,&H66000000,&H66000000,-1,0,0,0,100,100,0,0,1,2,1,3,0,30,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    # Watermark (always visible)
    wm_start = _ass_timestamp(0)
    wm_end = _ass_timestamp(timed_lines[-1].end + 5 if timed_lines else 60)
    wm_clean = WATERMARK_TEXT.encode("ascii", "ignore").decode("ascii")
    events = f"Dialogue: 0,{wm_start},{wm_end},Watermark,,0,0,0,,{wm_clean}\n"

    # For each line: one white base layer + one red overlay per word
    for tl in timed_lines:
        line_start = _ass_timestamp(tl.start)
        line_end = _ass_timestamp(tl.end)

        # Clean full line text
        full_text = " ".join(_clean_word(w.word) for w in tl.words)

        # Layer 1: full line in white (visible entire line duration)
        events += f"Dialogue: 1,{line_start},{line_end},LyricsWhite,,0,0,0,,{full_text}\n"

        # Layer 2: for each word, show the full line but only the active word in red
        # We do this by making all other words transparent and the active word red
        for wi, tw in enumerate(tl.words):
            word_start = _ass_timestamp(tw.start)
            word_end = _ass_timestamp(tw.end)

            # Build line with all words transparent except the current one in red
            parts = []
            for wj, tw2 in enumerate(tl.words):
                clean = _clean_word(tw2.word)
                if wj == wi:
                    # This is the active word — red
                    parts.append(clean)
                else:
                    # Invisible (alpha FF = fully transparent)
                    parts.append(f"{{\\1a&HFF&}}{clean}{{\\1a&H00&}}")

            red_text = " ".join(parts)
            events += f"Dialogue: 2,{word_start},{word_end},LyricsRed,,0,0,0,,{red_text}\n"

    ass_content = header + events
    output_path.write_text(ass_content, encoding="utf-8-sig")  # BOM for ASS compatibility
    log.info(f"[caption] Generated ASS subtitle: {output_path}")
    return output_path


async def burn_captions(
    video_path: Path,
    ass_path: Path,
    output_path: Path,
) -> Path:
    """Burn ASS subtitles onto video using ffmpeg."""
    output_dir = output_path.parent

    # Copy font to output dir for ffmpeg to find
    local_font = output_dir / "Roboto-Bold.ttf"
    if not local_font.exists():
        shutil.copy2(_FONT_ABS, local_font)

    # Use the ass filter with fontsdir pointing to output dir
    ass_name = ass_path.name
    fonts_dir = str(output_dir).replace("\\", "/")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path.resolve()),
        "-vf", f"ass={ass_name}:fontsdir=.",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        output_path.name,
    ]

    log.info(f"[caption] Burning karaoke subtitles onto video...")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(output_dir),
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        log.error(f"[caption] ffmpeg stderr:\n{stderr.decode()}")
        raise RuntimeError(f"ffmpeg captioning failed with exit code {proc.returncode}")

    log.info(f"[caption] Captioned video saved to {output_path}")
    return output_path


async def caption_video(
    video_path: Path,
    audio_path: Path,
    lyrics_path: Path,
    output_path: Path,
    whisper_model: str = "base",
) -> Path:
    """Full captioning pipeline: transcribe → match → generate ASS → burn."""
    # Step 1: Transcribe audio with Whisper
    whisper_words = transcribe_audio(audio_path, model_name=whisper_model)

    # Save transcription for debugging
    transcript_file = output_path.parent / "whisper_transcript.json"
    transcript_file.write_text(
        json.dumps(whisper_words, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info(f"[caption] Saved transcript to {transcript_file}")

    # Step 2: Parse lyrics
    lyric_lines = parse_lyrics_lines(lyrics_path)
    log.info(f"[caption] Parsed {len(lyric_lines)} lyric lines")

    # Step 3: Match lyrics to word-level timestamps
    timed_lines = match_lyrics_to_timestamps(lyric_lines, whisper_words)
    log.info(f"[caption] Matched {len(timed_lines)} lines:")
    for tl in timed_lines:
        log.info(f"  [{tl.start:.1f}s - {tl.end:.1f}s] {tl.text} ({len(tl.words)} words)")

    # Save timed lines for debugging
    timed_file = output_path.parent / "timed_lyrics.json"
    timed_data = []
    for t in timed_lines:
        timed_data.append({
            "text": t.text,
            "start": t.start,
            "end": t.end,
            "words": [{"word": w.word, "start": w.start, "end": w.end} for w in t.words],
        })
    timed_file.write_text(json.dumps(timed_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Step 4: Generate ASS subtitle file
    ass_path = output_path.parent / "lyrics.ass"
    generate_ass_subtitles(timed_lines, ass_path)

    # Step 5: Burn onto video
    return await burn_captions(video_path, ass_path, output_path)


if __name__ == "__main__":
    import argparse
    import os
    from dotenv import load_dotenv
    from datetime import date
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    from modules.utils import find_run_dir

    parser = argparse.ArgumentParser(description="Burn karaoke-style synced lyrics onto video")
    parser.add_argument("--date", type=str, default=None, help="Output date folder (default: today)")
    parser.add_argument("--run", type=str, default=None, help="Run folder: number, slug, or name (default: latest)")
    parser.add_argument("--input", type=str, default=None, help="Input video path")
    parser.add_argument("--audio", type=str, default=None, help="Audio path")
    parser.add_argument("--lyrics", type=str, default=None, help="Lyrics file path")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--whisper-model", type=str, default="base",
                        help="Whisper model size: tiny, base, small, medium, large")
    args = parser.parse_args()

    async def main():
        if args.input:
            video = Path(args.input)
            audio = Path(args.audio) if args.audio else video.parent / "song.mp3"
            lyrics = Path(args.lyrics) if args.lyrics else video.parent / "lyrics.txt"
            output = Path(args.output) if args.output else video.parent / "final_captioned.mp4"
        else:
            out_dir = find_run_dir(args.date, args.run)
            video = out_dir / "final.mp4"
            audio = out_dir / "song.mp3"
            lyrics = out_dir / "lyrics.txt"
            output = out_dir / "final_captioned.mp4"

        if not video.exists():
            print(f"Video not found: {video}")
            return
        if not audio.exists():
            print(f"Audio not found: {audio}")
            return
        if not lyrics.exists():
            print(f"Lyrics not found: {lyrics}")
            return

        result = await caption_video(
            video_path=video,
            audio_path=audio,
            lyrics_path=lyrics,
            output_path=output,
            whisper_model=args.whisper_model,
        )
        print(f"Captioned video: {result}")

    asyncio.run(main())
