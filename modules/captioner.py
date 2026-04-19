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
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

_FONT_ABS = Path(__file__).parent.parent / "assets" / "fonts" / "impact.ttf"

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
        if line.startswith("STYLE:"):
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
    karaoke: bool = True,
) -> Path:
    """Generate an ASS subtitle file with karaoke word highlighting.

    Each word starts white and turns red when sung.
    """
    font_name = "Impact"
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
Style: LyricsRed,{font_name},{FONT_SIZE},&H000000FF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,5,40,40,{margin_v},1
Style: LyricsSolo,{font_name},{int(FONT_SIZE * 3)},&H000000FF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,4,0,5,40,40,{margin_v},1
Style: Watermark,{font_name},33,&HAAFFFFFF,&HAAFFFFFF,&H88000000,&H88000000,0,0,0,0,100,100,0,0,1,2,0,2,0,0,480,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    # Watermark (always visible)
    wm_start = _ass_timestamp(0)
    wm_end = _ass_timestamp(timed_lines[-1].end + 5 if timed_lines else 60)
    wm_clean = WATERMARK_TEXT.encode("ascii", "ignore").decode("ascii")
    events = f"Dialogue: 0,{wm_start},{wm_end},Watermark,,0,0,0,,{wm_clean}\n"

    active_fs = int(FONT_SIZE * 1.12)
    MAX_WORD_DURATION = 1.5

    for tl in timed_lines:
        line_start = _ass_timestamp(tl.start)
        line_end = _ass_timestamp(tl.end)

        clean_words = [_clean_word(w.word).upper() for w in tl.words]
        full_text = " ".join(clean_words)

        # Group words that share the same time window
        word_groups: list[tuple[float, float, list[int]]] = []
        for wi, tw in enumerate(tl.words):
            if word_groups and abs(tw.start - word_groups[-1][0]) < 0.05 and abs(tw.end - word_groups[-1][1]) < 0.05:
                word_groups[-1][2].append(wi)
            else:
                word_groups.append((tw.start, tw.end, [wi]))

        word_groups = [
            (s, min(e, s + MAX_WORD_DURATION), idxs)
            for s, e, idxs in word_groups
        ]

        if karaoke:
            prev_end = tl.start
            for g_start, g_end, active_indices in word_groups:
                if g_start > prev_end + 0.05:
                    events += f"Dialogue: 1,{_ass_timestamp(prev_end)},{_ass_timestamp(g_start)},LyricsWhite,,0,0,0,,{full_text}\n"

                active_set = set(active_indices)

                white_parts = []
                for wj, cw in enumerate(clean_words):
                    if wj in active_set:
                        white_parts.append(f"{{\\1a&HFF&\\3a&HFF&}}{cw}{{\\1a&H00&\\3a&H00&}}")
                    else:
                        white_parts.append(cw)
                events += f"Dialogue: 1,{_ass_timestamp(g_start)},{_ass_timestamp(g_end)},LyricsWhite,,0,0,0,,{' '.join(white_parts)}\n"

                red_parts = []
                for wj, cw in enumerate(clean_words):
                    if wj in active_set:
                        red_parts.append(f"{{\\fs{active_fs}}}{cw}{{\\fs{FONT_SIZE}}}")
                    else:
                        red_parts.append(f"{{\\1a&HFF&\\3a&HFF&}}{cw}{{\\1a&H00&\\3a&H00&}}")
                events += f"Dialogue: 2,{_ass_timestamp(g_start)},{_ass_timestamp(g_end)},LyricsRed,,0,0,0,,{' '.join(red_parts)}\n"

                prev_end = g_end

            if tl.end > prev_end + 0.05:
                events += f"Dialogue: 1,{_ass_timestamp(prev_end)},{line_end},LyricsWhite,,0,0,0,,{full_text}\n"
        else:
            for g_start, g_end, active_indices in word_groups:
                active_set = set(active_indices)
                solo_text = " ".join(clean_words[i] for i in sorted(active_set))
                solo_text = solo_text.replace(",", "").replace(".", "")
                events += f"Dialogue: 1,{_ass_timestamp(g_start)},{_ass_timestamp(g_end)},LyricsSolo,,0,0,0,,{solo_text}\n"

    ass_content = header + events
    output_path.write_text(ass_content, encoding="utf-8-sig")  # BOM for ASS compatibility
    log.info(f"[caption] Generated ASS subtitle: {output_path}")
    return output_path


def _sanitize_hook_line(line: str) -> str:
    """Strip non-ASCII and ASS-unsafe characters from a hook caption line."""
    clean = line.encode("ascii", "ignore").decode("ascii")
    clean = clean.replace("\\", "").replace("{", "").replace("}", "")
    return clean.strip()


def _generate_hook_ass(
    hook_caption_text: str,
    duration: float,
    video_width: int,
    video_height: int,
) -> str:
    """Build a standalone ASS file for the 2-line opening hook caption."""
    lines = [_sanitize_hook_line(l) for l in hook_caption_text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("hook_caption_text is empty after sanitizing")
    # Enforce 2 lines max — join trailing lines to line 2 if model overshot
    if len(lines) > 2:
        lines = [lines[0], " ".join(lines[1:])]
    text = "\\N".join(lines)

    font_name = "Impact"
    font_size = 64
    # Alignment=8 = top-center; MarginV positions from top edge
    # 75% down the vertical scale = 0.75 * video_height
    margin_v = int(video_height * 0.75)
    # BorderStyle=3: opaque box; box color = OutlineColour (not BackColour)
    # PrimaryColour=black text, OutlineColour=white box, Outline=20 for padding
    header = f"""[Script Info]
Title: TikTok Hook Caption
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: HookCaption,{font_name},{font_size},&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,20,0,8,80,80,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    start = _ass_timestamp(0.0)
    end = _ass_timestamp(duration)
    event = f"Dialogue: 5,{start},{end},HookCaption,,0,0,0,,{text}\n"
    return header + event


async def burn_hook_caption(
    video_path: Path,
    output_path: Path,
    hook_caption_text: str,
    duration: float = 2.0,
    video_width: int = 1080,
    video_height: int = 1920,
) -> Path:
    """Burn a 2-line white-box hook caption onto the first N seconds of the video.

    Writes `tiktok_hook.ass` next to the output and burns it onto `video_path`
    using ffmpeg's subtitles filter. Intended to produce the TikTok-only
    `final_tiktok.mp4` layered on top of the YouTube-ready `final_captioned.mp4`.
    """
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    local_font = output_dir / "impact.ttf"
    if not local_font.exists():
        shutil.copy2(_FONT_ABS, local_font)

    ass_content = _generate_hook_ass(hook_caption_text, duration, video_width, video_height)
    ass_path = output_dir / "tiktok_hook.ass"
    ass_path.write_text(ass_content, encoding="utf-8-sig")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path.resolve()),
        "-vf", f"ass={ass_path.name}:fontsdir=.",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "21",
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        output_path.name,
    ]

    log.info(f"[caption] Burning TikTok hook caption ({duration:.1f}s) onto video...")

    result = await asyncio.to_thread(
        subprocess.run,
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(output_dir),
    )

    if result.returncode != 0:
        log.error(f"[caption] ffmpeg stderr:\n{result.stderr.decode()}")
        raise RuntimeError(f"ffmpeg hook-caption burn failed with exit code {result.returncode}")

    log.info(f"[caption] TikTok video saved to {output_path}")
    return output_path


async def burn_captions(
    video_path: Path,
    ass_path: Path,
    output_path: Path,
) -> Path:
    """Burn ASS subtitles onto video using ffmpeg."""
    output_dir = output_path.parent

    # Copy font to output dir for ffmpeg to find
    local_font = output_dir / "impact.ttf"
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
        "-crf", "21",
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        output_path.name,
    ]

    log.info(f"[caption] Burning karaoke subtitles onto video...")

    result = await asyncio.to_thread(
        subprocess.run,
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(output_dir),
    )

    if result.returncode != 0:
        log.error(f"[caption] ffmpeg stderr:\n{result.stderr.decode()}")
        raise RuntimeError(f"ffmpeg captioning failed with exit code {result.returncode}")

    log.info(f"[caption] Captioned video saved to {output_path}")
    return output_path


def build_timed_lines_from_suno(
    aligned_words: list[dict],
    lyric_lines: list[str],
) -> list[TimedLine]:
    """Build TimedLines from Suno's alignedWords, matched to lyric lines.

    Suno's word list embeds section tags ([HOOK], [VERSE], [CHORUS]) and
    repeats sections (e.g. hook sung twice before verse). We use the tags
    to find the LAST occurrence of each section start, then consume words
    sequentially from that point — this skips repeated intros and aligns
    to the actual sung section.
    """
    import re

    # Parse the raw word list into tagged segments, preserving section boundaries
    # Each entry: {"tag": "HOOK"|"VERSE"|"CHORUS"|None, "word": str, "start": float, "end": float}
    segments: list[dict] = []
    for aw in aligned_words:
        raw = aw.get("word", "")
        start = float(aw.get("startS", 0))
        end = float(aw.get("endS", 0))

        # Extract section tag if present (e.g. "[HOOK]\nGemini's")
        tag_match = re.match(r"\[([A-Z]+)\]\n?(.*)", raw, re.DOTALL)
        if tag_match:
            tag = tag_match.group(1)
            remainder = tag_match.group(2).strip()
            segments.append({"tag": tag, "word": None, "start": start, "end": end})
            if remainder:
                # Strip trailing newlines/section markers from remainder
                remainder = re.sub(r"\n\[.*", "", remainder).strip("\n ")
                if remainder:
                    segments.append({"tag": None, "word": remainder, "start": start, "end": end})
        else:
            # Strip from first newline onwards — Suno sometimes merges the last word
            # of one line with the first word of the next (e.g. "throne,\nQ2 ").
            # Keeping the cross-line text breaks token matching for all subsequent lines.
            word = raw.strip("\n ")
            word = re.sub(r"\n.*", "", word).strip()
            if not word:
                continue
            # Suno sometimes emits multi-word tokens (e.g. "District 11, ").
            # Split them and distribute the timestamp proportionally so each
            # sub-token gets its own slot — otherwise all downstream lyric
            # tokens that map to this span end up with identical timestamps
            # and collapse into a single caption group.
            sub_tokens = word.split()
            if len(sub_tokens) > 1:
                span = end - start
                per_tok = span / len(sub_tokens)
                for k, tok in enumerate(sub_tokens):
                    segments.append({
                        "tag": None,
                        "word": tok,
                        "start": start + k * per_tok,
                        "end": start + (k + 1) * per_tok,
                    })
            else:
                segments.append({"tag": None, "word": word, "start": start, "end": end})

    # Find the LAST occurrence of each section tag — Suno often repeats the hook
    # before the verse, so we want to start from the final appearance of each section
    section_order = []  # ordered list of (tag, start_idx_in_segments)
    seen_tags: dict[str, int] = {}
    for i, seg in enumerate(segments):
        if seg["tag"]:
            seen_tags[seg["tag"]] = i  # overwrite — keeps last occurrence

    # Rebuild ordered section starts from the last occurrences
    for i, seg in enumerate(segments):
        if seg["tag"] and seen_tags.get(seg["tag"]) == i:
            section_order.append((seg["tag"], i))

    # Build flat word list starting from the first relevant section start
    # (skip repeated earlier sections)
    first_section_idx = section_order[0][1] if section_order else 0
    clean_words = [
        s for s in segments[first_section_idx:]
        if s["word"] and s["tag"] is None
    ]

    if not clean_words:
        return []

    def _clean_token(s: str) -> str:
        return re.sub(r"[^\w]", "", s).lower()

    # Consume Suno words greedily per lyric token.
    # Suno sometimes splits a single word across multiple entries
    # (e.g. "Gemini's" → "Gemini'" + "s"), so we merge consecutive
    # Suno words into one lyric token until the concatenated text
    # fuzzy-matches the token, then advance.
    timed_lines: list[TimedLine] = []
    word_idx = 0

    for lyric_line in lyric_lines:
        line_tokens = lyric_line.split()
        if not line_tokens or word_idx >= len(clean_words):
            continue

        line_words: list[TimedWord] = []
        for token in line_tokens:
            if word_idx >= len(clean_words):
                break
            token_clean = _clean_token(token)
            # Accumulate Suno words until we've matched this token
            accumulated = ""
            start_time = clean_words[word_idx]["start"]
            end_time = clean_words[word_idx]["end"]
            while word_idx < len(clean_words):
                sw = clean_words[word_idx]
                accumulated += _clean_token(sw["word"])
                end_time = sw["end"]
                word_idx += 1
                if accumulated == token_clean or token_clean.startswith(accumulated):
                    if accumulated == token_clean:
                        break
                else:
                    # Gone too far — back up one
                    word_idx -= 1
                    break
            line_words.append(TimedWord(word=token, start=start_time, end=end_time))

        if line_words:
            timed_lines.append(TimedLine(
                text=lyric_line,
                start=line_words[0].start,
                end=line_words[-1].end,
                words=line_words,
            ))

    # Cap each line's end at the next line's start — prevents captions
    # bleeding into silence gaps (e.g. held notes with long endS)
    for i in range(len(timed_lines) - 1):
        next_start = timed_lines[i + 1].start
        if timed_lines[i].end > next_start:
            timed_lines[i].end = next_start

    return timed_lines


async def caption_video(
    video_path: Path,
    audio_path: Path,
    lyrics_path: Path,
    output_path: Path,
    whisper_model: str = "base",
    karaoke: bool = False,
) -> Path:
    """Full captioning pipeline: transcribe → match → generate ASS → burn."""
    # Step 2: Parse lyrics
    lyric_lines = parse_lyrics_lines(lyrics_path)
    log.info(f"[caption] Parsed {len(lyric_lines)} lyric lines")

    # Step 1: Use Suno's timestamped lyrics if available, else fall back to Whisper
    suno_timed_file = output_path.parent / "timed_lyrics.json"
    if suno_timed_file.exists():
        log.info(f"[caption] Using Suno timestamped lyrics from {suno_timed_file}")
        aligned_words = json.loads(suno_timed_file.read_text(encoding="utf-8"))
        timed_lines = build_timed_lines_from_suno(aligned_words, lyric_lines)
    else:
        log.info("[caption] No Suno timestamps found — falling back to Whisper")
        whisper_words = transcribe_audio(audio_path, model_name=whisper_model)
        transcript_file = output_path.parent / "whisper_transcript.json"
        transcript_file.write_text(
            json.dumps(whisper_words, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info(f"[caption] Saved transcript to {transcript_file}")
        timed_lines = match_lyrics_to_timestamps(lyric_lines, whisper_words)

    log.info(f"[caption] Matched {len(timed_lines)} lines:")
    for tl in timed_lines:
        log.info(f"  [{tl.start:.1f}s - {tl.end:.1f}s] {tl.text} ({len(tl.words)} words)")

    # Step 3: Generate ASS subtitle file
    ass_path = output_path.parent / "lyrics.ass"
    generate_ass_subtitles(timed_lines, ass_path, karaoke=karaoke)

    # Step 4: Burn onto video
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
    parser.add_argument("--karaoke", action="store_true",
                        help="Full-sentence karaoke mode: show entire line with active word in red")
    parser.add_argument("--hook-only", action="store_true",
                        help="Skip karaoke; burn only the 2s TikTok hook caption onto final_captioned.mp4")
    parser.add_argument("--hook-text", type=str, default=None,
                        help="Override hook caption text (use \\n between lines). Default: read from lyrics.txt.")
    parser.add_argument("--hook-duration", type=float, default=2.0,
                        help="Duration of the hook caption in seconds (default: 2.0)")
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

        if args.hook_only:
            # Input is the existing captioned video; output is final_tiktok.mp4
            captioned = output if output.exists() else video.parent / "final_captioned.mp4"
            if not captioned.exists():
                print(f"final_captioned.mp4 not found: {captioned}")
                return
            tiktok_out = captioned.parent / "final_tiktok.mp4"
            hook_text = args.hook_text.replace("\\n", "\n") if args.hook_text else None
            if hook_text is None and lyrics.exists():
                text = lyrics.read_text(encoding="utf-8")
                if "HOOK_CAPTION:" in text:
                    hook_block = text.split("HOOK_CAPTION:", 1)[1].strip()
                    hook_text = "\n".join(hook_block.splitlines()[:2])
            if not hook_text:
                print("No hook caption text provided (--hook-text or HOOK_CAPTION: in lyrics.txt)")
                return
            await burn_hook_caption(
                video_path=captioned,
                output_path=tiktok_out,
                hook_caption_text=hook_text,
                duration=args.hook_duration,
            )
            print(f"TikTok video: {tiktok_out}")
            return

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
            karaoke=args.karaoke,
        )
        print(f"Captioned video: {result}")

    asyncio.run(main())
