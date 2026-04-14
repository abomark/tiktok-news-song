"""
Step 1: Generate individual video clips via Runway text-to-video.

Usage:
    python -m modules.clip_generator                   # uses dummy data
    python -m modules.clip_generator --reuse           # reads headline.txt + lyrics.txt from today's output
    python -m modules.clip_generator --date 2026-04-12 # reads from a specific date folder
"""

from __future__ import annotations
import asyncio
import logging
import math
from pathlib import Path

import httpx

from modules.lyrics_generator import Lyrics, LyricSection

log = logging.getLogger(__name__)

_STYLE_FILE = Path(__file__).parent.parent / "assets" / "video_style.txt"

DEFAULT_STYLE = (
    "Bold editorial cartoon style, saturated colors, dramatic cinematic lighting, "
    "slightly satirical tone, modern digital art, smooth slow camera movement, "
    "no text or words visible"
)


def load_style() -> str:
    """Load visual style from assets/video_style.txt, or use default."""
    if _STYLE_FILE.exists():
        style = _STYLE_FILE.read_text(encoding="utf-8").strip()
        if style:
            return style
    return DEFAULT_STYLE


from modules.utils import log_api_call


def _pick_runway_durations(needed: float) -> list[int]:
    """Pick a list of 5s Runway clips that cover `needed` seconds.

    If the overshoot is 2s or less, skip the extra clip.
    """
    count_ceil = math.ceil(needed / 5)
    count_floor = max(1, math.floor(needed / 5))
    if needed - (count_floor * 5) <= 2:
        return [5] * count_floor
    return [5] * count_ceil


async def generate_clips(
    sections: list[LyricSection],
    headline: str,
    section_duration: float,
    output_dir: Path,
    runway_api_key: str,
    style: str | None = None,
) -> list[Path]:
    """Generate Runway video clips for all sections. Returns list of clip paths."""
    if not runway_api_key:
        raise RuntimeError("RUNWAYML_API_SECRET is required for clip generation.")

    if style is None:
        style = load_style()

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"[clips] Style: {style[:60]}...")
    log.info(f"[clips] Generating {len(sections)} clips ({section_duration:.1f}s each)...")

    tasks = [
        _generate_section_clips(
            section=section,
            headline=headline,
            style=style,
            section_duration=section_duration,
            index=i,
            output_dir=output_dir,
            runway_api_key=runway_api_key,
        )
        for i, section in enumerate(sections)
    ]
    return await asyncio.gather(*tasks)


async def _generate_section_clips(
    section: LyricSection,
    headline: str,
    style: str,
    section_duration: float,
    index: int,
    output_dir: Path,
    runway_api_key: str,
) -> Path:
    """Generate enough 5s clips to cover one section's duration, concat if needed."""
    from runwayml import RunwayML

    client = RunwayML(api_key=runway_api_key)
    durations = _pick_runway_durations(section_duration)

    lyric_snippet = " / ".join(section.lines[:2])
    prompt = (
        f"{style}. "
        f"News headline: {headline}. "
        f"Scene for this part of the song: {lyric_snippet}."
    )

    sub_clips = []
    for j, dur in enumerate(durations):
        clip_label = f"{index}.{j}" if len(durations) > 1 else str(index)
        log.info(f"[clips] Runway: generating clip [{clip_label}] ({dur}s) — {lyric_snippet[:50]}...")
        log_api_call("runway-text-to-video", {
            "index": index,
            "model": "gen4.5",
            "prompt_text": prompt,
            "ratio": "720:1280",
            "duration": dur,
            "sub_clip": j,
            "target_section_duration": section_duration,
        }, run_dir=output_dir)

        def _create_and_wait(d=dur):
            task = client.text_to_video.create(
                model="gen4.5",
                prompt_text=prompt,
                ratio="720:1280",
                duration=d,
            )
            return task.wait_for_task_output(timeout=3600)

        result = await asyncio.get_event_loop().run_in_executor(None, _create_and_wait)

        video_url = result.output[0]
        log_api_call("runway-text-to-video-response", {
            "index": index,
            "video_url": video_url,
            "sub_clip": j,
        }, run_dir=output_dir)

        clip_path = output_dir / f"clip_{index:02d}_{j}.mp4"
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as http:
            r = await http.get(video_url)
            r.raise_for_status()
            clip_path.write_bytes(r.content)
        sub_clips.append(clip_path)
        log.info(f"[clips] Runway: clip [{clip_label}] saved to {clip_path}")

    # If only one sub-clip, just rename it
    final_clip = output_dir / f"clip_{index:02d}.mp4"
    if len(sub_clips) == 1:
        sub_clips[0].rename(final_clip)
    else:
        # Concatenate sub-clips with ffmpeg
        concat_list = output_dir / f"_concat_{index}.txt"
        concat_list.write_text(
            "\n".join(f"file '{c.name}'" for c in sub_clips),
            encoding="utf-8",
        )
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list.name, "-c", "copy", final_clip.name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(output_dir),
        )
        await proc.communicate()
        concat_list.unlink(missing_ok=True)
        for c in sub_clips:
            c.unlink(missing_ok=True)

    log.info(f"[clips] Section [{index}] ready: {final_clip} (target: {section_duration:.1f}s)")
    return final_clip


def parse_lyrics_file(lyrics_path: Path) -> tuple[str, list[LyricSection]]:
    """Parse a lyrics.txt file into title and sections."""
    import re
    text = lyrics_path.read_text(encoding="utf-8")

    title = "Untitled"
    title_match = re.search(r"TITLE:\s*(.+)", text)
    if title_match:
        title = title_match.group(1).strip()

    sections = []
    body = text.split("---")[0] if "---" in text else text
    section_pattern = re.findall(r"\[(\w+)\]\s*\n(.*?)(?=\[|\Z)", body, re.DOTALL)
    for tag, content in section_pattern:
        lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
        if lines:
            sections.append(LyricSection(label=tag.lower(), lines=lines))

    return title, sections


if __name__ == "__main__":
    import argparse
    import os
    from dotenv import load_dotenv
    from datetime import date
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    from modules.utils import find_run_dir

    parser = argparse.ArgumentParser(description="Generate video clips via Runway")
    parser.add_argument("--reuse", action="store_true", help="Read headline.txt + lyrics.txt from output folder")
    parser.add_argument("--date", type=str, default=None, help="Output date folder (default: today)")
    parser.add_argument("--run", type=str, default=None, help="Run folder: number, slug, or name (default: latest)")
    parser.add_argument("--duration", type=float, default=40.0, help="Total song duration in seconds")
    args = parser.parse_args()

    async def main():
        out = find_run_dir(args.date, args.run)
        out.mkdir(parents=True, exist_ok=True)

        runway_key = os.getenv("RUNWAYML_API_SECRET", "")
        if not runway_key:
            print("Set RUNWAYML_API_SECRET in .env")
            return

        if args.reuse:
            headline_file = out / "headline.txt"
            lyrics_file = out / "lyrics.txt"
            if not headline_file.exists():
                print(f"No headline.txt in {out}")
                return
            if not lyrics_file.exists():
                print(f"No lyrics.txt in {out}")
                return

            headline = headline_file.read_text(encoding="utf-8").split("\n")[0].strip()
            title, sections = parse_lyrics_file(lyrics_file)
        else:
            headline = "New economic policies announced"
            sections = [
                LyricSection(label="hook", lines=["Money flying out the door"]),
                LyricSection(label="verse", lines=["Tariffs here, tariffs there", "Everybody pulling hair"]),
                LyricSection(label="chorus", lines=["Tax on this, tax on that", "Economy going flat"]),
            ]
            title = "Tariff Time"

        print(f"Headline: {headline}")
        print(f"Title: {title}")
        print(f"Sections: {len(sections)}")
        for s in sections:
            print(f"  [{s.label}] {' / '.join(s.lines[:2])}")

        section_duration = args.duration / len(sections)
        clip_paths = await generate_clips(
            sections=sections,
            headline=headline,
            section_duration=section_duration,
            output_dir=out,
            runway_api_key=runway_key,
        )
        print(f"\nGenerated {len(clip_paths)} clips:")
        for p in clip_paths:
            print(f"  {p}")

    asyncio.run(main())
