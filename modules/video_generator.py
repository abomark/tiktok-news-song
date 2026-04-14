"""
Full video generation pipeline — orchestrates the three steps:
  1. clip_generator  — fetch individual clips from Runway
  2. video_assembler — combine clips + audio into final.mp4
  3. captioner       — burn synced lyrics into final_captioned.mp4

Each step can also be run individually:
    python -m modules.clip_generator --reuse
    python -m modules.video_assembler
    python -m modules.captioner

Or run the full pipeline:
    python -m modules.video_generator --reuse
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from modules.lyrics_generator import Lyrics, LyricSection
from modules.clip_generator import generate_clips, parse_lyrics_file, load_style
from modules.video_assembler import assemble_video
from modules.captioner import caption_video

log = logging.getLogger(__name__)


@dataclass
class VideoResult:
    path: Path
    duration_seconds: float


async def generate_video(
    lyrics: Lyrics,
    audio_path: Path,
    audio_duration: float,
    headline: str,
    output_dir: Path,
    runway_api_key: str,
    whisper_model: str = "base",
    skip_clips: bool = False,
    skip_assemble: bool = False,
    skip_captions: bool = False,
) -> VideoResult:
    """Full video generation pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sections = lyrics.sections
    section_duration = audio_duration / len(sections)

    # ── Step 1: Generate clips ────────────────────────────────────
    if skip_clips:
        log.info("[video] Skipping clip generation — using existing clips.")
        clip_paths = sorted(output_dir.glob("clip_*.mp4"))
        if not clip_paths:
            raise FileNotFoundError(f"No clip_*.mp4 found in {output_dir}")
    else:
        clip_paths = await generate_clips(
            sections=sections,
            headline=headline,
            section_duration=section_duration,
            output_dir=output_dir,
            runway_api_key=runway_api_key,
        )

    # ── Step 2: Assemble video ────────────────────────────────────
    final_path = output_dir / "final.mp4"
    if skip_assemble:
        log.info("[video] Skipping assembly — using existing final.mp4.")
        if not final_path.exists():
            raise FileNotFoundError(f"No final.mp4 found in {output_dir}")
    else:
        await assemble_video(
            clip_paths=clip_paths,
            sections=sections,
            section_duration=section_duration,
            audio_path=audio_path,
            output_path=final_path,
        )

    # ── Step 3: Burn captions ─────────────────────────────────────
    captioned_path = output_dir / "final_captioned.mp4"
    if skip_captions:
        log.info("[video] Skipping captions.")
    else:
        lyrics_file = output_dir / "lyrics.txt"
        if lyrics_file.exists():
            await caption_video(
                video_path=final_path,
                audio_path=audio_path,
                lyrics_path=lyrics_file,
                output_path=captioned_path,
                whisper_model=whisper_model,
            )
        else:
            log.warning("[video] No lyrics.txt found — skipping captions.")

    result_path = captioned_path if captioned_path.exists() else final_path
    log.info(f"[video] Done: {result_path}")
    return VideoResult(path=result_path, duration_seconds=audio_duration)


if __name__ == "__main__":
    import argparse
    import os
    from dotenv import load_dotenv
    from datetime import date
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    from modules.utils import find_run_dir

    parser = argparse.ArgumentParser(description="Full video generation pipeline")
    parser.add_argument("--reuse", action="store_true", help="Read headline.txt + lyrics.txt from output folder")
    parser.add_argument("--date", type=str, default=None, help="Output date folder (default: today)")
    parser.add_argument("--run", type=str, default=None, help="Run folder: number, slug, or name (default: latest)")
    parser.add_argument("--duration", type=float, default=40.0, help="Total song duration in seconds")
    parser.add_argument("--skip-clips", action="store_true", help="Skip clip generation, use existing clips")
    parser.add_argument("--skip-assemble", action="store_true", help="Skip assembly, use existing final.mp4")
    parser.add_argument("--skip-captions", action="store_true", help="Skip captioning step")
    parser.add_argument("--whisper-model", type=str, default="base", help="Whisper model: tiny/base/small/medium/large")
    args = parser.parse_args()

    async def main():
        out = find_run_dir(args.date, args.run)
        out.mkdir(parents=True, exist_ok=True)

        runway_key = os.getenv("RUNWAYML_API_SECRET", "")

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
            title = "Tariff Time"
            sections = [
                LyricSection(label="hook", lines=["Money flying out the door"]),
                LyricSection(label="verse", lines=["Tariffs here, tariffs there", "Everybody pulling hair"]),
                LyricSection(label="chorus", lines=["Tax on this, tax on that", "Economy going flat"]),
            ]

        print(f"Headline: {headline}")
        print(f"Title: {title}")
        print(f"Sections: {len(sections)}")
        for s in sections:
            print(f"  [{s.label}] {' / '.join(s.lines[:2])}")

        lyrics = Lyrics(
            title=title,
            sections=sections,
            style_prompt="",
            caption="",
            topic_tags=[],
        )

        result = await generate_video(
            lyrics=lyrics,
            audio_path=out / "song.mp3",
            audio_duration=args.duration,
            headline=headline,
            output_dir=out,
            runway_api_key=runway_key,
            whisper_model=args.whisper_model,
            skip_clips=args.skip_clips,
            skip_assemble=args.skip_assemble,
            skip_captions=args.skip_captions,
        )
        print(f"Video: {result.path}")

    asyncio.run(main())
