"""
Full video generation pipeline — orchestrates the steps:
  0. scene_planner    — LLM generates a visual story plan
  1. pollo_generator  — Pollo AI renders each scene as a 5s clip
  2. video_assembler  — combine clips + audio into final.mp4
  3. captioner        — burn synced lyrics into final_captioned.mp4

Or run the full pipeline:
    python -m modules.video_generator --reuse
"""

from __future__ import annotations
import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from modules.lyrics_generator import Lyrics, LyricSection
from modules.clip_generator import parse_lyrics_file
from modules.pollo_generator import generate_clips_from_plan, parse_headline_file
from modules.scene_planner import plan_scenes
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
    pollo_api_key: str,
    image_url: str | None = None,
    summary: str = "",
    llm_api_key: str = "",
    llm_model: str = "",
    llm_base_url: str | None = None,
    whisper_model: str = "base",
    skip_clips: bool = False,
    skip_assemble: bool = False,
    skip_captions: bool = False,
    video_model: str | None = None,
) -> VideoResult:
    """Full video generation pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_duration = 5.0  # each Pollo clip is exactly 5s

    # ── Step 0: Plan visual scenes via LLM ────────────────────────
    if skip_clips:
        log.info("[video] Skipping clip generation — using existing clips.")
        clip_paths = sorted(output_dir.glob("clip_*.mp4"))
        if not clip_paths:
            raise FileNotFoundError(f"No clip_*.mp4 found in {output_dir}")
    else:
        log.info("[video] Planning visual scenes...")
        scene_plan = await plan_scenes(
            headline=headline,
            summary=summary,
            lyrics=lyrics,
            song_duration=audio_duration,
            image_url=image_url,
            api_key=llm_api_key,
            model=llm_model,
            base_url=llm_base_url,
            output_dir=output_dir,
        )
        # Save for debugging
        plan_data = [
            {"clip_index": s.clip_index, "prompt": s.prompt,
             "is_image_to_video": s.is_image_to_video, "visual_action": s.visual_action}
            for s in scene_plan.scenes
        ]
        (output_dir / "scene_plan.json").write_text(
            json.dumps(plan_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # ── Step 1: Generate clips ────────────────────────────────
        from modules.pollo_generator import DEFAULT_MODEL
        effective_video_model = video_model or DEFAULT_MODEL
        log.info(f"[video] Generating video clips (model={effective_video_model})...")
        clip_paths = await generate_clips_from_plan(
            plan=scene_plan,
            output_dir=output_dir,
            pollo_api_key=pollo_api_key,
            image_url=image_url,
            run_dir=output_dir,
            model=effective_video_model,
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
            sections=lyrics.sections,
            section_duration=clip_duration,
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
    parser.add_argument("--provider", type=str, default="ollama", choices=["ollama", "grok"],
                        help="LLM provider for scene planning (default: ollama)")
    parser.add_argument("--video-model", type=str, default=None,
                        help="Pollo video model (e.g. seedance-pro-1-5, veo3-1-fast). Default: seedance-pro-1-5")
    args = parser.parse_args()

    async def main():
        out = find_run_dir(args.date, args.run)
        out.mkdir(parents=True, exist_ok=True)

        pollo_key = os.getenv("POLLO_API_KEY", "")

        headline = "New economic policies announced"
        summary = ""
        image_url = None
        title = "Tariff Time"
        sections = [
            LyricSection(label="hook", lines=["Money flying out the door"]),
            LyricSection(label="verse", lines=["Tariffs here, tariffs there", "Everybody pulling hair"]),
            LyricSection(label="chorus", lines=["Tax on this, tax on that", "Economy going flat"]),
        ]

        if args.reuse:
            headline_file = out / "headline.txt"
            lyrics_file = out / "lyrics.txt"
            if not headline_file.exists():
                print(f"No headline.txt in {out}")
                return
            if not lyrics_file.exists():
                print(f"No lyrics.txt in {out}")
                return

            headline, image_url = parse_headline_file(headline_file)
            hl_lines = headline_file.read_text(encoding="utf-8").splitlines()
            summary = hl_lines[2] if len(hl_lines) > 2 else ""
            title, sections = parse_lyrics_file(lyrics_file)

        # LLM provider config
        if args.provider == "grok":
            llm_key = os.environ.get("XAI_API_KEY", "")
            llm_base = "https://api.x.ai/v1"
            llm_model = "grok-3-fast"
        else:
            llm_key = "ollama"
            llm_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            llm_model = os.getenv("OLLAMA_MODEL", "gemma3")

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
            pollo_api_key=pollo_key,
            image_url=image_url,
            summary=summary,
            llm_api_key=llm_key,
            llm_model=llm_model,
            llm_base_url=llm_base,
            whisper_model=args.whisper_model,
            skip_clips=args.skip_clips,
            skip_assemble=args.skip_assemble,
            skip_captions=args.skip_captions,
            video_model=args.video_model,
        )
        print(f"Video: {result.path}")

    asyncio.run(main())
