"""
Main pipeline orchestrator.

Usage:
    python pipeline.py              # Full run + post to TikTok
    python pipeline.py --dry-run    # Run everything except posting
    python pipeline.py --skip-news  --headline "Custom headline"  --summary "Custom summary"
"""

from __future__ import annotations
import argparse
import asyncio
import logging
import sys
from datetime import date
from pathlib import Path

# Suppress "Event loop is closed" noise from httpx cleanup on Windows Python 3.9
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from config import (
    NEWS_API_KEY,
    OPENAI_API_KEY,
    SUNOAPI_KEY,
    SUNOAPI_BASE,
    TIKTOK_CLIENT_KEY,
    TIKTOK_CLIENT_SECRET,
    TIKTOK_REFRESH_TOKEN,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    FIXED_HASHTAGS,
    NEWS_COUNTRY,
    OUTPUT_DIR,
    POLLO_API_KEY,
)
from modules.news_fetcher import fetch_news_candidates
from modules.social_scorer import score_candidates
from modules.story_classifier import classify_story, classify_story_dual
from modules.story_selector import score_and_flag
from modules.lyrics_generator import generate_lyrics
from modules.music_generator import generate_music
from modules.video_generator import generate_video
from modules.tiktok_publisher import publish_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "pipeline.log", encoding="utf-8"),
    ],
)
# Ensure stdout handler doesn't crash on emoji/unicode on Windows
logging.getLogger().handlers[0].stream = open(
    sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False
)
log = logging.getLogger(__name__)


import re

def _slugify(text: str, max_len: int = 40) -> str:
    """Convert text to a filesystem-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = slug.strip("-")
    return slug[:max_len]


def _make_run_dir(base_dir: Path, title: str) -> Path:
    """Create a numbered + slugified subfolder like 01-ceasefire-shuffle."""
    base_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(title)

    # Find next available number
    existing = sorted(base_dir.iterdir()) if base_dir.exists() else []
    next_num = 1
    for p in existing:
        if p.is_dir() and p.name[:2].isdigit():
            try:
                next_num = max(next_num, int(p.name[:2]) + 1)
            except ValueError:
                pass

    folder_name = f"{next_num:02d}-{slug}" if slug else f"{next_num:02d}"
    run_dir = base_dir / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _lyrics_exist_for(headline: str, day_dir: Path) -> bool:
    """Return True if a lyrics.txt already exists for this headline today."""
    for existing in sorted(day_dir.glob("*/headline.txt")):
        try:
            first_line = existing.read_text(encoding="utf-8").split("\n")[0].strip()
            if first_line == headline:
                return True
        except OSError:
            pass
    return False


def _find_run_dir_for(headline: str, day_dir: Path) -> Path | None:
    """Return the run folder for a headline if it exists, else None."""
    for existing in sorted(day_dir.glob("*/headline.txt")):
        try:
            first_line = existing.read_text(encoding="utf-8").split("\n")[0].strip()
            if first_line == headline:
                return existing.parent
        except OSError:
            pass
    return None


def _song_exists_for(headline: str, day_dir: Path) -> bool:
    """Return True if a song.mp3 already exists for this headline today."""
    run_dir = _find_run_dir_for(headline, day_dir)
    return run_dir is not None and (run_dir / "song.mp3").exists()


def _video_complete_for(headline: str, day_dir: Path) -> bool:
    """Return True if final_tiktok.mp4 already exists for this headline today."""
    run_dir = _find_run_dir_for(headline, day_dir)
    if run_dir is None:
        return False
    return (run_dir / "final_tiktok.mp4").exists()


def _provider_config(provider: str) -> tuple[str, str, str]:
    """Return (api_key, base_url, model) for the given provider."""
    import os
    if provider == "grok":
        return os.environ.get("XAI_API_KEY", ""), "https://api.x.ai/v1", "grok-3-fast"
    return "ollama", OLLAMA_BASE_URL, OLLAMA_MODEL


async def _generate_and_save_lyrics(
    headline: str,
    summary: str,
    angle: str,
    url: str,
    day_dir: Path,
    provider: str,
    image_url: str | None = None,
) -> tuple:
    """Generate lyrics for one story, write outputs, return (lyrics, output_dir)."""
    from modules.lyrics_generator import Lyrics
    api_key, base_url, model = _provider_config(provider)

    temp_dir = day_dir / "_generating"
    temp_dir.mkdir(parents=True, exist_ok=True)

    lyrics = await generate_lyrics(
        headline=headline,
        summary=summary,
        api_key=api_key,
        model=model,
        fixed_hashtags=FIXED_HASHTAGS,
        angle=angle,
        base_url=base_url,
        output_dir=temp_dir,
    )
    log.info(f"[pipeline] Song title: '{lyrics.title}'")

    output_dir = _make_run_dir(day_dir, lyrics.title)
    for f in temp_dir.iterdir():
        f.rename(output_dir / f.name)
    temp_dir.rmdir()

    headline_parts = [headline, "", summary, url]
    if image_url:
        headline_parts.append(f"IMAGE: {image_url}")
    (output_dir / "headline.txt").write_text(
        "\n".join(headline_parts), encoding="utf-8"
    )
    lyrics_body = (
        f"TITLE: {lyrics.title}\nSTYLE: {lyrics.style_prompt}\n\n"
        f"{lyrics.full_text}\n\n---\nCAPTION:\n{lyrics.caption}"
    )
    if lyrics.hook_caption:
        lyrics_body += f"\n\n---\nHOOK_CAPTION:\n{lyrics.hook_caption}"
    (output_dir / "lyrics.txt").write_text(lyrics_body, encoding="utf-8")
    log.info(f"[pipeline] Output: {output_dir}")
    return lyrics, output_dir


async def run_all_flagged(provider: str = "ollama") -> None:
    """Generate lyrics for every flagged story today that doesn't have lyrics yet."""
    import json
    today = date.today().isoformat()
    day_dir = OUTPUT_DIR / today
    day_dir.mkdir(parents=True, exist_ok=True)

    flagged_log = Path("logs/flagged_stories.jsonl")
    if not flagged_log.exists():
        log.warning("[pipeline] No flagged_stories.jsonl found — run the scheduler first.")
        return

    entries = []
    for line in flagged_log.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            if e.get("date") == today:
                entries.append(e)
        except json.JSONDecodeError:
            pass

    if not entries:
        log.warning(f"[pipeline] No flagged stories for {today}.")
        return

    log.info(f"[pipeline] {len(entries)} flagged stories for {today}")

    generated = 0
    skipped = 0
    for e in entries:
        headline = e.get("headline", "")
        if not headline:
            continue
        if _lyrics_exist_for(headline, day_dir):
            log.info(f"[pipeline] Lyrics already exist — skipping: {headline[:70]}")
            skipped += 1
            continue

        log.info(f"[pipeline] Generating lyrics for: {headline[:70]}")
        await _generate_and_save_lyrics(
            headline=headline,
            summary=e.get("summary", ""),
            angle=e.get("angle", ""),
            url=e.get("url", ""),
            day_dir=day_dir,
            provider=provider,
        )  # return value not needed — output already saved to disk
        generated += 1

    log.info(f"[pipeline] All-flagged done — {generated} generated, {skipped} skipped.")


async def run_all_flagged_music(provider: str = "ollama") -> None:
    """Generate lyrics + music for every flagged story today that doesn't have a song yet."""
    import json
    today = date.today().isoformat()
    day_dir = OUTPUT_DIR / today
    day_dir.mkdir(parents=True, exist_ok=True)

    flagged_log = Path("logs/flagged_stories.jsonl")
    if not flagged_log.exists():
        log.warning("[pipeline] No flagged_stories.jsonl found — run the scheduler first.")
        return

    entries = []
    for line in flagged_log.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            if e.get("date") == today:
                entries.append(e)
        except json.JSONDecodeError:
            pass

    if not entries:
        log.warning(f"[pipeline] No flagged stories for {today}.")
        return

    log.info(f"[pipeline] {len(entries)} flagged stories for {today}")

    generated = 0
    skipped = 0
    for e in entries:
        headline = e.get("headline", "")
        if not headline:
            continue

        if _song_exists_for(headline, day_dir):
            log.info(f"[pipeline] Song already exists — skipping: {headline[:70]}")
            skipped += 1
            continue

        # Generate lyrics first if not already done
        run_dir = _find_run_dir_for(headline, day_dir)
        if run_dir is None:
            log.info(f"[pipeline] Generating lyrics for: {headline[:70]}")
            try:
                _, run_dir = await _generate_and_save_lyrics(
                    headline=headline,
                    summary=e.get("summary", ""),
                    angle=e.get("angle", ""),
                    url=e.get("url", ""),
                    day_dir=day_dir,
                    provider=provider,
                    image_url=e.get("image_url"),
                )
            except Exception as exc:
                log.error(f"[pipeline] Lyrics generation failed for '{headline[:60]}': {exc}")
                continue
        else:
            log.info(f"[pipeline] Lyrics already exist, generating music for: {headline[:70]}")

        # Generate music
        lyrics_path = run_dir / "lyrics.txt"
        lyrics_raw = lyrics_path.read_text(encoding="utf-8")
        # lyrics.txt format: "TITLE: ...\n\n<lyrics>\n\n---\nCAPTION:\n<caption>"
        # Split on the separator to get only the pure lyrics body
        title_line = next((l for l in lyrics_raw.splitlines() if l.startswith("TITLE:")), "TITLE: Untitled")
        title = title_line.removeprefix("TITLE:").strip()
        lyrics_body = lyrics_raw.split("\n\n---\n")[0].removeprefix(title_line).strip()

        log.info(f"[pipeline] Generating music (sunoapi.org) for: {title}")
        try:
            audio = await generate_music(
                lyrics_text=lyrics_body,
                style_prompt="upbeat pop, punchy drums, catchy hook, modern production",
                title=title,
                output_dir=run_dir,
                sunoapi_key=SUNOAPI_KEY,
                sunoapi_base=SUNOAPI_BASE,
            )
            log.info(f"[pipeline] Music saved: {audio.path} ({audio.duration_seconds:.1f}s)")
            generated += 1
        except Exception as exc:
            log.error(f"[pipeline] Music generation failed for '{headline[:60]}': {exc}")

    log.info(f"[pipeline] All-flagged-music done — {generated} generated, {skipped} skipped.")


async def _run_one_story(
    story,
    angle: str,
    day_dir: Path,
    dry_run: bool,
    dry_run_full: bool,
    lyrics_only: bool,
    provider: str,
    video_model: str | None,
) -> None:
    """Run the full pipeline for a single story (lyrics → music → video → publish)."""
    log.info(f"[pipeline] Story: {story.headline}")
    if angle:
        log.info(f"[pipeline] Satirical angle: {angle}")

    if _video_complete_for(story.headline, day_dir):
        log.info("[pipeline] Video already complete for this story — skipping.")
        return

    # ── Step 2: Lyrics ────────────────────────────────────────────
    # Check if we already have a run dir with lyrics for this headline — reuse it.
    existing_run_dir = _find_run_dir_for(story.headline, day_dir)
    if existing_run_dir is not None and (existing_run_dir / "lyrics.txt").exists():
        log.info(f"[pipeline] Reusing existing run dir: {existing_run_dir.name}")
        from modules.lyrics_generator import Lyrics, LyricSection
        from modules.clip_generator import parse_lyrics_file
        lyrics_file = existing_run_dir / "lyrics.txt"
        _, sections = parse_lyrics_file(lyrics_file)
        text = lyrics_file.read_text(encoding="utf-8")
        title_line = next((l for l in text.splitlines() if l.startswith("TITLE:")), "TITLE: Unknown")
        style_line = next((l for l in text.splitlines() if l.startswith("STYLE:")), "STYLE:")
        # Parse CAPTION: section — ends at next "---" separator if present
        caption = ""
        if "CAPTION:\n" in text:
            caption_tail = text.split("CAPTION:\n", 1)[1]
            caption = caption_tail.split("\n---\n", 1)[0].strip()
        # Parse optional HOOK_CAPTION: section; fall back to title-based card for legacy files
        title_val = title_line.replace("TITLE:", "").strip()
        if "HOOK_CAPTION:" in text:
            hook_block = text.split("HOOK_CAPTION:", 1)[1].strip()
            hook_caption = "\n".join(hook_block.splitlines()[:2])
        else:
            hook_caption = f"{title_val.upper()}\n#news"
        lyrics = Lyrics(
            title=title_val,
            style_prompt=style_line.replace("STYLE:", "").strip(),
            sections=sections,
            caption=caption,
            topic_tags=[],
            hook_caption=hook_caption,
        )
        output_dir = existing_run_dir
    else:
        log.info("[pipeline] Generating lyrics...")
        lyrics, output_dir = await _generate_and_save_lyrics(
            headline=story.headline,
            summary=story.summary,
            angle=angle,
            url=story.url,
            day_dir=day_dir,
            provider=provider,
            image_url=getattr(story, "image_url", None),
        )

        # ── Step 2b: Classify lyrics ─────────────────────────────
        log.info("[pipeline] Classifying lyrics (dual: gemma3 + grok)...")
        try:
            import os as _os
            from modules.lyrics_classifier import classify_lyrics_dual
            lyrics_classification = await classify_lyrics_dual(
                headline=story.headline,
                title=lyrics.title,
                lyrics_text=lyrics.full_text,
                ollama_model=OLLAMA_MODEL,
                ollama_base_url=OLLAMA_BASE_URL,
                grok_api_key=_os.environ.get("XAI_API_KEY", ""),
            )
            log.info(f"[pipeline] Lyrics LVI={lyrics_classification.lvi:.1f} ({lyrics_classification.lvi_label})")
        except Exception as exc:
            log.warning(f"[pipeline] Lyrics classification failed (continuing): {exc}")

    if lyrics_only:
        log.info(f"[pipeline] LYRICS ONLY — done. Output: {output_dir}")
        return

    # ── Step 3: Music ─────────────────────────────────────────────
    existing_audio = output_dir / "song.mp3"
    if existing_audio.exists():
        log.info(f"[pipeline] Reusing existing song: {existing_audio.name}")
        from modules.music_generator import AudioResult, _get_duration
        duration = await _get_duration(existing_audio)
        audio = AudioResult(path=existing_audio, duration_seconds=duration, title=lyrics.title)
        log.info(f"[pipeline] Audio: {existing_audio.name} ({duration:.1f}s)")
    elif dry_run_full:
        log.info("[pipeline] DRY RUN FULL — skipping Suno music generation.")
    else:
        log.info("[pipeline] Generating music (Suno)...")
        audio = await generate_music(
            lyrics_text=lyrics.full_text,
            style_prompt=lyrics.style_prompt,
            title=lyrics.title,
            output_dir=output_dir,
            sunoapi_key=SUNOAPI_KEY,
            sunoapi_base=SUNOAPI_BASE,
        )
        log.info(f"[pipeline] Audio: {audio.path} ({audio.duration_seconds:.1f}s)")

    # ── Step 4: Video ─────────────────────────────────────────────
    if dry_run_full:
        log.info("[pipeline] DRY RUN FULL — skipping video generation.")
    else:
        log.info("[pipeline] Generating video...")
        api_key, base_url, model = _provider_config("grok")  # scene planner always uses Grok

        # Check if clips already exist — skip expensive clip generation if so
        existing_clips = sorted(output_dir.glob("clip_*.mp4"))
        skip_clips = len(existing_clips) > 0
        if skip_clips:
            log.info(f"[pipeline] Found {len(existing_clips)} existing clips — skipping generation")

        # Skip assembly if final.mp4 already exists — go straight to captioning
        skip_assemble = (output_dir / "final.mp4").exists()
        if skip_assemble:
            log.info("[pipeline] final.mp4 exists — skipping assembly, captioning only")

        # Skip karaoke burn if final_captioned.mp4 already exists — only run the TikTok hook burn
        skip_karaoke = (output_dir / "final_captioned.mp4").exists()
        if skip_karaoke:
            log.info("[pipeline] final_captioned.mp4 exists — skipping karaoke burn, hook-caption only")

        video = await generate_video(
            lyrics=lyrics,
            audio_path=audio.path,
            audio_duration=audio.duration_seconds,
            headline=story.headline,
            output_dir=output_dir,
            pollo_api_key=POLLO_API_KEY,
            image_url=getattr(story, "image_url", None),
            summary=story.summary,
            llm_api_key=api_key,
            llm_model=model,
            llm_base_url=base_url,
            video_model=video_model,
            skip_clips=skip_clips,
            skip_assemble=skip_assemble,
            skip_karaoke=skip_karaoke,
            hook_caption=lyrics.hook_caption,
        )
        log.info(f"[pipeline] Video: {video.path}")

    # ── Step 5: Publish ───────────────────────────────────────────
    if dry_run:
        log.info("[pipeline] DRY RUN — skipping TikTok publish.")
        if not dry_run_full:
            log.info(f"[pipeline] Video ready at: {video.path}")
            log.info(f"[pipeline] Caption:\n{lyrics.caption}")
        return

    log.info("[pipeline] Publishing to TikTok...")
    result = await publish_video(
        video_path=video.path,
        caption=lyrics.caption,
        client_key=TIKTOK_CLIENT_KEY,
        client_secret=TIKTOK_CLIENT_SECRET,
        refresh_token=TIKTOK_REFRESH_TOKEN,
    )
    log.info(f"[pipeline] Posted! Publish ID: {result.publish_id}")


async def run(dry_run: bool = False, dry_run_full: bool = False, lyrics_only: bool = False, provider: str = "ollama", headline: str | None = None, summary: str | None = None, video_model: str | None = None, max_stories: int = 3) -> None:
    today = date.today().isoformat()
    day_dir = OUTPUT_DIR / today
    day_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(f"Pipeline starting — {today}{' [DRY RUN]' if dry_run else ''}")
    log.info("=" * 60)

    # ── Step 1: News — fetch, classify all, select top flagged ───
    if headline and summary:
        log.info(f"[pipeline] Using provided headline: {headline}")
        from modules.news_fetcher import NewsStory
        story = NewsStory(headline=headline, summary=summary, url="", source="manual")
        stories_to_run = [(story, "")]
    else:
        log.info("[pipeline] Fetching news candidates...")
        stories = await fetch_news_candidates(NEWS_API_KEY, NEWS_COUNTRY)
        from modules.news_fetcher import log_new_candidates
        log_new_candidates(stories)
        log.info("[pipeline] Scoring candidates...")
        scored = await score_candidates(stories)
        log.info(f"[pipeline] Classifying {len(scored)} candidates (dual: gemma3 + grok)...")
        import os
        grok_key = os.environ.get("XAI_API_KEY", "")
        raw_classifications = await asyncio.gather(
            *[
                classify_story_dual(
                    story=s.story,
                    ollama_model=OLLAMA_MODEL,
                    ollama_base_url=OLLAMA_BASE_URL,
                    grok_api_key=grok_key,
                )
                for s in scored
            ],
            return_exceptions=True,
        )
        classifications = [None if isinstance(r, Exception) else r for r in raw_classifications]
        log.info(f"[pipeline] Flagging stories from {len(scored)} candidates...")
        candidates = score_and_flag(scored, classifications)
        flagged = [c for c in candidates if c.flagged]

        if not flagged:
            log.warning("[pipeline] No flagged stories found — nothing to do.")
            return

        top = flagged[:max_stories]
        log.info(f"[pipeline] {len(flagged)} flagged stories — processing top {len(top)} (max {max_stories}):")
        for i, c in enumerate(top, 1):
            log.info(f"  {i}. [{c.combined_score:.3f}] {c.story.headline[:70]}")

        stories_to_run = [
            (c.story, c.classification.angle if c.classification else "")
            for c in top
        ]

    # ── Process each selected story ───────────────────────────────
    for i, (story, angle) in enumerate(stories_to_run, 1):
        log.info(f"\n[pipeline] ── Story {i}/{len(stories_to_run)} ──────────────────────────")
        try:
            await _run_one_story(
                story=story,
                angle=angle,
                day_dir=day_dir,
                dry_run=dry_run,
                dry_run_full=dry_run_full,
                lyrics_only=lyrics_only,
                provider=provider,
                video_model=video_model,
            )
        except Exception as exc:
            import traceback
            log.error(f"[pipeline] Story failed — skipping: {exc}")
            log.error(f"[pipeline] Traceback: {traceback.format_exc()}")

    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="TikTok News Song Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Skip TikTok posting")
    parser.add_argument("--dry-run-full", action="store_true", help="Skip Suno, video, and TikTok (test news+lyrics only)")
    parser.add_argument("--lyrics-only", action="store_true", help="Only fetch news and generate lyrics, then stop")
    parser.add_argument("--all-flagged", action="store_true", help="Generate lyrics for all of today's flagged stories (skips those already done)")
    parser.add_argument("--all-flagged-music", action="store_true", help="Generate lyrics + music for all of today's flagged stories (skips stories that already have a song)")
    parser.add_argument("--provider", type=str, default="grok", choices=["ollama", "grok"],
                        help="LLM provider for lyrics (default: grok)")
    parser.add_argument("--headline", type=str, default=None, help="Override news headline")
    parser.add_argument("--summary", type=str, default=None, help="Override news summary")
    parser.add_argument("--video-model", type=str, default=None,
                        help="Pollo video model (e.g. seedance-pro-1-5, veo3-1-fast). Default: seedance-pro-1-5")
    parser.add_argument("--max-stories", type=int, default=3,
                        help="Maximum number of top flagged stories to process per run (default: 3)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all_flagged:
        asyncio.run(run_all_flagged(provider=args.provider))
        return

    if args.all_flagged_music:
        asyncio.run(run_all_flagged_music(provider=args.provider))
        return

    asyncio.run(run(
        dry_run=args.dry_run or args.dry_run_full or args.lyrics_only,
        dry_run_full=args.dry_run_full or args.lyrics_only,
        lyrics_only=args.lyrics_only,
        provider=args.provider,
        headline=args.headline,
        summary=args.summary,
        video_model=args.video_model,
        max_stories=args.max_stories,
    ))


if __name__ == "__main__":
    main()
