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
    RUNWAY_API_KEY,
)
from modules.news_fetcher import fetch_news_candidates
from modules.social_scorer import score_candidates
from modules.story_classifier import classify_story
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
        logging.FileHandler(OUTPUT_DIR / "pipeline.log"),
    ],
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

    (output_dir / "headline.txt").write_text(
        f"{headline}\n\n{summary}\n{url}", encoding="utf-8"
    )
    (output_dir / "lyrics.txt").write_text(
        f"TITLE: {lyrics.title}\n\n{lyrics.full_text}\n\n---\nCAPTION:\n{lyrics.caption}",
        encoding="utf-8",
    )
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


async def run(dry_run: bool = False, dry_run_full: bool = False, lyrics_only: bool = False, provider: str = "ollama", headline: str | None = None, summary: str | None = None) -> None:
    today = date.today().isoformat()
    day_dir = OUTPUT_DIR / today
    day_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(f"Pipeline starting — {today}{' [DRY RUN]' if dry_run else ''}")
    log.info("=" * 60)

    # ── Step 1: News — fetch, classify all, select best ──────────
    flagged: list = []
    if headline and summary:
        log.info(f"[pipeline] Using provided headline: {headline}")
        from modules.news_fetcher import NewsStory
        story = NewsStory(headline=headline, summary=summary, url="", source="manual")
    else:
        log.info("[pipeline] Fetching news candidates...")
        stories = await fetch_news_candidates(NEWS_API_KEY, NEWS_COUNTRY)
        log.info("[pipeline] Scoring candidates...")
        scored = await score_candidates(stories)
        log.info(f"[pipeline] Classifying {len(scored)} candidates...")
        raw_classifications = await asyncio.gather(
            *[
                classify_story(
                    story=s.story,
                    model=OLLAMA_MODEL,
                    base_url=OLLAMA_BASE_URL,
                )
                for s in scored
            ],
            return_exceptions=True,
        )
        classifications = [None if isinstance(r, Exception) else r for r in raw_classifications]
        log.info(f"[pipeline] Flagging stories from {len(scored)} candidates...")
        candidates = score_and_flag(scored, classifications)
        flagged = [c for c in candidates if c.flagged]
        story = flagged[0].story  # top-scoring flagged story

    # Grab the angle from the flagged story (empty for manual --headline runs)
    angle = ""
    if flagged:
        angle = flagged[0].classification.angle if flagged[0].classification else ""

    log.info(f"[pipeline] Story: {story.headline}")
    if angle:
        log.info(f"[pipeline] Satirical angle: {angle}")

    # ── Check: skip if lyrics already generated for this headline today ───────
    if _lyrics_exist_for(story.headline, day_dir):
        log.info("[pipeline] Lyrics already exist for this story — skipping.")
        return

    # ── Step 2: Lyrics ────────────────────────────────────────────
    log.info("[pipeline] Generating lyrics...")
    lyrics, output_dir = await _generate_and_save_lyrics(
        headline=story.headline,
        summary=story.summary,
        angle=angle,
        url=story.url,
        day_dir=day_dir,
        provider=provider,
    )

    if lyrics_only:
        log.info(f"[pipeline] LYRICS ONLY — done. Output: {output_dir}")
        return

    # ── Step 3: Music ─────────────────────────────────────────────
    if dry_run_full:
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
        video = await generate_video(
            lyrics=lyrics,
            audio_path=audio.path,
            audio_duration=audio.duration_seconds,
            headline=story.headline,
            output_dir=output_dir,
            runway_api_key=RUNWAY_API_KEY,
        )
        log.info(f"[pipeline] Video: {video.path}")

    # ── Step 5: Publish ───────────────────────────────────────────
    if dry_run:
        log.info("[pipeline] DRY RUN — skipping TikTok publish.")
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
    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="TikTok News Song Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Skip TikTok posting")
    parser.add_argument("--dry-run-full", action="store_true", help="Skip Suno, video, and TikTok (test news+lyrics only)")
    parser.add_argument("--lyrics-only", action="store_true", help="Only fetch news and generate lyrics, then stop")
    parser.add_argument("--all-flagged", action="store_true", help="Generate lyrics for all of today's flagged stories (skips those already done)")
    parser.add_argument("--provider", type=str, default="ollama", choices=["ollama", "grok"],
                        help="LLM provider for lyrics (default: ollama)")
    parser.add_argument("--headline", type=str, default=None, help="Override news headline")
    parser.add_argument("--summary", type=str, default=None, help="Override news summary")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all_flagged:
        asyncio.run(run_all_flagged(provider=args.provider))
        return

    asyncio.run(run(
        dry_run=args.dry_run or args.dry_run_full or args.lyrics_only,
        dry_run_full=args.dry_run_full or args.lyrics_only,
        lyrics_only=args.lyrics_only,
        provider=args.provider,
        headline=args.headline,
        summary=args.summary,
    ))


if __name__ == "__main__":
    main()
