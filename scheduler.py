"""
Scheduler for the CurrentNoise pipeline.

Hourly job (runs on startup, then every hour):
  1. Fetch news headlines — log new ones to news_candidates.jsonl
  2. Social score  — score any candidate not yet in social_scores.jsonl
  3. Classify      — VPI-classify any candidate not yet in story_classifications.jsonl
  4. Flag          — re-run score_and_flag to refresh logs/flagged_stories.jsonl

Daily job at 09:00:
  Full pipeline (news → lyrics → music → video → TikTok)

Usage:
    python scheduler.py

For Windows Task Scheduler or Linux cron, use pipeline.py directly instead:
    # Linux cron (crontab -e):
    0 9 * * * cd /path/to/tiktok-news-song && python pipeline.py >> output/pipeline.log 2>&1

    # Windows Task Scheduler:
    Action: python C:\\path\\to\\tiktok-news-song\\pipeline.py
    Trigger: Daily at 09:00
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

import schedule
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
LOG_FILE = BASE_DIR / "output" / "scheduler.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE),
    ],
)
log = logging.getLogger(__name__)

PIPELINE_RUN_TIME = "09:00"   # 24-hour local time


# ---------------------------------------------------------------------------
# Helpers — read logged headlines from a jsonl file
# ---------------------------------------------------------------------------

def _logged_headlines(jsonl_path: Path) -> set[str]:
    """Return the set of headlines already present in a log file."""
    if not jsonl_path.exists():
        return set()
    seen: set[str] = set()
    for line in jsonl_path.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if line:
            try:
                seen.add(json.loads(line).get("headline", ""))
            except json.JSONDecodeError:
                pass
    return seen


def _load_jsonl(jsonl_path: Path) -> list[dict]:
    if not jsonl_path.exists():
        return []
    entries = []
    for line in jsonl_path.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


# ---------------------------------------------------------------------------
# Hourly job: fetch → score → classify
# ---------------------------------------------------------------------------

def hourly_job() -> None:
    """Fetch news, then score, classify, and flag any candidates not yet processed."""
    log.info("[scheduler] --- Hourly job starting ---")
    try:
        asyncio.run(_hourly_async())
    except Exception as e:
        log.error(f"[scheduler] Hourly job failed: {e}", exc_info=True)
    log.info("[scheduler] --- Hourly job done ---")


async def _hourly_async() -> None:
    from modules.news_fetcher import fetch_news_candidates, log_new_candidates, NewsStory
    from modules.social_scorer import score_candidates
    from modules.story_classifier import classify_story

    news_api_key = os.getenv("NEWS_API_KEY", "")
    country = os.getenv("NEWS_COUNTRY", "us")
    ollama_model = os.getenv("OLLAMA_MODEL", "gemma3")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    # ── Step 1: Fetch ──────────────────────────────────────────────────────────
    log.info("[scheduler] Step 1/3 — Fetching news...")
    stories = await fetch_news_candidates(news_api_key, country)
    log_new_candidates(stories)

    # Build a full NewsStory list from the entire candidates log so steps 2 and 3
    # can catch any headline that was logged previously but never scored/classified.
    all_candidates: list[NewsStory] = [
        NewsStory(
            headline=e["headline"],
            summary=e.get("summary", ""),
            url=e.get("url", ""),
            source=e.get("source", ""),
            published_at=e.get("published_at"),
        )
        for e in _load_jsonl(LOGS_DIR / "news_candidates.jsonl")
        if e.get("headline")
    ]
    # Dedupe by headline while preserving order (keep first occurrence)
    seen_hl: set[str] = set()
    unique_candidates: list[NewsStory] = []
    for s in all_candidates:
        if s.headline not in seen_hl:
            unique_candidates.append(s)
            seen_hl.add(s.headline)

    log.info(f"[scheduler] {len(unique_candidates)} total unique candidates in log")

    # ── Step 2: Social score any candidate not yet in social_scores.jsonl ─────
    scored_headlines = _logged_headlines(LOGS_DIR / "social_scores.jsonl")
    unscored = [s for s in unique_candidates if s.headline not in scored_headlines]

    if unscored:
        log.info(f"[scheduler] Step 2/3 — Scoring {len(unscored)} unscored candidates...")
        await score_candidates(unscored)
    else:
        log.info("[scheduler] Step 2/3 — All candidates already scored, skipping.")

    # ── Step 3: Classify any candidate not yet classified today ───────────────
    # Key: (headline, date) — same story on a new day gets re-classified
    today = date.today().isoformat()
    classified_keys: set[tuple[str, str]] = {
        (e.get("headline", ""), e.get("date", ""))
        for e in _load_jsonl(LOGS_DIR / "story_classifications.jsonl")
    }
    unclassified = [s for s in unique_candidates if (s.headline, today) not in classified_keys]

    if unclassified:
        log.info(f"[scheduler] Step 3/4 — Classifying {len(unclassified)} unclassified candidates...")
        results = await asyncio.gather(
            *[
                classify_story(
                    story=s,
                    model=ollama_model,
                    base_url=ollama_base_url,
                )
                for s in unclassified
            ],
            return_exceptions=True,
        )
        n_ok = sum(1 for r in results if not isinstance(r, Exception))
        n_fail = len(results) - n_ok
        log.info(f"[scheduler] Step 3/4 — Classified {n_ok}/{len(unclassified)}" +
                 (f", {n_fail} failed" if n_fail else ""))
    else:
        log.info("[scheduler] Step 3/4 — All candidates already classified, skipping.")

    # ── Step 4: Re-flag all today's candidates ─────────────────────────────────
    # Rebuild ScoredStory + StoryClassification lists from logs and re-run flagging
    # so flagged_stories.jsonl is always fresh after each hourly enrichment pass.
    log.info("[scheduler] Step 4/4 — Refreshing flagged stories...")
    from modules.social_scorer import ScoredStory
    from modules.story_classifier import StoryClassification, FactorScore, _FACTOR_LABELS
    from modules.story_selector import score_and_flag
    from modules.news_fetcher import NewsStory

    social_by_headline: dict[str, dict] = {
        e["headline"]: e
        for e in _load_jsonl(LOGS_DIR / "social_scores.jsonl")
        if e.get("headline")
    }
    clf_by_headline: dict[str, dict] = {}
    for e in _load_jsonl(LOGS_DIR / "story_classifications.jsonl"):
        hl = e.get("headline", "")
        if hl and e.get("date") == today:
            clf_by_headline[hl] = e

    scored_today: list[ScoredStory] = []
    classifications_today: list[StoryClassification | None] = []

    for s in unique_candidates:
        social = social_by_headline.get(s.headline)
        if social is None:
            continue  # not yet scored — skip
        scored_today.append(ScoredStory(
            story=s,
            score=social.get("social_score", 0.0),
            reddit_score=social.get("reddit_score", 0.0),
            hn_score=social.get("hn_score", 0.0),
            trends_score=social.get("trends_score", 0.0),
        ))
        clf_entry = clf_by_headline.get(s.headline)
        if clf_entry is None:
            classifications_today.append(None)
        else:
            try:
                factors = {
                    k: FactorScore(
                        score=clf_entry[k]["score"],
                        rationale=clf_entry[k]["rationale"],
                    )
                    for k in _FACTOR_LABELS
                    if k in clf_entry and isinstance(clf_entry[k], dict)
                }
                classifications_today.append(StoryClassification(
                    headline=clf_entry["headline"],
                    summary=clf_entry.get("summary", ""),
                    source=clf_entry.get("source", ""),
                    url=clf_entry.get("url", ""),
                    timestamp=clf_entry.get("timestamp", ""),
                    run_dir=clf_entry.get("run_dir"),
                    angle=clf_entry.get("angle", ""),
                    vpi=clf_entry.get("vpi", 0.0),
                    **factors,
                ))
            except (KeyError, TypeError):
                classifications_today.append(None)

    if scored_today:
        score_and_flag(scored_today, classifications_today)
        log.info(f"[scheduler] Step 4/4 — Flagging done ({len(scored_today)} candidates)")
    else:
        log.info("[scheduler] Step 4/4 — No scored candidates yet, skipping flagging.")


# ---------------------------------------------------------------------------
# Daily full pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    log.info(f"[scheduler] Triggering pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "pipeline.py")],
        cwd=str(BASE_DIR),
        capture_output=False,
    )
    if result.returncode == 0:
        log.info("[scheduler] Pipeline completed successfully.")
    else:
        log.error(f"[scheduler] Pipeline FAILED with exit code {result.returncode}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    log.info(f"[scheduler] Starting — hourly enrichment + daily pipeline at {PIPELINE_RUN_TIME}")
    log.info("[scheduler] Press Ctrl+C to stop.")

    schedule.every().hour.do(hourly_job)
    schedule.every().day.at(PIPELINE_RUN_TIME).do(run_pipeline)

    # Run immediately on startup
    hourly_job()

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
