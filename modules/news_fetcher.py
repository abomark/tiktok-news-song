"""
Fetches news headline candidates from external sources.

Primary:  NewsAPI — up to 10 top headlines for a given country
Fallback: Google Trends RSS — single trending topic

Returns plain NewsStory objects with no scoring attached.
Scoring is handled separately by social_scorer.py.
"""

from __future__ import annotations
import json
import logging
import httpx
import feedparser
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

log = logging.getLogger(__name__)

_LOGS_DIR = Path(__file__).parent.parent / "logs"


@dataclass
class NewsStory:
    headline: str
    summary: str
    url: str
    source: str
    published_at: str | None = None   # ISO 8601 from the source, e.g. "2026-04-13T08:00:00Z"


# ---------------------------------------------------------------------------
# NewsAPI
# ---------------------------------------------------------------------------

async def _try_newsapi(api_key: str, country: str) -> list[NewsStory] | None:
    if not api_key:
        return None
    url = "https://newsapi.org/v2/top-headlines"
    params = {"country": country, "pageSize": 10, "apiKey": api_key}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            articles = [a for a in articles if a.get("description") and a.get("title")]
            if not articles:
                return None
            return [
                NewsStory(
                    headline=a["title"].split(" - ")[0].strip(),
                    summary=a.get("description", ""),
                    url=a.get("url", ""),
                    source=a.get("source", {}).get("name", "NewsAPI"),
                    published_at=a.get("publishedAt"),   # e.g. "2026-04-13T08:34:00Z"
                )
                for a in articles
            ]
    except Exception as e:
        log.warning(f"[news] NewsAPI error: {e}")
        return None


# ---------------------------------------------------------------------------
# Google Trends RSS fallback
# ---------------------------------------------------------------------------

async def _try_google_trends(country: str) -> NewsStory | None:
    geo = country.upper()
    rss_url = f"https://trends.google.com/trends/trendingsearches/daily/rss?geo={geo}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(rss_url)
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            entries = feed.entries
            if not entries:
                return None
            top = entries[0]
            title = top.get("title", "Unknown Trend")
            summary = top.get("ht_news_item_snippet", top.get("summary", title))
            url = top.get("ht_news_item_url", top.get("link", ""))

            # feedparser exposes published_parsed as a time.struct_time
            published_at: str | None = None
            if top.get("published_parsed"):
                try:
                    published_at = datetime(*top.published_parsed[:6]).isoformat()
                except Exception:
                    pass

            return NewsStory(
                headline=title,
                summary=summary,
                url=url,
                source="Google Trends",
                published_at=published_at,
            )
    except Exception as e:
        log.warning(f"[news] Google Trends RSS error: {e}")
        return None


# ---------------------------------------------------------------------------
# Candidate log — logs/news_candidates.jsonl
# ---------------------------------------------------------------------------

def _load_seen_headlines() -> set[str]:
    """Return all headlines ever logged, regardless of date."""
    log_file = _LOGS_DIR / "news_candidates.jsonl"
    if not log_file.exists():
        return set()
    seen: set[str] = set()
    for line in log_file.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if line:
            try:
                seen.add(json.loads(line).get("headline", ""))
            except json.JSONDecodeError:
                pass
    return seen


def log_new_candidates(stories: list[NewsStory]) -> list[NewsStory]:
    """Append stories not seen before to logs/news_candidates.jsonl.

    Each record includes:
      - fetched_at   : when this pipeline run fetched the story
      - published_at : when the source says the article was published (may be None)

    Returns only the stories that were new (not previously logged).
    """
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _LOGS_DIR / "news_candidates.jsonl"
    seen = _load_seen_headlines()
    today = date.today().isoformat()
    fetched_at = datetime.now().isoformat()

    new_stories = [s for s in stories if s.headline not in seen]
    if new_stories:
        with open(log_file, "a", encoding="utf-8") as f:
            for s in new_stories:
                f.write(json.dumps({
                    "date": today,
                    "fetched_at": fetched_at,
                    "published_at": s.published_at,
                    "headline": s.headline,
                    "summary": s.summary,
                    "source": s.source,
                    "url": s.url,
                }, ensure_ascii=False) + "\n")
        log.info(f"[news] Logged {len(new_stories)} new headlines ({len(stories) - len(new_stories)} already seen)")
    else:
        log.info(f"[news] No new headlines — all {len(stories)} already logged")

    return new_stories


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def fetch_news_candidates(
    news_api_key: str,
    country: str = "us",
) -> list[NewsStory]:
    """Fetch up to 10 headline candidates.

    Falls back to a single story from Google Trends RSS if NewsAPI fails.
    Returns plain NewsStory objects — call social_scorer.score_candidates() next.
    """
    stories = await _try_newsapi(news_api_key, country)

    if not stories:
        log.warning("[news] NewsAPI failed or no key — falling back to Google Trends RSS")
        story = await _try_google_trends(country)
        if story:
            log.info(f"[news] Google Trends fallback: {story.headline}")
            return [story]
        raise RuntimeError("Could not fetch any news story from any source.")

    log.info(f"[news] Fetched {len(stories)} candidates from NewsAPI")
    return stories


if __name__ == "__main__":
    import asyncio, os
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    async def main():
        stories = await fetch_news_candidates(os.getenv("NEWS_API_KEY", ""))
        for i, s in enumerate(stories, 1):
            print(f"{i:2}. [{s.source}] {s.published_at or 'no date'} — {s.headline}")

    asyncio.run(main())
