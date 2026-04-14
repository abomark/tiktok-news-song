"""
Scores news headline candidates using three social signals:
  - Reddit   (r/news, r/worldnews, r/politics — upvotes + comments)
  - Hacker News (points + comments via Algolia)
  - Google Trends (1-day interest for top keywords)

Each signal is normalised 0–100, then blended by configurable weights.
Results are logged to logs/social_scores.jsonl (deduped by headline+date).

Join key for left-joining with story_classifications.jsonl: (headline, date)
"""

from __future__ import annotations
import asyncio
import json
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import NamedTuple

import httpx

from modules.news_fetcher import NewsStory

log = logging.getLogger(__name__)

_LOGS_DIR = Path(__file__).parent.parent / "logs"

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
    "of", "and", "or", "but", "for", "with", "as", "by", "from", "that",
    "this", "it", "be", "has", "have", "had", "will", "would", "could",
    "should", "its", "their", "his", "her", "our", "your", "into", "after",
    "over", "about", "up", "out", "not", "no", "new", "more",
}


class ScoredStory(NamedTuple):
    story: NewsStory
    score: float           # blended 0–100
    reddit_score: float = 0.0
    hn_score: float = 0.0
    trends_score: float = 0.0


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

def _extract_keywords(headline: str, max_words: int = 5) -> str:
    words = re.sub(r"[^a-zA-Z0-9 ]", "", headline).split()
    keywords = [w for w in words if w.lower() not in _STOP_WORDS]
    return " ".join(keywords[:max_words])


# ---------------------------------------------------------------------------
# Individual scorers — each returns a raw float, 0.0 on any failure
# ---------------------------------------------------------------------------

async def _score_reddit(headline: str) -> float:
    query = _extract_keywords(headline)
    if not query:
        return 0.0
    url = "https://www.reddit.com/search.json"
    params = {"q": query, "sort": "top", "t": "day", "limit": 10}
    headers = {"User-Agent": "tiktok-news-scorer/1.0 (pipeline)"}
    subreddits = {"news", "worldnews", "politics"}
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            children = resp.json().get("data", {}).get("children", [])
            total = 0.0
            for child in children:
                d = child.get("data", {})
                if d.get("subreddit", "").lower() in subreddits:
                    total += d.get("score", 0) + d.get("num_comments", 0)
            return total
    except Exception as e:
        log.warning(f"[scorer/reddit] {e}")
        return 0.0


async def _score_hn(headline: str) -> float:
    query = _extract_keywords(headline)
    if not query:
        return 0.0
    url = "https://hn.algolia.com/api/v1/search"
    params = {"query": query, "tags": "story", "hitsPerPage": 10}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            hits = resp.json().get("hits", [])
            return float(sum(h.get("points", 0) + h.get("num_comments", 0) for h in hits))
    except Exception as e:
        log.warning(f"[scorer/hn] {e}")
        return 0.0


async def _score_trends(headline: str) -> float:
    keywords = _extract_keywords(headline, max_words=3).split()
    if not keywords:
        return 0.0

    def _sync_fetch() -> float:
        try:
            from pytrends.request import TrendReq
        except ImportError:
            return 0.0
        try:
            pt = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
            pt.build_payload(keywords[:5], cat=0, timeframe="now 1-d", geo="US")
            df = pt.interest_over_time()
            if df.empty:
                return 0.0
            cols = [c for c in keywords[:5] if c in df.columns]
            if not cols:
                return 0.0
            return float(df[cols].values.mean())
        except Exception as e:
            log.warning(f"[scorer/trends] {e}")
            return 0.0

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_fetch)


# ---------------------------------------------------------------------------
# Normalisation and blending
# ---------------------------------------------------------------------------

def _normalize(scores: list[float]) -> list[float]:
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi == lo:
        # All raw scores identical — no signal to differentiate on.
        # Return 0 if everyone scored zero, 50 if they all tied at a non-zero value.
        return [0.0 if hi == 0.0 else 50.0] * len(scores)
    return [100.0 * (s - lo) / (hi - lo) for s in scores]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_social_scores(scored: list[ScoredStory]) -> None:
    """Append to logs/social_scores.jsonl, deduped by (headline, date)."""
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _LOGS_DIR / "social_scores.jsonl"
    today = date.today().isoformat()
    now = datetime.now().isoformat()

    known: set[tuple[str, str]] = set()
    if log_file.exists():
        for line in log_file.read_text(encoding="utf-8").strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    e = json.loads(line)
                    known.add((e.get("headline", ""), e.get("date", "")))
                except json.JSONDecodeError:
                    pass

    new_entries = []
    for s in scored:
        key = (s.story.headline, today)
        if key in known:
            continue
        new_entries.append({
            "date": today,
            "timestamp": now,
            "headline": s.story.headline,
            "source": s.story.source,
            "social_score": round(s.score, 2),
            "reddit_score": round(s.reddit_score, 2),
            "hn_score": round(s.hn_score, 2),
            "trends_score": round(s.trends_score, 2),
        })
        known.add(key)

    if new_entries:
        with open(log_file, "a", encoding="utf-8") as f:
            for entry in new_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        log.info(f"[scorer] Logged {len(new_entries)} social scores")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def score_candidates(
    stories: list[NewsStory],
    weights: dict[str, float] | None = None,
) -> list[ScoredStory]:
    """Score a list of NewsStory objects across Reddit, HN, and Google Trends.

    Returns ScoredStory items sorted in original order.
    Logs results to logs/social_scores.jsonl.
    """
    if weights is None:
        import sys, os
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import SCORING_WEIGHTS
        weights = SCORING_WEIGHTS

    if len(stories) == 1:
        log.info("[scorer] Single story — skipping social scoring")
        return [ScoredStory(story=stories[0], score=0.0)]

    log.info(f"[scorer] Scoring {len(stories)} candidates across Reddit, HN, and Google Trends...")

    reddit_raw, hn_raw, trends_raw = await asyncio.gather(
        asyncio.gather(*[_score_reddit(s.headline) for s in stories]),
        asyncio.gather(*[_score_hn(s.headline) for s in stories]),
        asyncio.gather(*[_score_trends(s.headline) for s in stories]),
    )

    reddit_norm = _normalize(list(reddit_raw))
    hn_norm     = _normalize(list(hn_raw))
    trends_norm = _normalize(list(trends_raw))

    w_r = weights.get("reddit", 0.40)
    w_h = weights.get("hn",     0.30)
    w_t = weights.get("trends", 0.30)
    total_w = w_r + w_h + w_t

    scored: list[ScoredStory] = []
    for i, story in enumerate(stories):
        final = (w_r * reddit_norm[i] + w_h * hn_norm[i] + w_t * trends_norm[i]) / total_w
        scored.append(ScoredStory(
            story=story,
            score=final,
            reddit_score=reddit_norm[i],
            hn_score=hn_norm[i],
            trends_score=trends_norm[i],
        ))
        log.debug(
            f"[scorer] '{story.headline[:60]}' "
            f"reddit={reddit_norm[i]:.1f} hn={hn_norm[i]:.1f} "
            f"trends={trends_norm[i]:.1f} => {final:.1f}"
        )

    _log_social_scores(scored)
    return scored


if __name__ == "__main__":
    import asyncio, os
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    from modules.news_fetcher import fetch_news_candidates

    async def main():
        stories = await fetch_news_candidates(os.getenv("NEWS_API_KEY", ""))
        scored = await score_candidates(stories)
        for s in sorted(scored, key=lambda x: x.score, reverse=True):
            print(f"{s.score:5.1f}  reddit={s.reddit_score:5.1f}  hn={s.hn_score:5.1f}  trends={s.trends_score:5.1f}  {s.story.headline[:70]}")

    asyncio.run(main())
