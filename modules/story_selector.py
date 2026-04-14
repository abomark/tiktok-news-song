"""
Blends social scores and VPI classifications to flag the best stories.

Inputs (both pre-computed upstream):
  - list[ScoredStory]             from social_scorer
  - list[StoryClassification | None]  from story_classifier (None = failed)

Final score = SOCIAL_WEIGHT * social_norm + VPI_WEIGHT * vpi_norm
              where both inputs are normalised to 0–1 before blending.

All stories at or above `threshold` (default 0.5) are flagged.
If nothing clears the threshold, the single highest-scored story is flagged
so the pipeline always has something to work with.

Outputs written to logs/:
  flagged_stories.jsonl   — one entry per flagged story per run (pipeline reads this)
  selection_decisions.jsonl — full candidate ranking per run (audit trail)
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from modules.news_fetcher import NewsStory
from modules.social_scorer import ScoredStory
from modules.story_classifier import StoryClassification

log = logging.getLogger(__name__)

_LOGS_DIR = Path(__file__).parent.parent / "logs"

# Blend weights — must sum to 1.0
SOCIAL_WEIGHT = 0.40
VPI_WEIGHT    = 0.60

# Default combined-score threshold for flagging (0–1)
DEFAULT_THRESHOLD = 0.5


@dataclass
class CandidateResult:
    story: NewsStory
    social_score: float          # raw 0–100 from social_scorer
    reddit_score: float
    hn_score: float
    trends_score: float
    social_norm: float           # min-max normalised 0–1
    vpi_norm: float              # min-max normalised 0–1
    classification: StoryClassification | None
    combined_score: float        # blended 0–1
    flagged: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _load_jsonl(log_file: Path) -> list[dict]:
    if not log_file.exists():
        return []
    entries = []
    for line in log_file.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def _append_line(log_file: Path, entry: dict) -> None:
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _log_flagged(candidates: list[CandidateResult], today: str, timestamp: str, threshold: float) -> None:
    """Append flagged stories to logs/flagged_stories.jsonl (deduped by headline+date)."""
    log_file = _LOGS_DIR / "flagged_stories.jsonl"
    existing_keys: set[tuple[str, str]] = {
        (e.get("headline", ""), e.get("date", ""))
        for e in _load_jsonl(log_file)
    }
    for c in candidates:
        if not c.flagged:
            continue
        key = (c.story.headline, today)
        if key in existing_keys:
            continue
        entry = {
            "date": today,
            "timestamp": timestamp,
            "headline": c.story.headline,
            "source": c.story.source,
            "url": c.story.url,
            "summary": c.story.summary,
            "combined_score": c.combined_score,
            "social_score": c.social_score,
            "vpi": c.classification.vpi if c.classification else None,
            "vpi_label": c.classification.vpi_label if c.classification else None,
            "angle": c.classification.angle if c.classification else None,
            "threshold": threshold,
        }
        _append_line(log_file, entry)
        existing_keys.add(key)


def _log_decision(candidates: list[CandidateResult], today: str, timestamp: str, threshold: float) -> None:
    """Append full ranked candidate list to logs/selection_decisions.jsonl (audit trail)."""
    log_file = _LOGS_DIR / "selection_decisions.jsonl"
    flagged = [c for c in candidates if c.flagged]

    # Dedupe by flagged-headline set + date — skip if exact same set already logged today
    existing = _load_jsonl(log_file)
    flagged_headlines = sorted(c.story.headline for c in flagged)
    for e in existing:
        if e.get("date") == today and sorted(e.get("flagged_headlines", [])) == flagged_headlines:
            log.debug("[selector] Decision already logged — skipping duplicate")
            return

    entry = {
        "date": today,
        "timestamp": timestamp,
        "threshold": threshold,
        "flagged_headlines": flagged_headlines,
        "n_flagged": len(flagged),
        "candidates": [
            {
                "headline": c.story.headline,
                "source": c.story.source,
                "social_score": c.social_score,
                "reddit_score": c.reddit_score,
                "hn_score": c.hn_score,
                "trends_score": c.trends_score,
                "social_norm": c.social_norm,
                "vpi_norm": c.vpi_norm,
                "vpi": c.classification.vpi if c.classification else None,
                "vpi_label": c.classification.vpi_label if c.classification else None,
                "angle": c.classification.angle if c.classification else None,
                "combined_score": c.combined_score,
                "flagged": c.flagged,
            }
            for c in sorted(candidates, key=lambda c: c.combined_score, reverse=True)
        ],
    }
    _append_line(log_file, entry)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def score_and_flag(
    scored: list[ScoredStory],
    classifications: list[StoryClassification | None],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[CandidateResult]:
    """
    Blend social scores with VPI classifications and flag stories above threshold.

    Args:
        scored:          Output of social_scorer.score_candidates().
        classifications: Output of story_classifier.classify_story() for each story,
                         in the same order as `scored`. Use None for failed classifications.
        threshold:       Combined-score cutoff (0–1). Stories at or above this are flagged.

    Returns:
        All candidates as CandidateResult, sorted by combined_score descending.
        At least one story will always be flagged.
    """
    if len(scored) != len(classifications):
        raise ValueError(
            f"scored ({len(scored)}) and classifications ({len(classifications)}) must have the same length"
        )

    n_failed = sum(1 for c in classifications if c is None)
    if n_failed:
        log.warning(f"[selector] {n_failed}/{len(classifications)} classifications missing — VPI treated as 0 for those")

    # Normalise social scores (0–100 → 0–1)
    social_raw = [s.score for s in scored]
    social_norm = _normalize(social_raw) if len(social_raw) > 1 else [1.0]

    # Normalise VPI scores (1–10 → 0–1); failed classifications = 0
    vpi_raw = [clf.vpi if clf is not None else 0.0 for clf in classifications]
    all_zero = all(v == 0.0 for v in vpi_raw)
    vpi_norm = _normalize(vpi_raw) if (len(vpi_raw) > 1 and not all_zero) else [0.0] * len(vpi_raw)

    # If every classification failed, fall back to pure social signal
    effective_social_w = 1.0 if all_zero else SOCIAL_WEIGHT
    effective_vpi_w    = 0.0 if all_zero else VPI_WEIGHT

    results: list[CandidateResult] = []
    for i, (s, clf) in enumerate(zip(scored, classifications)):
        combined = effective_social_w * social_norm[i] + effective_vpi_w * vpi_norm[i]
        results.append(CandidateResult(
            story=s.story,
            social_score=round(s.score, 2),
            reddit_score=round(s.reddit_score, 2),
            hn_score=round(s.hn_score, 2),
            trends_score=round(s.trends_score, 2),
            social_norm=round(social_norm[i], 4),
            vpi_norm=round(vpi_norm[i], 4),
            classification=clf,
            combined_score=round(combined, 4),
        ))

    results.sort(key=lambda c: c.combined_score, reverse=True)

    # Stories already in flagged_stories.jsonl stay flagged regardless of current scores.
    # This prevents min-max renormalisation from un-flagging a story that cleared the
    # threshold in an earlier hourly run.
    today = date.today().isoformat()
    already_flagged: set[str] = {
        e.get("headline", "")
        for e in _load_jsonl(_LOGS_DIR / "flagged_stories.jsonl")
        if e.get("date") == today
    }

    for c in results:
        if c.story.headline in already_flagged or c.combined_score >= threshold:
            c.flagged = True

    # Always flag at least the top story so the pipeline is never empty-handed
    if not any(c.flagged for c in results):
        log.warning(f"[selector] No story cleared threshold {threshold:.2f} — flagging top story only")
        results[0].flagged = True

    flagged = [c for c in results if c.flagged]
    log.info(
        f"[selector] Flagged {len(flagged)}/{len(results)} stories "
        f"(threshold={threshold:.2f}, weights: social={effective_social_w:.0%} vpi={effective_vpi_w:.0%})"
    )
    for c in flagged:
        vpi_str = f"{c.classification.vpi:.1f}" if c.classification else "n/a"
        log.info(
            f"[selector]   ✓ combined={c.combined_score:.3f} social={c.social_score:.1f} "
            f"vpi={vpi_str} — {c.story.headline[:70]}"
        )

    timestamp = datetime.now().isoformat()
    _log_flagged(results, today, timestamp, threshold)
    _log_decision(results, today, timestamp, threshold)

    return results
