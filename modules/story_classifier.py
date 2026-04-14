"""
Classifies a news story for satirical TikTok music video potential.

Scores 10 virality factors (1–10 each) and computes a Viral Potential Index (VPI).
Results are appended to logs/story_classifications.jsonl.
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path

import openai

from modules.news_fetcher import NewsStory

log = logging.getLogger(__name__)

_LOGS_DIR = Path(__file__).parent.parent / "logs"

_PROMPT_FILE = Path(__file__).parent.parent / "assets" / "classifier_prompt.md"

def _load_prompts() -> tuple[str, str]:
    """Load system prompt and user template from assets/classifier_prompt.md."""
    text = _PROMPT_FILE.read_text(encoding="utf-8")
    parts = text.split("\n---\n", 1)
    if len(parts) != 2:
        raise ValueError(f"classifier_prompt.md must contain exactly one '---' separator, got {len(parts)} parts")
    system = parts[0].removeprefix("# SYSTEM\n").strip()
    user = parts[1].removeprefix("# USER\n").strip()
    return system, user

_SYSTEM_PROMPT, _USER_TEMPLATE = _load_prompts()

_FACTOR_LABELS = {
    "absurdity": "Absurdity Score",
    "character_punchability": "Character Punchability",
    "cultural_reach": "Cultural Reach",
    "emotional_heat": "Emotional Heat",
    "memeability": "Memeability",
    "musical_fit": "Musical Fit",
    "timestamp_sensitivity": "Timestamp Sensitivity",
    "moral_clarity": "Moral Clarity",
    "visual_potential": "Visual Potential",
    "safe_harbor": "Safe Harbor",
}


@dataclass
class FactorScore:
    score: int
    rationale: str


@dataclass
class StoryClassification:
    headline: str
    summary: str
    source: str
    url: str
    timestamp: str
    run_dir: str | None

    absurdity: FactorScore = field(default_factory=lambda: FactorScore(0, ""))
    character_punchability: FactorScore = field(default_factory=lambda: FactorScore(0, ""))
    cultural_reach: FactorScore = field(default_factory=lambda: FactorScore(0, ""))
    emotional_heat: FactorScore = field(default_factory=lambda: FactorScore(0, ""))
    memeability: FactorScore = field(default_factory=lambda: FactorScore(0, ""))
    musical_fit: FactorScore = field(default_factory=lambda: FactorScore(0, ""))
    timestamp_sensitivity: FactorScore = field(default_factory=lambda: FactorScore(0, ""))
    moral_clarity: FactorScore = field(default_factory=lambda: FactorScore(0, ""))
    visual_potential: FactorScore = field(default_factory=lambda: FactorScore(0, ""))
    safe_harbor: FactorScore = field(default_factory=lambda: FactorScore(0, ""))

    angle: str = ""
    vpi: float = 0.0

    @property
    def vpi_label(self) -> str:
        if self.vpi >= 8:
            return "Drop everything, make this now"
        if self.vpi >= 6:
            return "Strong candidate, needs the right angle"
        if self.vpi >= 4:
            return "Workable with the right hook"
        return "Tough sell — niche or too divisive"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["vpi_label"] = self.vpi_label
        return d


def _extract_json(raw: str) -> str:
    """Strip markdown code fences and surrounding whitespace from model output."""
    text = raw.strip()
    # Remove ```json ... ``` or ``` ... ``` wrappers
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text
        # Remove language tag on first line (e.g. "json\n{...")
        if "\n" in text:
            first, rest = text.split("\n", 1)
            if not first.strip().startswith("{"):
                text = rest
        # Remove trailing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _parse_response(raw: str, story: NewsStory, run_dir: str | None) -> StoryClassification:
    """Parse model JSON reply into a StoryClassification."""
    cleaned = _extract_json(raw)
    if not cleaned:
        raise ValueError(f"Empty response from model for headline: {story.headline[:60]!r}")
    log.debug(f"[classifier] Parsing JSON ({len(cleaned)} chars): {cleaned[:120]}")
    data = json.loads(cleaned)

    factors = {}
    for key in _FACTOR_LABELS:
        entry = data.get(key, {})
        factors[key] = FactorScore(
            score=int(entry.get("score", 0)),
            rationale=entry.get("rationale", ""),
        )

    scores = [f.score for f in factors.values()]
    vpi = round(sum(scores) / len(scores), 2) if scores else 0.0

    return StoryClassification(
        headline=story.headline,
        summary=story.summary,
        source=story.source,
        url=story.url,
        timestamp=datetime.now().isoformat(),
        run_dir=run_dir,
        angle=data.get("angle", ""),
        vpi=vpi,
        **factors,
    )


def _load_cached(headline: str, today: str) -> StoryClassification | None:
    """Return an existing classification for this headline+date, or None."""
    log_file = _LOGS_DIR / "story_classifications.jsonl"
    if not log_file.exists():
        return None
    for line in log_file.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            if entry.get("headline") == headline and entry.get("date") == today:
                # Reconstruct from stored dict
                factors = {
                    k: FactorScore(
                        score=entry[k]["score"],
                        rationale=entry[k]["rationale"],
                    )
                    for k in _FACTOR_LABELS
                    if k in entry and isinstance(entry[k], dict)
                }
                return StoryClassification(
                    headline=entry["headline"],
                    summary=entry.get("summary", ""),
                    source=entry.get("source", ""),
                    url=entry.get("url", ""),
                    timestamp=entry.get("timestamp", ""),
                    run_dir=entry.get("run_dir"),
                    angle=entry.get("angle", ""),
                    vpi=entry.get("vpi", 0.0),
                    **factors,
                )
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def _append_log(classification: StoryClassification, today: str) -> None:
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _LOGS_DIR / "story_classifications.jsonl"
    entry = classification.to_dict()
    entry["date"] = today
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


async def classify_story(
    story: NewsStory,
    api_key: str = "ollama",
    model: str = "gemma3",
    base_url: str = "http://localhost:11434/v1",
    run_dir: str | None = None,
) -> StoryClassification:
    """
    Score the story on all 10 virality factors using a local Ollama model.

    Returns cached result if this headline was already classified today.
    Appends new results to logs/story_classifications.jsonl.
    """
    today = date.today().isoformat()

    cached = _load_cached(story.headline, today)
    if cached is not None:
        log.info(f"[classifier] Cache hit for today — skipping model: {story.headline[:70]}")
        return cached

    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    user_prompt = (
        _USER_TEMPLATE
        .replace("{headline}", story.headline)
        .replace("{summary}", story.summary)
    )

    log.info(f"[classifier] Classifying: {story.headline[:80]}")

    response = await client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content
    log.debug(f"[classifier] Raw content type={type(content)} value={content!r:.200}")
    raw = (content or "").strip()
    log.debug(f"[classifier] Raw response: {raw[:200]}")

    classification = _parse_response(raw, story, run_dir)

    log.info(
        f"[classifier] VPI={classification.vpi:.1f} ({classification.vpi_label}) — {story.headline[:60]}"
    )

    _append_log(classification, today)
    return classification
