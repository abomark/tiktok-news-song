"""
Classifies generated lyrics for TikTok viral potential.

Scores across 4 categories: Hook Mechanics, Cultural Payload, Creator Bait, Platform Risk.
Results are appended to logs/lyrics_classifications.jsonl.
"""

from __future__ import annotations
import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path

import openai

log = logging.getLogger(__name__)

_LOGS_DIR = Path(__file__).parent.parent / "logs"
_PROMPT_FILE = Path(__file__).parent.parent / "assets" / "lyrics_classifier_prompt.md"


def _load_prompts() -> tuple[str, str]:
    text = _PROMPT_FILE.read_text(encoding="utf-8")
    parts = text.split("\n---\n", 1)
    if len(parts) != 2:
        raise ValueError("lyrics_classifier_prompt.md must contain exactly one '---' separator")
    system = parts[0].removeprefix("# SYSTEM\n").strip()
    user = parts[1].removeprefix("# USER\n").strip()
    return system, user

_SYSTEM_PROMPT, _USER_TEMPLATE = _load_prompts()


# ── Factor definitions ───────────────────────────────────────────────────────

@dataclass
class NumericFactor:
    score: int
    rationale: str

@dataclass
class CategoricalFactor:
    value: str
    rationale: str

@dataclass
class ShadowbanFactor:
    count: int
    words: list[str]
    rationale: str


@dataclass
class LyricsClassification:
    headline: str
    title: str
    timestamp: str

    # Hook Mechanics
    hook_strength: NumericFactor = field(default_factory=lambda: NumericFactor(0, ""))
    hook_position: CategoricalFactor = field(default_factory=lambda: CategoricalFactor("", ""))
    earworm_factor: NumericFactor = field(default_factory=lambda: NumericFactor(0, ""))
    singability: CategoricalFactor = field(default_factory=lambda: CategoricalFactor("", ""))

    # Cultural Payload
    topicality: CategoricalFactor = field(default_factory=lambda: CategoricalFactor("", ""))
    recognition_trigger: CategoricalFactor = field(default_factory=lambda: CategoricalFactor("", ""))
    controversy_level: NumericFactor = field(default_factory=lambda: NumericFactor(0, ""))
    satire_type: CategoricalFactor = field(default_factory=lambda: CategoricalFactor("", ""))
    ingroup_signal: CategoricalFactor = field(default_factory=lambda: CategoricalFactor("", ""))

    # Creator Bait
    visual_hook_potential: CategoricalFactor = field(default_factory=lambda: CategoricalFactor("", ""))
    meme_format_fit: CategoricalFactor = field(default_factory=lambda: CategoricalFactor("", ""))
    quotability: NumericFactor = field(default_factory=lambda: NumericFactor(0, ""))
    participation_hook: NumericFactor = field(default_factory=lambda: NumericFactor(0, ""))

    # Platform Risk
    takedown_risk: CategoricalFactor = field(default_factory=lambda: CategoricalFactor("", ""))
    algorithm_risk: CategoricalFactor = field(default_factory=lambda: CategoricalFactor("", ""))
    shadowban_words: ShadowbanFactor = field(default_factory=lambda: ShadowbanFactor(0, [], ""))

    verdict: str = ""
    lvi: float = 0.0  # Lyrics Virality Index

    @property
    def lvi_label(self) -> str:
        if self.lvi >= 8:
            return "Banger — ship it"
        if self.lvi >= 6:
            return "Strong, minor tweaks possible"
        if self.lvi >= 4:
            return "Decent but needs work"
        return "Weak — consider rewriting"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["lvi_label"] = self.lvi_label
        return d


_NUMERIC_FIELDS = [
    "hook_strength", "earworm_factor", "controversy_level",
    "quotability", "participation_hook",
]

_CATEGORICAL_FIELDS = [
    "hook_position", "singability", "topicality", "recognition_trigger",
    "satire_type", "ingroup_signal", "visual_hook_potential", "meme_format_fit",
    "takedown_risk", "algorithm_risk",
]

_RISK_MAP = {"low": 10, "medium": 5, "high": 1}
_VISUAL_MAP = {"low": 3, "medium": 6, "high": 9}
_SINGABILITY_MAP = {"easy": 9, "medium": 6, "hard": 3}
_TOPICALITY_MAP = {"breaking": 9, "trending": 7, "evergreen": 5}
_INGROUP_MAP = {"mass": 9, "niche": 5, "deep_niche": 2}


def _compute_lvi(c: LyricsClassification) -> float:
    """Compute Lyrics Virality Index (0-10) from all factors."""
    scores = [
        c.hook_strength.score,
        c.earworm_factor.score,
        _SINGABILITY_MAP.get(c.singability.value, 5),
        _TOPICALITY_MAP.get(c.topicality.value, 5),
        c.controversy_level.score,
        _INGROUP_MAP.get(c.ingroup_signal.value, 5),
        _VISUAL_MAP.get(c.visual_hook_potential.value, 5),
        c.quotability.score,
        c.participation_hook.score,
        _RISK_MAP.get(c.takedown_risk.value, 5),
        _RISK_MAP.get(c.algorithm_risk.value, 5),
    ]
    return round(sum(scores) / len(scores), 2) if scores else 0.0


def _extract_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text
        if "\n" in text:
            first, rest = text.split("\n", 1)
            if not first.strip().startswith("{"):
                text = rest
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _parse_response(raw: str, headline: str, title: str) -> LyricsClassification:
    cleaned = _extract_json(raw)
    if not cleaned:
        raise ValueError(f"Empty response for lyrics classification: {title[:60]!r}")
    data = json.loads(cleaned)

    def _nf(key: str) -> NumericFactor:
        e = data.get(key, {})
        return NumericFactor(score=int(e.get("score", 0)), rationale=e.get("rationale", ""))

    def _cf(key: str) -> CategoricalFactor:
        e = data.get(key, {})
        return CategoricalFactor(value=e.get("value", ""), rationale=e.get("rationale", ""))

    sb = data.get("shadowban_words", {})
    shadowban = ShadowbanFactor(
        count=int(sb.get("count", 0)),
        words=sb.get("words", []),
        rationale=sb.get("rationale", ""),
    )

    c = LyricsClassification(
        headline=headline,
        title=title,
        timestamp=datetime.now().isoformat(),
        hook_strength=_nf("hook_strength"),
        hook_position=_cf("hook_position"),
        earworm_factor=_nf("earworm_factor"),
        singability=_cf("singability"),
        topicality=_cf("topicality"),
        recognition_trigger=_cf("recognition_trigger"),
        controversy_level=_nf("controversy_level"),
        satire_type=_cf("satire_type"),
        ingroup_signal=_cf("ingroup_signal"),
        visual_hook_potential=_cf("visual_hook_potential"),
        meme_format_fit=_cf("meme_format_fit"),
        quotability=_nf("quotability"),
        participation_hook=_nf("participation_hook"),
        takedown_risk=_cf("takedown_risk"),
        algorithm_risk=_cf("algorithm_risk"),
        shadowban_words=shadowban,
        verdict=data.get("verdict", ""),
    )
    c.lvi = _compute_lvi(c)
    return c


def _load_cached(title: str, today: str, classifier: str | None = None) -> LyricsClassification | None:
    log_file = _LOGS_DIR / "lyrics_classifications.jsonl"
    if not log_file.exists():
        return None
    for line in log_file.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            if entry.get("title") == title and entry.get("date") == today:
                if classifier and entry.get("classifier") != classifier:
                    continue
                return _reconstruct(entry)
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def _reconstruct(entry: dict) -> LyricsClassification:
    def _nf(key: str) -> NumericFactor:
        e = entry.get(key, {})
        return NumericFactor(score=int(e.get("score", 0)), rationale=e.get("rationale", ""))

    def _cf(key: str) -> CategoricalFactor:
        e = entry.get(key, {})
        return CategoricalFactor(value=e.get("value", ""), rationale=e.get("rationale", ""))

    sb = entry.get("shadowban_words", {})

    c = LyricsClassification(
        headline=entry.get("headline", ""),
        title=entry.get("title", ""),
        timestamp=entry.get("timestamp", ""),
        hook_strength=_nf("hook_strength"),
        hook_position=_cf("hook_position"),
        earworm_factor=_nf("earworm_factor"),
        singability=_cf("singability"),
        topicality=_cf("topicality"),
        recognition_trigger=_cf("recognition_trigger"),
        controversy_level=_nf("controversy_level"),
        satire_type=_cf("satire_type"),
        ingroup_signal=_cf("ingroup_signal"),
        visual_hook_potential=_cf("visual_hook_potential"),
        meme_format_fit=_cf("meme_format_fit"),
        quotability=_nf("quotability"),
        participation_hook=_nf("participation_hook"),
        takedown_risk=_cf("takedown_risk"),
        algorithm_risk=_cf("algorithm_risk"),
        shadowban_words=ShadowbanFactor(
            count=int(sb.get("count", 0)),
            words=sb.get("words", []),
            rationale=sb.get("rationale", ""),
        ),
        verdict=entry.get("verdict", ""),
        lvi=entry.get("lvi", 0.0),
    )
    return c


def _append_log(classification: LyricsClassification, today: str, classifier: str | None = None, extra: dict | None = None) -> None:
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _LOGS_DIR / "lyrics_classifications.jsonl"
    entry = classification.to_dict()
    entry["date"] = today
    if classifier:
        entry["classifier"] = classifier
    if extra:
        entry.update(extra)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    try:
        from db.client import get_client
        get_client().table("lyrics_classifications").insert(entry).execute()
    except Exception:
        pass


async def classify_lyrics(
    headline: str,
    title: str,
    lyrics_text: str,
    api_key: str = "ollama",
    model: str = "gemma3",
    base_url: str = "http://localhost:11434/v1",
) -> LyricsClassification:
    """Classify lyrics for TikTok viral potential using a single LLM."""
    today = date.today().isoformat()

    cached = _load_cached(title, today, classifier=model)
    if cached is not None:
        log.info(f"[lyrics-classifier] Cache hit ({model}) — skipping: {title}")
        return cached

    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    user_prompt = (
        _USER_TEMPLATE
        .replace("{headline}", headline)
        .replace("{title}", title)
        .replace("{lyrics}", lyrics_text)
    )

    log.info(f"[lyrics-classifier] Classifying lyrics: {title} (model={model})")

    response = await client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = (response.choices[0].message.content or "").strip()
    classification = _parse_response(raw, headline, title)

    log.info(f"[lyrics-classifier] LVI={classification.lvi:.1f} ({classification.lvi_label}) — {title}")

    _append_log(classification, today, classifier=model)
    return classification


def _merge_classifications(
    a: LyricsClassification,
    b: LyricsClassification,
    label_a: str,
    label_b: str,
) -> tuple[LyricsClassification, dict]:
    """Average numeric scores from two classifications."""
    def _avg_nf(key: str) -> NumericFactor:
        fa: NumericFactor = getattr(a, key)
        fb: NumericFactor = getattr(b, key)
        return NumericFactor(
            score=round((fa.score + fb.score) / 2),
            rationale=f"[{label_a}] {fa.rationale} | [{label_b}] {fb.rationale}",
        )

    def _pick_cf(key: str) -> CategoricalFactor:
        fa: CategoricalFactor = getattr(a, key)
        fb: CategoricalFactor = getattr(b, key)
        value = fa.value if a.lvi >= b.lvi else fb.value
        return CategoricalFactor(
            value=value,
            rationale=f"[{label_a}] {fa.rationale} | [{label_b}] {fb.rationale}",
        )

    merged = LyricsClassification(
        headline=a.headline,
        title=a.title,
        timestamp=datetime.now().isoformat(),
        hook_strength=_avg_nf("hook_strength"),
        hook_position=_pick_cf("hook_position"),
        earworm_factor=_avg_nf("earworm_factor"),
        singability=_pick_cf("singability"),
        topicality=_pick_cf("topicality"),
        recognition_trigger=_pick_cf("recognition_trigger"),
        controversy_level=_avg_nf("controversy_level"),
        satire_type=_pick_cf("satire_type"),
        ingroup_signal=_pick_cf("ingroup_signal"),
        visual_hook_potential=_pick_cf("visual_hook_potential"),
        meme_format_fit=_pick_cf("meme_format_fit"),
        quotability=_avg_nf("quotability"),
        participation_hook=_avg_nf("participation_hook"),
        takedown_risk=_pick_cf("takedown_risk"),
        algorithm_risk=_pick_cf("algorithm_risk"),
        shadowban_words=a.shadowban_words if a.lvi >= b.lvi else b.shadowban_words,
        verdict=a.verdict if a.lvi >= b.lvi else b.verdict,
    )
    merged.lvi = _compute_lvi(merged)
    extra = {f"lvi_{label_a}": a.lvi, f"lvi_{label_b}": b.lvi}
    return merged, extra


async def classify_lyrics_dual(
    headline: str,
    title: str,
    lyrics_text: str,
    ollama_model: str = "gemma3",
    ollama_base_url: str = "http://localhost:11434/v1",
    grok_api_key: str = "",
    grok_base_url: str = "https://api.x.ai/v1",
    grok_model: str = "grok-3-fast",
) -> LyricsClassification:
    """Classify lyrics with both Ollama and Grok, merge results."""
    today = date.today().isoformat()

    cached = _load_cached(title, today, classifier="dual")
    if cached is not None:
        log.info(f"[lyrics-classifier] Dual cache hit — skipping: {title}")
        return cached

    tasks = [
        classify_lyrics(
            headline=headline, title=title, lyrics_text=lyrics_text,
            api_key="ollama", model=ollama_model, base_url=ollama_base_url,
        ),
    ]
    has_grok = bool(grok_api_key)
    if has_grok:
        tasks.append(
            classify_lyrics(
                headline=headline, title=title, lyrics_text=lyrics_text,
                api_key=grok_api_key, model=grok_model, base_url=grok_base_url,
            ),
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    ollama_result = None if isinstance(results[0], Exception) else results[0]
    grok_result = None if (not has_grok or isinstance(results[1], Exception)) else results[1]

    if ollama_result and grok_result:
        merged, extra = _merge_classifications(ollama_result, grok_result, "gemma3", "grok")
        log.info(
            f"[lyrics-classifier] Dual LVI={merged.lvi:.1f} "
            f"(gemma3={ollama_result.lvi:.1f}, grok={grok_result.lvi:.1f}) — {title}"
        )
        _append_log(merged, today, classifier="dual", extra=extra)
        return merged

    fallback = ollama_result or grok_result
    if fallback is None:
        raise RuntimeError(f"Both lyrics classifiers failed for: {title}")

    label = "gemma3" if ollama_result else "grok"
    log.warning(f"[lyrics-classifier] Only {label} succeeded — using single model for: {title}")
    return fallback


if __name__ == "__main__":
    import argparse
    import os
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    from modules.utils import find_run_dir
    from modules.clip_generator import parse_lyrics_file

    parser = argparse.ArgumentParser(description="Classify lyrics for TikTok viral potential")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--run", type=str, default=None)
    parser.add_argument("--provider", type=str, default="ollama", choices=["ollama", "grok", "dual"])
    args = parser.parse_args()

    async def main():
        out = find_run_dir(args.date, args.run)
        lyrics_file = out / "lyrics.txt"
        headline_file = out / "headline.txt"

        if not lyrics_file.exists():
            print(f"No lyrics.txt in {out}")
            return

        title, sections = parse_lyrics_file(lyrics_file)
        lyrics_text = "\n".join(
            f"[{s.label.upper()}]\n" + "\n".join(s.lines)
            for s in sections
        )

        headline = ""
        if headline_file.exists():
            headline = headline_file.read_text(encoding="utf-8").splitlines()[0].strip()

        if args.provider == "dual":
            result = await classify_lyrics_dual(
                headline=headline, title=title, lyrics_text=lyrics_text,
                grok_api_key=os.environ.get("XAI_API_KEY", ""),
            )
        elif args.provider == "grok":
            result = await classify_lyrics(
                headline=headline, title=title, lyrics_text=lyrics_text,
                api_key=os.environ.get("XAI_API_KEY", ""),
                model="grok-3-fast", base_url="https://api.x.ai/v1",
            )
        else:
            result = await classify_lyrics(
                headline=headline, title=title, lyrics_text=lyrics_text,
            )

        print(f"\nLyrics Virality Index: {result.lvi:.1f} — {result.lvi_label}")
        print(f"Verdict: {result.verdict}")
        print(f"\nHook Strength: {result.hook_strength.score}/10")
        print(f"Earworm Factor: {result.earworm_factor.score}/10")
        print(f"Singability: {result.singability.value}")
        print(f"Quotability: {result.quotability.score}/10")
        print(f"Controversy: {result.controversy_level.score}/10")
        print(f"Takedown Risk: {result.takedown_risk.value}")
        print(f"Algorithm Risk: {result.algorithm_risk.value}")
        if result.shadowban_words.count > 0:
            print(f"Shadowban Words: {result.shadowban_words.words}")

    asyncio.run(main())
