"""
Generates song lyrics + TikTok caption/hashtags using Claude.
Target: 30-45 seconds of audio content (1 verse + chorus).
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import openai

log = logging.getLogger(__name__)

from modules.utils import log_api_call

PLATFORM_TAGS = ["#fyp", "#foryou"]


@dataclass
class LyricSection:
    label: str   # "verse" | "chorus" | "hook"
    lines: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass
class Lyrics:
    title: str
    sections: list[LyricSection]
    style_prompt: str        # passed to Suno
    caption: str             # full TikTok post caption incl. hashtags
    topic_tags: list[str]    # e.g. ["#trump", "#tariffs"]

    @property
    def full_text(self) -> str:
        """All lyrics joined for Suno submission."""
        parts = []
        for s in self.sections:
            parts.append(f"[{s.label.upper()}]")
            parts.append(s.text)
        return "\n\n".join(parts)


_ASSETS_DIR = Path(__file__).parent.parent / "assets"

_DEFAULT_SYSTEM_PROMPT = """You are a viral TikTok songwriter. You write short, catchy, satirical pop/hip-hop songs
about current news events. Your songs are funny, punchy, and designed for the TikTok algorithm.
You always respond with valid JSON only — no markdown fences, no extra text."""

_DEFAULT_USER_PROMPT_TEMPLATE = """Write a short TikTok song about this news story:

HEADLINE: {headline}
SUMMARY: {summary}

Requirements:
- Total song length: 30-45 seconds when sung at a moderate pop tempo (~100 BPM)
- Structure: 1 hook (opening 3 seconds, one punchy line) + 1 verse (4-6 lines) + 1 chorus (4 lines, repeated feel)
- Tone: satirical, fun, slightly sarcastic — like a meme in song form
- Style: upbeat pop or hip-hop, modern, radio-friendly

Also generate:
- A Suno style prompt (music genre tags + mood, e.g. "upbeat pop, punchy drums, catchy hook, modern production")
- 1-2 specific topic hashtags based on the actual news subject (e.g. #trump, #gaza, #royalfamily — NOT generic like #news or #trending)
- A short TikTok caption (max 100 chars) that hooks viewers

Respond with this exact JSON structure:
{{
  "title": "Song title here",
  "hook": ["One punchy opening line"],
  "verse": ["line 1", "line 2", "line 3", "line 4"],
  "chorus": ["line 1", "line 2", "line 3", "line 4"],
  "style_prompt": "upbeat pop, ...",
  "caption": "Short hook caption here",
  "topic_tags": ["#specifictag1", "#specifictag2"]
}}"""


def _load_prompt(filename: str, default: str) -> str:
    """Load a prompt from assets/, or use the default."""
    path = _ASSETS_DIR / filename
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return default


def load_system_prompt() -> str:
    return _load_prompt("lyrics_system_prompt.txt", _DEFAULT_SYSTEM_PROMPT)


def load_user_prompt_template() -> str:
    return _load_prompt("lyrics_user_prompt.txt", _DEFAULT_USER_PROMPT_TEMPLATE)


def _flatten_string_list(val: list) -> list[str]:
    """Flatten any accidentally nested lists the LLM returns, keeping only strings."""
    out = []
    for item in val:
        if isinstance(item, list):
            out.extend(str(s) for s in item)
        else:
            out.append(str(item))
    return out


async def generate_lyrics(
    headline: str,
    summary: str,
    api_key: str,
    model: str,
    fixed_hashtags: list[str],
    angle: str = "",
    day_number: int | None = None,
    ollama_base_url: str = "http://localhost:11434/v1",
    base_url: str | None = None,
    output_dir: Path | None = None,
    max_retries: int = 3,
) -> Lyrics:
    """Call an OpenAI-compatible LLM and parse the lyrics response.

    If base_url is provided, uses that (e.g. xAI/Grok). Otherwise falls back to ollama_base_url.
    Retries up to max_retries times if the model returns malformed JSON.
    """
    effective_base = base_url or ollama_base_url
    effective_key = api_key if api_key != "ollama" else "ollama"
    client = openai.AsyncOpenAI(base_url=effective_base, api_key=effective_key)

    system_prompt = load_system_prompt()
    user_prompt_template = load_user_prompt_template()
    prompt = user_prompt_template.format(
        headline=headline,
        summary=summary,
        angle=angle or "derive a satirical angle from the headline and summary",
    )

    # Log the API call input
    log_api_call("lyrics-llm", {
        "base_url": effective_base,
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": prompt,
    }, run_dir=output_dir)

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        log.info(f"[lyrics] Calling {effective_base} ({model}) — attempt {attempt}/{max_retries}...")
        message = await client.chat.completions.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
        )

        raw = message.choices[0].message.content.strip()

        # Log the API response
        log_api_call("lyrics-llm-response", {
            "raw_response": raw,
            "attempt": attempt,
        }, run_dir=output_dir)

        # Strip any accidental markdown fences
        raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

        try:
            data = json.loads(raw)
            break  # success
        except json.JSONDecodeError as e:
            last_error = e
            log.warning(f"[lyrics] JSON parse failed (attempt {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                # Last resort: try json_repair before giving up
                try:
                    from json_repair import repair_json
                    repaired = repair_json(raw)
                    data = json.loads(repaired)
                    log.warning("[lyrics] JSON recovered via json_repair")
                    break
                except Exception:
                    raise ValueError(f"LLM returned invalid JSON after {max_retries} attempts: {e}") from e
            continue

    sections = [
        LyricSection(label="hook",   lines=_flatten_string_list(data["hook"])),
        LyricSection(label="verse",  lines=_flatten_string_list(data["verse"])),
        LyricSection(label="chorus", lines=_flatten_string_list(data["chorus"])),
    ]

    # Build hashtag string
    # Alternate #fyp / #foryou by day
    if day_number is None:
        day_number = date.today().toordinal()
    platform_tag = PLATFORM_TAGS[day_number % len(PLATFORM_TAGS)]

    topic_tags = [t if t.startswith("#") else f"#{t}" for t in data.get("topic_tags", [])][:2]
    all_tags = fixed_hashtags + topic_tags + [platform_tag]
    tags_str = " ".join(all_tags)

    caption_text = data.get("caption", data["title"])
    caption = f"{caption_text}\n\n{tags_str}"

    lyrics = Lyrics(
        title=data["title"],
        sections=sections,
        style_prompt=data.get("style_prompt", "upbeat pop, catchy, TikTok"),
        caption=caption,
        topic_tags=topic_tags,
    )
    log.info(f"[lyrics] Generated: '{lyrics.title}'")
    return lyrics


if __name__ == "__main__":
    import argparse
    import asyncio, os
    from dotenv import load_dotenv
    import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config import FIXED_HASHTAGS, OLLAMA_BASE_URL, OLLAMA_MODEL
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate song lyrics from a headline")
    parser.add_argument("--provider", type=str, default="ollama", choices=["ollama", "grok"],
                        help="LLM provider (default: ollama)")
    parser.add_argument("--model", type=str, default=None, help="Model name override")
    parser.add_argument("--headline", type=str, default="Trump announces sweeping new tariffs on all imports")
    parser.add_argument("--summary", type=str, default="President Trump declared a 25% tariff on goods from all trading partners, sending markets tumbling.")
    parser.add_argument("--date", type=str, default=None, help="Output date folder for logging (default: today)")
    parser.add_argument("--run", type=str, default=None, help="Run folder for logging (default: latest)")
    args = parser.parse_args()

    async def main():
        if args.provider == "grok":
            api_key = os.environ.get("XAI_API_KEY", "")
            if not api_key:
                print("Set XAI_API_KEY in .env")
                return
            base_url = "https://api.x.ai/v1"
            model = args.model or "grok-3-fast"
        else:
            api_key = "ollama"
            base_url = OLLAMA_BASE_URL
            model = args.model or OLLAMA_MODEL

        # Set up output dir for logging
        from modules.utils import find_run_dir
        out_dir = find_run_dir(args.date, args.run)
        out_dir.mkdir(parents=True, exist_ok=True)

        lyrics = await generate_lyrics(
            headline=args.headline,
            summary=args.summary,
            api_key=api_key,
            model=model,
            fixed_hashtags=FIXED_HASHTAGS,
            base_url=base_url,
            output_dir=out_dir,
        )
        print(f"\nTitle   : {lyrics.title}")
        print(f"Style   : {lyrics.style_prompt}")
        print(f"\n{lyrics.full_text}")
        # Encode caption safely for Windows terminal
        caption_safe = lyrics.caption.encode("ascii", "ignore").decode("ascii")
        print(f"\nCaption :\n{caption_safe}")

    asyncio.run(main())
