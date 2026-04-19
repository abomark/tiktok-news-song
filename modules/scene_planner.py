"""
Generates a scene-by-scene visual story plan for a music video using an LLM.

The first scene animates the news article's hero image (image-to-video).
Subsequent scenes are text-to-video with an evolving visual story and a viral twist.
"""

from __future__ import annotations
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import openai

from modules.lyrics_generator import Lyrics
from modules.utils import log_api_call

log = logging.getLogger(__name__)

CLIP_DURATION = 4  # seconds per Pollo clip

_ASSETS_DIR = Path(__file__).parent.parent / "assets"

_DEFAULT_SYSTEM_PROMPT = """You are a viral TikTok music video director. You create visually compelling scene-by-scene plans
for short-form news music videos. Each scene is exactly 4 seconds of video.

Your scenes tell a VISUAL STORY that evolves — each scene builds on the previous one.
Every video must include at least one "viral moment" — an unexpected visual twist, dramatic reveal,
or absurd escalation that makes viewers replay the video.

For each scene you produce ONE primary prompt and THREE alternative prompts. The alternatives must be
genuinely visually distinct from each other and the primary — different camera angles, different
settings, different levels of abstraction. All four must remain thematically tied to the lyric moment
and the headline. Do NOT just rephrase the primary prompt with minor word changes.

You always respond with valid JSON only — no markdown fences, no extra text."""

_DEFAULT_USER_PROMPT_TEMPLATE = """Create a scene-by-scene visual story plan for a TikTok news music video.

NEWS HEADLINE: {headline}
NEWS SUMMARY: {summary}

FULL SONG LYRICS:
{lyrics_text}

LYRICS TIMED TO EACH CLIP (use this to match visuals to what is being sung):
{clip_lyrics}

TOTAL DURATION: {duration}s
NUMBER OF 4-SECOND CLIPS NEEDED: {num_clips}

FIRST CLIP (clip 0): IMAGE-TO-VIDEO — the news article's hero image comes to life.
  - Describe cinematic motion that complements the existing image
  - Match the mood of what is sung during 0s-4s

SUBSEQUENT CLIPS (clips 1 through {last_clip_index}): TEXT-TO-VIDEO only.
  - Each clip must look DIFFERENT from every other clip (no repeated locations or compositions)
  - Visually match what is being sung during that clip's time window
  - Story arc: setup → escalation → viral twist → resolution
  - At MOST one city-skyline / aerial scene per video
  - Vary: interiors, close-ups, abstract, studio, nature, crowds, objects, textures, POV shots

For each scene produce:
  - "prompt": primary 1-2 sentence video prompt, specific and vivid
  - "alt_prompts": array of exactly 3 alternatives, each a genuinely different visual approach:
      [0] Different camera angle or shot size (e.g. extreme close-up, bird's-eye, tracking POV)
      [1] Abstract or symbolic visual (metaphor, motion blur, surreal, graphic)
      [2] Different physical setting or environment (indoors vs outdoors, documentary vs cinematic)

Respond with this exact JSON structure:
{{
  "scenes": [
    {{
      "clip_index": 0,
      "visual_action": "One sentence: what the viewer sees happening",
      "prompt": "Primary cinematic prompt for clip 0...",
      "alt_prompts": [
        "Different angle alternative for clip 0...",
        "Abstract/symbolic alternative for clip 0...",
        "Different setting alternative for clip 0..."
      ]
    }},
    {{
      "clip_index": 1,
      "visual_action": "One sentence: what the viewer sees happening",
      "prompt": "Primary cinematic prompt for clip 1...",
      "alt_prompts": [
        "Different angle alternative for clip 1...",
        "Abstract/symbolic alternative for clip 1...",
        "Different setting alternative for clip 1..."
      ]
    }}
  ]
}}

Generate exactly {num_clips} scenes (clip_index 0 through {last_clip_index}).
Make visuals match the sung lyrics. Make the story escalate with one unexpected twist."""


@dataclass
class SceneDescription:
    clip_index: int
    prompt: str
    is_image_to_video: bool
    visual_action: str
    alt_prompts: list[str] | None = None       # 3 diverse alternatives from LLM
    prompt_variants: list[str] | None = None   # ≥10 filter-safe variants for Pollo retry


@dataclass
class ScenePlan:
    scenes: list[SceneDescription]


# ── Desensitization ladder ────────────────────────────────────────────────────
# Each level's dict contains ONLY that level's replacements; _sanitize() stacks
# all levels ≤ target. Patterns are case-insensitive word-boundary regexes.
_SENSITIVE_TERMS: dict[int, dict[str, str]] = {
    # L1 — brand / corporate names → generic category
    1: {
        r"\bnetflix\b": "a streaming service",
        r"\bdisney\+?\b": "a streaming service",
        r"\bhbo( max)?\b": "a streaming service",
        r"\bspotify\b": "a music streaming service",
        r"\bamazon\b": "a tech giant",
        r"\bgoogle\b": "a tech giant",
        r"\bmeta\b": "a social media company",
        r"\bfacebook\b": "a social media platform",
        r"\binstagram\b": "a social media platform",
        r"\btiktok\b": "a short-video platform",
        r"\btwitter\b": "a social media platform",
        r"\bx \(formerly twitter\)\b": "a social media platform",
        r"\byoutube\b": "a video platform",
        r"\bapple\b": "a tech giant",
        r"\bmicrosoft\b": "a tech giant",
        r"\bopenai\b": "an AI company",
        r"\btesla\b": "an electric vehicle company",
        r"\bmcdonald'?s\b": "a fast-food chain",
        r"\bstarbucks\b": "a coffee chain",
        r"\bnike\b": "a sportswear brand",
        r"\bwalmart\b": "a retail chain",
    },
    # L2 — violence / weapon terms → neutral impact language
    2: {
        r"\bbombs?\b": "blasts",
        r"\bbombing\b": "a loud blast",
        r"\bexplosions?\b": "bright bursts",
        r"\bgunshots?\b": "sharp cracks",
        r"\bshoot(ing|s|ers?)?\b": "a flash",
        r"\bkill(ed|s|ing)?\b": "taken down",
        r"\bmurder(ed|s|ing)?\b": "taken down",
        r"\bassassinat(e|ed|ion)\b": "a dramatic fall",
        r"\bblood(y)?\b": "impact",
        r"\bwar\b": "crisis",
        r"\bwarzone\b": "a tense region",
        r"\battack(s|ing|ed)?\b": "confrontation",
        r"\bweapons?\b": "objects",
        r"\bgun(s|ner|fire)?\b": "gear",
        r"\brifles?\b": "gear",
        r"\bmissiles?\b": "streaks of light",
        r"\bdrone strikes?\b": "streaks across the sky",
        r"\bdead\b": "still",
        r"\bdeath\b": "stillness",
        r"\bcorpses?\b": "figures",
    },
    # L3 — named politicians → role
    3: {
        r"\bdonald trump\b": "a US leader",
        r"\btrump\b": "a US leader",
        r"\bjoe biden\b": "a US leader",
        r"\bbiden\b": "a US leader",
        r"\bkamala harris\b": "a US official",
        r"\bharris\b": "a US official",
        r"\bvladimir putin\b": "a world leader",
        r"\bputin\b": "a world leader",
        r"\bxi jinping\b": "a world leader",
        r"\bxi\b": "a world leader",
        r"\bbenjamin netanyahu\b": "a world leader",
        r"\bnetanyahu\b": "a world leader",
        r"\bkim jong[- ]un\b": "a world leader",
        r"\bzelensky\b": "a world leader",
        r"\bzelenskyy\b": "a world leader",
        r"\bemmanuel macron\b": "a world leader",
        r"\bmacron\b": "a world leader",
        r"\brishi sunak\b": "a world leader",
        r"\bsunak\b": "a world leader",
        r"\bkeir starmer\b": "a world leader",
        r"\bstarmer\b": "a world leader",
        r"\bjustin trudeau\b": "a world leader",
        r"\btrudeau\b": "a world leader",
        r"\bthe pope\b": "a religious figure",
        r"\bpope\b": "a religious figure",
    },
    # L4 — named celebrities / public figures → role
    4: {
        r"\belon musk\b": "a tech entrepreneur",
        r"\bmusk\b": "a tech entrepreneur",
        r"\bmark zuckerberg\b": "a tech executive",
        r"\bzuckerberg\b": "a tech executive",
        r"\bjeff bezos\b": "a tech entrepreneur",
        r"\bbezos\b": "a tech entrepreneur",
        r"\bbill gates\b": "a tech entrepreneur",
        r"\btaylor swift\b": "a pop star",
        r"\bkanye( west)?\b": "a rapper",
        r"\bbeyonc[eé]\b": "a pop star",
        r"\bdrake\b": "a rapper",
        r"\brihanna\b": "a pop star",
        r"\bkardashian\b": "a reality star",
        r"\blebron( james)?\b": "an athlete",
        r"\bmessi\b": "an athlete",
        r"\bronaldo\b": "an athlete",
    },
    # L5 — charged political terms → neutral
    5: {
        r"\bregime\b": "administration",
        r"\bdeport(ed|ing|ation)?\b": "relocated",
        r"\binvade(d|s|r|rs)?\b": "entered",
        r"\binvasion\b": "arrival",
        r"\bterroris(m|t|ts)\b": "unrest",
        r"\bextremis(m|t|ts)\b": "tension",
        r"\bprotest(s|ers|ing|ed)?\b": "a crowd gathering",
        r"\briot(s|ers|ing|ed)?\b": "a crowd scene",
        r"\bcoup\b": "a transition",
        r"\bsanctions?\b": "restrictions",
        r"\bimpeach(ed|ing|ment)?\b": "challenged",
        r"\btariffs?\b": "trade measures",
    },
    # L6 — specific location names → generic
    6: {
        r"\bwhite house\b": "a government building",
        r"\bcapitol( hill)?\b": "a government building",
        r"\boval office\b": "a formal office",
        r"\bpentagon\b": "a government headquarters",
        r"\bkremlin\b": "a government headquarters",
        r"\bgaza\b": "a contested region",
        r"\bisrael\b": "a country in the region",
        r"\bpalestine\b": "a contested region",
        r"\bukraine\b": "an eastern european country",
        r"\brussia\b": "a large eastern country",
        r"\bchina\b": "a large asian country",
        r"\biran\b": "a middle eastern country",
        r"\bnorth korea\b": "an isolated country",
        r"\btaiwan\b": "an island nation",
        r"\bwashington(, d\.?c\.?)?\b": "a capital city",
        r"\bmoscow\b": "a capital city",
        r"\bbeijing\b": "a capital city",
        r"\btehran\b": "a capital city",
        r"\bjerusalem\b": "a historic city",
    },
}


def _apply_level(text: str, level: int) -> str:
    """Apply all replacements for levels 1..level (additive)."""
    out = text
    for lvl in range(1, level + 1):
        for pattern, repl in _SENSITIVE_TERMS.get(lvl, {}).items():
            out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out


def _strip_proper_nouns(text: str) -> str:
    """L7: strip runs of capitalized tokens that aren't sentence-initial."""
    # Collapse runs of 2+ capitalized words (likely proper nouns) into "a figure"
    text = re.sub(r"(?<!\. )(?<!^)\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\b", "a figure", text)
    # Single capitalized tokens mid-sentence → lowercase if not start of sentence
    def _lc(m: re.Match) -> str:
        tok = m.group(0)
        # keep common proper nouns that are also regular words (avoid breaking)
        if tok.lower() in {"i"}:
            return tok
        return tok.lower()
    text = re.sub(r"(?<=[a-z,] )[A-Z][a-zA-Z]+", _lc, text)
    return text


def _lyrics_for_clip(timed_words: list[dict], clip_index: int, clip_duration: float) -> str:
    """Return a string of words sung during clip_index's time window."""
    t_start = clip_index * clip_duration
    t_end = t_start + clip_duration
    words = []
    for w in timed_words:
        s = float(w.get("startS", 0))
        if t_start <= s < t_end:
            raw = w.get("word", "")
            # Strip section tags and cross-line residue
            raw = re.sub(r"\[[A-Z]+\]\n?", "", raw)
            raw = re.sub(r"\n.*", "", raw).strip().strip(",.")
            if raw:
                words.append(raw.upper())
    return " ".join(words) if words else "(instrumental)"


def _build_clip_lyrics_block(timed_words: list[dict], num_clips: int, clip_duration: float) -> str:
    """Build the per-clip lyrics section for the scene planner prompt."""
    lines = []
    for i in range(num_clips):
        t_start = i * clip_duration
        t_end = t_start + clip_duration
        lyric = _lyrics_for_clip(timed_words, i, clip_duration)
        lines.append(f"  Clip {i} ({t_start:.0f}s-{t_end:.0f}s): \"{lyric}\"")
    return "\n".join(lines)


def _sanitize(text: str, level: int) -> str:
    """Stack levels 1..min(level,7); levels 8–9 are handled by _build_prompt_variants."""
    if not text:
        return text
    out = _apply_level(text, min(level, 6))
    if level >= 7:
        out = _strip_proper_nouns(out)
    # Normalize whitespace
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _build_prompt_variants(
    prompt: str,
    visual_action: str,
    alt_prompts: list[str] | None = None,
    num_variants: int = 10,
) -> list[str]:
    """Return a list of ≥num_variants filter-safe prompt variants for Pollo retry.

    Strategy: use the LLM-generated primary + alt_prompts as genuinely different
    visual bases, then apply two levels of sensitivity sanitization to each base.
    This produces real variety (different angles/settings/abstractions) while
    ensuring progressively safer content-filter odds.

    Order: primary L0 → alts L0 → primary L3-sanitized → alts L3-sanitized →
           primary L6-sanitized → alts L6-sanitized → visual_action cinematic →
           visual_action abstract → padding if still < num_variants.
    """
    prompt = (prompt or "").strip()
    visual_action = (visual_action or "").strip()
    bases = [prompt] + [a.strip() for a in (alt_prompts or []) if a and a.strip()]

    # Build ordered candidates: each base at L0 → L3 → L6
    candidates: list[str] = []
    for sanitize_level in (0, 3, 6):
        for base in bases:
            if sanitize_level == 0:
                candidates.append(base)
            else:
                candidates.append(_sanitize(base, sanitize_level))

    # Add visual_action–derived safe fallbacks (unique per scene)
    core = _sanitize(visual_action, 7) or _sanitize(prompt, 7) or "an evocative moment"
    candidates.append(
        f"Cinematic scene. {core}. Dynamic camera movement, vibrant colors, 9:16 vertical."
    )
    candidates.append(
        f"Abstract, impressionistic — {core}. Soft focus, slow camera drift, vibrant colors, 9:16 vertical."
    )

    # Dedupe in order
    seen: set[str] = set()
    deduped: list[str] = []
    for v in candidates:
        key = v.strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(key)

    # Pad to num_variants if needed (rare — usually we have 3 alts × 3 levels = 9+ unique)
    pad_suffixes = [" —", " ·", " ~", " |"]
    i = 0
    while len(deduped) < num_variants:
        base = deduped[-1]
        candidate = base.rstrip(" .—·~|") + pad_suffixes[i % len(pad_suffixes)]
        if candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
        i += 1
        if i > 40:
            break
    return deduped


def _load_prompt(filename: str, default: str) -> str:
    path = _ASSETS_DIR / filename
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return default


def load_system_prompt() -> str:
    return _load_prompt("scene_system_prompt.txt", _DEFAULT_SYSTEM_PROMPT)


def load_user_prompt_template() -> str:
    return _load_prompt("scene_user_prompt.txt", _DEFAULT_USER_PROMPT_TEMPLATE)


async def plan_scenes(
    headline: str,
    summary: str,
    lyrics: Lyrics,
    song_duration: float,
    image_url: str | None,
    api_key: str,
    model: str,
    base_url: str | None = None,
    ollama_base_url: str = "http://localhost:11434/v1",
    output_dir: Path | None = None,
    max_retries: int = 3,
    timed_lyrics_path: Path | None = None,
) -> ScenePlan:
    """Call an LLM to generate a visual scene plan for the music video."""
    num_clips = max(1, math.ceil(song_duration / CLIP_DURATION))
    effective_base = base_url or ollama_base_url
    effective_key = api_key if api_key != "ollama" else "ollama"
    client = openai.AsyncOpenAI(base_url=effective_base, api_key=effective_key)

    # Build per-clip lyrics block from Suno timed data if available
    clip_lyrics_block = ""
    if timed_lyrics_path and timed_lyrics_path.exists():
        try:
            timed_words = json.loads(timed_lyrics_path.read_text(encoding="utf-8"))
            clip_lyrics_block = _build_clip_lyrics_block(timed_words, num_clips, CLIP_DURATION)
            log.info(f"[scenes] Built per-clip lyrics from {timed_lyrics_path.name}")
        except Exception as e:
            log.warning(f"[scenes] Could not parse timed_lyrics.json: {e}")

    if not clip_lyrics_block:
        clip_lyrics_block = "  (timed lyrics not available — use full lyrics above)"

    system_prompt = load_system_prompt()
    user_prompt_template = load_user_prompt_template()
    prompt = user_prompt_template.format(
        headline=headline,
        summary=summary,
        lyrics_text=lyrics.full_text,
        clip_lyrics=clip_lyrics_block,
        duration=song_duration,
        num_clips=num_clips,
        last_clip_index=num_clips - 1,
    )

    log_api_call("scene-planner-request", {
        "base_url": effective_base,
        "model": model,
        "num_clips": num_clips,
        "song_duration": song_duration,
        "headline": headline,
    }, run_dir=output_dir)

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        log.info(f"[scenes] Calling {effective_base} ({model}) — attempt {attempt}/{max_retries}...")
        message = await client.chat.completions.create(
            model=model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
        )

        raw = message.choices[0].message.content.strip()

        log_api_call("scene-planner-response", {
            "raw_response": raw,
            "attempt": attempt,
        }, run_dir=output_dir)

        raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

        try:
            data = json.loads(raw)
            break
        except json.JSONDecodeError as e:
            last_error = e
            log.warning(f"[scenes] JSON parse failed (attempt {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                try:
                    from json_repair import repair_json
                    repaired = repair_json(raw)
                    data = json.loads(repaired)
                    log.warning("[scenes] JSON recovered via json_repair")
                    break
                except Exception:
                    raise ValueError(f"LLM returned invalid JSON after {max_retries} attempts: {e}") from e
            continue

    raw_scenes = data.get("scenes", [])
    scenes: list[SceneDescription] = []
    for s in raw_scenes:
        idx = int(s.get("clip_index", len(scenes)))
        prompt_text = s.get("prompt", "")
        visual_action = s.get("visual_action", "")
        alt_prompts = [a for a in s.get("alt_prompts", []) if a and a.strip()]
        scenes.append(SceneDescription(
            clip_index=idx,
            prompt=prompt_text,
            is_image_to_video=(idx == 0 and image_url is not None),
            visual_action=visual_action,
            alt_prompts=alt_prompts,
            prompt_variants=_build_prompt_variants(prompt_text, visual_action, alt_prompts),
        ))

    # Fill missing clips with scene-specific gap prompts
    if len(scenes) < num_clips:
        log.warning(f"[scenes] LLM returned {len(scenes)} scenes, expected {num_clips} — filling gaps")
        existing_indices = {s.clip_index for s in scenes}
        for i in range(num_clips):
            if i not in existing_indices:
                t_start, t_end = i * CLIP_DURATION, (i + 1) * CLIP_DURATION
                gap_prompt = f"Cinematic news scene about {headline}. Dynamic camera, vibrant colors, 9:16 vertical."
                gap_action = f"Visual moment at {t_start}-{t_end}s: {headline}"
                # Varied gap alts so these clips don't all look the same
                gap_alts = [
                    f"Extreme close-up detail shot related to: {headline}. Macro lens, 9:16 vertical.",
                    f"Abstract symbolic visual representing: {headline}. Motion blur, vibrant colors, 9:16 vertical.",
                    f"Wide environmental establishing shot evoking the mood of: {headline}. 9:16 vertical.",
                ]
                scenes.append(SceneDescription(
                    clip_index=i,
                    prompt=gap_prompt,
                    is_image_to_video=(i == 0 and image_url is not None),
                    visual_action=gap_action,
                    alt_prompts=gap_alts,
                    prompt_variants=_build_prompt_variants(gap_prompt, gap_action, gap_alts),
                ))
    scenes.sort(key=lambda s: s.clip_index)
    scenes = scenes[:num_clips]

    plan = ScenePlan(scenes=scenes)
    log.info(f"[scenes] Plan ready: {len(plan.scenes)} scenes for {song_duration:.0f}s video")
    return plan


if __name__ == "__main__":
    import argparse
    import asyncio
    import os
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    from modules.utils import find_run_dir
    from modules.clip_generator import parse_lyrics_file
    from modules.pollo_generator import parse_headline_file
    from modules.lyrics_generator import Lyrics, LyricSection

    parser = argparse.ArgumentParser(description="Generate visual scene plan via LLM")
    parser.add_argument("--reuse", action="store_true", help="Read from output folder")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--run", type=str, default=None)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--provider", type=str, default="ollama", choices=["ollama", "grok"])
    args = parser.parse_args()

    async def main():
        out = find_run_dir(args.date, args.run)

        headline = "Breaking news"
        summary = "A major event happened"
        image_url = None
        title = "Test Song"
        sections = [
            LyricSection(label="hook", lines=["One punchy line"]),
            LyricSection(label="verse", lines=["Line 1", "Line 2"]),
            LyricSection(label="chorus", lines=["Chorus 1", "Chorus 2"]),
        ]

        if args.reuse:
            headline_file = out / "headline.txt"
            lyrics_file = out / "lyrics.txt"
            if headline_file.exists():
                headline, image_url = parse_headline_file(headline_file)
                lines = headline_file.read_text(encoding="utf-8").splitlines()
                summary = lines[2] if len(lines) > 2 else ""
            if lyrics_file.exists():
                title, sections = parse_lyrics_file(lyrics_file)

        lyrics = Lyrics(title=title, sections=sections, style_prompt="", caption="", topic_tags=[])

        if args.provider == "grok":
            api_key = os.environ.get("XAI_API_KEY", "")
            base_url = "https://api.x.ai/v1"
            model = "grok-3-fast"
        else:
            api_key = "ollama"
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            model = os.getenv("OLLAMA_MODEL", "gemma3")

        plan = await plan_scenes(
            headline=headline,
            summary=summary,
            lyrics=lyrics,
            song_duration=args.duration,
            image_url=image_url,
            api_key=api_key,
            model=model,
            base_url=base_url,
            output_dir=out,
        )

        print(f"\nScene Plan ({len(plan.scenes)} clips):")
        for s in plan.scenes:
            mode = "IMG>VID" if s.is_image_to_video else "TXT>VID"
            action = s.visual_action[:60].encode("ascii", "replace").decode()
            prompt_preview = s.prompt[:80].encode("ascii", "replace").decode()
            print(f"  [{s.clip_index}] ({mode}) {action}...")
            print(f"       prompt: {prompt_preview}...")

    asyncio.run(main())
