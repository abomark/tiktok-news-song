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

CLIP_DURATION = 5  # seconds per Pollo clip

_ASSETS_DIR = Path(__file__).parent.parent / "assets"

_DEFAULT_SYSTEM_PROMPT = """You are a viral TikTok music video director. You create visually compelling scene-by-scene plans
for short-form news music videos. Each scene is exactly 5 seconds of video.

Your scenes tell a VISUAL STORY that evolves — each scene builds on the previous one.
Every video must include at least one "viral moment" — an unexpected visual twist, dramatic reveal,
or absurd escalation that makes viewers replay the video.

You always respond with valid JSON only — no markdown fences, no extra text."""

_DEFAULT_USER_PROMPT_TEMPLATE = """Create a scene-by-scene visual story plan for a TikTok news music video.

NEWS HEADLINE: {headline}
NEWS SUMMARY: {summary}

SONG LYRICS:
{lyrics_text}

TOTAL DURATION: {duration}s
NUMBER OF 5-SECOND CLIPS NEEDED: {num_clips}

FIRST CLIP (clip 0): This will use IMAGE-TO-VIDEO starting from the news article's hero image.
  - Describe how the still image should come to life with cinematic motion
  - The prompt should complement the existing image, not contradict it

SUBSEQUENT CLIPS (clips 1 through {last_clip_index}): These are TEXT-TO-VIDEO only.
  - Each clip should describe a unique visual scene
  - Scenes must visually evolve to tell a story arc (setup -> escalation -> viral twist -> resolution)
  - Include one "viral moment" clip — something unexpected, absurd, or dramatic

VISUAL STYLE: Cinematic, dynamic camera movement, vibrant colors, 9:16 vertical format,
TikTok-ready, no visible text or words in the video.

Respond with this exact JSON structure:
{{
  "scenes": [
    {{
      "clip_index": 0,
      "visual_action": "The news photo comes alive — camera slowly pushes in as...",
      "prompt": "Cinematic news scene, [detailed image-to-video prompt]. Dynamic camera push-in, vibrant colors, 9:16 vertical."
    }},
    {{
      "clip_index": 1,
      "visual_action": "Description of what the viewer sees...",
      "prompt": "Detailed text-to-video prompt for the AI video generator..."
    }}
  ]
}}

Generate exactly {num_clips} scene objects (clip_index 0 through {last_clip_index}).
Each prompt should be 1-2 sentences, vivid and specific. Avoid generic descriptions.
Make the story escalate and include at least one unexpected twist."""


@dataclass
class SceneDescription:
    clip_index: int
    prompt: str
    is_image_to_video: bool
    visual_action: str


@dataclass
class ScenePlan:
    scenes: list[SceneDescription]


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
) -> ScenePlan:
    """Call an LLM to generate a visual scene plan for the music video."""
    num_clips = max(1, math.ceil(song_duration / CLIP_DURATION))
    effective_base = base_url or ollama_base_url
    effective_key = api_key if api_key != "ollama" else "ollama"
    client = openai.AsyncOpenAI(base_url=effective_base, api_key=effective_key)

    system_prompt = load_system_prompt()
    user_prompt_template = load_user_prompt_template()
    prompt = user_prompt_template.format(
        headline=headline,
        summary=summary,
        lyrics_text=lyrics.full_text,
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
        scenes.append(SceneDescription(
            clip_index=idx,
            prompt=s.get("prompt", ""),
            is_image_to_video=(idx == 0 and image_url is not None),
            visual_action=s.get("visual_action", ""),
        ))

    # Fill missing clips with fallback prompts
    if len(scenes) < num_clips:
        log.warning(f"[scenes] LLM returned {len(scenes)} scenes, expected {num_clips} — filling gaps")
        existing_indices = {s.clip_index for s in scenes}
        for i in range(num_clips):
            if i not in existing_indices:
                scenes.append(SceneDescription(
                    clip_index=i,
                    prompt=f"Cinematic news scene about {headline}. Dynamic camera, vibrant colors, 9:16 vertical.",
                    is_image_to_video=(i == 0 and image_url is not None),
                    visual_action="Fallback scene",
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
