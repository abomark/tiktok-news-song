"""
Generates video clips via Pollo AI image-to-video API.

For each lyric section:
  - If a news article image URL is available, uses it as the starting frame
  - Falls back to text-to-video if no image is provided
  - Generates one clip per section, polls until complete, downloads .mp4

Setup:
  1. Sign up at https://pollo.ai/api-platform and get an API key
  2. Set POLLO_API_KEY=your_key_here in your .env

API docs: https://docs.pollo.ai
"""

from __future__ import annotations
import asyncio
import logging
import math
from pathlib import Path

import httpx

from modules.lyrics_generator import LyricSection

log = logging.getLogger(__name__)

POLL_INTERVAL = 10   # seconds between status checks
MAX_WAIT = 600       # seconds before giving up (v2.0 can take a few minutes)
DEFAULT_BASE = "https://pollo.ai/api/platform"

# Pollo clips are 4s each; we generate enough to cover each section
CLIP_DURATION = 4

# ── Model registry ────────────────────────────────────────────────────────────
# Each entry: endpoint_prefix, supported clip lengths, default clip length
_MODEL_REGISTRY: dict[str, dict] = {
    "seedance-pro-1-5": {
        "endpoint": "bytedance/seedance-1-5-pro",
        "clip_length": 5,       # supports 4-12s
    },
    "veo3-1": {
        "endpoint": "google/veo3-1",
        "clip_length": 4,       # 4s, 720p, text-to-video
    },
    "veo3-1-fast": {
        "endpoint": "google/veo3-1-fast",
        "clip_length": 6,       # only supports 4, 6, 8
    },
    "veo3-fast": {
        "endpoint": "google/veo3-fast",
        "clip_length": 6,
    },
    "kling-v3": {
        "endpoint": "kling-v3",
        "clip_length": 5,
    },
    "pollo-v2-0": {
        "endpoint": "pollo/pollo-v2-0",
        "clip_length": 5,
    },
    "pollo-v1-6": {
        "endpoint": "pollo/pollo-v1-6",
        "clip_length": 5,
    },
}

DEFAULT_MODEL = "veo3-1"


def _clips_needed(section_duration: float) -> int:
    """How many 5s clips to cover a section (at least 1)."""
    count = math.ceil(section_duration / CLIP_DURATION)
    # If overshoot is ≤2s, use one fewer clip
    if count > 1 and (count * CLIP_DURATION - section_duration) >= (CLIP_DURATION - 2):
        count -= 1
    return max(1, count)


async def generate_clips(
    sections: list[LyricSection],
    headline: str,
    section_duration: float,
    output_dir: Path,
    pollo_api_key: str,
    image_url: str | None = None,
    pollo_base: str = DEFAULT_BASE,
    run_dir: Path | None = None,
) -> list[Path]:
    """Generate Pollo video clips for all sections. Returns list of clip paths."""
    if not pollo_api_key:
        raise RuntimeError("POLLO_API_KEY is required for clip generation.")

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"[clips] Generating {len(sections)} sections via Pollo AI ({section_duration:.1f}s each)...")
    if image_url:
        log.info(f"[clips] Using article image: {image_url}")
    else:
        log.info("[clips] No article image — using text-to-video mode")

    effective_run_dir = run_dir or output_dir
    tasks = [
        _generate_section_clip(
            section=section,
            headline=headline,
            section_duration=section_duration,
            index=i,
            output_dir=output_dir,
            api_key=pollo_api_key,
            image_url=image_url,
            base=pollo_base,
            run_dir=effective_run_dir,
        )
        for i, section in enumerate(sections)
    ]
    return await asyncio.gather(*tasks)


async def generate_clips_from_plan(
    plan,  # ScenePlan from scene_planner
    output_dir: Path,
    pollo_api_key: str,
    image_url: str | None = None,
    pollo_base: str = DEFAULT_BASE,
    run_dir: Path | None = None,
    model: str = DEFAULT_MODEL,
) -> list[Path]:
    """Generate clips from a ScenePlan. Clip 0 uses image-to-video, rest text-to-video."""
    if not pollo_api_key:
        raise RuntimeError("POLLO_API_KEY is required for clip generation.")

    output_dir.mkdir(parents=True, exist_ok=True)
    effective_run_dir = run_dir or output_dir
    semaphore = asyncio.Semaphore(4)

    log.info(f"[clips] Generating {len(plan.scenes)} clips from scene plan via Pollo AI (model={model})...")

    async def _gen_one(scene) -> Path:
        async with semaphore:
            scene_image = image_url if scene.is_image_to_video else None
            mode = "IMG→VID" if scene_image else "TXT→VID"
            log.info(f"[clips] Pollo [{scene.clip_index}] ({mode}) — {scene.visual_action[:50]}...")

            variants = scene.prompt_variants or [scene.prompt]
            log.debug(f"[clips] Pollo [{scene.clip_index}] variants: L0={variants[0][:80]!r} ... Llast={variants[-1][:80]!r}")

            last_err: Exception | None = None
            for attempt, prompt in enumerate(variants):
                try:
                    video_url = await _submit(
                        prompt=prompt,
                        image_url=scene_image if attempt == 0 else None,
                        api_key=pollo_api_key,
                        base=pollo_base,
                        model=model,
                        run_dir=effective_run_dir,
                    )
                    clip_path = output_dir / f"clip_{scene.clip_index:02d}.mp4"
                    await _download_file(video_url, clip_path)
                    if attempt > 0:
                        log.warning(f"[clips] Pollo [{scene.clip_index}] accepted at variant {attempt}/{len(variants)-1} (after content filter)")
                    log.info(f"[clips] Pollo [{scene.clip_index}] saved → {clip_path}")
                    return clip_path
                except RuntimeError as e:
                    last_err = e
                    err_text = str(e).lower()
                    if any(tok in err_text for tok in ("sensitive", "harmful", "content", "policy", "blocked", "nsfw")):
                        log.warning(
                            f"[clips] Pollo [{scene.clip_index}] content filter at variant {attempt}/{len(variants)-1} "
                            f"— retrying with safer variant: {prompt[:120]!r}"
                        )
                        continue
                    raise
            raise RuntimeError(
                f"[clips] Pollo [{scene.clip_index}] — all {len(variants)} variants blocked by content filter. "
                f"Last error: {last_err}"
            )

    results = await asyncio.gather(*[_gen_one(s) for s in plan.scenes])
    # Sort by clip index since gather may return out of order
    return sorted(results, key=lambda p: p.name)


async def _generate_section_clip(
    section: LyricSection,
    headline: str,
    section_duration: float,
    index: int,
    output_dir: Path,
    api_key: str,
    image_url: str | None,
    base: str,
    run_dir: Path | None = None,
) -> Path:
    """Generate clip(s) for one section, concatenate if multiple needed."""
    n_clips = _clips_needed(section_duration)
    lyric_snippet = " / ".join(section.lines[:2])
    prompt = f"News story: {headline}. Scene: {lyric_snippet}. Cinematic, dynamic camera movement, vibrant colors."

    sub_clips = []
    for j in range(n_clips):
        clip_label = f"{index}.{j}" if n_clips > 1 else str(index)
        log.info(f"[clips] Pollo: generating clip [{clip_label}] ({CLIP_DURATION}s) — {lyric_snippet[:50]}...")

        video_url = await _submit(
            prompt=prompt,
            image_url=image_url,
            api_key=api_key,
            base=base,
            run_dir=run_dir,
        )

        clip_path = output_dir / f"clip_{index:02d}_{j}.mp4"
        await _download_file(video_url, clip_path)
        sub_clips.append(clip_path)
        log.info(f"[clips] Pollo: clip [{clip_label}] saved to {clip_path}")

    final_clip = output_dir / f"clip_{index:02d}.mp4"
    if len(sub_clips) == 1:
        sub_clips[0].rename(final_clip)
    else:
        # Concatenate with ffmpeg
        concat_list = output_dir / f"_concat_{index}.txt"
        concat_list.write_text(
            "\n".join(f"file '{c.name}'" for c in sub_clips),
            encoding="utf-8",
        )
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list.name, "-c", "copy", final_clip.name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(output_dir),
        )
        await proc.communicate()
        concat_list.unlink(missing_ok=True)
        for c in sub_clips:
            c.unlink(missing_ok=True)

    log.info(f"[clips] Section [{index}] ready: {final_clip}")
    return final_clip


async def _submit(
    prompt: str,
    image_url: str | None,
    api_key: str,
    base: str,
    model: str = DEFAULT_MODEL,
    run_dir: Path | None = None,
) -> str:
    """POST to Pollo API and return the video URL (synchronous — result is in the response)."""
    from modules.utils import log_api_call

    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    # Resolve endpoint and clip length from registry, fall back to raw model name
    registry_entry = _MODEL_REGISTRY.get(model)
    if registry_entry:
        endpoint = f"{base}/generation/{registry_entry['endpoint']}"
        clip_len = registry_entry["clip_length"]
    else:
        endpoint = f"{base}/generation/{model}"
        clip_len = CLIP_DURATION

    input_body: dict = {
        "prompt": prompt,
        "resolution": "720p",
        "aspectRatio": "9:16",
        "length": clip_len,
        "generateAudio": False,
    }
    if image_url:
        input_body["image"] = image_url
    payload = {"input": input_body}

    log.info(f"[clips] Pollo request → {endpoint}")
    log_api_call("pollo-request", {
        "endpoint": endpoint,
        "payload": payload,
    }, run_dir=run_dir)

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(endpoint, json=payload, headers=headers)
        if not resp.is_success:
            log.error(f"[clips] Pollo submit failed {resp.status_code}: {resp.text}")
        resp.raise_for_status()
        data = resp.json()

    # v1.6 returns results immediately; v2.0 returns a taskId to poll
    results = data.get("result", [])
    if results:
        first = results[0]
        log_api_call("pollo-response", {
            "status": first.get("status"),
            "videoUrl": first.get("videoUrl"),
            "credit": first.get("credit"),
            "failMsg": first.get("failMsg"),
        }, run_dir=run_dir)
        if first.get("status") != "succeed":
            raise RuntimeError(f"Pollo generation failed: {first.get('failMsg', 'unknown')}")
        video_url = first.get("videoUrl")
        if not video_url:
            raise RuntimeError(f"Pollo succeeded but no videoUrl in result: {first}")
        log.info(f"[clips] Pollo: video ready — {video_url} ({first.get('credit')} credits)")
        return video_url

    # Async model (v2.0+): poll for completion
    task_data = data.get("data", {})
    task_id = task_data.get("taskId")
    if task_id:
        log.info(f"[clips] Pollo: async task {task_id} — polling...")
        log_api_call("pollo-response", {"taskId": task_id, "status": "waiting"}, run_dir=run_dir)
        video_url = await _poll_until_ready(task_id, api_key, base, run_dir=run_dir)
        return video_url

    log_api_call("pollo-response", {"error": "unknown response format", "raw": data}, run_dir=run_dir)
    raise RuntimeError(f"Pollo returned unexpected response: {data}")


async def _poll_until_ready(task_id: str, api_key: str, base: str, run_dir: Path | None = None) -> str:
    """Poll GET /api/platform/generation/{taskId}/status until succeed."""
    from modules.utils import log_api_call

    headers = {"X-API-KEY": api_key}
    poll_url = f"{base}/generation/{task_id}/status"
    elapsed = 0

    async with httpx.AsyncClient(timeout=30) as client:
        while elapsed < MAX_WAIT:
            await asyncio.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL

            try:
                resp = await client.get(poll_url, headers=headers)
                resp.raise_for_status()
                body = resp.json()
                # Response: {"code":"SUCCESS","data":{"taskId":"...","generations":[...]}}
                data = body.get("data", body)

                if elapsed <= POLL_INTERVAL:
                    log.info(f"[clips] Pollo first poll: {body}")

                gens = data.get("generations", [])
                if gens:
                    gen = gens[0]
                    status = gen.get("status", "")
                else:
                    status = data.get("status", "")

                if status == "succeed":
                    video_url = gens[0].get("url") if gens else None
                    if video_url:
                        log_api_call("pollo-poll-complete", {
                            "taskId": task_id, "videoUrl": video_url, "elapsed": elapsed,
                        }, run_dir=run_dir)
                        log.info(f"[clips] Pollo: task done ({elapsed}s) — {video_url}")
                        return video_url
                    log.warning(f"[clips] Pollo: succeed but no URL — data: {body}")

                elif status == "failed":
                    fail_msg = gens[0].get("failMsg", "unknown") if gens else "unknown"
                    raise RuntimeError(f"Pollo generation failed: {fail_msg}")

                log.info(f"[clips] Pollo: status={status} ({elapsed}s)")

            except httpx.HTTPError as e:
                log.warning(f"[clips] Pollo: poll error (retrying): {e}")

    raise TimeoutError(f"Pollo did not complete within {MAX_WAIT}s for task {task_id}")


async def _download_file(url: str, dest: Path) -> None:
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=8192):
                    f.write(chunk)


def parse_headline_file(headline_path: Path) -> tuple[str, str | None]:
    """Parse headline.txt and return (headline, image_url)."""
    text = headline_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    headline = lines[0].strip() if lines else ""
    image_url = None
    for line in lines:
        if line.startswith("IMAGE: "):
            image_url = line.removeprefix("IMAGE: ").strip()
            break
    return headline, image_url


if __name__ == "__main__":
    import argparse
    import os
    from dotenv import load_dotenv
    from datetime import date
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    from modules.utils import find_run_dir
    from modules.clip_generator import parse_lyrics_file
    from modules.lyrics_generator import LyricSection

    parser = argparse.ArgumentParser(description="Generate video clips via Pollo AI")
    parser.add_argument("--reuse", action="store_true", help="Read headline.txt + lyrics.txt from output folder")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--run", type=str, default=None)
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()

    async def main():
        out = find_run_dir(args.date, args.run)
        out.mkdir(parents=True, exist_ok=True)

        pollo_key = os.getenv("POLLO_API_KEY", "")
        if not pollo_key:
            print("Set POLLO_API_KEY in .env")
            return

        headline = "Breaking news story"
        image_url = None
        sections = [
            LyricSection(label="hook",   lines=["One punchy line"]),
            LyricSection(label="verse",  lines=["Line 1", "Line 2"]),
            LyricSection(label="chorus", lines=["Chorus line 1", "Chorus line 2"]),
        ]

        if args.reuse:
            headline_file = out / "headline.txt"
            lyrics_file = out / "lyrics.txt"
            if headline_file.exists():
                headline, image_url = parse_headline_file(headline_file)
            if lyrics_file.exists():
                _, sections = parse_lyrics_file(lyrics_file)

        section_duration = args.duration / len(sections)
        clip_paths = await generate_clips(
            sections=sections,
            headline=headline,
            section_duration=section_duration,
            output_dir=out,
            pollo_api_key=pollo_key,
            image_url=image_url,
        )
        print(f"\nGenerated {len(clip_paths)} clips:")
        for p in clip_paths:
            print(f"  {p}")

    asyncio.run(main())
