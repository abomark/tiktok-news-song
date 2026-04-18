"""
Generates a song MP3 via sunoapi.org (hosted Suno API).

Setup:
  1. Sign up at https://sunoapi.org and get an API key.
  2. Set SUNOAPI_KEY=your_key_here in your .env

API docs: https://docs.sunoapi.org
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

POLL_INTERVAL = 10   # seconds between status checks
MAX_WAIT = 600       # seconds before giving up (generations take 2-5 min)
DEFAULT_BASE = "https://api.sunoapi.org"


@dataclass
class AudioResult:
    path: Path
    duration_seconds: float
    title: str


async def generate_music(
    lyrics_text: str,
    style_prompt: str,
    title: str,
    output_dir: Path,
    sunoapi_key: str,
    sunoapi_base: str = DEFAULT_BASE,
) -> AudioResult:
    """Submit lyrics to sunoapi.org, poll until done, download .mp3."""
    from modules.utils import log_api_call

    log.info("[music] Submitting to sunoapi.org...")

    # Append [END] so Suno stops at the end of the written lyrics (~30-40s)
    # without it Suno often extends the song to 2+ minutes
    lyrics_with_end = lyrics_text.rstrip() + "\n\n[END]"

    task_id = await _submit(
        lyrics=lyrics_with_end,
        style=style_prompt,
        title=title,
        api_key=sunoapi_key,
        base=sunoapi_base,
        run_dir=output_dir,
    )
    log.info(f"[music] Task ID: {task_id}")

    audio_url, duration, audio_id = await _poll_until_ready(
        task_id=task_id,
        api_key=sunoapi_key,
        base=sunoapi_base,
        run_dir=output_dir,
    )
    log.info(f"[music] Audio ready: {audio_url}")
    log_api_call("suno-complete", {
        "taskId": task_id,
        "audioId": audio_id,
        "audioUrl": audio_url,
        "duration": duration,
        "title": title,
    }, run_dir=output_dir)

    mp3_path = output_dir / "song.mp3"
    await _download_file(audio_url, mp3_path)
    log.info(f"[music] Saved to {mp3_path}")

    # Use API-reported duration if available; fall back to ffprobe
    if not duration:
        duration = await _get_duration(mp3_path)
    log.info(f"[music] Duration: {duration:.1f}s")

    # Fetch Suno's own word-level timestamps — much more accurate than Whisper
    if audio_id:
        await _fetch_timestamped_lyrics(
            task_id=task_id,
            audio_id=audio_id,
            api_key=sunoapi_key,
            base=sunoapi_base,
            output_dir=output_dir,
        )

    return AudioResult(path=mp3_path, duration_seconds=duration, title=title)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _submit(lyrics: str, style: str, title: str, api_key: str, base: str, run_dir: Path | None = None) -> str:
    """POST to sunoapi.org and return the task ID."""
    from modules.utils import log_api_call

    # callBackUrl is required by sunoapi.org but we don't use webhooks — we poll instead.
    # Any valid HTTPS URL satisfies the validation; we ignore whatever they POST to it.
    payload = {
        "customMode": True,
        "instrumental": False,
        "model": "V5",
        "prompt": lyrics,
        "style": f"{style}, short song, exactly 30-40 seconds, no intro, no outro, no instrumental break, lyrics start immediately, end after chorus",
        "title": title,
        "callBackUrl": "https://example.com/noop",
    }
    log_api_call("suno-request", {
        "endpoint": f"{base}/api/v1/generate",
        "payload": {k: v for k, v in payload.items() if k != "callBackUrl"},
    }, run_dir=run_dir)

    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{base}/api/v1/generate", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 200:
            log.error(f"[music] sunoapi.org rejected request — code={data.get('code')} msg={data.get('msg')} data={data}")
            raise RuntimeError(f"sunoapi.org error {data.get('code')}: {data.get('msg')}")
        task_id = data["data"]["taskId"]
        log_api_call("suno-response", {"taskId": task_id}, run_dir=run_dir)
        return task_id


async def _poll_until_ready(task_id: str, api_key: str, base: str, run_dir: Path | None = None) -> tuple[str, float, str]:
    """Poll record-info until SUCCESS; return (audio_url, duration_seconds, audio_id)."""
    headers = {"Authorization": f"Bearer {api_key}"}
    elapsed = 0
    FAILED = {"CREATE_TASK_FAILED", "GENERATE_AUDIO_FAILED", "CALLBACK_EXCEPTION", "SENSITIVE_WORD_ERROR"}

    async with httpx.AsyncClient(timeout=30) as client:
        while elapsed < MAX_WAIT:
            await asyncio.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
            try:
                resp = await client.get(
                    f"{base}/api/v1/generate/record-info",
                    params={"taskId": task_id},
                    headers=headers,
                )
                resp.raise_for_status()
                body = resp.json()
                if elapsed <= POLL_INTERVAL:
                    log.info(f"[music] First poll response: {body}")
                item = body.get("data", {})
                status = item.get("status", "")

                if status in ("SUCCESS", "FIRST_SUCCESS"):
                    # API returns sunoData nested inside a 'response' dict
                    clips = item.get("sunoData") or []
                    response = item.get("response")
                    if not clips and isinstance(response, dict):
                        clips = response.get("sunoData") or []
                    elif not clips and isinstance(response, list):
                        clips = response
                    # Pick first clip that has a usable audio URL
                    for clip in clips:
                        audio_url = clip.get("audioUrl") or clip.get("streamAudioUrl")
                        duration = float(clip.get("duration") or 0)
                        audio_id = clip.get("id") or clip.get("audioId") or ""
                        if audio_url:
                            return audio_url, duration, audio_id

                elif status in FAILED:
                    raise RuntimeError(f"sunoapi.org generation failed with status: {status}")

                log.info(f"[music] Status: {status} ({elapsed}s elapsed)")

            except httpx.HTTPError as e:
                log.warning(f"[music] Poll error (retrying): {e}")

    raise TimeoutError(f"sunoapi.org did not complete within {MAX_WAIT}s")


async def _fetch_timestamped_lyrics(
    task_id: str,
    audio_id: str,
    api_key: str,
    base: str,
    output_dir: Path,
) -> None:
    """Fetch Suno's word-level timestamps and save to timed_lyrics.json."""
    from modules.utils import log_api_call
    import json

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{base}/api/v1/generate/get-timestamped-lyrics"
    payload = {"taskId": task_id, "audioId": audio_id}

    log.info(f"[music] Fetching timestamped lyrics for audioId={audio_id}...")
    log_api_call("suno-timestamps-request", {"taskId": task_id, "audioId": audio_id}, run_dir=output_dir)

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if not resp.is_success:
            log.warning(f"[music] Timestamped lyrics failed {resp.status_code}: {resp.text}")
            return
        data = resp.json()

    log_api_call("suno-timestamps-response", data, run_dir=output_dir)

    aligned_words = data.get("data", {}).get("alignedWords", [])
    if not aligned_words:
        log.warning("[music] No alignedWords in timestamped lyrics response")
        return

    # Save as timed_lyrics.json for the captioner to consume
    timed_lyrics_path = output_dir / "timed_lyrics.json"
    timed_lyrics_path.write_text(
        json.dumps(aligned_words, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info(f"[music] Saved {len(aligned_words)} timed words → {timed_lyrics_path}")


async def _download_file(url: str, dest: Path) -> None:
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=8192):
                    f.write(chunk)


async def _get_duration(mp3_path: Path) -> float:
    """Use ffprobe to get audio duration in seconds."""
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(mp3_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    try:
        return float(stdout.decode().strip())
    except ValueError:
        return 40.0   # fallback estimate


if __name__ == "__main__":
    import asyncio, os
    from dotenv import load_dotenv
    from pathlib import Path
    from datetime import date
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    import argparse
    from modules.utils import find_run_dir
    from modules.clip_generator import parse_lyrics_file

    parser = argparse.ArgumentParser(description="Generate music via sunoapi.org")
    parser.add_argument("--reuse", action="store_true", help="Read lyrics.txt from output folder")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--run", type=str, default=None)
    args = parser.parse_args()

    async def main():
        out = find_run_dir(args.date, args.run)
        out.mkdir(parents=True, exist_ok=True)

        lyrics_text = "[HOOK]\nBreaking news today\n\n[VERSE]\nSomething happened out there\nNobody seems to care\n\n[CHORUS]\nThis is the song\nThis is the song"
        style_prompt = "upbeat pop, punchy drums, catchy hook, satirical, modern production"
        title = "News Song"

        if args.reuse:
            lyrics_file = out / "lyrics.txt"
            if lyrics_file.exists():
                title, sections = parse_lyrics_file(lyrics_file)
                lyrics_text = "\n\n".join(
                    f"[{s.label.upper()}]\n" + "\n".join(s.lines)
                    for s in sections
                )
                # Read style from lyrics.txt STYLE line if present
                for line in lyrics_file.read_text(encoding="utf-8").splitlines():
                    if line.startswith("STYLE:"):
                        style_prompt = line.removeprefix("STYLE:").strip()
                        break

        result = await generate_music(
            lyrics_text=lyrics_text,
            style_prompt=style_prompt,
            title=title,
            output_dir=out,
            sunoapi_key=os.environ["SUNOAPI_KEY"],
        )
        print(f"\nMP3: {result.path}")
        print(f"Duration: {result.duration_seconds:.1f}s")

    asyncio.run(main())
