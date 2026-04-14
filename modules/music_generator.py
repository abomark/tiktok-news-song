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
    )
    log.info(f"[music] Task ID: {task_id}")

    audio_url, duration = await _poll_until_ready(
        task_id=task_id,
        api_key=sunoapi_key,
        base=sunoapi_base,
    )
    log.info(f"[music] Audio ready: {audio_url}")

    mp3_path = output_dir / "song.mp3"
    await _download_file(audio_url, mp3_path)
    log.info(f"[music] Saved to {mp3_path}")

    # Use API-reported duration if available; fall back to ffprobe
    if not duration:
        duration = await _get_duration(mp3_path)
    log.info(f"[music] Duration: {duration:.1f}s")

    return AudioResult(path=mp3_path, duration_seconds=duration, title=title)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _submit(lyrics: str, style: str, title: str, api_key: str, base: str) -> str:
    """POST to sunoapi.org and return the task ID."""
    # callBackUrl is required by sunoapi.org but we don't use webhooks — we poll instead.
    # Any valid HTTPS URL satisfies the validation; we ignore whatever they POST to it.
    payload = {
        "customMode": True,
        "instrumental": False,
        "model": "V4",
        "prompt": lyrics,
        "style": style,
        "title": title,
        "callBackUrl": "https://example.com/noop",
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{base}/api/v1/generate", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 200:
            log.error(f"[music] sunoapi.org rejected request — code={data.get('code')} msg={data.get('msg')} data={data}")
            raise RuntimeError(f"sunoapi.org error {data.get('code')}: {data.get('msg')}")
        return data["data"]["taskId"]


async def _poll_until_ready(task_id: str, api_key: str, base: str) -> tuple[str, float]:
    """Poll record-info until SUCCESS; return (audio_url, duration_seconds)."""
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

                if status == "SUCCESS":
                    clips = item.get("sunoData", [])
                    if clips:
                        clip = clips[0]
                        audio_url = clip.get("audioUrl") or clip.get("streamAudioUrl")
                        duration = float(clip.get("duration") or 0)
                        if audio_url:
                            return audio_url, duration

                elif status in FAILED:
                    raise RuntimeError(f"sunoapi.org generation failed with status: {status}")

                log.info(f"[music] Status: {status} ({elapsed}s elapsed)")

            except httpx.HTTPError as e:
                log.warning(f"[music] Poll error (retrying): {e}")

    raise TimeoutError(f"sunoapi.org did not complete within {MAX_WAIT}s")


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

    async def main():
        out = Path("output") / date.today().isoformat()
        out.mkdir(parents=True, exist_ok=True)
        result = await generate_music(
            lyrics_text="[HOOK]\nMoney's flying out the door\n\n[VERSE]\nTariffs here, tariffs there\nEverybody's pulling hair\nStocks went down, markets cry\nSomebody tell me why\n\n[CHORUS]\nTax on this, tax on that\nEconomy going flat\nTrade war's back, here we go\nWatch the prices steal the show",
            style_prompt="upbeat pop, punchy drums, catchy hook, satirical, modern production",
            title="Tariff Time",
            output_dir=out,
            sunoapi_key=os.environ["SUNOAPI_KEY"],
        )
        print(f"\nMP3: {result.path}")
        print(f"Duration: {result.duration_seconds:.1f}s")

    asyncio.run(main())
