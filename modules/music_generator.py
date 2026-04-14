"""
Generates a song MP3 via the suno-api Docker wrapper.

Setup:
  git clone https://github.com/gcui-art/suno-api.git && cd suno-api
  docker compose up -d   (set SUNO_COOKIE in suno-api/.env first)
  Then set SUNO_API_BASE=http://localhost:3000 in your project .env

The suno-api project: https://github.com/gcui-art/suno-api
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

POLL_INTERVAL = 5   # seconds between status checks
MAX_WAIT = 300       # seconds before giving up


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
    suno_cookie: str,
    suno_api_base: str,
) -> AudioResult:
    """Submit lyrics to Suno, poll until done, download .mp3."""
    log.info("[music] Submitting to Suno...")

    song_id = await _submit_to_suno(
        lyrics=lyrics_text,
        style=style_prompt,
        title=title,
        cookie=suno_cookie,
        base=suno_api_base,
    )
    log.info(f"[music] Suno job ID: {song_id}")

    audio_url = await _poll_until_ready(song_id=song_id, base=suno_api_base)
    log.info(f"[music] Audio ready: {audio_url}")

    mp3_path = output_dir / "song.mp3"
    await _download_file(audio_url, mp3_path)
    log.info(f"[music] Saved to {mp3_path}")

    duration = await _get_duration(mp3_path)
    log.info(f"[music] Duration: {duration:.1f}s")

    return AudioResult(path=mp3_path, duration_seconds=duration, title=title)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _submit_to_suno(lyrics: str, style: str, title: str, cookie: str, base: str) -> str:
    """POST to suno-api and return the song clip ID."""
    payload = {
        "prompt": lyrics,
        "tags": style,
        "title": title,
        "make_instrumental": False,
        "wait_audio": False,
    }
    headers = {"Cookie": cookie}
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{base}/api/custom_generate", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        # suno-api returns a list; take first clip
        if isinstance(data, list):
            return data[0]["id"]
        return data["id"]


async def _poll_until_ready(song_id: str, base: str) -> str:
    """Poll the suno-api feed endpoint until the clip is complete."""
    elapsed = 0
    async with httpx.AsyncClient(timeout=30) as client:
        while elapsed < MAX_WAIT:
            await asyncio.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
            try:
                resp = await client.get(f"{base}/api/feed", params={"ids": song_id})
                resp.raise_for_status()
                items = resp.json()
                if isinstance(items, list) and items:
                    item = items[0]
                    status = item.get("status", "")
                    if status == "complete":
                        audio_url = item.get("audio_url") or item.get("stream_audio_url")
                        if audio_url:
                            return audio_url
                    elif status in ("error", "failed"):
                        raise RuntimeError(f"Suno generation failed: {item}")
                    log.debug(f"[music] Suno status: {status} ({elapsed}s elapsed)")
            except httpx.HTTPError as e:
                log.warning(f"[music] Poll error (retrying): {e}")

    raise TimeoutError(f"Suno did not complete within {MAX_WAIT}s")


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
            suno_cookie=os.environ["SUNO_COOKIE"],
            suno_api_base=os.getenv("SUNO_API_BASE", "http://localhost:3000"),
        )
        print(f"\nMP3: {result.path}")
        print(f"Duration: {result.duration_seconds:.1f}s")

    asyncio.run(main())
