"""
Publishes a video to TikTok using the Content Posting API v2.
Handles OAuth token refresh automatically.

API docs: https://developers.tiktok.com/doc/content-posting-api-get-started
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

TIKTOK_API_BASE = "https://open.tiktokapis.com"
CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB chunks


@dataclass
class PublishResult:
    publish_id: str
    post_url: str | None = None


async def publish_video(
    video_path: Path,
    caption: str,
    client_key: str,
    client_secret: str,
    refresh_token: str,
) -> PublishResult:
    """Full upload + publish flow. Returns the publish_id."""

    # 1. Refresh access token
    access_token = await _refresh_access_token(client_key, client_secret, refresh_token)
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

    video_size = video_path.stat().st_size
    chunk_count = math.ceil(video_size / CHUNK_SIZE)

    log.info(f"[tiktok] Initialising upload ({video_size / 1e6:.1f} MB, {chunk_count} chunk(s))...")

    # 2. Init upload
    init_payload = {
        "post_info": {
            "title": caption[:150],   # TikTok caption limit is 2200 but keep it punchy
            "privacy_level": "PUBLIC_TO_EVERYONE",
            "disable_duet": False,
            "disable_comment": False,
            "disable_stitch": False,
        },
        "source_info": {
            "source": "FILE_UPLOAD",
            "video_size": video_size,
            "chunk_size": CHUNK_SIZE,
            "total_chunk_count": chunk_count,
        },
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{TIKTOK_API_BASE}/v2/post/publish/video/init/",
            headers=headers,
            json=init_payload,
        )
        _check_response(resp, "init upload")
        data = resp.json()["data"]
        publish_id = data["publish_id"]
        upload_url = data["upload_url"]

    log.info(f"[tiktok] Publish ID: {publish_id}")

    # 3. Upload chunks
    await _upload_chunks(video_path, upload_url, video_size, chunk_count)

    log.info(f"[tiktok] Upload complete. TikTok is processing the video...")
    return PublishResult(publish_id=publish_id)


async def _refresh_access_token(client_key: str, client_secret: str, refresh_token: str) -> str:
    """Exchange refresh token for a new access token."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://open.tiktokapis.com/v2/oauth/token/",
            data={
                "client_key": client_key,
                "client_secret": client_secret,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        _check_response(resp, "token refresh")
        token_data = resp.json()
        log.debug(f"[tiktok] Token refreshed, expires in {token_data.get('expires_in')}s")
        return token_data["access_token"]


async def _upload_chunks(
    video_path: Path,
    upload_url: str,
    video_size: int,
    chunk_count: int,
) -> None:
    async with httpx.AsyncClient(timeout=120) as client:
        with video_path.open("rb") as f:
            for chunk_index in range(chunk_count):
                chunk_data = f.read(CHUNK_SIZE)
                start = chunk_index * CHUNK_SIZE
                end = min(start + len(chunk_data) - 1, video_size - 1)

                headers = {
                    "Content-Type": "video/mp4",
                    "Content-Range": f"bytes {start}-{end}/{video_size}",
                    "Content-Length": str(len(chunk_data)),
                }
                resp = await client.put(upload_url, content=chunk_data, headers=headers)
                _check_response(resp, f"chunk {chunk_index + 1}/{chunk_count}")
                log.info(f"[tiktok] Uploaded chunk {chunk_index + 1}/{chunk_count}")


def _check_response(resp: httpx.Response, context: str) -> None:
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        log.error(f"[tiktok] {context} failed: {resp.status_code} — {resp.text}")
        raise RuntimeError(f"TikTok API error during {context}: {resp.status_code}") from e


if __name__ == "__main__":
    import asyncio, os
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    async def main():
        result = await publish_video(
            video_path=Path("output/test/final.mp4"),
            caption="Testing @currentnoise auto-poster\n\n#newsatire #politicalsatire #aimusic #fyp",
            client_key=os.environ["TIKTOK_CLIENT_KEY"],
            client_secret=os.environ["TIKTOK_CLIENT_SECRET"],
            refresh_token=os.environ["TIKTOK_REFRESH_TOKEN"],
        )
        print(f"Publish ID: {result.publish_id}")

    asyncio.run(main())
