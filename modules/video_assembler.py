"""
Step 2: Assemble video clips + audio into final.mp4 with watermark.

Usage:
    python -m modules.video_assembler                   # uses today's output folder
    python -m modules.video_assembler --date 2026-04-12
    python -m modules.video_assembler --duration 30.0
"""

from __future__ import annotations
import asyncio
import logging
import shutil
import textwrap
from pathlib import Path

from modules.lyrics_generator import LyricSection
from modules.clip_generator import parse_lyrics_file

log = logging.getLogger(__name__)

WIDTH = 1080
HEIGHT = 1920
WATERMARK = "@currentnoise"
_FONT_ABS = Path(__file__).parent.parent / "assets" / "fonts" / "Roboto-Bold.ttf"


async def assemble_video(
    clip_paths: list[Path],
    sections: list[LyricSection],
    section_duration: float,
    audio_path: Path,
    output_path: Path,
) -> Path:
    """Assemble video clips into final.mp4 with watermark and audio."""
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy font locally (avoids C: in ffmpeg filter paths on Windows)
    local_font = output_dir / "Roboto-Bold.ttf"
    if not local_font.exists():
        shutil.copy2(_FONT_ABS, local_font)
    font_path = local_font.name

    temp_files = []
    wm_file = output_dir / "_watermark.txt"
    wm_file.write_bytes(WATERMARK.encode("ascii", "ignore"))
    temp_files.append(wm_file)

    filter_parts = []
    inputs = []
    n = len(clip_paths)

    for i in range(n):
        inputs += ["-i", str(clip_paths[i].resolve())]

        # Scale, crop, and trim each clip to exact dimensions and duration
        scale_trim = (
            f"[{i}:v]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={WIDTH}:{HEIGHT},"
            f"setpts=PTS-STARTPTS,"
            f"trim=duration={section_duration},setpts=PTS-STARTPTS,"
            f"fps=30[v{i}]"
        )

        # Watermark overlay
        watermark_dt = (
            f"[v{i}]drawtext="
            f"textfile={wm_file.name}:"
            f"fontfile={font_path}:"
            f"fontsize=42:"
            f"fontcolor=white@0.6:"
            f"borderw=2:bordercolor=black@0.4:"
            f"x=w-text_w-30:"
            f"y=60"
            f"[out{i}]"
        )

        filter_parts += [scale_trim, watermark_dt]

    # Concatenate all segments
    concat_inputs = "".join(f"[out{i}]" for i in range(n))
    filter_parts.append(f"{concat_inputs}concat=n={n}:v=1:a=0[outv]")

    # Write filter to script file
    filter_script = output_dir / "_filter.txt"
    filter_script.write_text(";\n".join(filter_parts), encoding="utf-8")
    temp_files.append(filter_script)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-i", str(audio_path.resolve()),
        "-filter_complex_script", filter_script.name,
        "-map", "[outv]",
        "-map", f"{n}:a",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        output_path.name,
    ]

    log.info(f"[assemble] Assembling {n} clips into {output_path.name}...")
    log.debug(f"[assemble] ffmpeg command: {' '.join(cmd)}")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(output_dir),
    )
    _, stderr = await proc.communicate()

    for tf in temp_files:
        tf.unlink(missing_ok=True)

    if proc.returncode != 0:
        log.error(f"[assemble] ffmpeg stderr:\n{stderr.decode()}")
        raise RuntimeError(f"ffmpeg assembly failed with exit code {proc.returncode}")

    log.info(f"[assemble] Done: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    import os
    from dotenv import load_dotenv
    from datetime import date
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    from modules.utils import find_run_dir

    parser = argparse.ArgumentParser(description="Assemble clips + audio into final.mp4")
    parser.add_argument("--date", type=str, default=None, help="Output date folder (default: today)")
    parser.add_argument("--run", type=str, default=None, help="Run folder: number, slug, or name (default: latest)")
    parser.add_argument("--duration", type=float, default=40.0, help="Total song duration in seconds")
    args = parser.parse_args()

    async def main():
        out = find_run_dir(args.date, args.run)

        # Find clips
        clip_paths = sorted(out.glob("clip_*.mp4"))
        if not clip_paths:
            print(f"No clip_*.mp4 found in {out}")
            return

        # Parse lyrics for section count
        lyrics_file = out / "lyrics.txt"
        if lyrics_file.exists():
            _, sections = parse_lyrics_file(lyrics_file)
        else:
            # Fallback: one section per clip
            sections = [LyricSection(label=f"section_{i}", lines=[""]) for i in range(len(clip_paths))]

        audio_path = out / "song.mp3"
        if not audio_path.exists():
            print(f"No song.mp3 found in {out}")
            return

        section_duration = args.duration / len(clip_paths)
        output_path = out / "final.mp4"

        print(f"Clips: {len(clip_paths)}")
        print(f"Section duration: {section_duration:.1f}s")
        print(f"Total: {args.duration:.1f}s")

        result = await assemble_video(
            clip_paths=clip_paths,
            sections=sections,
            section_duration=section_duration,
            audio_path=audio_path,
            output_path=output_path,
        )
        print(f"Video: {result}")

    asyncio.run(main())
