"""
Repair / resume incomplete pipeline runs for a given date.

Walks every run directory, identifies the most upstream missing step, and
builds everything downstream from there. Skips steps whose output already exists.

Pipeline order:
  1. lyrics.txt          ← needs headline.txt + LLM
  2. song.mp3            ← needs lyrics.txt + Suno
  3. scene_plan.json     ← needs lyrics.txt + song.mp3 + LLM
  4. clip_NN.mp4 (all)   ← needs scene_plan.json + Pollo
  5. final.mp4           ← needs all clips + song.mp3
  6. final_captioned.mp4 ← needs final.mp4
  7. final_tiktok.mp4    ← needs final_captioned.mp4 + HOOK_CAPTION in lyrics.txt

Usage:
    python repair.py                          # today, grok provider
    python repair.py --date 2026-04-19
    python repair.py --date 2026-04-19 --run 01-strait
    python repair.py --dry-run                # report only, no generation
    python repair.py --provider ollama        # use local Ollama for LLM steps
"""

from __future__ import annotations
import argparse
import asyncio
import json
import logging
import os
import subprocess
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _audio_duration(mp3: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(mp3)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, TypeError):
        return 30.0


def _load_scene_plan(run_dir: Path) -> list[dict] | None:
    path = run_dir / "scene_plan.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, list) else raw.get("scenes", raw)


def _load_hook_caption(lyrics_path: Path) -> str:
    if not lyrics_path.exists():
        return ""
    text = lyrics_path.read_text(encoding="utf-8")
    if "HOOK_CAPTION:" not in text:
        return ""
    block = text.split("HOOK_CAPTION:", 1)[1].strip()
    return "\n".join(block.splitlines()[:2])


def _parse_lyrics_txt(lyrics_path: Path):
    """Parse lyrics.txt into a Lyrics object (mirrors pipeline.py resume logic)."""
    from modules.lyrics_generator import Lyrics, LyricSection
    from modules.clip_generator import parse_lyrics_file

    text = lyrics_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    title = ""
    style_prompt = ""
    for line in lines:
        if line.startswith("TITLE:"):
            title = line.removeprefix("TITLE:").strip()
        elif line.startswith("STYLE:"):
            style_prompt = line.removeprefix("STYLE:").strip()

    _, sections = parse_lyrics_file(lyrics_path)

    caption = ""
    hook_caption = ""
    if "CAPTION:" in text:
        caption_tail = text.split("CAPTION:", 1)[1].strip()
        caption = caption_tail.split("\n---\n", 1)[0].strip()
    if "HOOK_CAPTION:" in text:
        block = text.split("HOOK_CAPTION:", 1)[1].strip()
        hook_caption = "\n".join(block.splitlines()[:2])

    return Lyrics(
        title=title,
        sections=sections,
        style_prompt=style_prompt,
        caption=caption,
        topic_tags=[],
        hook_caption=hook_caption,
    )


def _find_run_dirs(date_str: str, run_filter: str | None) -> list[Path]:
    base = Path("output") / date_str
    if not base.exists():
        return []
    dirs = sorted(p for p in base.iterdir() if p.is_dir())
    if run_filter:
        dirs = [d for d in dirs if run_filter in d.name]
    return dirs


def _llm_config(provider: str) -> dict:
    if provider == "grok":
        return {
            "api_key": os.environ.get("XAI_API_KEY", ""),
            "base_url": "https://api.x.ai/v1",
            "model": "grok-3-fast",
        }
    return {
        "api_key": "ollama",
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "model": os.getenv("OLLAMA_MODEL", "gemma3"),
    }


def _ok(msg: str):   print(f"  \033[32m✓\033[0m {msg}")
def _miss(msg: str): print(f"  \033[33m→\033[0m {msg}")
def _err(msg: str):  print(f"  \033[31m✗\033[0m {msg}")
def _skip(msg: str): print(f"  · {msg}")


# ── per-step repair ────────────────────────────────────────────────────────────

async def _ensure_lyrics(run_dir: Path, llm: dict, dry_run: bool) -> bool:
    lyrics_file = run_dir / "lyrics.txt"
    if lyrics_file.exists():
        _ok("lyrics.txt exists")
        return True

    headline_file = run_dir / "headline.txt"
    if not headline_file.exists():
        _err("lyrics.txt missing and no headline.txt — cannot repair")
        return False

    _miss("lyrics.txt missing — regenerating from headline...")
    if dry_run:
        return False

    from modules.lyrics_generator import generate_lyrics
    from modules.pollo_generator import parse_headline_file

    headline, _ = parse_headline_file(headline_file)
    hl_lines = headline_file.read_text(encoding="utf-8").splitlines()
    summary = hl_lines[2] if len(hl_lines) > 2 else ""

    try:
        lyrics = await generate_lyrics(
            headline=headline,
            summary=summary,
            api_key=llm["api_key"],
            model=llm["model"],
            fixed_hashtags=["#fyp", "#newsatire"],
            base_url=llm["base_url"],
            output_dir=run_dir,
        )
        body = (
            f"TITLE: {lyrics.title}\nSTYLE: {lyrics.style_prompt}\n\n"
            f"{lyrics.full_text}\n\n---\nCAPTION:\n{lyrics.caption}"
        )
        if lyrics.hook_caption:
            body += f"\n\n---\nHOOK_CAPTION:\n{lyrics.hook_caption}"
        lyrics_file.write_text(body, encoding="utf-8")
        _ok("Regenerated lyrics.txt")
        return True
    except Exception as e:
        _err(f"Lyrics generation failed: {e}")
        return False


async def _ensure_music(run_dir: Path, dry_run: bool) -> bool:
    mp3 = run_dir / "song.mp3"
    if mp3.exists():
        _ok("song.mp3 exists")
        return True

    lyrics_file = run_dir / "lyrics.txt"
    if not lyrics_file.exists():
        _err("song.mp3 missing and no lyrics.txt — skipping")
        return False

    _miss("song.mp3 missing — regenerating music...")
    if dry_run:
        return False

    from modules.music_generator import generate_music
    from config import SUNOAPI_KEY, SUNOAPI_BASE

    lyrics = _parse_lyrics_txt(lyrics_file)
    try:
        result = await generate_music(
            lyrics_text=lyrics.full_text,
            style_prompt=lyrics.style_prompt,
            title=lyrics.title,
            output_dir=run_dir,
            sunoapi_key=SUNOAPI_KEY,
            sunoapi_base=SUNOAPI_BASE,
        )
        _ok(f"Regenerated song.mp3 ({result.duration_seconds:.1f}s)")
        return True
    except Exception as e:
        _err(f"Music generation failed: {e}")
        return False


async def _ensure_scene_plan(run_dir: Path, llm: dict, dry_run: bool) -> bool:
    if (run_dir / "scene_plan.json").exists():
        _ok("scene_plan.json exists")
        return True

    lyrics_file = run_dir / "lyrics.txt"
    mp3 = run_dir / "song.mp3"
    if not lyrics_file.exists() or not mp3.exists():
        _err("scene_plan.json missing — needs lyrics.txt + song.mp3 first")
        return False

    _miss("scene_plan.json missing — regenerating scene plan...")
    if dry_run:
        return False

    from modules.scene_planner import plan_scenes
    from modules.pollo_generator import parse_headline_file

    headline_file = run_dir / "headline.txt"
    headline, image_url = parse_headline_file(headline_file) if headline_file.exists() else ("", None)
    hl_lines = headline_file.read_text(encoding="utf-8").splitlines() if headline_file.exists() else []
    summary = hl_lines[2] if len(hl_lines) > 2 else ""

    lyrics = _parse_lyrics_txt(lyrics_file)
    duration = _audio_duration(mp3)

    try:
        plan = await plan_scenes(
            headline=headline,
            summary=summary,
            lyrics=lyrics,
            song_duration=duration,
            image_url=image_url,
            api_key=llm["api_key"],
            model=llm["model"],
            base_url=llm["base_url"],
            output_dir=run_dir,
            timed_lyrics_path=run_dir / "timed_lyrics.json",
        )
        plan_data = [
            {"clip_index": s.clip_index, "prompt": s.prompt,
             "is_image_to_video": s.is_image_to_video, "visual_action": s.visual_action,
             "alt_prompts": s.alt_prompts or [],
             "prompt_variants": s.prompt_variants or []}
            for s in plan.scenes
        ]
        (run_dir / "scene_plan.json").write_text(
            json.dumps(plan_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _ok(f"Regenerated scene_plan.json ({len(plan.scenes)} scenes)")
        return True
    except Exception as e:
        _err(f"Scene planning failed: {e}")
        return False


async def _ensure_clips(run_dir: Path, pollo_key: str, video_model: str, dry_run: bool) -> bool:
    scenes = _load_scene_plan(run_dir)
    if scenes is None:
        _err("No scene_plan.json — cannot check clips")
        return False

    expected = {s["clip_index"] for s in scenes}
    existing = {int(p.stem.split("_")[1]) for p in run_dir.glob("clip_*.mp4")}
    missing = sorted(expected - existing)

    if not missing:
        _ok(f"All {len(expected)} clips present")
        return True

    _miss(f"Missing clips: {missing}")
    if dry_run:
        return False

    from modules.scene_planner import SceneDescription, ScenePlan, _build_prompt_variants
    from modules.pollo_generator import generate_clips_from_plan, parse_headline_file

    headline_file = run_dir / "headline.txt"
    image_url = None
    if headline_file.exists():
        _, image_url = parse_headline_file(headline_file)

    missing_set = set(missing)
    repair_scenes = []
    for s in scenes:
        if s["clip_index"] not in missing_set:
            continue
        prompt = s.get("prompt", "")
        visual_action = s.get("visual_action", "")
        alt_prompts = s.get("alt_prompts") or []
        variants = s.get("prompt_variants") or _build_prompt_variants(prompt, visual_action, alt_prompts)
        repair_scenes.append(SceneDescription(
            clip_index=s["clip_index"],
            prompt=prompt,
            is_image_to_video=s.get("is_image_to_video", False),
            visual_action=visual_action,
            alt_prompts=alt_prompts,
            prompt_variants=variants,
        ))

    try:
        await generate_clips_from_plan(
            plan=ScenePlan(scenes=repair_scenes),
            output_dir=run_dir,
            pollo_api_key=pollo_key,
            image_url=image_url,
            run_dir=run_dir,
            model=video_model,
        )
        _ok(f"Generated clips: {missing}")
        return True
    except Exception as e:
        err_msg = str(e)
        if "variants blocked by content filter" in err_msg:
            sentinel = run_dir / "CONTENT_BLOCKED"
            sentinel.write_text(
                f"All prompt variants blocked by Pollo content filter.\n\nError: {err_msg}\n",
                encoding="utf-8",
            )
            _err(f"Content filter exhausted — wrote CONTENT_BLOCKED sentinel. This run will be skipped on future repair attempts.")
        else:
            _err(f"Clip generation failed: {e}")
        return False


async def _ensure_final(run_dir: Path, dry_run: bool) -> bool:
    final = run_dir / "final.mp4"
    if final.exists():
        _ok("final.mp4 exists")
        return True

    # Check all expected clips are present
    scenes = _load_scene_plan(run_dir)
    if scenes is None:
        _err("final.mp4 missing — no scene_plan.json to know clip count")
        return False

    expected = {s["clip_index"] for s in scenes}
    existing = {int(p.stem.split("_")[1]) for p in run_dir.glob("clip_*.mp4")}
    if expected != existing:
        missing = sorted(expected - existing)
        _miss(f"final.mp4 waiting — clips still missing: {missing}")
        return False

    _miss("final.mp4 missing — assembling...")
    if dry_run:
        return False

    from modules.video_assembler import assemble_video
    from modules.clip_generator import parse_lyrics_file
    from modules.lyrics_generator import LyricSection

    mp3 = run_dir / "song.mp3"
    clip_paths = sorted(run_dir.glob("clip_*.mp4"), key=lambda p: int(p.stem.split("_")[1]))
    duration = _audio_duration(mp3)
    section_dur = duration / len(clip_paths)

    lyrics_file = run_dir / "lyrics.txt"
    if lyrics_file.exists():
        _, sections = parse_lyrics_file(lyrics_file)
    else:
        sections = [LyricSection(label=str(i), lines=[""]) for i in range(len(clip_paths))]

    try:
        await assemble_video(
            clip_paths=clip_paths,
            sections=sections,
            section_duration=section_dur,
            audio_path=mp3,
            output_path=final,
        )
        _ok(f"Assembled final.mp4 ({duration:.1f}s, {len(clip_paths)} clips)")
        return True
    except Exception as e:
        _err(f"Assembly failed: {e}")
        return False


async def _ensure_captioned(run_dir: Path, dry_run: bool) -> bool:
    captioned = run_dir / "final_captioned.mp4"
    if captioned.exists():
        _ok("final_captioned.mp4 exists")
        return True

    final = run_dir / "final.mp4"
    if not final.exists():
        _miss("final_captioned.mp4 waiting — final.mp4 not ready yet")
        return False

    _miss("final_captioned.mp4 missing — captioning...")
    if dry_run:
        return False

    from modules.captioner import caption_video

    try:
        await caption_video(
            video_path=final,
            audio_path=run_dir / "song.mp3",
            lyrics_path=run_dir / "lyrics.txt",
            output_path=captioned,
            karaoke=False,
        )
        _ok("Captioned → final_captioned.mp4")
        return True
    except Exception as e:
        _err(f"Captioning failed: {e}")
        return False


async def _ensure_tiktok(run_dir: Path, dry_run: bool) -> bool:
    tiktok = run_dir / "final_tiktok.mp4"
    if tiktok.exists():
        _ok("final_tiktok.mp4 exists")
        return True

    captioned = run_dir / "final_captioned.mp4"
    if not captioned.exists():
        _miss("final_tiktok.mp4 waiting — final_captioned.mp4 not ready yet")
        return False

    hook = _load_hook_caption(run_dir / "lyrics.txt")
    if not hook:
        _skip("final_tiktok.mp4 — no HOOK_CAPTION in lyrics.txt, skipping")
        return True  # not a failure — hook is optional

    _miss("final_tiktok.mp4 missing — burning hook caption...")
    if dry_run:
        return False

    from modules.captioner import burn_hook_caption

    try:
        await burn_hook_caption(
            video_path=captioned,
            output_path=tiktok,
            hook_caption_text=hook,
        )
        _ok("Hook burned → final_tiktok.mp4")
        return True
    except Exception as e:
        _err(f"Hook burn failed: {e}")
        return False


# ── orchestrate one run ────────────────────────────────────────────────────────

async def repair_run(
    run_dir: Path,
    llm: dict,
    pollo_key: str,
    video_model: str,
    dry_run: bool,
) -> None:
    print(f"\n{'='*60}")
    print(f"Run: {run_dir.name}")
    print(f"{'='*60}")

    sentinel = run_dir / "CONTENT_BLOCKED"
    if sentinel.exists():
        _skip("Skipping — CONTENT_BLOCKED sentinel present (all Pollo variants were blocked by content filter)")
        return

    # Each step returns True only when its output is confirmed present.
    # We continue regardless so we can report the full status in dry-run mode.
    ok_lyrics    = await _ensure_lyrics(run_dir, llm, dry_run)
    ok_music     = await _ensure_music(run_dir, dry_run)
    ok_plan      = await _ensure_scene_plan(run_dir, llm, dry_run)
    ok_clips     = await _ensure_clips(run_dir, pollo_key, video_model, dry_run)
    ok_final     = await _ensure_final(run_dir, dry_run)
    ok_captioned = await _ensure_captioned(run_dir, dry_run)
    ok_tiktok    = await _ensure_tiktok(run_dir, dry_run)


# ── main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Repair / resume incomplete pipeline runs")
    parser.add_argument("--date", type=str, default=None,
                        help="Date to check (default: today, e.g. 2026-04-19)")
    parser.add_argument("--run", type=str, default=None,
                        help="Filter to a specific run (partial name match, e.g. 01-strait)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what is missing without generating anything")
    parser.add_argument("--provider", type=str, default="grok", choices=["grok", "ollama"],
                        help="LLM provider for lyrics + scene planning (default: grok)")
    parser.add_argument("--video-model", type=str, default="veo3-1",
                        help="Pollo video model for clip regeneration (default: veo3-1)")
    args = parser.parse_args()

    date_str  = args.date or date.today().isoformat()
    llm       = _llm_config(args.provider)
    pollo_key = os.getenv("POLLO_API_KEY", "")

    if not pollo_key and not args.dry_run:
        print("Warning: POLLO_API_KEY not set — clip generation will fail")

    run_dirs = _find_run_dirs(date_str, args.run)
    if not run_dirs:
        print(f"No run directories found under output/{date_str}/")
        return

    label = "DRY RUN — " if args.dry_run else ""
    print(f"{label}Date: {date_str}  |  Provider: {args.provider}  |  Runs: {len(run_dirs)}")

    for run_dir in run_dirs:
        await repair_run(
            run_dir=run_dir,
            llm=llm,
            pollo_key=pollo_key,
            video_model=args.video_model,
            dry_run=args.dry_run,
        )

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
