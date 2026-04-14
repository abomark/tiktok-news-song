# CLAUDE.md — CurrentNoise project guide

## What this project does

Fully automated pipeline: picks the most viral news story of the day, writes satirical song lyrics, generates a song (Suno), generates a music video (Runway), burns in captions (Whisper + ffmpeg), and posts to TikTok.

Brand name: **@currentnoise**

---

## Quick orientation

| File | Role |
|---|---|
| `pipeline.py` | Main orchestrator — run this |
| `scheduler.py` | Runs pipeline daily at 09:00 + hourly background scoring |
| `app.py` | Streamlit dashboard (`streamlit run app.py`) |
| `config.py` | All config and env-var loading |
| `modules/` | One file per pipeline step |
| `assets/` | Prompts + fonts |
| `logs/` | JSONL logs from every run |
| `output/` | Per-run artefacts (mp3, mp4, lyrics) |

See **PIPELINE.md** for the full architecture and step-by-step breakdown.

---

## Environment setup

1. Copy `.env.example` → `.env` and fill in keys.
2. `OPENAI_API_KEY` must be set to a non-empty string even when using Ollama (it's the passthrough key). Set it to `"ollama"` if you have no real key.
3. `SUNO_COOKIE` must be set even for `--lyrics-only` runs — this is a known bug in `config.py:16–17`. Workaround: set a dummy value.
4. Ollama must be running locally (`ollama serve`) with the `gemma3` model pulled.
5. Suno-api Docker container must be running on port 3000 for music generation.
6. `ffmpeg` and `ffprobe` must be on PATH (Windows: add to system PATH after installing).

---

## Common commands

```bash
# Test news + lyrics only (cheapest, no external paid APIs)
python pipeline.py --lyrics-only

# Test up to video assembly but don't post
python pipeline.py --dry-run

# Full run including TikTok post
python pipeline.py

# Use Grok (xAI) for lyrics instead of local Ollama
python pipeline.py --provider grok --lyrics-only

# Inject a custom story (skip news fetch)
python pipeline.py --lyrics-only --headline "..." --summary "..."

# Background scoring daemon + daily post
python scheduler.py

# Dashboard
streamlit run app.py
```

---

## Known bugs to fix before production

1. **`config.py:16–17`** — `os.environ["OPENAI_API_KEY"]` and `os.environ["SUNO_COOKIE"]` use hard-crash `environ[]` instead of `.getenv()`. This breaks `--lyrics-only` on a clean env. Fix: change both to `os.getenv(...)`.

2. **`pipeline.py:179–190`** — `audio` and `video` are undefined when `--dry-run-full` skips generation but code after the if-block references them. Fix: add early return after the dry-run-full music/video skip, or initialise to `None` and guard.

3. **`story_selector.py:209`** — `winner.classification.vpi` crashes when the winner has no classification (e.g., all Ollama calls timed out). Fix: guard with `if winner.classification else 0.0`.

4. **`pipeline.py:19`** — `ANTHROPIC_API_KEY` is imported but never used anywhere in the pipeline. Remove the import, and consider whether the Anthropic SDK (`anthropic` in `requirements.txt`) is actually needed yet.

---

## Architecture decisions

- **LLM for classification**: gemma3 via Ollama (free, local). Can swap via `OLLAMA_MODEL` env var.
- **LLM for lyrics**: Ollama default; `--provider grok` switches to `grok-3-fast` via xAI API.
- **Music**: Suno via self-hosted Docker wrapper (`gcui-art/suno-api`). Needs a valid Suno cookie.
- **Video clips**: Runway Gen-3 Alpha Turbo via `runwayml` SDK.
- **Captions**: Whisper `base` model transcribes the generated MP3, then ffmpeg burns word-synced subtitles.
- **Social scoring**: Reddit (PRAW-free public API) + Hacker News (Algolia) + Google Trends (pytrends).
- **Story scoring blend**: social 40% + VPI 60%.

---

## Output layout

```
output/2026-04-13/01-some-song-title/
    headline.txt
    lyrics.txt
    song.mp3
    clip_0.mp4 … clip_N.mp4
    final.mp4
    final_captioned.mp4   ← this is what gets posted
    timed_lyrics.json

logs/
    pipeline.log
    social_scores.jsonl
    story_classifications.jsonl
    selection_decisions.jsonl
    news_candidates.jsonl
    api_calls.jsonl
```

---

## Codebase conventions

- All modules expose a single async `generate_*/fetch_*/score_*/select_*` entry-point function.
- Dataclasses (not Pydantic models) used for internal data transfer (`NewsStory`, `Lyrics`, `AudioResult`, `VideoResult`, `PublishResult`).
- Logs written as JSONL (one JSON object per line) to `logs/`.
- All external HTTP done with `httpx` (async).
- `tenacity` used for retries on Suno submit.
- ffmpeg called via `subprocess` (not `ffmpeg-python` wrapper) for assembler and captioner.
