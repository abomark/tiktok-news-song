# CurrentNoise 🎵

Fully automated pipeline that picks the top news stories of the day, writes satirical song lyrics, generates a Suno track, plans a scene-by-scene music video, renders it with Pollo AI (Veo 3.1), burns in karaoke-style captions, and posts to TikTok — all without human intervention.

Brand: **@currentnoise**

---

## Architecture overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         SCHEDULER (scheduler.py)                     │
│  Hourly: fetch + score + classify (warms caches)                     │
│  09:00:  full pipeline run → TikTok post                             │
└──────────────────────────────────────────────────────────────────────┘

NewsAPI / Google Trends RSS
        │
        ▼
┌─────────────────┐
│  news_fetcher   │  Fetches up to 10 candidate headlines
└────────┬────────┘  logs/news_candidates.jsonl
         │
         ▼
┌─────────────────┐
│  social_scorer  │  Reddit + Hacker News + Google Trends (parallel)
└────────┬────────┘  logs/social_scores.jsonl
         │
         ▼
┌─────────────────┐
│story_classifier │  10-factor VPI scoring — dual gemma3 + Grok (averaged)
└────────┬────────┘  logs/story_classifications.jsonl
         │
         ▼
┌─────────────────┐
│ story_selector  │  Blends social 40% + VPI 60% → flags all ≥ threshold
└────────┬────────┘  logs/flagged_stories.jsonl
         │                logs/selection_decisions.jsonl
         ▼                  (iterates top-N flagged stories)
┌─────────────────┐
│lyrics_generator │  Hook + verse + chorus via Ollama or Grok (default: Grok)
└────────┬────────┘  output/<date>/<run>/lyrics.txt
         │
         ▼
┌─────────────────┐
│lyrics_classifier│  Scores lyrics on 16 factors (Hook, Cultural, Creator, Risk)
└────────┬────────┘  logs/lyrics_classifications.jsonl
         │
         ▼
┌─────────────────┐
│ music_generator │  sunoapi.org (V5.5) → song.mp3 + timed_lyrics.json
└────────┬────────┘  output/<date>/<run>/song.mp3
         │
         ▼
┌─────────────────────────────────────────┐
│           video_generator               │
│  ┌──────────────────────────────────┐   │
│  │ scene_planner (always Grok)      │   │  LLM plans 5s scenes; first is
│  │                                  │   │  image-to-video from news photo
│  └──────────────┬───────────────────┘   │
│  ┌──────────────▼───────────────────┐   │
│  │ pollo_generator (Veo 3.1)        │   │  One Pollo clip per scene
│  └──────────────┬───────────────────┘   │
│  ┌──────────────▼───────────────────┐   │
│  │ video_assembler                  │   │  ffmpeg concat + audio + watermark
│  └──────────────┬───────────────────┘   │
│  ┌──────────────▼───────────────────┐   │
│  │ captioner (ASS karaoke)          │   │  Suno word timings → red-on-white
│  └──────────────────────────────────┘   │
└────────┬────────────────────────────────┘
         │  output/<date>/<run>/final_captioned.mp4
         ▼
┌─────────────────┐
│tiktok_publisher │  Chunked upload via TikTok API v2
└─────────────────┘

        │
        ▼ (parallel write path)
┌─────────────────┐
│    Supabase     │  Cloud-queryable copy of all JSONL logs
└─────────────────┘  JSONL files remain local source of truth

        │
        ▼
┌─────────────────┐
│  app.py         │  Streamlit dashboard — Supabase or JSONL fallback
└─────────────────┘
```

---

## Pipeline steps

### 1. News fetching — `modules/news_fetcher.py`
- Calls NewsAPI `/v2/top-headlines` (US, up to 10 articles)
- Falls back to Google Trends RSS if NewsAPI unavailable
- Upserts candidates to Supabase `news_candidates`

### 2. Social scoring — `modules/social_scorer.py`
- Scores each story in parallel across three signals:
  - **Reddit** — r/news + r/worldnews + r/politics; sums upvotes + comments
  - **Hacker News** — Algolia search API; sums points + comments
  - **Google Trends** — pytrends 1-day interest for top 3 headline keywords
- Each signal normalised 0–100, blended: `reddit×0.40 + hn×0.30 + trends×0.30`

### 3. Story classification — `modules/story_classifier.py`
Scores each candidate on **10 Viral Potential Index (VPI) factors** (1–10 each):

| Factor | What it measures |
|---|---|
| Absurdity | How inherently ridiculous the story is |
| Character Punchability | Is there a clear villain to mock? |
| Cultural Reach | How many people will get the reference? |
| Emotional Heat | Anger, outrage, or passion |
| Memeability | Can it become a repeatable meme format? |
| Musical Fit | Does it lend itself to a song naturally? |
| Timestamp Sensitivity | Will it still be relevant in 24 h? |
| Moral Clarity | Is there a clear good/bad side? |
| Visual Potential | Can you picture a fun music video? |
| Safe Harbor | Low legal/brand risk for satire |

- **Dual-model classification** via `classify_story_dual`: runs both gemma3 (local Ollama) and Grok in parallel, averages factor scores for higher robustness
- VPI = average of all 10 factor scores
- Results cached per `(headline, date, classifier)` — re-runs skip already-classified stories

### 4. Story flagging — `modules/story_selector.py`
- Accepts pre-computed social scores + VPI classifications — no LLM calls
- Normalises both to 0–1 (min-max within the current batch)
- Combined score: `social×0.40 + VPI×0.60`
- Flags **every** story with `combined_score ≥ 0.5` (configurable threshold)
- Guarantees at least one flagged story (promotes top story if nothing clears threshold)
- Pipeline iterates over the top-N flagged stories (default 3, via `--max-stories`)

### 5. Lyrics generation — `modules/lyrics_generator.py`
- Calls an OpenAI-compatible endpoint (**default: Grok**; `--provider ollama` to use local gemma3)
- Structure: 1 hook (3 s) + 1 verse (4–6 lines) + 1 chorus (4 lines) ≈ 30–45 s
- Also generates: Suno style prompt, TikTok caption, 1–2 topic hashtags
- Appends `[END]` tag so Suno stops at ~30–40 s instead of extending the track
- Uses the classifier's **satirical angle** as part of the prompt
- Prompts overridable via `assets/lyrics_system_prompt.txt` / `assets/lyrics_user_prompt.txt`

### 6. Lyrics classification — `modules/lyrics_classifier.py`
Scores generated lyrics on 16 factors across 4 categories, computing a **Lyrics Virality Index (LVI)** 0–10:
- **Hook Mechanics** — hook strength, hook position, earworm factor, singability
- **Cultural Payload** — topicality, recognition trigger, controversy level, satire type, in-group signal
- **Creator Bait** — visual hook potential, meme format fit, quotability, participation hook
- **Platform Risk** — takedown risk, algorithm risk, shadowban words
- Prompt at `assets/lyrics_classifier_prompt.md`
- Results appended to `logs/lyrics_classifications.jsonl`

### 7. Music generation — `modules/music_generator.py`
- Uses **sunoapi.org** (hosted wrapper, model `V5_5`) — the old self-hosted Docker setup was replaced
- Polls for result, downloads MP3, and also fetches word-level `timed_lyrics.json` for karaoke captions
- `ffprobe` reads actual duration

### 8. Video generation — `modules/video_generator.py`
Four sub-steps:

**8a. Scene planning** (`scene_planner.py`) — **always uses Grok**. Produces `scene_plan.json` with one scene per 5s Pollo clip; first scene is image-to-video (from the news article's hero image), subsequent scenes are text-to-video with an evolving visual story and a designated "viral moment"

**8b. Clip generation** (`pollo_generator.py`) — calls Pollo AI (default model: **Veo 3.1, 6s, 720p, text-to-video**). Supports alternate models via `--video-model` (seedance-pro-1-5, veo3-1-fast, veo3-fast, kling-v3, pollo-v2-0, pollo-v1-6)

**8c. Assembly** (`video_assembler.py`) — ffmpeg concatenates clips, mixes in the MP3, adds `@currentnoise` watermark, outputs `final.mp4`

**8d. Karaoke captions** (`captioner.py`) — prefers Suno's `timed_lyrics.json` for exact word timing; falls back to local Whisper if missing. Renders **ASS karaoke subtitles**: white base sentence with the currently sung word highlighted in red and slightly larger. Per-word display capped at 1.5s. Optional `--no-karaoke` mode shows only the sung word, solo and centered. Output: `final_captioned.mp4` (YouTube-ready)

**8e. TikTok hook caption** (`captioner.burn_hook_caption`) — overlays a 2-line **white-background title card** on the first 2 seconds of the video. The text is generated by the lyrics LLM as `hook_caption` (line 1 = topic/keyword label to seed TikTok's algorithm; line 2 = curiosity teaser tied to the satirical angle). This step runs only if `hook_caption` is present and produces **`final_tiktok.mp4`** — `final_captioned.mp4` stays untouched for the future YouTube upload

### 9. TikTok publishing — `modules/tiktok_publisher.py`
- Refreshes OAuth token from `TIKTOK_REFRESH_TOKEN`
- Chunked upload (10 MB chunks) via `/v2/post/publish/video/init/`
- Public post; duet, comment, and stitch enabled

---

## Resume / idempotency

The pipeline is fully resumable — re-running the same day **never regenerates paid artefacts** that already exist:

| File present in run dir | Behaviour |
|---|---|
| `final_tiktok.mp4` | Skip story entirely |
| `final_captioned.mp4` | Skip karaoke burn, only re-run the TikTok hook burn |
| `final.mp4` | Skip assembly, run captioning |
| `clip_*.mp4` | Skip clip generation (no Pollo spend) |
| `song.mp3` | Skip music generation (no Suno spend) |
| `lyrics.txt` | Reuse folder + lyrics (no LLM spend) |

This means a failed captioning step can be re-run cheaply; a crashed Pollo run resumes where it left off on the next invocation.

---

## Data stores

| Store | Purpose |
|---|---|
| `logs/*.jsonl` | Append-only local log — source of truth |
| **Supabase** | Cloud-queryable mirror; `upsert` on `(headline, date)` to survive re-runs |
| `output/<date>/<run>/` | All media artefacts for that run |

### Log files

| File | Contents |
|---|---|
| `logs/news_candidates.jsonl` | All fetched headlines + metadata |
| `logs/social_scores.jsonl` | Reddit / HN / Trends scores per headline |
| `logs/story_classifications.jsonl` | VPI factor scores per headline (+ classifier label) |
| `logs/lyrics_classifications.jsonl` | LVI factor scores per generated lyric set |
| `logs/flagged_stories.jsonl` | Stories that cleared the combined threshold |
| `logs/selection_decisions.jsonl` | Full ranked candidate list per run |
| `logs/api_calls.jsonl` | Every LLM + Suno + Pollo API call (input + output) |
| `logs/pipeline.log` | Full pipeline stdout log |

---

## Output layout

```
output/
└── 2026-04-13/
    └── 01-mcilroys-meltdown/
        ├── headline.txt          # Headline + summary + source URL + image URL
        ├── lyrics.txt            # Title + lyrics + TikTok caption
        ├── scene_plan.json       # LLM-planned scenes (prompts per clip)
        ├── song.mp3              # Suno-generated audio
        ├── timed_lyrics.json     # Suno word-level timings (preferred over Whisper)
        ├── clip_0.mp4            # Pollo clip — scene 0 (image-to-video)
        ├── clip_1.mp4            # Pollo clip — scene 1
        ├── clip_N.mp4            # Pollo clip — scene N
        ├── final.mp4             # Assembled video with watermark, no captions
        ├── lyrics.ass             # Generated ASS karaoke subtitle
        ├── final_captioned.mp4    # Karaoke-captioned (YouTube-ready)
        ├── tiktok_hook.ass        # 2-line white-bg hook caption (first 2s)
        └── final_tiktok.mp4       # Posted to TikTok (karaoke + hook caption)
```

---

## Quick start

```bash
# 1. Copy and fill in keys
cp .env.example .env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Ollama with gemma3 (used for dual classification)
ollama serve
ollama pull gemma3

# 4. Test news + lyrics only (uses Grok by default — paid)
python pipeline.py --lyrics-only

# 5. Test news + lyrics with free local LLM only
python pipeline.py --lyrics-only --provider ollama

# 6. Full dry run (no TikTok post)
python pipeline.py --dry-run

# 7. Full run
python pipeline.py

# 8. Generate lyrics + music for all of today's flagged stories
python pipeline.py --all-flagged-music

# 9. Dashboard
streamlit run app.py
```

---

## Run modes

| Command | Runs |
|---|---|
| `python pipeline.py` | Full pipeline → TikTok post |
| `python pipeline.py --dry-run` | All steps except post |
| `python pipeline.py --dry-run-full` | News + lyrics only |
| `python pipeline.py --lyrics-only` | News + lyrics, then stop |
| `python pipeline.py --all-flagged` | Lyrics for all of today's flagged stories (skips already done) |
| `python pipeline.py --all-flagged-music` | Lyrics + music for all of today's flagged stories |
| `python pipeline.py --max-stories N` | Process top N flagged stories this run (default 3) |
| `python pipeline.py --provider ollama` | Use local Ollama gemma3 instead of Grok for lyrics |
| `python pipeline.py --video-model veo3-1-fast` | Pick Pollo video model (default: `veo3-1`) |
| `python pipeline.py --headline "..." --summary "..."` | Skip news fetch |
| `python scheduler.py` | Hourly background job + daily 09:00 post |
| `streamlit run app.py` | Dashboard |
| `python db/migrate_logs.py` | One-time JSONL → Supabase migration |

### Per-module CLIs

Most modules can be invoked standalone for debugging and resume:

```bash
python -m modules.news_fetcher
python -m modules.music_generator
python -m modules.scene_planner --date 2026-04-18 --run 01
python -m modules.pollo_generator --date 2026-04-18 --run 01 --model veo3-1
python -m modules.video_assembler --date 2026-04-18 --run 01
python -m modules.captioner --date 2026-04-18 --run 01                  # karaoke (default)
python -m modules.captioner --date 2026-04-18 --run 01 --no-karaoke     # solo words
python -m modules.video_generator --reuse --date 2026-04-18 --run 01
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `NEWS_API_KEY` | Recommended | NewsAPI key (falls back to Google Trends RSS) |
| `OPENAI_API_KEY` | Yes | Set to `"ollama"` if not using OpenAI directly — used as Ollama passthrough |
| `SUNOAPI_KEY` | Yes (music) | Key for sunoapi.org hosted API |
| `SUNOAPI_BASE` | No | Override base URL (default: `https://api.sunoapi.org`) |
| `POLLO_API_KEY` | Yes (video) | Pollo AI key (https://pollo.ai/api-platform) |
| `XAI_API_KEY` | Yes (default) | Grok / xAI key — default provider for lyrics and scene planning |
| `TIKTOK_CLIENT_KEY` | Yes (post) | TikTok developer app key |
| `TIKTOK_CLIENT_SECRET` | Yes (post) | TikTok developer app secret |
| `TIKTOK_REFRESH_TOKEN` | Yes (post) | Long-lived TikTok OAuth refresh token |
| `OLLAMA_BASE_URL` | No | Ollama API URL (default: `http://localhost:11434/v1`) |
| `OLLAMA_MODEL` | No | Model name (default: `gemma3`) |
| `SUPABASE_URL` | Optional | Enables Supabase sync in dashboard + migration |
| `SUPABASE_KEY` | Optional | Supabase service role key |
| `REDDIT_CLIENT_ID` | Optional | Reddit API credentials (falls back to public API) |
| `REDDIT_CLIENT_SECRET` | Optional | Reddit API credentials |
| `ANTHROPIC_API_KEY` | Unused | Declared but not wired into the pipeline yet |
| `RUNWAYML_API_SECRET` | Deprecated | Kept for reference only; video now uses Pollo |

---

## External services

| Service | Used for |
|---|---|
| [NewsAPI](https://newsapi.org) | Headline candidates |
| Reddit (public API / PRAW) | Social scoring |
| Hacker News (Algolia) | Social scoring |
| Google Trends (pytrends) | Social scoring |
| Ollama + gemma3 (local) | Dual-model classification |
| Grok / xAI | Classification (one half of dual), lyrics, scene planning |
| [sunoapi.org](https://sunoapi.org) | Music generation (Suno V5.5) |
| [Pollo AI](https://pollo.ai) | Video clip generation (default: Veo 3.1) |
| TikTok API v2 | Publishing |
| Supabase (optional) | Cloud log storage + dashboard queries |

---

## Key files

| File | Role |
|---|---|
| `pipeline.py` | Main orchestrator |
| `scheduler.py` | Cron-style runner (hourly scoring + daily post) |
| `app.py` | Streamlit dashboard |
| `config.py` | All config and env-var loading |
| `modules/` | One file per pipeline step |
| `assets/` | LLM prompts + fonts |
| `logs/` | JSONL event logs |
| `output/` | Per-run media artefacts |
| `db/schema.sql` | Supabase table definitions |
| `db/migrate_logs.py` | One-time JSONL → Supabase migration |
| `db/client.py` | Shared Supabase client |

---

## Dashboard pages

| Page | Shows |
|---|---|
| News Feed | All fetched headlines with social + VPI + combined scores |
| Flagged Stories | Stories that cleared the threshold, with score breakdown |
| Selection Decisions | Full ranked candidate list per run |
| Story Classifications | Per-headline VPI factor breakdown |
| Lyrics Classifications | Per-lyric LVI breakdown (Hook / Cultural / Creator / Risk) |
| Credits | API cost tracking (Pollo, Suno, Grok) |
| API Logs | Every LLM + Suno + Pollo call with inputs and outputs |
| Runs | Browse output folders by date |
| Run Detail | Inspect lyrics, audio, scene plan, and video for a specific run |
