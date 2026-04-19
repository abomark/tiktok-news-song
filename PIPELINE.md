# CurrentNoise — Pipeline Overview

End-to-end flow from news API to TikTok post.

---

## Architecture

```
NewsAPI / Google Trends RSS
        │
        ▼
┌─────────────────┐
│  news_fetcher   │  Fetches up to 10 candidate headlines (incl. hero image URL)
└────────┬────────┘  Writes: logs/news_candidates.jsonl, Supabase upsert
         │  list[NewsStory]
         ▼
┌─────────────────┐
│  social_scorer  │  Scores each story via Reddit + HN + Google Trends (parallel)
└────────┬────────┘  Writes: logs/social_scores.jsonl
         │  list[ScoredStory]
         ▼
┌─────────────────┐
│story_classifier │  Scores on 10 VPI factors — dual gemma3 + Grok (averaged)
└────────┬────────┘  Writes: logs/story_classifications.jsonl
         │  list[StoryClassification | None]
         ▼
┌─────────────────┐
│ story_selector  │  Blends social (40%) + VPI (60%) → flags stories ≥ threshold
└────────┬────────┘  Writes: logs/flagged_stories.jsonl, logs/selection_decisions.jsonl
         │  list[CandidateResult]  — pipeline iterates top-N (default 3)
         ▼
┌─────────────────┐
│lyrics_generator │  Generates hook + verse + chorus + caption via Grok (default)
└────────┬────────┘  Writes: output/<date>/<run>/lyrics.txt
         │  Lyrics
         ▼
┌─────────────────┐
│lyrics_classifier│  Scores lyrics across 4 categories → LVI
└────────┬────────┘  Writes: logs/lyrics_classifications.jsonl
         │  LyricsClassification
         ▼
┌─────────────────┐
│ music_generator │  sunoapi.org hosted Suno V5.5 — downloads mp3 + timed_lyrics
└────────┬────────┘  Output: song.mp3, timed_lyrics.json
         │  AudioResult
         ▼
┌─────────────────┐
│ video_generator │  Orchestrates four sub-steps (see below)
└────────┬────────┘  Output: final_captioned.mp4
         │  VideoResult
         ▼
┌─────────────────┐
│tiktok_publisher │  Refreshes OAuth token, chunks and uploads via TikTok API v2
└─────────────────┘
```

---

## Steps in detail

### Step 1 — News fetching (`modules/news_fetcher.py`)

- Calls NewsAPI `/v2/top-headlines` → up to 10 articles
- Falls back to Google Trends RSS if NewsAPI fails or has no key
- Captures hero image URL (`image_url`) from the article when available — used later as the first frame for image-to-video
- Upserts to Supabase `news_candidates` on `(headline, date)`

### Step 2 — Social scoring (`modules/social_scorer.py`)

- Scores each candidate in parallel across three signals:
  - **Reddit** — searches r/news, r/worldnews, r/politics; sums upvotes + comments
  - **Hacker News** — searches via Algolia; sums points + comments
  - **Google Trends** — pytrends 1-day interest, top 3 keywords from headline
- Each signal normalised 0–100, then blended: `reddit×0.40 + hn×0.30 + trends×0.30`
- Writes social scores to `logs/social_scores.jsonl` (join key: `headline + date`)
- Upserts to Supabase `social_scores`

### Step 3 — Story classification (`modules/story_classifier.py`)

- Scores each candidate on **10 VPI factors** (1–10 each):

  | Factor | What it measures |
  |---|---|
  | Absurdity | How inherently ridiculous the story is |
  | Character Punchability | Is there a clear villain to mock? |
  | Cultural Reach | How many people will get the reference? |
  | Emotional Heat | Anger, outrage, or passion in the story |
  | Memeability | Can it become a repeatable meme format? |
  | Musical Fit | Does it lend itself to a song naturally? |
  | Timestamp Sensitivity | Will it still be relevant in 24h? |
  | Moral Clarity | Is there a clear good/bad side? |
  | Visual Potential | Can you picture a fun music video? |
  | Safe Harbor | Low legal/brand risk for satire |

- **Dual classification** via `classify_story_dual`: runs `gemma3` (local Ollama) and `grok-3-fast` (xAI) in parallel and averages factor scores
- VPI = average of all 10 scores; merged entry saved with classifier label `"dual"`
- Per-classifier rows also cached so individual results are reusable across runs
- Prompt loaded from `assets/classifier_prompt.md`
- Results cached by `(headline, date, classifier)` — already-classified stories are skipped
- Upserts to Supabase `story_classifications` on `(headline, date)` to survive re-runs

### Step 4 — Story flagging (`modules/story_selector.py`)

- Accepts pre-computed `list[ScoredStory]` + `list[StoryClassification | None]` — no LLM calls inside
- Normalises both score types to 0–1 (min-max across the current batch)
- Combined score: `social×0.40 + vpi×0.60`
- Falls back to pure social score if all VPI classifications failed
- **Flags every story** with `combined_score ≥ threshold` (default 0.5)
- Guarantees at least one flagged story (flags the top story if nothing clears threshold)
- Writes to `logs/flagged_stories.jsonl` + `logs/selection_decisions.jsonl`
- Returns `list[CandidateResult]` sorted by combined score descending
- Pipeline iterates `flagged[0..max_stories]` (default 3 top flagged stories per run)

### Step 5 — Lyrics generation (`modules/lyrics_generator.py`)

- Default provider: **Grok** (`grok-3-fast` via xAI); switch to local Ollama gemma3 with `--provider ollama`
- Structure: hook (1 line) + verse (4–6 lines) + chorus (4 lines)
- Generates Suno style prompt, TikTok caption, 1–2 topic hashtags, and a 2-line **TikTok hook caption** (`hook_caption`) used later for the opening title card on `final_tiktok.mp4`
- Fixed hashtags (`#newsatire #politicalsatire #aimusic`) appended
- Platform tag (`#fyp` / `#foryou`) alternates daily
- Appends `[END]` sentinel to lyrics body so Suno stops at the end of the written section (~30–40s) instead of extending to 2 minutes
- Prompts loadable from `assets/lyrics_system_prompt.txt` / `assets/lyrics_user_prompt.txt`
- Output folder: `output/<date>/<NN>-<slug>/`

### Step 6 — Lyrics classification (`modules/lyrics_classifier.py`)

Scores the generated lyrics on 16 factors across 4 categories. Produces a **Lyrics Virality Index (LVI)** (0–10):

| Category | Factors |
|---|---|
| Hook Mechanics | `hook_strength` (1-10), `hook_position`, `earworm_factor` (1-10), `singability` |
| Cultural Payload | `topicality`, `recognition_trigger`, `controversy_level` (1-10), `satire_type`, `ingroup_signal` |
| Creator Bait | `visual_hook_potential`, `meme_format_fit`, `quotability` (1-10), `participation_hook` (1-10) |
| Platform Risk | `takedown_risk`, `algorithm_risk`, `shadowban_words` |

Categorical factors are remapped to numeric scores (e.g. `takedown_risk: low/med/high → 10/5/1`), then averaged. Prompt at `assets/lyrics_classifier_prompt.md`; results in `logs/lyrics_classifications.jsonl`.

### Step 7 — Music generation (`modules/music_generator.py`)

- Uses **sunoapi.org** (hosted Suno wrapper) — replaced the self-hosted Docker setup
- Model: `V5_5`
- Submits lyrics + style prompt, polls task status, downloads MP3
- Also fetches Suno's word-level **timed lyrics** (saved as `timed_lyrics.json`), used by the captioner for exact word timing
- `ffprobe` reads actual duration
- Returns `AudioResult` (path, duration_seconds, title)

### Step 8 — Video generation (`modules/video_generator.py`)

Five sub-steps:

**8a. Scene planning (`modules/scene_planner.py`)**
- **Always uses Grok** (hardcoded in `pipeline.py`) — richer scene descriptions than local Ollama
- Given headline, summary, lyrics, and song duration, plans one scene per 5s clip
- First scene is **image-to-video** starting from the news article's hero image (if available); rest are text-to-video
- Prompts clips to tell a visual story arc (setup → escalation → viral twist → resolution); at least one designated "viral moment"
- Saves `scene_plan.json` for debugging

**8b. Clip generation (`modules/pollo_generator.py`)**
- Calls **Pollo AI** platform — default model `veo3-1` (Google Veo 3.1, 6s, 720p)
- Other supported models (via `--video-model`): `veo3-1-fast`, `veo3-fast`, `seedance-pro-1-5`, `kling-v3`, `pollo-v2-0`, `pollo-v1-6`
- Clip 0 uses image-to-video if `image_url` is set; clips 1..N are text-to-video
- Polls each task until complete, downloads `clip_0.mp4`, `clip_1.mp4`, ...
- Replaces the previous Runway Gen-3 implementation

**8c. Video assembly (`modules/video_assembler.py`)**
- Concatenates clips with ffmpeg, scales/crops each to 1080×1920 @ 30 fps
- Burns `@currentnoise` watermark via `drawtext`
- Mixes in the MP3 audio track and trims to audio duration
- Uses `asyncio.to_thread(subprocess.run, …)` to avoid Windows `ProactorEventLoop` subprocess hangs
- Output: `final.mp4`

**8d. Karaoke captions (`modules/captioner.py`)**
- Prefers Suno's `timed_lyrics.json` for exact word timing; falls back to local **Whisper** (default model: `base`) if absent
- Generates an **ASS karaoke subtitle** with two layers:
  - Layer 1 (`LyricsWhite`): full line in white; the currently sung word is made transparent (both fill and border) so it doesn't fight layer 2
  - Layer 2 (`LyricsRed`): same layout with only the active word visible — rendered red and slightly larger (inline `\fs` tag, ~1.12×)
- Words sharing identical timestamps are grouped into one event to avoid flicker
- Per-word display capped at 1.5s so trailing words don't hang on screen
- Optional `--no-karaoke` mode: drops the white layer; each sung word appears solo, larger, centered
- ffmpeg burns the ASS file into the video → `final_captioned.mp4` (this is the **YouTube-ready** version)

**8e. TikTok hook caption (`modules/captioner.burn_hook_caption`)**
- Renders a separate `tiktok_hook.ass` containing one Dialogue event for the first 2 seconds
- Style `HookCaption`: `BorderStyle=3` (opaque white box), black text, top-center alignment, font ~64px, two lines separated by `\N`
- Text comes from `Lyrics.hook_caption`, generated by the lyrics LLM with two goals: (1) supply a topic/keyword signal for TikTok's For-You algorithm, (2) give profile-page viewers a teaser so they click
- ffmpeg burns the hook ASS onto `final_captioned.mp4` → **`final_tiktok.mp4`** (the TikTok upload source). `final_captioned.mp4` stays untouched for YouTube.
- If `Lyrics.hook_caption` is empty (e.g. legacy resumed run), the hook step is skipped and `final_captioned.mp4` is used as-is; a programmatic fallback (`TITLE`\n`#topic`) is populated for resumed runs so TikTok always gets a hook card.

Returns `VideoResult` (path is `final_tiktok.mp4` when present, else `final_captioned.mp4`, else `final.mp4`)

### Step 9 — TikTok publishing (`modules/tiktok_publisher.py`)

- Refreshes OAuth access token from stored `TIKTOK_REFRESH_TOKEN`
- Initialises chunked upload via `/v2/post/publish/video/init/`
- Uploads in 10 MB chunks with `Content-Range` headers
- Post is public; duet, comment, and stitch enabled by default
- Returns `PublishResult` (publish_id)

---

## Resume / idempotency

`pipeline.py` detects existing artefacts and skips paid work:

| Present in run dir | Behaviour |
|---|---|
| `final_tiktok.mp4` | Skip whole story (step 9 only runs with a new output) |
| `final_captioned.mp4` | Skip karaoke burn, only re-run the 2s hook-caption burn |
| `final.mp4` | Skip assembly, run captioning (karaoke + hook) |
| `clip_*.mp4` | Skip clip generation (no Pollo spend) |
| `song.mp3` | Skip music generation (no Suno spend) |
| `lyrics.txt` | Reuse run dir and lyrics (no LLM spend) |

This is driven by two helpers in `pipeline.py`:
- `_find_run_dir_for(headline, day_dir)` locates the existing folder by exact or slugified headline match
- `_video_complete_for(headline, day_dir)` returns `True` only when `final_captioned.mp4` exists

---

## Run modes

| Command | What runs |
|---|---|
| `python pipeline.py` | Full pipeline → posts to TikTok |
| `python pipeline.py --dry-run` | All steps except TikTok post |
| `python pipeline.py --dry-run-full` | News + lyrics only (no Suno, Pollo, TikTok) |
| `python pipeline.py --lyrics-only` | News + lyrics, then stop |
| `python pipeline.py --all-flagged` | Lyrics for all of today's flagged stories (skips already done) |
| `python pipeline.py --all-flagged-music` | Lyrics + music for all of today's flagged stories |
| `python pipeline.py --max-stories N` | Process top N flagged stories (default 3) |
| `python pipeline.py --provider ollama` | Use local Ollama gemma3 for lyrics (default is Grok) |
| `python pipeline.py --video-model veo3-1-fast` | Pick Pollo video model (default: `veo3-1`) |
| `python pipeline.py --headline "..." --summary "..."` | Skip news fetch, use custom story |

---

## Output structure

```
output/
└── 2026-04-13/
    └── 01-mcilroys-meltdown/
        ├── headline.txt          # Winning headline + summary + URL + image URL
        ├── lyrics.txt            # Title + full lyrics + caption
        ├── scene_plan.json       # LLM-planned scenes (one per 5s clip)
        ├── song.mp3              # Suno-generated audio
        ├── timed_lyrics.json     # Suno word-level timings (used by captioner)
        ├── clip_0.mp4            # Pollo clip — scene 0 (image-to-video)
        ├── clip_1.mp4            # Pollo clip — scene 1 (text-to-video)
        ├── clip_N.mp4            # Pollo clip — scene N
        ├── final.mp4              # Assembled video with watermark, no captions
        ├── lyrics.ass              # Generated ASS karaoke subtitle
        ├── final_captioned.mp4     # Karaoke-captioned (YouTube-ready)
        ├── tiktok_hook.ass         # 2-line white-bg hook caption (first 2s)
        └── final_tiktok.mp4        # Posted to TikTok (karaoke + opening hook card)

logs/
├── news_candidates.jsonl         # All fetched headlines with metadata
├── social_scores.jsonl           # Social scores per headline+date
├── story_classifications.jsonl   # VPI scores per headline+date (+ classifier label)
├── lyrics_classifications.jsonl  # LVI scores per generated lyric
├── flagged_stories.jsonl         # Stories above the combined threshold
├── selection_decisions.jsonl     # Full candidate ranking per run
├── api_calls.jsonl               # LLM + Suno + Pollo API call log
└── pipeline.log                  # Full pipeline log (stdout mirror)
```

---

## Key dependencies

| Dependency | Purpose |
|---|---|
| `openai` SDK | Ollama + Grok (OpenAI-compatible) |
| `httpx` | Async HTTP for all external APIs (Suno, Pollo, Reddit, HN, …) |
| `pytrends` | Google Trends scoring |
| `feedparser` | Google Trends RSS fallback |
| `tenacity` | Retry logic for Suno submit |
| `whisper` | Audio transcription fallback for captions |
| `ffmpeg` / `ffprobe` | Video assembly, caption burn, duration detection |
| `streamlit` | Dashboard (`app.py`) |
| `supabase-py` | Optional cloud log mirror |

---

## External services

| Service | Used for | Auth |
|---|---|---|
| NewsAPI | Headline candidates | `NEWS_API_KEY` |
| Reddit (public) | Social scoring | No key needed |
| Hacker News (Algolia) | Social scoring | No key needed |
| Google Trends | Social scoring | No key needed |
| Ollama (local) | Classification (half of dual) | No key (`OLLAMA_BASE_URL`) |
| Grok / xAI | Classification (half of dual), lyrics, scene planning | `XAI_API_KEY` |
| sunoapi.org | Music generation | `SUNOAPI_KEY`, `SUNOAPI_BASE` |
| Pollo AI | Video clip generation (Veo 3.1) | `POLLO_API_KEY` |
| TikTok API v2 | Publishing | `TIKTOK_CLIENT_KEY`, `TIKTOK_CLIENT_SECRET`, `TIKTOK_REFRESH_TOKEN` |
| Supabase (optional) | Cloud log mirror + dashboard queries | `SUPABASE_URL`, `SUPABASE_KEY` |

---

## Scheduler

`scheduler.py` runs two jobs:

| Job | Schedule | What it does |
|---|---|---|
| Hourly background job | Every 60 min (+ on startup) | Fetch headlines → social score → VPI classify (populates caches) |
| Daily full pipeline | 09:00 local time | Full news → lyrics → music → video → TikTok post |

For production on Windows, use Task Scheduler pointing at `pipeline.py` directly (see comment in `scheduler.py`).

---

## Dashboard

`streamlit run app.py` opens a web UI with pages:

- **News Feed** — all fetched headlines + their scores
- **Flagged Stories** — stories above the threshold with breakdown
- **Selection Decisions** — full candidate ranking per run
- **Story Classifications** — VPI breakdown per headline
- **Lyrics Classifications** — LVI breakdown per generated lyric
- **Credits** — cost tracking per run
- **API Logs** — LLM + Suno + Pollo call log
- **Runs** / **Run Detail** — browse output folders

---

## Known issues / open items

| Location | Issue |
|---|---|
| `pipeline.py:~180` | `audio` and `video` variables undefined when `--dry-run-full` skips generation |
| `config.py` | `ANTHROPIC_API_KEY` imported but not wired into the pipeline yet |
| `pipeline.py` | `--video-model` default help text still says `seedance-pro-1-5`; the actual default is `veo3-1` (from `pollo_generator.DEFAULT_MODEL`) |
| Windows / Python 3.9 | `asyncio.create_subprocess_exec` hangs or raises `NotImplementedError` — worked around in `video_assembler.py` and `captioner.py` via `asyncio.to_thread(subprocess.run, …)` |
