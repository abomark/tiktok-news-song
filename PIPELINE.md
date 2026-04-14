# CurrentNoise — Pipeline Overview

End-to-end flow from news API to TikTok post.

---

## Architecture

```
NewsAPI / Google Trends RSS
        │
        ▼
┌─────────────────┐
│  news_fetcher   │  Fetches up to 10 candidate headlines
└────────┬────────┘  Writes: logs/news_candidates.jsonl
         │  list[NewsStory]
         ▼
┌─────────────────┐
│  social_scorer  │  Scores each story via Reddit + HN + Google Trends (parallel)
└────────┬────────┘  Writes: logs/social_scores.jsonl
         │  list[ScoredStory]
         ▼
┌─────────────────┐
│story_classifier │  Scores each story on 10 VPI factors via gemma3/Ollama (parallel, cached)
└────────┬────────┘  Writes: logs/story_classifications.jsonl
         │  list[StoryClassification | None]
         ▼
┌─────────────────┐
│ story_selector  │  Blends social (40%) + VPI (60%) → flags all stories above threshold
└────────┬────────┘  Writes: logs/flagged_stories.jsonl, logs/selection_decisions.jsonl
         │  list[CandidateResult]  — pipeline picks flagged[0]
         ▼
┌─────────────────┐
│lyrics_generator │  Generates hook + verse + chorus + caption via Ollama or Grok
└────────┬────────┘  Writes: output/<date>/<run>/lyrics.txt
         │  Lyrics
         ▼
┌─────────────────┐
│ music_generator │  Submits lyrics to Suno via suno-api Docker wrapper, polls + downloads
└────────┬────────┘  Output: output/<date>/<run>/song.mp3
         │  AudioResult
         ▼
┌─────────────────┐
│ video_generator │  Orchestrates three sub-steps (see below)
└────────┬────────┘  Output: output/<date>/<run>/final_captioned.mp4
         │  VideoResult
         ▼
┌─────────────────┐
│tiktok_publisher │  Refreshes OAuth token, chunks and uploads video via TikTok API v2
└─────────────────┘
```

---

## Steps in detail

### Step 1 — News fetching (`modules/news_fetcher.py`)

- Calls NewsAPI `/v2/top-headlines` → up to 10 articles
- Falls back to Google Trends RSS if NewsAPI fails or has no key
- Scores every candidate in parallel across three signals:
  - **Reddit** — searches r/news, r/worldnews, r/politics; sums upvotes + comments
  - **Hacker News** — searches via Algolia; sums points + comments
  - **Google Trends** — pytrends 1-day interest, top 3 keywords from headline
- Each signal is normalised 0–100, then blended: `reddit×0.40 + hn×0.30 + trends×0.30`
- Writes social scores to `logs/social_scores.jsonl` (join key: `headline + date`)
- Returns `list[_ScoredStory]`

### Step 2 — Story classification (`modules/story_classifier.py`)

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

- VPI = average of all 10 scores
- Uses **gemma3 via Ollama** (`http://localhost:11434/v1`), OpenAI-compatible API
- Prompt loaded from `assets/classifier_prompt.md`
- Results cached by `(headline, date)` — already-classified stories are skipped
- Logs to `logs/story_classifications.jsonl`

### Step 3 — Story classification (`modules/story_classifier.py`)

Called for all candidates before selection (same module as step 2 but now invoked in `pipeline.py` and the scheduler's hourly job, not inside the selector). Results are cached per `(headline, date)`.

### Step 4 — Story flagging (`modules/story_selector.py`)

- Accepts pre-computed `list[ScoredStory]` + `list[StoryClassification | None]` — no LLM calls inside
- Normalises both score types to 0–1 (min-max across the current batch):
  - Social score (0–100) → 0–1
  - VPI (1–10) → 0–1 (failed classifications treated as 0)
- Combined score: `social×0.40 + vpi×0.60`
- Falls back to pure social score if all VPI classifications failed
- **Flags every story** with `combined_score >= threshold` (default 0.5)
- Guarantees at least one flagged story (flags the top story if nothing clears threshold)
- Writes flagged stories to `logs/flagged_stories.jsonl` (deduped by headline+date)
- Writes full ranked audit to `logs/selection_decisions.jsonl`
- Returns `list[CandidateResult]` sorted by combined score descending
- Pipeline picks `flagged[0]` (the top-scoring flagged story)

### Step 4 — Lyrics generation (`modules/lyrics_generator.py`)

- Calls an OpenAI-compatible endpoint (Ollama default, Grok via `--provider grok`)
- Output structure: hook (1 line) + verse (4–6 lines) + chorus (4 lines)
- Also generates: Suno style prompt, TikTok caption, 1–2 topic hashtags
- Fixed hashtags (`#newsatire #politicalsatire #aimusic`) always appended
- Platform tag (`#fyp` / `#foryou`) alternates daily
- Prompts loadable from `assets/lyrics_system_prompt.txt` / `assets/lyrics_user_prompt.txt`
- Output folder created: `output/<date>/<NN>-<slug>/`
- Writes `headline.txt` and `lyrics.txt` to run folder
- Returns `Lyrics` (title, sections, style_prompt, caption, topic_tags)

### Step 5 — Music generation (`modules/music_generator.py`)

- Submits to [suno-api](https://github.com/gcui-art/suno-api) Docker wrapper at `SUNO_API_BASE`
- Polls `/api/feed` every 5s, timeout after 300s
- Downloads MP3 to `output/<date>/<run>/song.mp3`
- Uses `ffprobe` to read actual duration
- Returns `AudioResult` (path, duration_seconds, title)

### Step 6 — Video generation (`modules/video_generator.py`)

Three sub-steps:

**6a. Clip generation (`modules/clip_generator.py`)**
- One Runway Gen-3 clip per lyric section (hook / verse / chorus)
- Each clip duration = `audio_duration / num_sections`
- Prompts are derived from section content + headline + style
- Polls Runway until complete, downloads `clip_0.mp4`, `clip_1.mp4`, ...

**6b. Video assembly (`modules/video_assembler.py`)**
- Concatenates clips with ffmpeg
- Mixes in the MP3 audio track
- Trims to audio duration
- Output: `final.mp4`

**6c. Captioning (`modules/captioner.py`)**
- Transcribes audio with **Whisper** (default model: `base`)
- Burns word-synced lyric captions into the video with ffmpeg
- Output: `final_captioned.mp4`

Returns `VideoResult` (path to `final_captioned.mp4` or `final.mp4` if captioning failed)

### Step 7 — TikTok publishing (`modules/tiktok_publisher.py`)

- Refreshes OAuth access token from stored `TIKTOK_REFRESH_TOKEN`
- Initialises chunked upload via `/v2/post/publish/video/init/`
- Uploads in 10 MB chunks with `Content-Range` headers
- Post is public; duet, comment, and stitch enabled by default
- Returns `PublishResult` (publish_id)

---

## Run modes

| Command | What runs |
|---|---|
| `python pipeline.py` | Full pipeline → posts to TikTok |
| `python pipeline.py --dry-run` | All steps except TikTok post |
| `python pipeline.py --dry-run-full` | News + lyrics only (no Suno, Runway, TikTok) |
| `python pipeline.py --lyrics-only` | News + lyrics, then stop |
| `python pipeline.py --provider grok` | Use Grok (xAI) for lyrics instead of Ollama |
| `python pipeline.py --headline "..." --summary "..."` | Skip news fetch, use custom story |

---

## Output structure

```
output/
└── 2026-04-13/
    └── 01-mcilroys-meltdown/
        ├── headline.txt          # Winning headline + summary + URL
        ├── lyrics.txt            # Title + full lyrics + caption
        ├── song.mp3              # Suno-generated audio
        ├── clip_0.mp4            # Runway clip — hook
        ├── clip_1.mp4            # Runway clip — verse
        ├── clip_2.mp4            # Runway clip — chorus
        ├── final.mp4             # Assembled video (no captions)
        ├── final_captioned.mp4   # Final video with burned-in captions
        └── timed_lyrics.json     # Whisper word timings

logs/
├── social_scores.jsonl           # Social scores per headline+date
├── story_classifications.jsonl   # VPI scores per headline+date
├── selection_decisions.jsonl     # Full candidate ranking per run
├── news_candidates.jsonl         # All fetched headlines with scores
├── api_calls.jsonl               # LLM + Runway API call log
└── pipeline.log                  # Full pipeline log (stdout mirror)
```

---

## Key dependencies

| Dependency | Purpose |
|---|---|
| `openai` SDK | Ollama + Grok (OpenAI-compatible) |
| `httpx` | Async HTTP for all external APIs |
| `pytrends` | Google Trends scoring |
| `feedparser` | Google Trends RSS fallback |
| `tenacity` | Retry logic for Suno submit |
| `whisper` | Audio transcription for captions |
| `ffmpeg` / `ffprobe` | Video assembly + duration detection |
| `streamlit` | Dashboard (`app.py`) |

---

## External services

| Service | Used for | Auth |
|---|---|---|
| NewsAPI | Headline candidates | `NEWS_API_KEY` |
| Reddit (public) | Social scoring | No key needed |
| Hacker News (Algolia) | Social scoring | No key needed |
| Google Trends | Social scoring | No key needed |
| Ollama (local) | Classification + lyrics | No key (`OLLAMA_BASE_URL`) |
| Grok / xAI | Lyrics (optional) | `XAI_API_KEY` |
| Suno (via Docker) | Music generation | `SUNO_COOKIE`, `SUNO_API_BASE` |
| Runway ML | Video clips | `RUNWAYML_API_SECRET` |
| TikTok API v2 | Publishing | `TIKTOK_CLIENT_KEY`, `TIKTOK_CLIENT_SECRET`, `TIKTOK_REFRESH_TOKEN` |

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
- **Credits** — cost tracking per run
- **Selection Decisions** — full candidate ranking per run
- **Story Classifications** — VPI breakdown per headline
- **API Logs** — LLM + Runway call log
- **Runs** / **Run Detail** — browse output folders

---

## Known issues

| Location | Issue |
|---|---|
| ~~`story_selector.py:209`~~ | Fixed — `classification` is now guarded throughout |
| `pipeline.py:179–190` | `audio` and `video` variables undefined when `--dry-run-full` skips generation |
| `config.py:16–17` | `os.environ["OPENAI_API_KEY"]` and `os.environ["SUNO_COOKIE"]` crash at import if missing, even for `--lyrics-only` runs |
| `pipeline.py:19` | `ANTHROPIC_API_KEY` imported but never used |
| `.env.example` | Documents `OPENAI_API_KEY` as "for DALL-E 3" but it is actually used as the Ollama passthrough key (set to any non-empty string) |
