# CurrentNoise 🎵

Fully automated pipeline that picks the most viral news story of the day, writes satirical song lyrics, generates a Suno track, creates a Runway music video, burns in captions, and posts to TikTok — all without human intervention.

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
│story_classifier │  10-factor VPI scoring via gemma3/Ollama (parallel)
└────────┬────────┘  logs/story_classifications.jsonl
         │
         ▼
┌─────────────────┐
│ story_selector  │  Blends social 40% + VPI 60% → flags all ≥ threshold
└────────┬────────┘  logs/flagged_stories.jsonl
         │                logs/selection_decisions.jsonl
         ▼
┌─────────────────┐
│lyrics_generator │  Hook + verse + chorus via Ollama or Grok
└────────┬────────┘  output/<date>/<run>/lyrics.txt
         │
         ▼
┌─────────────────┐
│ music_generator │  Suno via Docker wrapper → song.mp3
└────────┬────────┘  output/<date>/<run>/song.mp3
         │
         ▼
┌─────────────────────────────────┐
│        video_generator          │
│  ┌─────────────────────────┐    │
│  │  clip_generator         │    │  Runway Gen-3 — one clip per lyric section
│  └────────────┬────────────┘    │
│  ┌────────────▼────────────┐    │
│  │  video_assembler        │    │  ffmpeg concat + audio mix
│  └────────────┬────────────┘    │
│  ┌────────────▼────────────┐    │
│  │  captioner              │    │  Whisper transcription + ffmpeg subtitle burn
│  └─────────────────────────┘    │
└────────┬────────────────────────┘
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
- Returns `list[NewsStory]`

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

- VPI = average of all 10 factor scores
- Model: **gemma3 via Ollama** (OpenAI-compatible, `http://localhost:11434/v1`)
- Results cached per `(headline, date)` — re-runs skip already-classified stories

### 4. Story flagging — `modules/story_selector.py`
- Accepts pre-computed social scores + VPI classifications — **no LLM calls**
- Normalises both to 0–1 (min-max within the current batch)
- Combined score: `social×0.40 + VPI×0.60`
- Flags **every** story with `combined_score ≥ 0.5` (configurable threshold)
- Guarantees at least one flagged story (promotes top story if nothing clears threshold)
- Already-flagged stories from earlier hourly runs are preserved

### 5. Lyrics generation — `modules/lyrics_generator.py`
- Calls an OpenAI-compatible endpoint (Ollama default; `--provider grok` → xAI)
- Structure: 1 hook (3 s) + 1 verse (4–6 lines) + 1 chorus (4 lines) ≈ 30–45 s
- Also generates: Suno style prompt, TikTok caption, 1–2 topic hashtags
- Uses the classifier's **satirical angle** as part of the prompt
- Skips stories that already have lyrics in today's output folder
- Prompts overridable via `assets/lyrics_system_prompt.txt` / `assets/lyrics_user_prompt.txt`

### 6. Music generation — `modules/music_generator.py`
- Submits lyrics + style prompt to [suno-api](https://github.com/gcui-art/suno-api) Docker wrapper
- Polls `/api/feed` every 5 s (timeout: 300 s)
- Downloads MP3; `ffprobe` reads actual duration

### 7. Video generation — `modules/video_generator.py`
- **7a. Clip generation** (`clip_generator.py`) — one Runway Gen-3 Alpha Turbo clip per lyric section; prompt derived from section content + headline
- **7b. Assembly** (`video_assembler.py`) — ffmpeg concatenates clips and mixes in the MP3
- **7c. Captioning** (`captioner.py`) — Whisper `base` transcribes audio; ffmpeg burns word-synced subtitles → `final_captioned.mp4`

### 8. TikTok publishing — `modules/tiktok_publisher.py`
- Refreshes OAuth token from `TIKTOK_REFRESH_TOKEN`
- Chunked upload (10 MB chunks) via `/v2/post/publish/video/init/`
- Public post; duet, comment, and stitch enabled

---

## Data stores

| Store | Purpose |
|---|---|
| `logs/*.jsonl` | Append-only local log — source of truth |
| **Supabase** | Cloud-queryable mirror; upsert with `ON CONFLICT DO NOTHING` |
| `output/<date>/<run>/` | All media artefacts for that run |

### Log files

| File | Contents |
|---|---|
| `logs/news_candidates.jsonl` | All fetched headlines + metadata |
| `logs/social_scores.jsonl` | Reddit / HN / Trends scores per headline |
| `logs/story_classifications.jsonl` | VPI factor scores per headline |
| `logs/flagged_stories.jsonl` | Stories that cleared the combined threshold |
| `logs/selection_decisions.jsonl` | Full ranked candidate list per run |
| `logs/api_calls.jsonl` | Every LLM + Runway API call (input + output) |
| `logs/pipeline.log` | Full pipeline stdout log |

---

## Output layout

```
output/
└── 2026-04-13/
    └── 01-mcilroys-meltdown/
        ├── headline.txt          # Headline + summary + source URL
        ├── lyrics.txt            # Title + lyrics + TikTok caption
        ├── song.mp3              # Suno-generated audio
        ├── clip_0.mp4            # Runway clip — hook
        ├── clip_1.mp4            # Runway clip — verse
        ├── clip_2.mp4            # Runway clip — chorus
        ├── final.mp4             # Assembled video (no captions)
        ├── final_captioned.mp4   # Posted to TikTok
        └── timed_lyrics.json     # Whisper word timings
```

---

## Quick start

```bash
# 1. Copy and fill in keys
cp .env.example .env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Ollama with gemma3
ollama serve
ollama pull gemma3

# 4. Start Suno Docker wrapper
docker run -d -p 3000:3000 gcui-art/suno-api   # see suno-api docs for cookie setup

# 5. Test news + lyrics only (cheapest)
python pipeline.py --lyrics-only

# 6. Full dry run (no TikTok post)
python pipeline.py --dry-run

# 7. Full run
python pipeline.py

# 8. Generate lyrics for all of today's flagged stories
python pipeline.py --all-flagged --lyrics-only

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
| `python pipeline.py --all-flagged` | Lyrics for all flagged stories today |
| `python pipeline.py --provider grok` | Use Grok instead of Ollama for lyrics |
| `python pipeline.py --headline "..." --summary "..."` | Skip news fetch |
| `python scheduler.py` | Hourly background job + daily 09:00 post |
| `streamlit run app.py` | Dashboard |
| `python db/migrate_logs.py` | One-time JSONL → Supabase migration |
| `python db/migrate_logs.py --dry-run` | Preview migration without writing |

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `NEWS_API_KEY` | Recommended | NewsAPI key (falls back to Google Trends RSS) |
| `OPENAI_API_KEY` | Yes | Set to `"ollama"` if not using OpenAI directly |
| `SUNO_COOKIE` | Yes (music) | Suno session cookie for the Docker wrapper |
| `SUNO_API_BASE` | No | Suno wrapper URL (default: `http://localhost:3000`) |
| `RUNWAYML_API_SECRET` | Yes (video) | Runway ML API key |
| `TIKTOK_CLIENT_KEY` | Yes (post) | TikTok developer app key |
| `TIKTOK_CLIENT_SECRET` | Yes (post) | TikTok developer app secret |
| `TIKTOK_REFRESH_TOKEN` | Yes (post) | Long-lived TikTok OAuth refresh token |
| `XAI_API_KEY` | Optional | xAI/Grok key for `--provider grok` |
| `OLLAMA_BASE_URL` | No | Ollama API URL (default: `http://localhost:11434/v1`) |
| `OLLAMA_MODEL` | No | Model name (default: `gemma3`) |
| `SUPABASE_URL` | Optional | Enables Supabase sync in dashboard + migration |
| `SUPABASE_KEY` | Optional | Supabase service role key |
| `REDDIT_CLIENT_ID` | Optional | Reddit API credentials (falls back to public API) |
| `REDDIT_CLIENT_SECRET` | Optional | Reddit API credentials |

---

## External services

| Service | Used for |
|---|---|
| [NewsAPI](https://newsapi.org) | Headline candidates |
| Reddit (public API / PRAW) | Social scoring |
| Hacker News (Algolia) | Social scoring |
| Google Trends (pytrends) | Social scoring |
| Ollama + gemma3 (local) | VPI classification + lyrics |
| Grok / xAI (optional) | Lyrics via `--provider grok` |
| [suno-api](https://github.com/gcui-art/suno-api) Docker | Music generation |
| [Runway ML](https://runwayml.com) Gen-3 | Video clips |
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
| Credits | API cost tracking (Runway, Suno, Grok) |
| API Logs | Every LLM + Runway call with inputs and outputs |
| Runs | Browse output folders by date |
| Run Detail | Inspect lyrics, audio, and video for a specific run |
