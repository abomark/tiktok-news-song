from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_DIR = BASE_DIR / "output"
FONTS_DIR = ASSETS_DIR / "fonts"

# API keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SUNOAPI_KEY  = os.getenv("SUNOAPI_KEY", "")
SUNOAPI_BASE = os.getenv("SUNOAPI_BASE", "https://api.sunoapi.org")
RUNWAY_API_KEY = os.getenv("RUNWAYML_API_SECRET", "")  # kept for reference
POLLO_API_KEY  = os.getenv("POLLO_API_KEY", "")
TIKTOK_CLIENT_KEY = os.getenv("TIKTOK_CLIENT_KEY", "")
TIKTOK_CLIENT_SECRET = os.getenv("TIKTOK_CLIENT_SECRET", "")
TIKTOK_REFRESH_TOKEN = os.getenv("TIKTOK_REFRESH_TOKEN", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
SCORING_WEIGHTS = {
    "reddit": float(os.getenv("SCORE_WEIGHT_REDDIT", "0.40")),
    "hn":     float(os.getenv("SCORE_WEIGHT_HN",     "0.30")),
    "trends": float(os.getenv("SCORE_WEIGHT_TRENDS", "0.30")),
}

# Video settings — TikTok 9:16
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_FPS = 30

# Song style passed to Suno
SUNO_STYLE = "upbeat pop, modern, energetic, catchy hook, TikTok viral, radio quality"

# Watermark
WATERMARK_TEXT = "@currentnoise"

# Fixed hashtags (always appended)
FIXED_HASHTAGS = ["#newsatire", "#politicalsatire", "#aimusic"]

# News target region
NEWS_COUNTRY = "us"

# Claude model
CLAUDE_MODEL = "claude-sonnet-4-6"

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "gemma3")
