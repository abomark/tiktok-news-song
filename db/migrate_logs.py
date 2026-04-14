"""
One-time migration: reads all existing JSONL log files and upserts into Supabase.

Usage:
    python supabase/migrate_logs.py
    python supabase/migrate_logs.py --table flagged_stories   # single table only
    python supabase/migrate_logs.py --dry-run                 # print rows, don't write

Each table uses ON CONFLICT DO NOTHING so re-running is safe.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root or from supabase/ subfolder
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

LOGS_DIR = ROOT / "logs"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [warn] JSON parse error in {path.name}: {e}")
    return rows


def _upsert(sb, table: str, rows: list[dict], dry_run: bool) -> None:
    if not rows:
        print(f"  [skip] {table} — no rows")
        return
    if dry_run:
        print(f"  [dry-run] {table} — would upsert {len(rows)} rows")
        print(f"    sample: {json.dumps(rows[0], default=str)[:120]}")
        return
    # Supabase upsert with ignoreDuplicates=True is equivalent to ON CONFLICT DO NOTHING
    sb.table(table).upsert(rows, ignore_duplicates=True).execute()
    print(f"  [ok] {table} — upserted {len(rows)} rows")


# ── Per-table transformers ────────────────────────────────────────────────────

def _transform_news_candidates(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if not r.get("headline") or not r.get("date"):
            continue
        out.append({
            "headline":     r["headline"],
            "date":         r["date"],
            "fetched_at":   r.get("fetched_at") or r.get("timestamp"),
            "published_at": r.get("published_at"),
            "summary":      r.get("summary"),
            "source":       r.get("source"),
            "url":          r.get("url"),
        })
    return out


def _transform_social_scores(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if not r.get("headline") or not r.get("date"):
            continue
        out.append({
            "headline":     r["headline"],
            "date":         r["date"],
            "timestamp":    r.get("timestamp"),
            "source":       r.get("source"),
            "social_score": r.get("social_score"),
            "reddit_score": r.get("reddit_score"),
            "hn_score":     r.get("hn_score"),
            "trends_score": r.get("trends_score"),
        })
    return out


_FACTOR_KEYS = [
    "absurdity", "character_punchability", "cultural_reach", "emotional_heat",
    "memeability", "musical_fit", "timestamp_sensitivity", "moral_clarity",
    "visual_potential", "safe_harbor",
]

def _transform_story_classifications(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if not r.get("headline") or not r.get("date"):
            continue
        row = {
            "headline":   r["headline"],
            "date":       r["date"],
            "timestamp":  r.get("timestamp"),
            "summary":    r.get("summary"),
            "source":     r.get("source"),
            "url":        r.get("url"),
            "run_dir":    r.get("run_dir"),
            "angle":      r.get("angle"),
            "vpi":        r.get("vpi"),
            "vpi_label":  r.get("vpi_label"),
        }
        for key in _FACTOR_KEYS:
            row[key] = r.get(key)   # already a dict {score, rationale} or None
        out.append(row)
    return out


def _transform_flagged_stories(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if not r.get("headline") or not r.get("date"):
            continue
        out.append({
            "headline":      r["headline"],
            "date":          r["date"],
            "timestamp":     r.get("timestamp"),
            "source":        r.get("source"),
            "url":           r.get("url"),
            "summary":       r.get("summary"),
            "angle":         r.get("angle"),
            "combined_score": r.get("combined_score"),
            "social_score":  r.get("social_score"),
            "vpi":           r.get("vpi"),
            "vpi_label":     r.get("vpi_label"),
            "threshold":     r.get("threshold"),
        })
    return out


def _transform_selection_decisions(rows: list[dict]) -> list[dict]:
    """Handles both old schema (winner_headline) and new schema (flagged_headlines)."""
    out = []
    for r in rows:
        if not r.get("date") or not r.get("timestamp"):
            continue
        out.append({
            "date":                   r["date"],
            "timestamp":              r["timestamp"],
            "threshold":              r.get("threshold"),
            # new schema
            "flagged_headlines":      r.get("flagged_headlines"),
            "n_flagged":              r.get("n_flagged"),
            # old schema
            "winner_headline":        r.get("winner_headline"),
            "winner_combined_score":  r.get("winner_combined_score"),
            "winner_vpi":             r.get("winner_vpi"),
            "winner_social_score":    r.get("winner_social_score"),
            # candidates array — stored as JSONB
            "candidates":             r.get("candidates"),
        })
    return out


def _transform_api_calls(rows: list[dict]) -> list[dict]:
    """Non-key fields packed into a JSONB payload column."""
    out = []
    for r in rows:
        if not r.get("timestamp") or not r.get("api"):
            continue
        payload = {k: v for k, v in r.items()
                   if k not in ("timestamp", "api", "run_dir")}
        out.append({
            "timestamp": r["timestamp"],
            "api":       r["api"],
            "run_dir":   r.get("run_dir"),
            "payload":   payload,
        })
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

TABLES = {
    "news_candidates":      (LOGS_DIR / "news_candidates.jsonl",       _transform_news_candidates),
    "social_scores":        (LOGS_DIR / "social_scores.jsonl",         _transform_social_scores),
    "story_classifications":(LOGS_DIR / "story_classifications.jsonl", _transform_story_classifications),
    "flagged_stories":      (LOGS_DIR / "flagged_stories.jsonl",       _transform_flagged_stories),
    "selection_decisions":  (LOGS_DIR / "selection_decisions.jsonl",   _transform_selection_decisions),
    "api_calls":            (LOGS_DIR / "api_calls.jsonl",             _transform_api_calls),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate JSONL logs to Supabase")
    parser.add_argument("--table", default=None,
                        help=f"Migrate one table only. Choices: {', '.join(TABLES)}")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without actually writing")
    args = parser.parse_args()

    if args.table and args.table not in TABLES:
        print(f"Unknown table '{args.table}'. Choices: {', '.join(TABLES)}")
        sys.exit(1)

    sb = None
    if not args.dry_run:
        from db.client import get_client
        sb = get_client()

    targets = {args.table: TABLES[args.table]} if args.table else TABLES

    for table, (jsonl_path, transform) in targets.items():
        print(f"\n{table}")
        raw = _load_jsonl(jsonl_path)
        if not raw:
            continue
        rows = transform(raw)
        _upsert(sb, table, rows, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
