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
    # Insert in batches, skipping rows that already exist
    BATCH = 100
    inserted = 0
    skipped = 0
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        try:
            sb.table(table).upsert(batch, ignore_duplicates=True).execute()
            inserted += len(batch)
        except Exception:
            # Fall back to one-by-one to skip individual duplicates
            for row in batch:
                try:
                    sb.table(table).insert(row).execute()
                    inserted += 1
                except Exception:
                    skipped += 1
    print(f"  [ok] {table} — inserted {inserted} rows, skipped {skipped} duplicates")


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


def _transform_lyrics_classifications(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if not r.get("title") or not r.get("date"):
            continue
        out.append({
            "title":                r["title"],
            "date":                 r["date"],
            "timestamp":            r.get("timestamp"),
            "headline":             r.get("headline"),
            "classifier":           r.get("classifier"),
            "lvi":                  r.get("lvi"),
            "lvi_label":            r.get("lvi_label"),
            "lvi_gemma3":           r.get("lvi_gemma3"),
            "lvi_grok":             r.get("lvi_grok"),
            "verdict":              r.get("verdict"),
            "hook_strength":        r.get("hook_strength"),
            "hook_position":        r.get("hook_position"),
            "earworm_factor":       r.get("earworm_factor"),
            "singability":          r.get("singability"),
            "topicality":           r.get("topicality"),
            "recognition_trigger":  r.get("recognition_trigger"),
            "controversy_level":    r.get("controversy_level"),
            "satire_type":          r.get("satire_type"),
            "ingroup_signal":       r.get("ingroup_signal"),
            "visual_hook_potential":r.get("visual_hook_potential"),
            "meme_format_fit":      r.get("meme_format_fit"),
            "quotability":          r.get("quotability"),
            "participation_hook":   r.get("participation_hook"),
            "takedown_risk":        r.get("takedown_risk"),
            "algorithm_risk":       r.get("algorithm_risk"),
            "shadowban_words":      r.get("shadowban_words"),
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
    "news_candidates":        (LOGS_DIR / "news_candidates.jsonl",         _transform_news_candidates),
    "social_scores":          (LOGS_DIR / "social_scores.jsonl",           _transform_social_scores),
    "story_classifications":  (LOGS_DIR / "story_classifications.jsonl",   _transform_story_classifications),
    "flagged_stories":        (LOGS_DIR / "flagged_stories.jsonl",         _transform_flagged_stories),
    "selection_decisions":    (LOGS_DIR / "selection_decisions.jsonl",     _transform_selection_decisions),
    "lyrics_classifications": (LOGS_DIR / "lyrics_classifications.jsonl",  _transform_lyrics_classifications),
    "api_calls":              (LOGS_DIR / "api_calls.jsonl",               _transform_api_calls),
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
