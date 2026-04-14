"""Shared utilities for the pipeline modules."""

from __future__ import annotations
import json
from datetime import date, datetime
from pathlib import Path

_LOGS_DIR = Path(__file__).parent.parent / "logs"


def log_api_call(api_name: str, params: dict, run_dir: Path | None = None) -> None:
    """Append an API call to the central logs/api_calls.jsonl file.

    Each line is a self-contained JSON object (JSONL format) so appending is safe.
    """
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _LOGS_DIR / "api_calls.jsonl"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "api": api_name,
        "run_dir": str(run_dir) if run_dir else None,
        **params,
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def find_run_dir(date_str: str | None = None, run: str | None = None) -> Path:
    """Find the output run directory.

    Args:
        date_str: Date string like "2026-04-12" (default: today)
        run: Run identifier — a number like "1" or "01", a slug like "ceasefire-shuffle",
             or a full subfolder name like "01-ceasefire-shuffle".
             If None, uses the latest (highest numbered) run folder.

    Returns:
        Path to the run directory.
    """
    day = date_str or date.today().isoformat()
    day_dir = Path("output") / day

    if not day_dir.exists():
        # Fallback: maybe they're using the old flat structure
        return day_dir

    # List subdirectories
    subdirs = sorted([d for d in day_dir.iterdir() if d.is_dir()])

    if not subdirs:
        # No run folders yet — old flat structure or empty
        return day_dir

    if run is None:
        # Use the latest run
        return subdirs[-1]

    # Try to match by number, slug, or full name
    for d in subdirs:
        if d.name == run:
            return d
        if d.name.startswith(f"{int(run):02d}-") if run.isdigit() else False:
            return d
        if run.lower() in d.name.lower():
            return d

    # Not found — return as literal path
    candidate = day_dir / run
    if candidate.exists():
        return candidate

    print(f"Warning: run '{run}' not found in {day_dir}, using latest")
    return subdirs[-1] if subdirs else day_dir
