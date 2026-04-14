"""
Shared Supabase client.

Usage in any module:
    from db.client import get_client
    sb = get_client()
    sb.table("flagged_stories").insert(row).execute()

Requires in .env:
    SUPABASE_URL=https://<project>.supabase.co
    SUPABASE_KEY=<service_role_key>   # use service_role, not anon, for server-side writes
"""

from __future__ import annotations
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def get_client():
    try:
        from supabase import create_client, Client
    except ImportError:
        raise ImportError(
            "supabase-py is not installed. Run: pip install supabase"
        )

    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")

    if not url or not key:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_KEY must be set in .env"
        )

    return create_client(url, key)
