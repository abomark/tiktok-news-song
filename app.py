"""
Streamlit dashboard for the TikTok News Song pipeline.

Run with:
    streamlit run app.py
"""

import json
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

LOGS_DIR = Path("logs")
OUTPUT_DIR = Path("output")

st.set_page_config(page_title="CurrentNoise Dashboard", page_icon="🎵", layout="wide")
st.title("CurrentNoise Dashboard")


# ── Sidebar: navigation ──────────────────────────────────────────────────────

_HAS_LOCAL_OUTPUT = OUTPUT_DIR.exists() and any(
    p.is_dir() and p.name[:4].isdigit() for p in OUTPUT_DIR.iterdir()
)
_PAGES = ["News Feed", "Flagged Stories", "Selection Decisions", "Story Classifications", "Lyrics Classifications", "Credits", "API Logs"]
if _HAS_LOCAL_OUTPUT:
    _PAGES += ["Runs", "Run Detail"]
page = st.sidebar.radio("Page", _PAGES)


# ── Data loading — Supabase with JSONL fallback ───────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    """Read a local JSONL file into a list of dicts."""
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


@st.cache_resource
def _get_supabase():
    """Return a Supabase client, or None if not configured."""
    # st.secrets takes precedence (Streamlit Cloud); fall back to env vars (local)
    url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
    key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception:
        return None


def _load_table(table: str, jsonl_fallback: Path, order_col: str = "date") -> list[dict]:
    """
    Load data from Supabase if configured, otherwise fall back to local JSONL.
    Returns a list of dicts with the same shape in both cases.
    """
    sb = _get_supabase()
    if sb is not None:
        try:
            resp = sb.table(table).select("*").order(order_col).execute()
            rows = resp.data or []
            # Supabase returns candidates as a parsed list (JSONB) — normalise to str dates
            for row in rows:
                if "date" in row and row["date"]:
                    row["date"] = str(row["date"])[:10]
                # api_calls: unpack payload back into the row for display compatibility
                if table == "api_calls" and isinstance(row.get("payload"), dict):
                    for k, v in row["payload"].items():
                        if k not in row:
                            row[k] = v
            return rows
        except Exception as e:
            st.sidebar.warning(f"Supabase error ({table}): {e} — using local JSONL")
    return _load_jsonl(jsonl_fallback)


def _using_supabase() -> bool:
    return _get_supabase() is not None


if _get_supabase() is not None:
    st.sidebar.success("☁️ Supabase")
else:
    st.sidebar.info("📁 Local JSONL")


if page == "News Feed":
    st.header("News Feed")
    st.caption("All headlines fetched by the hourly job. Shows what data exists for each story.")

    candidates = _load_table("news_candidates", LOGS_DIR / "news_candidates.jsonl", "fetched_at")

    if not candidates:
        st.info("No headlines logged yet. Start the scheduler or run the pipeline.")
    else:
        # Build lookup indexes keyed by headline (most recent entry wins for dupes)
        social_by_headline: dict[str, dict] = {}
        for e in _load_table("social_scores", LOGS_DIR / "social_scores.jsonl"):
            social_by_headline[e.get("headline", "")] = e

        clf_by_headline: dict[str, dict] = {}
        for e in _load_table("story_classifications", LOGS_DIR / "story_classifications.jsonl"):
            clf_by_headline[e.get("headline", "")] = e

        flagged_headlines: set[str] = set()
        for e in _load_table("flagged_stories", LOGS_DIR / "flagged_stories.jsonl"):
            flagged_headlines.add(e.get("headline", ""))

        # Most recent selection_decisions entry keyed by headline — for norms + combined
        scoring_by_headline: dict[str, dict] = {}
        for e in _load_table("selection_decisions", LOGS_DIR / "selection_decisions.jsonl"):
            for c in (e.get("candidates") or []):
                hl = c.get("headline", "")
                if hl:
                    scoring_by_headline[hl] = c  # last write wins (most recent run)

        # ── Filters ──────────────────────────────────────────────────────────
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            date_options = sorted({e.get("date", "") for e in candidates}, reverse=True)
            filter_date = st.selectbox("Date", ["All"] + date_options)
        with col_f2:
            filter_social = st.selectbox("Social scored", ["All", "Yes", "No"])
        with col_f3:
            filter_clf = st.selectbox("Classified", ["All", "Yes", "No"])
        with col_f4:
            filter_selected = st.selectbox("Flagged", ["All", "Yes", "No"])

        # Apply filters
        filtered = list(reversed(candidates))   # newest first
        if filter_date != "All":
            filtered = [e for e in filtered if e.get("date") == filter_date]
        if filter_social == "Yes":
            filtered = [e for e in filtered if e.get("headline") in social_by_headline]
        elif filter_social == "No":
            filtered = [e for e in filtered if e.get("headline") not in social_by_headline]
        if filter_clf == "Yes":
            filtered = [e for e in filtered if e.get("headline") in clf_by_headline]
        elif filter_clf == "No":
            filtered = [e for e in filtered if e.get("headline") not in clf_by_headline]
        if filter_selected == "Yes":
            filtered = [e for e in filtered if e.get("headline") in flagged_headlines]
        elif filter_selected == "No":
            filtered = [e for e in filtered if e.get("headline") not in flagged_headlines]

        st.caption(f"Showing {len(filtered)} of {len(candidates)} headlines")

        # ── Summary table ─────────────────────────────────────────────────────
        import pandas as pd

        rows = []
        for e in filtered:
            hl = e.get("headline", "")
            social = social_by_headline.get(hl)
            clf = clf_by_headline.get(hl)
            is_flagged = hl in flagged_headlines
            rows.append({
                "Date": e.get("date", ""),
                "Published": (e.get("published_at") or "")[:16].replace("T", " "),
                "Fetched": (e.get("fetched_at") or e.get("timestamp") or "")[:16].replace("T", " "),
                "Source": e.get("source", ""),
                "Headline": hl[:80],
                "Social": "✓" if social else "—",
                "Score": round(social["social_score"], 2) if social else None,
                "Classified": "✓" if clf else "—",
                "VPI": round(clf["vpi"], 1) if clf else None,
                "Flagged": "🚩" if is_flagged else "",
            })

        df = pd.DataFrame(rows)

        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Detail expanders ──────────────────────────────────────────────────
        st.divider()
        st.subheader("Detail")

        for e in filtered:
            hl = e.get("headline", "")
            social = social_by_headline.get(hl)
            clf = clf_by_headline.get(hl)
            scoring = scoring_by_headline.get(hl)
            is_flagged = hl in flagged_headlines

            prefix = "🚩 " if is_flagged else ""
            combined_str = f"Combined {scoring['combined_score']:.3f} · " if scoring else ""
            vpi_str = f"VPI {clf['vpi']:.1f} · " if clf else ""
            score_str = f"Social {social['social_score']:.1f} · " if social else ""
            with st.expander(f"{prefix}{combined_str}{score_str}{vpi_str}{hl[:65]}", expanded=False):
                meta1, meta2 = st.columns(2)
                with meta1:
                    st.markdown(f"**Source:** {e.get('source', '')}  |  **Date:** {e.get('date', '')}")
                    st.markdown(f"**Published:** {e.get('published_at') or '—'}")
                    st.markdown(f"**Fetched:** {e.get('fetched_at') or e.get('timestamp') or '—'}")
                    if e.get("url"):
                        st.markdown(f"[Open article]({e['url']})")
                with meta2:
                    if is_flagged:
                        st.success("Flagged as high-potential story")
                    if not social:
                        st.warning("No social score yet")
                    if not clf:
                        st.warning("No VPI classification yet")

                if e.get("summary"):
                    st.markdown(f"**Summary:** {e['summary']}")

                # ── Combined score breakdown ──────────────────────────────────
                if scoring:
                    st.markdown("**Combined score breakdown** — `0.40 × social_norm + 0.60 × vpi_norm`")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Combined", f"{scoring['combined_score']:.4f}")
                    c2.metric("Social norm (×0.40)", f"{scoring.get('social_norm', 0):.4f}")
                    c3.metric("VPI norm (×0.60)", f"{scoring.get('vpi_norm', 0):.4f}")

                # ── Social signal breakdown ───────────────────────────────────
                if social:
                    st.markdown("**Social score** — `reddit×0.40 + hn×0.30 + trends×0.30` (raw, before batch normalisation)")
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Social (raw)", f"{social['social_score']:.2f}")
                    s2.metric("Reddit", f"{social['reddit_score']:.2f}")
                    s3.metric("HN", f"{social['hn_score']:.2f}")
                    s4.metric("Trends", f"{social['trends_score']:.2f}")

                # ── VPI breakdown ─────────────────────────────────────────────
                if clf:
                    st.markdown("**VPI classification** — average of 10 factors (1–10 each)")
                    v1, v2 = st.columns(2)
                    v1.metric("VPI (raw avg)", f"{clf['vpi']:.1f} / 10")
                    v2.markdown(f"_{clf.get('vpi_label', '')}_")
                    if clf.get("angle"):
                        st.info(f"**Satirical angle:** {clf['angle']}")


# ── Credits ──────────────────────────────────────────────────────────────────

def fetch_runway_credits(api_key: str) -> dict:
    """Fetch Runway ML credit balance."""
    if not api_key:
        return {"status": "no_key"}
    try:
        r = httpx.get(
            "https://api.dev.runwayml.com/v1/organization",
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Runway-Version": "2024-11-06",
            },
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            credits = data.get("creditBalance", data.get("credits", data.get("balance")))
            return {"status": "ok", "credits": credits, "raw": data}
        return {"status": "error", "code": r.status_code, "body": r.text[:300]}
    except Exception as e:
        return {"status": "exception", "error": str(e)}


def fetch_openai_credits(api_key: str) -> dict:
    """Verify OpenAI API key is valid; billing requires browser session."""
    if not api_key:
        return {"status": "no_key"}
    try:
        r = httpx.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if r.status_code == 200:
            return {"status": "ok_key_valid"}
        return {"status": "error", "code": r.status_code, "body": r.text[:300]}
    except Exception as e:
        return {"status": "exception", "error": str(e)}


def fetch_grok_credits(api_key: str) -> dict:
    """Check xAI/Grok API key validity and fetch credit balance."""
    if not api_key:
        return {"status": "no_key"}
    try:
        r = httpx.get(
            "https://api.x.ai/v1/api-key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            return {"status": "ok", "data": data}
        return {"status": "error", "code": r.status_code, "body": r.text[:300]}
    except Exception as e:
        return {"status": "exception", "error": str(e)}


def fetch_newsapi_remaining(api_key: str) -> dict:
    """Fetch NewsAPI remaining requests via response headers."""
    if not api_key:
        return {"status": "no_key"}
    try:
        r = httpx.get(
            "https://newsapi.org/v2/top-headlines",
            params={"country": "us", "pageSize": 1, "apiKey": api_key},
            timeout=10,
        )
        if r.status_code == 200:
            remaining = r.headers.get("X-RateLimit-Remaining") or r.headers.get("x-ratelimit-remaining")
            limit = r.headers.get("X-RateLimit-Limit") or r.headers.get("x-ratelimit-limit")
            total_results = r.json().get("totalResults")
            return {
                "status": "ok",
                "requests_remaining": remaining,
                "requests_limit": limit,
                "total_results_available": total_results,
                "headers": dict(r.headers),
            }
        return {"status": "error", "code": r.status_code, "body": r.text[:300]}
    except Exception as e:
        return {"status": "exception", "error": str(e)}


if page == "Credits":
    st.header("API Credits & Quotas")
    st.caption("Click 'Check credits' to fetch current status for all APIs.")

    if st.button("Check credits", type="primary"):
        def _secret(key: str) -> str:
            return st.secrets.get(key, os.getenv(key, ""))
        runway_key = _secret("RUNWAYML_API_SECRET")
        openai_key = _secret("OPENAI_API_KEY")
        news_key = _secret("NEWS_API_KEY")
        grok_key = _secret("XAI_API_KEY")

        col1, col2, col3, col4 = st.columns(4)

        # ── Runway ML ────────────────────────────────────────────────
        with col1:
            st.subheader("🎬 Runway ML")
            with st.spinner("Fetching..."):
                result = fetch_runway_credits(runway_key)
            if result["status"] == "no_key":
                st.warning("RUNWAYML_API_SECRET not set")
            elif result["status"] == "ok":
                credits = result["credits"]
                if credits is not None:
                    st.metric("Credits remaining", f"{credits:,}")
                else:
                    st.info("Response received but no credits field found")
                    st.json(result.get("raw", {}))
            else:
                st.error(f"Error: {result.get('code') or result.get('error')}")
                if "body" in result:
                    st.code(result["body"])

        # ── Grok / xAI ───────────────────────────────────────────────
        with col2:
            st.subheader("⚡ Grok (xAI)")
            with st.spinner("Fetching..."):
                result = fetch_grok_credits(grok_key)
            if result["status"] == "no_key":
                st.warning("XAI_API_KEY not set")
            elif result["status"] == "ok":
                data = result.get("data", {})
                # xAI returns key info — show whatever fields are available
                name = data.get("name") or data.get("api_key_id") or "—"
                st.success(f"Key valid: `{name}`")
                accrued = data.get("accrued_usage_usd")
                if accrued is not None:
                    st.metric("Accrued usage", f"${accrued:.4f}")
                team = data.get("team_id") or data.get("organization")
                if team:
                    st.caption(f"Team: {team}")
                with st.expander("Raw response"):
                    st.json(data)
            else:
                st.error(f"Error: {result.get('code') or result.get('error')}")
                if "body" in result:
                    st.code(result["body"])

        # ── OpenAI ───────────────────────────────────────────────────
        with col3:
            st.subheader("🤖 OpenAI")
            with st.spinner("Fetching..."):
                result = fetch_openai_credits(openai_key)
            if result["status"] == "no_key":
                st.warning("OPENAI_API_KEY not set")
            elif result["status"] == "ok_key_valid":
                st.success("API key is valid")
                st.info("OpenAI billing requires browser login — check balance at [platform.openai.com/usage](https://platform.openai.com/usage)")
            else:
                st.error(f"Error: {result.get('code') or result.get('error')}")
                if "body" in result:
                    st.code(result["body"])

        # ── NewsAPI ──────────────────────────────────────────────────
        with col4:
            st.subheader("📰 NewsAPI")
            with st.spinner("Fetching..."):
                result = fetch_newsapi_remaining(news_key)
            if result["status"] == "no_key":
                st.warning("NEWS_API_KEY not set")
            elif result["status"] == "ok":
                remaining = result.get("requests_remaining")
                limit = result.get("requests_limit")
                if remaining is not None:
                    st.metric("Requests remaining", remaining)
                if limit is not None:
                    st.metric("Daily limit", limit)
                if remaining is None and limit is None:
                    st.info("API responded OK but no rate-limit headers found")
                    st.caption("Check the NewsAPI dashboard for quota info")
            else:
                st.error(f"Error: {result.get('code') or result.get('error')}")
                if "body" in result:
                    st.code(result["body"])

        # ── Suno ─────────────────────────────────────────────────────
        st.divider()
        with st.expander("ℹ️ Suno — cookie-based, no standard API"):
            st.markdown(
                "Suno is accessed via cookie authentication and has no standard credits endpoint. "
                "Log in at [suno.com](https://suno.com) to check account status."
            )
    else:
        st.info("Trykk 'Hent kreditter' for å hente gjeldende status.")


# ── Flagged Stories ──────────────────────────────────────────────────────────

if page == "Flagged Stories":
    st.header("Flagged Stories")
    st.caption("Stories that cleared the combined-score threshold. The pipeline picks from this list.")

    entries = _load_table("flagged_stories", LOGS_DIR / "flagged_stories.jsonl")
    if not entries:
        st.info("No flagged stories yet. Run the scheduler or pipeline to populate.")
    else:
        import pandas as pd

        # ── Filters ──────────────────────────────────────────────────────────
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            date_options = sorted({e.get("date", "") for e in entries}, reverse=True)
            filter_date = st.selectbox("Date", ["All"] + date_options, key="flagged_date")
        with col_f2:
            filter_text = st.text_input("Search headline", placeholder="Filter...", key="flagged_search")

        filtered = list(reversed(entries))
        if filter_date != "All":
            filtered = [e for e in filtered if e.get("date") == filter_date]
        if filter_text:
            filtered = [e for e in filtered if filter_text.lower() in e.get("headline", "").lower()]

        st.caption(f"Showing {len(filtered)} of {len(entries)} flagged stories")

        # ── Summary table ─────────────────────────────────────────────────────
        rows = []
        for e in filtered:
            rows.append({
                "Date": e.get("date", ""),
                "Headline": e.get("headline", "")[:80],
                "Combined": round(e.get("combined_score", 0), 4),
                "VPI": round(e.get("vpi") or 0, 1),
                "VPI Label": e.get("vpi_label", ""),
                "Social": round(e.get("social_score", 0), 2),
                "Threshold": round(e.get("threshold", 0), 2),
                "Source": e.get("source", ""),
            })
        df = pd.DataFrame(rows)

        def _color_combined(val):
            if val >= 0.8:
                return "background-color: #1a7a1a; color: white"
            if val >= 0.6:
                return "background-color: #4a7a1a; color: white"
            return "background-color: #7a6a1a; color: white"

        st.dataframe(
            df.style.applymap(_color_combined, subset=["Combined"]),
            use_container_width=True,
            hide_index=True,
        )

        # ── Detail cards ──────────────────────────────────────────────────────
        st.divider()
        for e in filtered:
            combined = e.get("combined_score", 0)
            vpi = e.get("vpi") or 0
            hl = e.get("headline", "")
            ts = e.get("timestamp", "")[:16].replace("T", " ")

            if combined >= 0.8:
                icon = "🔥"
            elif combined >= 0.6:
                icon = "🚩"
            else:
                icon = "📌"

            with st.expander(f"{icon} {ts}  combined={combined:.3f}  VPI={vpi:.1f}  —  {hl[:65]}", expanded=False):
                col_l, col_r = st.columns([3, 1])
                with col_l:
                    st.markdown(f"**Source:** {e.get('source', '')}  |  **Date:** {e.get('date', '')}")
                    if e.get("url"):
                        st.markdown(f"[Open article]({e['url']})")
                    if e.get("summary"):
                        st.markdown(f"**Summary:** {e['summary']}")
                    if e.get("angle"):
                        st.info(f"**Satirical angle:** {e['angle']}")
                with col_r:
                    st.metric("Combined", f"{combined:.4f}")
                    st.caption("`0.40 × social_norm + 0.60 × vpi_norm`")
                    st.metric("VPI (raw)", f"{vpi:.1f} / 10")
                    st.caption(e.get("vpi_label", ""))
                    st.metric("Social (raw)", f"{e.get('social_score', 0):.2f}")
                    st.caption(f"Threshold: {e.get('threshold', 0):.2f}")


# ── Selection Decisions ───────────────────────────────────────────────────────

if page == "Selection Decisions":
    st.header("Selection Decisions")
    st.caption("One entry per pipeline run. Full candidate ranking with combined scores.")

    entries = _load_table("selection_decisions", LOGS_DIR / "selection_decisions.jsonl", "timestamp")
    if not entries:
        st.info("No decisions yet. Run the pipeline to generate.")
    else:
        import pandas as pd
        for entry in reversed(entries):
            ts = entry.get("timestamp", "")[:16].replace("T", " ")
            flagged = entry.get("flagged_headlines", [])
            n_flagged = entry.get("n_flagged", len(flagged))
            threshold = entry.get("threshold", 0.5)
            n_total = len(entry.get("candidates", []))
            label = flagged[0][:55] if flagged else "—"

            with st.expander(f"🚩 {ts}  —  {label}  (+{n_flagged - 1} more)" if n_flagged > 1 else f"🚩 {ts}  —  {label}", expanded=False):
                c1, c2 = st.columns(2)
                c1.metric("Flagged stories", f"{n_flagged} / {n_total}")
                c2.metric("Threshold", f"{threshold:.2f}")

                if flagged:
                    st.markdown("**Flagged headlines:**")
                    for hl in flagged:
                        st.markdown(f"- 🚩 {hl}")

                st.markdown("**All candidates (ranked by combined score)** — `combined = 0.40 × social_norm + 0.60 × vpi_norm`")
                rows = []
                for c in entry.get("candidates", []):
                    rows.append({
                        "": "🚩" if c.get("flagged") else "",
                        "Headline": c.get("headline", "")[:55],
                        "Combined": round(c.get("combined_score", 0), 4),
                        "Social norm": round(c.get("social_norm", 0), 4),
                        "VPI norm": round(c.get("vpi_norm", 0), 4),
                        "VPI (raw)": round(c.get("vpi") or 0, 1),
                        "Social (raw)": round(c.get("social_score", 0), 2),
                        "Reddit": round(c.get("reddit_score", 0), 2),
                        "HN": round(c.get("hn_score", 0), 2),
                        "Trends": round(c.get("trends_score", 0), 2),
                        "Angle": c.get("angle", "")[:50],
                    })
                df = pd.DataFrame(rows)

                def _hl_flagged(row):
                    if row[""] == "🚩":
                        return ["background-color: #1a3a1a; color: white"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    df.style.apply(_hl_flagged, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )


# ── Story Classifications ─────────────────────────────────────────────────────

_FACTOR_LABELS = {
    "absurdity": "Absurdity Score",
    "character_punchability": "Character Punchability",
    "cultural_reach": "Cultural Reach",
    "emotional_heat": "Emotional Heat",
    "memeability": "Memeability",
    "musical_fit": "Musical Fit",
    "timestamp_sensitivity": "Timestamp Sensitivity",
    "moral_clarity": "Moral Clarity",
    "visual_potential": "Visual Potential",
    "safe_harbor": "Safe Harbor",
}

if page == "Story Classifications":
    st.header("Story Classifications — Viral Potential Index")

    entries = _load_table("story_classifications", LOGS_DIR / "story_classifications.jsonl", "timestamp")
    if not entries:
        st.info("No classifications yet. Run the pipeline to generate.")
    else:
        # Load social scores and index by (headline, date) for left-join
        social_index: dict[tuple[str, str], dict] = {
            (s.get("headline", ""), s.get("date", "")): s
            for s in _load_table("social_scores", LOGS_DIR / "social_scores.jsonl")
        }

        # Summary table
        import pandas as pd
        rows = []
        for e in reversed(entries):
            vpi = e.get("vpi", 0)
            social = social_index.get((e.get("headline", ""), e.get("date", "")), {})
            rows.append({
                "Tidspunkt": e.get("timestamp", "")[:16].replace("T", " "),
                "Overskrift": e.get("headline", "")[:70],
                "VPI": vpi,
                "Social": round(social.get("social_score", 0) or 0, 2),
                "Reddit": round(social.get("reddit_score", 0) or 0, 2),
                "HN": round(social.get("hn_score", 0) or 0, 2),
                "Trends": round(social.get("trends_score", 0) or 0, 2),
                "Vurdering": e.get("vpi_label", ""),
                "Kilde": e.get("source", ""),
            })
        df = pd.DataFrame(rows)

        def _color_vpi(val):
            if val >= 8:
                return "background-color: #1a7a1a; color: white"
            if val >= 6:
                return "background-color: #4a7a1a; color: white"
            if val >= 4:
                return "background-color: #7a6a1a; color: white"
            return "background-color: #7a1a1a; color: white"

        st.dataframe(
            df.style.applymap(_color_vpi, subset=["VPI"]),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        # Detail cards
        for e in reversed(entries):
            vpi = e.get("vpi", 0)
            headline = e.get("headline", "")
            timestamp = e.get("timestamp", "")[:16].replace("T", " ")
            social = social_index.get((headline, e.get("date", "")), {})

            if vpi >= 8:
                icon = "🔥"
            elif vpi >= 6:
                icon = "✅"
            elif vpi >= 4:
                icon = "⚠️"
            else:
                icon = "❌"

            with st.expander(f"{icon} VPI {vpi:.1f} — {headline[:70]}  ({timestamp})", expanded=False):
                col_meta, col_vpi = st.columns([2, 1])
                with col_meta:
                    st.markdown(f"**Kilde:** {e.get('source', '')}  |  **Run:** `{e.get('run_dir') or '—'}`")
                    if e.get("url"):
                        st.markdown(f"[Åpne artikkel]({e['url']})")
                    st.markdown(f"**Summary:** {e.get('summary', '')}")
                    if e.get("angle"):
                        st.info(f"**Satirisk vinkel:** {e['angle']}")
                with col_vpi:
                    st.metric("VPI", f"{vpi:.1f} / 10")
                    st.caption(e.get("vpi_label", ""))
                    if social:
                        st.metric("Social", f"{social.get('social_score', 0):.2f}")
                        st.caption(f"Reddit {social.get('reddit_score', 0):.2f} · HN {social.get('hn_score', 0):.2f} · Trends {social.get('trends_score', 0):.2f}")

                st.markdown("**Faktorer:**")
                factor_cols = st.columns(2)
                for i, (key, label) in enumerate(_FACTOR_LABELS.items()):
                    factor = e.get(key, {})
                    score = factor.get("score", 0) if isinstance(factor, dict) else 0
                    rationale = factor.get("rationale", "") if isinstance(factor, dict) else ""
                    with factor_cols[i % 2]:
                        bar = "█" * score + "░" * (10 - score)
                        st.markdown(f"**{label}**  `{score}/10`")
                        st.caption(f"{bar}  {rationale}")


# ── Lyrics Classifications ───────────────────────────────────────────────────

if page == "Lyrics Classifications":
    st.header("Lyrics Classifications — Lyrics Virality Index")
    st.caption("LVI breakdown for generated lyrics across Hook Mechanics, Cultural Payload, Creator Bait, and Platform Risk.")

    entries = _load_table("lyrics_classifications", LOGS_DIR / "lyrics_classifications.jsonl", "timestamp")
    if not entries:
        st.info("No lyrics classifications yet. Run the pipeline to generate.")
    else:
        import pandas as pd

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            date_options = sorted({e.get("date", "") for e in entries}, reverse=True)
            filter_date = st.selectbox("Date", ["All"] + date_options, key="lc_date")
        with col_f2:
            filter_text = st.text_input("Search title", placeholder="Filter...", key="lc_search")

        filtered = list(reversed(entries))
        if filter_date != "All":
            filtered = [e for e in filtered if e.get("date") == filter_date]
        if filter_text:
            filtered = [e for e in filtered if filter_text.lower() in e.get("title", "").lower()]

        st.caption(f"Showing {len(filtered)} of {len(entries)} classifications")

        # Summary table
        rows = []
        for e in filtered:
            lvi = e.get("lvi", 0)
            rows.append({
                "Date": e.get("date", ""),
                "Title": e.get("title", "")[:55],
                "LVI": round(lvi, 1),
                "Label": e.get("lvi_label", ""),
                "Hook": (e.get("hook_strength") or {}).get("score", ""),
                "Earworm": (e.get("earworm_factor") or {}).get("score", ""),
                "Quotability": (e.get("quotability") or {}).get("score", ""),
                "Takedown Risk": (e.get("takedown_risk") or {}).get("value", ""),
                "Algo Risk": (e.get("algorithm_risk") or {}).get("value", ""),
                "Classifier": e.get("classifier", ""),
            })
        df = pd.DataFrame(rows)

        def _color_lvi(val):
            try:
                v = float(val)
            except (TypeError, ValueError):
                return ""
            if v >= 8:
                return "background-color: #1a7a1a; color: white"
            if v >= 6:
                return "background-color: #4a7a1a; color: white"
            if v >= 4:
                return "background-color: #7a6a1a; color: white"
            return "background-color: #7a1a1a; color: white"

        st.dataframe(
            df.style.applymap(_color_lvi, subset=["LVI"]),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        for e in filtered:
            lvi = e.get("lvi", 0)
            title = e.get("title", "")
            ts = e.get("timestamp", "")[:16].replace("T", " ")
            classifier = e.get("classifier", "")
            icon = "🔥" if lvi >= 8 else "✅" if lvi >= 6 else "⚠️" if lvi >= 4 else "❌"

            with st.expander(f"{icon} LVI {lvi:.1f} — {title}  ({ts})", expanded=False):
                col_l, col_r = st.columns([3, 1])
                with col_r:
                    st.metric("LVI", f"{lvi:.1f} / 10")
                    st.caption(e.get("lvi_label", ""))
                    if classifier:
                        st.caption(f"Classifier: `{classifier}`")
                    if e.get("lvi_gemma3") is not None:
                        st.caption(f"gemma3: {e['lvi_gemma3']:.1f} · grok: {e.get('lvi_grok', 0):.1f}")
                with col_l:
                    if e.get("headline"):
                        st.markdown(f"**Headline:** {e['headline']}")
                    if e.get("verdict"):
                        st.info(f"**Verdict:** {e['verdict']}")

                    # ── Hook Mechanics ────────────────────────────────────────
                    st.markdown("**Hook Mechanics**")
                    h_cols = st.columns(4)
                    h_cols[0].metric("Hook Strength", f"{(e.get('hook_strength') or {}).get('score', 0)}/10")
                    h_cols[0].caption((e.get('hook_strength') or {}).get('rationale', ''))
                    h_cols[1].metric("Earworm", f"{(e.get('earworm_factor') or {}).get('score', 0)}/10")
                    h_cols[1].caption((e.get('earworm_factor') or {}).get('rationale', ''))
                    h_cols[2].metric("Hook Position", (e.get('hook_position') or {}).get('value', ''))
                    h_cols[2].caption((e.get('hook_position') or {}).get('rationale', ''))
                    h_cols[3].metric("Singability", (e.get('singability') or {}).get('value', ''))
                    h_cols[3].caption((e.get('singability') or {}).get('rationale', ''))

                    # ── Cultural Payload ──────────────────────────────────────
                    st.markdown("**Cultural Payload**")
                    c_cols = st.columns(5)
                    c_cols[0].metric("Topicality", (e.get('topicality') or {}).get('value', ''))
                    c_cols[1].metric("Recognition", (e.get('recognition_trigger') or {}).get('value', ''))
                    c_cols[2].metric("Controversy", f"{(e.get('controversy_level') or {}).get('score', 0)}/10")
                    c_cols[3].metric("Satire Type", (e.get('satire_type') or {}).get('value', ''))
                    c_cols[4].metric("In-group", (e.get('ingroup_signal') or {}).get('value', ''))

                    # ── Creator Bait ──────────────────────────────────────────
                    st.markdown("**Creator Bait**")
                    b_cols = st.columns(4)
                    b_cols[0].metric("Visual Hook", (e.get('visual_hook_potential') or {}).get('value', ''))
                    b_cols[1].metric("Meme Format", (e.get('meme_format_fit') or {}).get('value', ''))
                    b_cols[2].metric("Quotability", f"{(e.get('quotability') or {}).get('score', 0)}/10")
                    b_cols[3].metric("Participation", f"{(e.get('participation_hook') or {}).get('score', 0)}/10")

                    # ── Platform Risk ─────────────────────────────────────────
                    st.markdown("**Platform Risk**")
                    r_cols = st.columns(3)
                    r_cols[0].metric("Takedown Risk", (e.get('takedown_risk') or {}).get('value', ''))
                    r_cols[0].caption((e.get('takedown_risk') or {}).get('rationale', ''))
                    r_cols[1].metric("Algorithm Risk", (e.get('algorithm_risk') or {}).get('value', ''))
                    r_cols[1].caption((e.get('algorithm_risk') or {}).get('rationale', ''))
                    sb = e.get('shadowban_words') or {}
                    r_cols[2].metric("Shadowban Words", sb.get('count', 0))
                    if sb.get('words'):
                        r_cols[2].caption(", ".join(sb['words']))


# ── API Logs ─────────────────────────────────────────────────────────────────

if page == "API Logs":
    st.header("API Call Logs")

    entries = _load_table("api_calls", LOGS_DIR / "api_calls.jsonl", "timestamp")

    if not entries:
        st.info("No API logs yet. Run the pipeline to generate logs.")
    else:
            # Filters
            col1, col2 = st.columns(2)
            api_names = sorted(set(e.get("api", "") for e in entries))
            with col1:
                filter_api = st.multiselect("Filter by API", api_names, default=api_names)
            with col2:
                filter_text = st.text_input("Search", placeholder="Search in logs...")

            # Apply filters
            filtered = [e for e in entries if e.get("api", "") in filter_api]
            if filter_text:
                filtered = [e for e in filtered if filter_text.lower() in json.dumps(e).lower()]

            st.caption(f"Showing {len(filtered)} of {len(entries)} entries")

            # Display in reverse chronological order
            for entry in reversed(filtered):
                timestamp = entry.get("timestamp", "")
                api = entry.get("api", "unknown")
                run_dir = entry.get("run_dir", "")

                # Format timestamp
                try:
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    time_str = timestamp

                # Color-code by API type
                if "response" in api:
                    icon = "📥"
                elif "runway" in api:
                    icon = "🎬"
                elif "lyrics" in api:
                    icon = "🎵"
                elif "suno" in api:
                    icon = "🎶"
                else:
                    icon = "📡"

                with st.expander(f"{icon} {api} — {time_str}", expanded=False):
                    # Show key fields prominently
                    if "model" in entry:
                        st.markdown(f"**Model:** `{entry['model']}`")
                    if "base_url" in entry:
                        st.markdown(f"**Provider:** `{entry['base_url']}`")
                    if run_dir:
                        st.markdown(f"**Run:** `{run_dir}`")
                    if "prompt_text" in entry:
                        st.markdown("**Prompt:**")
                        st.code(entry["prompt_text"], language=None)
                    if "user_prompt" in entry:
                        st.markdown("**User Prompt:**")
                        st.code(entry["user_prompt"], language=None)
                    if "raw_response" in entry:
                        st.markdown("**Response:**")
                        st.code(entry["raw_response"], language="json")
                    if "video_url" in entry:
                        st.markdown(f"**Video URL:** `{entry['video_url']}`")

                    # Full JSON
                    st.markdown("**Full entry:**")
                    st.json(entry)


# ── Runs ─────────────────────────────────────────────────────────────────────

elif page == "Runs":
    st.header("All Runs")

    if not OUTPUT_DIR.exists():
        st.info("No output yet.")
    else:
        # Collect all runs
        runs = []
        for day_dir in sorted(OUTPUT_DIR.iterdir(), reverse=True):
            if not day_dir.is_dir() or day_dir.name.startswith("."):
                continue
            subdirs = sorted([d for d in day_dir.iterdir() if d.is_dir()])
            if subdirs:
                for run_dir in subdirs:
                    runs.append(run_dir)
            else:
                # Old flat structure
                if any(day_dir.glob("*.txt")):
                    runs.append(day_dir)

        if not runs:
            st.info("No runs found.")
        else:
            for run_dir in runs:
                headline_file = run_dir / "headline.txt"
                lyrics_file = run_dir / "lyrics.txt"
                final_video = run_dir / "final_captioned.mp4"
                if not final_video.exists():
                    final_video = run_dir / "final.mp4"

                # Extract headline
                headline = ""
                if headline_file.exists():
                    headline = headline_file.read_text(encoding="utf-8").split("\n")[0].strip()

                # Extract title from lyrics
                title = ""
                if lyrics_file.exists():
                    for line in lyrics_file.read_text(encoding="utf-8").split("\n"):
                        if line.startswith("TITLE:"):
                            title = line.replace("TITLE:", "").strip()
                            break

                has_video = final_video.exists()
                status = "✅" if has_video else "⏳"

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"### {status} {title or run_dir.name}")
                    if headline:
                        st.caption(headline)
                    st.markdown(f"`{run_dir}`")
                with col2:
                    if has_video:
                        if st.button("View", key=str(run_dir)):
                            st.session_state["selected_run"] = str(run_dir)
                            st.rerun()

                st.divider()


# ── Run Detail ───────────────────────────────────────────────────────────────

elif page == "Run Detail":
    st.header("Run Detail")

    # Collect all runs for the selector
    all_runs = []
    if OUTPUT_DIR.exists():
        for day_dir in sorted(OUTPUT_DIR.iterdir(), reverse=True):
            if not day_dir.is_dir():
                continue
            subdirs = sorted([d for d in day_dir.iterdir() if d.is_dir()])
            if subdirs:
                all_runs.extend(subdirs)
            elif any(day_dir.glob("*.txt")):
                all_runs.append(day_dir)

    if not all_runs:
        st.info("No runs found.")
    else:
        run_names = [str(r) for r in all_runs]
        default_idx = 0
        if "selected_run" in st.session_state:
            sel = st.session_state["selected_run"]
            if sel in run_names:
                default_idx = run_names.index(sel)

        selected = st.selectbox("Select run", run_names, index=default_idx)
        run_dir = Path(selected)

        # Headline
        headline_file = run_dir / "headline.txt"
        if headline_file.exists():
            st.subheader("Headline")
            st.text(headline_file.read_text(encoding="utf-8").split("\n")[0])

        # Lyrics
        lyrics_file = run_dir / "lyrics.txt"
        if lyrics_file.exists():
            st.subheader("Lyrics")
            st.code(lyrics_file.read_text(encoding="utf-8"), language=None)

        # Video
        final_captioned = run_dir / "final_captioned.mp4"
        final = run_dir / "final.mp4"
        video_file = final_captioned if final_captioned.exists() else final if final.exists() else None
        if video_file:
            st.subheader("Video")
            st.video(str(video_file))

        # Audio
        audio_file = run_dir / "song.mp3"
        if audio_file.exists():
            st.subheader("Audio")
            st.audio(str(audio_file))

        # Timed lyrics
        timed_file = run_dir / "timed_lyrics.json"
        if timed_file.exists():
            st.subheader("Timed Lyrics")
            timed_data = json.loads(timed_file.read_text(encoding="utf-8"))
            for tl in timed_data:
                words = tl.get("words", [])
                word_info = f" ({len(words)} words)" if words else ""
                st.markdown(f"**[{tl['start']:.1f}s - {tl['end']:.1f}s]** {tl['text']}{word_info}")

        # Clips
        clips = sorted(run_dir.glob("clip_*.mp4"))
        if clips:
            st.subheader(f"Clips ({len(clips)})")
            cols = st.columns(min(len(clips), 3))
            for i, clip in enumerate(clips):
                with cols[i % 3]:
                    st.caption(clip.name)
                    st.video(str(clip))

        # Files listing
        st.subheader("All Files")
        for f in sorted(run_dir.iterdir()):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                st.markdown(f"- `{f.name}` ({size_kb:.1f} KB)")
