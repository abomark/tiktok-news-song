-- CurrentNoise — Supabase schema
-- Run this once in the Supabase SQL editor before migrating.
-- All tables use (headline, date) as the natural dedup key where applicable.

-- ── news_candidates ───────────────────────────────────────────────────────────
-- Source: logs/news_candidates.jsonl

CREATE TABLE IF NOT EXISTS news_candidates (
    headline        TEXT        NOT NULL,
    date            DATE        NOT NULL,
    fetched_at      TIMESTAMPTZ,
    published_at    TIMESTAMPTZ,
    summary         TEXT,
    source          TEXT,
    url             TEXT,

    PRIMARY KEY (headline, date)
);


-- ── social_scores ─────────────────────────────────────────────────────────────
-- Source: logs/social_scores.jsonl

CREATE TABLE IF NOT EXISTS social_scores (
    headline        TEXT        NOT NULL,
    date            DATE        NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    source          TEXT,
    social_score    NUMERIC,
    reddit_score    NUMERIC,
    hn_score        NUMERIC,
    trends_score    NUMERIC,

    PRIMARY KEY (headline, date)
);


-- ── story_classifications ─────────────────────────────────────────────────────
-- Source: logs/story_classifications.jsonl
-- Each VPI factor stored as JSONB: {"score": int, "rationale": str}

CREATE TABLE IF NOT EXISTS story_classifications (
    headline                TEXT        NOT NULL,
    date                    DATE        NOT NULL,
    timestamp               TIMESTAMPTZ NOT NULL,
    summary                 TEXT,
    source                  TEXT,
    url                     TEXT,
    run_dir                 TEXT,
    angle                   TEXT,
    vpi                     NUMERIC,
    vpi_label               TEXT,

    -- 10 VPI factors
    absurdity               JSONB,
    character_punchability  JSONB,
    cultural_reach          JSONB,
    emotional_heat          JSONB,
    memeability             JSONB,
    musical_fit             JSONB,
    timestamp_sensitivity   JSONB,
    moral_clarity           JSONB,
    visual_potential        JSONB,
    safe_harbor             JSONB,

    PRIMARY KEY (headline, date)
);


-- ── flagged_stories ───────────────────────────────────────────────────────────
-- Source: logs/flagged_stories.jsonl

CREATE TABLE IF NOT EXISTS flagged_stories (
    headline        TEXT        NOT NULL,
    date            DATE        NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    source          TEXT,
    url             TEXT,
    summary         TEXT,
    angle           TEXT,
    combined_score  NUMERIC,
    social_score    NUMERIC,
    vpi             NUMERIC,
    vpi_label       TEXT,
    threshold       NUMERIC,

    PRIMARY KEY (headline, date)
);


-- ── selection_decisions ───────────────────────────────────────────────────────
-- Source: logs/selection_decisions.jsonl
-- Handles both old schema (winner_headline) and new schema (flagged_headlines).
-- candidates is a JSONB array with per-candidate scores, norms, angle, flagged.

CREATE TABLE IF NOT EXISTS selection_decisions (
    date                    DATE        NOT NULL,
    timestamp               TIMESTAMPTZ NOT NULL,
    threshold               NUMERIC,

    -- new schema
    flagged_headlines       TEXT[],
    n_flagged               INTEGER,

    -- old schema (pre-refactor rows) — nullable
    winner_headline         TEXT,
    winner_combined_score   NUMERIC,
    winner_vpi              NUMERIC,
    winner_social_score     NUMERIC,

    -- full ranked candidate list
    candidates              JSONB,

    PRIMARY KEY (date, timestamp)
);


-- ── api_calls ─────────────────────────────────────────────────────────────────
-- Source: logs/api_calls.jsonl
-- Payload varies by api type so non-key fields stored as JSONB.

CREATE TABLE IF NOT EXISTS api_calls (
    id          BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL,
    api         TEXT        NOT NULL,
    run_dir     TEXT,
    payload     JSONB,

    UNIQUE (timestamp, api, run_dir)
);


-- ── Indexes ───────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_social_scores_date     ON social_scores (date);
CREATE INDEX IF NOT EXISTS idx_classifications_date   ON story_classifications (date);
CREATE INDEX IF NOT EXISTS idx_flagged_date           ON flagged_stories (date);
CREATE INDEX IF NOT EXISTS idx_decisions_date         ON selection_decisions (date);
CREATE INDEX IF NOT EXISTS idx_api_calls_ts           ON api_calls (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_api_calls_api          ON api_calls (api);
