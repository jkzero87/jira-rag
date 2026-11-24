-- Bronze layer: raw Apache JIRA payloads, untouched by downstream schemas.
-- Idempotent: safe to run repeatedly.
CREATE SCHEMA IF NOT EXISTS bronze;

CREATE TABLE IF NOT EXISTS bronze.jira_field_catalog (
    field_id    text PRIMARY KEY,
    name        text NOT NULL,
    custom      boolean NOT NULL,
    schema_type text,
    fetched_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS bronze.jira_issue_raw (
    issue_key  text PRIMARY KEY,
    issue_id   bigint NOT NULL,
    payload    jsonb NOT NULL,          -- full issue JSON, changelog included
    fetched_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS bronze.ingest_run (
    run_id      bigserial PRIMARY KEY,
    started_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    jql         text NOT NULL,
    pages_done  integer NOT NULL DEFAULT 0,  -- complete pages committed, counted from startAt=0
    issues_done integer NOT NULL DEFAULT 0,
    status      text NOT NULL DEFAULT 'running'
                CHECK (status IN ('running', 'interrupted', 'failed', 'done'))
);

-- Columns added after the first release of this file.
ALTER TABLE bronze.ingest_run ADD COLUMN IF NOT EXISTS retries integer NOT NULL DEFAULT 0;
ALTER TABLE bronze.ingest_run ADD COLUMN IF NOT EXISTS changelogs_refetched integer NOT NULL DEFAULT 0;
ALTER TABLE bronze.ingest_run ADD COLUMN IF NOT EXISTS last_error text;
