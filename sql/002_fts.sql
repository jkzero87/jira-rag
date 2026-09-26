-- FTS: generated tsvector column on jira.issues (summary=A, description=B)
-- plus a GIN index for fast keyword ranking via ts_rank_cd.

ALTER TABLE jira.issues
  ADD COLUMN IF NOT EXISTS fts tsvector
    GENERATED ALWAYS AS (
      setweight(to_tsvector('english', summary), 'A') ||
      setweight(to_tsvector('english', coalesce(description, '')), 'B')
    ) STORED;

CREATE INDEX IF NOT EXISTS issues_fts_gin ON jira.issues USING gin (fts);
