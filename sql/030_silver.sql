-- Silver layer: clean grain, one row per issue, built only from bronze.jira_issue_raw.
-- Project rule: nothing is ever deleted. Rarities are kept and marked with
-- boolean flags (flag_*), never excluded.
-- Idempotent: drops and rebuilds the silver tables, so the script can be run
-- again and again without breaking anything.

BEGIN;

CREATE SCHEMA IF NOT EXISTS silver;

-- Drop children first, then the parent, so constraints never block a re-run.
DROP TABLE IF EXISTS silver.issue_component;
DROP TABLE IF EXISTS silver.status_transition;
DROP TABLE IF EXISTS silver.field_catalog;
DROP TABLE IF EXISTS silver.issue;

-- One row per issue_key: the clean grain.
CREATE TABLE silver.issue (
    issue_key            text PRIMARY KEY,
    issue_id             bigint NOT NULL,
    issue_type           text NOT NULL,
    is_subtask           boolean NOT NULL,
    parent_key           text,            -- null when the issue is not a sub-task
    status               text NOT NULL,
    resolution           text,            -- null when the issue has none
    priority             text,
    created_at           timestamptz NOT NULL,
    resolved_at          timestamptz,
    updated_at           timestamptz NOT NULL,
    flag_impossible_date boolean NOT NULL, -- created before 2010, resolved before
                                            -- created, or created after now()
    flag_out_of_project  boolean NOT NULL, -- issue type outside the project's own types
    flag_no_description  boolean NOT NULL  -- description null or whitespace-only
);

-- One row per (issue_key, component) pair.
CREATE TABLE silver.issue_component (
    issue_key text NOT NULL,
    component text NOT NULL,
    PRIMARY KEY (issue_key, component)
);

-- One row per changelog item with field = 'status'; changed_at is the
-- timestamp of the history entry containing the change.
CREATE TABLE silver.status_transition (
    issue_key   text NOT NULL,
    changed_at  timestamptz NOT NULL,
    from_status text NOT NULL,
    to_status   text NOT NULL
);

-- Copy of bronze.jira_field_catalog plus is_automatic, true ONLY for the
-- system-maintained field names listed in the project metric definitions.
CREATE TABLE silver.field_catalog (
    field_id     text PRIMARY KEY,
    name         text NOT NULL,
    custom       boolean NOT NULL,
    schema_type  text,
    fetched_at   timestamptz NOT NULL,
    is_automatic boolean NOT NULL
);

INSERT INTO silver.issue (
    issue_key, issue_id, issue_type, is_subtask, parent_key,
    status, resolution, priority,
    created_at, resolved_at, updated_at,
    flag_impossible_date, flag_out_of_project, flag_no_description
)
SELECT
    r.issue_key,
    (r.payload->>'id')::bigint,
    r.payload->'fields'->'issuetype'->>'name',
    r.payload->'fields'->'issuetype'->>'name' = 'Sub-task',
    CASE WHEN r.payload->'fields'->'issuetype'->>'name' = 'Sub-task'
         THEN r.payload->'fields'->'parent'->>'key'
    END,
    r.payload->'fields'->'status'->>'name',
    r.payload->'fields'->'resolution'->>'name',
    r.payload->'fields'->'priority'->>'name',
    (r.payload->'fields'->>'created')::timestamptz,
    (r.payload->'fields'->>'resolutiondate')::timestamptz,
    (r.payload->'fields'->>'updated')::timestamptz,
    (r.payload->'fields'->>'created')::timestamptz < '2010-01-01'::timestamptz
     OR ((r.payload->'fields'->>'resolutiondate')::timestamptz IS NOT NULL
         AND (r.payload->'fields'->>'resolutiondate')::timestamptz
             < (r.payload->'fields'->>'created')::timestamptz)
     OR (r.payload->'fields'->>'created')::timestamptz > now(),
    r.payload->'fields'->'issuetype'->>'name'
        IN ('IT Help', 'New JIRA Project', 'Blog - New Blog Request'),
    r.payload->'fields'->>'description' IS NULL
     OR btrim(r.payload->'fields'->>'description') = ''
FROM bronze.jira_issue_raw r;

INSERT INTO silver.issue_component (issue_key, component)
SELECT DISTINCT
    r.issue_key,
    c->>'name'
FROM bronze.jira_issue_raw r
     CROSS JOIN LATERAL jsonb_array_elements(r.payload->'fields'->'components') c
WHERE r.payload->'fields'->'components' IS NOT NULL;

INSERT INTO silver.status_transition (issue_key, changed_at, from_status, to_status)
SELECT
    r.issue_key,
    (h->>'created')::timestamptz,
    i->>'fromString',
    i->>'toString'
FROM bronze.jira_issue_raw r
     CROSS JOIN LATERAL jsonb_array_elements(r.payload->'changelog'->'histories') h
     CROSS JOIN LATERAL jsonb_array_elements(h->'items') i
WHERE i->>'field' = 'status';

INSERT INTO silver.field_catalog
    (field_id, name, custom, schema_type, fetched_at, is_automatic)
SELECT
    field_id,
    name,
    custom,
    schema_type,
    fetched_at,
    name IN ('Rank', 'Global Rank', 'Rank (Obsolete)', 'Attachment count',
             'Last public comment date')
FROM bronze.jira_field_catalog;

COMMIT;
