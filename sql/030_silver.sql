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
    created_at_raw       text NOT NULL,   -- fields.created verbatim, exactly as it comes in
                                          -- the bronze payload
    created_at           timestamptz NOT NULL, -- created_at_raw cast to timestamptz,
                                          -- century restored when repaired
    resolved_at          timestamptz,
    resolved_at_raw      text,            -- fields.resolutiondate verbatim from the bronze payload
    updated_at           timestamptz NOT NULL,
    updated_at_raw       text NOT NULL,   -- fields.updated verbatim from the bronze payload
    date_fix_reason      text,            -- 'century_fix' when any of the issue's dates was
                                          -- repaired; null when nothing was repaired
    flag_century_repaired boolean NOT NULL, -- created_at_raw had a two-digit year
                                            -- (0010-0012 for 2010-2012); created_at adds 2000 years
    flag_impossible_date boolean NOT NULL, -- true when a date had to be repaired (the original
                                            -- value was impossible), or on the repaired created_at: before 2010,
                                            -- after resolved_at, or after now()
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
    created_at_raw, created_at, resolved_at, resolved_at_raw,
    updated_at, updated_at_raw, date_fix_reason,
    flag_century_repaired, flag_impossible_date, flag_out_of_project, flag_no_description
)
WITH d AS (
    SELECT r.issue_key,
           r.payload->'fields'->>'created'                       AS created_raw_text,
           (r.payload->'fields'->>'created')::timestamptz        AS created_raw,
           r.payload->'fields'->>'resolutiondate'                AS resolved_raw_text,
           (r.payload->'fields'->>'resolutiondate')::timestamptz AS resolved,
           r.payload->'fields'->>'updated'                       AS updated_raw_text,
           -- Older imported SPARK issues carry year 00YY in fields.created while
           -- their resolution, update and changelog dates are 20YY.
           extract(year FROM ((r.payload->'fields'->>'created')::timestamptz
                              AT TIME ZONE 'UTC')) < 100           AS repaired
    FROM bronze.jira_issue_raw r
), dd AS (
    SELECT d.*,
           CASE WHEN repaired
                THEN ((created_raw AT TIME ZONE 'UTC') + interval '2000 years') AT TIME ZONE 'UTC'
                ELSE created_raw
           END AS created
    FROM d
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
    dd.created_raw_text,
    dd.created,
    dd.resolved,
    dd.resolved_raw_text,
    (r.payload->'fields'->>'updated')::timestamptz,
    dd.updated_raw_text,
    CASE WHEN dd.repaired THEN 'century_fix' END,
    dd.repaired,
    dd.repaired
     OR dd.created < '2010-01-01'::timestamptz
     OR (dd.resolved IS NOT NULL AND dd.resolved < dd.created)
     OR dd.created > now(),
    r.payload->'fields'->'issuetype'->>'name'
        IN ('IT Help', 'New JIRA Project', 'Blog - New Blog Request'),
    r.payload->'fields'->>'description' IS NULL
     OR btrim(r.payload->'fields'->>'description') = ''
FROM bronze.jira_issue_raw r
JOIN dd USING (issue_key);

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
