-- Gold layer: published metrics computed on top of silver (docs/metric_definitions.md).
-- Reads only silver; writes only gold. Nothing is deleted from lower layers.
-- Idempotent: drops and rebuilds the gold tables inside one transaction.

BEGIN;

CREATE SCHEMA IF NOT EXISTS gold;

DROP TABLE IF EXISTS gold.metric;
DROP TABLE IF EXISTS gold.issue_timing;

-- One row per silver.issue: done classification and time milestones.
CREATE TABLE gold.issue_timing (
    issue_key         text PRIMARY KEY,
    is_subtask        boolean NOT NULL,
    is_closed         boolean NOT NULL,   -- status Resolved or Closed
    is_delivered      boolean NOT NULL,   -- closed with a work-done resolution
    created_at        timestamptz NOT NULL,
    first_done_at     timestamptz,        -- first transition to Resolved or Closed
    first_started_at  timestamptz,        -- first transition to In Progress
    started_before_done boolean,          -- null without first_done_at
    lead_days         double precision,   -- created_at -> first_done_at
    cycle_days        double precision,   -- first_started_at -> first_done_at, when started before done
    queue_days        double precision,   -- lead_days - cycle_days
    flag_impossible_date boolean NOT NULL,
    date_fix_reason      text,            -- 'century_fix' when an impossible original
                                          -- date was repaired; null when there is no fix
    flag_out_of_project  boolean NOT NULL
);

-- One row per published number, with its denominator and every exclusion.
CREATE TABLE gold.metric (
    metric            text NOT NULL,
    scope             text NOT NULL CHECK (scope IN ('all', 'non_subtask')),
    value             double precision,
    unit              text NOT NULL,
    n_included        integer NOT NULL,   -- issues the value is computed over
    n_excl_out_of_project  integer NOT NULL,
    n_excl_impossible_date integer NOT NULL,
    n_excl_not_applicable  integer NOT NULL, -- e.g. never done, never started before done
    note              text,
    PRIMARY KEY (metric, scope)
);

INSERT INTO gold.issue_timing
WITH t AS (
    SELECT issue_key,
           min(changed_at) FILTER (WHERE to_status IN ('Resolved', 'Closed')) AS first_done_at,
           min(changed_at) FILTER (WHERE to_status = 'In Progress')          AS first_started_at
    FROM silver.status_transition
    GROUP BY issue_key
)
SELECT i.issue_key,
       i.is_subtask,
       i.status IN ('Resolved', 'Closed'),
       i.status IN ('Resolved', 'Closed')
         AND i.resolution IN ('Fixed', 'Done', 'Implemented', 'Resolved'),
       i.created_at,
       t.first_done_at,
       t.first_started_at,
       CASE WHEN t.first_done_at IS NOT NULL
            THEN t.first_started_at IS NOT NULL AND t.first_started_at <= t.first_done_at END,
       extract(epoch FROM t.first_done_at - i.created_at) / 86400,
       CASE WHEN t.first_started_at <= t.first_done_at
            THEN extract(epoch FROM t.first_done_at - t.first_started_at) / 86400 END,
       CASE WHEN t.first_started_at <= t.first_done_at
            THEN extract(epoch FROM t.first_started_at - i.created_at) / 86400 END,
       i.flag_impossible_date,
       i.date_fix_reason,
       i.flag_out_of_project
FROM silver.issue i
LEFT JOIN t USING (issue_key);

-- Counts: denominators exclude only flag_out_of_project.
INSERT INTO gold.metric
SELECT m.metric, s.scope, m.value, m.unit, m.n_included, m.n_oop, 0, 0, m.note
FROM (VALUES ('all'), ('non_subtask')) s(scope)
CROSS JOIN LATERAL (
    SELECT count(*) FILTER (WHERE NOT flag_out_of_project) AS n,
           count(*) FILTER (WHERE flag_out_of_project)     AS n_oop,
           count(*) FILTER (WHERE NOT flag_out_of_project AND is_closed)    AS closed,
           count(*) FILTER (WHERE NOT flag_out_of_project AND is_delivered) AS delivered
    FROM gold.issue_timing
    WHERE s.scope = 'all' OR NOT is_subtask
) c
CROSS JOIN LATERAL (VALUES
    ('closed_count',    c.closed::double precision,    'issues', c.n, c.n_oop, 'status Resolved or Closed'),
    ('delivered_count', c.delivered::double precision, 'issues', c.n, c.n_oop,
        'closed with resolution Fixed, Done, Implemented or Resolved'),
    ('closed_share',    100.0 * c.closed / c.n,    'percent', c.n, c.n_oop, 'closed_count / issues in scope'),
    ('delivered_share', 100.0 * c.delivered / c.n, 'percent', c.n, c.n_oop, 'delivered_count / issues in scope'),
    ('delivered_of_closed', 100.0 * c.delivered / NULLIF(c.closed, 0), 'percent', c.closed, c.n_oop,
        'delivered_count / closed_count; n_included is closed_count')
) m(metric, value, unit, n_included, n_oop, note);

-- Time medians: exclude flag_out_of_project, then impossible dates that were
-- not repaired (flag_impossible_date with date_fix_reason NULL), then issues
-- where the metric does not apply.
INSERT INTO gold.metric
SELECT m.metric, s.scope, m.value, 'days', m.n_inc, b.n_oop, b.n_imp, b.n_base - m.n_inc, m.note
FROM (VALUES ('all'), ('non_subtask')) s(scope)
CROSS JOIN LATERAL (
    SELECT count(*) FILTER (WHERE flag_out_of_project) AS n_oop,
           count(*) FILTER (WHERE NOT flag_out_of_project AND flag_impossible_date
                             AND date_fix_reason IS NULL)               AS n_imp,
           count(*) FILTER (WHERE NOT flag_out_of_project AND NOT (flag_impossible_date
                             AND date_fix_reason IS NULL))             AS n_base
    FROM gold.issue_timing WHERE s.scope = 'all' OR NOT is_subtask
) b
CROSS JOIN LATERAL (
    SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY lead_days)  AS lead,
           count(lead_days)                                        AS n_lead,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY cycle_days) AS cyc,
           count(cycle_days)                                       AS n_cyc,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY queue_days) AS que,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY lead_days)
               FILTER (WHERE started_before_done = false)          AS lead_unstarted,
           count(*) FILTER (WHERE started_before_done = false)     AS n_unstarted
    FROM gold.issue_timing
    WHERE (s.scope = 'all' OR NOT is_subtask) AND NOT flag_out_of_project
      AND NOT (flag_impossible_date AND date_fix_reason IS NULL)
) x
CROSS JOIN LATERAL (VALUES
    ('lead_time_median',  x.lead, x.n_lead, 'created_at -> first Resolved/Closed; issues never done not applicable'),
    ('cycle_time_median', x.cyc,  x.n_cyc,  'first In Progress -> first Resolved/Closed; only issues started before first done'),
    ('queue_time_median', x.que,  x.n_cyc,  'created_at -> first In Progress, same issues as cycle_time_median'),
    ('lead_time_median_never_started', x.lead_unstarted, x.n_unstarted,
        'lead time of done issues with no In Progress before first done, reported separately')
) m(metric, value, n_inc, note);

COMMIT;
