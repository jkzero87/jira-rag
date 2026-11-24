# Metric definitions

These definitions are the contract for every number this project publishes.
They are built on top of the silver layer (`sql/030_silver.sql`).

## Guiding principle: full grain first, metrics on top

The silver layer keeps the complete grain: one row per issue, nothing
dropped, nothing cleaned away. Rarities and bad values are *flagged*, not
excluded. Every published metric is then computed on top of this full grain,
and every metric must say how many issues it excludes and by which flag.
This order — grain first, numbers on top — is what makes every figure
reproducible from the same base.

## Done

Two numbers are published, never one.

- **"Closed"** = issues with status `Resolved` or `Closed`.
- **"Delivered"** = issues with a resolution of work actually done (the exact
  list is TBD, see status x resolution table).

Both numbers come out of the complete status x resolution table, so the two
figures and the gap between them are always visible together.

## Sub-tasks

- **"Delivered work"** counts only issues that are not sub-tasks.
- **"Closed pieces"** counts everything, sub-tasks included.

Both numbers are published.

## Time

- **Lead time** = `created_at` → first arrival at `Resolved`/`Closed`.
- **Cycle time** = first arrival at `In Progress` → first arrival at
  `Resolved`/`Closed`.
- **Queue time** = lead time − cycle time.

The **median** is reported. Time metrics use the repaired `created_at`
(see `flag_century_repaired` below), never `created_at_raw`.

Issues that never passed through `In Progress` are counted and shown
separately; no start date is invented for them.

## Invalid data

Nothing is deleted. Every issue carries flags, and every published number
states how many issues it excluded and by which flag.

- `flag_impossible_date`, `flag_out_of_project`, and `flag_no_description`
  mark the data; they do not remove rows.
- `flag_century_repaired` marks 289 older SPARK issues whose JIRA `created`
  has a two-digit year (`0010`–`0012`) while their resolution, update and
  changelog dates are `2010`–`2012`. `created_at_raw` keeps the value as JIRA
  returns it; `created_at` adds 2000 years. Every repaired date falls before
  the issue's resolution, last update and first status change, so these
  issues stay in the time metrics.
- `flag_impossible_date` is evaluated on the repaired `created_at`: before
  2010, after `resolved_at`, or in the future. An issue with this flag is
  excluded from the time metrics.
- `flag_no_description` does **not** exclude an issue from the time metrics.
