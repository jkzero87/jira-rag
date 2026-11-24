# Metric definitions

These definitions are the contract for every number this project publishes.
They are built on top of the silver layer (`sql/030_silver.sql`) and computed
in the gold layer (`sql/040_gold.sql`, table `gold.metric`); the current
numbers are in `docs/metrics.md`.

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
- **"Delivered"** = closed issues whose resolution is `Fixed`, `Done`,
  `Implemented` or `Resolved`. Every other resolution (`Duplicate`,
  `Won't Fix`, `Invalid`, `Incomplete`, `Not A Problem`, `Cannot Reproduce`,
  `Later`, and the rest of the status x resolution table) is closed but not
  delivered.

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
- **Queue time** = lead time − cycle time, per issue (`created_at` → first
  arrival at `In Progress`).

The **median** is reported. Each median is taken over per-issue values, so
the median queue time is not the difference of the other two medians.

- Lead time applies to every issue that ever reached `Resolved`/`Closed`,
  including issues reopened since (182 at the time of writing).
- Cycle and queue time apply only when the first `In Progress` comes before
  the first `Resolved`/`Closed`. An issue that reached `In Progress` only after
  its first `Resolved`/`Closed` counts with those that never passed through
  `In Progress`.
- In SPARK, 91.8% of transitions to `In Progress` are made by the
  `apachespark` account, which moves an issue when a pull request linked to it
  is opened. `In Progress` therefore means "a pull request exists": cycle time
  is pull request to first `Resolved`/`Closed`, and queue time is issue
  creation to first pull request. Most issues are filed together with their
  pull request, which is why the median queue time is under an hour. Time metrics use the repaired `created_at`
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
- `flag_out_of_project` excludes an issue from every metric (denominators
  included): those issue types are JIRA administration requests, not SPARK
  work.
