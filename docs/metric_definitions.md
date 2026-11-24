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

The **median** is reported.

Issues that never passed through `In Progress` are counted and shown
separately; no start date is invented for them.

## Invalid data

Nothing is deleted. Every issue carries flags, and every published number
states how many issues it excluded and by which flag.

- `flag_impossible_date`, `flag_out_of_project`, and `flag_no_description`
  mark the data; they do not remove rows.
- `flag_no_description` does **not** exclude an issue from the time metrics.
