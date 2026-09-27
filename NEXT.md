# NEXT.md

## Last changes (this session)

1. **Offline tail-rescue sweep (`scratch/tail_rescue.py`, lists.json only)** —
   final top-10 = vector top-(10-m) + first m keyword hits not already in
   that head (fill from vector if the keyword list runs out). m=0 matched the
   live w=0 reference 40/40. No m beat m=0: m=1 and m=2 tie the baseline
   exactly (both rescue only SPARK-22286 for G03, which is already counted at
   rank 6 there), m=3 loses topic r@10, m=4 also loses lookup r@10.
2. **`search_rescue` added to `retrieve.py`** (m=1, minimal among the
   ties; `search_hybrid` untouched) and **`summary_desc_rescue` added to
   `run_eval.py`**, then one full run_eval pass with all five strategies.

## Current best strategy

`summary_desc_filtered` (live w=0 reference, `eval/cache/w0ref.json`).
`summary_desc_rescue` (m=1) matches it on overall/by-type r@10, r@5 and
MRR in the live run (`2026-09-27_8d3b652_baseline.json`), with 0 questions
worse and 0 better per MRR — so it does not beat the baseline and is NOT
the default.

Per-question, rescue swaps exactly one key (slot 10) on 33/40 questions
(the 7 without a change either have an empty keyword list, or their
cached kw #1 was already the vector slot 10). Two r@10 changes in the
live run: G03 1/3 → 2/3 and G28 2/3 → 1/3. Both contradict the offline
lists.json cache (offline, G03's tail key SPARK-22286 rescues gold and
G28's tail SPARK-46678 does not hurt gold): live, G03's top-10 already
contains SPARK-22286 and its slot-10 swap is 49770 → 46678 (46678 is in
neither cached G03 list — embedding re-run drift), while G28's swap is
22870 → 46678, where 46678 IS the cached G28 kw #1 (so the live keyword
list drifted from the cache, and the gold vector hit SPARK-22870 was not
in the live kw top-10 to be rescued). MRR is untouched in both cases
(first gold rank unchanged), which is why the live m=1 row equals the
baseline exactly.

## Next task (open)

No keyword weight or tail size improves the live baseline. Remaining ideas:
- cap the keyword list more tightly (top 100 by ts_rank_cd is already there;
  maybe the RRF constant k=60 is too small),
- only fuse / rescue when the keyword list actually contains the gold
  (adaptive),
- try `websearch_to_tsquery` (phrase + OR) instead of a pure OR rewrite.
