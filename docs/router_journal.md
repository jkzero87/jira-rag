# Issue router — research journal

This document records how the issue router (chapter 1 of jira-rag) is being built: the
problem, the decisions and why they were taken, every result (including negative ones), the
method mistakes that were caught and corrected, and the research that shapes the next steps.
It is a living log: each experiment adds an entry. Generated numbers live in
`docs/router_results.md` (written by `scripts/router_text_baseline.py`); this journal explains them.

## 1. Why this exists

The project narrative comes from real work, not from a job posting:

- **Amadeus (L2 support):** the hardest daily problem was deciding which L3 team should get
  each ticket. Triage meant searching the logs (by time and other data) for a letter code that
  identified the failing service, then opening the escalation in ServiceNow. Tickets without
  logs were sent back asking for more information.
- **EA (data):** the need is an agent over a data warehouse that answers exact figures fast.

jira-rag already had a cited-answer RAG over Apache Spark JIRA issues (retrieval recall@10
0.55 → 0.78, 39/40 answers correct, 0 invented citations) and an API layer (FastAPI, API keys,
rate limits, CI, image on ghcr.io). Chapter 1 adds the Amadeus problem on public data:
**given a new Spark issue, which component should own it?**

## 2. Data foundation (bronze / silver / gold)

All data lives in the single Postgres instance `de_postgres` (port 5432). Decisions:

- **Grain first, figures on top.** Get the full grain, then build metrics over it. Show the
  information exists (status × resolution, with and without sub-tasks) instead of picking one
  headline number.
- **Flag, never delete.** Invalid data gets flags in silver; corrected values (e.g. dates with a
  lost century) keep the original value, metadata and a trail.
- The ambiguous resolution "Resolved" is reported as its own row, outside "Delivered".
- Lead, cycle and queue time use the **median**.

## 3. Chapter 1 setup

- **Task:** text classifier over all issues → component. We did not force the Amadeus shape
  ("let's use what best exploits the data"); error signals are kept as a separate hint.
- **Time split (no peeking):** train < 2023-01-01, **validation = 2023**, test ≥ 2024.
  Every choice (hyper-parameters, variant, blend weight) is made on validation only. Test
  numbers are reference only.
- **Leakage guard for parent context:** sub-tasks use the parent's components **as of the
  sub-task's creation date** (`parent_components_asof`), never the parent's current components
  (`parent_components_now`). The `_now` column is read only to measure how much leakage it
  would have caused.

## 4. Results so far

### 4.1 Variants (2023 validation, top-1, n = 4733)

| Variant | What it is | Val top-1 |
|---|---|---|
| level 1 (C=8) | TF-IDF + logistic regression on issue text | 61.7% |
| V-rule | sub-tasks predict the parent's as-of components (ranked by train frequency); others = level 1 | 66.1% |
| **V-token (C=8) — selected** | level 1 + a token per parent as-of component (e.g. `__parentcomp_SQL`) | **68.7%** |
| V-token+title (C=8) | V-token + the parent's summary in the text | 67.0% |

Reference on test (not used to choose): V-token 66.5% top-1 / 89.3% top-3 (sub-tasks 71.0% /
91.2%, non-sub-tasks 62.6% / 87.7%).

### 4.2 Per component (test, reference only)

Strong: SQL (~92%), Build (~81%). Weak: **Declarative Pipelines (~5%)**, Pandas API on Spark
(~27%), Optimizer (~29%) — new or low-history components. The router **ages**: it cannot know
components that did not exist when it was trained.

### 4.3 Negative result: blending the parent rule into V-token (PR #14)

Hypothesis: V-rule and V-token win on different components, so a weighted blend
`p = (1-w)·p_model + w·p_rule` (one w for all, chosen on validation) should beat both.

Result on 2023 validation: best **w = 0.0** (68.7%); accuracy never rises as w grows (it is flat between 0.8 and 1.0)
(0.3 → 68.3%, 0.5 → 65.9%, 0.9 → 64.9%). Blending adds nothing: V-token already absorbs what
the rule knows. **V-token stays the selected variant.**

Bug fixed before merging PR #14: at w = 1.0 validation read 24.9%, which was impossible - the 3051 no-parent rows alone keep V-token's 62.3%. Verified cause: on rows where p_rule has no information, `blend_rank` still scaled the V-token probabilities by (1 - w); at w = 1.0 every score was exactly 0, the whole ranking tied and the first model class won. (The suspected uniform p_rule for parents whose as-of components were absent from train < 2023 never triggered: the 2023 slice has zero such rows, and a run with only that fix left the w grid unchanged.) Fix: `rule_weights` now returns `None` instead of a uniform spread and `blend_rank` keeps the plain model ranking on those rows, so they use V-token unchanged for every w (asserted for every w on every such row). At w = 1.0 validation now reads 64.9% (V-rule: 66.1%), and the chosen w is still 0.0.

### 4.4 Experiment 1: route only when confident

Hypothesis: allow the router to auto-route only when it is confident; a confidence
threshold then trades coverage for precision (plan, section 7, experiment 1).
Method: two signals from the V-token `predict_proba` — the top-1 probability and
the margin (top1 minus the second-highest) — are thresholded on a 0.05..0.95 grid.
Thresholds are chosen on the 2023 validation slice only; test is reference.
Full tables: [docs/router_confidence.md](docs/router_confidence.md) (generated by
`scripts/router_confidence.py`, byte-identical across runs).

Results (2023 validation, n = 4733; overall top-1 is 68.7%):

- 90% precision target: only the **margin** signal reaches it — t = 0.95, coverage
  8.9%, precision 94.0%. Top-1 never reaches 90% at any grid threshold (best
  88.4% at t = 0.95).
- 80% precision target: margin t = 0.50, coverage 54.4%, precision 80.8% (top-1:
  t = 0.70, coverage 61.5%).

Test reference, same thresholds, no re-selection: margin 90% → 8.8% coverage at
92.9% precision; 80% → 52.7% coverage at 79.0%. The below-threshold rest
(4314 validation / 14369 test issues) has 86.6% (val) / 88.7% (test) top-3
accuracy: a human sees most issues, but with at most three candidates.

Sub-tasks are both more routable and more accurate at the 90% margin threshold
(validation coverage 10.7% at 98.3% precision vs 7.8% at 90.8% for
non-sub-tasks). Per component (test, margin t = 0.95): Spark Core is the
confidently-wrong outlier — 51 wrong auto-routes at 21.5% auto precision. New
components are almost never auto-routed there, and what is auto-routed is all
wrong (Declarative Pipelines 3/206 → SQL, Spark Core; Pandas API on Spark
6/276 → PySpark; Optimizer 9/84 → SQL) — the old-component confusions the
novelty detector (plan, section 7, experiment 4) targets.

Update after experiment 7: with V-token+balanced (C = 8, selected in 4.6), the raw confidently-wrong counts of the two variants are not comparable, because at each variant's 90% threshold they auto-route different volumes (test coverage 8.8% vs 15.0%, +6.2 pp; at the 80% threshold 52.7% vs 49.6%, -3.1 pp; n = 15754 on test). Normalized as wrong per 100 auto-routed (test, each variant's 90% threshold; rows grouped by true component): 7.1 for V-token vs 9.7 for V-token+balanced overall; per component, SQL 0.1 (1 wrong of 1110 auto-routed) vs 0.6 (9 of 1583), Spark Core 78.5 (51 of 65) vs 65.3 (96 of 147), Connect 36.7 (11 of 30) vs 40.2 (47 of 117) — the extra coverage at 90% raises the wrong rate overall, and most for Connect (+3.5 per 100). Full tables: the V-token+balanced section of docs/router_confidence.md.

### 4.5 Experiment 5: the reassignment ("tossing") graph

Source: `scripts/router_tossing.py` -> [docs/router_tossing.md](docs/router_tossing.md).
Each issue's initial components are reconstructed (current list with every
Component changelog entry after created_at undone, the same `_asof` logic as the
parent context) and compared with the final components — the label the router
uses.

**How often do people toss?** Real tosses (initial != final, initial non-empty):
train 3120/40018 (7.8%), validation 181/4733
(3.8%), test 834/15754 (5.3%);
train additionally has 1762 (4.4%) late assignments
(a component set after creation, not a toss). People rarely change a component
once it is set.

**Who tosses what (train, <2023)?** Top pairs:
Spark Core -> SQL (155), SQL -> Structured Streaming (96), R -> SparkR (92), DStreams -> Structured Streaming (77), docs -> Documentation (76). Several top pairs are
component *renames* (R -> SparkR, docs -> Documentation, MLlib -> ML), counted
as tosses because the label changed. 9 of the top-20 pairs
involve Spark Core.

**Do the router's errors look like people's tosses (2023 validation, CV model)?**
Only 8 of 1483 top-1 errors (0.5%) predict a
component the person had set at creation and later removed. The base rate (an
error uniformly random among the model's 34 components) is 0.03%
(16 of 50422 component draws): the router does not
systematically repeat people's tosses, and people's tosses are too rare to be
the router's failure mode. Only 3 of the model's top-15 confusion pairs
are also in the people top-20 (Spark Core -> SQL, PySpark -> SQL, SQL ->
Spark Core).

**The Spark Core exception (test, reference only).** All 51 confidently
wrong auto-routes at margin >= 0.95 (experiment 1) land on (Spark Core ->
predicted) pairs the people have actually tossed: 51/51
match the full people graph, 51/51 the people top-20
(Spark Core -> SQL is the #1 people toss). For this one component the
router's confident errors *are* the people's toss pattern, so a toss-based
prior or rule-based correction could in principle catch all of them.

### 4.6 Experiment 7: label hygiene (renames, multi-component, rare classes)

Source: `scripts/router_label_hygiene.py` -> [docs/router_label_hygiene.md](docs/router_label_hygiene.md).
All selection on the 2023 slice (fit on < 2023, top-1 on 2023, C grid
{0.5, 2, 8}); test is reference only.

**Renames (evidence strictly before 2023-01-01).** Initial components are
the same `components_asof` reconstruction as 4.5. A rename X -> Y is
supported only if X is retired (no issue created in 2022 has X in its
initial list), Y is current (some 2022 issue has Y in its initial list),
and at least 5 X -> Y replacement events fall inside a 60-day window.
The mapping is `{MLilb: MLlib, docs: Documentation}`; it is robust -
unchanged for every whole-day retirement threshold between 83 and 427
days before 2023-01-01. The pairs flagged in 4.5 do not pass: R and MLlib
are still used on new 2022 issues (last new R issue 2022-11-25, last new
MLlib issue 2022-11-27), and `ps` -> `Pandas API on Spark` has only 2
replacement events before the cutoff; of the 40 pre-2023 issues that
started with `ps` and ended with `Pandas API on Spark`, 38 completed the
swap only after 2023-01-01 - an in-flight rename, not applied because the
pre-cutoff evidence is insufficient. Remapping the training labels
(`V-token+renames`) does not move the validation top-1 (68.7%,
3250/4733 at C = 8, exactly the V-token number).

**Multi-component.** Issues with >= 2 final components: train
7302/40018 (18.2%), validation 1109/4733 (23.4%), test 2435/15754 (15.5%).
The current scoring (`hit_counts`) already counts a top-1 hit when the
prediction is any of the final components - the two definitions are
identical by construction.

**Rare classes.** `class_weight="balanced"` on the same pipeline (best
C = 8) raises the validation top-1 from 68.7% (3250/4733) to 69.7%
(3298/4733) and is selected as **V-token+balanced**. The cost, on
validation: SQL errors (rows whose final components include SQL; an error
when top-1 is not SQL) go from 170 (1302/1472) to 275 (1197/1472); balanced
also hurts SQL (93.5% -> 88.2%) and Spark Core (61.2% -> 55.4%) per
component, while it helps most smaller components (Documentation
71.9% -> 79.5%, PySpark 76.8% -> 80.8%, Kubernetes 71.8% -> 83.5%,
Web UI 49.5% -> 60.2%). Test reference (never used for selection):
V-token+balanced top-1 68.2% (10737/15754), top-3 89.5%
(14096/15754) vs V-token 66.5% (10479/15754) and 89.3% (14070/15754). Since experiment 7, V-token+balanced is the selected variant: experiment 1 has been re-measured with it (see the 'Update after experiment 7' paragraph in 4.4).

### 4.7 Experiment 3: aging and recency

Source: `scripts/router_recency.py` -> [docs/router_recency.md](docs/router_recency.md).
All variants are V-token+balanced with C = 8 (from 4.6); every selection
(recency half-life, margin thresholds, variant) is made on the 2023
validation slice, test reference only. The aging baseline `static` is
trained once on created_at < 2023-01-01 (40018 issues) - one year older
than the selected final V-token+balanced model of 4.6, which was trained
on < 2024-01-01.

**Label rule.** The as-of builder already existed (`components_asof` in
`scripts/router_parent_context.py:58`, used in 4.5), so for each month M
the training labels are the components as known on the first day of M
(final list with every Component changelog entry on/after that day
undone), and issues whose as-of list is empty are dropped. Ground truth
for the issues created in M is their FINAL component list, as in every
other experiment; the RESOLVED-before fallback rule was not needed.

**Results (top-1).** `static`: 69.7% (3298/4733) on validation - the
documented V-token+balanced number, asserted in code - and 67.1%
(10569/15754) on test. For reference, the selected final model (trained
on < 2024-01-01) scores 68.2% (10737/15754) on the same test slice
([docs/router_label_hygiene.md](docs/router_label_hygiene.md), line 145).
Monthly retraining (`rolling`) reaches 73.1% (3462/4733) on validation and
73.5% (11579/15754) on test: +1010 hits (+6.4 pp) over `static` and +842
hits (+5.3 pp) over the final < 2024 model. `rolling` leads `static` in
all 12 test quarters; the gap is widest in 2026Q4 (57.9% vs 69.7%,
+11.8 pp). Recency weighting alone (`recency`) does not help: 69.7%
(3297/4733) on validation, 66.0% (10392/15754) on test, below `static`.
The three half-life values (h = 1, 2, 4) score 3297, 3295, 3289
validation hits - they differ by at most 8 hits - and the chosen h = 1
is the smallest value of the grid. `rolling+recency` is slightly worse
than `rolling` on validation (72.9%) but best on test (74.3%,
11700/15754); `rolling` is selected on validation. At the 90%-precision
auto-routing threshold (chosen on validation), `rolling` auto-routes
26.9% of test at 92.2% precision vs 16.3% (88.0%) for `static`; at the
80% target, 62.4% at 82.3% vs 52.1% (77.9%).

## 5. Method mistakes caught (and the rule each one produced)

1. **"Best variant per component" chosen on test.** An early per-component table picked the
   best variant for each row looking at test. Removed. Rule: any "best" is computed on
   validation only; test tables mark the validation-selected variant and say test was not used.
2. **The blend hypothesis came from the test table.** "The rule wins on Kubernetes (84.7% vs
   58.4%)" was a test number. Validation said the blend does not help. The method stopped a
   conclusion drawn from the exam. Rule: hypotheses may come from anywhere, but only
   validation can confirm them.
3. **Parent leakage.** Using the parent's current components would leak the future. Rule:
   always as-of the event's own timestamp.
4. **Reproducibility.** Every result doc must regenerate byte-identical on two full runs.

## 6. Research: why we are going in circles, and better approaches

So far every variant is the same idea: one text classifier forced to give **one** answer for
**every** issue, in a **closed** set of components. That framing has hit its ceiling (~68%).
The state of the art changes the question rather than tuning the model. Six findings:

### 6.1 Route only when confident (abstention)
Production systems do not try to be right on everything. Mozilla's BugBug assigns a component
only above a confidence threshold (60% at the time), reaching > 80% precision. Chinese
industrial ticket routing reports the same pattern: high-confidence tickets go straight to
dispatch, the rest go to human confirmation. This is exactly the Amadeus reality. Reporting
"X% auto-routed at 90% precision, the rest to a human, with top-3 suggestions" is more useful
than a single top-1 figure, and needs no new model.

### 6.2 Two memories (bio-inspired: complementary learning systems)
The brain uses a fast hippocampal memory (learns from a few examples) and a slow cortical one
(generalizes). Our model is only the "cortex", so it fails on new components. The
"hippocampus" is retrieval: nearest past issues (kNN over embeddings) and per-component
**prototypes** (mean embeddings). A prototype classifier adds a new class by computing one more
mean, without retraining. Chinese government ticket systems use the same idea: retrieve the
top-5 similar historical tickets and recommend "department X; 87% of similar cases in the last
3 months were handled there".

### 6.3 Detect the new instead of forcing it (bio-inspired: dentate gyrus)
The dentate gyrus, the hippocampus's entry, is associated with novelty detection and pattern
separation. The ML counterpart is open-world recognition: reject unknowns → discover new
classes → learn them continually. For the router: an issue far from every known component
should be flagged as "possibly new", not forced into SQL. This targets Declarative Pipelines.

### 6.4 Recency (forgetting curve)
Component prediction work at scale reports new components almost every month and handles
concept drift with recency-based sampling; Microsoft's DeepTriage retrains frequently. Old
data should weigh less, and the retraining cadence is itself an experiment.

### 6.5 Learn from mistakes: the reassignment (tossing) graph
JIRA records when an issue's component was changed. A bug-tossing knowledge graph built from
those changes improved component assignment by > 20% top-k over BugBug, because confusing
component designs and misleading names mislead reporters, developers and tools alike. The
Spark changelog gives us the real confusing pairs instead of guessing them.

### 6.6 The LLM as tie-breaker, not as classifier
Zero-shot LLMs consistently underperform fine-tuned models on domain taxonomies, and kNN
embedding pipelines beat zero-shot LLM classification. Where the LLM helps is unseen classes:
conditioning on category definitions recovered categories a fine-tuned BERT could never
predict. So the local 27B should only see low-confidence cases, choosing among the top-3
candidates, with component descriptions and boundary rules for the confusing pairs (6.5).

### 6.7 Considered and discarded
Ant-colony and bee-colony methods in bug triage are mostly feature weighting or developer load
balancing, not component prediction; they do not address our failure modes. The bio-inspired
ideas that do apply are about memory: two memories, novelty detection, forgetting.

## 7. Plan (each step measured on 2023 validation, test as reference)

| # | Experiment | Targets | Who |
|---|---|---|---|
| 1 | Confidence → precision curve (coverage at 90% precision) and top-3 (done, see 4.4) | reframes the task; baseline for the rest | local agent |
| 2 | Hippocampus: embedding kNN + per-component prototypes, combined with V-token on validation | new and rare components | Claude Code (needs GPU) |
| 3 | Recency weighting and monthly retraining with rolling evaluation (done, see 4.7) | aging | local agent |
| 4 | Novelty detector: does it catch the first Declarative Pipelines issues? | new components forced into old ones | local agent |
| 5 | Tossing graph from the JIRA changelog: confusing pairs (done, see 4.5) | real confusions | local agent |
| 6 | 27B tie-breaker on low-confidence cases, with descriptions and boundary rules | the last stretch | local agent |
| 7 | Label hygiene for the router labels: rename mapping from pre-2023 evidence, multi-component counts, balanced rare-class variant and SQL errors (done, see 4.6) | labels: renames, multi-component, rare classes | local agent |

## Sources

- Mozilla BugBug (confidence threshold, > 80% precision): https://hacks.mozilla.org/2019/04/teaching-machines-to-triage-firefox-bugs/
- Zhang & Challis, *Software Component Prediction for Bug Reports*, ACML 2019 (new components monthly, recency sampling): https://proceedings.mlr.press/v101/zhang19c.html
- DeepTriage, incident transfer at Microsoft (frequent retraining): https://arxiv.org/pdf/2012.03665
- Su et al., *Reducing Bug Triaging Confusion by Learning from Mistakes with a Bug Tossing Knowledge Graph*, ASE 2021: https://xin-xia.github.io/publication/ase215.pdf
- TRIANGLE, multi-agent incident triage (ASE 2025): https://www.microsoft.com/en-us/research/publication/triangle-empowering-incident-triage-with-multi-agents/
- *Triage in Software Engineering: A Systematic Review* (2025): https://arxiv.org/html/2511.08607v1
- Open-world machine learning review: https://arxiv.org/html/2403.01759
- Class-incremental learning survey (nearest-class-mean prototypes): https://arxiv.org/pdf/2302.03648
- Prototype-based continual learning on device (adding a class = one more mean): https://arxiv.org/pdf/2608.07192
- Dentate gyrus and novelty detection: https://elifesciences.org/reviewed-preprints/109611
- Zero-shot LLM vs fine-tuned on domain taxonomies: https://arxiv.org/html/2609.01564
- kNN embedding pipelines vs zero-shot LLMs: https://arxiv.org/html/2608.12875
- RAG classification for unseen categories: https://arxiv.org/pdf/2608.23241
- Chinese industrial ticket routing (confidence gate, human review): https://agent.csdn.net/6a69c5af662f9a54cb95d712.html
- Chinese government dispatch with retrieval of similar tickets: https://www.woshipm.com/pd/6187832.html
- Bio-inspired triage comparison (ACO/ABC): https://dl.acm.org/doi/fullHtml/10.1145/3607947.3608095
