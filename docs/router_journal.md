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
| 1 | Confidence → precision curve (coverage at 90% precision) and top-3 | reframes the task; baseline for the rest | local agent |
| 2 | Hippocampus: embedding kNN + per-component prototypes, combined with V-token on validation | new and rare components | Claude Code (needs GPU) |
| 3 | Recency weighting and monthly retraining with rolling evaluation | aging | local agent |
| 4 | Novelty detector: does it catch the first Declarative Pipelines issues? | new components forced into old ones | local agent |
| 5 | Tossing graph from the JIRA changelog: confusing pairs | real confusions | local agent |
| 6 | 27B tie-breaker on low-confidence cases, with descriptions and boundary rules | the last stretch | local agent |

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
