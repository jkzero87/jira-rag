# Real cases from the generation eval

Cases generated at commit `3d5a427` (`eval/results/answers_3d5a427.json`) and graded/committed at `df5a31e` (`eval/results/grade_3d5a427.json`). Model questions and answers are reproduced verbatim, unedited.

> **PENDING re-measurement.** These runs predate the fix for the G07 parser-prompt leak (see README, TL;DR). They stay as recorded until `eval/remeasure.sh` is re-run.

---

## G01 — OOM during shuffle (topic)

- **Type:** topic
- **Question:** My Spark SQL job crashes out of memory at the step where data is redistributed between machines. Known causes?

**Retrieved issues (in rank order):**
1. SPARK-22438 — OutOfMemoryError on very small data sets
2. SPARK-32141 — Repartition leads to out of memory
3. SPARK-19255 — SQL Listener is causing out of memory, in case of  data size is in petabytes.
4. SPARK-22166 — java.lang.OutOfMemoryError: error while calling spill()
5. SPARK-5314 — java.lang.OutOfMemoryError in SparkSQL with GROUP BY
6. SPARK-6104 — spark SQL shuffle OOM
7. SPARK-37100 — Pandas groupby UDFs would benefit from automatically redistributing data on the groupby key in order to prevent network issues running udf
8. SPARK-48956 — Spark Repartition Task Field Retry Cause Data Duplication
9. SPARK-48642 — False reported SparkOutOfMemoryError caused by killing task on spilling
10. SPARK-18443 — spark leak memeory and led to OOM

**Model answer:**

> Based on the provided Jira issues, the following are known causes for OutOfMemory (OOM) errors during data redistribution (shuffles/repartitions) in Spark SQL:
>
> 1.  **Lack of Aggregator in Spark SQL Shuffle:** Spark SQL shuffles did not originally define an aggregator, causing `ExternalAppendOnlyMap` to rely on `ShuffleDependency`'s aggregator. This design flaw easily causes OOM on large-scale data because it cannot efficiently combine fetched data [SPARK-6104].
> 2.  **High Shuffle Partition Count:** Setting a very high number of shuffle partitions (e.g., `spark.sql.shuffle.partitions=100000`) can cause the SQL Listener's internal metrics map (`_stageIdToStageMetrics`) to grow proportionally, leading to OOM on petabyte-scale datasets [SPARK-19255].
> 3.  **Large Data Volumes in Repartition:** Repartitioning large datasets (e.g., >4TB) on specific columns can lead to OOM or disk space exhaustion, particularly if the shuffle partition count is high (e.g., 6912) and the cluster resources are insufficient for the shuffle volume [SPARK-32141].
> 4.  **Memory Spilling Failures:** OOM can occur specifically when the system attempts to spill data to disk during sorting or shuffling. This includes errors like "error while calling spill()" on `UnsafeExternalSorter` or `ShuffleExternalSorter` when `TaskMemoryManager` fails to acquire execution memory [SPARK-22166] [SPARK-18443].
> 5.  **False Positives from Task Killing:** A reported `SparkOutOfMemoryError` may be false if it is caused by the system killing a task during the spilling process rather than actual memory exhaustion [SPARK-48642].
> 6.  **Join Operations on Small Data:** Even with small datasets, specific query structures (such as left joins) can cause unexpected memory spikes (e.g., >10GB) leading to OOM, potentially due to inefficient execution plans or memory management issues in the join logic [SPARK-22438].
> 7.  **GROUP BY Aggregation:** Large `GROUP BY` queries on large tables (e.g., 50GB) can trigger OOM due to GC overhead limits, particularly during the aggregation phase which involves shuffling [SPARK-5314].

**Verdict:** hit=True (cites SPARK-6104) · cite_ratio=0.8 (8/10 contexts) · expected gold=[SPARK-4515, SPARK-5421, SPARK-6104] · invented citations=0 (all in_context). **Note:** one sentence goes beyond the source — "potentially due to inefficient execution plans..." on SPARK-22438 is the model's own hedged speculation; the citation check does not catch this.

---

## G09 — Newest ZooKeeper version (lookup, lowest cite_ratio with hit)

- **Type:** lookup (lowest cite_ratio among lookups with hit=True; tie at 0.1 broken by order, G09)
- **Question:** What's the newest ZooKeeper version Spark is moving to?

**Retrieved issues (in rank order):**
1. SPARK-35954 — [Deploy] Upgrade Apache Curator Dependency to 4.2.0
2. SPARK-34110 — Upgrade ZooKeeper to 3.6.2
3. SPARK-49844 — PySpark requiring vulnerable Apache Zookeeper version 3.6.3
4. SPARK-59652 — Upgrade ZooKeeper to 3.9.6
5. SPARK-40070 — Upgrade Apache ZooKeeper to 3.8.0
6. SPARK-50135 — Upgrade ZooKeeper to 3.9.3
7. SPARK-42486 — Upgrade ZooKeeper from 3.6.3 to 3.6.4
8. SPARK-47402 — Upgrade `ZooKeeper` to 3.9.2
9. SPARK-15876 — Remove support for "zk://" master URL
10. SPARK-55894 — Upgrade Zookeeper to 3.9.5

**Model answer:**

> The newest ZooKeeper version Spark is moving to is 3.9.6 [SPARK-59652].

**Verdict:** hit=True (cites SPARK-59652) · cite_ratio=0.1 (1/10 contexts) · expected gold=[SPARK-59652] · invented citations=0.

---

## G23 — Open blockers since June 2026 (filtered with hit)

- **Type:** filtered
- **Question:** Which blocker bugs filed since June 2026 are still open?

**Retrieved issues (in rank order):**
1. SPARK-57343 — [SECURITY] Upgrade bundled Netty to 4.2.15.Final and ZooKeeper to 3.9.5 in PySpark to resolve Critical/High CVEs
2. SPARK-59122 — UnionExec fused into whole-stage codegen can fail with "key not found: numOutputRows"

**Model answer:**

> Based on the provided Jira issues, the blocker bugs filed since June 2026 that are still open are:
>
> *   [SPARK-57343]
> *   [SPARK-59122]

**Verdict:** hit=True · cite_ratio=1.0 (2/2 contexts) · expected gold=[SPARK-57343, SPARK-59122] · invented citations=0 (all in_context).

---

## G19 — Retrieval miss (lookup: gold did not reach the contexts)

- **Type:** lookup
- **Question:** CRAN checks for SparkR fail on the build machines because an R package is missing.

**Retrieved issues (in rank order):**
1. SPARK-19387 — CRAN tests do not run with SparkR source package
2. SPARK-31918 — SparkR CRAN check gives a warning with R 4.0.0 on OSX
3. SPARK-24152 — SparkR CRAN feasibility check server problem
4. SPARK-16507 — Add CRAN checks to SparkR
5. SPARK-20197 — CRAN check fail with package installation
6. SPARK-25923 — SparkR UT Failure (checking CRAN incoming feasibility)
7. SPARK-39959 — Recover SparkR CRAN check in GitHub Actions CI
8. SPARK-22167 — Spark Packaging w/R distro issues
9. SPARK-22889 — CRAN checks can fail if older Spark install exists
10. SPARK-25999 — make-distribution.sh failure with --r and -Phadoop-provided

**Model answer:**

> Based on the provided Jira issues, the specific claim that CRAN checks fail because an **R package is missing** is not explicitly stated. However, the issues describe failures related to missing dependencies, missing documentation, or missing installation components:
>
> *   **Missing Documentation/Dependencies:** [SPARK-39959] reports that CRAN checks fail in GitHub Actions CI due to issues with "latest dependences of R documentation build, e.g., rmarkdown or roxygen2," resulting in warnings for "missing documentation entries."
> *   **Missing Installation/Spark:** [SPARK-19387] states that `sparkR.session()` is not installing Spark, causing `R CMD check --as-cran` to fail.
> *   **Missing Files/Environment:** [SPARK-22889] notes that checks can fail if an older partial download or install of Spark is left behind on the machine. [SPARK-22167] mentions that the R directory was missing from the Hadoop distribution.
> *   **Missing Hadoop Dependencies:** [SPARK-25999] describes a failure where building R documentation depends on Hadoop dependencies that are not present when using `--hadoop-provided`.
>
> The provided issues do not contain a specific report identifying a single missing R package as the cause of failure on build machines.

**Verdict:** hit=False · cite_ratio=0.5 · expected gold=[SPARK-17191] · invented citations=0. **The failure is in retrieval, not generation:** the gold `SPARK-17191` (the actual issue for the missing R dependency) never reached the retrieved contexts (gold_in_context=False, the only one of the 40), so the correct answer was unreachable from what was given. The model honestly answered that, among the provided issues, no report identifies a single missing R package as the cause of the failure, and cited 5 legitimate in-context issues instead of inventing a key.
