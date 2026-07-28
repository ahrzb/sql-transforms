---
id: DRAFT-20
title: Serving pipelines in SQL — marginalized aggregates, fitted artifacts, and the leakage question
status: Draft
type: spike
created: 2026-07-28
---

## Where this came from

Design dialogue with AmirHossein, 2026-07-28. Stated goal: **express the entire
serving pipeline in SQL.** Not implemented, not scheduled — parked so the
conclusions survive.

## The architecture we converged on

**Aggregates over `__THIS__` are the static half of a binding-time analysis.**
The user writes one text; a training-side preprocess marginalizes every
aggregate over `__THIS__` into a params table and rewrites it to a join.
`PARTITION BY g` becomes the join key. AmirHossein's plan; pipelines appear as
UDAFs at the training end.

```sql
-- one text
SELECT (age - avg(age) OVER (PARTITION BY country)) / stddev(age) AS age_z FROM __THIS__
-- serving form after rewrite (already compilable today)
SELECT (age - p.mean) / p.scale FROM __THIS__ LEFT JOIN params p USING (country)
```

Consequences:

- **Most sklearn transformers become unnecessary.** StandardScaler, MinMax,
  SimpleImputer, target encoding, one-hot domains are all `avg`/`stddev`/
  `median`/`DISTINCT`/`GROUP BY`. No importer, no `State<T>` surface — the
  state struct is what the *rewriter emits*, not a user-facing concept.
- **A fitted transformer is a static table that gets joined.** Per-group
  families (per-country PCA) are the general case, not a special case — the
  group is the join key, and multi-column keys already work.
- **Params must be pivoted WIDE**, not tidy/long: a long params table would
  need `SUM(...) GROUP BY` to form a dot product, and serving rejects
  aggregation by design.
- **The engine's no-aggregation rule stops being a limitation** and becomes the
  type system: the rewrite's output is aggregate-free by construction, so
  anything the specializer refuses is a bug in the rewrite.
- **The payoff is training/serving skew elimination by construction**, not
  latency — one text, mechanically split.

## What actually needs building (small)

Everything reduces to **two instructions plus a params-materialization library**:

```
matvec         → linear, logistic, PCA, any projection (+ link fn)
tree_ensemble  → GBDT / RF / extra trees
everything else → desugar over existing lanes; ZERO engine change
```

Suggested order: (1) params-as-static-tables + desugar catalogue — no engine
change, covers most real pipelines; (2) `matvec` — unlocks linear/logistic,
the most-served model class, plus PCA; (3) `tree_ensemble`, sklearn-only,
when something real needs it.

Sizing (my estimate, from the wave work): a bit-exact tree ensemble for ONE
model family ≈ 1.5–2 weeks. The tree walk itself is a day. The cost is the
six-file instruction tax, the importer (per format), and the semantics pins —
float32-vs-f64 threshold comparison, `<` vs `<=` ties, per-node missing
direction, link/objective, base_score. Cost scales with *formats supported*,
not tree count. XGBoost +1wk, LightGBM (categorical bitset splits) +1–2wk;
that way lies a model-format zoo.

## Bit-exactness, inverted

- Scalers / imputers / encoders / trees: **bit-exact is reachable**. Tree walks
  are comparisons plus a fixed-order sum.
- Anything matmul-shaped (PCA, linear): **not bit-exact with sklearn** — numpy
  dispatches to BLAS, whose summation order varies by build, threading, and
  size. Needs a documented own-order + tolerance tier, loudly opt-in. It would
  be the first approximate answer this engine ever serves.

Oracle split: feature SQL keeps DuckDB (existing machinery, unchanged, and
functions defined AS their expansion stay verifiable — pin whether DuckDB
`CREATE MACRO` takes struct args). Model scoring's oracle becomes
sklearn/XGBoost. Closing that seam would mean shipping a DuckDB extension with
the same functions — roughly a doubling of the work.

## THE OPEN QUESTION — leakage (AmirHossein: "very tricky, brainstorm at some point")

**Cross-fitting breaks the one-text-one-rewrite symmetry**, which is the
property everything else rests on:

```
serving:  LEFT JOIN params         USING (country)
training: LEFT JOIN params_by_fold USING (country, fold)   -- fold ∉ serving
```

The marginalizing rewrite would no longer be the same transformation in both
phases. Sub-questions for that session:

1. How does the preprocessor learn **which columns are targets**? `avg(age)` is
   safe; `avg(target)` leaks. SQL carries no such metadata today.
2. Refuse, or cross-fit automatically? The design makes the leaky version the
   easiest thing to write.
3. Smoothed target encoding (m-estimate / prior) means the marginalized
   aggregate has **hyperparameters** — it isn't a pure `avg`.
4. High-cardinality group keys: a group of size 1 is the row's own target.

## Other open items

- Unknown group at serving is unavoidable (a country unseen in training). The
  join type is the policy — LEFT → null lanes, INNER → row drops, COALESCE to a
  global row → fallback. Must be an explicit per-table choice.
- `PARTITION BY g` requires `g` to be a serving input field — a build-time
  proof that should name the missing column.
- `avg(x)` silently means training-time, never request-batch. Safe (serving
  sees one row) but a real trap; wants an explain output.
- Params-table size ceiling for high-cardinality keys (statics are frozen in
  memory).
- Tfidf vocabularies fall on the blob side of the line, same as matrices/trees.

## The line

State-as-struct for fixed-arity scalar params. Opaque frozen blob plus a
dedicated instruction for anything whose size scales with model complexity —
matrices, ensembles, vocabularies. A GBDT as struct lanes would be ~150k
compile-time lanes; its structure *is* control flow, so it belongs in the
compiled code, not a data field.

End state: the deployable artifact is SQL text + Arrow params tables, no Python
at serving. Given the WASM spike ran near-native under Go (wazero) and Java
(chicory, with its opt-in compiler), the same artifact serves from a JVM or Go
service with nothing reimplemented.
