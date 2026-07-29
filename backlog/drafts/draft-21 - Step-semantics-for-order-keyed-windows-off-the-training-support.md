---
id: DRAFT-21
title: Step semantics for order-keyed windows off the training support
status: Draft
type: spike
created: 2026-07-29
---

## Where this came from

Design dialogue with AmirHossein, 2026-07-29, right after loop 3
(projection chains + the fit plan) merged. The question: **what should
`rank()` — and every order-keyed window — mean when the serving row's value
was never in the training set?** Parked with a converged direction so the
conclusions survive; sequenced after the schema-aware-resolution loop.

## The problem

Order-keyed windows (`rank`, `dense_rank`, `percent_rank`, `cume_dist`,
running aggregates under RANGE/GROUPS frames, `last_value`/`nth_value`)
marginalize with the order values in the join key set. Serving joins by
exact `IS NOT DISTINCT FROM` match — so an unseen order value (almost
*every* value, when the key is continuous like `salary`) misses the LEFT
JOIN and yields NULL. Truthful, but it makes the whole order-keyed family
nearly useless on real serving traffic.

## The converged answer: the step-function reading

Every order-discriminated window we admit is, per partition, a **step
function of the order value** — that is exactly *why* it was marginalizable.
The params table is that function sampled at the training points. The fitted
transform should mean: **evaluate the training-set step function at the
serving row's coordinate** (row excluded).

```
rank() OVER (PARTITION BY dep ORDER BY salary)
-- fitted step for dep='sales':  4800 → 1 (two peers), 5000 → 3
-- serving salary=4750 (unseen):
--   exact-join semantics (today):  NULL
--   step semantics:                1 + #{training rows < 4750} = 1
```

Why this is the right default, not an accommodation:

1. **It extends, never contradicts.** At observed coordinates the step
   function equals the exact join, so the training-set round-trip invariant
   is untouched. Exact-join is the degenerate restriction of step semantics.
2. **Partition keys stay exact.** Categorical keys have no "between";
   NULL for an unseen group remains the honest answer. Step semantics is for
   *order* keys only — partition keys are categorical, order keys ordinal.
3. **`cume_dist() OVER (ORDER BY x)` becomes the empirical CDF against
   training** — sklearn's QuantileTransformer, falling out of one SQL
   function. Strong evidence the step reading is the ML-correct semantics.
4. **Still oracle-testable.** Off-support behavior has a definitional form
   DuckDB can execute per probe (`SELECT 1 + count(*) FROM train WHERE
   dep = ? AND salary < ?`), so the gate extends rather than weakens.

## Mechanics (sketch)

AS-OF join instead of equality on the order-key part of the join predicate.
DuckDB has `ASOF JOIN` natively for the fit-side gate; on the Confit side it
is a binary search over a sorted params column — fits the static-lookup
architecture. Per family:

- running `sum`/`count`/`min`/`max` (RANGE ≤ frames): as-of pick of the
  cumulative value at the greatest observed `o' ≤ v`.
- `rank`: `1 + cum_count(o < v)` — one extra cumulative-count column in the
  params table; exact hits read the stored rank.
- `percent_rank`/`cume_dist`: derived from rank/cum_count and `n`.

## Pins needed before believing anything

- Tie boundaries: `<` vs `≤` per function, and DuckDB ASOF's strictness.
- NULL order values at serving: NULL is its own peer class — stays an exact
  match, never an as-of probe.
- `last_value` off-support is genuinely debatable: "the last peer of a value
  nobody has" is arguably the serving row itself — a different kind of
  answer than a frozen param. Decide per-function; refusing step semantics
  for `last_value`/`nth_value` and keeping them exact-only is a fine v0.
- Bounded RANGE frames (`RANGE BETWEEN 2 PRECEDING AND CURRENT ROW`): the
  step function has 2·n breakpoints, not n — the as-of table must sample
  frame *boundaries*, not just observed points. May stay exact-only at
  first.

## Decision state

Direction agreed ("ah that's nice I like it"): adopt step semantics as THE
meaning of order-keyed windows, in its own loop with its own pins and an
off-support gate, sequenced after schema-aware resolution. Until that loop,
NULL-off-support is the documented behavior.
