# Widening marginalization: almost all window queries

**Date:** 2026-07-29
**Status:** loop 2 of SQLProjection marginalization; extends
[loop 1](2026-07-29-sql-projection-marginalization-design.md), whose
architecture (oracle parser, two-stage fit, training-set round-trip
invariant) is unchanged.

## The one rule

A window expression is marginalizable iff its value is a **function of
row-visible values**: the partition keys, plus — once ORDER BY enters — the
order values. RANGE/GROUPS frame semantics make all order-peers share a
value, so the value is constant within (partition keys ∪ order values) and a
params table keyed on that tuple reproduces it exactly. What can never be
carried by a join key is **physical position**: row_number, ntile, lag/lead,
and bounded ROWS frames distinguish tied rows, so they stay refused.

Every window is assigned a discriminator set D:

| window | D (extra join keys beyond PARTITION BY) |
|---|---|
| aggregate, no ORDER BY (default frame) | ∅ |
| aggregate, any frame `UNBOUNDED PRECEDING..UNBOUNDED FOLLOWING` | ∅ |
| aggregate, ORDER BY + default / RANGE / GROUPS frame (constant bounds) | order exprs |
| `rank`, `dense_rank`, `percent_rank`, `cume_dist` | order exprs |
| `first_value`, default or whole-partition frame | ∅ |
| `first_value`/`last_value`/`nth_value` (constant n), RANGE/GROUPS frames | order exprs |
| **refused:** `row_number`, `ntile`, `lag`, `lead`; bounded ROWS frames; `EXCLUDE …`; non-constant frame bounds | — |

Join keys = partition entries ∪ D, and both may now be **expressions**
(`PARTITION BY lower(c)`, `ORDER BY o % 10`): validated row-wise (no
windows/aggregates/subqueries inside), projected in `windows_sql`, joined
back by the qualified expression `ON (lower(__cf_t.c) IS NOT DISTINCT FROM
__cf_p0.__cf_x0)`. Plain-column keys keep their natural name in the params
table; expression keys are named `__cf_xJ`. `COLLATE` on an order key is
stripped to its child for keying: raw-equal implies collated-equal implies
peer, so joining on the raw value preserves multiplicity (collated-equal but
raw-distinct values get their own params rows carrying the same value —
denormalized, still exact; joining on the collated value would match several
rows and break multiplicity).

## The aggregate allowlist is gone

Loop 1's allowlist encoded "deterministic under GROUP BY refit". The
two-stage fit removed that dependency: `windows_sql` reruns the original
computation and the collapse picks values, so *any* function the oracle
classifies as `WINDOW_AGGREGATE` is per-key-constant within the execution —
including order-sensitive ones (`string_agg`, `array_agg`, `first`) and
ordered-argument aggregates (`string_agg(x ORDER BY o)`), whose frozen value
is precisely the fitted semantics. `FILTER (WHERE …)` and `DISTINCT` inside
aggregates are likewise admitted (the filter expression validated row-wise).
`IGNORE NULLS` is admitted on value functions, refused on aggregates.
The bare-aggregate-without-OVER refusal (checked against
`duckdb_functions()`) is unchanged.

Named windows (`WINDOW w AS (…)` + `OVER w`) are inlined by the oracle's
parser — supported with zero code.

## Multiplicity, restated

For every admitted window, the value is constant within its key tuple (the
table above is exactly that claim, case by case), so `SELECT DISTINCT keys,
aggs FROM __CF_WINDOWS__` collapses to one row per key tuple, keys are
unique, and the LEFT JOIN matches exactly one params row per input row. The
training-set round-trip invariant remains the gate, now over the widened
surface: the fuzz generator gains ORDER BY windows, rank/value functions,
frames, FILTER, DISTINCT aggregates, expression keys, and order-sensitive
aggregates.

## Still refused after this loop

Position-dependent windows (`row_number`, `ntile`, `lag`, `lead`, bounded
ROWS frames, `EXCLUDE`) — a join key cannot carry physical position;
non-constant frame bounds and non-constant `nth_value` n; nested
windows/aggregates/subqueries inside window arguments, keys, or filters; and
every loop-1 structural refusal (WHERE, GROUP BY, joins, CTEs, …).
