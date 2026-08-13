# sql-transform

`SQLProjection` — projections over `__THIS__`, fit once, serve row-at-a-time.

The **fit half works today**: window aggregates over `__THIS__` are
*marginalized* — computed once over training data, materialized into params
tables, and the SQL rewritten to join them instead of recomputing:

```python
from sql_transform import SQLProjection

p = SQLProjection(
    "SELECT (age - avg(age) OVER (PARTITION BY country))"
    " / stddev_samp(age) OVER (PARTITION BY country) AS age_z FROM __THIS__"
).fit(train)   # train: pyarrow.Table

p.serving_sql
# SELECT (__cf_t.age - __cf_p0.__cf_a0) / __cf_p0.__cf_a1 AS age_z
# FROM __THIS__ AS __cf_t
# LEFT JOIN __CF_PARAMS_0__ AS __cf_p0
#   ON (__cf_t.country IS NOT DISTINCT FROM __cf_p0.country)
p.params
# {"__CF_PARAMS_0__": <pyarrow.Table: country, __cf_a0, __cf_a1>}
p.plan
# (FitStep(name='__CF_LEVEL_0__', sql='SELECT ...', reads=('__THIS__',)),
#  FitStep(name='__CF_PARAMS_0__', sql='SELECT DISTINCT ...', reads=('__CF_LEVEL_0__',)))
```

Fitting is an explicit **plan** — a topologically ordered DAG of named SQL
steps (nested aggregations across CTE levels genuinely depend on each other).
Every intermediate is inspectable by name and every step is plain SQL you can
run by hand.

The serving half (`infer`/`infer_batch` through [Confit](../confit)) is a later
loop and still raises `NotImplementedError`.

## The contract

A **chain of strict projections**: the final SELECT over a CTE or derived
table, projecting over another, down to `__THIS__` — flattened into one
serving projection. The window surface is wide: any aggregate (including
`FILTER`, `DISTINCT`, ordered arguments, and order-sensitive ones), running
windows and RANGE/GROUPS frames (the order values join the key set), the rank
family, `first_value`/`last_value`/`nth_value`, expression partition keys,
named windows — plus scalar and `EXISTS` subqueries over `__THIS__`, which
are provably uncorrelated (there is no syntax to reach the outer row) and run
verbatim at fit time.

Declaring a schema unlocks more: `SQLProjection(sql, this_schema=pa.schema(...))`
(the arrow schema is the authoritative `__THIS__` schema) makes unknown columns
refuse at construction, expands `COLUMNS('regex')` (matched by DuckDB's own
regex engine) and `* EXCLUDE/REPLACE/RENAME` at any level, and resolves
lateral aliases by DuckDB's column-wins rule. Schema-free construction keeps
working, with the ambiguous cases refused.

The rule deciding it all: a window is marginalizable iff its value is a
function of row-visible values. What depends on **physical row position** —
`row_number`, `ntile`, `lag`/`lead`, bounded ROWS frames, `EXCLUDE` — is
refused, as is everything non-projection (WHERE, GROUP BY, joins, set
operations, `IN`-subqueries), each with a named error. Serve-or-refuse, no
third mode.

Parsing is DuckDB's own (`json_serialize_sql` / `json_deserialize_sql`): the
oracle's grammar and the oracle's printer, no second SQL dialect anywhere.

## The gate and the metric

For every accepted query, the training-set round-trip invariant must hold
**bit-exactly**: fit + transform on the training set equals the original
query with `__THIS__` pointing at the training set — both executed by DuckDB
(single-threaded: DuckDB's parallel float window aggregation is not
bit-deterministic even against itself). NULL partition keys are one partition
and join back via `IS NOT DISTINCT FROM`; join multiplicity is provable, not
tested.

Progression is measured by **corpus replay** (`_corpus_test.py`): queries
mined from DuckDB's own window test suite plus a curated family list, each
either MARGINALIZED (gate passes), REFUSED (named error), or FAILED (always a
bug; asserted empty). The pinned scoreboard — currently **11/22 mined**
marginalized and **39 curated** families supported against **17** named
refusal families — is the progression record; widening a loop edits it
upward in a reviewable diff.

Design specs, in order:
[loop 1 — marginalization](../../docs/superpowers/specs/2026-07-29-sql-projection-marginalization-design.md),
[loop 2 — window widening](../../docs/superpowers/specs/2026-07-29-window-widening-design.md),
[loop 3 — chains and the fit plan](../../docs/superpowers/specs/2026-07-29-projection-chains-fit-plan-design.md).
