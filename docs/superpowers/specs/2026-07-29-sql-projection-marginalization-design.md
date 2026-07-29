# SQLProjection: marginalization over `__THIS__`

**Date:** 2026-07-29
**Status:** validated with AmirHossein (dialogue + spike), narrow slice of
[DRAFT-20](../../../backlog/drafts/draft-20%20-%20Serving-pipelines-in-SQL-marginalized-aggregates-fitted-artifacts-and-the-leakage-question.md)
**Scope:** rename `SQLTransform` → `SQLProjection`; implement bulletproof
marginalization. **No inference wiring** — `infer`/`infer_batch`/`backend`/
`boundary` keep raising `NotImplementedError`.

## What marginalization is

Aggregates over `__THIS__` are the static half of a binding-time analysis. At
fit time they are computed once over the training table, materialized into wide
params tables, and the SQL is rewritten to join them instead of recomputing:

```sql
-- input (one text, written by the user)
SELECT (age - avg(age) OVER (PARTITION BY country))
       / stddev_samp(age) OVER (PARTITION BY country) AS age_z,
       fare - avg(fare) OVER () AS fare_c
FROM __THIS__
```

```sql
-- serving_sql (generated)
SELECT (t.age - p0.__cf_a0) / p0.__cf_a1 AS age_z,
       t.fare - p1.__cf_a0 AS fare_c
FROM __THIS__ AS t
LEFT JOIN __CF_PARAMS_0__ AS p0 ON t.country IS NOT DISTINCT FROM p0.country
CROSS JOIN __CF_PARAMS_1__ AS p1
```

```sql
-- params[0].fit_sql   (keys = [country])
SELECT country, avg(age) AS __cf_a0, stddev_samp(age) AS __cf_a1
FROM __THIS__ GROUP BY country
-- params[1].fit_sql   (keys = [])
SELECT avg(fare) AS __cf_a0 FROM __THIS__
```

`fit(train)` runs each `fit_sql` over the training table via DuckDB and stores
the resulting tables. Serving (a later slice) hands `serving_sql` + params to
Confit as static tables.

## Architecture: parse with the oracle

Pure Python, entirely inside `sql_transform`. **Confit is untouched** — it is
the serving library and stays that way.

The parser is DuckDB's own, via `json_serialize_sql` / `json_deserialize_sql`:
parse SQL → JSON AST, rewrite the JSON in Python, let DuckDB print the SQL
back. One grammar, and it is the oracle's. Spike-validated on 1.5.5:

- Window aggregates round-trip; swapping a `WINDOW` node for a `COLUMN_REF`
  node in place deserializes to exactly the intended rewritten SQL.
- DuckDB-isms parse natively **and normalize away** (`SELECT x: age + 1` →
  `SELECT (age + 1) AS x`), so `serving_sql` comes out in vanilla form —
  conservative SQL for Confit's frontend later.
- The parser classifies for us: `type: WINDOW_AGGREGATE` vs
  `WINDOW_ROW_NUMBER` etc.; `FILTER` appears as `filter_expr`; the top-level
  select node exposes `where_clause` / `group_expressions` / `having` /
  `qualify` / `sample` / `cte_map` — every structural refusal is a field check.
- Parse errors are structured: `{error_type, error_message, position}`.

**Known cost:** the JSON AST is a DuckDB-internal format, not a stable API. The
node shapes we rely on get pinned with executed examples against DuckDB 1.5.5
(pins-first, like everything else); a version bump reruns the pins. If a
generated `serving_sql` ever drifts outside what Confit accepts, Confit's
refuse-at-build contract catches it loudly at wiring time — the backstop exists
regardless of parser.

`duckdb>=1.5.5` moves from the dev group to a real `sql-transform` dependency
(fit needs it; serving never will).

## Rewrite rules

1. **One params table per distinct partition-key set**, in first-appearance
   order: `__CF_PARAMS_0__`, `__CF_PARAMS_1__`, … Aggregate expressions that
   are structurally equal (same serialized AST node after DuckDB
   normalization) within a key set dedupe to one column.
2. **Join predicate is `IS NOT DISTINCT FROM`, never `USING`/`=`.** Window
   `PARTITION BY` groups NULL keys into one partition; equality joins would
   drop them. `OVER ()` (empty key set) becomes `CROSS JOIN` against a
   one-row table.
3. **Multiplicity by construction:** `fit_sql` is `GROUP BY keys` ⇒ keys are
   unique ⇒ LEFT JOIN matches ≤ 1 ⇒ exactly one row out per row in. Map shape
   is proven, not tested.
4. **`__CF_` is a reserved prefix** (params tables `__CF_PARAMS_N__`, columns
   `__cf_aN`). Input SQL containing it (case-insensitive) is refused — same
   idiom as Confit's `__glob_pat` reservation.
5. `__THIS__` gains the alias `t` in `serving_sql`; base-column references are
   qualified through it.

## Accepted surface and refusals

Strict projection: `SELECT <exprs> FROM __THIS__`, nothing else. Every
refusal raises a named, positioned error — serve-or-refuse, no third mode.

**Marginalizable:** `agg(expr…) OVER (PARTITION BY col, …)` and
`agg(expr…) OVER ()`, where:

- `agg` is on the **allowlist** (grown one measured entry at a time, like the
  corpus): `avg`, `sum`, `count`, `min`, `max`, `stddev`, `stddev_pop`,
  `stddev_samp`, `var_pop`, `var_samp`, `variance`, `median`. Order-sensitive
  aggregates (`first`, `last`, `string_agg`, `array_agg`, …) are refused by
  absence: their per-group value is nondeterministic, which would make both
  fit and the differential gate flaky.
- Partition keys are plain `__THIS__` columns (expressions: next loop).
- Aggregate arguments are arbitrary expressions over `__THIS__` columns —
  computed training-side, no restriction beyond what DuckDB accepts.

**Refused by name:**

| construct | why |
|---|---|
| `OVER (ORDER BY …)`, frames | running window, not a per-group constant |
| pure window functions (`row_number`, `lag`, `rank`, …) | position-dependent; classified by the parser's `type` tag |
| bare aggregate without `OVER` | not a projection |
| `FILTER (WHERE …)`, `DISTINCT` inside aggregate | next loops, refuse for now |
| `PARTITION BY <expression>` | v0: plain columns only |
| WHERE / GROUP BY / HAVING / QUALIFY / SAMPLE | not a projection |
| joins, subqueries, CTEs, set operations | static tables are the named next loop |
| `ORDER BY` / `LIMIT` at query level | meaningless row-at-a-time |
| multiple statements | one projection per SQLProjection |
| `__CF_` anywhere in input | reserved prefix |

## API surface

`packages/sql-transform/sql_transform/_projection.py` (renamed from
`_transform.py`):

```python
class SQLProjection:
    def __init__(self, sql: str | Template) -> None: ...
    @classmethod
    def from_file(cls, path: str) -> SQLProjection: ...
    def fit(self, table: pa.Table, /, this_model: type[BaseModel] | None = None) -> SQLProjection: ...
    @property
    def serving_sql(self) -> str: ...          # rewritten projection; fitted only
    @property
    def params(self) -> dict[str, pa.Table]: ...  # {"__CF_PARAMS_0__": …}; fitted only
    # still raising NotImplementedError — later slices:
    def infer(self, row): ...
    def infer_batch(self, rows): ...
    @property
    def backend(self) -> str: ...
    @property
    def boundary(self) -> str: ...
```

Internally, `marginalize(sql) -> Marginalized(serving_sql, params_specs)` is a
pure function (parse + rewrite + plan, no execution); `fit` = marginalize +
materialize. `this_model` is accepted for signature stability but unused this
slice (the training table brings its own schema).

## The bulletproof gate

DuckDB-vs-DuckDB differential — no inference code anywhere:

```python
original  = duck(sql, __THIS__=train)
rewritten = duck(m.serving_sql, __THIS__=train, **fitted_params)
assert original.equals(rewritten)   # bit-exact
```

Cases must include: NULL partition keys, NULLs in aggregate inputs, single-row
groups, one column used under two key sets, the same aggregate twice (dedupe),
`OVER ()` alone and mixed with keyed windows, every allowlisted aggregate, and
unicode/quoted identifiers. Plus:

- **Refusal tests:** every row of the refusal table asserts its named error.
- **AST pins:** executed `json_serialize_sql` examples for every node shape the
  walker relies on, so a DuckDB bump that moves the format fails loudly.
- Column-order and dtype equality in the differential, not just values.

## Out of scope (named next loops)

Static tables in the input SQL → computed partition keys → `FILTER`/`DISTINCT`
→ inference wiring through Confit (`IS NOT DISTINCT FROM` join support needed
there) → the DRAFT-20 leakage question. Loop-based, like Confit's corpus:
each loop widens the accepted surface and shrinks the refusal table.
