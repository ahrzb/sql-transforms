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
SELECT ((__cf_t.age - __cf_p0.__cf_a0) / __cf_p0.__cf_a1) AS age_z,
       (__cf_t.fare - __cf_p1.__cf_a0) AS fare_c
FROM __THIS__ AS __cf_t
LEFT JOIN __CF_PARAMS_0__ AS __cf_p0
  ON ((__cf_t.country IS NOT DISTINCT FROM __cf_p0.country))
LEFT JOIN __CF_PARAMS_1__ AS __cf_p1 ON ((1 = 1))
```

```sql
-- windows_sql: ONE fit-time execution — the original select list verbatim
-- (chain-pinning, discarded), each distinct window re-projected, the keys:
SELECT (age - avg(age) OVER (PARTITION BY country))
       / stddev_samp(age) OVER (PARTITION BY country) AS __cf_o0,
       (fare - avg(fare) OVER ()) AS __cf_o1,
       avg(age) OVER (PARTITION BY country) AS __cf_w0,
       stddev_samp(age) OVER (PARTITION BY country) AS __cf_w1,
       avg(fare) OVER () AS __cf_w2,
       country AS __cf_k0
FROM __THIS__
-- params[0].fit_sql   (keys = [country]) — pure value picking, no arithmetic:
SELECT DISTINCT __cf_k0 AS country, __cf_w0 AS __cf_a0, __cf_w1 AS __cf_a1
FROM __CF_WINDOWS__
-- params[1].fit_sql   (keys = [])
SELECT DISTINCT __cf_w2 AS __cf_a0 FROM __CF_WINDOWS__
```

`fit(train)` registers the training table, runs `windows_sql` once
(single-threaded, see below), registers the materialized result as
`__CF_WINDOWS__`, and collapses each params table out of it with SELECT
DISTINCT — every allowlisted aggregate is deterministic per group, so the
tuple is constant within a group and DISTINCT yields exactly one row per
group. Serving (a later slice) hands `serving_sql` + params to Confit as
static tables.

**Why fit is two-stage — measured, not designed.** The differential fuzz
killed two simpler forms. A `GROUP BY` fit drifts from the original text by
an ulp: DuckDB's group-by and window aggregates sum floats in different
orders. A standalone per-keyset window query drifts too: DuckDB chains window
operators, each reordering rows for the next, so a float aggregate's
summation order depends on which *other* windows share the query. Keeping the
original select items in `windows_sql` pins the operator chain, and
re-projecting an already-present window is CSE'd — the original text's value,
bit-exactly. The last ulp source is parallelism itself: DuckDB's parallel
window aggregation is schedule-dependent (measured 1/500 cases), so fit pins
`SET threads = 1`, making params deterministic and machine-reproducible; the
gate compares both sides single-threaded.

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
   drop them. `OVER ()` (empty key set) joins its one-row table via
   `LEFT JOIN … ON (1 = 1)` — never a CROSS join, because the oracle prints
   CROSS joins in comma form, which re-parses with different associativity
   once another join follows (fuzz-found).
3. **Multiplicity by construction:** `fit_sql` is `GROUP BY keys` ⇒ keys are
   unique ⇒ LEFT JOIN matches ≤ 1 ⇒ exactly one row out per row in. Map shape
   is proven, not tested.
4. **`__CF_` is a reserved prefix** (params tables `__CF_PARAMS_N__`, columns
   `__cf_aN`). Input SQL containing it (case-insensitive) is refused — same
   idiom as Confit's `__glob_pat` reservation.
5. `__THIS__` gains the alias `__cf_t` in `serving_sql`; base-column
   references are qualified through it (a short alias like `t` could collide
   with a user column; the reserved prefix cannot). Unaliased non-column
   select items get their derived name frozen as an explicit alias first —
   DuckDB names such columns by their own printed text, which qualification
   would change. Plain column refs are exempt: their name is the last path
   part, untouched by qualification.

## Accepted surface and refusals

Strict projection: `SELECT <exprs> FROM __THIS__`, nothing else. Every
refusal raises a named, positioned error — serve-or-refuse, no third mode.

**Marginalizable:** `agg(expr…) OVER (PARTITION BY col, …)` and
`agg(expr…) OVER ()`, where:

- `agg` is on the **allowlist** (grown one measured entry at a time, like the
  corpus): `avg`, `sum`, `count`, `count_star`, `min`, `max`, `stddev`,
  `stddev_pop`, `stddev_samp`, `var_pop`, `var_samp`, `variance`, `median`.
  Order-sensitive aggregates (`string_agg`, `array_agg`, …) are refused by
  absence: their per-group value is nondeterministic, which would make both
  fit and the differential gate flaky. (`first`/`last` never reach the
  allowlist check — the oracle classifies them as their own window types, so
  they are refused as position-dependent window functions.)
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

Internally, `marginalize(sql) -> Marginalized(serving_sql, windows_sql,
params)` is a pure function (parse + rewrite + plan, no execution); `fit` =
marginalize + materialize (two stages, see above). `this_model` is accepted
for signature stability but unused this slice (the training table brings its
own schema).

The raw AST dicts are handled under a two-tier typing discipline: subtrees
merely carried pass through as opaque `Node` dicts, while every node the
module *interprets* is read through a pydantic view that validates shape at
the read site — a DuckDB format change fails as one named "AST shape drift"
error, not a `KeyError` mid-walk.

## The bulletproof gate: the training-set round-trip invariant

The standing invariant, for this loop and every one after it:

> **fit + transform, applied to the training set, must be bit-equal to
> running the original query with `__THIS__` pointing at the training set.**

It is free — the training set itself is the oracle input, so no expected
values are ever written by hand — and it survives every future widening:
today "transform" is played by DuckDB executing `serving_sql` against the
fitted params; when `infer`/`infer_batch` land, the same assertion runs
through the real serving path (Confit) and gates the wiring end-to-end.

Concretely, DuckDB-vs-DuckDB differential — no inference code anywhere, both
sides at `threads = 1` (the only setting where the oracle's own float window
aggregation is bit-deterministic):

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
- **Seeded differential fuzz** (`MARGINALIZE_FUZZ_N`, default 25; deep runs at
  500+): random typed tables with NULLs everywhere × random projections. The
  fuzz found every deep bug in this slice: the CROSS-join comma-form
  associativity trap, both float summation-order drifts, and the degenerate
  `pa.null()`-typed-column coercion (an untyped training column is rejected
  territory for a later loop — the generator now types its columns
  explicitly).

## Out of scope (named next loops)

Static tables in the input SQL → computed partition keys → `FILTER`/`DISTINCT`
→ inference wiring through Confit (`IS NOT DISTINCT FROM` join support needed
there) → the DRAFT-20 leakage question. Loop-based, like Confit's corpus:
each loop widens the accepted surface and shrinks the refusal table.
