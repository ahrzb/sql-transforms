# `PARTITION BY` window aggregates — Design

**Goal:** Support `AGG(col) OVER (PARTITION BY k1, k2, …)` in `SQLTransform` —
per-partition learned state (e.g. target/categorical encoding) — while keeping
the transform strictly 1-to-1 (row-preserving) and both execution engines
(DataFusion batch, Rust `InferFn`) in agreement.

## Motivation

`OVER ()` aggregates freeze one global value at `fit()` and broadcast it to every
row. `PARTITION BY` generalizes that to *one value per partition key*: `AVG(target)
OVER (PARTITION BY city)` is a per-city mean — the core of target/categorical
encoding, the most-requested feature-engineering primitive this project lacks.
Today it raises `NotImplementedError`. The Rust engine already has the primitive
needed (a static keyed table joined to a row table), so this is largely wiring
plus one focused Rust addition.

## Governing invariant: the transform is strictly 1-to-1

Every join the rewrite emits is a **LEFT JOIN onto a unique-keyed state table**.
`GROUP BY` guarantees one row per key-set, so a LEFT JOIN matches at most one row
and never drops one → row count out always equals row count in. No INNER joins are
ever emitted by `SQLTransform` (an INNER join would drop unseen-partition rows,
violating 1-to-1). Unseen key → LEFT miss → NULL → still exactly one output row.

## Unseen partitions resolve to NULL

At inference a row may carry a partition key never seen at `fit()`. It resolves to
**NULL**, which propagates through the surrounding expression (NULL output for that
row). This is the LEFT-miss result and needs no fallback machinery; a user who
wants a fallback writes it in SQL (e.g. `COALESCE(...)`). Both engines must agree
on this.

## Unified state model: every state table is a LEFT-joined keyed table

Today global `OVER ()` state is a one-row Pydantic *row-table* cross-joined into
the query. This design **replaces** that special case with a single uniform
mechanism:

- **Partition state table** — keyed by the partition columns. For
  `AVG(target) OVER (PARTITION BY city)`: a table `(city, avg_target)`, one row per
  city, LEFT JOIN `ON __THIS__.city = <table>.city`.
- **Global state table** — the `OVER ()` case: a one-row table carrying a synthetic
  constant key column `__state_marker__ = 0` (reusing the marker already used for
  empty state in `_batch.py`), LEFT JOIN `ON <table>.__state_marker__ = 0`. Always
  hits → identical behavior to today, but through the same LEFT-lookup code path.

Both engines then run exactly one state mechanism: LEFT lookup join on unique-keyed
static tables. The strict inner `LookupJoin` is no longer emitted by `SQLTransform`
(it remains for user-authored `InferFn` joins).

### Naming

State tables are named deterministically from their key-set:
- Global (empty key-set): `__STATE__`.
- Partition by `city`: `__STATE_BY_city__`.
- Composite `city, region`: `__STATE_BY_city_region__`.

Same key-set → same name → natural dedup: all aggregates sharing a partition-key-set
collapse into one table with one value column each.

### Grouping and dedup

Window aggregates are grouped by their partition-key-set. Each group yields one
state table: its key columns plus one value column per distinct `(fn, col)`, named
by the existing `state_key(fn, col)` scheme (`avg_target`). The existing
case-collision `ValueError` (two columns differing only by case mapping to the same
`state_key`) applies per-table. Distinct key-sets with the same `(fn, col)` are
distinct tables — no collision.

## Value typing: preserve real types (no float coercion)

State value columns keep their **natural type** — `int`, `float`, `str`, `bool` —
instead of today's blanket `float` coercion (`extract_state` does
`values[key] = float(value)` and `synthesize_state_model` makes every field
`float`). This is essential for categorical work: `COUNT(*) OVER (PARTITION BY
city)` is an integer count-encoding, an ordinal id is an integer, not `20.0`.

The mechanism makes this *the natural path, not extra work*: state tables are built
directly from the `GROUP BY` / aggregate query results, whose columns already carry
correct Arrow types. Nothing coerces them. The per-table state model is synthesized
from the resulting Arrow schema (reusing `_schema.py`'s `_arrow_type_to_python`),
and every value field is **nullable** (`T | None`) because a LEFT-join miss yields
NULL. Partition **key** columns likewise keep their native types — they are join
keys.

This folds in what was the separate "aggregate result typing" backlog item, and it
applies to **both** the global `OVER ()` state and the `PARTITION BY` state (the
old global scalar path stops coercing too — its one-row table is built from the
query result's real column type). Verified: the Rust `InferFn` already carries
`int`/`float`/`str`/`bool` static-table columns through a lookup join with types
intact (`Value` enum has `Int`/`Float`/`Str`/`Bool`/`Null`), so typed values need
no Rust change — only the LEFT-lookup-join does. Multi-argument and expression
aggregate arguments remain out of scope (single plain column only).

## Architecture & data flow

```
fit(train_table):
  tree     = parse_and_validate(sql)          # PARTITION BY no longer rejected
  windows  = find_window_aggregates(tree)     # each WindowAgg carries partition_cols
  groups   = group windows by partition-key-set
  for each group:
    run  SELECT <keys>, <agg1> AS k1, ... FROM __THIS__ GROUP BY <keys>   (DataFusion)
    build a pyarrow state table straight from the result: key columns + value
    columns, all keeping their natural Arrow types (no float coercion)
    (empty key-set -> one row + __state_marker__=0 column)
  rewritten = rewrite_sql(tree, windows, state_tables)   # LEFT JOINs on key equality
  store: self._rewritten_sql, self._state_tables (dict[name -> pa.Table])
  build InferFn(rewritten, row_tables={"__THIS__": this_model},
                static_tables=self._state_tables)

transform(batch)  -> DataFusion:
  ctx.register __THIS__ = batch
  ctx.register each state table by name
  run rewritten SQL (LEFT JOINs); DataFusion yields NULL on miss natively.

infer/infer_batch(rows)  -> Rust InferFn:
  self._infer_fn.infer({"__THIS__": rows})     # state tables are static, bound at build
  Rust LEFT lookup join yields a NULL row on miss.
```

## Components / file changes

- **`sql_transform/_sql.py`** — `WindowAgg` gains `partition_cols: tuple[str, ...]`
  (empty for `OVER ()`), populated from the sqlglot `partition_by` nodes (each must
  be a plain column; otherwise `ValueError`). `has_partition` becomes derivable
  (`bool(partition_cols)`) — keep or drop per implementation preference. ORDER BY
  still rejected via `has_order`.
- **`sql_transform/_state.py`** — `extract_state` is replaced by a builder that
  returns `dict[str, pa.Table]` (state-table-name → table): groups windows by
  key-set, runs one `GROUP BY` query per group (empty key-set → a single-row query
  plus a `__state_marker__` column), and builds pyarrow tables straight from the
  results with **no float coercion** — value columns keep their natural Arrow type.
  `state_key` unchanged. The `NotImplementedError` for `has_partition` is removed;
  the one for `has_order` stays. A shared `state_table_name(partition_cols)` helper
  gives the deterministic table name (`__STATE__` / `__STATE_BY_city__`), used by
  both `_state` and `_rewrite`.
- **`sql_transform/_schema.py`** — `synthesize_state_model` is **removed**. State is
  no longer a Pydantic row-model: state tables are passed to `InferFn` as *static*
  `pa.Table`s (read directly, columns validated against the table's own schema) and
  registered directly in DataFusion. Types (int/float/str/bool, nullable) flow from
  the Arrow tables with no model synthesis. `synthesize_this_model` is unchanged.
- **`sql_transform/_rewrite.py`** — `rewrite_sql` takes the state-table set and emits
  one LEFT JOIN per key-set with ANDed key equalities (`__THIS__.k = T.k`), replacing
  each window node with `T.<state_key>`. Global state joined on the marker constant.
- **`sql_transform/_batch.py`** — `run_batch` registers every state table in the
  SessionContext (generalized from the single `__STATE__`), then runs the rewritten
  SQL. `_state_to_table` folds into the `_state.py` builder or stays as the
  empty-marker helper.
- **`sql_transform/__init__.py`** — `fit` stores `self._state_tables`; `transform`
  passes them to `run_batch`; `infer`/`infer_batch` no longer pass `__STATE__` as a
  row (state is bound into `InferFn` as `static_tables` at build time).
- **Rust (`src/plan.rs`, `src/lookup.rs`, `src/lib.rs`)** — LEFT lookup join:
  - `src/plan.rs` builder: accept `JoinOperator::LeftOuter` (today only inner `Join`/
    `Inner` at `plan.rs:144`); mark the resulting `LookupJoin` node with an `outer`
    flag.
  - `LookupJoin` execute (`plan.rs:497`): on key miss, when `outer`, insert a row of
    NULLs for the static table's columns instead of raising `MissingKey`. Requires the
    lookup index (`src/lookup.rs`) to expose its column names so the null row can be
    built with the right shape.
  - Expr eval already does three-valued NULL propagation, so a NULL joined column
    flows through arithmetic to a NULL result with no further change.

## Edge cases

- **Unseen partition key** → LEFT miss → NULL (both engines). No fallback.
- **NULL partition key** (training or inference) → never matches (SQL NULL-join
  semantics; Rust `plan.rs:473`, DataFusion native) → NULL result. The trained
  NULL-key group is harmless dead weight. Not special-cased.
- **Composite key** → ANDed per-column equality join; keys keep native types.
- **Mixed `OVER ()` + multiple `PARTITION BY`** → one state table + LEFT join per
  distinct key-set; all 1-to-1.
- **`PARTITION BY` + `ORDER BY`** together → still `NotImplementedError` (ORDER BY
  unchanged).
- **Empty batch / unseen everything** → row count preserved; values NULL where
  missing.

## Testing

- **`_sql`** — `partition_cols` populated for single and composite keys; non-column
  partition expression rejected with `ValueError`.
- **`_state`** — `GROUP BY` builds the correct per-partition table; dedup within a
  key-set (repeated `(fn, col)` → one value column); distinct key-sets → distinct
  tables; empty key-set → one-row marker table.
- **Value typing** — `COUNT(*) OVER (PARTITION BY city)` yields an **integer** value
  column and integer output (count encoding), not `float`; a string-valued
  aggregate (e.g. `MIN(name)`) yields a `str` column; state-model value fields are
  nullable. Covers both the global and partition paths (the old global path no
  longer coerces to float either).
- **`_rewrite`** — correct LEFT JOIN emission for single and composite keys; global
  state joined on the marker; window nodes replaced by `T.<state_key>`.
- **Rust (`tests/test_interpreter.py`)** — LEFT lookup join: hit returns the value,
  miss returns NULL (not `MissingKey`); the strict inner join still errors on miss.
- **Cross-engine equivalence** — `transform` vs `infer_batch` on a batch containing a
  seen and an **unseen** partition → identical values, both NULL for the unseen row.
- **End-to-end target encoding** — `AVG(target) OVER (PARTITION BY city)`: fit on
  training cities; `infer` a seen city (its mean) and an unseen city (NULL); `transform`
  a batch and confirm 1-to-1 row count.

## Non-goals

- `ORDER BY` / window frames (running/cumulative/moving aggregates) — a separate,
  harder effort (order-dependent streaming state); stays rejected.
- Global-aggregate fallback / smoothing priors for unseen partitions — NULL only;
  users compose fallbacks in SQL.
- Multi-argument or expression aggregate arguments — still single plain column.
