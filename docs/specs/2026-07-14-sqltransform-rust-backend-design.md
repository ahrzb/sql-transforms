# SQLTransform on the Rust Interpreter — Design Spec

## Goal

Replace `SQLTransform`'s Python-codegen inference path (`_codegen.py`'s `generate_infer_fn`, which `exec()`s generated Python source) with the Rust `InferFn` interpreter. `fit()` now rewrites the user's SQL into a form the Rust engine can run directly — window-aggregate references become plain column references against a synthesized `__STATE__` row table — instead of walking the DataFusion plan to emit and `exec()` Python source.

## Motivation

`SQLTransform` currently has two independent execution paths for essentially the same problem the Rust interpreter (`src/`) already solves: safe, schema-validated, row-at-a-time SQL evaluation. `_codegen.py` re-implements a small expression compiler in Python (`Column`/`BinaryExpr`/`Alias` → Python source string → `exec()`) that only understands a couple of DataFusion expression node types and has no schema validation. Routing through `InferFn` instead means `SQLTransform` inherits the Rust engine's column validation, type inference, and output-model synthesis for free, and Python-side `exec()` of generated source goes away entirely.

## Scope

- **In scope:** non-partitioned window aggregates only — source SQL must write them as `AGG(col) OVER ()` (empty, unpartitioned window), e.g. `SELECT age / AVG(age) OVER () AS age_norm FROM __THIS__`. This is exactly what `_state.py`/`_codegen.py` support today (see their existing tests — none exercise a bare, OVER-less aggregate).
- **Out of scope, deferred:** `OVER (PARTITION BY ...)`. The Rust engine has no window-function support at all yet; adding partitioned-state support means adding it there first (as a `LookupJoin` against a per-partition state table, keyed by partition column — already discussed and deferred, not part of this spec). `extract_state` raises `NotImplementedError` if it finds a partitioned window aggregate, rather than silently mishandling it.
- **Breaking change:** `FROM __THIS__` becomes a required convention — users write `__THIS__` as the table name in their SQL directly (single-input transforms only). This replaces the current arbitrary-table-name convention (`FROM data` in the README/tests). Per [[feedback_v0_no_backward_compat]], this is a direct breaking change, no compat shim.

## Public API

```python
from sql_transform import SQLTransform
import pyarrow as pa

sql = "SELECT age / AVG(age) OVER () AS age_norm FROM __THIS__"

# this_model omitted: synthesized from the input table's arrow schema
t = SQLTransform(sql)
t.fit(data)                 # data: pa.Table
out = t.transform(data)     # batch
row = t._infer({"age": 42}) # single row

# this_model supplied explicitly
from pydantic import BaseModel
class Row(BaseModel):
    age: int

t = SQLTransform(sql)
t.fit(data, this_model=Row)
```

- `SQLTransform.fit(table, this_model=None)`:
  - `this_model: type[BaseModel] | None = None` — if given, used directly as `__THIS__`'s row schema (passed straight to `InferFn`'s `row_tables`). If omitted, synthesized from `table.schema` (arrow → pydantic, same base-type mapping the Rust side already uses for static tables in `schema.rs::arrow_type_to_base`, done here in Python since `row_tables` needs an actual pydantic class). No separate Python-side compatibility check against `table`'s schema — `InferFn`'s own build-time column/type validation catches mismatches (unknown column, incompatible type) and surfaces a clear `ValueError`.
- `transform`/`_infer` signatures are unchanged from today.

## Architecture

```
sql_transform/_state.py    (rewritten) — extract_state() walks the DataFusion
                              plan display text for window-agg nodes (same
                              regex-based detection as today), but:
                              - rejects PARTITION BY (NotImplementedError)
                              - computes each DISTINCT (fn, col) once, not
                                once per alias — dedups repeated aggregates
                              - keys the state dict by `{fn}_{col}`
                                (lowercase, bare column name, qualifier
                                stripped, NO leading underscore — Pydantic
                                v2 treats leading-underscore field names as
                                private attributes, excluded from
                                model_fields/model_validate, so the field
                                name itself can't have one; only the SQL
                                reference is qualified, e.g.
                                `__STATE__.avg_age`) instead of by output
                                alias
                              - returns a pydantic model INSTANCE
                                (StateModel(**values)), not a raw dict

sql_transform/_codegen.py  (rewritten) — same tree-walk shape as today
                              (Column / BinaryExpr / Alias over the plan's
                              projections), retargeted to emit SQL TEXT
                              instead of Python source:
                              - window-agg Column -> `__STATE__.fn_col`
                              - plain Column       -> `__THIS__.col`
                              - BinaryExpr          -> `(<left> <op> <right>)`
                              Produces `SELECT <exprs> FROM __THIS__, __STATE__`.
                              No more exec() — this is now a pure string
                              builder.

sql_transform/_schema.py   (new)      — pa.Table.schema -> synthesized
                              pydantic model (this_model default path) and
                              a state dict -> synthesized pydantic StateModel
                              (all-float fields, one per fn_col key).
                              Both via pydantic.create_model, mirroring the
                              output_model synthesis InferFn already does
                              on the Rust side (schema.rs), just in Python
                              because row_tables/the state row need a real
                              class + instance, not just an Arrow schema.

sql_transform/__init__.py  (rewritten) — SQLTransform.fit():
                              1. this_model = given or synthesize_this_model(table.schema)
                              2. register table as "__THIS__" in a DataFusion
                                 SessionContext, get the logical plan
                              3. state = extract_state(plan, ctx, "__THIS__")
                                 (a StateModel instance)
                              4. rewritten_sql = rewrite_sql(plan, state)
                              5. self._infer_fn = InferFn(
                                     rewritten_sql,
                                     row_tables={"__THIS__": this_model,
                                                 "__STATE__": type(state)},
                                     static_tables={},
                                 )
                              6. self._state = state  (kept for every infer() call)

                            SQLTransform.transform()/._infer():
                              rows -> SimpleNamespace (duck-typed, per the
                              existing InferFn duck-typing support -- no
                              pydantic validation needed on the row side),
                              call self._infer_fn.infer({"__THIS__": rows,
                              "__STATE__": [self._state]}), convert the
                              resulting output_model instances back to
                              dict / pa.Table.
```

`_codegen.py` and `_state.py` keep their current file responsibilities (state extraction vs. expression-tree walking) — only their *output* changes (SQL text instead of Python source; a typed state model instance instead of a dict).

## Why `__STATE__` Is a Row Table, Not a Static Table

Nothing joins into `__STATE__` by key — it's always a single constant row, cross-joined against every `__THIS__` row (`FROM __THIS__, __STATE__` with no `ON`, i.e. `RelNode::CrossJoin`). A static table (`LookupIndex`, keyed lookup) is the wrong tool for something with no join key at all. Making it a row table means:
- No Arrow serialization step for state — the synthesized pydantic `StateModel` instance passed straight into `row_tables`/`.infer()`'s `tables` dict, no `pa.Table.from_pylist()` round-trip.
- `.infer()` supplies `"__STATE__": [self._state]` (the single instance, wrapped in a 1-element list) on every call — `CrossJoin` of 1×N naturally repeats it for every `__THIS__` row. No new Rust-side machinery needed; `RelNode::CrossJoin` already does exactly this.

## State Key Naming and Dedup

State dict keys are `{fn}_{col}`, lowercase, with the column's table qualifier stripped (`AVG(t.age)` and `AVG(age)` both key to `avg_age`) and **no leading underscore** — Pydantic v2 treats any leading-underscore field name as a private attribute (excluded from `model_fields`, unsettable via `model_validate`/constructor kwargs), so a field actually named `__avg_age` or `_avg_age` would be invisible to both `create_model` and `InferFn`'s `getattr`-based row conversion. The double-underscore *table* names (`__THIS__`, `__STATE__`) are unaffected — those are plain string keys in `row_tables`/`tables` dicts, not attribute names, so no such restriction applies to them.

This is a change from today's alias-keyed `_state.py` (`state["age_norm"] = ...`): if the same `(fn, col)` pair appears in two different projections (e.g. `age / AVG(age) OVER () AS age_norm, AVG(age) OVER () AS age_avg`), it's computed and stored exactly once, and both projections' rewritten SQL reference the same `__STATE__.avg_age` field. `extract_state` collects the distinct `(fn, col)` pairs first, then runs one DataFusion query per distinct pair (not per occurrence).

## Error Handling

No new error variants — everything routes through errors that already exist:
- Partitioned window aggregate in source SQL → `NotImplementedError` raised directly in `_state.py::extract_state`, before any DataFusion/Rust work happens. Clear, Python-side, not a Rust build error.
- Unknown column, incompatible `this_model` type, malformed SQL, etc. → `InferFn`'s existing build-time `ValueError` (from `plan.rs`/`schema.rs` validation) — not duplicated in Python.
- Missing `__THIS__` in source SQL (user forgot the required table name) → surfaces naturally as `InferFn`'s "Unknown table in FROM clause" / column-resolution error; no special-cased message needed for v1.

## Non-Goals

- **`OVER (PARTITION BY ...)` support.** Explicitly deferred — see Scope. Will require Rust-side window/lookup support first.
- **Arbitrary `FROM` table names.** `__THIS__` is a required, hardcoded convention in this version — no rewriting of user-chosen table names.
- **Validating `this_model` against `table`'s actual schema before construction.** Trusting `InferFn`'s own build-time validation is sufficient (see Public API).
- **Multi-table `SQLTransform` inputs.** `__THIS__` + `__STATE__` are the only two tables in play; joins against other row/static tables are out of scope for `SQLTransform` in this pass (still fully available directly via `InferFn` itself).

## Testing Strategy

Port `_state_test.py`, `_codegen_test.py`, and `__init___test.py`'s existing non-partitioned cases onto the new pipeline; drop or explicitly mark-deferred the partitioned-aggregate tests (`test_extract_partitioned_window_agg`, `test_generate_partitioned_window_agg`, `test_partitioned_agg_transform`, `test_partitioned_single_row_inference`, and the partitioned assertion inside `test_e2e_three_transforms`) — replace one of them with a test asserting `NotImplementedError` on partitioned input. New coverage:

| Test | Covers |
|---|---|
| Bare `AVG(...) OVER ()` end-to-end fit/transform/infer | Existing behavior preserved through the new pipeline |
| Two projections referencing the same `(fn, col)` | Computed once, both rewritten refs point at the same `__STATE__` field |
| `this_model` omitted | Synthesized correctly from `table.schema` (types + nullability) |
| `this_model` supplied, compatible | Used as-is, `InferFn` builds successfully |
| `this_model` missing a column the query references | `InferFn`'s build-time `ValueError` propagates (Rust's row-table validation only checks column *presence*, not type — a wrong-typed-but-present field is not a build-time error here, unlike `output_model`) |
| Partitioned `OVER (PARTITION BY ...)` in source SQL | `NotImplementedError` from `extract_state`, before any Rust call |
| `self._state` is a real pydantic instance | Typed access works (e.g. `state.__avg_age` is a `float`), not a dict |
| SQL missing `__THIS__` | Clear error surfaces (whatever `InferFn` itself raises) |
