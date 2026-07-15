# Split `transform` into a DataFusion batch path and a Rust inference path — Design

**Goal:** Give `SQLTransform` two execution backends over the same fitted
transform: `transform()` runs vectorized batch through **DataFusion**, while
`infer()`/`infer_batch()` run row-at-a-time through the **Rust `InferFn`**. Both
apply the *frozen* training state and, on the normal numeric path, return
identical values — they differ only in execution strategy and I/O shape.

## Motivation

Today `transform()` already goes through the Rust `InferFn` row-at-a-time, even
for large batches. That is the wrong engine for offline/batch work: DataFusion is
columnar and vectorized. Conversely, low-latency online inference wants the lean
Rust interpreter, not a SQL engine. VISION.md's two goals map directly onto this
split — "easy authoring" (fit) and "fast inference" (both batch and online).

The split does **not** change transform semantics: a window aggregate like
`MEAN(age) OVER ()` resolves to the value frozen at `fit()` time, not recomputed
over the input batch. This preserves sklearn fit/transform semantics (no
test-set leakage) and keeps both engines numerically equivalent.

## Architecture & data flow

Both paths consume `fit()`'s outputs: the rewritten SQL and the frozen state.
The split happens only at execution time.

```
fit(train_table):
  tree      = parse_and_validate(sql)
  windows   = find_window_aggregates(tree)
  state     = extract_state(windows, ctx, "__THIS__")   # frozen training stats
  rewritten = rewrite_sql(tree, windows)                # __THIS__ CROSS JOIN __STATE__
  store: self._rewritten_sql, self._state, self._infer_fn (Rust, built from rewritten)

transform(batch_table)  -> DataFusion:
  ctx = fresh SessionContext
  ctx.register __THIS__  = batch_table
  ctx.register __STATE__ = _state_to_table(state)       # exactly one row, frozen
  return ctx.sql(self._rewritten_sql).collect()  -> pa.Table

infer(row) / infer_batch(rows)  -> Rust InferFn:
  self._infer_fn.infer({"__THIS__": rows, "__STATE__": [state]})
```

**Key invariant:** `__STATE__` is the same one-row frozen table in both paths.
That is what guarantees `transform(batch)` and `infer_batch(rows)` agree on the
normal numeric path, and it becomes an explicit equivalence test.

`fit()` gains one new stored field, `self._rewritten_sql` — today the rewritten
SQL is discarded after building `InferFn`. Everything else already exists.

## Public API

```python
def transform(self, table: pa.Table, /) -> pa.Table
    # DataFusion batch. Fresh SessionContext per call.
    # Registers __THIS__ = table, __STATE__ = one-row state table.
    # Runs self._rewritten_sql, returns the collected pa.Table.
    # Raises RuntimeError if not fitted.

def infer(self, row: dict[str, Any] | BaseModel, /) -> BaseModel
    # Rust single-row inference. Returns the typed output_model instance.
    # Raises RuntimeError if not fitted.

def infer_batch(self, rows: list[dict[str, Any] | BaseModel], /) -> list[BaseModel]
    # Rust many-rows inference. Returns list of typed output_model instances.
    # Raises RuntimeError if not fitted.
```

Behavior:

- All three require `fit()` first; otherwise `RuntimeError` (same guard as today).
- **Input normalization** for `infer`/`infer_batch`: a `dict` becomes
  `SimpleNamespace(**d)`; a `BaseModel` becomes `SimpleNamespace(**m.model_dump())`.
  Mixed lists are allowed in `infer_batch`.
- **Return types:** `transform` is pyarrow in / pyarrow out (sklearn-shaped,
  columnar). `infer`/`infer_batch` accept dict-or-model in and return the typed
  `output_model` Pydantic instance(s) that `InferFn` already builds; callers who
  want a dict call `.model_dump()`.
- **v0, no back-compat** (project rule): `transform` keeps its name but swaps
  engine. The private `_infer`/`_infer_rows` are removed and replaced by public
  `infer`/`infer_batch`. A private `_infer_rows` helper may remain as shared
  plumbing under both.

## Module structure

`__init__.py` stays thin and orchestrates; the DataFusion batch execution moves
into its own focused, independently testable unit.

```
sql_transform/_batch.py        (new)  — run_batch(rewritten_sql, table, state) -> pa.Table
                                         _state_to_table(state) -> pa.Table (one row)
sql_transform/_batch_test.py   (new)  — DataFusion path, incl. empty-state / empty-batch
sql_transform/__init__.py      (mod)  — fit() stashes _rewritten_sql; transform/infer/infer_batch
sql_transform/__init___test.py (mod)  — rename _infer -> infer; add infer_batch + equivalence + xfail
```

`_batch.py` depends only on `datafusion`, `pyarrow`, and the state model — no
coupling to the Rust interpreter or to sqlglot. `__init__.py` wires the two
backends to the fitted state.

## Edge cases

- **Empty state** (no window aggregates, e.g. `SELECT age AS x FROM __THIS__`):
  `state.model_dump() == {}`. A zero-column Arrow table cannot hold one row, so
  `_state_to_table` emits a one-row placeholder `{"__state_marker__": [0]}` when
  the state is empty. The rewritten SQL still says `CROSS JOIN __STATE__`, and
  the explicit projection never selects the marker, so it disappears from the
  output. The same builder handles the non-empty case (add the real fields).
- **Empty batch** (zero input rows): DataFusion returns zero rows with the
  correct output schema — strictly better than today's `pa.table({})`, which
  loses the schema.
- **Error-semantics divergence** (see below): on the *normal* numeric path the
  two engines are verified identical (including float division by zero → `inf`
  on both, and integer truncation on both). They diverge on **error type**: an
  integer division/modulo by zero makes the Rust `InferFn` raise a clean
  `ValueError`, while DataFusion raises its own `Exception`
  ("DataFusion error: Arrow error: Divide by zero error"). This is accepted for
  now, documented in VISION.md, and pinned by an `xfail` test rather than left
  implicit.

## Testing

**Passing — the split works:**

- Existing `transform` tests now exercise the DataFusion path unchanged and must
  still pass (same numbers) — free cross-engine coverage.
- Rename `_infer` usages to `infer`; assert on typed model attributes
  (`result.age_norm`) rather than dict keys.
- New `infer_batch` test: list in -> list of typed models out.
- New pydantic-input test: `infer(ThisRow(age=40))` and a `BaseModel` in a batch.
- New **cross-engine equivalence** test: `transform(batch)` and
  `infer_batch(rows)` produce identical values on the normal numeric path.
- `_batch_test.py`: empty-state placeholder produces correct output; empty batch
  preserves schema.

**The divergence gap (tracked, not swept under):**

1. Add `test_transform_raises_clean_valueerror_on_div_by_zero`: fit on integer
   columns, then `transform` a batch that divides by zero
   (`SELECT a / b AS x FROM __THIS__`, batch `b == 0`). Assert `transform` raises
   `ValueError` — the same clean error the Rust `infer` path raises.
2. Run it and **confirm it fails** — the Rust path raises a clean `ValueError`,
   but DataFusion raises its own `Exception` ("DataFusion error: Arrow error:
   Divide by zero error"), which `pytest.raises(ValueError)` does not catch.
3. Mark it `@pytest.mark.xfail(reason="batch (DataFusion) surfaces its own error
   type, not the clean ValueError the Rust inference path raises — see
   VISION.md", strict=True)`. It keeps running, stays red-tracked, and auto-flips
   to a pass (failing loudly) the day the gap is closed. A plain `skip` would
   silently hide it, so `xfail(strict=True)` is used instead.

**VISION.md roadmap item (new):** "Unify batch (DataFusion) vs inference (Rust)
*error* semantics: an integer division/modulo by zero raises a clean `ValueError`
from the Rust `InferFn` but a raw DataFusion `Exception` from the batch path.
Normal numeric results are already identical across engines. Tracked by the
`xfail` test `test_transform_raises_clean_valueerror_on_div_by_zero`."

## Non-goals

- No change to `fit()`'s semantics or to state extraction.
- No new SQL constructs; scope is identical to today's supported subset.
- Not resolving the engine divergence now — only capturing it as a tracked TODO.
- No async / streaming / out-of-core batch execution beyond what DataFusion's
  `ctx.sql(...).collect()` already provides.
