# Transform/Infer Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `SQLTransform`'s single Rust-backed `transform()` into a DataFusion-backed batch `transform()` plus Rust-backed `infer()`/`infer_batch()`, both applying the same frozen fit-time state.

**Architecture:** `fit()` already produces a rewritten SQL (`__THIS__ CROSS JOIN __STATE__`) and a frozen state model. The two execution paths share those: `transform()` runs the rewritten SQL through DataFusion with the frozen state as a one-row `__STATE__` table (vectorized batch); `infer()`/`infer_batch()` run it through the Rust `InferFn` row-at-a-time (low latency). Same rewritten SQL + same frozen state → identical values on the normal numeric path.

**Tech Stack:** Python 3.13, DataFusion (`datafusion` package, batch execution), the Rust `InferFn` (pyo3), Pydantic v2, pyarrow, pytest.

## Global Constraints

- v0, no backward compatibility required — make breaking API changes directly, no compat shims. The private `_infer`/`_infer_rows` are removed and replaced by public `infer`/`infer_batch`.
- Transform semantics are unchanged: a window aggregate resolves to the value frozen at `fit()` time, never recomputed over the input batch.
- No new error *types* introduced by this change except where explicitly tracked: the batch path currently raises a raw DataFusion `Exception` on integer div/mod by zero (vs the Rust path's clean `ValueError`) — this gap is captured by an `xfail` test, not fixed here.
- `transform` is pyarrow in / pyarrow out. `infer`/`infer_batch` accept `dict | BaseModel` in and return the typed `output_model` Pydantic instance(s).
- All three public methods require `fit()` first; otherwise `RuntimeError`.
- Every existing test must keep passing (the existing `transform` tests now exercise the DataFusion path unchanged), except the deliberate `_infer` → `infer` renames in Task 2.

---

## File Structure

```
sql_transform/_batch.py        (new)  — run_batch(), _state_to_table(): DataFusion batch engine
sql_transform/_batch_test.py   (new)  — tests for _batch.py
sql_transform/__init__.py      (mod)  — fit() stashes _rewritten_sql; transform()=DataFusion; infer()/infer_batch()=Rust
sql_transform/__init___test.py (mod)  — rename _infer→infer; add infer_batch/pydantic/equivalence/xfail tests
VISION.md                      (mod)  — "how it works today" reflects two engines; new roadmap item
README.md                      (mod)  — Quick Start shows infer()/infer_batch()
```

---

### Task 1: `_batch.py` — DataFusion batch execution engine

**Files:**
- Create: `sql_transform/_batch.py`
- Test: `sql_transform/_batch_test.py`

**Interfaces:**
- Consumes: nothing from other tasks (leaf module; depends only on `datafusion`, `pyarrow`, `pydantic`).
- Produces:
  - `run_batch(rewritten_sql: str, table: pa.Table, state: pydantic.BaseModel) -> pa.Table` — used by Task 2 (`__init__.py`).
  - `_state_to_table(state: pydantic.BaseModel) -> pa.Table` — internal, tested directly via `run_batch`.

- [ ] **Step 1: Write the failing tests**

Create `sql_transform/_batch_test.py`:

```python
"""Tests for the DataFusion batch execution path."""

import pyarrow as pa

from sql_transform._batch import run_batch
from sql_transform._schema import synthesize_state_model


def _state(values: dict[str, float]):
    model = synthesize_state_model(values)
    return model(**values)


def test_run_batch_applies_frozen_state():
    state = _state({"avg_age": 30.0})
    table = pa.table({"age": [30.0, 60.0]})
    sql = (
        "SELECT __THIS__.age / __STATE__.avg_age AS age_norm "
        "FROM __THIS__ CROSS JOIN __STATE__"
    )
    out = run_batch(sql, table, state)
    assert out.column("age_norm").to_pylist() == [1.0, 2.0]


def test_run_batch_empty_state_uses_placeholder():
    state = _state({})  # no window aggregates -> zero-field state model
    table = pa.table({"age": [1, 2, 3]})
    sql = "SELECT __THIS__.age AS age FROM __THIS__ CROSS JOIN __STATE__"
    out = run_batch(sql, table, state)
    assert out.column("age").to_pylist() == [1, 2, 3]
    assert out.schema.names == ["age"]  # placeholder marker column absent


def test_run_batch_empty_batch_preserves_schema():
    state = _state({"avg_age": 30.0})
    table = pa.table({"age": pa.array([], type=pa.float64())})
    sql = (
        "SELECT __THIS__.age / __STATE__.avg_age AS age_norm "
        "FROM __THIS__ CROSS JOIN __STATE__"
    )
    out = run_batch(sql, table, state)
    assert out.num_rows == 0
    assert out.schema.names == ["age_norm"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest sql_transform/_batch_test.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sql_transform._batch'`

- [ ] **Step 3: Write the implementation**

Create `sql_transform/_batch.py`:

```python
"""DataFusion batch execution for a fitted SQLTransform.

Runs the rewritten SQL (`__THIS__ CROSS JOIN __STATE__`) over a batch table,
using the frozen fit-time state as a one-row `__STATE__` table. This is the
vectorized counterpart to the row-at-a-time Rust `InferFn` path -- same
rewritten SQL, same frozen state, so both produce identical values on the
normal numeric path.
"""

from __future__ import annotations

import datafusion
import pyarrow as pa
from pydantic import BaseModel


def run_batch(
    rewritten_sql: str,
    table: pa.Table,
    state: BaseModel,
) -> pa.Table:
    """Execute `rewritten_sql` against `table` (registered as __THIS__) and the
    frozen `state` (registered as a one-row __STATE__ table) via DataFusion,
    returning the result as a pyarrow Table."""
    ctx = datafusion.SessionContext()
    ctx.from_arrow(table, name="__THIS__")
    ctx.from_arrow(_state_to_table(state), name="__STATE__")
    df = ctx.sql(rewritten_sql)
    # collect() returns [] for a zero-row result, and pa.Table.from_batches([])
    # raises -- so pass the DataFrame's schema explicitly to preserve it.
    return pa.Table.from_batches(df.collect(), schema=df.schema())


def _state_to_table(state: BaseModel) -> pa.Table:
    """Build the one-row __STATE__ table from a frozen state model.

    A zero-field state (a query with no window aggregates) would produce a
    zero-column Arrow table, which cannot hold one row; emit a single
    placeholder column instead. The rewritten SQL never selects it, so it does
    not appear in the output."""
    data = {key: [value] for key, value in state.model_dump().items()}
    if not data:
        data = {"__state_marker__": [0]}
    return pa.table(data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest sql_transform/_batch_test.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add sql_transform/_batch.py sql_transform/_batch_test.py
git commit -m "feat: DataFusion batch execution engine (_batch.run_batch)"
```

---

### Task 2: `__init__.py` — split `transform` / `infer` / `infer_batch`

**Files:**
- Modify: `sql_transform/__init__.py`
- Modify: `sql_transform/__init___test.py`

**Interfaces:**
- Consumes: `run_batch(rewritten_sql, table, state) -> pa.Table` (Task 1); existing `parse_and_validate`, `find_window_aggregates`, `extract_state`, `rewrite_sql`, `synthesize_this_model`, `InferFn`.
- Produces:
  - `SQLTransform.transform(table: pa.Table) -> pa.Table` — DataFusion batch (public API changed engine).
  - `SQLTransform.infer(row: dict | BaseModel) -> BaseModel` — Rust single-row.
  - `SQLTransform.infer_batch(rows: list[dict | BaseModel]) -> list[BaseModel]` — Rust many-rows.

- [ ] **Step 1: Update the tests**

Make these edits to `sql_transform/__init___test.py`.

(a) Replace `test_infer_before_fit_raises_runtime_error` (uses the removed `_infer`):

```python
def test_infer_before_fit_raises_runtime_error():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age FROM __THIS__")
    with pytest.raises(RuntimeError):
        t.infer({"age": 1})
```

(b) Replace `test_single_row_inference` (rename `_infer` → `infer`, attribute access):

```python
def test_single_row_inference():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
    t.fit(pa.table({"age": [25, 30, 35]}))
    result = t.infer({"age": 40})
    assert abs(result.age_norm - 40 / 30) < 0.001
```

(c) Replace `test_from_file` (rename + attribute access):

```python
def test_from_file(tmp_path):
    from sql_transform import SQLTransform

    sql_file = tmp_path / "features.sql"
    sql_file.write_text("SELECT age / MEAN(age) OVER () AS x FROM __THIS__")

    t = SQLTransform.from_file(str(sql_file))
    t.fit(pa.table({"age": [1, 2, 3]}))
    result = t.infer({"age": 10})
    assert hasattr(result, "x")
```

(d) In `test_e2e_two_transforms_and_dedup`, replace the single-row block. Change:

```python
    row = {"age": 50, "income": 100_000}
    result = t._infer(row)

    mean_age = 32.5
    assert abs(result["age_norm"] - 50 / mean_age) < 0.001

    total_income = 260_000.0
    assert abs(result["income_share"] - 100_000 / total_income) < 0.001
```
to:
```python
    row = {"age": 50, "income": 100_000}
    result = t.infer(row)

    mean_age = 32.5
    assert abs(result.age_norm - 50 / mean_age) < 0.001

    total_income = 260_000.0
    assert abs(result.income_share - 100_000 / total_income) < 0.001
```

(e) In `test_this_model_omitted_synthesizes_from_table_schema`, change:
```python
    result = t._infer({"age": 5})
    assert result["age"] == 5
```
to:
```python
    result = t.infer({"age": 5})
    assert result.age == 5
```

(f) In `test_this_model_supplied_compatible`, change:
```python
    result = t._infer({"age": 7})
    assert result["age"] == 7
```
to:
```python
    result = t.infer({"age": 7})
    assert result.age == 7
```

(g) Add three new tests at the end of the file:

```python
def test_infer_accepts_pydantic_model():
    from sql_transform import SQLTransform

    class Row(BaseModel):
        age: int

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
    t.fit(pa.table({"age": [25, 30, 35]}), this_model=Row)
    result = t.infer(Row(age=40))
    assert abs(result.age_norm - 40 / 30) < 0.001


def test_infer_batch_returns_typed_models():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
    t.fit(pa.table({"age": [25, 30, 35]}))
    out = t.infer_batch([{"age": 40}, {"age": 50}])
    assert len(out) == 2
    assert all(isinstance(o, BaseModel) for o in out)
    assert abs(out[0].age_norm - 40 / 30) < 0.001
    assert abs(out[1].age_norm - 50 / 30) < 0.001


def test_transform_and_infer_batch_agree():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
    t.fit(pa.table({"age": [25, 30, 35]}))

    test = pa.table({"age": [40, 50, 60]})
    batch = t.transform(test)
    rows = t.infer_batch([{"age": 40}, {"age": 50}, {"age": 60}])

    assert_approx_equal(
        batch.column("age_norm").to_pylist(), [r.age_norm for r in rows]
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest sql_transform/__init___test.py -v`
Expected: FAIL — `t.infer`, `t.infer_batch` don't exist yet (`AttributeError`); the renamed/attribute-access assertions fail against the current dict-returning `_infer`.

- [ ] **Step 3: Write the implementation**

In `sql_transform/__init__.py`:

(a) Add the `run_batch` import. Replace:
```python
from sql_transform._interpreter import InferFn
from sql_transform._rewrite import rewrite_sql
from sql_transform._schema import synthesize_this_model
from sql_transform._sql import find_window_aggregates, parse_and_validate
from sql_transform._state import extract_state
```
with:
```python
from sql_transform._batch import run_batch
from sql_transform._interpreter import InferFn
from sql_transform._rewrite import rewrite_sql
from sql_transform._schema import synthesize_this_model
from sql_transform._sql import find_window_aggregates, parse_and_validate
from sql_transform._state import extract_state
```

(b) In `__init__`, add the `_rewritten_sql` field. Replace:
```python
    def __init__(self, sql: str) -> None:
        self._sql = sql
        self._state: BaseModel | None = None
        self._infer_fn: InferFn | None = None
```
with:
```python
    def __init__(self, sql: str) -> None:
        self._sql = sql
        self._state: BaseModel | None = None
        self._rewritten_sql: str | None = None
        self._infer_fn: InferFn | None = None
```

(c) In `fit()`, stash the rewritten SQL. Replace:
```python
        self._state = extract_state(windows, ctx, "__THIS__")
        rewritten_sql = rewrite_sql(tree, windows)
        self._infer_fn = InferFn(
            rewritten_sql,
            row_tables={"__THIS__": this_model, "__STATE__": type(self._state)},
            static_tables={},
        )
        return self
```
with:
```python
        self._state = extract_state(windows, ctx, "__THIS__")
        self._rewritten_sql = rewrite_sql(tree, windows)
        self._infer_fn = InferFn(
            self._rewritten_sql,
            row_tables={"__THIS__": this_model, "__STATE__": type(self._state)},
            static_tables={},
        )
        return self
```

(d) Replace the entire block from `_infer_rows` through `_infer` (the old `_infer_rows`, `transform`, and `_infer` methods) with the new `transform`, `infer`, and `infer_batch` methods:

Replace:
```python
    def _infer_rows(self, this_rows: list[SimpleNamespace]) -> list[BaseModel]:
        """Run InferFn.infer() for the given __THIS__ rows against learned state."""
        if self._infer_fn is None:
            raise RuntimeError("Must call fit() before inference")
        return self._infer_fn.infer({"__THIS__": this_rows, "__STATE__": [self._state]})

    def transform(self, table: pa.Table, /) -> pa.Table:
        """Apply transforms to batch data using learned state, via InferFn."""
        rows = table.to_pylist()
        out_rows = self._infer_rows([SimpleNamespace(**row) for row in rows])
        out_dicts = [r.model_dump() for r in out_rows]
        return (
            pa.table({k: [r[k] for r in out_dicts] for k in out_dicts[0]})
            if out_dicts
            else pa.table({})
        )

    def _infer(self, row: dict[str, Any]) -> dict[str, Any]:
        """Single-row inference via InferFn."""
        out_rows = self._infer_rows([SimpleNamespace(**row)])
        return out_rows[0].model_dump()
```
with:
```python
    def transform(self, table: pa.Table, /) -> pa.Table:
        """Batch-transform `table` through DataFusion using the frozen fit-time
        state. Runs the rewritten SQL (`__THIS__ CROSS JOIN __STATE__`)
        vectorized; returns a pyarrow Table. Use infer()/infer_batch() for
        low-latency row-at-a-time inference through the Rust engine instead."""
        if self._infer_fn is None:
            raise RuntimeError("Must call fit() before transform")
        return run_batch(self._rewritten_sql, table, self._state)

    def infer(self, row: dict[str, Any] | BaseModel, /) -> BaseModel:
        """Single-row inference through the Rust InferFn against the frozen
        state. Accepts a dict or a Pydantic model; returns the typed output
        model instance."""
        return self.infer_batch([row])[0]

    def infer_batch(
        self, rows: list[dict[str, Any] | BaseModel], /
    ) -> list[BaseModel]:
        """Many-rows inference through the Rust InferFn against the frozen
        state. Accepts dicts and/or Pydantic models; returns a list of typed
        output model instances."""
        if self._infer_fn is None:
            raise RuntimeError("Must call fit() before inference")
        this_rows = [_to_namespace(row) for row in rows]
        return self._infer_fn.infer(
            {"__THIS__": this_rows, "__STATE__": [self._state]}
        )
```

(e) Add the module-level `_to_namespace` helper. Insert it immediately after the `__all__` line (before the `class SQLTransform` line):

```python
def _to_namespace(row: dict[str, Any] | BaseModel) -> SimpleNamespace:
    """Normalize an inference input row (dict or Pydantic model) into the
    SimpleNamespace of attributes the Rust InferFn reads."""
    if isinstance(row, BaseModel):
        return SimpleNamespace(**row.model_dump())
    return SimpleNamespace(**row)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest sql_transform/__init___test.py -v`
Expected: PASS (all tests in the file, including the 3 new ones).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: All tests pass (the existing `transform` tests now run through DataFusion and still pass).

- [ ] **Step 6: Run ruff**

Run: `uv run ruff check . && uv run ruff format .`
Expected: `All checks passed!` and any reformatting applied cleanly. Re-run `uv run pytest -q` if files were reformatted; expected all pass.

- [ ] **Step 7: Commit**

```bash
git add sql_transform/__init__.py sql_transform/__init___test.py
git commit -m "feat: split transform (DataFusion batch) from infer/infer_batch (Rust)"
```

---

### Task 3: Track the engine-divergence gap + update docs

**Files:**
- Modify: `sql_transform/__init___test.py` (add the `xfail` test)
- Modify: `VISION.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: `SQLTransform.transform` (Task 2).
- Produces: nothing consumed by other tasks (test + docs only).

- [ ] **Step 1: Add the divergence xfail test**

Append to `sql_transform/__init___test.py`:

```python
@pytest.mark.xfail(
    reason="batch (DataFusion) surfaces its own error type, not the clean "
    "ValueError the Rust inference path raises -- see VISION.md",
    strict=True,
)
def test_transform_raises_clean_valueerror_on_div_by_zero():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT a / b AS x FROM __THIS__")
    t.fit(pa.table({"a": [1], "b": [1]}))
    with pytest.raises(ValueError):
        t.transform(pa.table({"a": [1], "b": [0]}))
```

- [ ] **Step 2: Run it to confirm it is a tracked expected-failure**

Run: `uv run pytest sql_transform/__init___test.py::test_transform_raises_clean_valueerror_on_div_by_zero -v`
Expected: reported as `XFAIL` (the batch path raises a DataFusion `Exception`, not `ValueError`, so `pytest.raises(ValueError)` does not catch it → the test fails → `xfail` records it as expected). It must NOT report `XPASS`; an `XPASS` under `strict=True` would fail the suite and would mean the divergence was silently closed.

- [ ] **Step 3: Add the VISION.md roadmap item and correct "how it works today"**

In `VISION.md`, replace this block under "How it works today":
```markdown
- **`transform(table)` / `_infer(row)`** — the rewritten SQL runs through a small
  Rust interpreter (`InferFn`, via pyo3), row-at-a-time, against the fitted state.
  No DataFusion, no aggregation engine, at inference time — just expression eval,
  scans, and joins against typed Pydantic rows.

This split is what makes goal 2 possible: fit pays the cost of a real query engine
once; inference pays only for a lean interpreter walking a plan.
```
with:
```markdown
- **`transform(table)`** — batch execution through DataFusion: the rewritten SQL
  (`__THIS__ CROSS JOIN __STATE__`) runs against the batch as `__THIS__` and the
  frozen fit-time state as a one-row `__STATE__` table. Vectorized, columnar.
- **`infer(row)` / `infer_batch(rows)`** — low-latency execution through the small
  Rust interpreter (`InferFn`, via pyo3), row-at-a-time, against the same frozen
  state. No SQL engine at call time — just expression eval, scans, and joins
  against typed Pydantic rows. Accepts dicts or Pydantic models; returns typed
  output models.

Both paths run the *same* rewritten SQL against the *same* frozen state, so they
return identical values on the normal numeric path. The split just picks the
engine: DataFusion for large batches, the Rust interpreter for online inference.
This is what makes goal 2 possible: fit pays the cost of a real query engine
once; inference pays only for a lean interpreter walking a plan.
```

Then add this bullet at the top of the "Open questions / roadmap candidates" list:
```markdown
- **Unify batch vs inference error semantics.** `transform` (DataFusion) and
  `infer`/`infer_batch` (Rust `InferFn`) return identical values on the normal
  numeric path, but an integer division/modulo by zero raises a clean
  `ValueError` from the Rust path and a raw DataFusion `Exception`
  ("DataFusion error: Arrow error: Divide by zero error") from the batch path.
  Tracked by the `xfail` test
  `test_transform_raises_clean_valueerror_on_div_by_zero`; closing it means
  catching DataFusion's error in `_batch.run_batch` and re-raising the same
  clean `ValueError` the interpreter raises.
```

- [ ] **Step 4: Update README.md Quick Start**

In `README.md`, replace the end of the Basic Usage block:
```python
# Fit and transform
transformer = SQLTransform(sql)
transformer.fit(data)
result = transformer.transform(data)
print(result)
```
with:
```python
# Fit, then batch-transform through DataFusion (pyarrow in / pyarrow out)
transformer = SQLTransform(sql)
transformer.fit(data)
result = transformer.transform(data)
print(result)

# Low-latency inference through the Rust engine (dict or Pydantic model in,
# typed model out). infer() for one row, infer_batch() for many.
one = transformer.infer({"feature1": 2.0, "feature2": 20})
print(one.feature1_norm)
many = transformer.infer_batch([{"feature1": 2.0, "feature2": 20}])
```

- [ ] **Step 5: Verify docs render coherently and suite is green**

Run: `uv run pytest -q`
Expected: all pass, with one `xfail` reported (the divergence test). Read `VISION.md`'s "How it works today" section once to confirm no remaining claim that `transform` runs through the Rust interpreter.

- [ ] **Step 6: Commit**

```bash
git add sql_transform/__init___test.py VISION.md README.md
git commit -m "docs+test: track batch/inference error-semantics gap; document the split"
```

---

## Post-Plan Verification

- [ ] Run `mise run check` (fmt + full test suite) and confirm it's clean, with exactly one `xfail` (`test_transform_raises_clean_valueerror_on_div_by_zero`) and no `XPASS`.
- [ ] Confirm the Rust path is gone from `transform`: `grep -n "to_pylist\|SimpleNamespace" sql_transform/__init__.py` shows `SimpleNamespace` only inside `_to_namespace` (used by `infer_batch`), and no `to_pylist` in `transform`.
- [ ] Confirm no remaining references to the removed private methods: `grep -rn "_infer_rows\|\._infer\b" sql_transform/` returns nothing.
- [ ] Confirm `transform` and `infer_batch` agree on the normal numeric path (covered by `test_transform_and_infer_batch_agree`).
