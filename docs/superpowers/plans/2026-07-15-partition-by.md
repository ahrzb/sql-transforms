# `PARTITION BY` Window Aggregates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support `AGG(col) OVER (PARTITION BY k1, …)` in `SQLTransform` — per-partition learned state (target/categorical encoding) — with real value types preserved, a strictly 1-to-1 transform, and both engines (DataFusion batch, Rust `InferFn`) agreeing that an unseen partition → NULL.

**Architecture:** At `fit()`, group window aggregates by partition-key-set and run one `GROUP BY` per group to build a per-partition pyarrow **state table** (keys + typed value columns). State tables are passed as *static* tables to `InferFn` and registered in DataFusion. The rewrite emits a **LEFT JOIN** from `__THIS__` to each state table on its keys; the global `OVER ()` state is a one-row table joined on a constant marker key. Unseen key → LEFT miss → NULL. A new Rust LEFT-lookup-join emits a NULL row on miss instead of erroring.

**Tech Stack:** Python 3.13, sqlglot, DataFusion (`datafusion` package), Rust + pyo3 (`InferFn`, built with maturin), pyarrow, Pydantic v2, pytest, cargo.

## Global Constraints

- **Strictly 1-to-1 transform:** the rewrite emits only `LEFT JOIN`s onto unique-keyed (`GROUP BY`-produced) state tables. Never an INNER join. Row count out == row count in, always.
- **Unseen partition → NULL** (LEFT miss). No fallback, no smoothing.
- **Real value types preserved:** state value columns keep their natural Arrow type (int/float/str/bool), nullable — no `float` coercion. `synthesize_state_model` is removed; state is static `pa.Table`s, not Pydantic models.
- **v0, no backward compatibility** — remove `_infer`-era state-model plumbing directly; update call sites and tests, no shims.
- State table naming is deterministic: empty key-set → `__STATE__`; `PARTITION BY city` → `__STATE_BY_city__`; composite `city, region` → `__STATE_BY_city_region__`. Same key-set → same table (dedup).
- `ORDER BY` window aggregates remain `NotImplementedError`. Multi-arg / expression aggregate arguments remain a single-plain-column `ValueError`.
- Both engines must agree on the normal path; the existing div-by-zero `xfail` stays.

---

## File Structure

```
sql_transform/_sql.py          (modify) — WindowAgg.partition_cols
sql_transform/_sql_test.py     (modify) — partition_cols tests
sql_transform/_state.py        (rewrite) — build_state_tables() -> dict[str, pa.Table]; state_table_name()
sql_transform/_state_test.py   (rewrite) — typed per-partition state tables
sql_transform/_rewrite.py      (modify) — LEFT JOINs on keys + global marker
sql_transform/_rewrite_test.py (modify) — LEFT-join expected SQL
src/plan.rs                    (modify) — LEFT lookup join (outer flag, null row on miss)
src/lookup.rs                  (modify) — LookupIndex.value_columns
tests/test_interpreter.py      (modify) — LEFT lookup join tests
sql_transform/_batch.py        (modify) — run_batch registers all state tables
sql_transform/__init__.py      (modify) — fit stores _state_tables; static_tables plumbing
sql_transform/__init___test.py (modify) — end-to-end partition-by + typed + equivalence
```

---

### Task 1: `_sql.py` — `WindowAgg.partition_cols`

**Files:**
- Modify: `sql_transform/_sql.py`
- Test: `sql_transform/_sql_test.py`

**Interfaces:**
- Consumes: nothing new (sqlglot only).
- Produces: `WindowAgg.partition_cols: tuple[str, ...]` (empty tuple for `OVER ()`), consumed by Tasks 2 and 3.

- [ ] **Step 1: Write the failing tests**

Add to `sql_transform/_sql_test.py`:

```python
def test_find_window_aggregates_partition_cols_empty_for_bare_over():
    tree = parse_and_validate("SELECT AVG(age) OVER () AS x FROM __THIS__")
    windows = find_window_aggregates(tree)
    assert windows[0].partition_cols == ()


def test_find_window_aggregates_single_partition_col():
    tree = parse_and_validate(
        "SELECT AVG(target) OVER (PARTITION BY city) AS x FROM __THIS__"
    )
    windows = find_window_aggregates(tree)
    assert windows[0].partition_cols == ("city",)
    assert windows[0].has_partition is True


def test_find_window_aggregates_composite_partition_cols():
    tree = parse_and_validate(
        "SELECT AVG(target) OVER (PARTITION BY city, region) AS x FROM __THIS__"
    )
    windows = find_window_aggregates(tree)
    assert windows[0].partition_cols == ("city", "region")


def test_find_window_aggregates_rejects_non_column_partition():
    tree = parse_and_validate(
        "SELECT AVG(target) OVER (PARTITION BY city || 'x') AS y FROM __THIS__"
    )
    with pytest.raises(ValueError, match="PARTITION BY"):
        find_window_aggregates(tree)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest sql_transform/_sql_test.py -v`
Expected: FAIL — `WindowAgg` has no `partition_cols` attribute.

- [ ] **Step 3: Write the implementation**

In `sql_transform/_sql.py`, add the field to `WindowAgg` (after `col`):

```python
@dataclass(frozen=True)
class WindowAgg:
    """A single window-aggregate reference found in a SELECT list.

    `node` is the actual sqlglot Window node -- rewrite_sql() matches
    against it by identity to know which node to replace, so callers must
    not re-parse the SQL between find_window_aggregates() and using the
    returned WindowAggs.
    """

    node: exp.Window
    fn: str
    col: str
    partition_cols: tuple[str, ...]
    has_partition: bool
    has_order: bool
```

In `find_window_aggregates`, before building the `WindowAgg`, extract the partition columns (each must be a plain column):

```python
        col = args[0].name

        partition_by = node.args.get("partition_by") or []
        partition_cols: list[str] = []
        for p in partition_by:
            if not isinstance(p, exp.Column):
                raise ValueError(
                    "PARTITION BY must be a list of plain columns: "
                    f"{node.sql()!r}"
                )
            partition_cols.append(p.name)

        windows.append(
            WindowAgg(
                node=node,
                fn=fn,
                col=col,
                partition_cols=tuple(partition_cols),
                has_partition=bool(node.args.get("partition_by")),
                has_order=bool(node.args.get("order")),
            )
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest sql_transform/_sql_test.py -v`
Expected: PASS (existing + 4 new).

- [ ] **Step 5: Commit**

```bash
git add sql_transform/_sql.py sql_transform/_sql_test.py
git commit -m "feat: WindowAgg carries partition_cols from PARTITION BY"
```

---

### Task 2: `_state.py` — build typed per-partition state tables

**Files:**
- Modify (rewrite): `sql_transform/_state.py`
- Test (rewrite): `sql_transform/_state_test.py`

**Interfaces:**
- Consumes: `WindowAgg` incl. `partition_cols` (Task 1).
- Produces:
  - `state_table_name(partition_cols: tuple[str, ...]) -> str` — used by Task 3 and internally.
  - `state_key(fn_name, col_name) -> str` — unchanged, used by Task 3.
  - `build_state_tables(windows: list[WindowAgg], ctx, table_name: str) -> dict[str, pa.Table]` — used by Tasks 5. Replaces `extract_state`.

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `sql_transform/_state_test.py`:

```python
"""Tests for building typed per-partition state tables."""

import datafusion
import pytest

from sql_transform._sql import find_window_aggregates, parse_and_validate
from sql_transform._state import build_state_tables, state_key, state_table_name


def _windows(sql: str):
    return find_window_aggregates(parse_and_validate(sql))


def _ctx(data: dict, name: str = "__THIS__"):
    ctx = datafusion.SessionContext()
    ctx.from_pydict(data, name=name)
    return ctx


def test_state_key_lowercases():
    assert state_key("AVG", "age") == "avg_age"
    assert state_key("avg", "AGE") == "avg_age"


def test_state_table_name_global_and_partitioned():
    assert state_table_name(()) == "__STATE__"
    assert state_table_name(("city",)) == "__STATE_BY_city__"
    assert state_table_name(("city", "region")) == "__STATE_BY_city_region__"


def test_global_state_table_has_marker_and_value():
    ctx = _ctx({"age": [25, 30, 35]})
    windows = _windows("SELECT age / MEAN(age) OVER () AS x FROM __THIS__")
    tables = build_state_tables(windows, ctx, "__THIS__")

    assert set(tables) == {"__STATE__"}
    t = tables["__STATE__"]
    assert t.num_rows == 1
    assert t.column("avg_age").to_pylist() == [30.0]
    assert t.column("__state_marker__").to_pylist() == [0]


def test_partition_state_table_per_key():
    ctx = _ctx(
        {"city": ["a", "b", "a", "b"], "target": [1.0, 3.0, 2.0, 4.0]}
    )
    windows = _windows(
        "SELECT MEAN(target) OVER (PARTITION BY city) AS enc FROM __THIS__"
    )
    tables = build_state_tables(windows, ctx, "__THIS__")

    assert set(tables) == {"__STATE_BY_city__"}
    t = tables["__STATE_BY_city__"]
    got = dict(zip(t.column("city").to_pylist(), t.column("avg_target").to_pylist()))
    assert got == {"a": 1.5, "b": 3.5}
    assert "__state_marker__" not in t.schema.names


def test_partition_value_type_preserved_for_count():
    ctx = _ctx({"city": ["a", "a", "b"], "target": [1, 2, 3]})
    windows = _windows(
        "SELECT COUNT(target) OVER (PARTITION BY city) AS n FROM __THIS__"
    )
    tables = build_state_tables(windows, ctx, "__THIS__")
    t = tables["__STATE_BY_city__"]
    # COUNT is an integer count-encoding, not a float.
    assert pa_is_integer(t.column("count_target").type)


def pa_is_integer(t) -> bool:
    import pyarrow as pa

    return pa.types.is_integer(t)


def test_dedup_repeated_aggregate_in_group():
    ctx = _ctx({"city": ["a", "b"], "target": [1.0, 2.0]})
    windows = _windows(
        "SELECT MEAN(target) OVER (PARTITION BY city) AS a, "
        "MEAN(target) OVER (PARTITION BY city) AS b FROM __THIS__"
    )
    tables = build_state_tables(windows, ctx, "__THIS__")
    t = tables["__STATE_BY_city__"]
    # One value column despite two projections.
    assert [n for n in t.schema.names if n != "city"] == ["avg_target"]


def test_distinct_key_sets_distinct_tables():
    ctx = _ctx(
        {
            "city": ["a", "b"],
            "region": ["x", "y"],
            "target": [1.0, 2.0],
        }
    )
    windows = _windows(
        "SELECT MEAN(target) OVER (PARTITION BY city) AS a, "
        "SUM(target) OVER (PARTITION BY region) AS b FROM __THIS__"
    )
    tables = build_state_tables(windows, ctx, "__THIS__")
    assert set(tables) == {"__STATE_BY_city__", "__STATE_BY_region__"}


def test_composite_partition_key():
    ctx = _ctx(
        {
            "city": ["a", "a"],
            "region": ["x", "x"],
            "target": [2.0, 4.0],
        }
    )
    windows = _windows(
        "SELECT MEAN(target) OVER (PARTITION BY city, region) AS e FROM __THIS__"
    )
    tables = build_state_tables(windows, ctx, "__THIS__")
    t = tables["__STATE_BY_city_region__"]
    assert set(t.schema.names) == {"city", "region", "avg_target"}
    assert t.column("avg_target").to_pylist() == [3.0]


def test_no_windows_returns_empty_dict():
    ctx = _ctx({"age": [1, 2, 3]})
    windows = _windows("SELECT age AS x FROM __THIS__")
    assert build_state_tables(windows, ctx, "__THIS__") == {}


def test_order_by_still_not_implemented():
    ctx = _ctx({"age": [1, 2, 3]})
    windows = _windows(
        "SELECT MEAN(age) OVER (ORDER BY age) AS r FROM __THIS__"
    )
    with pytest.raises(NotImplementedError):
        build_state_tables(windows, ctx, "__THIS__")


def test_case_collision_raises():
    ctx = _ctx({"age": [1.0], "Age": [2.0]})
    windows = _windows(
        'SELECT MEAN(age) OVER () + MEAN("Age") OVER () AS c FROM __THIS__'
    )
    with pytest.raises(ValueError, match="[Aa]mbiguous"):
        build_state_tables(windows, ctx, "__THIS__")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest sql_transform/_state_test.py -v`
Expected: FAIL — `build_state_tables` / `state_table_name` don't exist.

- [ ] **Step 3: Write the implementation**

Replace the full contents of `sql_transform/_state.py`:

```python
"""Build typed per-partition state tables from SQLTransform's window aggregates.

Groups window aggregates by their PARTITION BY key-set and runs one DataFusion
GROUP BY query per group, producing a pyarrow state table (key columns + one
value column per distinct (fn, col)) whose value columns keep their natural
Arrow type -- no float coercion. The global OVER () group (empty key-set) gets a
one-row table plus a constant __state_marker__ column so the rewrite can LEFT
JOIN it uniformly. State tables are consumed as static tables by both engines
(InferFn static_tables and DataFusion registration) -- there is no Pydantic
state model anymore.
"""

from __future__ import annotations

import datafusion
import pyarrow as pa

from sql_transform._sql import WindowAgg

STATE_MARKER = "__state_marker__"


def state_key(fn_name: str, col_name: str) -> str:
    """The state value-column name for an aggregate function + column,
    e.g. state_key("AVG", "age") == "avg_age"."""
    return f"{fn_name.lower()}_{col_name.lower()}"


def state_table_name(partition_cols: tuple[str, ...]) -> str:
    """Deterministic state-table name for a partition-key-set. Empty key-set
    (the global OVER () state) -> "__STATE__"; otherwise
    "__STATE_BY_<cols joined by _>__"."""
    if not partition_cols:
        return "__STATE__"
    return "__STATE_BY_" + "_".join(partition_cols) + "__"


def build_state_tables(
    windows: list[WindowAgg],
    ctx: datafusion.SessionContext,
    table_name: str,
) -> dict[str, pa.Table]:
    """Return a dict of state-table-name -> pyarrow table, one per distinct
    PARTITION BY key-set present in `windows`. Value columns keep their real
    Arrow type. Raises NotImplementedError for ORDER BY window aggregates and
    ValueError for a case-collision between two aggregates in the same table."""
    # Group windows by partition-key-set, preserving discovery order.
    groups: dict[tuple[str, ...], list[WindowAgg]] = {}
    for w in windows:
        if w.has_order:
            raise NotImplementedError(
                "ORDER BY window aggregates are not yet supported by "
                "the Rust-backed SQLTransform pipeline"
            )
        groups.setdefault(w.partition_cols, []).append(group_member(w))

    tables: dict[str, pa.Table] = {}
    for partition_cols, members in groups.items():
        # Dedup (fn, col) within the group; detect state_key collisions.
        selected: dict[str, tuple[str, str]] = {}
        for fn_name, col_name in members:
            key = state_key(fn_name, col_name)
            existing = selected.get(key)
            if existing is not None and existing != (fn_name, col_name):
                raise ValueError(
                    f"Ambiguous window aggregate: {fn_name}({col_name}) "
                    f"normalizes to the same state key {key!r} as another "
                    "aggregate in this query -- names that differ only by case "
                    "aren't distinguished"
                )
            selected[key] = (fn_name, col_name)

        value_exprs = [
            f'{fn_name}("{col_name}") AS {key}'
            for key, (fn_name, col_name) in selected.items()
        ]

        if partition_cols:
            key_list = ", ".join(f'"{c}"' for c in partition_cols)
            sql = (
                f"SELECT {key_list}, {', '.join(value_exprs)} "
                f"FROM {table_name} GROUP BY {key_list}"
            )
            table = _collect(ctx, sql)
        else:
            sql = f"SELECT {', '.join(value_exprs)} FROM {table_name}"
            table = _collect(ctx, sql)
            table = table.append_column(
                STATE_MARKER, pa.array([0], type=pa.int64())
            )

        tables[state_table_name(partition_cols)] = table

    return tables


def group_member(w: WindowAgg) -> tuple[str, str]:
    """The (fn, col) pair identifying an aggregate within its partition group.

    Preserves the column's real case -- it is quoted into the DataFusion query,
    and two columns differing only by case are genuinely distinct here (the
    state_key() name-collision check is separate and intentional)."""
    return (w.fn, w.col)


def _collect(ctx: datafusion.SessionContext, sql: str) -> pa.Table:
    df = ctx.sql(sql)
    return pa.Table.from_batches(df.collect(), schema=df.schema())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest sql_transform/_state_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sql_transform/_state.py sql_transform/_state_test.py
git commit -m "feat: build typed per-partition state tables (no float coercion, no state model)"
```

---

### Task 3: `_rewrite.py` — LEFT JOINs to state tables

**Files:**
- Modify: `sql_transform/_rewrite.py`
- Test (rewrite): `sql_transform/_rewrite_test.py`

**Interfaces:**
- Consumes: `WindowAgg` incl. `partition_cols` (Task 1); `state_table_name`, `state_key`, `STATE_MARKER` (Task 2).
- Produces: `rewrite_sql(select: exp.Select, windows: list[WindowAgg]) -> str` — same signature; now emits LEFT JOINs. Used by Task 5.

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `sql_transform/_rewrite_test.py`:

```python
"""Tests for SQL rewriting into LEFT-joined state-table SQL."""

import pytest

from sql_transform._rewrite import rewrite_sql
from sql_transform._sql import find_window_aggregates, parse_and_validate


def _rewrite(sql: str) -> str:
    tree = parse_and_validate(sql)
    windows = find_window_aggregates(tree)
    return rewrite_sql(tree, windows)


def test_simple_column_no_state():
    # No window aggregates -> no state join at all.
    sql = _rewrite("SELECT age AS just_age FROM __THIS__")
    assert sql == "SELECT __THIS__.age AS just_age FROM __THIS__"


def test_global_agg_left_joins_marker():
    sql = _rewrite("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
    assert sql == (
        "SELECT __THIS__.age / __STATE__.avg_age AS age_norm "
        "FROM __THIS__ LEFT JOIN __STATE__ ON __STATE__.__state_marker__ = 0"
    )


def test_partition_agg_left_joins_on_key():
    sql = _rewrite(
        "SELECT MEAN(target) OVER (PARTITION BY city) AS enc FROM __THIS__"
    )
    assert sql == (
        "SELECT __STATE_BY_city__.avg_target AS enc "
        "FROM __THIS__ LEFT JOIN __STATE_BY_city__ "
        "ON __THIS__.city = __STATE_BY_city__.city"
    )


def test_composite_partition_key_anded():
    sql = _rewrite(
        "SELECT MEAN(target) OVER (PARTITION BY city, region) AS e FROM __THIS__"
    )
    assert sql == (
        "SELECT __STATE_BY_city_region__.avg_target AS e "
        "FROM __THIS__ LEFT JOIN __STATE_BY_city_region__ "
        "ON __THIS__.city = __STATE_BY_city_region__.city "
        "AND __THIS__.region = __STATE_BY_city_region__.region"
    )


def test_mixed_global_and_partition():
    sql = _rewrite(
        "SELECT age / MEAN(age) OVER () AS n, "
        "MEAN(target) OVER (PARTITION BY city) AS enc FROM __THIS__"
    )
    assert sql == (
        "SELECT __THIS__.age / __STATE__.avg_age AS n, "
        "__STATE_BY_city__.avg_target AS enc "
        "FROM __THIS__ "
        "LEFT JOIN __STATE__ ON __STATE__.__state_marker__ = 0 "
        "LEFT JOIN __STATE_BY_city__ ON __THIS__.city = __STATE_BY_city__.city"
    )


def test_unaliased_expression_raises():
    with pytest.raises(ValueError, match="needs an alias"):
        _rewrite("SELECT age / MEAN(age) OVER () FROM __THIS__")


def test_bad_column_qualifier_raises():
    with pytest.raises(ValueError, match="does not refer to __THIS__"):
        _rewrite("SELECT foo.age AS x FROM __THIS__")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest sql_transform/_rewrite_test.py -v`
Expected: FAIL — current `rewrite_sql` emits `CROSS JOIN __STATE__`, not LEFT joins.

- [ ] **Step 3: Write the implementation**

Replace the full contents of `sql_transform/_rewrite.py`:

```python
"""Rewrite SQLTransform's SQL into LEFT-joined state-table SQL for the engines.

Each window-aggregate reference becomes a column reference into its partition's
state table (state_table_name(partition_cols).state_key(fn, col)); every plain
column becomes __THIS__.col; and one LEFT JOIN per distinct partition-key-set is
appended. The global OVER () state (empty key-set) is joined on a constant marker
key. All joins are LEFT onto unique-keyed (GROUP BY) tables, so the rewrite is
strictly 1-to-1: an unseen key yields NULL, never a dropped or duplicated row.
"""

from __future__ import annotations

from sqlglot import exp

from sql_transform._sql import WindowAgg
from sql_transform._state import STATE_MARKER, state_key, state_table_name


def rewrite_sql(select: exp.Select, windows: list[WindowAgg]) -> str:
    """Return SQL equivalent to `select` with window aggregates replaced by
    state-table column references and one LEFT JOIN per partition-key-set.

    Mutates `select` in place -- callers should not reuse it afterwards."""
    window_ref = {
        id(w.node): (state_table_name(w.partition_cols), state_key(w.fn, w.col))
        for w in windows
    }

    for e in select.expressions:
        out_name = e.alias_or_name
        if not out_name:
            raise ValueError(
                f"Expression in SELECT list needs an alias (AS name): {e.sql()!r}"
            )

        for win_node in list(e.find_all(exp.Window)):
            table, col = window_ref[id(win_node)]
            win_node.replace(exp.column(col, table=table))

        for col_node in list(e.find_all(exp.Column)):
            if col_node.table and col_node.table.startswith("__STATE"):
                continue  # already rewritten above
            if col_node.table and col_node.table != "__THIS__":
                raise ValueError(
                    f"Column qualifier {col_node.table!r} does not refer "
                    "to __THIS__"
                )
            col_node.replace(exp.column(col_node.name, table="__THIS__"))

    # One LEFT JOIN per distinct partition-key-set, in first-seen order.
    seen: dict[tuple[str, ...], None] = {}
    for w in windows:
        seen.setdefault(w.partition_cols, None)

    for partition_cols in seen:
        table = state_table_name(partition_cols)
        if not partition_cols:
            on = f"{table}.{STATE_MARKER} = 0"
        else:
            on = " AND ".join(
                f"__THIS__.{c} = {table}.{c}" for c in partition_cols
            )
        select.join(exp.to_table(table), on=on, join_type="LEFT", copy=False)

    return select.sql()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest sql_transform/_rewrite_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sql_transform/_rewrite.py sql_transform/_rewrite_test.py
git commit -m "feat: rewrite emits LEFT JOINs to per-partition state tables"
```

---

### Task 4: Rust — LEFT lookup join (NULL row on miss)

**Files:**
- Modify: `src/plan.rs`, `src/lookup.rs`
- Test: `tests/test_interpreter.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (independent Rust change).
- Produces: `InferFn` accepts `LEFT JOIN ... ON` against a static table, returning a NULL-filled static row on a key miss instead of raising. Used end-to-end by Task 5.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_interpreter.py`:

```python
def test_left_lookup_join_hit_returns_value():
    import pyarrow as pa
    from types import SimpleNamespace
    from pydantic import BaseModel
    from sql_transform import InferFn

    class Row(BaseModel):
        city: str

    ref = pa.table({"city": ["a", "b"], "enc": [1.5, 3.5]})
    sql = "SELECT ref.enc FROM data LEFT JOIN ref ON data.city = ref.city"
    fn = InferFn(sql, row_tables={"data": Row}, static_tables={"ref": ref})
    out = fn.infer({"data": [SimpleNamespace(city="a")]})
    assert out[0].enc == 1.5


def test_left_lookup_join_miss_returns_null():
    import pyarrow as pa
    from types import SimpleNamespace
    from pydantic import BaseModel
    from sql_transform import InferFn

    class Row(BaseModel):
        city: str

    ref = pa.table({"city": ["a"], "enc": [1.5]})
    sql = "SELECT ref.enc FROM data LEFT JOIN ref ON data.city = ref.city"
    fn = InferFn(sql, row_tables={"data": Row}, static_tables={"ref": ref})
    out = fn.infer({"data": [SimpleNamespace(city="zzz")]})  # unseen key
    assert out[0].enc is None


def test_inner_lookup_join_miss_still_errors():
    import pyarrow as pa
    from types import SimpleNamespace
    from pydantic import BaseModel
    from sql_transform import InferFn

    class Row(BaseModel):
        city: str

    ref = pa.table({"city": ["a"], "enc": [1.5]})
    sql = "SELECT ref.enc FROM data JOIN ref ON data.city = ref.city"
    fn = InferFn(sql, row_tables={"data": Row}, static_tables={"ref": ref})
    with pytest.raises(ValueError):
        fn.infer({"data": [SimpleNamespace(city="zzz")]})
```

- [ ] **Step 2: Build and run to verify failure**

Run: `uv run maturin develop && uv run pytest tests/test_interpreter.py -v -k lookup`
Expected: the two LEFT-join tests FAIL — today `JoinOperator::LeftOuter` is rejected at build (`Unsupported JOIN type`), so `InferFn(...)` raises during construction.

- [ ] **Step 3: `src/lookup.rs` — capture value-column names**

The null row needs the static table's non-key column names even when no row matched. Add a `value_columns` field to `LookupIndex` and populate it from the pyarrow table's schema (robust to empty tables).

Change the struct:

```rust
pub struct LookupIndex {
    pub index: HashMap<Vec<Value>, HashMap<String, Value>>,
    pub value_columns: Vec<String>,
}
```

In `build_index`, before the row loop, read all column names from the table and derive the non-key ones:

```rust
    let all_columns: Vec<String> = bound
        .getattr("column_names")
        .and_then(|c| c.extract())
        .map_err(|e| InterpError::Build(format!("Failed to read static table columns: {e}")))?;
    let value_columns: Vec<String> = all_columns
        .into_iter()
        .filter(|c| !key_columns.contains(c))
        .collect();
```

and return it:

```rust
    Ok(LookupIndex { index, value_columns })
```

- [ ] **Step 4: `src/plan.rs` — thread an `outer` flag and emit NULL on miss**

(a) Add `outer: bool` to both `RelNode::Join` and `RelNode::LookupJoin` in the enum definition (around lines 45-58):

```rust
    Join {
        left: Box<RelNode>,
        right: Box<RelNode>,
        on: Vec<(Expr, Expr)>,
        outer: bool,
    },
    ...
    LookupJoin {
        input: Box<RelNode>,
        table: String,
        keys: Vec<Expr>,
        outer: bool,
    },
```

(b) In `build_join` (around line 143), accept `LeftOuter` and set `outer` on the produced `Join`. Replace the inner-join arm:

```rust
        JoinOperator::Join(constraint) | JoinOperator::Inner(constraint) => {
            let on_expr = require_on(constraint)?;
            let on = extract_equality_keys(on_expr)?;
            Ok(RelNode::Join {
                left: Box::new(left),
                right: Box::new(right),
                on,
                outer: false,
            })
        }
        JoinOperator::LeftOuter(constraint) => {
            let on_expr = require_on(constraint)?;
            let on = extract_equality_keys(on_expr)?;
            Ok(RelNode::Join {
                left: Box::new(left),
                right: Box::new(right),
                on,
                outer: true,
            })
        }
```

(c) In `optimize_rel`'s `RelNode::Join` arm (around line 288), destructure `outer`, thread it into both `LookupJoin` constructions, reject an outer row-to-row join, and keep it on the plain-join fallthrough:

```rust
        RelNode::Join { left, right, on, outer } => {
            let left = optimize_rel(*left, static_tables, specs)?;
            let right = optimize_rel(*right, static_tables, specs)?;
            let left_static = scan_table_name(&left).filter(|t| static_tables.contains(*t));
            let right_static = scan_table_name(&right).filter(|t| static_tables.contains(*t));
            match (left_static, right_static) {
                (Some(_), Some(_)) => Err(InterpError::Build(
                    "Joining two static tables together is not supported".to_string(),
                )),
                (None, Some(table)) => {
                    let table = table.to_string();
                    let (keys, key_columns) = split_keys(&on, &table)?;
                    specs.push(LookupSpec { static_table: table.clone(), key_columns });
                    Ok(RelNode::LookupJoin { input: Box::new(left), table, keys, outer })
                }
                (Some(table), None) => {
                    let table = table.to_string();
                    let (keys, key_columns) = split_keys(&on, &table)?;
                    specs.push(LookupSpec { static_table: table.clone(), key_columns });
                    Ok(RelNode::LookupJoin { input: Box::new(right), table, keys, outer })
                }
                (None, None) => {
                    if outer {
                        return Err(InterpError::Build(
                            "LEFT JOIN is only supported against a static lookup table"
                                .to_string(),
                        ));
                    }
                    Ok(RelNode::Join {
                        left: Box::new(left),
                        right: Box::new(right),
                        on,
                        outer,
                    })
                }
            }
        }
```

(d) In `execute_rel`'s `RelNode::LookupJoin` arm (around line 497), destructure `outer` and, on a miss, either insert a NULL row (outer) or raise (inner):

```rust
        RelNode::LookupJoin { input, table, keys, outer } => {
            let rows = execute_rel(input, tables, lookups)?;
            let index = lookups.get(table).ok_or_else(|| {
                InterpError::Build(format!("No lookup index built for table: {table}"))
            })?;
            let mut out = Vec::with_capacity(rows.len());
            for mut row in rows {
                let key: Vec<Value> = keys
                    .iter()
                    .map(|k| crate::expr::eval(k, &row))
                    .collect::<Result<_, _>>()?;
                match index.index.get(&key) {
                    Some(hit) => {
                        row.insert(table.clone(), hit.clone());
                    }
                    None if *outer => {
                        let null_row: HashMap<String, Value> = index
                            .value_columns
                            .iter()
                            .map(|c| (c.clone(), Value::Null))
                            .collect();
                        row.insert(table.clone(), null_row);
                    }
                    None => {
                        let key_repr: Vec<String> =
                            key.iter().map(crate::expr::display_value).collect();
                        return Err(InterpError::MissingKey(format!(
                            "No row in static table '{table}' matches key ({})",
                            key_repr.join(", ")
                        )));
                    }
                }
                out.push(row);
            }
            Ok(out)
        }
```

(e) The remaining `RelNode::Join { left, right, on }` match sites need the new field. Sites that bind fields: `execute_rel` (~line 461) and the walker (~line 632) both bind `{ left, right, on }` — add `, .. ` or `, outer: _`. Sites already using `..` (lines ~551, ~555, ~643) need no change. After editing, `cargo build` will name any site you missed — fix each by adding `..` to the pattern. Grep to enumerate: `grep -n "RelNode::Join\|RelNode::LookupJoin" src/plan.rs`.

- [ ] **Step 5: Build and run tests**

Run: `uv run maturin develop && uv run pytest tests/test_interpreter.py -v`
Expected: all pass, including the three new lookup tests. If `cargo`/`maturin` reports a missing-field or non-exhaustive-match error, fix the named site (add the `outer` field or `..`) and rebuild.

- [ ] **Step 6: Commit**

```bash
git add src/plan.rs src/lookup.rs tests/test_interpreter.py
git commit -m "feat: LEFT lookup join in the Rust interpreter (NULL row on key miss)"
```

---

### Task 5: `__init__.py` + `_batch.py` — plumb static state tables into both engines

**Files:**
- Modify: `sql_transform/__init__.py`, `sql_transform/_batch.py`
- Modify: `sql_transform/__init___test.py`

**Interfaces:**
- Consumes: `build_state_tables` (Task 2), `rewrite_sql` (Task 3), the LEFT-lookup-join `InferFn` (Task 4), `run_batch` (updated here).
- Produces: `SQLTransform` with PARTITION BY support end-to-end; unchanged public method signatures.

- [ ] **Step 1: Update `_batch.py`**

`run_batch` now registers an arbitrary set of state tables (not one `__STATE__`). Replace the full contents of `sql_transform/_batch.py`:

```python
"""DataFusion batch execution for a fitted SQLTransform.

Registers __THIS__ plus every fit-time state table, then runs the rewritten SQL
(LEFT JOINs to those state tables). DataFusion yields NULL on an unseen partition
natively -- matching the Rust engine's LEFT-lookup-join. This is the vectorized
counterpart to the row-at-a-time InferFn path.
"""

from __future__ import annotations

import datafusion
import pyarrow as pa


def run_batch(
    rewritten_sql: str,
    table: pa.Table,
    state_tables: dict[str, pa.Table],
) -> pa.Table:
    """Execute `rewritten_sql` against `table` (as __THIS__) and every state
    table (registered by name) via DataFusion, returning a pyarrow Table."""
    ctx = datafusion.SessionContext()
    ctx.from_arrow(table, name="__THIS__")
    for name, state_table in state_tables.items():
        ctx.from_arrow(state_table, name=name)
    df = ctx.sql(rewritten_sql)
    # collect() returns [] for a zero-row result, so pass the schema explicitly.
    return pa.Table.from_batches(df.collect(), schema=df.schema())
```

- [ ] **Step 2: Update the tests**

In `sql_transform/__init___test.py`:

(a0) **Delete** `test_partitioned_agg_raises_not_implemented` — it asserts `fit()`
raises `NotImplementedError` for `PARTITION BY`, which is exactly the behavior this
task removes. (ORDER BY is still covered by `_state_test.py`'s
`test_order_by_still_not_implemented`.)

(a) Replace `test_state_is_typed_pydantic_instance` (there is no state model anymore) with a state-table check:

```python
def test_state_tables_hold_typed_values():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
    t.fit(pa.table({"age": [25, 30, 35]}))
    # State is a dict of pyarrow tables, not a Pydantic model.
    assert t._state_tables["__STATE__"].column("avg_age").to_pylist() == [30.0]
```

(b) Add end-to-end PARTITION BY tests at the end of the file:

```python
def test_partition_by_target_encoding_seen_and_unseen():
    from sql_transform import SQLTransform

    t = SQLTransform(
        "SELECT MEAN(target) OVER (PARTITION BY city) AS enc FROM __THIS__"
    )
    t.fit(pa.table({"city": ["a", "b", "a", "b"], "target": [1.0, 3.0, 2.0, 4.0]}))

    seen = t.infer({"city": "a", "target": 0.0})
    assert seen.enc == 1.5

    unseen = t.infer({"city": "zzz", "target": 0.0})
    assert unseen.enc is None  # unseen partition -> NULL


def test_partition_by_count_encoding_is_integer():
    from sql_transform import SQLTransform

    t = SQLTransform(
        "SELECT COUNT(target) OVER (PARTITION BY city) AS n FROM __THIS__"
    )
    t.fit(pa.table({"city": ["a", "a", "b"], "target": [1, 2, 3]}))
    out = t.infer({"city": "a", "target": 0})
    assert out.n == 2
    assert isinstance(out.n, int)  # count encoding stays an int, not 2.0


def test_partition_by_transform_is_one_to_one_and_matches_infer():
    from sql_transform import SQLTransform

    t = SQLTransform(
        "SELECT MEAN(target) OVER (PARTITION BY city) AS enc FROM __THIS__"
    )
    t.fit(pa.table({"city": ["a", "b", "a", "b"], "target": [1.0, 3.0, 2.0, 4.0]}))

    batch = pa.table({"city": ["a", "b", "zzz"], "target": [0.0, 0.0, 0.0]})
    out = t.transform(batch)
    assert out.num_rows == 3  # strictly 1-to-1, unseen row preserved

    rows = t.infer_batch(
        [{"city": "a", "target": 0.0}, {"city": "b", "target": 0.0}, {"city": "zzz", "target": 0.0}]
    )
    assert out.column("enc").to_pylist() == [r.enc for r in rows]
    assert out.column("enc").to_pylist()[2] is None  # unseen -> NULL, both engines
```

- [ ] **Step 3: Update `__init__.py`**

(a) Update imports — drop the state-model import, add `build_state_tables`:

Replace:
```python
from sql_transform._batch import run_batch
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
from sql_transform._state import build_state_tables
```

(b) In `__init__`, replace the `_state` field with `_state_tables`:

Replace:
```python
    def __init__(self, sql: str) -> None:
        self._sql = sql
        self._state: BaseModel | None = None
        self._rewritten_sql: str | None = None
        self._infer_fn: InferFn | None = None
```
with:
```python
    def __init__(self, sql: str) -> None:
        self._sql = sql
        self._state_tables: dict[str, pa.Table] | None = None
        self._rewritten_sql: str | None = None
        self._infer_fn: InferFn | None = None
```

(c) Replace the body of `fit()` (from `this_model =` through `return self`):

```python
        this_model = this_model or synthesize_this_model(table.schema)

        tree = parse_and_validate(self._sql)
        windows = find_window_aggregates(tree)

        ctx = datafusion.SessionContext()
        ctx.from_arrow(table, name="__THIS__")

        self._state_tables = build_state_tables(windows, ctx, "__THIS__")
        self._rewritten_sql = rewrite_sql(tree, windows)
        self._infer_fn = InferFn(
            self._rewritten_sql,
            row_tables={"__THIS__": this_model},
            static_tables=self._state_tables,
        )
        return self
```

(d) Replace `transform` (pass state tables) and `infer_batch` (no `__STATE__` row):

Replace `transform`'s body:
```python
        if self._infer_fn is None:
            raise RuntimeError("Must call fit() before transform")
        return run_batch(self._rewritten_sql, table, self._state_tables)
```

Replace `infer_batch`'s body:
```python
        if self._infer_fn is None:
            raise RuntimeError("Must call fit() before inference")
        this_rows = [_to_namespace(row) for row in rows]
        return self._infer_fn.infer({"__THIS__": this_rows})
```

`infer` (delegates to `infer_batch`) and `_to_namespace` are unchanged. Remove the now-unused `BaseModel` import only if nothing else uses it — `synthesize_this_model`'s return and `_to_namespace`'s `isinstance(row, BaseModel)` still use it, so keep it.

- [ ] **Step 4: Run the file tests, then the full suite**

Run: `uv run pytest sql_transform/__init___test.py -v`
Expected: PASS (existing OVER () tests now run through static state tables + LEFT joins; new PARTITION BY tests pass).

Run: `uv run pytest -q`
Expected: all pass, one `xfail` (the pre-existing div-by-zero test).

- [ ] **Step 5: Run ruff**

Run: `uv run ruff check . && uv run ruff format .`
Expected: clean; re-run `uv run pytest -q` if anything reformatted.

- [ ] **Step 6: Commit**

```bash
git add sql_transform/__init__.py sql_transform/_batch.py sql_transform/__init___test.py
git commit -m "feat: wire PARTITION BY state tables through transform and infer"
```

---

### Task 6: Docs — `SQL_SUPPORT.md`, `VISION.md`, backlog

**Files:**
- Modify: `docs/SQL_SUPPORT.md`, `docs/BACKLOG.md`

**Interfaces:**
- Consumes: nothing (docs only).
- Produces: nothing consumed by other tasks.

- [ ] **Step 1: Update `docs/SQL_SUPPORT.md`**

In the Layer 2 table, change the two rejected PARTITION/ORDER rows and add typing:
- `PARTITION BY window aggregates` — status `❌ explicitly rejected` → `✅` | Source `_state.py` `build_state_tables` + `_rewrite.py` LEFT join + Rust LEFT lookup join.
- Add a row: `Per-partition state value types preserved (int/float/str/bool)` → `✅` | `_state.py` (no float coercion).
- Leave `ORDER BY window aggregates` as `❌ explicitly rejected` (`_state.py` raises `NotImplementedError`).

- [ ] **Step 2: Update `docs/BACKLOG.md`**

Move the `PARTITION BY` item out of "In progress" (it's shipped). Delete that entry. The `ORDER BY / window frames` item stays in Open items.

- [ ] **Step 3: Commit**

```bash
git add docs/SQL_SUPPORT.md docs/BACKLOG.md
git commit -m "docs: PARTITION BY + value typing supported; backlog updated"
```

---

## Post-Plan Verification

- [ ] `uv run maturin develop && mise run check` — clean, one `xfail`, no `XPASS`.
- [ ] Target encoding works end-to-end: seen partition → its mean, unseen → NULL, on both `infer` and `transform`.
- [ ] `transform` is 1-to-1 on a batch containing an unseen partition (row count preserved, NULL value).
- [ ] Count encoding stays integer-typed (`isinstance(out.n, int)`), confirming no float coercion.
- [ ] `grep -rn "extract_state\|synthesize_state_model\|self\._state\b" sql_transform/` returns nothing (old state-model path fully removed).
