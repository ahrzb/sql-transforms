# sqlglot-Based SQL Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `_state.py`/`_rewrite.py`'s DataFusion-plan-text regex parsing with a proper sqlglot AST walk of the original SQL text, keeping `SQLTransform`'s functional behavior exactly as it is today.

**Architecture:** A new shared module, `sql_transform/_sql.py`, parses SQL with sqlglot, validates it against SQLTransform's supported subset (single `SELECT`, exactly `FROM __THIS__`, no `WHERE`/`JOIN`/`GROUP BY`/`HAVING`/`ORDER BY`/`LIMIT`), and structurally locates every window-aggregate node (`WindowAgg` records: function, column, whether it has `PARTITION BY`/`ORDER BY`). `_state.py` and `_rewrite.py` both consume that same `WindowAgg` list — one source of truth, instead of two independently-maintained regexes that can (and did) disagree. `_rewrite.py` mutates the parsed tree in place via sqlglot's `.replace()`/`.join()` and serializes the result with `.sql()`, instead of hand-built f-strings. DataFusion's role shrinks to exactly one thing: executing the small per-aggregate value queries during `fit()` — it never parses or plans `SQLTransform`'s original SQL anymore.

**Tech Stack:** Python 3.13, sqlglot (new dependency, already added via `uv add sqlglot`, resolved to `30.12.0`), DataFusion (`datafusion` package, execution-only role from here on), Pydantic v2, the existing Rust `InferFn`.

## Global Constraints

- Functional scope is **identical** to today — this is a foundation swap, not a capability expansion. Anything not already supported stays unsupported.
- `SQLTransform`'s SQL must be a single `SELECT` against exactly `FROM __THIS__` (no alias) — `WHERE`, `JOIN`, `GROUP BY`, `HAVING`, `ORDER BY` (top-level), `LIMIT` all raise a clear `ValueError` naming the clause, at parse-validation time, before any DataFusion work.
- Window aggregates: `PARTITION BY`/`ORDER BY` (the window's own `OVER (...)` clause) still raise `NotImplementedError`, same messages as today — now driven by structural `has_partition`/`has_order` flags instead of regex.
- `MEAN` → `AVG` synonym normalization is added explicitly (`_FUNCTION_SYNONYMS = {"MEAN": "AVG"}` in `_sql.py`) to preserve today's exact `state_key` naming — DataFusion gave this for free via plan-display normalization; sqlglot does not. No other function-name allowlist is introduced — any DataFusion-recognized aggregate function name works, matching today's genericity.
- Column case handling from commit `2b3171c` (quote the column name in the DataFusion value query; preserve real case in the dedup key; raise a clear `ValueError` on a genuine case-collision) carries forward unchanged.
- No new error *types* — everything is `ValueError` or `NotImplementedError`, matching today's taxonomy.
- Every test in `_state_test.py`, `_rewrite_test.py`, and `__init___test.py` that exercises current behavior must keep passing unchanged (same public behavior, same error types/messages) — this is the acceptance bar.

---

## File Structure

```
sql_transform/_sql.py           (new)     — parse_and_validate(), WindowAgg, find_window_aggregates()
sql_transform/_sql_test.py      (new)     — tests for _sql.py
sql_transform/_state.py         (rewrite) — extract_state(windows, ctx, table_name) -- no longer parses SQL
sql_transform/_state_test.py    (rewrite) — tests for the rewritten _state.py
sql_transform/_rewrite.py       (rewrite) — rewrite_sql(select, windows) -- sqlglot AST mutation, not DataFusion plan walk
sql_transform/_rewrite_test.py  (rewrite) — tests for the rewritten _rewrite.py
sql_transform/__init__.py       (modify)  — fit() wires the new pipeline together
sql_transform/__init___test.py  (modify)  — add scope-validation test coverage; existing tests unchanged
```

---

### Task 1: `_sql.py` — parsing, scope validation, window-aggregate discovery

**Files:**
- Create: `sql_transform/_sql.py`
- Test: `sql_transform/_sql_test.py`

**Interfaces:**
- Consumes: nothing from other tasks (leaf module, only depends on `sqlglot`).
- Produces:
  - `parse_and_validate(sql: str) -> sqlglot.exp.Select` — used by Task 4 (`__init__.py`).
  - `WindowAgg` (frozen dataclass: `node: exp.Window`, `fn: str`, `col: str`, `has_partition: bool`, `has_order: bool`) — used by Tasks 2 and 3.
  - `find_window_aggregates(select: exp.Select) -> list[WindowAgg]` — used by Task 4.

- [ ] **Step 1: Write the failing tests**

Create `sql_transform/_sql_test.py`:

```python
"""Tests for SQL parsing, scope validation, and window-aggregate discovery."""

import pytest
from sqlglot import exp

from sql_transform._sql import find_window_aggregates, parse_and_validate


def test_parse_valid_simple_select():
    tree = parse_and_validate("SELECT age FROM __THIS__")
    assert isinstance(tree, exp.Select)
    assert tree.sql() == "SELECT age FROM __THIS__"


def test_parse_rejects_wrong_from_table():
    with pytest.raises(ValueError, match="__THIS__"):
        parse_and_validate("SELECT age FROM data")


def test_parse_rejects_aliased_this():
    with pytest.raises(ValueError, match="__THIS__"):
        parse_and_validate("SELECT age FROM __THIS__ AS t")


def test_parse_rejects_where():
    with pytest.raises(ValueError, match="WHERE"):
        parse_and_validate("SELECT age FROM __THIS__ WHERE age > 1")


def test_parse_rejects_join():
    with pytest.raises(ValueError, match="JOIN"):
        parse_and_validate(
            "SELECT __THIS__.x FROM __THIS__ JOIN b ON __THIS__.id = b.id"
        )


def test_parse_rejects_group_by():
    with pytest.raises(ValueError, match="GROUP BY"):
        parse_and_validate("SELECT age FROM __THIS__ GROUP BY age")


def test_parse_rejects_order_by():
    with pytest.raises(ValueError, match="ORDER BY"):
        parse_and_validate("SELECT age FROM __THIS__ ORDER BY age")


def test_parse_rejects_limit():
    with pytest.raises(ValueError, match="LIMIT"):
        parse_and_validate("SELECT age FROM __THIS__ LIMIT 5")


def test_parse_rejects_multiple_statements():
    with pytest.raises(ValueError, match="one SQL statement"):
        parse_and_validate("SELECT age FROM __THIS__; SELECT age FROM __THIS__")


def test_parse_rejects_non_select():
    with pytest.raises(ValueError, match="SELECT"):
        parse_and_validate("CREATE TABLE t (id INT)")


def test_find_window_aggregates_detects_avg():
    tree = parse_and_validate("SELECT AVG(age) OVER () AS x FROM __THIS__")
    windows = find_window_aggregates(tree)
    assert len(windows) == 1
    assert windows[0].fn == "AVG"
    assert windows[0].col == "age"
    assert windows[0].has_partition is False
    assert windows[0].has_order is False


def test_find_window_aggregates_normalizes_mean_to_avg():
    tree = parse_and_validate("SELECT MEAN(age) OVER () AS x FROM __THIS__")
    windows = find_window_aggregates(tree)
    assert windows[0].fn == "AVG"


def test_find_window_aggregates_detects_partition_by():
    tree = parse_and_validate(
        "SELECT AVG(age) OVER (PARTITION BY city) AS x FROM __THIS__"
    )
    windows = find_window_aggregates(tree)
    assert windows[0].has_partition is True


def test_find_window_aggregates_detects_order_by():
    tree = parse_and_validate(
        "SELECT AVG(age) OVER (ORDER BY age) AS x FROM __THIS__"
    )
    windows = find_window_aggregates(tree)
    assert windows[0].has_order is True


def test_find_window_aggregates_rejects_non_column_argument():
    tree = parse_and_validate("SELECT AVG(age + 1) OVER () AS x FROM __THIS__")
    with pytest.raises(ValueError, match="single plain column"):
        find_window_aggregates(tree)


def test_find_window_aggregates_empty_when_no_windows():
    tree = parse_and_validate("SELECT age FROM __THIS__")
    assert find_window_aggregates(tree) == []


def test_find_window_aggregates_multiple_distinct():
    tree = parse_and_validate(
        "SELECT AVG(age) OVER () AS a, SUM(score) OVER () AS b FROM __THIS__"
    )
    windows = find_window_aggregates(tree)
    assert len(windows) == 2
    assert {(w.fn, w.col) for w in windows} == {("AVG", "age"), ("SUM", "score")}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest sql_transform/_sql_test.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sql_transform._sql'`

- [ ] **Step 3: Write the implementation**

Create `sql_transform/_sql.py`:

```python
"""Parse and validate SQLTransform's SQL via sqlglot, and locate window
aggregates structurally.

Shared by _state.py and _rewrite.py so there is exactly one place that
knows what SQLTransform's supported SQL subset looks like in the sqlglot
AST -- avoids the class of bug where two independently-maintained regexes
disagreed about the same window aggregate (fixed in commit 2b3171c).
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

_FUNCTION_SYNONYMS = {"MEAN": "AVG"}

_UNSUPPORTED_CLAUSES = {
    "joins": "JOIN",
    "where": "WHERE",
    "group": "GROUP BY",
    "having": "HAVING",
    "order": "ORDER BY",
    "limit": "LIMIT",
}


def parse_and_validate(sql: str) -> exp.Select:
    """Parse `sql` and enforce SQLTransform's supported SQL subset: a
    single SELECT against exactly `FROM __THIS__` (no alias), with none
    of JOIN/WHERE/GROUP BY/HAVING/ORDER BY/LIMIT. Raises ValueError naming
    the first unsupported construct found."""
    statements = sqlglot.parse(sql)
    if len(statements) != 1:
        raise ValueError("Expected exactly one SQL statement")
    tree = statements[0]
    if not isinstance(tree, exp.Select):
        raise ValueError("Only SELECT queries are supported")

    from_ = tree.args.get("from_")
    if from_ is None or not isinstance(from_.this, exp.Table):
        raise ValueError("FROM clause is required and must be a plain table")
    table = from_.this
    if table.name != "__THIS__" or table.alias:
        raise ValueError(
            "FROM clause must be exactly __THIS__ (no alias); found "
            f"{table.sql()!r}"
        )

    for key, label in _UNSUPPORTED_CLAUSES.items():
        if tree.args.get(key):
            raise ValueError(f"{label} is not yet supported by SQLTransform")

    return tree


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
    has_partition: bool
    has_order: bool


def find_window_aggregates(select: exp.Select) -> list[WindowAgg]:
    """Find every window-aggregate node in `select`'s projection list.

    Raises ValueError if a window aggregate's argument isn't a single
    plain column -- multi-arg and expression aggregates aren't supported.
    """
    windows: list[WindowAgg] = []
    for node in select.find_all(exp.Window):
        func = node.this
        if isinstance(func, exp.Anonymous):
            fn = func.this.upper()
            args = func.expressions
        else:
            fn = func.sql_name()
            args = [func.this]
        fn = _FUNCTION_SYNONYMS.get(fn, fn)

        if len(args) != 1 or not isinstance(args[0], exp.Column):
            raise ValueError(
                "Window aggregate argument must be a single plain column: "
                f"{node.sql()!r}"
            )
        col = args[0].name

        windows.append(
            WindowAgg(
                node=node,
                fn=fn,
                col=col,
                has_partition=bool(node.args.get("partition_by")),
                has_order=bool(node.args.get("order")),
            )
        )
    return windows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest sql_transform/_sql_test.py -v`
Expected: PASS (17 passed)

- [ ] **Step 5: Commit**

```bash
git add sql_transform/_sql.py sql_transform/_sql_test.py
git commit -m "feat: sqlglot-based SQL parsing, scope validation, window-agg discovery"
```

---

### Task 2: `_state.py` — `extract_state` consumes `WindowAgg` list instead of parsing SQL

**Files:**
- Modify (full rewrite): `sql_transform/_state.py`
- Test (full rewrite): `sql_transform/_state_test.py`

**Interfaces:**
- Consumes: `WindowAgg` (Task 1).
- Produces:
  - `state_key(fn_name: str, col_name: str) -> str` — unchanged signature, still used by Task 3.
  - `extract_state(windows: list[WindowAgg], ctx: datafusion.SessionContext, table_name: str) -> pydantic.BaseModel` — signature changed from `extract_state(plan, ctx, table_name)`; used by Task 4.

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `sql_transform/_state_test.py`:

```python
"""Tests for state extraction from window-aggregate discovery."""

import datafusion
import pytest

from sql_transform._sql import find_window_aggregates, parse_and_validate
from sql_transform._state import extract_state, state_key


def _windows(sql: str):
    return find_window_aggregates(parse_and_validate(sql))


def test_state_key_lowercases_and_strips_qualifier():
    assert state_key("AVG", "age") == "avg_age"
    assert state_key("avg", "AGE") == "avg_age"


def test_extract_constant_window_agg():
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"age": [25, 30, 35]}, name="data")

    windows = _windows("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
    state = extract_state(windows, ctx, "data")

    assert state.avg_age == 30.0


def test_extract_dedups_repeated_aggregate():
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"age": [25, 30, 35]}, name="data")

    windows = _windows(
        "SELECT age / MEAN(age) OVER () AS age_norm, "
        "MEAN(age) OVER () AS age_avg FROM __THIS__"
    )
    state = extract_state(windows, ctx, "data")

    # Both projections reference the same (fn, col) pair -> one field.
    assert state.model_dump() == {"avg_age": 30.0}


def test_extract_multiple_distinct_aggregates():
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"age": [25, 30, 35], "score": [10, 20, 30]}, name="data")

    windows = _windows(
        "SELECT age / MEAN(age) OVER () AS age_norm, "
        "score / SUM(score) OVER () AS score_norm FROM __THIS__"
    )
    state = extract_state(windows, ctx, "data")

    assert state.avg_age == 30.0
    assert state.sum_score == 60.0


def test_extract_no_aggregates_returns_empty_state():
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"age": [1, 2, 3]}, name="data")

    windows = _windows("SELECT age FROM __THIS__")
    state = extract_state(windows, ctx, "data")

    assert state.model_dump() == {}


def test_extract_partitioned_window_agg_raises_not_implemented():
    ctx = datafusion.SessionContext()
    ctx.from_pydict(
        {"city": ["a", "b", "a", "b"], "target": [1.0, 2.0, 3.0, 4.0]},
        name="data",
    )

    windows = _windows(
        "SELECT MEAN(target) OVER (PARTITION BY city) AS city_enc FROM __THIS__"
    )
    with pytest.raises(NotImplementedError):
        extract_state(windows, ctx, "data")


def test_extract_multi_column_partitioned_window_agg_raises_not_implemented():
    ctx = datafusion.SessionContext()
    ctx.from_pydict(
        {
            "city": ["a", "b", "a", "b"],
            "region": ["x", "y", "x", "y"],
            "target": [1.0, 2.0, 3.0, 4.0],
        },
        name="data",
    )

    windows = _windows(
        "SELECT MEAN(target) OVER (PARTITION BY city, region) AS enc FROM __THIS__"
    )
    with pytest.raises(NotImplementedError):
        extract_state(windows, ctx, "data")


def test_extract_order_by_window_agg_raises_not_implemented():
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"age": [25, 30, 35]}, name="data")

    windows = _windows(
        "SELECT MEAN(age) OVER (ORDER BY age) AS running_avg FROM __THIS__"
    )
    with pytest.raises(NotImplementedError):
        extract_state(windows, ctx, "data")


def test_extract_preserves_column_case_in_query():
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"Age": [25.0, 30.0, 35.0]}, name="data")

    windows = _windows(
        'SELECT "Age" / MEAN("Age") OVER () AS age_norm FROM __THIS__'
    )
    state = extract_state(windows, ctx, "data")

    assert state.avg_age == 30.0


def test_extract_case_differing_columns_raises_ambiguous_error():
    ctx = datafusion.SessionContext()
    ctx.from_pydict(
        {"age": [25.0, 30.0, 35.0], "Age": [100.0, 200.0, 300.0]}, name="data"
    )

    windows = _windows(
        'SELECT age / MEAN(age) OVER () + "Age" / MEAN("Age") OVER () '
        "AS combo FROM __THIS__"
    )
    with pytest.raises(ValueError, match="Ambiguous window aggregate"):
        extract_state(windows, ctx, "data")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest sql_transform/_state_test.py -v`
Expected: FAIL — `extract_state` still expects `(plan, ctx, table_name)`, not `(windows, ctx, table_name)`.

- [ ] **Step 3: Write the implementation**

Replace the full contents of `sql_transform/_state.py`:

```python
"""Extract learned state from SQLTransform's window aggregates.

Runs one DataFusion query per DISTINCT (function, column) pair found by
_sql.py's find_window_aggregates(), and synthesizes a typed Pydantic
model instance ("StateModel") keyed by `{fn}_{col}` (no leading
underscore -- see state_key), suitable for use as InferFn's __STATE__ row
table. DataFusion's only role here is executing those small per-aggregate
queries -- it never parses or plans SQLTransform's original SQL.
"""

from __future__ import annotations

import datafusion
from pydantic import BaseModel

from sql_transform._schema import synthesize_state_model
from sql_transform._sql import WindowAgg


def state_key(fn_name: str, col_name: str) -> str:
    """The __STATE__ field name for a given aggregate function + column,
    e.g. state_key("AVG", "age") == "avg_age". No leading underscore --
    Pydantic v2 would treat that as a private attribute."""
    return f"{fn_name.lower()}_{col_name.lower()}"


def extract_state(
    windows: list[WindowAgg],
    ctx: datafusion.SessionContext,
    table_name: str,
) -> BaseModel:
    """Return a synthesized StateModel instance with one float field per
    distinct (fn, col) window aggregate in `windows`.

    Raises NotImplementedError if any window aggregate uses PARTITION BY
    or ORDER BY -- not yet supported by the Rust-backed pipeline.
    """
    pairs: dict[tuple[str, str], None] = {}
    for w in windows:
        if w.has_partition:
            raise NotImplementedError(
                "PARTITION BY window aggregates are not yet supported by "
                "the Rust-backed SQLTransform pipeline"
            )
        if w.has_order:
            raise NotImplementedError(
                "ORDER BY window aggregates are not yet supported by "
                "the Rust-backed SQLTransform pipeline"
            )
        # Preserve the column's real case here -- lower-casing it would
        # break the query below against a mixed-case schema, and would
        # collide two distinct case-differing columns onto the same
        # dedup key (state_key() below normalizes the STATE FIELD name
        # to lowercase, which is a separate, intentional choice).
        pairs[(w.fn, w.col)] = None

    values: dict[str, float] = {}
    for fn_name, col_name in pairs:
        # Quote the column name so DataFusion resolves it against the
        # schema's real (possibly mixed-case) field name rather than
        # case-folding an unquoted identifier to lowercase.
        sql = f'SELECT {fn_name}("{col_name}") FROM {table_name}'
        result = ctx.sql(sql).collect()
        value = result[0].column(0)[0].as_py()
        key = state_key(fn_name, col_name)
        if key in values:
            raise ValueError(
                f"Ambiguous window aggregate: {fn_name}({col_name}) "
                f"normalizes to the same state key {key!r} as another "
                "aggregate in this query -- column names that differ only "
                "by case aren't distinguished"
            )
        values[key] = float(value)

    state_model = synthesize_state_model(values)
    return state_model(**values)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest sql_transform/_state_test.py -v`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add sql_transform/_state.py sql_transform/_state_test.py
git commit -m "feat: extract_state consumes WindowAgg list instead of parsing plan text"
```

---

### Task 3: `_rewrite.py` — `rewrite_sql` operates on the sqlglot tree

**Files:**
- Modify (full rewrite): `sql_transform/_rewrite.py`
- Test (full rewrite): `sql_transform/_rewrite_test.py`

**Interfaces:**
- Consumes: `WindowAgg` (Task 1), `state_key` (Task 2).
- Produces: `rewrite_sql(select: exp.Select, windows: list[WindowAgg]) -> str` — signature changed from `rewrite_sql(plan)`; used by Task 4. Mutates `select` in place.

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `sql_transform/_rewrite_test.py`:

```python
"""Tests for SQL rewriting via the sqlglot AST."""

import pytest

from sql_transform._rewrite import rewrite_sql
from sql_transform._sql import find_window_aggregates, parse_and_validate


def _rewrite(sql: str) -> str:
    tree = parse_and_validate(sql)
    windows = find_window_aggregates(tree)
    return rewrite_sql(tree, windows)


def test_rewrite_simple_column_pass_through():
    sql = _rewrite("SELECT age AS just_age FROM __THIS__")
    assert sql == "SELECT __THIS__.age AS just_age FROM __THIS__ CROSS JOIN __STATE__"


def test_rewrite_constant_window_agg():
    sql = _rewrite("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
    assert sql == (
        "SELECT __THIS__.age / __STATE__.avg_age AS age_norm "
        "FROM __THIS__ CROSS JOIN __STATE__"
    )


def test_rewrite_bare_window_agg_alias():
    sql = _rewrite("SELECT MEAN(age) OVER () AS age_avg FROM __THIS__")
    assert sql == "SELECT __STATE__.avg_age AS age_avg FROM __THIS__ CROSS JOIN __STATE__"


def test_rewrite_multiple_projections():
    sql = _rewrite(
        "SELECT age / MEAN(age) OVER () AS age_norm, "
        "score / SUM(score) OVER () AS score_norm FROM __THIS__"
    )
    assert sql == (
        "SELECT __THIS__.age / __STATE__.avg_age AS age_norm, "
        "__THIS__.score / __STATE__.sum_score AS score_norm "
        "FROM __THIS__ CROSS JOIN __STATE__"
    )


def test_rewrite_unaliased_expression_raises_clear_error():
    with pytest.raises(ValueError, match="needs an alias"):
        _rewrite("SELECT age / MEAN(age) OVER () FROM __THIS__")


def test_rewrite_unaliased_bare_window_agg_raises_clear_error():
    with pytest.raises(ValueError, match="needs an alias"):
        _rewrite("SELECT MEAN(age) OVER () FROM __THIS__")


def test_rewrite_bad_column_qualifier_raises():
    with pytest.raises(ValueError, match="does not refer to __THIS__"):
        _rewrite("SELECT foo.age AS x FROM __THIS__")


def test_rewrite_already_qualified_column_stays_this():
    sql = _rewrite("SELECT __THIS__.age AS x FROM __THIS__")
    assert sql == "SELECT __THIS__.age AS x FROM __THIS__ CROSS JOIN __STATE__"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest sql_transform/_rewrite_test.py -v`
Expected: FAIL — `rewrite_sql` still expects a DataFusion `plan` argument, not `(select, windows)`.

- [ ] **Step 3: Write the implementation**

Replace the full contents of `sql_transform/_rewrite.py`:

```python
"""Rewrite SQLTransform's SQL into SQL runnable by the Rust InferFn.

Given the Select tree and WindowAgg list from _sql.py, replaces every
window-aggregate reference with a __STATE__ column reference, qualifies
every plain column as __THIS__.col, and appends __STATE__ as a cross-
joined FROM entry (__STATE__ is always exactly one row, so a cross join
just repeats it against every __THIS__ row).
"""

from __future__ import annotations

from sqlglot import exp

from sql_transform._sql import WindowAgg
from sql_transform._state import state_key


def rewrite_sql(select: exp.Select, windows: list[WindowAgg]) -> str:
    """Return SQL text equivalent to `select`'s projection, with every
    window-aggregate reference replaced by a __STATE__ column reference.

    Mutates `select` in place via node.replace() -- callers should not
    reuse `select` afterwards.
    """
    window_key = {id(w.node): state_key(w.fn, w.col) for w in windows}

    for e in select.expressions:
        out_name = e.alias_or_name
        if not out_name:
            raise ValueError(
                "Expression in SELECT list needs an alias (AS name): "
                f"{e.sql()!r}"
            )

        for win_node in list(e.find_all(exp.Window)):
            win_node.replace(
                exp.column(window_key[id(win_node)], table="__STATE__")
            )

        for col_node in list(e.find_all(exp.Column)):
            if col_node.table == "__STATE__":
                continue  # already rewritten by the pass above
            if col_node.table and col_node.table != "__THIS__":
                raise ValueError(
                    f"Column qualifier {col_node.table!r} does not refer "
                    "to __THIS__"
                )
            col_node.replace(exp.column(col_node.name, table="__THIS__"))

    select.join("__STATE__", join_type="CROSS", copy=False)
    return select.sql()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest sql_transform/_rewrite_test.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add sql_transform/_rewrite.py sql_transform/_rewrite_test.py
git commit -m "feat: rewrite_sql mutates the sqlglot tree instead of walking a DataFusion plan"
```

---

### Task 4: `__init__.py` — wire `fit()` to the new pipeline

**Files:**
- Modify: `sql_transform/__init__.py`
- Modify: `sql_transform/__init___test.py`

**Interfaces:**
- Consumes:
  - `parse_and_validate(sql: str) -> exp.Select`, `find_window_aggregates(select) -> list[WindowAgg]` (Task 1)
  - `extract_state(windows, ctx, table_name) -> BaseModel` (Task 2)
  - `rewrite_sql(select, windows) -> str` (Task 3)
- Produces: `SQLTransform` public class — unchanged public API/behavior; this is the integration point, nothing downstream in this plan consumes it further.

- [ ] **Step 1: Write the failing tests**

Add these two new tests to `sql_transform/__init___test.py` — insert after the existing `test_infer_before_fit_raises_runtime_error` function (do not remove any existing test in this file; all of them must keep passing):

```python
def test_fit_rejects_where_clause():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age FROM __THIS__ WHERE age > 1")
    with pytest.raises(ValueError, match="WHERE"):
        t.fit(pa.table({"age": [1, 2, 3]}))


def test_fit_rejects_wrong_from_table():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age FROM data")
    with pytest.raises(ValueError, match="__THIS__"):
        t.fit(pa.table({"age": [1, 2, 3]}))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest sql_transform/__init___test.py -v`
Expected: The two new tests FAIL:
- `test_fit_rejects_where_clause` — today's `fit()` silently ignores `WHERE` (only the projection list is walked), so no exception is raised at all; the test's `pytest.raises(ValueError, ...)` fails with "DID NOT RAISE".
- `test_fit_rejects_wrong_from_table` — today's `fit()` does raise a `ValueError` for the missing `data` table (DataFusion itself rejects it), but with message `"Error during planning: table 'datafusion.public.data' not found"`, which doesn't match the test's `match="__THIS__"` — fails on the message assertion, not on exception type.

- [ ] **Step 3: Write the implementation**

In `sql_transform/__init__.py`, replace the imports block and the `fit()` method body:

Replace:
```python
from sql_transform._rewrite import rewrite_sql
from sql_transform._interpreter import InferFn
from sql_transform._schema import synthesize_this_model
from sql_transform._state import extract_state
```
with:
```python
from sql_transform._interpreter import InferFn
from sql_transform._rewrite import rewrite_sql
from sql_transform._schema import synthesize_this_model
from sql_transform._sql import find_window_aggregates, parse_and_validate
from sql_transform._state import extract_state
```

Replace the body of `fit()`:
```python
    def fit(
        self,
        table: pa.Table,
        /,
        this_model: type[BaseModel] | None = None,
    ) -> SQLTransform:
        this_model = this_model or synthesize_this_model(table.schema)

        ctx = datafusion.SessionContext()
        ctx.from_arrow(table, name="__THIS__")
        df = ctx.sql(self._sql)
        plan = df.logical_plan()

        self._state = extract_state(plan, ctx, "__THIS__")
        rewritten_sql = rewrite_sql(plan)
        self._infer_fn = InferFn(
            rewritten_sql,
            row_tables={"__THIS__": this_model, "__STATE__": type(self._state)},
            static_tables={},
        )
        return self
```
with:
```python
    def fit(
        self,
        table: pa.Table,
        /,
        this_model: type[BaseModel] | None = None,
    ) -> SQLTransform:
        this_model = this_model or synthesize_this_model(table.schema)

        tree = parse_and_validate(self._sql)
        windows = find_window_aggregates(tree)

        ctx = datafusion.SessionContext()
        ctx.from_arrow(table, name="__THIS__")

        self._state = extract_state(windows, ctx, "__THIS__")
        rewritten_sql = rewrite_sql(tree, windows)
        self._infer_fn = InferFn(
            rewritten_sql,
            row_tables={"__THIS__": this_model, "__STATE__": type(self._state)},
            static_tables={},
        )
        return self
```

The rest of the file (`__init__`, `from_file`, `_infer_rows`, `transform`, `_infer`) is unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest sql_transform/__init___test.py -v`
Expected: PASS (17 passed — 15 existing + 2 new)

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest`
Expected: All tests pass. Final count: `sql_transform/_sql_test.py` (17) + `sql_transform/_state_test.py` (10) + `sql_transform/_rewrite_test.py` (8) + `sql_transform/_schema_test.py` (6, unchanged) + `sql_transform/__init___test.py` (17) + `tests/test_interpreter.py` (30, unchanged) = 88 passed.

- [ ] **Step 6: Run ruff**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: Clean (the pre-existing `S608` warning on `_rewrite.py`'s `select.sql()`-adjacent code, if it resurfaces, is pre-existing project lint debt — not introduced by this change; do not suppress it ad hoc, just confirm it's the same pre-existing warning by checking `git stash` + re-running on the prior version if in doubt).

- [ ] **Step 7: Commit**

```bash
git add sql_transform/__init__.py sql_transform/__init___test.py
git commit -m "feat: wire SQLTransform.fit() to the sqlglot-based rewrite pipeline"
```

---

### Task 5: Update `SQL_SUPPORT.md`

**Files:**
- Modify: `SQL_SUPPORT.md`

**Interfaces:**
- Consumes: nothing (docs only).
- Produces: nothing consumed by other tasks.

- [ ] **Step 1: Update Layer 2's status rows**

In `SQL_SUPPORT.md`'s Layer 2 table, update the `Source` column for these rows (functional status `✅`/`❌` for each stays the same — only the implementation source changes from DataFusion-plan-walk to sqlglot):

- `Window aggregate, no PARTITION BY/ORDER BY` → Source: `_sql.py` `find_window_aggregates`
- `Plain column reference in SELECT` → Source: `_rewrite.py` `rewrite_sql`
- `Binary-op arithmetic in SELECT` → Source: `_rewrite.py` `rewrite_sql`
- `Required alias on every SELECT item` → Source: `_rewrite.py` `rewrite_sql`
- `PARTITION BY window aggregates` → Source: `_state.py` `extract_state` (via `WindowAgg.has_partition`)
- `ORDER BY window aggregates` → Source: `_state.py` `extract_state` (via `WindowAgg.has_order`)

Also replace the "Parser swap in progress" paragraph (added in commit `1c696f0`) with a short "Parser swap complete" note: sqlglot now does 100% of the SQL parsing/analysis for Layer 2; DataFusion's only remaining role in `fit()` is executing the per-aggregate value queries. Add three new rows to the table reflecting the new, explicit scope-validation errors that didn't exist before (previously these constructs just fell through to a confusing DataFusion or `InferFn` error rather than being explicitly named):

| Feature | Status | Source |
|---|---|---|
| Explicit `ValueError` naming the unsupported clause (`WHERE`/`JOIN`/`GROUP BY`/`HAVING`/`ORDER BY`/`LIMIT`) instead of a downstream failure | ✅ | `_sql.py` `parse_and_validate` |
| `MEAN` → `AVG` synonym (preserves pre-sqlglot behavior) | ✅ | `_sql.py` `_FUNCTION_SYNONYMS` |

- [ ] **Step 2: Verify the doc still reads coherently**

Read the full file once after editing to confirm no leftover references to "DataFusion plan-walk" as the *current* mechanism (historical mentions describing why the swap happened are fine to keep).

- [ ] **Step 3: Commit**

```bash
git add SQL_SUPPORT.md
git commit -m "docs: SQL_SUPPORT.md reflects the completed sqlglot parser swap"
```

---

## Post-Plan Verification

- [ ] Run `mise run check` (fmt + full test suite) and confirm it's clean.
- [ ] Confirm no remaining references to `datafusion.plan.LogicalPlan`, `display_indent`, or `to_variant` in `sql_transform/_state.py` or `sql_transform/_rewrite.py` (`grep -n "display_indent\|to_variant\|LogicalPlan" sql_transform/_state.py sql_transform/_rewrite.py` should return nothing) — confirms the DataFusion-plan-walk is fully gone from these two files.
- [ ] Confirm `sql_transform/_state.py` and `sql_transform/_rewrite.py` no longer `import re` (the regexes are gone) — `grep -n "^import re" sql_transform/_state.py sql_transform/_rewrite.py` should return nothing.
