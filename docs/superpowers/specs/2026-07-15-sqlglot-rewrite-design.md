# sqlglot-Based SQL Rewrite — Design Spec

## Goal

Replace `_state.py`/`_rewrite.py`'s DataFusion-plan-text parsing (regexes over `plan.display_indent()` and generated `Column.name()` strings) with a proper AST walk of the *original* SQL text via [sqlglot](https://github.com/tobymao/sqlglot). Functional scope stays **identical** to what `SQLTransform` supports today — plain columns, arithmetic, non-partitioned window aggregates, required aliases. This is a foundation swap, not a capability expansion.

## Motivation

Two regexes currently encode the same knowledge about DataFusion's generated plan-text format in two different files (`_state.py::_WINDOW_AGG_RE`, `_rewrite.py::_WINDOW_COL_RE`), and a code review already caught real bugs from them silently disagreeing (ORDER BY / multi-column PARTITION BY bypassing the `NotImplementedError` guard — fixed in commit `2b3171c`, but the underlying fragility remains).

Attempting to fix this by walking DataFusion's Python plan *objects* instead of plan *text* hits a harder wall: `to_variant()` raises `"Converting Expr::WindowFunction to a Python object is not implemented"` for window functions in the installed `datafusion` version, requiring an undocumented `node.method(raw_expr)` calling convention to work around (see conversation research, and [[project_phase2_interpreter_pyo3_gotchas]]). That's a DataFusion Python-*binding* completeness gap, not a query-planning limitation — and it would recur for any other construct the rewrite tries to introspect.

This project already has precedent for the fix: the Rust `InferFn` interpreter parses SQL with `sqlparser` directly rather than via DataFusion's logical planner, for an analogous reason (schema unavailability there; binding incompleteness here — same lesson: don't lean on DataFusion's plan objects for general introspection, use a tool built for that job). sqlglot is that tool for the Python side — a stable, well-documented SQL AST library, previously used in this project's very first implementation and later dropped only because DataFusion's plan was, at the time, sufficient for the narrower job it was doing.

## Scope

**In scope — functional behavior identical to today:**
- Plain column references in the SELECT list (`age`).
- Binary-op arithmetic (`age / MEAN(age) OVER ()`).
- Non-partitioned, non-ordered window aggregates (`AVG(age) OVER ()`), any DataFusion-recognized aggregate function name (no new allowlist — same genericity as the current regex, which also didn't restrict function names).
- Required explicit alias on any non-bare-column SELECT expression (same as today, enforced by commit `2b3171c`).
- `MEAN` → `AVG` synonym normalization, added explicitly in this pass (see below) to preserve today's exact `state_key` naming and existing test expectations — DataFusion provided this for free via its own plan-display normalization; sqlglot does not (`MEAN(age)` parses as `exp.Anonymous`, not `exp.Avg`).

**Explicitly out of scope, rejected with a clear `ValueError` at parse-validation time (not a downstream crash):**
- `WHERE`, `JOIN`, `GROUP BY`, `HAVING`, `ORDER BY` (top-level SELECT clause), `LIMIT`.
- Multiple SQL statements in one string.
- Any `FROM` table other than exactly `__THIS__` (no aliasing, no additional tables).
- Any column qualified with a table name other than `__THIS__` (nothing else exists in scope).

**Still rejected with `NotImplementedError` (unchanged from today, just detected structurally instead of via regex):**
- `PARTITION BY` on a window aggregate.
- `ORDER BY` on a window aggregate (the window's own `OVER (ORDER BY ...)`, not the SELECT-level clause above).

**Not in this spec** (tracked in `VISION.md`'s roadmap as a distinct follow-up): join-based `PARTITION BY` support via synthesized per-partition state tables, and any resulting growth of `SQLTransform`'s public API to supply additional tables.

## Architecture

```
sql_transform/_sql.py       (new)      — sqlglot.parse_one(), scope validation,
                                          window-aggregate discovery. Shared by
                                          _state.py and _rewrite.py so there is
                                          exactly one place that knows what a
                                          "window aggregate" looks like in the
                                          sqlglot AST.

sql_transform/_state.py     (rewrite)  — extract_state() takes the WindowAgg
                                          list from _sql.py instead of parsing
                                          plan text itself; unchanged
                                          responsibility otherwise (run one
                                          DataFusion query per distinct (fn,
                                          col), synthesize a typed StateModel).
                                          DataFusion's role shrinks to exactly
                                          this: it never parses/plans the
                                          user's original SQL anymore, only
                                          executes the small per-aggregate
                                          value queries.

sql_transform/_rewrite.py   (rewrite)  — rewrite_sql() takes the same WindowAgg
                                          list plus the parsed (and copied)
                                          sqlglot tree, replaces each window
                                          node with a __STATE__ column
                                          reference via node.replace(),
                                          re-qualifies plain columns as
                                          __THIS__.col, appends __STATE__ as a
                                          cross-joined FROM entry, and returns
                                          tree.sql() -- a real AST
                                          serialization, not hand-built
                                          f-strings.

sql_transform/__init__.py   (modify)   — fit() calls _sql.py's parse+validate
                                          once, passes the result to
                                          extract_state() and rewrite_sql().
                                          Still builds a DataFusion
                                          SessionContext with __THIS__
                                          registered (needed for state-value
                                          execution), but no longer calls
                                          ctx.sql(self._sql) for structural
                                          analysis.

pyproject.toml               (modify)   — add sqlglot dependency (already
                                          added via `uv add sqlglot`,
                                          resolved to 30.12.0).
```

## `_sql.py`: Parsing and Scope Validation

```python
def parse_and_validate(sql: str) -> exp.Select:
    """Parse `sql`, enforce v1 scope, return the validated Select tree."""
```

- `sqlglot.parse(sql)` (not `parse_one`) and check `len(statements) == 1` — mirrors the Rust engine's own "Expected exactly one SQL statement" check (`plan.rs::build_plan`), same error shape.
- The single statement must be `isinstance(stmt, exp.Select)`.
- `tree.args.get("from_")` must be exactly a bare table reference named `__THIS__` (`isinstance(tree.args["from_"].this, exp.Table)`, `.this.name == "__THIS__"`, and `.this.alias` empty — `Table.alias` is `""` when unaliased, the alias string otherwise). Anything else → `ValueError` naming what was found.
- Each of `tree.args.get("joins")`, `.get("where")`, `.get("group")`, `.get("having")`, `.get("order")`, `.get("limit")` must be `None`/empty. Any present → `ValueError` naming the specific clause ("JOIN is not yet supported by SQLTransform", etc.) — this is the "fail loud at the boundary" mechanism `SQL_SUPPORT.md` commits to.

## `_sql.py`: Window Aggregate Discovery

```python
@dataclass(frozen=True)
class WindowAgg:
    node: exp.Window       # the actual node, for rewrite_sql to .replace()
    fn: str                 # canonical, e.g. "AVG" (post-synonym-mapping)
    col: str                 # bare column name, real case preserved
    has_partition: bool
    has_order: bool

def find_window_aggregates(select: exp.Select) -> list[WindowAgg]:
    ...
```

- `select.find_all(exp.Window)` — flat, since v1 forbids subqueries/joins that could nest another `Select`.
- For each `Window` node `w`:
  - Function name: `w.this.sql_name()` if `w.this` is a recognized sqlglot function class (e.g. `exp.Avg`, `exp.Sum`), else `w.this.this.upper()` for `exp.Anonymous`. Apply `_FUNCTION_SYNONYMS = {"MEAN": "AVG"}` (only entry needed — no other synonym is exercised by any current test or documented usage).
  - Column: single-arg only (matches today's scope — no multi-arg aggregates supported). `w.this.this` if it's already a bare `exp.Column` (recognized function classes store their one arg in `.this`), else `w.this.expressions[0]` for `exp.Anonymous`. If the resolved arg isn't an `exp.Column`, raise `ValueError` ("window aggregate argument must be a plain column").
  - `has_partition = bool(w.args.get("partition_by"))`, `has_order = bool(w.args.get("order"))` — direct structural checks, no text matching, no possibility of the two flags disagreeing with what `_rewrite.py` sees (same `WindowAgg` record feeds both).

## `_state.py`: extract_state (revised)

```python
def extract_state(
    windows: list[WindowAgg],
    ctx: datafusion.SessionContext,
    table_name: str,
) -> BaseModel:
```

- For each `WindowAgg` in `windows`: if `has_partition` → `NotImplementedError("PARTITION BY ...")`; if `has_order` → `NotImplementedError("ORDER BY ...")` (same messages as today, just triggered by the structural flags instead of a regex miss).
- Dedup by `(fn, col)` (case preserved, per the case-collision fix already in place) — build the `pairs` set, then for each distinct pair run `SELECT {fn}("{col}") FROM {table_name}` (quoted column, per the case-preservation fix), same as today.
- `state_key(fn, col)` and the ambiguous-collision `ValueError` guard are unchanged (already correct, already tested).

`_WINDOW_AGG_RE` is deleted entirely.

## `_rewrite.py`: rewrite_sql (revised)

```python
def rewrite_sql(select: exp.Select, windows: list[WindowAgg]) -> str:
```

- Build `window_to_key: dict[int, str]` mapping `id(w.node)` → `state_key(w.fn, w.col)` for every `WindowAgg` (identity-keyed, since the same `Window` node object from `find_window_aggregates` is what appears in the tree being rewritten — `_state.py` and `_rewrite.py` both consume the *same* discovery pass, so there's no risk of the two disagreeing about which nodes are window aggregates, unlike today's two-independent-regexes situation).
- For each top-level `select.expressions` item `expr`:
  - `out_name = expr.alias_or_name`. Empty (`""`) → `ValueError("Expression in SELECT list needs an alias (AS name)")` — this single check replaces both the `AttributeError` crash and the invalid-generated-SQL bug fixed in `2b3171c`; sqlglot's `alias_or_name` already returns `""` for exactly the "no meaningful name" case that DataFusion's plan-derived names produced garbage for.
  - Walk `expr` and, for every `exp.Window` node found, `node.replace(exp.column(window_to_key[id(node)], table="__STATE__"))`.
  - Walk the (now Window-free) remainder and, for every remaining `exp.Column` node: if `.table` is set and isn't `"__THIS__"` → `ValueError` (shouldn't be reachable given `_sql.py`'s scope validation already rejects other tables, but defends against a future scope-validation gap rather than silently mis-qualifying); otherwise `node.replace(exp.column(node.name, table="__THIS__"))`.
- Append `__STATE__` to the FROM clause as a plain comma-joined table (matching today's `FROM __THIS__, __STATE__` — a cross join, since `__STATE__` is always exactly one row; `InferFn`'s Rust parser already accepts this form, exercised by `tests/test_interpreter.py::test_cross_join`).
- Return `select.sql()` — the whole rewritten tree serialized once, not string-concatenated piecemeal.

`_WINDOW_COL_RE` and `_VALID_IDENT_RE` are deleted entirely (the alias-required check subsumes what `_VALID_IDENT_RE` was defending against).

## `__init__.py`: fit() (revised)

```python
def fit(self, table, /, this_model=None) -> SQLTransform:
    this_model = this_model or synthesize_this_model(table.schema)

    tree = parse_and_validate(self._sql)
    windows = find_window_aggregates(tree)

    ctx = datafusion.SessionContext()
    ctx.from_arrow(table, name="__THIS__")

    self._state = extract_state(windows, ctx, "__THIS__")
    rewritten_sql = rewrite_sql(tree.copy(), windows)
    self._infer_fn = InferFn(
        rewritten_sql,
        row_tables={"__THIS__": this_model, "__STATE__": type(self._state)},
        static_tables={},
    )
    return self
```

Note `tree.copy()` passed to `rewrite_sql` — `extract_state` only reads the tree (via the `WindowAgg` list, not the tree directly), but `rewrite_sql` mutates it via `.replace()`; passing a copy keeps `fit()`'s own `tree`/`windows` untouched in case of future callers that need both (defensive, cheap — sqlglot trees are small for this scope).

## Error Taxonomy

No new error *types* — everything is `ValueError` (scope violations, missing alias, bad qualifier, ambiguous state key) or `NotImplementedError` (`PARTITION BY`/`ORDER BY`), same as today. What changes is *where* each is raised: at `_sql.py`'s validation step (immediately after parsing, before any DataFusion work) rather than scattered across a plan-text regex miss and a downstream `InferFn` failure.

## Non-Goals

- **Widening `SQLTransform`'s accepted SQL surface.** This spec is a foundation swap; `WHERE`/`JOIN`/`GROUP BY`/etc. are explicitly rejected, not silently accepted. Tracked as a future decision in `SQL_SUPPORT.md`/`VISION.md`.
- **`PARTITION BY` support via join-based per-partition state tables.** Deliberately deferred — see `VISION.md`'s roadmap. This spec's `WindowAgg.has_partition` flag exists so that future work has a clean, already-structural signal to build on, but building the join-synthesis itself is out of scope here.
- **Growing `SQLTransform`'s public API** (e.g. accepting extra tables for joins). Not needed until the above lands.
- **A general-purpose sqlglot-based SQL passthrough.** `_sql.py` validates a narrow, explicit allowlist of what's structurally present in the parsed tree — it is not designed to eventually accept "anything sqlglot can parse."

## Testing Strategy

Every existing test in `_state_test.py`, `_rewrite_test.py`, and the SQL-shape-relevant parts of `__init___test.py` continues to pass unchanged (same public behavior, same error types, same messages) — this is the acceptance bar, not a rewrite of test intent. New coverage:

| Test | Covers |
|---|---|
| `MEAN(age) OVER ()` and `AVG(age) OVER ()` produce the same `state_key` | Synonym normalization preserves today's naming |
| `SUM`, or any other DataFusion-recognized aggregate, still works | No new function-name allowlist introduced |
| `SELECT age WHERE age > 1 FROM __THIS__` | `ValueError` naming `WHERE`, raised before any DataFusion call |
| `SELECT a.age FROM __THIS__ JOIN b ...` | `ValueError` naming `JOIN` |
| `SELECT age FROM data` (wrong table name) | `ValueError` naming the bad `FROM` target |
| `SELECT foo.age FROM __THIS__` (bad qualifier, no such table) | `ValueError` from the rewrite-time qualifier check |
| Two statements in one string (`SELECT 1; SELECT 2`) | `ValueError`, mirrors `InferFn`'s own multi-statement rejection |
| `_state.py` and `_rewrite.py` given the same `WindowAgg` list from one `find_window_aggregates` call | No possibility of the two disagreeing (regression guard against the class of bug fixed in `2b3171c`) |
| Existing ORDER BY / multi-column PARTITION BY regression tests (`2b3171c`) | Still raise `NotImplementedError`, now via `has_order`/`has_partition` instead of regex |
