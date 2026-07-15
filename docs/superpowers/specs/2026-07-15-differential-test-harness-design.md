# Differential Test Harness for the Rust Engine — Design

**Goal:** Make testing the Rust `InferFn` interpreter highly automated by giving it
a small differential-testing harness: a single `check(query, tables, expect=)`
helper that runs a query through both **DataFusion (the oracle)** and the **Rust
`InferFn`** over the same typed input, and asserts their output values match.
Tests are ordinary `@pytest.mark.parametrize` decision tables — no YAML, no custom
loader.

## Motivation

`tests/test_interpreter.py` already tests the Rust engine differentially by hand:
each test builds a DataFusion oracle (`_expected`), runs `InferFn` on the same
data, and compares `model_dump()` to the oracle. That pattern is correct but
verbose and re-typed per test. This harness extracts it into one reusable helper
plus a few table builders, so a new engine test is a one-line `check(...)` inside
a native parametrized test — the "bind variables, check results in a table" shape,
expressed as pytest parametrization.

Native pytest (over a YAML fixture format) was chosen deliberately: test titles are
function names + param ids; `xfail` and error expectations use native
`pytest.mark.xfail(strict=True)` and `pytest.raises`; each decision-table row is an
independently runnable/debuggable case; and there is no schema mini-language or
parser to maintain.

## The harness (`tests/differential.py`)

```python
def check(query: str, tables: dict[str, Table], expect: list[dict] | None = None) -> None:
    """Run `query` through DataFusion (oracle) AND the Rust InferFn over the same
    typed tables; assert their output rows match (order-insensitive, float-tolerant,
    NULL-aware). If `expect` is given, also assert the output equals it."""

def row(**cols) -> Table:            # single-row `row` table; types inferred from the values
def rows(schema, data) -> Table:     # multi-row `row` table; explicit schema
def static(schema, data) -> Table:   # preloaded static table (InferFn static_tables)
```

- **`Table`** is a small dataclass: `kind` (`"row"` | `"static"`), `schema`
  (`pa.Schema`), `rows` (`list[dict]`).
- **`schema`** accepts either a **type-spec dict** — `{"a": "int", "b": "float?",
  "name": "str"}` where the base is `int`/`float`/`str`/`bool` and a trailing `?`
  marks the column nullable — **or a Pydantic `BaseModel` subclass** (used directly
  as the `InferFn` row model, and reflected to a `pa.Schema` for DataFusion). Python
  builtins (`int`, `float`, `str`, `bool`) are accepted as type values too.
- `row(**cols)` infers each column's type from its value (`int`→int64,
  `float`→float64, `str`→string, `bool`→bool; a `None` value makes the column
  nullable). It is the terse default for scalar decision tables; use `rows(...)`
  when you need explicit types or nullable-empty columns.

### What `check` does

1. Build a `pa.Schema` per table from its `schema`.
2. **DataFusion oracle:** register every table (row *and* static) via
   `ctx.from_arrow(pa.Table.from_pylist(rows, schema))`; `ctx.sql(query).collect()`
   → `to_pylist()`.
3. **Rust `InferFn`:** `row` tables → `synthesize_this_model(schema)` (existing
   `_schema.py` helper) unless a `BaseModel` was supplied → `row_tables`; `static`
   tables → `pa.Table` → `static_tables`. `InferFn(query, row_tables,
   static_tables).infer({name: rows for row-tables})` → `model_dump()`.
4. **Compare** with `_rows_equal` and, if `expect` given, compare to it too.

### `_rows_equal` — comparison semantics

- **Order-insensitive.** The Rust engine preserves input row order; DataFusion does
  not guarantee output order across a join, and the Rust Layer-1 subset has no
  `ORDER BY`, so ordering is not a meaningful axis to assert here. Compare as
  multisets: sort both sides by a canonical key, then element-wise.
- **Column set must match** exactly (same output column names).
- **Float-tolerant:** numeric values compared with `abs(a - b) <= 1e-9` (matching
  the existing `assert_approx_equal`); exact for ints/str/bool.
- **NULL-aware:** `None` equals `None`, and `None` never equals a value.

## Markers — divergences and errors, natively

- **Known divergence** (engines legitimately differ, e.g. int mod-by-zero error
  type): mark the parametrize case
  `pytest.param(..., marks=pytest.mark.xfail(reason="…", strict=True))`. It runs,
  is expected to fail the value match, and flips to a loud failure (`XPASS` under
  `strict`) if the engines ever start agreeing.
- **Both engines must reject** a query (unknown column, self-join, unsupported
  clause): wrap the `check(...)` in `with pytest.raises(Exception)` (or the specific
  type). `check` raising means *at least one* engine rejected; for cases where we
  want to assert *both* reject with a shared message, a `check_both_raise(query,
  tables, match=...)` variant asserts each engine raises and its message matches.

## Test organization

Group by capability (each file is native parametrized tests calling `check`):

```
tests/differential.py              (new) — check(), row/rows/static, _rows_equal
tests/test_diff_expressions.py     (new) — arithmetic, comparisons, AND/OR/NOT,
                                            CAST, UPPER/LOWER/TRIM/SUBSTR/CONCAT,
                                            ABS/ROUND/COALESCE/NULLIF, NULL propagation
tests/test_diff_relational.py      (new) — WHERE, INNER/CROSS join, static LookupJoin,
                                            LEFT lookup join (hit→value, miss→NULL),
                                            multi-row order preservation
tests/test_diff_errors.py          (new) — unknown col, self-join, unsupported clause
                                            (pytest.raises); known divergences (xfail)
```

`tests/test_interpreter.py`'s existing value-comparison tests are migrated to
`check(...)` (they are already differential); its construction/validation tests
(`test_module_imports_and_constructs`, the `ValueError` construction checks) move to
`test_diff_errors.py` or stay as focused unit tests. The three Task-4 LEFT-lookup
tests are re-expressed via `check`. Net: `test_interpreter.py` shrinks to anything
genuinely not differential; the differential surface lives in the new files.

## Coverage the harness must make easy (acceptance)

Each of these is a short parametrized test calling `check`:

- Arithmetic `+ − * / %` including int/int truncation, int vs float promotion, and
  negative operands.
- Comparisons and boolean logic with three-valued NULL truth tables.
- `CAST` across INT/FLOAT/STR/BOOL, including float→int truncation.
- String builtins and `ABS/ROUND/COALESCE/NULLIF`.
- NULL propagation through arithmetic and functions.
- `WHERE` row filtering (N in → M out).
- `INNER`/`CROSS` join; row⋈static `LookupJoin`; LEFT lookup join hit and miss
  (miss → NULL, both engines).
- Errors rejected by both engines; known divergences tracked as `xfail`.

## Non-goals

- No YAML / external fixture files, no custom loader or type mini-language.
- Not a fuzzer — cases are hand-written decision tables, not randomly generated.
- Not testing `SQLTransform`'s fit/rewrite pipeline (covered by its own tests) —
  this harness targets the Rust `InferFn` execution surface (Layer 1) directly.
- No `ORDER BY`/aggregate coverage — outside the Rust engine's supported subset.
- Not changing any engine behavior; this is tests-only. (Divergences it surfaces
  are tracked as `xfail`, fixed separately.)
