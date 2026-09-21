# Per-row aggregation and the static-only fold, reading (2026-09-21)

> Current definitions: [scope](../goal.md#scope) and the
> [comparison contract](../oracle/05-the-comparison-contract.md).
> The dated measurements below are unchanged.

**What this is.** The dated reading behind three changes to `packages/confit/docs/goal.md`
made on this date: claim: float-reduction-bound, the move of per-row aggregation out of
exclusion: whole-relation-shapes, and the retirement of that row's static-tables-only
carve-out. The goal holds the target; this file holds what was measured and how far today's
engine is from it. Every block below was executed and prints what it printed.

## 1. Environment and reproduction {#environment-and-repro}

DuckDB 1.5.5 (Python wheel), Windows 11, `confit.BUILD_PROFILE == "release"`. The engine
was built from `694ecf6`; `git diff 694ecf6 origin/master -- packages/confit/src
packages/confit/confit packages/confit/fuzz packages/confit/tests/test_corpus_replay.py` is
empty against master `0fc02ae`, so every reading here is a reading of master.

The examples share this preamble:

```python
import pyarrow as pa
from confit import DuckDBInferFn
ROW = pa.schema([("price", pa.float64()), ("city", pa.string())])
PRICES = pa.table({"city": ["de", "de", "fr"], "price": [1.0, 3.0, 5.0]})
S = pa.table({"v": pa.array([1, 2, 3], pa.int64())})
```

## 2. finding: float-sum-run-variance {#float-sum-run-variance}

The premise of claim: float-reduction-bound. One table, one query, one connection, twenty
runs at each thread count; the result's bits are collected into a set.

```python
c = duckdb.connect()
c.execute("create table s as select (random()-0.5)*1e6 as v, (i%7) as g "
          "from range(3000000) t(i)")
c.execute("set threads=8")   # then threads=1
{bits(c.execute("select sum(v) from s").fetchone()[0]) for _ in range(20)}
```

| `threads` | distinct `sum(v)` bit patterns in 20 runs | distinct `avg(v)` |
|---|---|---|
| 8 | **20** (`41bc3232fa31fd55`, `41bc3232fa31fdc0`, `41bc3232fa31fec2`, ...) | **20** |
| 1 | 1 | 1 |

A plain in-order loop — one IEEE add per row, in table order — was then compared with the
single-threaded answer over a second table built the same way:

| reduction | loop == DuckDB at `threads=1` |
|---|---|
| `sum(v)` | yes, `c190a228e7277a0f` both |
| `avg(v)` (loop sum / n) | yes, `c0374156bfa83c98` both |
| `sum(v) GROUP BY g` | 7 of 7 groups |

Read together: the run-to-run spread is an ordering effect, there is no single
multi-threaded bit pattern to be identical to, and the bound in the goal is a bound between
orderings. Seven groups and two reductions is a small reading; it is a premise check, not a
parity gate, and the goal marks the bound itself `Unverified`.

## 3. gap: per-row-aggregation {#per-row-aggregation}

*(was part of `exclusion: whole-relation-shapes`.)* An aggregate over the static rows one
input row matches. The target serves it; today both spellings refuse:

```python
DuckDBInferFn(
    "SELECT (SELECT avg(p.price) FROM prices AS p WHERE p.city = t.city) AS city_avg "
    "FROM __THIS__ AS t",
    row_tables={"__THIS__": ROW}, static_tables={"prices": PRICES})
# ValueError: unsupported: expression: (SELECT avg(p.price) FROM prices AS p
#             WHERE p.city = t.city)

DuckDBInferFn(
    "SELECT t.city, avg(p.price) AS city_avg FROM __THIS__ AS t "
    "JOIN prices AS p ON p.city = t.city GROUP BY t.city",
    row_tables={"__THIS__": ROW}, static_tables={"prices": PRICES}, shape="many")
# ValueError: unsupported: GROUP BY / HAVING / aggregation
```

What is already there: a join under `shape="many"` walks every static row an input row
matches. What is missing is an accumulator over that walk in place of one emitted row per
match, and the binder support for the spelling.

The two spellings are not equivalent, which is the design question this gap opens and does
not answer. The correlated scalar subquery is per-row by construction. `JOIN` plus
`GROUP BY` is per-row only when the grouping yields exactly one group per input row; two
input rows sharing `t.city` collapse into one output row, which is the batch dependence
exclusion: whole-relation-shapes exists to keep out.

The same aggregate over the input rows stays out by decision and is unchanged:

```python
DuckDBInferFn("SELECT sum(price) AS total FROM __THIS__",
              row_tables={"__THIS__": ROW}, static_tables={})
# ValueError: unsupported: aggregate function sum (no aggregation in v0)
```

## 4. gap: static-only-fold {#static-only-fold}

*(was the static-tables-only carve-out of `exclusion: whole-relation-shapes`.)* The target
refuses a query that reads no row table. Today's engine serves it: when the row path refuses,
`DuckDBInferFn::new` hands the whole SQL text to a fresh DuckDB connection that holds only
the statics (`eval_static_only`, `packages/confit/src/duckdb/mod.rs`), and freezes the rows
that come back. It is a whole-query fallback; there is no analysis of static-only parts of a
query that also reads the row table.

```python
fn = DuckDBInferFn("SELECT max(v) AS top FROM s",
                   row_tables={"__THIS__": ROW}, static_tables={"s": S})
fn.backend, fn.boundary, fn.infer_rows([])   # ('constant', 'constant', [{'top': 3}])

fn = DuckDBInferFn("SELECT v AS o FROM s", row_tables={"__THIS__": ROW},
                   static_tables={"s": S})
fn.backend                                   # 'constant'
```

**What the path serves, counted.** `test_corpus_replay.py`'s replay was run with the
constructor wrapped to record `fn.backend` for every case:

| outcome, backend | cases |
|---|---|
| match, `cranelift` | 517 |
| match, `interpreter` | 21 |
| match, `constant` | 4 |
| unsupported | 136 |

The four `constant` matches read a table function and no table:
`FROM generate_series(1000, 2000)`, `FROM range(4)` twice, `FROM test_all_types()`. The
wrapped replay matched 542 where the suite's floor is 547; the five-case difference is the
`SELECT COUNT(*) FROM t` statements of one source file, which match in the suite because the
fresh connection resolves `t` to a pyarrow table that happens to be a local variable of the
calling test frame, and stop matching when a wrapper frame sits in between. That attribution
is an inference from the count and the statements, not a per-case trace. Either way, **no
corpus match is a query over a static table the caller supplied.**

The campaign generator gives the path its own arm: `fuzz/gen.py:1257`, the `static_agg`
branch, six points of the query-shape draw when the case has statics.

**What closing the gap touches.** `eval_static_only` and `Engine::Constant` in
`src/duckdb/mod.rs`; the `"constant"` value of `backend` and `boundary` (public, and
documented in `packages/sql-transform/sql_transform/_projection.py:503-508`); the
`static_agg` arm and the `constant-ordered` / `constant-unordered` compare modes in `fuzz/`;
the oracle spec's chapters that describe them; `known-limitations.md:95-120`; the row-limit
refusal test in `test_arrow_schema_api.py`. The corpus floor drops by the nine matches above,
547 to 538, for one named reason.

**What it closes.** finding: static-only-tie-order
(`2026-09-02-goal-baseline.md`) is a symptom of this path — a frozen order the query does
not determine — and has no subject once the path is gone. The unmerged branch
`refuse-static-tie-order`, which refuses such shapes one class at a time, is superseded and
is not to be merged. The decision record is
`packages/confit/docs/decisions/trustworthy-fold.md`.
