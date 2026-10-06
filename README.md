# SQL Transforms

Define ML feature transforms as SQL, fit once, then serve them row-at-a-time
with sub-microsecond latency.

## Packages

This repository is a workspace of two packages:

| package | what it is |
|---|---|
| [`packages/confit`](packages/confit) | **Confit** — the serving engine. SQL plus static tables frozen at fit time are partially evaluated, once, into a native function. Serves bit-exact with DuckDB (optimizer-off reading) or refuses at build time. Usable on its own. |
| [`packages/sql-transform`](packages/sql-transform) | The authoring surface: `SQLProjection`. The **fit half works**: window aggregates over `__THIS__` are marginalized into materialized params tables plus a rewritten serving SQL. The serving half (through Confit) is a later loop. |

## Installation

```bash
pip install sql-transform      # authoring + serving
pip install confit             # the serving engine alone
```

### Development

```bash
git clone https://github.com/ahrzb/sql-transforms.git
cd sql-transforms
mise run install        # uv sync — installs both packages, builds Confit's extension
```

With Nix, `nix develop` gives a shell with the pinned toolchain (Python 3.14,
uv, Rust, JDK 21 for the Spark gate); run `uv sync --group spark` inside it.

Confit ships a Rust/PyO3 extension (`confit._engine`) that
[maturin](https://www.maturin.rs/) builds. sql-transform is pure Python. After
you change Rust code, run `uv run maturin develop` in `packages/confit`. The
tests also rebuild the extension when a `.rs` file is newer than the built
module.

Run the whole gate from the repository root:

```bash
uv run pytest -q && cargo test --release
```

## Quick Start

Confit, the serving engine, works today:

```python
import pyarrow as pa
from pydantic import BaseModel
from confit import DuckDBInferFn

class Row(BaseModel):
    age: float

fn = DuckDBInferFn(
    "SELECT (age - 30.0) / 10.0 AS age_z FROM __THIS__",
    row_tables={"__THIS__": Row},
    static_tables={},
    shape="map",
)
fn.infer_rows([Row(age=40.0)])          # row objects in, row objects out
fn.infer_arrow(pa.table({"age": [40.0]}))  # pa.Table in, pa.Table out
```

`sql_transform.SQLProjection` is the authoring layer on top of Confit. You
write SQL with window aggregates, and `fit()` computes them once. The fit half
works today. See [packages/sql-transform](packages/sql-transform).

```python
from sql_transform import SQLProjection

p = SQLProjection(
    "SELECT (age - avg(age) OVER (PARTITION BY country)) AS d FROM __THIS__"
).fit(train)
p.serving_sql   # the rewritten projection: params joins instead of aggregates
p.params        # {"__CF_PARAMS_0__": <pyarrow.Table>}
```

## Architecture

The two-phase shape — **fit works, the wiring between the phases is the next
loop**:

```
SQL over __THIS__
      │
      ▼
   fit(train) ── marginalize each window aggregate (e.g. avg(age) OVER
      │          (PARTITION BY country)) into a materialized params table and
      │          rewrite the SQL to LEFT JOIN it instead of recomputing.
      │          Bit-exact with DuckDB by differential gate.      [WORKS]
      │
      │  rewritten SQL + frozen params          [wiring: NOT IMPLEMENTED]
      ▼
   Confit ── partially evaluates the pair into a native function: binding-time
      │      analysis collapses every static lookup into a prepare-time probe,
      │      so nothing general remains at call time.             [WORKS]
      ▼
 infer(row) / infer_batch(rows)
```

Confit's contract has two outcomes. In the first, the SQL and its static
tables become a function that is bit-exact with DuckDB. In the second, the
build stops with an error that names the construct that Confit does not
serve. The reference is
DuckDB with `PRAGMA disable_optimizer` (the optimizer-off reading).
[Confit's known limitations](packages/confit/docs/known-limitations.md) says
why, and lists the refusals.

## What Confit supports

[`packages/confit`](packages/confit) documents what Confit serves:
expressions, joins to static tables, row shapes (`map`/`filter`/`many`), and
Arrow input and output.
[known-limitations.md](packages/confit/docs/known-limitations.md) lists what it
refuses. The corpus is 678 SQL
statements from DuckDB's own test suite. Confit serves **543 of 678**
bit-exact, and refuses the others with a named error (2026-10-05,
[counts](packages/confit/docs/reports/corpus-counts.json)).

In the authoring layer, `fit` of window aggregates (marginalization) works.
The next work is typed input and output, serving through Confit, and static
tables in authored SQL. After that comes the DRAFT-20 program, which stores
fitted models as params tables.

## Reports

- [The architecture of Confit](packages/confit/docs/reports/confit-architecture.md)
- [Pins-first: building a bit-exact engine twin](packages/confit/docs/reports/pins-first-methodology.md)
- [Performance: the serving regime, measured](packages/confit/docs/reports/performance-report.md)

## Development

```bash
mise run test     # uv run pytest
mise run fmt      # ruff check + format
mise run check    # fmt + test
mise tasks        # list all tasks
```
