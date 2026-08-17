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

Confit ships a Rust/PyO3 extension (`confit._engine`) built by
[maturin](https://www.maturin.rs/); sql-transform is pure Python. After changing
Rust code, rebuild with `uv run maturin develop` from `packages/confit`; the test
suite also rebuilds automatically when a `.rs` file is newer than the built
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

`sql_transform.SQLProjection` is the authoring layer on top of it — write SQL
with window aggregates, `fit()` to freeze them. The fit half works today; see
[packages/sql-transform](packages/sql-transform).

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

Confit's contract: SQL plus frozen tables either specialize into a function
bit-exact with DuckDB (the optimizer-off reading — `PRAGMA
disable_optimizer`, and [why](docs/known-limitations.md)), or construction
raises and names the construct it will not serve — see
[Confit's known limitations](docs/known-limitations.md).

## What Confit supports

The expression surface, joins to static tables, the row-shape contract
(`map`/`filter`/`many`) and the Arrow boundary are documented in
[`packages/confit`](packages/confit) and
[docs/known-limitations.md](docs/known-limitations.md): **550 of 678** statements
mined from DuckDB's own test suite replay bit-exact, with the remainder clean,
named build-time rejections.

The authoring layer's window-aggregate `fit` (marginalization) works; typed
I/O, serving through Confit, static tables in authored SQL, and the broader
DRAFT-20 program (fitted model artifacts as params tables) are the named next
loops.

## Reports

- [The architecture of Confit](docs/reports/confit-architecture.md)
- [Pins-first: building a bit-exact engine twin](docs/reports/pins-first-methodology.md)
- [Performance: the serving regime, measured](docs/reports/performance-report.md)

## Development

```bash
mise run test     # uv run pytest
mise run fmt      # ruff check + format
mise run check    # fmt + test
mise tasks        # list all tasks
```
