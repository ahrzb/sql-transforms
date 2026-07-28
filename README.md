# SQL Transforms

Define ML feature transforms as SQL, fit once, then serve them row-at-a-time
with sub-microsecond latency.

## Packages

This repository is a workspace of two packages:

| package | what it is |
|---|---|
| [`packages/confit`](packages/confit) | **Confit** — the serving engine. SQL plus static tables frozen at fit time are partially evaluated, once, into a native function. Serves bit-exact with DuckDB or refuses at build time. Usable on its own. |
| [`packages/sql-transform`](packages/sql-transform) | The authoring layer: `SQLTransform` — fit window-aggregate state from training data, then serve through Confit. |

> **sql-transform was reset.** The DataFusion-differentiated native engine, the
> Python codegen backend, and the batch `transform()` path were removed; the
> package is being rebuilt on Confit. Confit itself is unaffected.

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

```python
import pyarrow as pa
from sql_transform import SQLTransform

data = pa.table({
    "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
    "feature2": [10, 20, 30, 40, 50],
})

# The input table is always referenced as __THIS__.
sql = """
SELECT
    feature1 / MEAN(feature1) OVER () AS feature1_norm,
    feature2 / SUM(feature2) OVER () AS feature2_share
FROM __THIS__
"""

t = SQLTransform(sql)
t.fit(data)

# Serving: dict or Pydantic model in, typed model out.
one = t.infer({"feature1": 2.0, "feature2": 20})
print(one.feature1_norm)
many = t.infer_batch([{"feature1": 2.0, "feature2": 20}])
```

Per-group statistics use `OVER (PARTITION BY ...)` — the group means/counts are
frozen at `fit` and looked up per row at inference:

```python
sql = "SELECT target / MEAN(target) OVER (PARTITION BY city) AS enc FROM __THIS__"
```

## Architecture

Two phases, one rewritten query:

```
SQL over __THIS__
      │
      ▼
   fit(train) ── DataFusion runs the SQL, freezes each window aggregate (e.g.
      │          MEAN(age)) into a typed __STATE__ table, and rewrites the SQL
      │          to reference __STATE__ + the raw row __THIS__ instead of
      │          recomputing aggregates.
      │
      │  rewritten SQL + frozen state
      ▼
   Confit ── partially evaluates the pair into a native function:
      │      binding-time analysis collapses every static lookup into a
      │      prepare-time probe, so nothing general remains at call time.
      ▼
 infer(row) / infer_batch(rows)
```

`fit` pays for a real query engine once; serving pays only for straight-line
native code over the frozen state. Confit's contract carries through: the fitted
SQL either serves bit-exact with DuckDB, or `fit()` raises and names the
construct it will not serve — see
[Confit's known limitations](docs/known-limitations.md).

## What it supports

- **Window aggregates**, computed once at `fit` and frozen: whole-table `OVER ()`
  and per-group `OVER (PARTITION BY ...)` (`MEAN`, `SUM`, `COUNT`, `STDDEV`, …).
- **Everything Confit serves** at inference — the expression surface, joins to
  static tables, and the row-shape contract are documented in
  [`packages/confit`](packages/confit) and
  [docs/known-limitations.md](docs/known-limitations.md).
- **Typed I/O**: Pydantic models for the input row and output, validated when the
  transform is fitted and again at call time.

Not currently supported, pending the rebuild: batch `transform()`, sklearn
transformer references, and composing one `SQLTransform` into another.

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
