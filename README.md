# SQL Transforms

Define ML feature transforms as SQL, fit once, then run them at batch or
low-latency single-row speed.

## Installation

```bash
pip install sql-transform
```

### Development

```bash
git clone https://github.com/ahrzb/sql-transforms.git
cd sql-transforms
mise run install        # uv sync — installs deps and builds the Rust extension
```

The inference engine is a Rust/PyO3 module built by [maturin](https://www.maturin.rs/).
After changing Rust code, rebuild it with `uv run maturin develop`.

## Quick Start

```python
from sql_transform import SQLTransform
import pyarrow as pa

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

transformer = SQLTransform(sql)
transformer.fit(data)

# Batch transform through DataFusion (pyarrow in / pyarrow out).
result = transformer.transform(data)
print(result)

# Low-latency inference through the native engine (dict or Pydantic model in,
# typed model out). infer() for one row, infer_batch() for many.
one = transformer.infer({"feature1": 2.0, "feature2": 20})
print(one.feature1_norm)
many = transformer.infer_batch([{"feature1": 2.0, "feature2": 20}])
```

Per-group statistics use `OVER (PARTITION BY ...)` — the group means/counts/etc.
are frozen at `fit` and looked up per row at inference:

```python
sql = "SELECT target / MEAN(target) OVER (PARTITION BY city) AS enc FROM __THIS__"
```

## What it supports

- **Window aggregates**, computed once at `fit` and frozen: whole-table `OVER ()`
  and per-group `OVER (PARTITION BY ...)` (e.g. `MEAN`, `SUM`, `COUNT`, `STDDEV`).
- **Expressions** (batch and inference): arithmetic, comparisons, `CAST`,
  `UPPER`/`LOWER`/`TRIM`/`SUBSTR`/`CONCAT`, `ABS`, `ROUND`, `COALESCE`, `NULLIF`,
  with SQL NULL-propagation semantics.
- **Joins**: `INNER`/`CROSS`, plus a static-table lookup join (a row joined to a
  preloaded `pyarrow.Table` by key — no per-row Python callback).
- **Typed I/O**: Pydantic models for the input row and output, validated when the
  transformer is built and again at call time. Output is a typed model; the input
  schema is auto-synthesized or user-supplied.

See [docs/SQL_SUPPORT.md](docs/SQL_SUPPORT.md) for the feature-by-feature tracker.

## Architecture

Two phases, two engines, one rewritten query:

```
SQL over __THIS__
      │
      ▼
   fit(train) ── DataFusion runs the SQL, freezes each window-aggregate (e.g.
      │          MEAN(age)) into a typed __STATE__, and rewrites the SQL to
      │          reference __STATE__ + the raw row __THIS__ instead of
      │          recomputing aggregates.
      │
      │  rewritten SQL + frozen state
      ├───────────────────────────────┬───────────────────────────────┐
      ▼                               ▼
 transform(batch)                infer(row) / infer_batch(rows)
 DataFusion, vectorized           native InferFn interpreter, row-at-a-time,
 columnar over the batch          no SQL engine at call time
```

Both paths run the **same** rewritten SQL against the **same** frozen state, so
they return identical values on the normal numeric path. `fit` pays for a real
query engine once; inference pays only for a lean interpreter walking a plan. This
separation of **fit** (compute statistics) and **transform/infer** (apply them)
is the standard ML pattern — fit on training data, apply to training and serving.

## Development

```bash
mise run test     # uv run pytest
mise run fmt      # ruff check + format
mise run check    # fmt + test
mise tasks        # list all tasks
```

## Project docs

- [docs/ROADMAP.md](docs/ROADMAP.md) — the sequenced milestones (narrative) and progress.
- [docs/BACKLOG.md](docs/BACKLOG.md) — the reasoning archive: *why* things were
  deferred, decision context, source citations. Actionable tasks live in Backlog.md.
- **Backlog.md** (`backlog/`) — live task board (`backlog board` / `backlog task
  list`) grouped by **milestones** (`backlog milestone list`), architecture
  **decisions** (`backlog/decisions/`), and reference **docs** (`backlog/docs/`):
  [Vision](<backlog/docs/doc-3 - Vision.md>) (what the project is and where it's
  headed), the [DataFusion function catalogue](<backlog/docs/doc-1 - DataFusion-function-catalogue.md>)
  (interpreter parity target, auto-generated), and the
  [sklearn transformer plan](<backlog/docs/doc-2 - sklearn-transformer-implementation-plan.md>)
  (tiers + native-machinery status), and the
  [Epic B design brief](<backlog/docs/doc-4 - Epic-B-multi-language-inference-runtimes-design-brief.md>)
  (multi-language serving — unscoped).

## Contributing

1. Pick an item from [docs/ROADMAP.md](docs/ROADMAP.md) or
   [docs/BACKLOG.md](docs/BACKLOG.md).
2. Open an issue.
3. Implement with tests.
4. Submit a PR.

## License

MIT
