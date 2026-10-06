# SQL Transforms

Author compositional SQL over fit and request data. Fit once, then use batch
execution or compile row-local projections through Confit.

Read the [system specification](docs/specs/README.md) for current behavior,
interfaces, and refusals.

## Packages

This repository is a workspace of two packages:

| package | what it is |
|---|---|
| [`packages/confit`](packages/confit) | **Confit** — the serving engine. SQL plus static tables frozen at fit time are partially evaluated, once, into a native function. Serves bit-exact with DuckDB (optimizer-off reading) or refuses at build time. Usable on its own. |
| [`packages/sql-transform`](packages/sql-transform) | The authoring package: general `SQLTransform` composition and row-local `SQLProjection`. Fit binds fit data and writes params. Confit compiles admitted fitted projections or names a construct it does not serve. |

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

`sql_transform` authors SQL over fit data (`__FIT__`) and request data
(`__THIS__`). Fit binds the fit data and returns a fitted artifact:

```python
import pyarrow as pa
from sql_transform import SQLProjection

fit_data = pa.table({"v": [10.0, 20.0]})
requests = pa.table({"v": [12.0, 22.0]})
fitted = SQLProjection(
    "SELECT t.v - p.m AS d FROM __THIS__ t, "
    "(SELECT avg(v) AS m FROM __FIT__) p"
).fit(fit_data)

assert fitted.transform(requests).to_pydict() == {"d": [-3.0, 7.0]}
fn = fitted.compile()
assert fn.infer_rows(requests.to_pylist()) == [{"d": -3.0}, {"d": 7.0}]
```

Use `SQLProjection.marginalize(...)` for the bounded one-level window
convenience. The ordinary constructor expects explicit fit/request SQL.
The [authoring contract](packages/sql-transform/docs/contract.md) defines
composition, executable examples, public serving artifacts, and refusals.

## Architecture

The authoring package and Confit have separate admission checks:

```text
Authored SQL: (fit data, request data) -> result
                         |
          authoring plan + SQLProjection row-local check
                         |
             fit(fit data) + params join probes
                         |
          fitted SQL + params + UDFs + request schema
                         |
             compile() -> Confit build or refusal
                         |
                  infer_rows / infer_arrow
```

General `SQLTransform` supports composition and batch execution without the
row-local rule. Fitted `SQLProjection.transform` also executes in batch.
Only Confit-admitted projections compile for row-local serving.

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

The [system specification](docs/specs/README.md) describes the current system
by topic. Unsupported authoring syntax and its explicit alternatives are in
the [authoring contract](packages/sql-transform/docs/contract.md#8-unsupported-forms-and-explicit-alternatives).

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
