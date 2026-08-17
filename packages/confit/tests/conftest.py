"""Rebuild the native extension, and point every DuckDB connection at THE ORACLE.

The rebuild guard must run at conftest import time -- `import confit` eagerly
loads the native module, so a stale build would already be in the process by
the time a test body runs.

# THE ORACLE IS DuckDB WITH THE OPTIMIZER OFF

Decided 2026-08-17. Every `duckdb.connect()` in the test suite comes back with
`PRAGMA disable_optimizer` already applied, so a test that compares against
DuckDB is comparing against the oracle whether or not its author thought about
it.

Why here and not at the 42 call sites: the oracle is a property of the REPO,
not a per-test choice, and a call site can be forgotten. A new test that reaches
for DuckDB gets the oracle by construction.

Why optimizer-off is the oracle at all, measured the same day:

  * `PRAGMA disable_optimizer` == disabling all 33 named optimizers.
  * The BINDER is untouched, so output TYPES are identical (checked across
    narrow ints and decimals), constant folding still happens (`1 + 2` is
    int32 3), and bind-time constant errors still fire.
  * Execution-level LAZINESS is untouched: an untaken CASE arm, AND/OR
    short-circuit in both operand orders, and coalesce's later arguments all
    behave exactly as with the optimizer on.
  * What is removed is the plan rewriting -- statistics_propagation,
    expression_rewriter, filter pushdown/pullup, CSE, join reordering.

So the oracle is "the query as written, run by DuckDB's execution model, with
no plan rewriting" -- which is the same shape as this engine, and that is what
makes it matchable. The optimizer-on reading is NOT matchable in principle:
`statistics_propagation` reads a column's null statistic, so it answers the
same query over the same rows differently depending on the table's insert
history (proved in known_divergences/test_trap_elision.py).

A test that WANTS the optimizer -- because it documents what the optimizer
does -- says so in its own body:

    con = duckdb.connect()
    con.execute("PRAGMA enable_optimizer")   # this test is ABOUT the optimizer

which reads as the deliberate exception it is.

Done as an autouse FIXTURE rather than an import-time assignment on purpose:
`duckdb` is a shared module, so assigning to `duckdb.connect` at import time
leaks into every other package's tests for the rest of the session. It did --
sql_transform's single-evaluation tests count sklearn calls made through
DuckDB, and losing CSE doubled them. monkeypatch keeps the oracle inside this
directory and undoes it per test.
"""

from __future__ import annotations

import duckdb
import pytest
from _native_guard import ensure_native_built

ensure_native_built()


@pytest.fixture(autouse=True)
def _duckdb_is_the_oracle(monkeypatch):
    raw_connect = duckdb.connect

    def connect(*args, **kwargs):
        con = raw_connect(*args, **kwargs)
        con.execute("PRAGMA disable_optimizer")
        return con

    monkeypatch.setattr(duckdb, "connect", connect)
