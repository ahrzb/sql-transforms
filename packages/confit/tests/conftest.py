"""Rebuild the native extension, and hand out THE ORACLE.

The rebuild guard must run at conftest import time -- `import confit` eagerly
loads the native module, so a stale build would already be in the process by
the time a test body runs.

Everything that compares against DuckDB goes through `confit.oracle`, which is
where the optimizer-off decision and the reasoning behind it live. The ban on
raw connections is enforced in test_oracle.py, by reading the sources, and
NOT by patching `duckdb.connect` here, because the door is shared: the engine
folds a static-tables-only query at build time by handing it to DuckDB
(src/duckdb/mod.rs, `eval_static_only`), and it reaches for the same module
attribute. Measured: a Python frame cannot tell the engine's call from a
test's -- both arrive with the calling test's frame and differ only in line
number -- so a patch here has two possible effects, and neither is wanted. It
either refuses the engine's own fold (every constant-emitter test fails), or,
as the pragma-applying version of this fixture did, it silently folds those
queries with the optimizer OFF while production folds them with it ON.
"""

from __future__ import annotations

import pytest
from _native_guard import ensure_native_built
from confit.oracle import Oracle

ensure_native_built()


@pytest.fixture
def oracle():
    """The oracle, one per test, closed at teardown."""
    with Oracle() as o:
        yield o
