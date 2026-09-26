"""Rebuild the native extension, and hand out THE ORACLE.

The rebuild guard must run at conftest import time -- `import confit` eagerly
loads the native module, so a stale build would already be in the process by
the time a test body runs.

Everything that compares against DuckDB goes through `confit.oracle`, which is
where the optimizer-off decision and the reasoning behind it live. The ban on
raw connections is enforced in test_oracle.py by reading the sources, which
also covers the tests a run never reaches.
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
