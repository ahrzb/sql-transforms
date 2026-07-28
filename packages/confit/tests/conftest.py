"""Rebuild the native extension before anything imports `confit`.

The guard must run at conftest import time -- `import confit` eagerly loads the
native module, so a stale build would already be in the process by the time a
test body runs.
"""

from __future__ import annotations

from _native_guard import ensure_native_built

ensure_native_built()
