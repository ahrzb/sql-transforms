"""Run every differential test once per serving engine.

The diff tests call differential.check(), which compares ONE engine against the
DataFusion oracle. Rather than touch 80 call sites, parametrize the modules that
use the harness over the available backends and let the fixture point check() at
each in turn -- so "native" and "codegen" are each proven against the same oracle,
and appear as separate test IDs when one breaks.

autouse is load-bearing and NOT stylistic. A non-autouse fixture named by
`metafunc.fixturenames.append(...)` produces the parametrized IDs but is NEVER
INSTANTIATED -- measured 2026-07-17: the ID said "codegen" while the engine in
use was still "native", so the whole suite ran the native engine twice and reported a
green bar for an engine that never executed. autouse forces instantiation.
Modules that aren't parametrized have no request.param and fall through.
"""

from __future__ import annotations

# Rebuild the native extension if src/*.rs is newer than the built _interpreter,
# BEFORE `import differential` -> `import sql_transform` loads it (TASK-33).
from _native_guard import ensure_native_built

ensure_native_built()

import differential  # noqa: E402
import pytest  # noqa: E402

_HARNESS_MODULES = ("test_diff_", "test_differential", "test_backend_wiring")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if metafunc.module.__name__.startswith(_HARNESS_MODULES):
        metafunc.parametrize("_backend", list(differential.BACKENDS), indirect=True)


@pytest.fixture(autouse=True)
def _backend(request: pytest.FixtureRequest):
    param = getattr(request, "param", None)
    if param is None:  # not a harness module -- leave the default alone
        yield None
        return
    differential.set_backend(param)
    yield param
    differential.set_backend("native")


@pytest.fixture
def xfail_on_native(request):
    """Mark the current test xfail on the native backend only.

    For a residual case where the native engine still disagrees with the
    DataFusion oracle while codegen matches it. strict=True so that fixing the
    native engine turns the xpass into a failure -- the reminder to remove the
    marker.
    """

    def _mark(reason: str) -> None:
        if differential._backend == "native":
            request.applymarker(pytest.mark.xfail(reason=reason, strict=True))

    return _mark
