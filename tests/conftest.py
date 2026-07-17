"""Run every differential test once per serving engine.

The diff tests call differential.check(), which compares ONE engine against the
DataFusion oracle. Rather than touch 80 call sites, parametrize the modules that
use the harness over the available backends and let the fixture point check() at
each in turn -- so "rust" and "codegen" are each proven against the same oracle,
and appear as separate test IDs when one breaks.

autouse is load-bearing and NOT stylistic. A non-autouse fixture named by
`metafunc.fixturenames.append(...)` produces the parametrized IDs but is NEVER
INSTANTIATED -- measured 2026-07-17: the ID said "codegen" while the engine in
use was still "rust", so the whole suite ran the rust engine twice and reported a
green bar for an engine that never executed. autouse forces instantiation.
Modules that aren't parametrized have no request.param and fall through.
"""

from __future__ import annotations

import differential
import pytest

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
    differential.set_backend("rust")
