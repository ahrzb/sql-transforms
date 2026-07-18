"""Guard the two-backend wiring itself.

A parametrized ID is not evidence the backend switched: a non-autouse fixture
yields correct-looking IDs while never running, so the suite silently exercises
one engine twice and reports green. This test fails in exactly that case.
"""

from __future__ import annotations

import differential


def test_diff_backend_fixture_actually_switches_the_engine(request):
    expected = request.node.callspec.params["_backend"]
    actual = differential._backend
    assert actual == expected, (
        f"backend wiring is broken: test ID says {expected!r} but the engine in "
        f"use is {actual!r} — the suite is testing one engine twice"
    )
