"""Guard the codegen engine's deferred surface.

Skips are how the harness handles what codegen defers (containers/UNNEST). That
is fine only while the skip set is exactly the deferred surface and nothing has
quietly fallen out of coverage -- this test is what makes that true.
"""

from __future__ import annotations

import pytest
from differential import _run_codegen, rows, static

from sql_transform._codegen import UnsupportedInCodegen

# Every committed-surface shape. None of these may raise UnsupportedInCodegen.
_COMMITTED = [
    ("SELECT a AS x FROM t", {"t": rows({"a": "int"}, [{"a": 1}])}),
    ("SELECT a + 1 AS x FROM t", {"t": rows({"a": "int"}, [{"a": 1}])}),
    ("SELECT -a AS x FROM t", {"t": rows({"a": "int"}, [{"a": 5}])}),
    ("SELECT a || '!' AS x FROM t", {"t": rows({"a": "str"}, [{"a": "hi"}])}),
    (
        "SELECT a / b AS x FROM t",
        {"t": rows({"a": "int", "b": "int"}, [{"a": 7, "b": 2}])},
    ),
    (
        "SELECT a % b AS x FROM t",
        {"t": rows({"a": "int", "b": "int"}, [{"a": 7, "b": 2}])},
    ),
    ("SELECT NOT (a > 1) AS x FROM t", {"t": rows({"a": "int"}, [{"a": 1}])}),
    ("SELECT a > 1 AND a < 5 AS x FROM t", {"t": rows({"a": "int"}, [{"a": 1}])}),
    ("SELECT UPPER(s) AS x FROM t", {"t": rows({"s": "str"}, [{"s": "a"}])}),
    ("SELECT LOWER(s) AS x FROM t", {"t": rows({"s": "str"}, [{"s": "A"}])}),
    ("SELECT TRIM(s) AS x FROM t", {"t": rows({"s": "str"}, [{"s": " a "}])}),
    ("SELECT SUBSTR(s, 1, 2) AS x FROM t", {"t": rows({"s": "str"}, [{"s": "abc"}])}),
    ("SELECT CONCAT(s, s) AS x FROM t", {"t": rows({"s": "str"}, [{"s": "a"}])}),
    ("SELECT ABS(a) AS x FROM t", {"t": rows({"a": "int"}, [{"a": -1}])}),
    ("SELECT ROUND(f) AS x FROM t", {"t": rows({"f": "float"}, [{"f": 1.5}])}),
    ("SELECT COALESCE(a, 0) AS x FROM t", {"t": rows({"a": "int?"}, [{"a": None}])}),
    # COALESCE(a, a) -- repeated args: the per-function arg extractor must NOT
    # de-dup equal sub-expressions down to arity 1 (a sqlglot-shape trap).
    ("SELECT COALESCE(a, a) AS x FROM t", {"t": rows({"a": "int?"}, [{"a": None}])}),
    # COALESCE(int, float) types as the numeric supertype (float), so the float
    # result is not pydantic-coerced back to int (same bug class as ROUND).
    (
        "SELECT COALESCE(a, b) AS x FROM t",
        {"t": rows({"a": "int?", "b": "float"}, [{"a": 3, "b": 2.5}])},
    ),
    ("SELECT NULLIF(a, 1) AS x FROM t", {"t": rows({"a": "int"}, [{"a": 1}])}),
    (
        "SELECT CASE WHEN a > 0 THEN 1 ELSE 0 END AS x FROM t",
        {"t": rows({"a": "int"}, [{"a": 1}])},
    ),
    (
        "SELECT CASE a WHEN 1 THEN 'one' ELSE 'other' END AS x FROM t",
        {"t": rows({"a": "int"}, [{"a": 1}])},
    ),
    ("SELECT CAST(a AS VARCHAR) AS x FROM t", {"t": rows({"a": "int"}, [{"a": 1}])}),
    ("SELECT CAST(s AS BIGINT) AS x FROM t", {"t": rows({"s": "str"}, [{"s": "1"}])}),
    ("SELECT CAST(a AS DOUBLE) AS x FROM t", {"t": rows({"a": "int"}, [{"a": 1}])}),
    ("SELECT CAST(a AS BOOLEAN) AS x FROM t", {"t": rows({"a": "int"}, [{"a": 1}])}),
    ("SELECT a AS x FROM t WHERE a > 0", {"t": rows({"a": "int"}, [{"a": 1}])}),
    ("SELECT z.a AS x FROM t AS z", {"t": rows({"a": "int"}, [{"a": 1}])}),
    (
        "SELECT a AS x, b AS y FROM t CROSS JOIN u",
        {"t": rows({"a": "int"}, [{"a": 1}]), "u": rows({"b": "int"}, [{"b": 2}])},
    ),
    (
        "SELECT a AS x FROM t JOIN u ON t.k = u.k",
        {
            "t": rows({"k": "int", "a": "int"}, [{"k": 1, "a": 1}]),
            "u": rows({"k": "int"}, [{"k": 1}]),
        },
    ),
    (
        "SELECT v AS x FROM t JOIN s ON t.k = s.k",
        {
            "t": rows({"k": "int"}, [{"k": 1}]),
            "s": static({"k": "int", "v": "float"}, [{"k": 1, "v": 1.0}]),
        },
    ),
    (
        "SELECT v AS x FROM t LEFT JOIN s ON t.k = s.k",
        {
            "t": rows({"k": "int"}, [{"k": 9}]),
            "s": static({"k": "int", "v": "float"}, [{"k": 1, "v": 1.0}]),
        },
    ),
    ("SELECT s AS x FROM t", {"t": rows({"s": "struct{x:int}"}, [{"s": {"x": 1}}])}),
    ("SELECT l AS x FROM t", {"t": rows({"l": "list[int]"}, [{"l": [1]}])}),
    (
        "SELECT named_struct('a', x, 'b', y) AS s FROM t",
        {"t": rows({"x": "int", "y": "int"}, [{"x": 1, "y": 2}])},
    ),
    (
        "SELECT array(x, y) AS l FROM t",
        {"t": rows({"x": "int", "y": "int"}, [{"x": 1, "y": 2}])},
    ),
    ("SELECT s.x AS v FROM t", {"t": rows({"s": "struct{x:int}"}, [{"s": {"x": 1}}])}),
    (
        "SELECT (s = s) AS x FROM t",
        {"t": rows({"s": "struct{x:int}"}, [{"s": {"x": 1}}])},
    ),
]


@pytest.mark.parametrize("query, tables", _COMMITTED, ids=lambda v: None)
def test_committed_surface_is_never_deferred(query, tables):
    """If this raises, codegen has silently dropped committed surface -- which
    the differential harness would otherwise report as a harmless skip."""
    try:
        _run_codegen(query, tables)
    except UnsupportedInCodegen as e:
        pytest.fail(f"committed surface must not be deferred: {query!r} raised {e}")


_DEFERRED = [
    ("SELECT unnest(l) AS x FROM t", {"t": rows({"l": "list[int]"}, [{"l": [1]}])}),
    # Equality is the ONLY op defined on containers; a struct in any other op
    # (here dpipe) must still defer, not fall through to the scalar STR/arith
    # path and render "<object>" -- pins the container-operand guard.
    (
        "SELECT s || 'x' AS r FROM t",
        {"t": rows({"s": "struct{x:int}"}, [{"s": {"x": 1}}])},
    ),
]


@pytest.mark.parametrize("query, tables", _DEFERRED, ids=lambda v: None)
def test_deferred_surface_raises_rather_than_answering_wrongly(query, tables):
    with pytest.raises(UnsupportedInCodegen):
        _run_codegen(query, tables)
