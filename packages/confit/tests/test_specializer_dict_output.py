"""The raw-dict output mode: output="dict" on DuckDBInferFn.

Opt-in only — the typed pydantic contract is the untouched default. Same
engine, same lanes; the marshaller skips model construction, returning
per-row dicts that agree with the typed mode field-for-field (the modes
share the dict the model would have been built from).
"""

from __future__ import annotations

import pytest
from confit import DuckDBInferFn
from pydantic import create_model
from test_duckdb_interpreter import duck_check, static

Row = create_model("Row", a=(int, ...), s=(str | None, None))
SQL = "SELECT a * 2 AS d, upper(coalesce(s, 'x')) AS u FROM __THIS__ WHERE a > 0"
ROWS = [
    {"a": 3, "s": "hi"},
    {"a": -1, "s": None},
    {"a": 5, "s": None},
]


def _fns():
    kw = {"row_tables": {"__THIS__": Row}, "static_tables": {}}
    return (
        DuckDBInferFn(SQL, **kw),
        DuckDBInferFn(SQL, **kw, output="dict"),
    )


def test_dict_mode_agrees_with_typed_mode():
    m, d = _fns()
    rows = [Row(**r) for r in ROWS]
    typed = [r.model_dump() for r in m.infer({"__THIS__": rows})]
    dicts = d.infer({"__THIS__": rows})
    assert typed == dicts
    assert all(type(r) is dict for r in dicts)
    assert m.output == "model" and d.output == "dict"


def test_dict_mode_with_join_and_nulls():
    dim = static(
        {"id": "int", "name": "str"},
        [{"id": 1, "name": "one"}, {"id": 3, "name": "three"}],
    )
    fn = DuckDBInferFn(
        "SELECT k, name FROM __THIS__ LEFT JOIN dim ON k = dim.id",
        row_tables={"__THIS__": create_model("K", k=(int, ...))},
        static_tables={"dim": dim},
        output="dict",
    )
    K = create_model("K", k=(int, ...))
    got = fn.infer({"__THIS__": [K(k=1), K(k=2)]})
    assert got == [{"k": 1, "name": "one"}, {"k": 2, "name": None}]


def test_dict_mode_returns_fresh_dicts_per_call():
    _, d = _fns()
    rows = [Row(a=1)]
    first = d.infer({"__THIS__": rows})
    first[0]["d"] = "mutated"
    again = d.infer({"__THIS__": rows})
    assert again == [{"d": 2, "u": "X"}]


def test_dict_mode_on_constant_engine():
    fn = DuckDBInferFn(
        "SELECT 1 AS one, 'x' AS s",
        row_tables={"__THIS__": Row},
        static_tables={},
        output="dict",
    )
    got = fn.infer({"__THIS__": []})
    assert all(type(r) is dict for r in got)
    # Mutating a returned dict must not leak into the next call.
    if got:
        got[0]["one"] = "mutated"
        assert fn.infer({"__THIS__": []}) != got


def test_dict_mode_guards():
    kw = {"row_tables": {"__THIS__": Row}, "static_tables": {}}
    with pytest.raises(ValueError, match="mutually exclusive"):
        DuckDBInferFn(SQL, **kw, output="dict", output_model=Row)
    with pytest.raises(ValueError, match="'model' or 'dict'"):
        DuckDBInferFn(SQL, **kw, output="bogus")


def test_dict_mode_oracle_parity():
    # Same duck_check oracle discipline: dict-mode dumps == duckdb rows.
    m, d = _fns()
    rows = [Row(**r) for r in ROWS]
    assert d.infer({"__THIS__": rows}) == [
        r.model_dump() for r in m.infer({"__THIS__": rows})
    ]
    duck_check(SQL, {"a": "int", "s": "str?"}, ROWS)
