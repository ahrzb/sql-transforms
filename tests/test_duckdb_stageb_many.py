"""Stage-B multiplicity vs the duckdb oracle (TASK-59), multiset parity.

Pins: docs/superpowers/specs/2026-07-28-stageB-multiplicity-pins.md —
DuckDB's join output ORDER is a hash-join accident, so comparison is
SORTED; the engine's own order (probe outer, insertion inner) is a
documented contract of its own.
"""

from __future__ import annotations

import duckdb
import pyarrow as pa
from pydantic import create_model

from sql_transform._interpreter import DuckDBInferFn

T = create_model("T", pid=(int | None, None))
ROWS = [{"pid": 1}, {"pid": 2}, {"pid": 3}, {"pid": None}]
DIM = pa.table({"id": [1, 2, 1, 2, 1], "v": ["a", "b", "c", "d", "e"]})


def _many_check(sql: str):
    """Engine (shape='many') vs DuckDB, sorted-row multiset."""
    fn = DuckDBInferFn(
        sql.replace("__THIS__", "__THIS__"),
        row_tables={"__THIS__": T},
        static_tables={"d": DIM},
        output="dict",
        shape="many",
    )
    got = [tuple(r.values()) for r in fn.infer({"__THIS__": [T(**r) for r in ROWS]})]

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (pid BIGINT)")
    for r in ROWS:
        con.execute("INSERT INTO __THIS__ VALUES (?)", [r["pid"]])
    con.register("__arrow_d", DIM)
    con.execute('CREATE TABLE d AS SELECT * FROM "__arrow_d"')
    want = con.execute(sql).fetchall()
    key = lambda t: tuple((x is None, x) for x in t)  # noqa: E731
    assert sorted(got, key=key) == sorted(want, key=key), f"{sql}\n{got}\n{want}"


def test_dup_key_fanout_vs_oracle():
    _many_check("SELECT pid, v FROM __THIS__ JOIN d ON pid = d.id")
    _many_check("SELECT pid, v FROM __THIS__ LEFT JOIN d ON pid = d.id")
    _many_check("SELECT pid, v FROM __THIS__ LEFT JOIN d ON pid = d.id AND d.v > 'b'")
    _many_check("SELECT pid, v FROM __THIS__ LEFT JOIN d ON pid = d.id WHERE v IS NULL")
    _many_check("SELECT upper(v) AS u FROM __THIS__ JOIN d ON pid = d.id WHERE pid > 1")


def test_cross_and_inequality_vs_oracle():
    _many_check("SELECT pid, id, v FROM __THIS__, d")
    _many_check("SELECT pid, id FROM __THIS__ JOIN d ON pid > d.id")
    _many_check("SELECT pid, id FROM __THIS__ LEFT JOIN d ON pid > d.id")
    _many_check("SELECT pid, id FROM __THIS__ LEFT JOIN d ON NULL = 2")
    _many_check("SELECT pid, id FROM __THIS__, d WHERE pid >= id AND v <> 'c'")


def test_engine_order_contract():
    # The engine's OWN documented deterministic order: probe rows in input
    # order, matches contiguous in build INSERTION order, null-extension
    # in place.
    fn = DuckDBInferFn(
        "SELECT pid, v FROM __THIS__ LEFT JOIN d ON pid = d.id",
        row_tables={"__THIS__": T},
        static_tables={"d": DIM},
        output="dict",
        shape="many",
    )
    got = [tuple(r.values()) for r in fn.infer({"__THIS__": [T(**r) for r in ROWS]})]
    assert got == [
        (1, "a"),
        (1, "c"),
        (1, "e"),
        (2, "b"),
        (2, "d"),
        (3, None),
        (None, None),
    ]
