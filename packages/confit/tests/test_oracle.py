"""The oracle's one constructor, exercised on its own terms.

Every test builds its oracle through `confit.oracle.Oracle` rather than a bare
`duckdb.connect()`: the module exists so that the optimizer-off pragma, the
native-table load and the raw arrow answer come from a single place, and a
test that reached for `duckdb` directly would not be testing that.

The tests run under this directory's autouse fixture, which patches
`duckdb.connect` to hand back an optimizer-off connection. `Oracle` must be
independent of it -- it captures the real `duckdb.connect` at import -- so
nothing here may rely on the fixture being in force.
"""

from __future__ import annotations

import dataclasses

import duckdb
import pyarrow as pa
import pytest
from confit.oracle import Oracle, Trap

# `PRAGMA disable_optimizer` writes no setting `current_setting` can read
# back, so the probe is plan-shaped: the constant-true filter survives as a
# FILTER node only while the expression rewriter is off.
_PROBE = "EXPLAIN SELECT a FROM probe WHERE 1 = 1"


def _plan(oracle: Oracle) -> str:
    oracle.execute("CREATE TABLE IF NOT EXISTS probe(a INTEGER)")
    return oracle.execute(_PROBE).fetchall()[0][1]


def test_construction_applies_the_pragma():
    with Oracle() as oracle:
        assert "FILTER" in _plan(oracle)


def test_optimizer_on_flips_the_same_connection():
    with Oracle() as oracle:
        con = oracle.con
        assert "FILTER" in _plan(oracle)
        assert oracle.optimizer_on() is oracle
        assert oracle.con is con
        assert "FILTER" not in _plan(oracle)


def test_table_creates_and_inserts_rows():
    with Oracle() as oracle:
        rows = [(1, "x"), (2, None)]
        assert oracle.table("t", "a INTEGER, b VARCHAR", rows) is oracle
        assert oracle.answer("SELECT * FROM t ORDER BY a").to_pylist() == [
            {"a": 1, "b": "x"},
            {"a": 2, "b": None},
        ]


def test_table_keeps_the_declaration_including_not_null():
    with Oracle() as oracle:
        oracle.table("t", "a INTEGER NOT NULL, b VARCHAR")
        assert oracle.catalog() == [
            ("t", [("a", "INTEGER", False), ("b", "VARCHAR", True)])
        ]


def test_load_materializes_a_native_table_with_widths_intact():
    arrow = pa.table({"a": pa.array([1, 2], pa.int8()), "b": pa.array(["x", "y"])})
    with Oracle() as oracle:
        assert oracle.load("s", arrow) is oracle
        assert oracle.execute(
            "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'main'"
        ).fetchall() == [("s",)]
        # TINYINT, not INTEGER: the width survives the CTAS. Nullability does
        # not -- a CTAS column is always nullable.
        assert oracle.catalog() == [
            ("s", [("a", "TINYINT", True), ("b", "VARCHAR", True)])
        ]
        assert oracle.answer("SELECT a FROM s ORDER BY a").to_pylist() == [
            {"a": 1},
            {"a": 2},
        ]


def test_load_unregisters_its_alias():
    arrow = pa.table({"a": pa.array([1], pa.int64())})
    with Oracle() as oracle:
        oracle.load("s", arrow)
        # SHOW TABLES lists registered relations alongside real tables, so a
        # leftover alias would show up here.
        assert oracle.execute("SHOW TABLES").fetchall() == [("s",)]


def test_answer_returns_arrow_unnormalized():
    with Oracle() as oracle:
        answer = oracle.answer("SELECT 1 AS a, 2 AS a")
        assert isinstance(answer, pa.Table)
        assert answer.column_names == ["a", "a"]


def test_try_answer_returns_the_table_when_the_query_runs():
    with Oracle() as oracle:
        assert oracle.try_answer("SELECT 1 AS a").to_pylist() == [{"a": 1}]


def test_try_answer_returns_a_trap_when_the_query_fails():
    with Oracle() as oracle:
        trap = oracle.try_answer("SELECT nosuchcol")
        assert isinstance(trap, Trap)
        assert trap.kind == "BinderException"
        assert "nosuchcol" in trap.message
        assert str(trap) == f"BinderException: {trap.message}"


def test_trap_is_frozen():
    trap = Trap("BinderException", "boom")
    with pytest.raises(dataclasses.FrozenInstanceError):
        trap.kind = "other"


def test_error_is_duckdbs():
    assert Oracle.Error is duckdb.Error


def test_replay_setup_drops_and_retries_a_duplicate_create():
    with Oracle() as oracle:
        assert (
            oracle.replay_setup(
                [
                    "CREATE TABLE t(a INTEGER)",
                    "INSERT INTO t VALUES (1)",
                    'CREATE TABLE "t"(b VARCHAR)',
                ]
            )
            is oracle
        )
        # The re-create won, and it took the first table's rows with it.
        assert oracle.catalog() == [("t", [("b", "VARCHAR", True)])]
        assert oracle.answer("SELECT count(*) AS n FROM t").to_pylist() == [{"n": 0}]


def test_replay_setup_raises_on_any_other_catalog_error():
    with Oracle() as oracle:
        with pytest.raises(duckdb.CatalogException):
            oracle.replay_setup(["DROP TABLE nope"])


def test_catalog_shape_over_two_tables():
    with Oracle() as oracle:
        oracle.table("b", "x INTEGER NOT NULL")
        oracle.table("a", "y VARCHAR, z DOUBLE")
        assert dict(oracle.catalog()) == {
            "a": [("y", "VARCHAR", True), ("z", "DOUBLE", True)],
            "b": [("x", "INTEGER", False)],
        }


def test_connection_passthrough():
    with Oracle() as oracle:
        oracle.register("r", pa.table({"a": pa.array([7], pa.int64())}))
        assert oracle.execute("SELECT a FROM r").fetchall() == [(7,)]


def test_exit_closes_the_connection():
    oracle = Oracle()
    with oracle:
        pass
    with pytest.raises(duckdb.Error):
        oracle.execute("SELECT 1")
