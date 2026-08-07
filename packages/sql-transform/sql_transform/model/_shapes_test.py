"""Does the AST walk survive SQL shapes the design examples never used?

Nothing here parses SQL. DuckDB's ``json_serialize_sql`` produces the tree and
``json_deserialize_sql`` prints it back; in between the walk swaps nodes in an
untyped JSON dict. That middle step rests on two structural facts, and these
tests are what keeps them facts:

- a **query node** is exactly a dict carrying ``cte_map``
- a **TableRef** is exactly a dict carrying ``sample`` and no ``cte_map``

Both are internal DuckDB details, not a documented API. So the last test
replays the corpus mined from DuckDB's own window suite and requires the
answer to be identical to what DuckDB gives for the untouched text. That is
the one that would notice a serialization format change.
"""

import duckdb
import pyarrow as pa
import pytest

from sql_transform.model import SQLTransform, TransformError, run
from sql_transform.model._transform import _QUERY, _serialize, _under

D = pa.table({"cat": ["a", "a", "b"], "price": [1.0, 3.0, 10.0]})


def shapes(sql: str) -> tuple[set[str], set[str]]:
    """(TableRef types, query-node types) as the two discriminators see them."""
    doc = _serialize(sql)
    refs, queries = set(), set()
    for _, _, v in _under(doc, deep=True):
        if _QUERY in v:
            queries.add(v.get("type"))
        elif "sample" in v:
            refs.add(v.get("type"))
    return refs, queries


def test_the_two_discriminators_hold():
    """Every dict with ``sample`` is a TableRef; every one with ``cte_map`` is
    a query node. Measured, not assumed — a counterexample silently breaks the
    walk rather than raising."""
    refs, queries = shapes("""
        WITH a AS (SELECT 1 x)
        SELECT * FROM (VALUES (1), (2)) v(y), __THIS__ t, a,
               LATERAL (SELECT t.price * 2 z) s, range(2)
        UNION ALL SELECT 9, 9, 9, 9, 9, 9
    """)
    assert refs == {
        "BASE_TABLE",
        "EMPTY",
        "EXPRESSION_LIST",
        "JOIN",
        "SUBQUERY",
        "TABLE_FUNCTION",
    }
    assert queries == {"SELECT_NODE", "SET_OPERATION_NODE"}


def test_recursive_cte_sees_its_own_name():
    """A RECURSIVE CTE is in scope inside its own body.

    A plain CTE is not — there the same name means whatever the caller's frame
    binds — so the two cannot share one rule, and the inner node type is what
    tells them apart.
    """
    t = SQLTransform("""
        WITH RECURSIVE c(i) AS (
            SELECT 1 UNION ALL SELECT i + 1 FROM c WHERE i < 3
        )
        SELECT sum(i) AS s FROM c, (SELECT 1 FROM __THIS__ LIMIT 1)
    """)
    assert run(t, D).to_pylist() == [{"s": 6}]


def test_a_plain_cte_does_not_shadow_its_own_body():
    """`WITH c AS (SELECT * FROM c)` reads the *outer* c, so it still has to
    resolve from the caller's frame."""
    c = pa.table({"i": [7]})
    assert c is not None
    t = SQLTransform("WITH c AS (SELECT * FROM c) SELECT i FROM c, __THIS__ LIMIT 1")
    assert run(t, D).to_pylist() == [{"i": 7}]


def test_values_lists_and_set_operations_survive():
    t = SQLTransform(
        "SELECT price FROM __FIT__ UNION ALL SELECT y FROM (VALUES (99.0)) v(y)"
    )
    assert sorted(r["price"] for r in run(t, D).to_pylist()) == [1.0, 3.0, 10.0, 99.0]


def test_a_builtin_table_function_is_not_a_member_call():
    """``range`` is DuckDB's, so it is never looked up in the caller's frame."""
    t = SQLTransform("SELECT count(*) AS n FROM range(3), __THIS__")
    assert run(t, D).to_pylist() == [{"n": 9}]


def test_a_non_select_statement_refuses_by_name():
    """DuckDB only serializes SELECT, so PIVOT refuses at construction (P7)
    rather than reaching fit."""
    with pytest.raises(TransformError, match="SELECT"):
        SQLTransform("PIVOT __THIS__ ON cat USING sum(price)")


def test_the_walk_never_changes_duckdbs_answer():
    """The corpus replay: serialize, walk, print back, and require the result
    to equal what DuckDB computes for the text it was handed.

    This is the gate that would catch DuckDB changing its serialization
    format, which is an internal detail with no stability promise.
    """
    from sql_transform._corpus_test import (  # noqa: PLC0415
        CURATED_MARGINALIZED,
        CURATED_REFUSED,
        EMPSALARY,
        MINED,
    )

    con = duckdb.connect()
    con.register("__THIS__", EMPSALARY)
    con.register("__FIT__", EMPSALARY)

    def rows(table):
        # Order-insensitive: without a top-level ORDER BY, two executions of
        # the same text need not agree on row order.
        return sorted(repr(tuple(r.values())) for r in table.to_pylist())

    checked = 0
    for sql in dict.fromkeys(MINED + CURATED_MARGINALIZED + CURATED_REFUSED):
        try:
            expected = rows(con.execute(sql).to_arrow_table())
        except duckdb.Error:
            continue  # not runnable against empsalary; not the walk's business
        transform = SQLTransform(sql)
        assert rows(run(transform, EMPSALARY)) == expected, sql
        assert rows(transform.fit(EMPSALARY).transform(EMPSALARY)) == expected, sql
        checked += 1

    assert checked >= 70, f"corpus shrank to {checked}; the gate is losing teeth"
