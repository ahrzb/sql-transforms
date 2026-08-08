"""The gaps the typed-walk migration's mutation check exposed.

Five mutations survived the first pass. None was dead code and none was a
defect — each was real behaviour that nothing held down, so the walk could
have been rewritten to lose it and the suite would have stayed green. They
predate the migration; it is only that mutation-checking a refactor asks the
question a feature never does.

Every case here is a *usage*, not a mechanism: the query a person would write,
and the number of params it must cost. That is the shape a gate has to have —
one built from the diff passes for the same reason the diff does.
"""

import pyarrow as pa
import pytest

from sql_transform.model import SQLTransform
from sql_transform.model._ast import _parse
from sql_transform.model._nodes import CteMap, Opaque, is_query, is_ref

D = pa.table({"cat": ["a", "a", "b"], "price": [1.0, 3.0, 10.0]})


def approx(table, places=6):
    return [
        tuple(round(v, places) if isinstance(v, float) else v for v in row.values())
        for row in table.to_pylist()
    ]


# ------------------------------------------------- the predicates, on a tag
# we do not type


def test_a_query_node_we_do_not_type_is_still_a_query_node():
    """All three query tags are typed today, so this branch is unreachable on
    1.5.5 and no query can exercise it. It is not decoration: answering False
    here would make the walk treat a whole subquery as an expression, freeze it
    as a leaf, and never look inside for `__THIS__`. Asked of the predicate
    directly, since the only other way to ask is a DuckDB that does not exist
    yet."""
    future = Opaque(tag_name="LATERAL_JOIN_NODE", fields={"cte_map": CteMap(map=[])})
    assert is_query(future)
    assert not is_ref(future)


def test_a_table_ref_we_do_not_type_is_still_a_table_ref():
    future = Opaque(tag_name="PIVOT_REF", fields={"sample": None, "alias": "p"})
    assert is_ref(future)
    assert not is_query(future)


# ------------------------------------------------------------------ freezing


def test_a_frozen_subtree_carries_the_ctes_it_references():
    """`freeze` prepends the enclosing CTEs to the step it emits. Without them
    the step is a standalone statement with an unbound name, and it dies at
    fit in DuckDB's words rather than serving."""
    t = SQLTransform("""
        WITH lo AS (SELECT 2.0 AS floor)
        SELECT t.price - x.m AS z
        FROM __THIS__ t,
             (SELECT greatest(avg(f.price), max(lo.floor)) AS m FROM __FIT__ f, lo) x
    """)
    fitted = t.fit(D)
    assert set(fitted.params) == {"__param_0"}
    assert len(fitted.params["__param_0"]) == 1
    assert approx(fitted(D), 4) == [(-3.6667,), (-1.6667,), (5.3333,)]


def test_a_frozen_cte_stops_reading_fit_for_everything_after_it():
    """What a CTE reads is re-read *after* it is visited, not before.

    Before: `a` reads `__FIT__`, so the derived table referencing it looks
    `__FIT__`-only too and freezes a second time — a params table nothing
    needs. After: the frozen `a` reads nothing, so the derived table is
    ordinary and one params table does the job.

    The difference is invisible in the output and visible in `len(params)`,
    which is the whole point of reifying the closure.
    """
    t = SQLTransform("""
        WITH a AS (SELECT avg(price) AS m FROM __FIT__)
        SELECT t.price - b.m AS z FROM __THIS__ t, (SELECT m FROM a) b
    """)
    fitted = t.fit(D)
    assert set(fitted.params) == {"__param_a"}, "the derived table froze twice"
    assert len(fitted.params["__param_a"]) == 1
    assert approx(fitted(D), 4) == [(-3.6667,), (-1.6667,), (5.3333,)]


def test_a_recursive_cte_reading_fit_ships_the_training_set_and_says_so():
    """A recursive CTE cannot be hoisted — its self-reference is bound by the
    enclosing entry key — so `__FIT__` inside one is repointed at the training
    set instead. Left unrewritten, `__FIT__` is simply unbound at serve.

    `__param_fit` with `len(D)` rows is the honest report of what that costs.
    """
    t = SQLTransform("""
        WITH RECURSIVE r(n) AS (
            SELECT count(*) AS n FROM __FIT__
            UNION ALL
            SELECT n - 1 FROM r WHERE n > 0)
        SELECT t.price, (SELECT max(n) FROM r) AS k FROM __THIS__ t
    """)
    fitted = t.fit(D)
    assert set(fitted.params) == {"__param_fit"}
    assert len(fitted.params["__param_fit"]) == len(D)
    assert approx(fitted(D)) == [(1.0, 3), (3.0, 3), (10.0, 3)]


# ---------------------------------------------------------------- resolution


def test_a_member_call_inside_a_cte_body_is_spliced():
    """Resolution descends into CTE bodies and the result is kept. Dropped, the
    call stays in the text and DuckDB is asked for a table function named after
    a Python variable."""
    # Read out of the caller's frame by name, which is the mechanism on trial:
    # to the linter it is unused, and to the transform below it is everything.
    inner = SQLTransform(  # noqa: F841
        "SELECT t.cat, t.price - (SELECT avg(price) FROM __FIT__) AS d FROM __THIS__ t"
    )
    outer = SQLTransform("""
        WITH s AS (SELECT * FROM inner(__FIT__, __THIS__))
        SELECT s.cat, s.d * 2 AS dd FROM s
    """)
    assert "inner(" not in outer.sql, "the call survived into the text"
    fitted = outer.fit(D)
    assert approx(fitted(D), 4) == [("a", -7.3333), ("a", -3.3333), ("b", 10.6667)]


@pytest.mark.parametrize(
    "sql",
    [
        "WITH s AS (SELECT * FROM __FIT__) SELECT count(*) AS n FROM s",
        "WITH s AS (SELECT * FROM __THIS__) SELECT count(*) AS n FROM s",
    ],
)
def test_a_cte_body_is_walked_at_all(sql):
    """The cheapest possible statement of the same thing: a CTE body that
    mentions a parameter has to be seen, whichever parameter it is."""
    assert _parse(sql).statements[0].node is not None
    assert SQLTransform(sql).fit(D)(D).num_rows == 1
