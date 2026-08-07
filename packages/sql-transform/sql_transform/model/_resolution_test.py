"""TASK-71: the walk must resolve names, not compare strings.

Four failures, one cause. Every one is loud, so C5 holds — but P7 does not:
three surfaced as raw DuckDB errors at fit, on queries ``run`` executes fine.

The general gate is the last test in this module: nothing ``run`` accepts may
die at ``fit`` with an error that is not ours. Each specific case below is a
row of that.
"""

import duckdb
import pyarrow as pa
import pytest

from sql_transform.model import CorrelatedFit, SQLTransform, TransformError, run

D = pa.table({"grp": ["a", "a", "b"], "price": [1.0, 3.0, 100.0]})
T = pa.table({"grp": ["a", "b"], "price": [7.0, 9.0]})


# ------------------------------------------- _reads must follow a CTE (F6)


def test_a_subtree_reading_this_through_a_cte_is_not_frozen():
    """``_reads`` scanned for literal ``BASE_TABLE`` nodes, so a CTE reference
    hid ``__THIS__`` and the subtree was frozen — then died at fit, because
    ``__THIS__`` is not bound when parameters are computed."""
    t = SQLTransform(
        "WITH w AS (SELECT * FROM __THIS__) "
        "SELECT t.grp, (SELECT count(*) FROM w, __FIT__) AS c FROM __THIS__ t"
    )
    assert t.fit(D).transform(D).to_pylist() == run(t, D).to_pylist()


def test_a_cte_reading_only_fit_still_freezes():
    """The fix must not cost the freezing it was meant to protect."""
    t = SQLTransform(
        "WITH w AS (SELECT avg(price) m FROM __FIT__) "
        "SELECT t.price / w.m AS z FROM __THIS__ t, w"
    )
    fitted = t.fit(D)
    assert "__FIT__" not in fitted.sql
    assert fitted.transform(D).to_pylist() == run(t, D).to_pylist()


def test_a_chain_of_ctes_propagates_what_it_reads():
    t = SQLTransform(
        "WITH a AS (SELECT * FROM __THIS__), b AS (SELECT * FROM a) "
        "SELECT t.grp, (SELECT count(*) FROM b, __FIT__) AS c FROM __THIS__ t"
    )
    assert t.fit(D).transform(D).to_pylist() == run(t, D).to_pylist()


# ------------------------------- an unqualified correlated reference (F7)


def test_an_unqualified_correlated_fit_reference_refuses_by_name():
    """``_correlation`` only inspected qualified references, so this was
    frozen and died with a raw ``BinderException``.

    Whether an unqualified name resolves inward or outward cannot be known
    before the data exists — ``__FIT__`` has no schema at construction — so
    unlike its qualified sibling this one is caught at fit. It is *named*,
    which is the part that matters.
    """
    # `cat` exists only in the outer relation, so DuckDB resolves it outward.
    # Had `__FIT__` carried a `cat` too it would bind inward and be correct —
    # which is exactly why this cannot be decided without the data.
    t = SQLTransform(
        "SELECT t.cat, (SELECT avg(f.price) FROM __FIT__ f WHERE f.grp = cat) AS m "
        "FROM (SELECT grp AS cat, price FROM __THIS__) t"
    )
    with pytest.raises(TransformError) as caught:
        t.fit(D)
    assert not isinstance(caught.value, duckdb.Error)
    assert "cat" in str(caught.value)


def test_an_unqualified_name_that_binds_inward_is_left_alone():
    """The same shape, except ``__FIT__`` has the column. DuckDB binds it
    locally, nothing is correlated, and the freeze is correct."""
    t = SQLTransform(
        "SELECT t.grp, (SELECT avg(f.price) FROM __FIT__ f WHERE f.grp = grp) AS m "
        "FROM __THIS__ t"
    )
    assert t.fit(D).transform(D).to_pylist() == run(t, D).to_pylist()


def test_the_qualified_form_still_refuses_at_construction():
    """The case we *can* see without data keeps its construction-time refusal."""
    with pytest.raises(CorrelatedFit):
        SQLTransform(
            "SELECT (SELECT avg(price) FROM __FIT__ f WHERE f.grp = t.grp) AS m "
            "FROM __THIS__ t"
        )


def test_an_ordinary_unqualified_column_is_untouched():
    """The canonical example in the guide is unqualified. Refusing on the mere
    presence of an unqualified name would refuse almost everything."""
    t = SQLTransform(
        "SELECT t.price / s.m AS z FROM __THIS__ t, "
        "(SELECT avg(price) m FROM __FIT__) s"
    )
    assert t.fit(D).transform(D).to_pylist() == run(t, D).to_pylist()


# ------------------------------------------------- a recursive CTE (F13)


def test_a_recursive_cte_reading_fit_agrees_with_run():
    """``freeze`` hoisted the body into a standalone statement, but a
    recursive CTE's self-reference is bound by the enclosing entry key, not by
    anything inside the body."""
    t = SQLTransform(
        "WITH RECURSIVE c(i) AS ("
        "  SELECT 1 UNION ALL"
        "  SELECT i+1 FROM c WHERE i < (SELECT count(*) FROM __FIT__)"
        ") SELECT t.grp, (SELECT count(*) FROM c) AS n FROM __THIS__ t"
    )
    assert t.fit(D).transform(D).to_pylist() == run(t, D).to_pylist()


def test_a_plain_cte_named_like_its_own_body_still_freezes():
    t = SQLTransform(
        "WITH c AS (SELECT count(*) n FROM __FIT__) "
        "SELECT t.grp, c.n FROM __THIS__ t, c"
    )
    assert t.fit(D).transform(D).to_pylist() == run(t, D).to_pylist()


# ------------------------------------------------- case-insensitivity (F14)


@pytest.mark.parametrize(
    "defined,used", [("Sales", "sales"), ("sales", "SALES"), ("SaLeS", "sAlEs")]
)
def test_cte_names_match_the_way_duckdb_matches_them(defined, used):
    """``ctes`` was a set of exact strings; DuckDB's binder is
    case-insensitive, so valid SQL was refused as an unknown free name."""
    t = SQLTransform(
        f"WITH {defined} AS (SELECT avg(price) m FROM __FIT__) "
        f"SELECT t.price / {used}.m AS z FROM __THIS__ t, {used}"
    )
    assert t.fit(D).transform(D).to_pylist() == run(t, D).to_pylist()


# ----------------------------------------------------------- the general gate


ACCEPTED = [
    "WITH w AS (SELECT * FROM __THIS__) "
    "SELECT t.grp, (SELECT count(*) FROM w, __FIT__) AS c FROM __THIS__ t",
    "WITH a AS (SELECT * FROM __THIS__), b AS (SELECT * FROM a) "
    "SELECT t.grp, (SELECT count(*) FROM b, __FIT__) AS c FROM __THIS__ t",
    "WITH Up AS (SELECT avg(price) m FROM __FIT__) "
    "SELECT t.price / up.m AS z FROM __THIS__ t, up",
    "WITH RECURSIVE c(i) AS ("
    "  SELECT 1 UNION ALL"
    "  SELECT i+1 FROM c WHERE i < (SELECT count(*) FROM __FIT__)"
    ") SELECT t.grp, (SELECT count(*) FROM c) AS n FROM __THIS__ t",
    "SELECT t.price / s.m AS z FROM __THIS__ t, (SELECT avg(price) m FROM __FIT__) s",
]


@pytest.mark.parametrize("sql", ACCEPTED)
def test_nothing_run_accepts_dies_at_fit_with_someone_elses_error(sql):
    """The property the four specific gates are instances of.

    A construct either serves or refuses *by our name*. A raw
    ``CatalogException`` or ``BinderException`` escaping from ``fit``, on a
    query ``run`` executes happily, is the model failing to know its own mind.
    """
    t = SQLTransform(sql)
    expected = run(t, D).to_pylist()
    try:
        assert t.fit(D).transform(D).to_pylist() == expected
    except TransformError as refusal:
        assert not isinstance(refusal, duckdb.Error)
