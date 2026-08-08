"""The correlated ``__FIT__`` subquery, as a keyed table instead of a refusal.

Kim's ``NEST-JA`` (TODS 7(3), 1982) is the one rule in the lineage whose
temporary relation names the *inner* relation alone, which is the only kind we
can build: at fit there is no ``__THIS__`` to reach into. Everything published
since that repairs Kim does so by pulling the outer relation in.

The oracle here is DuckDB running the author's own text with both parameters
bound to *different* relations — ``run`` binds them to the same one, which
cannot see a serving key that fit never saw, and that is where every
interesting divergence lives.
"""

import duckdb
import pyarrow as pa
import pytest

from sql_transform.model import CorrelatedFit, SQLTransform, TransformError, run

# `a` has two prices, `b` has only NULLs, and the NULL category is reachable
# only through IS NOT DISTINCT FROM. `zz` in the serving relation is a key fit
# never saw.
F = pa.table(
    {
        "cat": ["a", "a", "a", "b", None],
        "price": [10.0, 30.0, 999.0, None, 7.0],
        "ok": [True, True, False, True, True],
    }
)
T = pa.table(
    {
        "cat": ["a", "a", "b", "zz", None],
        "region": ["EU", "US", "EU", "EU", "EU"],
    }
)


def oracle(sql: str, fit: pa.Table = F, this: pa.Table = T) -> list[dict]:
    """The author's text, run with the two parameters bound to two relations."""
    con = duckdb.connect()
    con.register("__FIT__", fit)
    con.register("__THIS__", this)
    return con.execute(sql).to_arrow_table().to_pylist()


def served(sql: str, fit: pa.Table = F, this: pa.Table = T) -> list[dict]:
    return SQLTransform(sql).fit(fit).transform(this).to_pylist()


def both(sql: str, fit: pa.Table = F, this: pa.Table = T) -> None:
    assert served(sql, fit, this) == oracle(sql, fit, this)


JA = (
    "SELECT t.cat, t.region,"
    " (SELECT avg(f.price) FROM __FIT__ f WHERE f.cat = t.cat) AS m"
    " FROM __THIS__ t ORDER BY t.cat, t.region"
)


# ------------------------------------------------------------------ admitted


def test_the_canonical_shape_is_no_longer_refused():
    """The shape the whole survey is about, and it used to raise at construction."""
    both(JA)


def test_an_unseen_key_takes_the_subquerys_own_empty_value():
    """P14 said *unseen group implies NULL*. That is a window rule. Here the
    answer is whatever the aggregate returns on no rows at all, which for
    ``count`` is 0 — a probe reads it off ``__FIT__``'s schema, never its rows."""
    both(
        "SELECT t.cat, (SELECT count(*) FROM __FIT__ f WHERE f.cat = t.cat) AS n"
        " FROM __THIS__ t ORDER BY t.cat"
    )


def test_two_spellings_of_one_count_keep_their_own_empty_values():
    """``count_if(x)`` is NULL on empty and ``count(x) FILTER (WHERE x)`` is 0.
    No hand-kept list of aggregates can be right; the probe does not need one."""
    both(
        "SELECT t.cat,"
        " (SELECT count_if(f.price > 0) FROM __FIT__ f WHERE f.cat = t.cat) AS a,"
        " (SELECT count(f.price) FILTER (WHERE f.price > 0) FROM __FIT__ f"
        "  WHERE f.cat = t.cat) AS b"
        " FROM __THIS__ t ORDER BY t.cat"
    )


def test_a_present_group_whose_value_is_null_is_not_a_miss():
    """``COALESCE(value, empty)`` gets this wrong — group ``b``'s prices are all
    NULL, so its honest answer is the sentinel the author wrote, not the
    empty-input one. Hit-ness is counted, never inferred from the value."""
    both(
        "SELECT t.cat, (SELECT CASE WHEN count(*) = 0 THEN -1 ELSE max(f.price) END"
        " FROM __FIT__ f WHERE f.cat = t.cat) AS m FROM __THIS__ t ORDER BY t.cat"
    )


def test_a_this_only_conjunct_becomes_a_guard_not_a_dropped_predicate():
    """The sharpest edge, and it has no counterpart in the literature: a plan
    rewrite has both relations in hand, so a conjunct over the outer one alone
    never needs moving. Dropped, it returns a plausible number."""
    both(
        "SELECT t.cat, t.region, (SELECT avg(f.price) FROM __FIT__ f"
        " WHERE f.cat = t.cat AND t.region = 'EU') AS m"
        " FROM __THIS__ t ORDER BY t.cat, t.region"
    )


def test_a_false_guard_means_the_group_is_empty_not_that_the_answer_is_null():
    """The same guard under ``count``. Filtering every row out is an empty
    input, and ``count`` of an empty input is 0 — guarding to NULL is wrong."""
    both(
        "SELECT t.cat, t.region, (SELECT count(*) FROM __FIT__ f"
        " WHERE f.cat = t.cat AND t.region = 'EU') AS n"
        " FROM __THIS__ t ORDER BY t.cat, t.region"
    )


def test_a_fit_only_conjunct_stays_in_the_params_query():
    both(
        "SELECT t.cat, (SELECT avg(f.price) FROM __FIT__ f"
        " WHERE f.cat = t.cat AND f.ok) AS m FROM __THIS__ t ORDER BY t.cat"
    )


def test_the_join_predicate_mirrors_the_operator_the_author_wrote():
    """``=`` never matches a NULL key, so that group is unreachable and is not
    shipped. ``IS NOT DISTINCT FROM`` does match it, so it is kept. P4 says
    *always INDF*, and P4 is a window rule — here it would invent an answer."""
    both(
        "SELECT t.cat, (SELECT avg(f.price) FROM __FIT__ f"
        " WHERE f.cat IS NOT DISTINCT FROM t.cat) AS m"
        " FROM __THIS__ t ORDER BY t.cat"
    )


def test_a_composite_key_may_mix_the_two_spellings():
    fit = pa.table(
        {
            "cat": ["a", "a", "a", "b"],
            "grp": [1, 1, None, 2],
            "price": [10.0, 30.0, 50.0, 5.0],
        }
    )
    this = pa.table({"cat": ["a", "a", "b", "zz"], "grp": [1, None, 2, 9]})
    both(
        "SELECT t.cat, t.grp, (SELECT avg(f.price) FROM __FIT__ f"
        " WHERE f.cat = t.cat AND f.grp IS NOT DISTINCT FROM t.grp) AS m"
        " FROM __THIS__ t ORDER BY t.cat, t.grp",
        fit,
        this,
    )


def test_the_enclosing_query_is_left_alone():
    """The rewrite stays inside the subquery, so the enclosing GROUP BY, the
    select list and the row count are untouched by construction."""
    both(
        "SELECT t.region, sum((SELECT avg(f.price) FROM __FIT__ f"
        " WHERE f.cat = t.cat)) AS m FROM __THIS__ t GROUP BY t.region"
        " ORDER BY t.region"
    )


def test_the_subquery_may_sit_in_a_where_clause():
    both(
        "SELECT t.cat FROM __THIS__ t WHERE t.cat >"
        " (SELECT min(f.cat) FROM __FIT__ f WHERE f.cat = t.cat)"
        " OR t.cat IS NULL ORDER BY t.cat"
    )


def test_the_correlation_may_reach_a_fit_only_relation_instead():
    """Correlating into a ``__FIT__``-side name used to fall through and ship
    the whole training set. The mechanics do not care which side it reaches."""
    both(
        "WITH lo AS (SELECT cat, min(price) AS floor FROM __FIT__ GROUP BY cat)"
        " SELECT t.cat, lo.floor, (SELECT avg(f.price) FROM __FIT__ f"
        " WHERE f.cat = lo.cat) AS m"
        " FROM __THIS__ t JOIN lo ON lo.cat = t.cat ORDER BY t.cat"
    )


def test_freezing_is_still_faithful():
    """The law: binding both parameters to one relation must equal fitting on
    it and transforming it."""
    t = SQLTransform(
        "SELECT t.cat, (SELECT avg(f.price) FROM __FIT__ f WHERE f.cat = t.cat) AS m"
        " FROM __THIS__ t ORDER BY t.cat"
    )
    assert t.fit(F).transform(F).to_pylist() == run(t, F).to_pylist()


# -------------------------------------------------------------- disclosure


def test_the_params_are_one_row_per_distinct_key_not_one_per_training_row():
    """The point of the whole exercise. 1000 rows, 3 categories, 3 rows kept
    (plus the one-row empty-input probe)."""
    big = pa.table({"cat": ["a", "b", "c"] * 400, "price": [1.0, 2.0, 3.0] * 400})
    fitted = SQLTransform(
        "SELECT t.cat, (SELECT avg(f.price) FROM __FIT__ f WHERE f.cat = t.cat) AS m"
        " FROM __THIS__ t"
    ).fit(big)
    assert sum(len(p) for p in fitted.params.values()) == 4
    assert len(big) == 1200


def test_a_group_the_join_can_never_reach_is_not_shipped():
    """``=`` cannot match a NULL key, so the NULL group is dead weight in the
    artifact — and dead weight made of training rows."""
    fitted = SQLTransform(
        "SELECT t.cat, (SELECT avg(f.price) FROM __FIT__ f WHERE f.cat = t.cat) AS m"
        " FROM __THIS__ t"
    ).fit(F)
    keyed = next(p for p in fitted.params.values() if len(p) > 1)
    assert None not in keyed.column("__key_0").to_pylist()


# ------------------------------------------------- nothing ships F entire


def test_a_bare_fit_beside_this_refuses_instead_of_shipping_the_training_set():
    """It used to answer, with every row and every column of ``__FIT__`` in the
    artifact. The refusal names the edit that makes the retention explicit."""
    with pytest.raises(TransformError) as caught:
        SQLTransform("SELECT t.price - f.price AS d FROM __THIS__ t, __FIT__ f")
    assert "training set" in str(caught.value)


def test_asking_for_the_training_set_in_so_many_words_still_works():
    """The escape hatch the refusal points at. Retention is fine; retention
    nobody wrote down is not."""
    fitted = SQLTransform(
        "SELECT t.price - f.price AS d FROM __THIS__ t, (SELECT price FROM __FIT__) f"
    ).fit(F)
    assert sum(len(p) for p in fitted.params.values()) == len(F)


def test_a_recursive_cte_over_fit_refuses_instead_of_shipping_it():
    with pytest.raises(TransformError, match="training set"):
        SQLTransform(
            "WITH RECURSIVE r(n) AS (SELECT count(*) FROM __FIT__"
            " UNION ALL SELECT n - 1 FROM r WHERE n > 0)"
            " SELECT t.cat, (SELECT max(n) FROM r) AS d FROM __THIS__ t"
        )


# ---------------------------------------------------------------- refusals
# Every one of these is named. `reason` is the metric surface: the set of
# reasons is the refusal list, and it is meant to stay short.


@pytest.mark.parametrize(
    "reason,sql",
    [
        (
            "not-an-equality",
            "SELECT t.cat, (SELECT avg(f.price) FROM __FIT__ f"
            " WHERE f.price <= t.price) AS m FROM __THIS__ t",
        ),
        (
            "not-an-equality",
            "SELECT t.cat, (SELECT avg(f.price) FROM __FIT__ f"
            " WHERE f.cat = t.cat OR f.ok) AS m FROM __THIS__ t",
        ),
        (
            "outside-where",
            "SELECT t.cat, (SELECT avg(f.price) - t.price FROM __FIT__ f"
            " WHERE f.cat = t.cat) AS m FROM __THIS__ t",
        ),
        (
            "not-aggregated",
            "SELECT t.cat, (SELECT f.price FROM __FIT__ f"
            " WHERE f.cat = t.cat) AS m FROM __THIS__ t",
        ),
        (
            "modifier",
            "SELECT t.cat, (SELECT f.price FROM __FIT__ f"
            " WHERE f.cat = t.cat ORDER BY f.price DESC LIMIT 1) AS m"
            " FROM __THIS__ t",
        ),
        (
            "grouping",
            "SELECT t.cat, (SELECT avg(f.price) FROM __FIT__ f"
            " WHERE f.cat = t.cat GROUP BY f.ok) AS m FROM __THIS__ t",
        ),
        (
            "window",
            "SELECT t.cat, (SELECT max(avg(f.price) OVER ()) FROM __FIT__ f"
            " WHERE f.cat = t.cat) AS m FROM __THIS__ t",
        ),
        (
            "not-a-scalar-subquery",
            "SELECT t.cat FROM __THIS__ t"
            " WHERE EXISTS (SELECT 1 FROM __FIT__ f WHERE f.cat = t.cat)",
        ),
    ],
)
def test_the_refusals_are_named_and_at_construction(reason, sql):
    with pytest.raises(CorrelatedFit) as caught:
        SQLTransform(sql)
    assert caught.value.reason == reason, str(caught.value)


def test_every_refusal_reason_is_documented():
    """The metric is only a metric if the list is written down. Each reason has
    an entry in the unsupported-cases page, with what it would take to lift."""
    import pathlib

    from sql_transform.model._correlate import REASONS

    page = (
        pathlib.Path(__file__).resolve().parents[4]
        / "docs"
        / "decorrelation-unsupported.md"
    ).read_text(encoding="utf-8")
    missing = [r for r in REASONS if f"`{r}`" not in page]
    assert not missing, f"undocumented refusal reasons: {missing}"


def test_a_cross_type_key_is_a_named_error_not_a_quiet_answer():
    """The one hazard the AST cannot see: ``'1'`` and ``'01'`` are two groups
    at fit and one key at serving, so the lookup matches twice. Loud."""
    fit = pa.table({"c": ["1", "01"], "p": [10.0, 30.0]})
    this = pa.table({"c": [1]})
    t = SQLTransform(
        "SELECT (SELECT avg(f.p) FROM __FIT__ f WHERE f.c = t.c) AS m FROM __THIS__ t"
    )
    with pytest.raises(duckdb.Error, match="compare the correlation key"):
        t.fit(fit).transform(this)
