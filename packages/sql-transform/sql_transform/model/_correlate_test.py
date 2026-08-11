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


def test_the_inequality_refusal_names_the_way_out():
    """The one refusal that is not a weird case says what to write instead.

    An inequality correlation has an exact hand-written form — aggregate
    __FIT__ to one row per lookup coordinate in a CTE, then join it — so this
    refusal is a redirect rather than a wall. Saying so is the difference
    between a short refusal list and a short *usable* one.
    """
    with pytest.raises(CorrelatedFit) as caught:
        SQLTransform(
            "SELECT t.cat, (SELECT avg(f.price) FROM __FIT__ f"
            " WHERE f.ts <= t.ts) AS m FROM __THIS__ t"
        )
    assert caught.value.reason == "not-an-equality"
    assert "CTE" in str(caught.value)
    assert "docs/decorrelation-unsupported.md" in str(caught.value)


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


# ------------------------------------------------- what the audit turned up
# Four ways to be accepted and answered wrong, plus one over-refusal. Each is
# the shortest shape that shows it.


@pytest.mark.parametrize("agg", ["entropy(f.i)", "approx_count_distinct(f.i)"])
def test_the_miss_value_is_duckdbs_own_not_the_aggregates_empty_input(agg):
    """The probe was ``<agg> FROM __FIT__ WHERE false``, which is the value the
    aggregate takes on an empty *input*. That is not what DuckDB gives for a
    correlated *miss*: it repairs the count bug for ``count`` alone and yields
    NULL for everything else, whatever the empty-input value would be. The two
    coincide for 65 of DuckDB's 68 aggregates and part company for three.

    P9 settles which one is right — DuckDB is the oracle, so the target is what
    it does, not what the aggregate means. The probe is a guaranteed correlated
    miss now, so it inherits both behaviours without knowing either.
    """
    fit = pa.table({"cat": ["a", "a", "b"], "i": [1, 2, 3]})
    both(
        f"SELECT t.cat, (SELECT {agg} FROM __FIT__ f WHERE f.cat = t.cat) AS m"
        " FROM __THIS__ t ORDER BY t.cat",
        fit,
        pa.table({"cat": ["a", "zz"]}),
    )


def test_the_miss_value_keeps_freezing_faithful_on_a_null_key():
    """The same defect with no external oracle. `=` rejects the NULL key, so a
    NULL serving key is always a miss — and ``run``, which does no freezing at
    all, disagreed with the frozen form on exactly that row."""
    data = pa.table({"cat": ["a", "a", None], "i": [1, 2, 3]})
    t = SQLTransform(
        "SELECT t.cat, (SELECT entropy(f.i) FROM __FIT__ f WHERE f.cat = t.cat) AS m"
        " FROM __THIS__ t ORDER BY t.cat NULLS LAST"
    )
    assert t.fit(data).transform(data).to_pylist() == run(t, data).to_pylist()


@pytest.mark.parametrize(
    "sql",
    [
        # The inner alias published to the parent, capturing a name the parent
        # already bound. `f` is a CTE outside and `f.price > 20` is a constant
        # filter on it; after the merge it silently filtered training rows.
        "WITH f AS (SELECT 999.0 AS price)"
        " SELECT t.cat, (SELECT avg(price) FROM (SELECT * FROM __FIT__ f"
        " WHERE f.cat = t.cat) WHERE f.price > 20) AS m"
        " FROM __THIS__ t, f ORDER BY t.cat",
        # The same line in the other direction: the outer alias pushed onto the
        # base, so `__FIT__.cat` inside no longer resolved and fit died on a
        # query the oracle answers.
        "SELECT t.cat, (SELECT avg(x.price) FROM"
        " (SELECT * FROM __FIT__ WHERE __FIT__.cat = t.cat) x) AS m"
        " FROM __THIS__ t ORDER BY t.cat",
    ],
)
def test_flattening_never_changes_what_the_base_relation_is_called(sql):
    """``_flatten`` merged with ``ref.alias or base.alias``, which renames the
    base in one direction and leaks the inner alias in the other. Both are a
    *different query*, not a refusal — one answered wrongly and one crashed.

    Merging is allowed only when the set of names the base answers to is
    unchanged; where it is not, the correlation stays where the author put it,
    which is outside the WHERE clause, and that already has a name.
    """
    with pytest.raises(CorrelatedFit) as caught:
        SQLTransform(sql)
    assert caught.value.reason == "outside-where"


def test_a_sample_on_the_table_reference_refuses_like_one_on_the_query():
    """``USING SAMPLE`` lands on the SELECT node and was checked; ``TABLESAMPLE``
    written after the table lands on the *ref* and was not, so fit froze one
    draw and shipped it. Three fits of the same relation gave three different
    models — exactly the harm the refusal exists to prevent."""
    for sql in (
        "SELECT t.cat, (SELECT avg(f.price) FROM __FIT__ f TABLESAMPLE bernoulli(10%)"
        " WHERE f.cat = t.cat) AS m FROM __THIS__ t",
        "SELECT t.cat, (SELECT avg(f.price) FROM __FIT__ f"
        " WHERE f.cat = t.cat USING SAMPLE 10%) AS m FROM __THIS__ t",
    ):
        with pytest.raises(CorrelatedFit) as caught:
            SQLTransform(sql)
        assert caught.value.reason == "sample", sql


def test_a_cte_inside_the_subquery_is_lifted_like_any_other_spelling():
    """The lookup was copied from the subquery and kept its ``cte_map``, which
    is dead — the lookup reads only the keyed table — but still named
    ``__FIT__``, so the walk saw an unfrozen reference and refused. The same
    query written as a derived table was accepted, which is the tell."""
    both(
        "SELECT t.cat, (WITH z AS (SELECT * FROM __FIT__)"
        " SELECT avg(z.price) FROM z WHERE z.cat = t.cat) AS m"
        " FROM __THIS__ t ORDER BY t.cat"
    )


def test_a_nested_column_that_shadows_the_outer_alias_refuses():
    """The one classification the AST cannot make. DuckDB binds ``t.cat`` to a
    nested column ``t`` of ``__FIT__`` in preference to the outer relation, so
    what looks like the flagship type-JA is field access and the subquery is
    not correlated at all. Lifted anyway, it built a keyed table on a
    correlation nobody wrote and served plausible numbers.

    Measured: STRUCT and MAP columns win, a plain column of the same name does
    not, and a STRUCT without the field is a binder error. So the check is on
    nestedness, and it is at fit because that is where the schema first exists.
    """
    fit = pa.table(
        {
            "cat": ["a", "a", "b"],
            "price": [10.0, 30.0, 5.0],
            "t": [{"cat": "b"}, {"cat": "b"}, {"cat": "b"}],
        }
    )
    t = SQLTransform(
        "SELECT t.cat, (SELECT avg(f.price) FROM __FIT__ f WHERE f.cat = t.cat) AS m"
        " FROM __THIS__ t ORDER BY 1"
    )
    with pytest.raises(CorrelatedFit) as caught:
        t.fit(fit)
    assert caught.value.reason == "shadowed-by-a-nested-column"


def test_a_plain_column_of_the_same_name_is_not_shadowing():
    """The check must not fire on a name that DuckDB binds outward anyway —
    only a nested column takes precedence."""
    fit = pa.table(
        {"cat": ["a", "a", "b"], "price": [10.0, 30.0, 5.0], "t": ["x", "x", "x"]}
    )
    both(
        "SELECT t.cat, (SELECT avg(f.price) FROM __FIT__ f WHERE f.cat = t.cat) AS m"
        " FROM __THIS__ t ORDER BY 1",
        fit,
        pa.table({"cat": ["a", "b", "zz"]}),
    )


def test_no_refusal_escapes_the_documented_list():
    """The gate walks ``REASONS`` to the page and so cannot see a reason that
    is not in ``REASONS``. ``CorrelatedFit``'s default reason was exactly that:
    reachable, unnamed, and undocumented."""
    import inspect

    from sql_transform.model import _errors

    signature = inspect.signature(_errors.CorrelatedFit.__init__)
    assert signature.parameters["reason"].default is inspect.Parameter.empty, (
        "a default reason is a refusal nobody has to name"
    )


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
