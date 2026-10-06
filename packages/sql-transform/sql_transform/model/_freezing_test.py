"""Slice 1 gates: the two parameters, freezing, ``fit``/``transform``.

These cases exercise the current authoring contract's freezing and execution
rules (`packages/sql-transform/docs/contract.md`). The
ordered-frame gate is written first, deliberately: it pins a result that looks
like a bug and must not be "corrected".
"""

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from sql_transform.model import (
    CorrelatedFit,
    SQLTransform,
    TransformError,
    run,
)
from sql_transform.model._analysis import _reads
from sql_transform.model._ast import FIT, _serialize

D = pa.table(
    {
        "ts": [1, 2, 3],
        "price": [10.0, 20.0, 30.0],
        "cat": ["a", "a", "b"],
    }
)
D2 = pa.table(
    {
        "ts": [1, 2, 3],
        "price": [100.0, 200.0, 300.0],
        "cat": ["a", "a", "b"],
    }
)

Z_SQL = """
SELECT (t.price - s.a) / s.s AS z
FROM __THIS__ t, (SELECT avg(price) a, stddev_pop(price) s FROM __FIT__) s
"""


def rows(table: pa.Table) -> list[tuple]:
    return list(zip(*(c.to_pylist() for c in table.columns), strict=True))


def approx(table: pa.Table, places: int = 9) -> list[tuple]:
    return [
        tuple(round(v, places) if isinstance(v, float) else v for v in r)
        for r in rows(table)
    ]


def reads_fit(sql: str) -> bool:
    """Does the residual still *read* ``__FIT__``?

    A substring check would also trip on the alias a spliced member leaves
    behind (``FROM <arg> AS __FIT__``), which reads nothing.
    """
    return FIT in _reads(_serialize(sql)["statements"][0]["node"])


def z_ref(fit_t: pa.Table, this_t: pa.Table) -> pa.Table:
    """The opaque composite ``(F, T) -> R``. Shares no code with the SQL path."""
    a = pc.mean(fit_t["price"]).as_py()
    s = pc.stddev(fit_t["price"], ddof=0).as_py()
    return pa.table({"z": pc.divide(pc.subtract(this_t["price"], a), s)})


# --------------------------------------------------------------- permissiveness

ORDERED_SQL = """
SELECT t.ts, ma.m
FROM __THIS__ t
LEFT JOIN (SELECT ts, avg(price) OVER (ORDER BY ts ROWS 2 PRECEDING) AS m
           FROM __FIT__) ma
  ON t.ts = ma.ts
"""


def test_ordered_frame_over_fit_is_deliberately_legal():
    """DELIBERATE — do not "fix" this.

    An order-keyed frame over ``__FIT__`` freezes and joins by the order key.
    A seen key returns the *training* value, ignoring the serving row's own
    price; an unseen key returns NULL through the LEFT JOIN (P14, the one NULL
    story). Both look wrong and both are what the text says. The previous
    design refused this because freezing was implicit; now the author writes
    the join key, so refusing would overrule something they can see.
    """
    fitted = SQLTransform(ORDERED_SQL).fit(D)
    out = fitted(pa.table({"ts": [3, 9], "price": [999.0, 50.0]}))
    assert rows(out) == [(3, 20.0), (9, None)]


def test_this_side_aggregation_stays_live():
    """An aggregate over ``__THIS__`` means what DuckDB means: batch-dependent."""
    t = SQLTransform(
        "SELECT price / (SELECT avg(price) FROM __THIS__) AS z FROM __THIS__"
    )
    fitted = t.fit(D)
    assert fitted.params == {}
    assert approx(fitted(D)) == [(0.5,), (1.0,), (1.5,)]
    assert approx(fitted(pa.table({"price": [1.0, 3.0]}))) == [(0.5,), (1.5,)]


# ---------------------------------------------------------------- the oracle


def test_pair_equals_composite():
    """``z.fit(F).transform(T) == z_ref(F, T)`` — the composite sets the numbers."""
    fitted = SQLTransform(Z_SQL).fit(D)
    assert approx(fitted(D2)) == approx(z_ref(D, D2))


def test_fitted_is_callable_and_transform_is_the_same_thing():
    fitted = SQLTransform(Z_SQL).fit(D)
    assert approx(fitted(D2)) == approx(fitted.transform(D2))


def test_partial_application_spelling():
    """``fit`` is partial application; calling the transform is the same."""
    z = SQLTransform(Z_SQL)
    assert approx(z(D)(D2)) == approx(z.fit(D).transform(D2))


# ------------------------------------------------------------------- freezing

TWO_CTES_SQL = """
WITH a AS (SELECT avg(price) m FROM __FIT__),
     b AS (SELECT stddev_pop(price) s FROM __FIT__)
SELECT (t.price - a.m) / b.s AS z FROM __THIS__ t, a, b
"""


def test_freezing_is_complete():
    """Every maximal ``__FIT__``-only subtree is in params; none survives."""
    fitted = SQLTransform(TWO_CTES_SQL).fit(D)
    assert set(fitted.params) == {"__param_a", "__param_b"}
    assert not reads_fit(fitted.sql)


def test_freezing_is_faithful():
    """``t.fit(D).transform(D) == run(t, D)`` — the reference is a binding."""
    for sql in (Z_SQL, TWO_CTES_SQL, ORDERED_SQL):
        t = SQLTransform(sql)
        assert approx(t.fit(D).transform(D)) == approx(run(t, D)), sql


def test_freezing_is_observable():
    """Load-bearing: without this, faithfulness passes on a no-op fit.

    Fit on ``D``, serve ``D2``. Frozen parameters give D's mean; recomputing
    from the serving batch would give 1.0 for every row.
    """
    t = SQLTransform(
        "SELECT price / (SELECT avg(price) FROM __FIT__) AS z FROM __THIS__"
    )
    assert approx(t.fit(D)(D2), 4) == [(5.0,), (10.0,), (15.0,)]
    assert approx(t.fit(D2)(D2), 4) == [(0.5,), (1.0,), (1.5,)]


def test_freezing_is_deterministic():
    a = SQLTransform(Z_SQL).fit(D)
    b = SQLTransform(Z_SQL).fit(D)
    assert a.params.keys() == b.params.keys()
    assert all(a.params[k].equals(b.params[k]) for k in a.params)


def test_fit_ignores_this():
    """``fit`` reads only ``__FIT__``; no serving batch need exist."""
    fitted = SQLTransform(Z_SQL).fit(D)
    assert list(fitted.params) == ["__param_0"]


def test_statelessness_is_real():
    """A transform never naming ``__FIT__`` has empty params and a no-op fit."""
    t = SQLTransform("SELECT price * 2 AS z FROM __THIS__")
    assert t.fit(D).params == {}
    assert approx(t.fit(D)(D2)) == approx(t.fit(D2)(D2))


def test_params_are_measurable():
    """A well-behaved transform is O(1) in |D|; a retaining one reports |D|."""
    well_behaved = SQLTransform(Z_SQL).fit(D)
    assert sum(len(p) for p in well_behaved.params.values()) == 1

    # Retaining is allowed where the author wrote a query whose value *is*
    # those rows. `FROM __THIS__ t, __FIT__ f` is refused instead: same
    # artifact, but its size would be a fact about freezing rather than about
    # the text.
    retains = SQLTransform(
        "SELECT t.price - f.price AS d FROM __THIS__ t, (SELECT price FROM __FIT__) f"
    )
    fitted = retains.fit(D)
    assert sum(len(p) for p in fitted.params.values()) == len(D)


# ------------------------------------------------------------------- refusals


def test_this_correlated_fit_refuses_at_construction():
    """P7: refused at construction, naming what it cannot do. Not at fit, not
    at serve.

    The equality case is lifted into a keyed table now (`_correlate_test.py`);
    what is left refuses. Here the correlation is an inequality, so no
    ``GROUP BY`` reproduces its equivalence classes.
    """
    sql = (
        "SELECT (SELECT avg(price) FROM __FIT__ f WHERE f.price <= t.price) AS m "
        "FROM __THIS__ t"
    )
    with pytest.raises(CorrelatedFit, match="equalit") as caught:
        SQLTransform(sql)
    assert caught.value.reason == "not-an-equality"


def test_internal_correlation_is_allowed():
    """A LATERAL inside a *closed* ``__FIT__`` subtree evaluates once, as a whole."""
    sql = """
    SELECT t.cat, t.price / w.m AS z
    FROM __THIS__ t
    LEFT JOIN (SELECT g.cat, s.m
               FROM (SELECT DISTINCT cat FROM __FIT__) g,
                    LATERAL (SELECT avg(price) m FROM __FIT__ f
                             WHERE f.cat = g.cat) s) w
      ON t.cat = w.cat
    """
    fitted = SQLTransform(sql).fit(D)
    assert len(fitted.params) == 1
    assert not reads_fit(fitted.sql)
    assert approx(fitted(D), 4) == [("a", 0.6667), ("a", 1.3333), ("b", 1.0)]


def test_unparseable_sql_refuses_at_construction():
    with pytest.raises(TransformError):
        SQLTransform("SELECT FROM WHERE __THIS__")


def test_no_third_mode_for_the_slice_1_corpus():
    """C5: every transform serves or refuses by name — never both, never neither."""
    for sql in (Z_SQL, TWO_CTES_SQL, ORDERED_SQL):
        assert isinstance(SQLTransform(sql).fit(D)(D2), pa.Table)


# ------------------------------------------------- frozen picks, the fit DAG

CARRIER = "c AS (SELECT cat, avg(price) OVER (PARTITION BY cat) AS m FROM __FIT__)"
PICK_SQL = f"""
WITH {CARRIER},
     p AS (SELECT DISTINCT c.cat, c.m FROM c)
SELECT t.price - p.m AS z FROM __THIS__ t JOIN p ON t.cat = p.cat ORDER BY t.price
"""


def test_a_distinct_pick_of_a_frozen_cte_ships_alone():
    """The pick freezes after its carrier, and the carrier — every fit row —
    is neither shipped nor left in the residual."""
    t = SQLTransform(PICK_SQL)
    fitted = t.fit(D)
    assert set(fitted.params) == {"__param_p"}
    assert "__param_c" not in fitted.sql
    assert not reads_fit(fitted.sql)
    assert approx(fitted(D2), 4) == [(85.0,), (185.0,), (270.0,)]
    assert approx(fitted(D)) == approx(run(t, D))


def test_a_pick_scales_with_its_keys_not_the_fit_rows():
    big = pa.table({"cat": ["a", "b"] * 1500, "price": [float(i) for i in range(3000)]})
    fitted = SQLTransform(PICK_SQL).fit(big)
    assert len(fitted.params["__param_p"]) == 2


def test_a_carrier_lives_until_its_last_pick():
    """Two picks read one carrier. Given back after the first, the second would
    fail to bind; both answering is what proves the order."""
    t = SQLTransform("""
        WITH c AS (SELECT cat, avg(price) OVER (PARTITION BY cat) AS m,
                          max(price) OVER () AS hi FROM __FIT__),
             p AS (SELECT DISTINCT c.cat, c.m FROM c),
             q AS (SELECT DISTINCT c.hi FROM c)
        SELECT t.price - p.m AS z, t.price / q.hi AS r
        FROM __THIS__ t JOIN p ON t.cat = p.cat, q ORDER BY t.price
    """)
    fitted = t.fit(D)
    assert set(fitted.params) == {"__param_p", "__param_q"}
    assert (len(fitted.params["__param_p"]), len(fitted.params["__param_q"])) == (2, 1)
    assert approx(fitted(D2), 4) == [
        (85.0, 3.3333),
        (185.0, 6.6667),
        (270.0, 10.0),
    ]


def test_a_residual_that_reads_the_carrier_retains_it():
    """Retention the author wrote is kept: the pick still freezes, and the
    carrier ships because the residual itself reads it."""
    t = SQLTransform(f"""
        WITH {CARRIER},
             p AS (SELECT DISTINCT c.cat, c.m FROM c)
        SELECT t.price - p.m AS z, (SELECT count(*) FROM c) AS n
        FROM __THIS__ t JOIN p ON t.cat = p.cat ORDER BY t.price
    """)
    fitted = t.fit(D)
    assert set(fitted.params) == {"__param_c", "__param_p"}
    assert len(fitted.params["__param_c"]) == len(D)
    assert approx(fitted(D2), 4) == [(85.0, 3), (185.0, 3), (270.0, 3)]


@pytest.mark.parametrize(
    "pick",
    [
        "SELECT DISTINCT cat, m FROM c",  # unqualified
        "SELECT DISTINCT c.cat, c.m + 0 AS m FROM c",  # an expression
        "SELECT DISTINCT c.cat, c.m FROM c WHERE c.m > 0",  # a clause
        "SELECT DISTINCT ON (c.cat) c.cat, c.m FROM c",
        "SELECT DISTINCT c.cat, c.m FROM c ORDER BY c.cat LIMIT 5",
        "SELECT c.cat, max(c.m) AS m FROM c GROUP BY c.cat",  # not DISTINCT
    ],
)
def test_anything_but_the_flat_pick_stays_live(pick):
    """Over the same frozen carrier, every near miss is ordinary SQL over a
    parameter: it stays in the residual, and the carrier ships."""
    t = SQLTransform(f"""
        WITH {CARRIER}, p AS ({pick})
        SELECT t.price - p.m AS z FROM __THIS__ t JOIN p ON t.cat = p.cat
        ORDER BY t.price
    """)
    fitted = t.fit(D)
    assert set(fitted.params) == {"__param_c"}
    assert approx(fitted(D2), 4) == [(85.0,), (185.0,), (270.0,)]


def test_a_pick_of_a_live_cte_stays_live():
    """A source reading ``__THIS__`` is not a fit-step parameter, so its pick
    answers per batch."""
    t = SQLTransform("""
        WITH c AS (SELECT cat FROM __THIS__), p AS (SELECT DISTINCT c.cat FROM c)
        SELECT count(*) AS n FROM p
    """)
    fitted = t.fit(D)
    assert fitted.params == {}
    assert rows(fitted(D)) == [(2,)]
    assert rows(fitted(pa.table({"cat": ["x"]}))) == [(1,)]


def test_a_dead_cte_is_neither_fit_nor_shipped():
    t = SQLTransform(
        "WITH unused AS (SELECT avg(price) AS m FROM __FIT__) "
        "SELECT price * 2 AS z FROM __THIS__"
    )
    fitted = t.fit(D)
    assert fitted.params == {}
    assert fitted(D2).to_pylist() == [{"z": 200.0}, {"z": 400.0}, {"z": 600.0}]
