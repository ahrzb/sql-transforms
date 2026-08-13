"""Ordered fits — fit/transform-split slice 4.

An order-sensitive transformer declares it (``OrderSensitive`` wrapper)
and the query names the order via in-call ``ORDER BY`` — DuckDB's own
ordered-aggregate spelling. The fit scope is stably sorted by the named
keys (DuckDB comparisons, input order breaking ties) before ``est.fit``.
Mechanism promise only: we sort by what you name. Spec:
docs/superpowers/specs/2026-08-05-fit-transform-split-design.md.
"""

import numpy as np
import pyarrow as pa
import pytest
from sklearn.preprocessing import StandardScaler

from sql_transform import MarginalizeError, OrderSensitive, SQLProjection

from ._transformers_test import TRAIN, _by_name

ROW = TRAIN.schema


class SeqMean:
    """Order-sensitive on purpose: fit folds rows into an exponentially
    weighted sum, so any reordering changes the fitted value."""

    def __init__(self):
        self.w = None

    def fit(self, X):
        self.w = sum(float(v[0]) / 2**i for i, v in enumerate(X))
        return self

    def transform(self, X):
        return [[float(v[0]) - self.w] for v in X]

    def get_feature_names_out(self, input_features=None):
        return list(input_features)


def _fit(sql: str) -> SQLProjection:
    return SQLProjection(
        sql,
        this_schema=ROW,
        transformers={"sm": OrderSensitive(SeqMean()), "sc": StandardScaler()},
    ).fit(TRAIN)


def _w(ages_in_order):
    return sum(a / 2**i for i, a in enumerate(ages_in_order))


AGES = TRAIN.column("age").to_pylist()
FARES = TRAIN.column("fare").to_pylist()
COUNTRIES = TRAIN.column("country").to_pylist()
NAMES = TRAIN.column("name").to_pylist()


def test_ordered_fit_global():
    p = _fit(
        "SELECT sm_transform(sm_fit(age ORDER BY fare) OVER (), age).age AS z,"
        " name FROM __THIS__"
    )
    order = sorted(range(len(AGES)), key=lambda i: FARES[i])
    w = _w([AGES[i] for i in order])
    got = _by_name(p.transform(TRAIN), "z")
    for i, n in enumerate(NAMES):
        np.testing.assert_allclose(got[n], AGES[i] - w, rtol=1e-12)
    row = dict(
        zip(
            TRAIN.column_names,
            [c[0].as_py() for c in TRAIN.columns],
            strict=True,
        )
    )
    np.testing.assert_allclose(p.infer(row)["z"], AGES[0] - w, rtol=1e-12)


def test_ordered_fit_partitioned():
    p = _fit(
        "SELECT sm_transform(sm_fit(age ORDER BY fare)"
        " OVER (PARTITION BY country), age).age AS z, name FROM __THIS__"
    )
    got = _by_name(p.transform(TRAIN), "z")
    for i, n in enumerate(NAMES):
        member = [j for j in range(len(AGES)) if COUNTRIES[j] == COUNTRIES[i]]
        member.sort(key=lambda j: FARES[j])
        w = _w([AGES[j] for j in member])
        np.testing.assert_allclose(got[n], AGES[i] - w, rtol=1e-12)


def test_desc_and_nulls_follow_duckdb_defaults():
    """Measured: DuckDB default null placement is NULLS LAST for both
    directions; ties keep input order (our stable-sort promise)."""
    p = _fit(
        "SELECT sm_transform(sm_fit(age ORDER BY country DESC) OVER (), age)"
        ".age AS a,"
        " sm_transform(sm_fit(age ORDER BY country NULLS FIRST) OVER (), age)"
        ".age AS b, name FROM __THIS__"
    )
    nn = [i for i in range(len(AGES)) if COUNTRIES[i] is not None]
    nulls = [i for i in range(len(AGES)) if COUNTRIES[i] is None]
    desc = sorted(nn, key=lambda i: COUNTRIES[i], reverse=True) + nulls
    nf = nulls + sorted(nn, key=lambda i: COUNTRIES[i])
    w_a, w_b = _w([AGES[i] for i in desc]), _w([AGES[i] for i in nf])
    out = p.transform(TRAIN)
    got_a, got_b = _by_name(out, "a"), _by_name(out, "b")
    for i, n in enumerate(NAMES):
        np.testing.assert_allclose(got_a[n], AGES[i] - w_a, rtol=1e-12)
        np.testing.assert_allclose(got_b[n], AGES[i] - w_b, rtol=1e-12)


def test_multi_key_order_and_stable_ties():
    p = _fit(
        "SELECT sm_transform(sm_fit(age ORDER BY country, fare) OVER (), age)"
        ".age AS z, name FROM __THIS__"
    )
    nn = [i for i in range(len(AGES)) if COUNTRIES[i] is not None]
    nulls = [i for i in range(len(AGES)) if COUNTRIES[i] is None]
    order = sorted(nn, key=lambda i: (COUNTRIES[i], FARES[i])) + sorted(
        nulls, key=lambda i: FARES[i]
    )
    w = _w([AGES[i] for i in order])
    got = _by_name(p.transform(TRAIN), "z")
    for i, n in enumerate(NAMES):
        np.testing.assert_allclose(got[n], AGES[i] - w, rtol=1e-12)


def test_ordered_plus_filter():
    p = _fit(
        "SELECT sm_transform(sm_fit(age ORDER BY fare)"
        " FILTER (WHERE fare > 6) OVER (), age).age AS z, name FROM __THIS__"
    )
    passing = [i for i in range(len(AGES)) if FARES[i] > 6]
    passing.sort(key=lambda i: FARES[i])
    w = _w([AGES[i] for i in passing])
    got = _by_name(p.transform(TRAIN), "z")
    for i, n in enumerate(NAMES):
        np.testing.assert_allclose(got[n], AGES[i] - w, rtol=1e-12)


def test_order_blind_fit_accepts_and_honors_order_by():
    """The oracle accepts in-call ORDER BY on order-blind aggregates; an
    order-blind fit gives the same answer either way."""
    ordered = _fit(
        "SELECT sc_transform(sc_fit(age ORDER BY fare) OVER (), age).age AS z,"
        " name FROM __THIS__"
    )
    plain = _fit(
        "SELECT sc_transform(sc_fit(age) OVER (), age).age AS z, name FROM __THIS__"
    )
    got = _by_name(ordered.transform(TRAIN), "z")
    want = _by_name(plain.transform(TRAIN), "z")
    for n, v in want.items():
        np.testing.assert_allclose(got[n], v, rtol=1e-12)


def test_distinct_orders_mint_distinct_steps():
    p = _fit(
        "SELECT sm_transform(sm_fit(age ORDER BY fare) OVER (), age).age AS a,"
        " sm_transform(sm_fit(age ORDER BY fare DESC) OVER (), age).age AS b,"
        " name FROM __THIS__"
    )
    assert len([s for s in p.plan if s.kind == "fit"]) == 2
    asc = sorted(range(len(AGES)), key=lambda i: FARES[i])
    w_a = _w([AGES[i] for i in asc])
    w_b = _w([AGES[i] for i in reversed(asc)])
    out = p.transform(TRAIN)
    got_a, got_b = _by_name(out, "a"), _by_name(out, "b")
    for i, n in enumerate(NAMES):
        np.testing.assert_allclose(got_a[n], AGES[i] - w_a, rtol=1e-12)
        np.testing.assert_allclose(got_b[n], AGES[i] - w_b, rtol=1e-12)


def test_theta_lateral_ordered_equals_inline():
    lateral = _fit(
        "SELECT sm_fit(age ORDER BY fare) OVER () AS _th,"
        " sm_transform(_th, age).age AS z, name FROM __THIS__"
    )
    inline = _fit(
        "SELECT sm_transform(sm_fit(age ORDER BY fare) OVER (), age).age AS z,"
        " name FROM __THIS__"
    )
    assert lateral.serving_sql == inline.serving_sql


def test_collate_key_is_honored():
    """Review round: the collation annotation dropped through Arrow and the
    sort ran binary — the named collation must do the comparing."""
    import duckdb

    t = pa.table({"rid": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "s": list("bABaCc")})
    model = t.schema
    p = SQLProjection(
        "SELECT sm_transform(sm_fit(rid ORDER BY s COLLATE NOCASE) OVER (), rid)"
        ".rid AS z, rid FROM __THIS__",
        this_schema=model,
        transformers={"sm": OrderSensitive(SeqMean())},
    ).fit(t)
    con = duckdb.connect()
    con.register("t", t)
    (oracle_order,) = con.execute(
        "SELECT list(rid ORDER BY s COLLATE NOCASE, rid) FROM t"
    ).fetchone()
    con.close()
    w = _w(list(oracle_order))
    got = {r["rid"]: r["z"] for r in p.transform(t).to_pylist()}
    for rid in t.column("rid").to_pylist():
        np.testing.assert_allclose(got[rid], rid - w, rtol=1e-12)


def test_named_wrapping_order_sensitive_still_requires_order():
    """Review round: Named had no attribute forwarding, so it silently
    cancelled the inner order-sensitivity declaration."""
    from sql_transform import Named

    with pytest.raises(MarginalizeError, match="order-sensitive"):
        SQLProjection(
            "SELECT sm_transform(sm_fit(age) OVER (), age).age AS z FROM __THIS__",
            this_schema=ROW,
            transformers={"sm": Named(OrderSensitive(SeqMean()), returns=("age",))},
        )


def test_wrapper_survives_pickle_roundtrip():
    """Review round: __getattr__ raised KeyError (not AttributeError) on
    empty instance state, crashing pickle/copy protocols."""
    import pickle

    # S301: test-local roundtrip of our own object, no untrusted data.
    back = pickle.loads(pickle.dumps(OrderSensitive(StandardScaler())))  # noqa: S301
    assert back.order_sensitive is True
    assert isinstance(back.estimator, StandardScaler)


def test_integer_literal_key_is_a_constant():
    """Measured: DuckDB treats in-call ORDER BY 1 as a constant (not
    positional) — all ties, input order."""
    ordered = _fit(
        "SELECT sm_transform(sm_fit(age ORDER BY 1) OVER (), age).age AS z,"
        " name FROM __THIS__"
    )
    w = _w(AGES)
    got = _by_name(ordered.transform(TRAIN), "z")
    for i, n in enumerate(NAMES):
        np.testing.assert_allclose(got[n], AGES[i] - w, rtol=1e-12)


REFUSALS = [
    # An order-sensitive transformer must name its order.
    (
        "SELECT sm_transform(sm_fit(age) OVER (), age).age AS z FROM __THIS__",
        "order-sensitive",
    ),
    ("SELECT sm(age).age AS z FROM __THIS__", "order-sensitive"),
    # Order keys resolve like any fit-side expression.
    (
        "SELECT sm_transform(sm_fit(age ORDER BY nope) OVER (), age).age"
        " AS z FROM __THIS__",
        "unknown column nope",
    ),
    (
        "SELECT sm_transform(sm_fit(age ORDER BY sc(fare).fare) OVER (), age)"
        ".age AS z FROM __THIS__",
        "inside an ORDER BY key",
    ),
    # Review round: collation edges refuse by name at construction.
    (
        "SELECT sm_transform(sm_fit(age ORDER BY name COLLATE nosuch)"
        " OVER (), age).age AS z FROM __THIS__",
        "collation",
    ),
    (
        "SELECT sm_transform(sm_fit(age ORDER BY (name COLLATE NOCASE) || 'x')"
        " OVER (), age).age AS z FROM __THIS__",
        "COLLATE inside",
    ),
    # DuckDB's binder rule: a non-integer literal key has no effect.
    (
        "SELECT sm_transform(sm_fit(age ORDER BY 'a') OVER (), age).age"
        " AS z FROM __THIS__",
        "non-integer literal",
    ),
    # In-call ORDER BY binds only on aggregates (measured) — every scalar
    # spelling refuses instead of crashing at serving or dropping silently.
    (
        "SELECT round(age ORDER BY fare) AS r FROM __THIS__",
        "ORDER BY inside the scalar call",
    ),
    ("SELECT sc(age ORDER BY fare).age AS z FROM __THIS__", "ORDER BY"),
    (
        "SELECT sc_transform(sc_fit(age) OVER (), age ORDER BY fare).age"
        " AS z FROM __THIS__",
        "ORDER BY",
    ),
]


@pytest.mark.parametrize("sql,match", REFUSALS)
def test_ordered_refusals(sql, match):
    with pytest.raises(MarginalizeError, match=match):
        SQLProjection(
            sql,
            this_schema=ROW,
            transformers={"sm": OrderSensitive(SeqMean()), "sc": StandardScaler()},
        )
