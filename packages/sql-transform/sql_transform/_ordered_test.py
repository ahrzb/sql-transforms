"""Ordered fits — fit/transform-split slice 4.

An order-sensitive transformer declares it (``OrderSensitive`` wrapper)
and the query names the order via in-call ``ORDER BY`` — DuckDB's own
ordered-aggregate spelling. The fit scope is stably sorted by the named
keys (DuckDB comparisons, input order breaking ties) before ``est.fit``.
Mechanism promise only: we sort by what you name. Spec:
docs/superpowers/specs/2026-08-05-fit-transform-split-design.md.
"""

import numpy as np
import pydantic
import pytest
from sklearn.preprocessing import StandardScaler

from sql_transform import MarginalizeError, OrderSensitive, SQLProjection

from ._transformers_test import TRAIN, _by_name

ROW = pydantic.create_model("Row", **dict.fromkeys(TRAIN.column_names, (object, None)))


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
        this_model=ROW,
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
    np.testing.assert_allclose(p.infer(row).z, AGES[0] - w, rtol=1e-12)


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
]


@pytest.mark.parametrize("sql,match", REFUSALS)
def test_ordered_refusals(sql, match):
    with pytest.raises(MarginalizeError, match=match):
        SQLProjection(
            sql,
            this_model=ROW,
            transformers={"sm": OrderSensitive(SeqMean()), "sc": StandardScaler()},
        )
