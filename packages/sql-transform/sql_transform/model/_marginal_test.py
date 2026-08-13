"""``SQLProjection.marginalize`` — the ``__FIT__`` half derived from a
``__THIS__``-only text.

The law (spec M2): ``marginalize(text).fit(F).transform(F)`` equals
``run(SQLTransform(text), F)`` — freezing is invisible on the fit data, and
divergence exists only at unseen-partition misses (P14 NULL). Gates ``law``,
``freeze``, ``params``, ``serving`` and ``attribution`` of
`docs/superpowers/specs/2026-08-13-marginalize-design.md`.
"""

import math

import pyarrow as pa
import pytest

from sql_transform.model import SQLProjection, SQLTransform, TransformError, run

F = pa.table(
    {
        "store": ["S1", "S1", "S1", "S2", "S2", None],
        "price": [10.0, 20.0, 30.0, 100.0, 300.0, 7.0],
    }
)
X = pa.table(
    {
        "store": ["S2", "NEW", None, "S1"],
        "price": [200.0, 7.0, 14.0, 10.0],
    }
)

LAWFUL = {
    "global": "SELECT store, price / avg(price) OVER () AS r FROM __THIS__",
    "per_key": (
        "SELECT store, price - avg(price) OVER (PARTITION BY store) AS d FROM __THIS__"
    ),
    "shared_key": """
        SELECT store,
               (price - avg(price) OVER (PARTITION BY store))
               / stddev_pop(price) OVER (PARTITION BY store) AS z
        FROM __THIS__
    """,
    "key_expression": (
        "SELECT store, price - min(price) OVER (PARTITION BY substr(store, 1, 1))"
        " AS d FROM __THIS__"
    ),
    "filter_rides": (
        "SELECT store, price - avg(price) FILTER (WHERE price > 8.0)"
        " OVER (PARTITION BY store) AS d FROM __THIS__"
    ),
    "spine_alias": (
        "SELECT t.store, t.price / sum(t.price) OVER (PARTITION BY t.store) AS s"
        " FROM __THIS__ t"
    ),
    "mixed_scopes": (
        "SELECT store, avg(price) OVER () - avg(price) OVER (PARTITION BY store)"
        " AS gap FROM __THIS__"
    ),
}


def _sorted(rows: list[dict]) -> list[dict]:
    """Content-sorted, NaN made comparable: nan != nan would fail a lawful
    0/0 that both sides produced identically."""
    canon = [
        {
            k: "NaN" if isinstance(v, float) and math.isnan(v) else v
            for k, v in r.items()
        }
        for r in rows
    ]
    return sorted(canon, key=lambda r: tuple((v is None, str(v)) for v in r.values()))


@pytest.mark.parametrize("text", LAWFUL.values(), ids=LAWFUL.keys())
def test_the_law_frozen_equals_transductive_on_the_fit_data(text):
    frozen = SQLProjection.marginalize(text).fit(F).transform(F).to_pylist()
    transductive = run(SQLTransform(text), F).to_pylist()
    assert _sorted(frozen) == _sorted(transductive)


def test_divergence_is_only_at_misses():
    """On new data: seen partitions answer from frozen θ, the unseen partition
    is NULL (P14), and a NULL key joins its own partition — window semantics."""
    fitted = SQLProjection.marginalize(LAWFUL["per_key"]).fit(F)
    assert fitted.transform(X).to_pylist() == [
        {"store": "S2", "d": 0.0},  # θ(S2) = 200.0, frozen
        {"store": "NEW", "d": None},  # no θ: NULL out, where transductive refits
        {"store": None, "d": 7.0},  # NULL key → the NULL partition's θ (7.0)
        {"store": "S1", "d": -10.0},
    ]


def test_scopes_sharing_a_key_share_one_derived_join():
    two = SQLProjection.marginalize(LAWFUL["shared_key"])
    assert two.sql.count("JOIN") == 1
    assert "GROUP BY" in two.sql  # M3: the derived text is SQL you can read


def test_frozen_thetas_are_ordinary_params():
    fitted = SQLProjection.marginalize(LAWFUL["per_key"]).fit(F)
    (params,) = fitted.params.values()
    by_key = {r["cf_k0"]: r["cf_w0"] for r in params.to_pylist()}
    assert by_key == {"S1": 20.0, "S2": 200.0, None: 7.0}
    assert fitted.instances == {}


def test_a_marginalized_projection_serves():
    fitted = SQLProjection.marginalize(LAWFUL["per_key"]).fit(F)
    rows = [r.model_dump() for r in fitted.compile().infer_rows(X.to_pylist())]
    assert rows == fitted.transform(X).to_pylist()


def test_the_fresh_prefix_dodges_the_authors_names():
    """The derived names live in author space (the derived text is an ordinary
    text), so they must step aside when the author already uses the prefix."""
    text = (
        "SELECT store AS cf_k0, price - avg(price) OVER (PARTITION BY store) AS d"
        " FROM __THIS__"
    )
    frozen = SQLProjection.marginalize(text).fit(F).transform(F).to_pylist()
    transductive = run(SQLTransform(text), F).to_pylist()
    assert _sorted(frozen) == _sorted(transductive)


REFUSED = [
    ("WHERE", "SELECT price FROM __THIS__ WHERE price > 0"),
    ("GROUP BY", "SELECT avg(price) AS m FROM __THIS__ GROUP BY store"),
    ("GROUP BY", "SELECT avg(price) AS m FROM __THIS__ GROUP BY ALL"),
    ("QUALIFY", "SELECT price FROM __THIS__ QUALIFY sum(price) OVER () > 0"),
    (
        "set operation",
        "SELECT price FROM __THIS__ UNION ALL SELECT price FROM __THIS__",
    ),
    ("CTE", "WITH c AS (SELECT 1 AS one) SELECT price FROM __THIS__"),
    ("subquery", "SELECT (SELECT 1) AS one, price FROM __THIS__"),
    (r"OVER \(\)", "SELECT avg(price) AS m FROM __THIS__"),  # bare aggregate
    (
        "ORDER BY",
        "SELECT sum(price) OVER (PARTITION BY store ORDER BY price) AS s FROM __THIS__",
    ),
    ("positional", "SELECT row_number() OVER (PARTITION BY store) AS n FROM __THIS__"),
    (
        "frame",
        "SELECT avg(price) OVER (ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS a"
        " FROM __THIS__",
    ),
    (r"\*", "SELECT *, avg(price) OVER () AS m FROM __THIS__"),
    (
        "__FIT__",
        "SELECT t.price / f.m AS r"
        " FROM __THIS__ t, (SELECT avg(price) AS m FROM __FIT__) f",
    ),
    (
        f"more than {'__THIS__'}",
        "SELECT a.price FROM __THIS__ a JOIN codes c ON a.store = c.store",
    ),
    ("ORDER BY", "SELECT price FROM __THIS__ ORDER BY price"),
    ("inside", "SELECT avg(sum(price)) OVER (PARTITION BY store) AS a FROM __THIS__"),
    ("alias", "SELECT price - avg(price) OVER () FROM __THIS__"),  # unnamed scope
]


@pytest.mark.parametrize(("token", "text"), REFUSED, ids=[t for t, _ in REFUSED])
def test_refusals_fire_pre_rewrite_in_the_authors_vocabulary(token, text):
    with pytest.raises(TransformError, match=token):
        SQLProjection.marginalize(text)


# --- projection scopes (slice 3) -------------------------------------------

zscore = SQLProjection("""
    SELECT round((t.price - f.m) / f.s, 4) AS z
    FROM __THIS__ t,
         (SELECT avg(price) AS m, stddev_pop(price) AS s FROM __FIT__) f
""")

PROJECTION_LAWFUL = {
    "bare_global": (
        "SELECT store, zscore(struct_pack(price := price)).z AS z FROM __THIS__"
    ),
    "split_per_key": """
        SELECT store,
               zscore_transform(
                   zscore_fit(struct_pack(price := price)) OVER (PARTITION BY store),
                   struct_pack(price := price)).z AS z
        FROM __THIS__
    """,
    "mixed_with_plain": """
        SELECT store,
               zscore_transform(
                   zscore_fit(struct_pack(price := price)) OVER (PARTITION BY store),
                   struct_pack(price := price)).z AS z,
               price - avg(price) OVER (PARTITION BY store) AS d
        FROM __THIS__
    """,
}


@pytest.mark.parametrize(
    "text", PROJECTION_LAWFUL.values(), ids=PROJECTION_LAWFUL.keys()
)
def test_the_law_holds_for_projection_scopes(text):
    frozen = SQLProjection.marginalize(text).fit(F).transform(F).to_pylist()
    transductive = run(SQLTransform(text), F).to_pylist()
    assert _sorted(frozen) == _sorted(transductive)


def test_a_projection_scope_shares_the_join_with_plain_scopes():
    """One key tuple, one derived join — a θ column and a plain aggregate
    column side by side in the same params table."""
    p = SQLProjection.marginalize(PROJECTION_LAWFUL["mixed_with_plain"])
    assert p.sql.count("JOIN") == 1


def test_projection_theta_misses_are_null():
    fitted = SQLProjection.marginalize(PROJECTION_LAWFUL["split_per_key"]).fit(F)
    by = {r["store"]: r["z"] for r in fitted.transform(X).to_pylist()}
    assert by["NEW"] is None  # P14, through the leaf: NULL θ in, NULL out


def test_a_projection_scope_serves_in_batch_and_refuses_the_row_path_loudly():
    """The frozen θ crosses the derived join as a struct, and the residual
    reads it with struct_extract — which Confit's v0 catalogue lacks. Batch
    is unaffected; compile() refuses with Confit's own message, by name
    (recorded gap: spec Deferred)."""
    fitted = SQLProjection.marginalize(PROJECTION_LAWFUL["split_per_key"]).fit(F)
    assert fitted.transform(X).to_pylist()[1] == {"store": "NEW", "z": None}
    with pytest.raises(ValueError, match="struct_extract"):
        fitted.compile()


PROJECTION_REFUSED = [
    # the deleted sugar stays deleted: the OVER belongs on the fit half
    (
        r"zscore_fit",
        "SELECT zscore(struct_pack(price := price)) OVER (PARTITION BY store) AS z"
        " FROM __THIS__",
    ),
    # a fit call with no scope — even the global scope is spelled OVER ()
    (
        r"OVER",
        """SELECT zscore_transform(
               zscore_fit(struct_pack(price := price)),
               struct_pack(price := price)).z AS z FROM __THIS__""",
    ),
    # FILTER on a projection fit scope has no frozen spelling yet
    (
        r"FILTER",
        """SELECT zscore_transform(
               zscore_fit(struct_pack(price := price))
                   FILTER (WHERE price > 0) OVER (PARTITION BY store),
               struct_pack(price := price)).z AS z FROM __THIS__""",
    ),
    # a fit scope inside a bare call's bundle
    (
        r"inside",
        "SELECT zscore(struct_pack(price := price - avg(price) OVER ())).z AS z"
        " FROM __THIS__",
    ),
]


@pytest.mark.parametrize(
    ("token", "text"), PROJECTION_REFUSED, ids=[t for t, _ in PROJECTION_REFUSED]
)
def test_projection_scope_refusals_in_the_authors_vocabulary(token, text):
    with pytest.raises(TransformError, match=token):
        SQLProjection.marginalize(text)
