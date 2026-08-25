"""``SQLProjection.marginalize`` — the ``__FIT__`` half derived from a
``__THIS__``-only text.

The law (spec M2): ``marginalize(text).fit(F).transform(F)`` equals
``run(SQLTransform(text), F)`` — freezing is invisible on the fit data, and
divergence exists only at unseen-partition misses (P14 NULL). Gates ``law``,
``freeze``, ``params``, ``serving`` and ``attribution`` of
`docs/specs/2026-08-13-marginalize-design.md`.
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
    rows = fitted.compile().infer_rows(X.to_pylist())
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


def test_struct_paths_survive_the_spine_qualifier():
    """`t.p.v` strips to `p.v`, never to bare `v` — a decoy column named `v`
    makes truncation a law violation instead of a bind error."""
    S = pa.table(
        {
            "store": ["S1", "S1", "S2"],
            "v": [1.0, 1.0, 1.0],
            "p": pa.array(
                [{"v": 10.0}, {"v": 20.0}, {"v": 100.0}],
                type=pa.struct([("v", pa.float64())]),
            ),
        }
    )
    text = (
        "SELECT t.store, t.p.v - avg(t.p.v) OVER (PARTITION BY t.store) AS d"
        " FROM __THIS__ t"
    )
    frozen = SQLProjection.marginalize(text).fit(S).transform(S).to_pylist()
    assert _sorted(frozen) == _sorted(run(SQLTransform(text), S).to_pylist())


def test_struct_paths_survive_as_partition_keys():
    S = pa.table(
        {
            "k": ["A", "A", "B"],
            "x": [1.0, 2.0, 40.0],
            "g": pa.array(
                [{"k": "A"}, {"k": "B"}, {"k": "B"}],
                type=pa.struct([("k", pa.string())]),
            ),
        }
    )
    text = "SELECT t.x, avg(t.x) OVER (PARTITION BY t.g.k) AS m FROM __THIS__ t"
    frozen = SQLProjection.marginalize(text).fit(S).transform(S).to_pylist()
    assert _sorted(frozen) == _sorted(run(SQLTransform(text), S).to_pylist())


def test_an_integer_literal_partition_key_is_a_constant_not_an_ordinal():
    """In a window, `PARTITION BY 2` is the constant; the derived GROUP BY
    must not let it decay into a positional ordinal."""
    text = "SELECT store, price - avg(price) OVER (PARTITION BY 2) AS d FROM __THIS__"
    frozen = SQLProjection.marginalize(text).fit(F).transform(F).to_pylist()
    assert _sorted(frozen) == _sorted(run(SQLTransform(text), F).to_pylist())


def test_a_schema_qualified_aggregate_freezes():
    text = (
        "SELECT store, price - main.avg(price) OVER (PARTITION BY store) AS d"
        " FROM __THIS__"
    )
    frozen = SQLProjection.marginalize(text).fit(F).transform(F).to_pylist()
    assert _sorted(frozen) == _sorted(run(SQLTransform(text), F).to_pylist())


@pytest.mark.xfail(
    strict=True,
    reason="Arrow cannot carry a BIT (nor TIMETZ/UNION) key faithfully through "
    "the params table; the frozen key misses its own row. Recorded gap, "
    "spec Deferred.",
)
def test_an_arrow_hostile_partition_key_type_holds_the_law():
    Q = pa.table(
        {
            "store": ["S1", "S1", "S2"],
            "qty": [1, 1, 2],
            "price": [10.0, 20.0, 100.0],
        }
    )
    text = "SELECT store, avg(price) OVER (PARTITION BY qty::BIT) AS v FROM __THIS__"
    frozen = SQLProjection.marginalize(text).fit(Q).transform(Q).to_pylist()
    assert _sorted(frozen) == _sorted(run(SQLTransform(text), Q).to_pylist())


# --- the widened window vocabulary (slice 5) --------------------------------

ORD = pa.table(
    {
        "store": ["S1", "S1", "S1", "S1", "S2", "S2", None],
        "d": [1, 1, 2, None, 1, 2, 1],
        "price": [10.0, 20.0, 30.0, 5.0, 100.0, 300.0, 7.0],
    }
)

ORDERED_LAWFUL = {
    "cumulative": (
        "SELECT store, d, sum(price) OVER (PARTITION BY store ORDER BY d) AS s"
        " FROM __THIS__"
    ),
    "range_offset": (
        "SELECT store, d, sum(price) OVER (PARTITION BY store ORDER BY d"
        " RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) AS s FROM __THIS__"
    ),
    "groups": (
        "SELECT store, d, sum(price) OVER (PARTITION BY store ORDER BY d"
        " GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s FROM __THIS__"
    ),
    "desc_nulls_first": (
        "SELECT store, d, sum(price) OVER (PARTITION BY store"
        " ORDER BY d DESC NULLS FIRST) AS s FROM __THIS__"
    ),
    "two_orders": (
        "SELECT store, d, sum(price) OVER (PARTITION BY store ORDER BY d, price)"
        " AS s FROM __THIS__"
    ),
    "no_partition": "SELECT d, sum(price) OVER (ORDER BY d) AS s FROM __THIS__",
}


@pytest.mark.parametrize("text", ORDERED_LAWFUL.values(), ids=ORDERED_LAWFUL.keys())
def test_the_law_holds_for_order_discriminating_frames(text):
    """RANGE/GROUPS peers share order values, so the window's value is a
    function of (partition keys ⊕ order values) — the frozen join carries
    exactly that key. Ties and a NULL order value are in the fixture."""
    frozen = SQLProjection.marginalize(text).fit(ORD).transform(ORD).to_pylist()
    transductive = run(SQLTransform(text), ORD).to_pylist()
    assert _sorted(frozen) == _sorted(transductive)


def test_an_unseen_order_value_is_a_miss():
    """Frozen keys are the fit data's (store, d) pairs: a new d is NULL where
    true RANGE semantics would refit — the same divergence-only-at-misses."""
    fitted = SQLProjection.marginalize(ORDERED_LAWFUL["cumulative"]).fit(ORD)
    X3 = pa.table({"store": ["S1", "S1"], "d": [2, 3], "price": [1.0, 1.0]})
    assert fitted.transform(X3).to_pylist() == [
        {"store": "S1", "d": 2, "s": 60.0},  # frozen cumulative at d=2
        {"store": "S1", "d": 3, "s": None},  # unseen order value
    ]


def test_an_ordered_scope_serves():
    fitted = SQLProjection.marginalize(ORDERED_LAWFUL["cumulative"]).fit(ORD)
    rows = fitted.compile().infer_rows(ORD.to_pylist())
    assert rows == fitted.transform(ORD).to_pylist()


SUBQUERY_LAWFUL = {
    "global_max": (
        "SELECT store, price / (SELECT max(price) FROM __THIS__) AS r FROM __THIS__"
    ),
    "aliased_where": (
        "SELECT store, price - (SELECT avg(i.price) FROM __THIS__ i"
        " WHERE i.price > 8) AS d FROM __THIS__"
    ),
    "order_limit": (
        "SELECT store, price - (SELECT price FROM __THIS__ ORDER BY price LIMIT 1)"
        " AS d FROM __THIS__"
    ),
    "count_star": (
        "SELECT store, price * (SELECT count(*) FROM __THIS__) AS n FROM __THIS__"
    ),
}


@pytest.mark.parametrize("text", SUBQUERY_LAWFUL.values(), ids=SUBQUERY_LAWFUL.keys())
def test_the_law_holds_for_scalar_subqueries(text):
    """An uncorrelated single-level subquery over __THIS__ freezes verbatim
    over __FIT__ — one value, joined one-row."""
    frozen = SQLProjection.marginalize(text).fit(F).transform(F).to_pylist()
    transductive = run(SQLTransform(text), F).to_pylist()
    assert _sorted(frozen) == _sorted(transductive)


def test_a_frozen_subquery_serves():
    fitted = SQLProjection.marginalize(SUBQUERY_LAWFUL["global_max"]).fit(F)
    rows = fitted.compile().infer_rows(X.to_pylist())
    assert rows == fitted.transform(X).to_pylist()


def test_theta_parks_in_a_lateral_alias_and_is_read_twice():
    """θ of a keyless projection is a value (D1): park it `AS t`, then read
    it from sibling items with two different bundles — one fit, many reads.
    Works by composition (lateral aliases + the ordinary splice); pinned so
    it stays working. Measured 2026-08-13."""
    text = """
        SELECT store,
               zscore_fit(struct_pack(price := price))
                   OVER (PARTITION BY store) AS t,
               zscore_transform(t, struct_pack(price := price)).z AS z,
               zscore_transform(t, struct_pack(price := price * 2)).z AS z2
        FROM __THIS__
    """
    frozen = SQLProjection.marginalize(text).fit(F).transform(F).to_pylist()
    assert _sorted(frozen) == _sorted(run(SQLTransform(text), F).to_pylist())


# --- key composition (slice 4, RFC M5) --------------------------------------

keyed = SQLProjection("""
    SELECT t.price / f.m AS r
    FROM __THIS__ t
    LEFT JOIN (SELECT store, avg(price) AS m FROM __FIT__ GROUP BY store) f
      ON t.store = f.store
""")

CITY = pa.table(
    {
        "city": ["C1", "C1", "C1", "C2", "C2"],
        "store": ["S1", "S1", "S2", "S1", None],
        "price": [10.0, 30.0, 100.0, 4.0, 9.0],
    }
)

KEYED_TEXT = """
    SELECT city, store,
           keyed_transform(
               keyed_fit(struct_pack(store := store, price := price))
                   OVER (PARTITION BY city),
               struct_pack(store := store, price := price)).r AS r
    FROM __THIS__
"""


def test_key_composition_equals_per_scope_standalone_fits():
    """The definitional gate: each scope's answer is the keyed projection
    fitted standalone on that scope's rows. Effective key = city ⊕ store;
    the internal `=` keeps its lookup semantics (NULL store misses)."""
    out = SQLProjection.marginalize(KEYED_TEXT).fit(CITY).transform(CITY).to_pylist()
    for i, row in enumerate(out):
        group = CITY.filter(pa.compute.equal(CITY["city"], row["city"]))
        expected = keyed.fit(group).transform(CITY.slice(i, 1)).to_pylist()[0]["r"]
        assert row["r"] == expected, (i, row)


def test_key_composition_misses_are_null_on_either_half():
    fitted = SQLProjection.marginalize(KEYED_TEXT).fit(CITY)
    X2 = pa.table(
        {
            "city": ["C9", "C1", "C1"],
            "store": ["S1", "S9", "S1"],
            "price": [10.0, 10.0, 40.0],
        }
    )
    assert fitted.transform(X2).to_pylist() == [
        {"city": "C9", "store": "S1", "r": None},  # scope-key miss
        {"city": "C1", "store": "S9", "r": None},  # internal-key miss
        {"city": "C1", "store": "S1", "r": 40.0 / 20.0},
    ]


def test_key_composition_serves():
    """A keyed scope's params are flat columns — no struct θ — so unlike the
    keyless projection scope, the row path works."""
    fitted = SQLProjection.marginalize(KEYED_TEXT).fit(CITY)
    rows = fitted.compile().infer_rows(CITY.to_pylist())
    assert rows == fitted.transform(CITY).to_pylist()


def test_bare_sugar_on_a_keyed_projection_is_the_internal_key_alone():
    """`keyed(bundle)` has no scope keys: one global fit, per-store lookup."""
    text = (
        "SELECT store, keyed(struct_pack(store := store, price := price)).r AS r"
        " FROM __THIS__"
    )
    out = SQLProjection.marginalize(text).fit(CITY).transform(CITY).to_pylist()
    expected = keyed.fit(CITY).transform(CITY).to_pylist()
    assert [r["r"] for r in out] == [r["r"] for r in expected]


def test_a_keyed_fit_scope_outside_its_transform_refuses_by_name():
    with pytest.raises(TransformError, match=r"keyed.*transform"):
        SQLProjection.marginalize("""
            SELECT keyed_fit(struct_pack(store := store, price := price))
                       OVER (PARTITION BY city) AS th
            FROM __THIS__
        """)


def test_a_keyed_projection_beyond_one_grouped_step_refuses_by_name():
    twostep = SQLProjection("""
        SELECT t.price / f.m + g.a AS r
        FROM __THIS__ t
        LEFT JOIN (SELECT store, avg(price) AS m FROM __FIT__ GROUP BY store) f
          ON t.store = f.store
        LEFT JOIN (SELECT avg(price) AS a FROM __FIT__) g ON 1 = 1
    """)
    assert twostep is not None
    with pytest.raises(TransformError, match=r"twostep"):
        SQLProjection.marginalize("""
            SELECT twostep_transform(
                twostep_fit(struct_pack(store := store, price := price))
                    OVER (PARTITION BY city),
                struct_pack(store := store, price := price)).r AS r
            FROM __THIS__
        """)


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
        "ROWS",
        "SELECT sum(price) OVER (PARTITION BY store ORDER BY price"
        " ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s FROM __THIS__",
    ),
    (
        "EXCLUDE",
        "SELECT sum(price) OVER (PARTITION BY store ORDER BY price"
        " RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
        " EXCLUDE CURRENT ROW) AS s FROM __THIS__",
    ),
    (
        "correlat",
        "SELECT (SELECT max(i.price) FROM __THIS__ i WHERE i.store = t.store) AS m"
        " FROM __THIS__ t",
    ),
    (
        "shadow",
        "SELECT (SELECT max(t.price) FROM __THIS__ t) AS m FROM __THIS__ t",
    ),
    (
        "nest",
        "SELECT (SELECT max(price) + (SELECT min(price) FROM __THIS__)"
        " FROM __THIS__) AS m FROM __THIS__",
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
    (
        "inside",
        "SELECT avg(price - avg(price) OVER ()) OVER (PARTITION BY store) AS a"
        " FROM __THIS__",
    ),
    ("alias", "SELECT price - avg(price) OVER () FROM __THIS__"),  # unnamed scope
    ("whole", "SELECT store, max(t) OVER () AS m FROM __THIS__ t"),
    (
        "lambda",
        "SELECT avg(list_sum(list_transform(arr, x -> x.v))) OVER () AS s"
        " FROM __THIS__",
    ),
    ("positional", "SELECT #1 AS a, avg(price) OVER () AS m FROM __THIS__"),
    ("column-alias list", "SELECT a FROM __THIS__ t(a, b)"),
    ("SAMPLE", "SELECT price FROM __THIS__ TABLESAMPLE reservoir(2 ROWS)"),
    (
        "sibling",
        "SELECT price AS p, avg(p) OVER (PARTITION BY store) AS m FROM __THIS__",
    ),
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
    # an ordered fit scope is a running fit — per-row θ, still deferred
    (
        r"running",
        """SELECT zscore_transform(
               zscore_fit(struct_pack(price := price)) OVER (ORDER BY price),
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
