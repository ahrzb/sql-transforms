"""``SQLProjection.marginalize`` — the ``__FIT__`` half derived from a
``__THIS__``-only text.

The law: ``marginalize(text).fit(F).transform(F)`` equals the original text
run over ``F`` — freezing is invisible on the fit data, and divergence exists
only at unseen-key misses (NULL). The original query is the oracle; the
window carrier keeps its numbers exact, so the new cases compare every float
by its bits.
"""

import math
import struct

import duckdb
import pyarrow as pa
import pytest

from sql_transform import SQLProjection, SQLTransform, TransformError, run

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


def _original(text: str, table: pa.Table) -> pa.Table:
    """The author's own text over ``table`` as ``__THIS__``, at threads=1."""
    con = duckdb.connect()
    try:
        con.execute("SET threads = 1")
        con.register("__THIS__", table)
        return con.execute(text).to_arrow_table()
    finally:
        con.close()


def _bits(v):
    if isinstance(v, float):
        return struct.pack(">d", v).hex()  # signed zero and NaN, exactly
    if isinstance(v, dict):
        return {k: _bits(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_bits(x) for x in v]
    return v


def _exact(table: pa.Table) -> tuple[list, list]:
    """The schema's names and types, and the rows as a multiset with every
    float compared by its bits."""
    schema = [(f.name, str(f.type)) for f in table.schema]
    return schema, sorted((_bits(r) for r in table.to_pylist()), key=repr)


def _law(text: str, table: pa.Table) -> None:
    frozen = SQLProjection.marginalize(text).fit(table).transform(table)
    assert _exact(frozen) == _exact(_original(text, table))


@pytest.mark.parametrize("text", LAWFUL.values(), ids=LAWFUL.keys())
def test_the_law_frozen_equals_transductive_on_the_fit_data(text):
    frozen = SQLProjection.marginalize(text).fit(F).transform(F).to_pylist()
    transductive = run(SQLTransform(text), F).to_pylist()
    assert _sorted(frozen) == _sorted(transductive)


@pytest.mark.parametrize("text", LAWFUL.values(), ids=LAWFUL.keys())
def test_the_law_holds_bit_for_bit_against_the_original_query(text):
    _law(text, F)


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


def test_frozen_thetas_are_ordinary_params():
    fitted = SQLProjection.marginalize(LAWFUL["per_key"]).fit(F)
    (params,) = fitted.params.values()
    by_key = {r["cf_k0"]: r["cf_w0"] for r in params.to_pylist()}
    assert by_key == {"S1": 20.0, "S2": 200.0, None: 7.0}
    assert params.num_rows == 3  # one row per fitted key, not per fit row
    assert fitted.instances == {}


def test_params_scale_with_fitted_keys_and_the_carrier_never_ships():
    """Every scope's params table holds one row per fitted key; the fit-time
    carrier, one row per fit row, is not among them."""
    big = pa.table(
        {
            "store": ["S1", "S2", None] * 50,
            "price": [float(i) for i in range(150)],
        }
    )
    fitted = SQLProjection.marginalize(LAWFUL["mixed_scopes"]).fit(big)
    assert sorted(t.num_rows for t in fitted.params.values()) == [1, 3]


def test_an_empty_fit_misses_every_request():
    fitted = SQLProjection.marginalize(LAWFUL["mixed_scopes"]).fit(F.slice(0, 0))
    assert [r["gap"] for r in fitted.transform(X).to_pylist()] == [None] * 4


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
    """`t.p.v` stays the struct path `p.v`, never bare `v` — a decoy column
    named `v` makes truncation a law violation instead of a bind error."""
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
    """In a window, `PARTITION BY 2` is the constant, and so is the lookup
    key it becomes."""
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
    "the params table; the frozen key misses its own row. Recorded gap.",
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


# --- the carrier: original numbers, original names -------------------------

# Seed 20260729, generated case 206 of the 1,500-case differential: DuckDB's
# `avg(x) OVER ()` lands on different last bits when evaluated alone than
# beside this query's ordered `sum`. The carrier evaluates the original items,
# so the frozen value is the original's.
WITNESS = pa.table(
    {
        "k1": ["b", None, None, "c", "a", "b", "c", None, "a", None, "b", None],
        "k2": [1, None, 1, 1, 2, 1, None, 1, 2, 2, 2, 1],
        "x": [
            7.164686377126717,
            4.704282410335283,
            None,
            None,
            None,
            3.87249411114834,
            -7.3797585981390155,
            -5.592795861334189,
            None,
            None,
            None,
            -2.904195008746247,
        ],
        "y": pa.array(
            [None, None, 38, None, None, None, 17, 88, None, 79, 56, 79],
            type=pa.int64(),
        ),
    }
)
WITNESS_TEXT = (
    "SELECT sum(y) OVER (ORDER BY y RANGE BETWEEN 2 PRECEDING AND CURRENT ROW)"
    " AS e0, avg(x) OVER () AS e1 FROM __THIS__"
)


def test_the_carrier_keeps_the_original_reduction_order():
    original = _original(WITNESS_TEXT, WITNESS)
    # The value the full original query computes (DuckDB 1.5.5, threads=1);
    # the same average evaluated alone ends in ...c380 instead.
    assert {_bits(v) for v in original["e1"].to_pylist()} == {"bf9716c2aab4c355"}
    fitted = SQLProjection.marginalize(WITNESS_TEXT).fit(WITNESS)
    assert _exact(fitted.transform(WITNESS)) == _exact(original)


def test_an_unaliased_scope_keeps_duckdbs_own_output_name():
    text = "SELECT store, price - avg(price) OVER (PARTITION BY store) FROM __THIS__"
    _law(text, F)


def test_a_star_reads_the_batch_and_never_the_derived_params():
    text = "SELECT *, avg(price) OVER (PARTITION BY store) AS m FROM __THIS__"
    _law(text, F)
    fitted = SQLProjection.marginalize(text).fit(F)
    assert fitted.transform(X).column_names == ["store", "price", "m"]


@pytest.mark.parametrize(
    "text",
    [
        "SELECT *, avg(v) OVER () AS m FROM __THIS__",
        "SELECT t.*, avg(t.v) OVER () AS m FROM __THIS__ t",
        "SELECT *, avg(v) OVER (PARTITION BY g) AS m FROM __THIS__",
        "SELECT t.*, avg(t.v) OVER (PARTITION BY t.g) AS m FROM __THIS__ t",
    ],
)
def test_a_hostile_input_column_named_like_a_derived_one_survives(text):
    """The star hides the input's columns from the fresh-prefix scan, so an
    input column can wear a derived name; the qualified star keeps it."""
    H = pa.table({"g": ["a", "a"], "v": [10, 20], "cf_w0": [999, 999], "cf_k0": [7, 7]})
    fitted = SQLProjection.marginalize(text).fit(H)
    out = fitted.transform(H)
    assert out.column_names == ["g", "v", "cf_w0", "cf_k0", "m"]
    assert [(r["cf_w0"], r["m"]) for r in out.to_pylist()] == [(999, 15.0)] * 2
    assert _exact(out) == _exact(_original(text, H))
    assert fitted.compile().infer_rows(H.to_pylist()) == out.to_pylist()


def test_an_ordinary_lateral_alias_needs_no_fit_rewrite():
    text = (
        "SELECT price * 2 AS p2, p2 - avg(price) OVER (PARTITION BY store) AS d"
        " FROM __THIS__"
    )
    _law(text, F)
    fitted = SQLProjection.marginalize(text).fit(F)
    assert fitted.compile().infer_rows(X.to_pylist()) == fitted.transform(X).to_pylist()


def test_signed_zero_and_nan_cross_the_params_table_exactly():
    SZ = pa.table(
        {
            "g": ["a", "a", "b", "b", "c"],
            "v": [-0.0, -0.0, 0.0, 1.0, math.nan],
        }
    )
    text = (
        "SELECT g, min(v) OVER (PARTITION BY g) AS lo,"
        " max(v) OVER (PARTITION BY g) AS hi FROM __THIS__"
    )
    _law(text, SZ)
    out = SQLProjection.marginalize(text).fit(SZ).transform(SZ).to_pylist()
    assert _bits(out[0]["lo"]) == _bits(-0.0)


def test_a_collated_key_looks_up_its_own_value():
    """The window evaluates under the collation; the lookup key is the raw
    value, so each spelling the fit saw finds its partition's value."""
    N = pa.table({"name": ["a", "A", "b"], "v": [1.0, 2.0, 4.0]})
    text = (
        "SELECT name, sum(v) OVER (PARTITION BY name COLLATE nocase) AS s FROM __THIS__"
    )
    _law(text, N)


# --- the widened window vocabulary ------------------------------------------

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
    "whole_frame_with_order": (
        "SELECT store, sum(price) OVER (PARTITION BY store ORDER BY d"
        " ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS s"
        " FROM __THIS__"
    ),
    "rank": (
        "SELECT store, d, rank() OVER (PARTITION BY store ORDER BY d) AS r"
        " FROM __THIS__"
    ),
    "dense_rank": (
        "SELECT store, d, dense_rank() OVER (PARTITION BY store"
        " ORDER BY d DESC NULLS FIRST) AS r FROM __THIS__"
    ),
    "percent_rank": (
        "SELECT store, d, percent_rank() OVER (PARTITION BY store ORDER BY d)"
        " AS r FROM __THIS__"
    ),
    "cume_dist": "SELECT d, cume_dist() OVER (ORDER BY d) AS r FROM __THIS__",
    "rank_without_order": (
        "SELECT store, rank() OVER (PARTITION BY store) AS r FROM __THIS__"
    ),
    "first_value": (
        "SELECT store, first_value(price) OVER (PARTITION BY store ORDER BY price)"
        " AS f FROM __THIS__"
    ),
    "first_value_ignore_nulls": (
        "SELECT store, price, first_value(d IGNORE NULLS) OVER (PARTITION BY store"
        " ORDER BY price DESC) AS f FROM __THIS__"
    ),
    "last_value": (
        "SELECT store, price, last_value(price) OVER (PARTITION BY store"
        " ORDER BY price) AS l FROM __THIS__"
    ),
    "last_value_whole": (
        "SELECT store, last_value(price) OVER (PARTITION BY store ORDER BY price"
        " ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS l"
        " FROM __THIS__"
    ),
    "nth_value": (
        "SELECT store, nth_value(price, 2) OVER (PARTITION BY store ORDER BY price"
        " RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS n"
        " FROM __THIS__"
    ),
    "ordered_string_agg": (
        "SELECT store, string_agg(CAST(price AS VARCHAR), ',' ORDER BY price DESC)"
        " OVER (PARTITION BY store) AS s FROM __THIS__"
    ),
    "ordered_list": (
        "SELECT store, list(price ORDER BY price) OVER (PARTITION BY store) AS l"
        " FROM __THIS__"
    ),
    "named_window": (
        "SELECT store, price - avg(price) OVER w AS dv, max(price) OVER w AS mx"
        " FROM __THIS__ WINDOW w AS (PARTITION BY store)"
    ),
    "filter_and_distinct": (
        "SELECT store, count(DISTINCT d) FILTER (WHERE price > 8)"
        " OVER (PARTITION BY store) AS n FROM __THIS__"
    ),
}


@pytest.mark.parametrize("text", ORDERED_LAWFUL.values(), ids=ORDERED_LAWFUL.keys())
def test_the_law_holds_for_the_window_vocabulary(text):
    """Every admitted window's value is a function of its lookup keys — the
    partitions, plus the order values when the frame moves with them. Ties
    and a NULL order value are in the fixture."""
    _law(text, ORD)


def test_an_unseen_order_value_is_a_miss():
    """Frozen keys are the fit data's (store, d) pairs: a new d is NULL where
    true RANGE semantics would refit — the same divergence-only-at-misses."""
    fitted = SQLProjection.marginalize(ORDERED_LAWFUL["cumulative"]).fit(ORD)
    X3 = pa.table({"store": ["S1", "S1"], "d": [2, 3], "price": [1.0, 1.0]})
    assert fitted.transform(X3).to_pylist() == [
        {"store": "S1", "d": 2, "s": 60.0},  # frozen cumulative at d=2
        {"store": "S1", "d": 3, "s": None},  # unseen order value
    ]


@pytest.mark.parametrize("name", ["cumulative", "rank", "first_value"])
def test_an_ordered_scope_serves(name):
    fitted = SQLProjection.marginalize(ORDERED_LAWFUL[name]).fit(ORD)
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
    "exists": (
        "SELECT store, EXISTS (SELECT 1 FROM __THIS__ WHERE price > 250) AS big"
        " FROM __THIS__"
    ),
    "not_exists_with_clauses": (
        "SELECT store, NOT EXISTS (SELECT 1 FROM __THIS__ i WHERE i.price < 0"
        " ORDER BY i.price LIMIT 1) AS clean FROM __THIS__"
    ),
    "exists_beside_a_window": (
        "SELECT store, CASE WHEN EXISTS (SELECT 1 FROM __THIS__ WHERE store IS NULL)"
        " THEN price - avg(price) OVER (PARTITION BY store) END AS d FROM __THIS__"
    ),
}


@pytest.mark.parametrize("text", SUBQUERY_LAWFUL.values(), ids=SUBQUERY_LAWFUL.keys())
def test_the_law_holds_for_uncorrelated_subqueries(text):
    """An uncorrelated single-level subquery over __THIS__ freezes verbatim
    over __FIT__ — one value, joined one-row."""
    _law(text, F)


def test_an_exists_answer_is_frozen_at_fit():
    """The fit data has a price above 250 and the request does not: the
    frozen EXISTS answers for the fit data on every request row."""
    fitted = SQLProjection.marginalize(SUBQUERY_LAWFUL["exists"]).fit(F)
    assert [r["big"] for r in fitted.transform(X).to_pylist()] == [True] * 4


def test_a_frozen_subquery_serves():
    fitted = SQLProjection.marginalize(SUBQUERY_LAWFUL["global_max"]).fit(F)
    rows = fitted.compile().infer_rows(X.to_pylist())
    assert rows == fitted.transform(X).to_pylist()


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


def test_key_composition_beside_a_carried_window():
    """The keyed scope keeps its grouped composition; the window beside it is
    carried — the two lowerings in one text, each answering as it does
    alone."""
    text = """
        SELECT city, store,
               keyed_transform(
                   keyed_fit(struct_pack(store := store, price := price))
                       OVER (PARTITION BY city),
                   struct_pack(store := store, price := price)).r AS r,
               price - avg(price) OVER (PARTITION BY city) AS d
        FROM __THIS__
    """
    out = SQLProjection.marginalize(text).fit(CITY).transform(CITY).to_pylist()
    alone = SQLProjection.marginalize(KEYED_TEXT).fit(CITY).transform(CITY)
    assert [r["r"] for r in out] == alone["r"].to_pylist()
    plain = "SELECT price - avg(price) OVER (PARTITION BY city) AS d FROM __THIS__"
    expected = _original(plain, CITY)["d"].to_pylist()
    assert sorted(_bits(r["d"]) for r in out) == sorted(_bits(d) for d in expected)


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
    (
        "explicitly over __FIT__ and __THIS__",
        "WITH c AS (SELECT price * 2 AS p FROM __THIS__)"
        " SELECT p - avg(p) OVER () AS d FROM c",
    ),
    (
        "explicitly over __FIT__ and __THIS__",
        "SELECT p - avg(p) OVER () AS d FROM (SELECT price * 2 AS p FROM __THIS__) s",
    ),
    ("subquery", "SELECT (SELECT 1) AS one, price FROM __THIS__"),
    ("subquery", "SELECT EXISTS (SELECT 1 FROM codes) AS e, price FROM __THIS__"),
    ("ANY", "SELECT price IN (SELECT price FROM __THIS__) AS hit FROM __THIS__"),
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
        "non-constant",
        "SELECT sum(price) OVER (ORDER BY d RANGE BETWEEN d PRECEDING"
        " AND CURRENT ROW) AS s FROM __THIS__",
    ),
    (
        "correlat",
        "SELECT (SELECT max(i.price) FROM __THIS__ i WHERE i.store = t.store) AS m"
        " FROM __THIS__ t",
    ),
    (
        "correlat",
        "SELECT EXISTS (SELECT 1 FROM __THIS__ i WHERE i.price > t.price) AS e"
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
    ("positional", "SELECT ntile(2) OVER (ORDER BY price) AS n FROM __THIS__"),
    ("positional", "SELECT lag(price) OVER (ORDER BY price) AS p FROM __THIS__"),
    ("positional", "SELECT lead(price, 1, 0) OVER (ORDER BY price) AS p FROM __THIS__"),
    (
        "nth_value",
        "SELECT nth_value(price, d) OVER (ORDER BY price) AS n FROM __THIS__",
    ),
    (
        "nth_value",
        "SELECT nth_value(price, 0) OVER (ORDER BY price) AS n FROM __THIS__",
    ),
    (
        "frame",
        "SELECT avg(price) OVER (ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS a"
        " FROM __THIS__",
    ),
    ("COLUMNS", "SELECT COLUMNS('p.*'), avg(price) OVER () AS m FROM __THIS__"),
    ("EXCLUDE", "SELECT * EXCLUDE (store), avg(price) OVER () AS m FROM __THIS__"),
    ("RENAME", "SELECT * RENAME (store AS s), avg(price) OVER () AS m FROM __THIS__"),
    (
        r"\* inside an expression",
        "SELECT [*COLUMNS(*)] AS l, avg(price) OVER () AS m FROM __THIS__",
    ),
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
    # One fit scope, applied inline twice with two bundles: the replacement
    # for a θ parked in a lateral alias.
    "one_fit_two_applies": """
        SELECT store,
               zscore_transform(
                   zscore_fit(struct_pack(price := price)) OVER (PARTITION BY store),
                   struct_pack(price := price)).z AS z,
               zscore_transform(
                   zscore_fit(struct_pack(price := price)) OVER (PARTITION BY store),
                   struct_pack(price := price * 2)).z AS z2
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


def test_projection_theta_misses_are_null():
    fitted = SQLProjection.marginalize(PROJECTION_LAWFUL["split_per_key"]).fit(F)
    by = {r["store"]: r["z"] for r in fitted.transform(X).to_pylist()}
    assert by["NEW"] is None  # P14, through the leaf: NULL θ in, NULL out


@pytest.mark.parametrize(
    "text", PROJECTION_LAWFUL.values(), ids=PROJECTION_LAWFUL.keys()
)
def test_a_projection_scope_serves_on_the_row_path(text):
    """The frozen θ crosses the derived join as a struct, and the residual
    reads it with struct_extract; the compiled row path answers exactly what
    batch does."""
    fitted = SQLProjection.marginalize(text).fit(F)
    rows = fitted.compile().infer_rows(X.to_pylist())
    assert rows == fitted.transform(X).to_pylist()


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
    # a θ parked in a lateral alias: apply it inline, or write the fit
    # explicitly over __FIT__
    (
        r"inline.*explicit CTE",
        """SELECT store,
               zscore_fit(struct_pack(price := price))
                   OVER (PARTITION BY store) AS t,
               zscore_transform(t, struct_pack(price := price)).z AS z
           FROM __THIS__""",
    ),
]


@pytest.mark.parametrize(
    ("token", "text"), PROJECTION_REFUSED, ids=[t for t, _ in PROJECTION_REFUSED]
)
def test_projection_scope_refusals_in_the_authors_vocabulary(token, text):
    with pytest.raises(TransformError, match=token):
        SQLProjection.marginalize(text)
