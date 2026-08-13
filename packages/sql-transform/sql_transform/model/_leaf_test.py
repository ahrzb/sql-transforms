"""A projection as a leaf: both halves spliced as SQL, θ as data (D1, D2).

``p_fit(bundle)`` becomes a struct of window/grouped aggregates — θ carries
the parameters, not a pointer — and ``p_transform(θ, bundle)`` becomes the
residual's expressions over struct reads. Nothing is registered: no Python in
the row path, and the artifact's size is visible in the params table.

Gates ``leaf`` and ``capture`` of
`docs/superpowers/specs/2026-08-11-row-wise-projections-design.md`.
"""

import pyarrow as pa
import pytest

from sql_transform.model import SQLProjection, SQLTransform, TransformError, run

F = pa.table(
    {
        "store": ["S1", "S1", "S1", "S2", "S2", "S2"],
        "price": [10.0, 20.0, 30.0, 100.0, 300.0, 500.0],
    }
)
X = pa.table(
    {
        "store": ["S2", "NEW", "S1"],
        "price": [200.0, 7.0, 10.0],
    }
)

# The leaf under test: a global z-score. Its own params are one row, so a
# host window or GROUP BY decides the fit scope — that is the whole point.
zscore = SQLProjection("""
    SELECT round((t.price - f.m) / f.s, 4) AS z
    FROM __THIS__ t,
         (SELECT avg(price) AS m, stddev_pop(price) AS s FROM __FIT__) f
""")

PER_STORE_HOST = """
    SELECT t.store,
           zscore_transform(f.theta, struct_pack(price := t.price)).z AS z
    FROM __THIS__ t
    LEFT JOIN (SELECT store, zscore_fit(struct_pack(price := price)) AS theta
               FROM __FIT__ GROUP BY store) f
      ON t.store = f.store
    ORDER BY t.store
"""


def test_leaf_per_group_equals_standalone_per_group():
    """The leaf gate: θ crosses the join as a value, and each group's answer
    is the same projection fitted standalone on that group's rows."""
    host = SQLTransform(PER_STORE_HOST)
    out = host.fit(F).transform(X).to_pylist()

    def standalone(store: str, price: float) -> float:
        group = F.filter(pa.compute.equal(F["store"], store))
        one = pa.table({"store": [store], "price": [price]})
        return zscore.fit(group).transform(one).to_pylist()[0]["z"]

    assert out == [
        {"store": "NEW", "z": None},  # θ is a join miss: NULL in, NULL out
        {"store": "S1", "z": standalone("S1", 10.0)},
        {"store": "S2", "z": standalone("S2", 200.0)},
    ]


def test_theta_is_data_in_the_params_table():
    """D1, observable: the artifact holds the parameters themselves, one
    struct per group, not a pointer into a registry of Python objects."""
    host = SQLTransform(PER_STORE_HOST)
    fitted = host.fit(F)
    (params,) = fitted.params.values()
    thetas = {r["store"]: r["theta"] for r in params.to_pylist()}
    assert thetas["S1"]["__param_0"]["m"] == 20.0
    assert thetas["S2"]["__param_0"]["m"] == 300.0
    assert fitted.instances == {}  # no registry: there is nothing to point at


def test_transductive_window_fit():
    """`p_fit(...) OVER (PARTITION BY k)` refits per partition, inline."""
    t = SQLTransform("""
        SELECT k,
               zscore_transform(
                   zscore_fit(struct_pack(price := v)) OVER (PARTITION BY k),
                   struct_pack(price := v)).z AS z
        FROM __THIS__ ORDER BY k, z
    """)
    d = pa.table({"k": ["a", "a", "a", "b", "b"], "v": [1.0, 2.0, 3.0, 10.0, 30.0]})
    assert run(t, d).to_pylist() == [
        {"k": "a", "z": -1.2247},
        {"k": "a", "z": 0.0},
        {"k": "a", "z": 1.2247},
        {"k": "b", "z": -1.0},
        {"k": "b", "z": 1.0},
    ]


def test_bare_call_is_the_global_split():
    """The ONE sugar: `tfm(x)` ≡ `tfm_transform(tfm_fit(x) OVER (), x)`."""
    sugar = SQLTransform("""
        SELECT store, zscore(struct_pack(price := price)).z AS z
        FROM __THIS__ ORDER BY store, z
    """)
    split = SQLTransform("""
        SELECT store, zscore_transform(
                   zscore_fit(struct_pack(price := price)) OVER (),
                   struct_pack(price := price)).z AS z
        FROM __THIS__ ORDER BY store, z
    """)
    assert run(sugar, F).to_pylist() == run(split, F).to_pylist()


def test_the_deleted_over_sugar_refuses_by_name():
    """`tfm(x) OVER w` has no oracle reading (fit-transform-split spec) and
    stays deleted: refused at construction, pointing at the split spelling —
    not a DuckDB unknown-function error at fit time."""
    with pytest.raises(TransformError, match=r"zscore.*zscore_fit"):
        SQLTransform("""
            SELECT zscore(struct_pack(price := v)) OVER (PARTITION BY k) AS s
            FROM __THIS__
        """)


def test_host_names_do_not_capture_the_leafs_names():
    """The capture gate: the host reuses the leaf's own internal aliases
    (t, f) and its column name (m) — the spliced text still reads the
    leaf's values, because nothing of the leaf's names survives the splice."""
    host = SQLTransform("""
        SELECT t.store,
               zscore_transform(f.theta, struct_pack(price := t.m)).z AS z
        FROM (SELECT store, price AS m FROM __THIS__) t
        LEFT JOIN (SELECT store, zscore_fit(struct_pack(price := price)) AS theta
                   FROM __FIT__ GROUP BY store) f
          ON t.store = f.store
        ORDER BY t.store
    """)
    out = host.fit(F).transform(X).to_pylist()
    assert [r["z"] for r in out] == [None, -1.2247, -0.6124]


def test_the_bundle_must_supply_what_the_leaf_reads():
    with pytest.raises(TransformError, match=r"zscore.*price"):
        SQLTransform("""
            SELECT zscore_transform(
                zscore_fit(struct_pack(cost := v)) OVER (),
                struct_pack(cost := v)).z AS z
            FROM __THIS__
        """)


def test_leaf_refusals_are_attributed_to_the_projection_by_name():
    """The attribution gate (D3). After the splice there is one merged text,
    so whatever cannot splice must say which projection and why *before*
    merging — every leaf refusal opens with the projection's own name."""
    unqual = SQLProjection("SELECT price - 1 AS d FROM __THIS__")
    filtered = SQLProjection("""
        SELECT t.price / f.m AS z
        FROM __THIS__ t,
             (SELECT avg(price) AS m FROM __FIT__ WHERE price > 0) f
    """)
    assert (unqual, filtered) is not None
    hosts = {
        "unqual": """
            SELECT unqual_transform(
                unqual_fit(struct_pack(price := v)) OVER (),
                struct_pack(price := v)).d AS d
            FROM __THIS__
        """,
        "filtered": """
            SELECT filtered_transform(
                filtered_fit(struct_pack(price := v)) OVER (),
                struct_pack(price := v)).z AS z
            FROM __THIS__
        """,
    }
    for stem, sql in hosts.items():
        with pytest.raises(
            TransformError, match=rf"{stem} is a projection used as a leaf"
        ):
            SQLTransform(sql)


def test_a_keyed_projection_refuses_the_leaf_role_by_name():
    """A leaf fits per scope; keys inside it would need θ to carry tables.
    Refused with the projection's name, at the host's construction."""
    keyed = SQLProjection("""
        SELECT t.price / f.m AS r
        FROM __THIS__ t
        LEFT JOIN (SELECT store, avg(price) AS m FROM __FIT__ GROUP BY store) f
          ON t.store = f.store
    """)
    assert keyed is not None
    with pytest.raises(TransformError, match=r"keyed"):
        SQLTransform("""
            SELECT keyed_transform(
                keyed_fit(struct_pack(store := store, price := price)) OVER (),
                struct_pack(store := store, price := price)).r AS r
            FROM __THIS__
        """)
