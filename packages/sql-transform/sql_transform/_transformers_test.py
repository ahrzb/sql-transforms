"""Transformers and UDFs as in-place scalar calls (DRAFT-22, step 1).

A transformer window rewrites in place to ``__cf_tf{j}(id, features...)``
over its own params join, so mid-expression use works; a width-1 transform
is scalar-valued, width-k yields a list column. The gate's oracle for
transformer columns is an independent clone/fit/transform reference; author
UDF queries round-trip exactly (the same object runs on both sides).
"""

import duckdb
import numpy as np
import pyarrow as pa
import pytest
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sql_transform import MarginalizeError, PythonUDF, SQLProjection, UDFError

TRAIN = pa.table(
    {
        "country": ["US", "US", None, "DE", None, "DE", "FR"],
        "age": [40.0, 30.0, 20.0, 25.0, 50.0, 45.0, 35.0],
        "fare": [7.0, 8.0, 6.0, 9.0, 10.0, 11.0, 5.0],
        "name": ["x", "y", "z", "w", "v", "u", "t"],
    }
)


def _reference(proto, feats, keys):
    """Independent oracle: clone-per-group fit_transform, row-aligned."""
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    out = [None] * len(keys)
    for _k, idx in groups.items():
        est = clone(proto)
        block = np.asarray(est.fit(feats[idx]).transform(feats[idx]))
        if block.ndim == 1:
            block = block.reshape(-1, 1)
        for row, vals in zip(idx, block, strict=True):
            out[row] = [float(v) for v in vals]
    return out


def _by_name(table, value_col):
    names = table.column("name").to_pylist()
    vals = table.column(value_col).to_pylist()
    return dict(zip(names, vals, strict=True))


def test_global_scaler_from_scope_is_scalar_valued():
    sc = StandardScaler()
    p = SQLProjection("SELECT sc(age) OVER () AS z, name FROM __THIS__").fit(TRAIN)
    (step,) = [s for s in p.plan if s.kind == "fit"]
    assert step.transformer == "sc" and step.keys == ()
    assert p.udfs["__cf_tf0"].returns == ("f64",)
    got = _by_name(p.transform(TRAIN), "z")
    assert all(isinstance(v, float) for v in got.values())  # width 1 -> DOUBLE
    feats = np.array([TRAIN.column("age").to_pylist()], dtype=float).T
    ref = _reference(sc, feats, [()] * TRAIN.num_rows)
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i][0], rtol=1e-12)


def test_transformer_mid_expression():
    sc = StandardScaler()
    p = SQLProjection(
        "SELECT sc(age) OVER (PARTITION BY country) * 10 + 1 AS z, name FROM __THIS__",
        transformers={"sc": sc},
    ).fit(TRAIN)
    got = _by_name(p.transform(TRAIN), "z")
    feats = np.array([TRAIN.column("age").to_pylist()], dtype=float).T
    ref = _reference(sc, feats, [(c,) for c in TRAIN.column("country").to_pylist()])
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i][0] * 10 + 1, rtol=1e-12)


def test_pipeline_pca2_with_struct_bundle_yields_list():
    embed = Pipeline([("s", StandardScaler()), ("p", PCA(n_components=2))])
    p = SQLProjection(
        "SELECT embed(struct_pack(a := age, f := fare)) OVER () AS e, name"
        " FROM __THIS__",
        transformers={"embed": embed},
    ).fit(TRAIN)
    assert p.udfs["__cf_tf0"].returns == ("f64", "f64")
    got = _by_name(p.transform(TRAIN), "e")
    feats = np.array(
        [TRAIN.column("age").to_pylist(), TRAIN.column("fare").to_pylist()],
        dtype=float,
    ).T
    ref = _reference(embed, feats, [()] * TRAIN.num_rows)
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i], rtol=1e-9)


def test_per_country_pipeline_pca1_with_struct_bundle():
    embed = Pipeline([("s", StandardScaler()), ("p", PCA(n_components=1))])
    p = SQLProjection(
        "SELECT embed(struct_pack(a := age, f := fare))"
        " OVER (PARTITION BY country) AS e, name FROM __THIS__",
        transformers={"embed": embed},
    ).fit(TRAIN)
    got = _by_name(p.transform(TRAIN), "e")
    feats = np.array(
        [TRAIN.column("age").to_pylist(), TRAIN.column("fare").to_pylist()],
        dtype=float,
    ).T
    ref = _reference(embed, feats, [(c,) for c in TRAIN.column("country").to_pylist()])
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i][0], rtol=1e-9)


def test_params_table_carries_instance_ids():
    sc = StandardScaler()
    p = SQLProjection(
        "SELECT sc(age) OVER (PARTITION BY country) AS z FROM __THIS__",
        transformers={"sc": sc},
    ).fit(TRAIN)
    (params,) = p.params.values()
    assert "__cf_est" in params.column_names
    countries = params.column("country").to_pylist()
    assert None in countries  # NULL partition key is a real group
    ids = params.column("__cf_est").to_pylist()
    assert sorted(ids) == list(range(len(set(countries))))


def test_unseen_group_gets_null():
    sc = StandardScaler()
    p = SQLProjection(
        "SELECT sc(age) OVER (PARTITION BY country) AS z, name FROM __THIS__",
        transformers={"sc": sc},
    ).fit(TRAIN)
    new = pa.table({"country": ["JP"], "age": [1.0], "fare": [1.0], "name": ["q"]})
    assert p.transform(new).column("z").to_pylist() == [None]


def test_mixed_sql_and_transformer_columns():
    sc = StandardScaler()
    p = SQLProjection(
        "SELECT age - avg(age) OVER () AS c, sc(fare) OVER () AS z, name FROM __THIS__",
        transformers={"sc": sc},
    ).fit(TRAIN)
    out = p.transform(TRAIN)
    mean = np.mean(TRAIN.column("age").to_pylist())
    got = _by_name(out, "c")
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], TRAIN.column("age")[i].as_py() - mean)
    assert out.column_names == ["c", "z", "name"]


def test_serving_sql_calls_in_place():
    sc = StandardScaler()
    p = SQLProjection(
        "SELECT sc(age) OVER (PARTITION BY country) + 1 AS z FROM __THIS__",
        transformers={"sc": sc},
    )
    assert "__cf_tf0(__cf_p0.__cf_est, __cf_t.age)" in p.serving_sql
    assert "IS NOT DISTINCT FROM" in p.serving_sql
    (spec,) = p._marginalized.udfs
    assert (spec.name, spec.step) == ("__cf_tf0", "__CF_PARAMS_0__")
    names = [s.name for s in p.plan]
    assert names == ["__CF_LEVEL_0__", "__CF_PARAMS_0__"]
    assert p.plan[1].kind == "fit"


def test_transformer_at_final_level_of_chain_with_renamed_feature():
    sc = StandardScaler()
    p = SQLProjection(
        "WITH a AS (SELECT age * 2 AS age2, country, name FROM __THIS__)"
        " SELECT sc(age2) OVER (PARTITION BY country) - 1 AS z, name FROM a",
        transformers={"sc": sc},
    ).fit(TRAIN)
    assert "__cf_tf0(__cf_p0.__cf_est, (__cf_t.age * 2))" in p.serving_sql
    assert [(s.name, s.kind) for s in p.plan] == [
        ("__CF_LEVEL_0__", "sql"),
        ("__CF_LEVEL_1__", "sql"),
        ("__CF_PARAMS_0__", "fit"),
    ]
    got = _by_name(p.transform(TRAIN), "z")
    feats = np.array([TRAIN.column("age").to_pylist()], dtype=float).T * 2
    ref = _reference(sc, feats, [(c,) for c in TRAIN.column("country").to_pylist()])
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i][0] - 1, rtol=1e-12)


# --- author UDFs ---------------------------------------------------------------


def test_author_udf_round_trips_exactly():
    halve = PythonUDF(
        "halve", lambda x: None if x is None else x / 2.0, ("f64",), ("f64",)
    )
    p = SQLProjection(
        "SELECT halve(age) - avg(halve(age)) OVER () AS d, name FROM __THIS__",
        transformers={"halve": halve},
    ).fit(TRAIN)
    out = p.transform(TRAIN)
    con = duckdb.connect()
    try:
        con.execute("SET threads = 1")
        con.register("__THIS__", TRAIN)
        halve.register(con)
        orig = con.execute(
            "SELECT halve(age) - avg(halve(age)) OVER () AS d, name FROM __THIS__"
        ).to_arrow_table()
    finally:
        con.close()
    assert out.equals(orig)


def test_author_udf_from_scope_in_identity_projection():
    shout = PythonUDF(
        "shout", lambda s: None if s is None else s.upper(), ("str",), ("str",)
    )
    assert shout is not None  # resolved from this scope by name
    p = SQLProjection("SELECT shout(name) AS n FROM __THIS__").fit(TRAIN)
    assert p._marginalized.scalar_udfs == ("shout",)
    assert p.transform(TRAIN).column("n").to_pylist() == [
        "X",
        "Y",
        "Z",
        "W",
        "V",
        "U",
        "T",
    ]


def test_author_udf_inside_transformer_bundle():
    halve = PythonUDF(
        "halve", lambda x: None if x is None else x / 2.0, ("f64",), ("f64",)
    )
    sc = StandardScaler()
    p = SQLProjection(
        "SELECT sc(struct_pack(h := halve(age))) OVER () AS z, name FROM __THIS__",
        transformers={"halve": halve, "sc": sc},
    ).fit(TRAIN)
    assert p._marginalized.scalar_udfs == ("halve",)
    got = _by_name(p.transform(TRAIN), "z")
    feats = np.array([TRAIN.column("age").to_pylist()], dtype=float).T / 2
    ref = _reference(sc, feats, [()] * TRAIN.num_rows)
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i][0], rtol=1e-12)


def test_registry_beats_scope():
    sc = StandardScaler()  # noqa: F841 — the scope decoy
    p = SQLProjection(
        "SELECT sc(age) OVER () AS z FROM __THIS__",
        transformers={"sc": Pipeline([("s", StandardScaler())])},
    )
    assert isinstance(p.transformers["sc"], Pipeline)


def test_python_transform_missing_id_raises():
    from sql_transform import PythonTransform

    t = PythonTransform("t", instances={}, takes=("f64",), returns=("f64",))
    assert t(None, 1.0) is None
    with pytest.raises(UDFError, match="different fits"):
        t(0, 1.0)


def test_udf_declaration_violation_traps():
    liar = PythonUDF("liar", lambda x: "not a float", ("f64",), ("f64",))
    p = SQLProjection(
        "SELECT liar(age) AS z FROM __THIS__", transformers={"liar": liar}
    ).fit(TRAIN)
    with pytest.raises(Exception, match="(?i)convert|conversion|cast"):
        p.transform(TRAIN)


@pytest.mark.parametrize(
    "sql,match",
    [
        ("SELECT nope(age) OVER () FROM __THIS__", "unknown window function nope"),
        (
            "SELECT sc(age) OVER (ORDER BY age) FROM __THIS__",
            "ORDER BY on a transformer",
        ),
        ("SELECT sc(*) OVER () FROM __THIS__", "zero arguments"),
        ("SELECT sc(age, fare) OVER () FROM __THIS__", "exactly one bundle"),
        ("SELECT sc(struct_pack(age)) OVER () FROM __THIS__", "must be named"),
        (
            "WITH a AS (SELECT sc(age) OVER () AS z FROM __THIS__) SELECT z FROM a",
            "non-final level",
        ),
        ("SELECT sk.sc(age) OVER () FROM __THIS__", "namespaced transformer"),
        (
            "SELECT avg(sc(age) OVER ()) OVER () FROM __THIS__",
            "inside a window aggregate",
        ),
        ("SELECT sc(age) FROM __THIS__", "without OVER"),
        ("SELECT mystery(age) FROM __THIS__", "unknown function mystery"),
        (
            "SELECT (SELECT max(mystery(age)) FROM __THIS__) FROM __THIS__",
            "inside a subquery",
        ),
    ],
)
def test_refusals(sql, match):
    sc = StandardScaler()  # noqa: F841 — resolved from scope where relevant
    with pytest.raises(MarginalizeError, match=match):
        SQLProjection(sql)


def test_udf_arity_mismatch_refuses():
    one = PythonUDF("one", lambda x: x, ("f64",), ("f64",))
    with pytest.raises(MarginalizeError, match="declares 1 arguments, called with 2"):
        SQLProjection("SELECT one(age, fare) FROM __THIS__", transformers={"one": one})


def test_udf_name_mismatch_refuses():
    other = PythonUDF("something_else", lambda x: x, ("f64",), ("f64",))
    with pytest.raises(MarginalizeError, match="resolves to an object named"):
        SQLProjection("SELECT one(age) FROM __THIS__", transformers={"one": other})
