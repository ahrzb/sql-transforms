"""Typed estimator and scalar-UDF projections, checked against independent fits."""

import duckdb
import numpy as np
import pyarrow as pa
import pytest
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sql_transform import PythonUDF, SQLProjection, TransformError, UDFError

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
    p = SQLProjection.marginalize("SELECT sc(age).age AS z, name FROM __THIS__").fit(
        TRAIN
    )
    (udf,) = p.udfs.values()
    assert udf.returns == pa.struct([("age", pa.float64())])
    assert len(p.instances) == 1
    got = _by_name(p.transform(TRAIN), "z")
    assert all(isinstance(v, float) for v in got.values())
    feats = np.array([TRAIN.column("age").to_pylist()], dtype=float).T
    ref = _reference(sc, feats, [()] * TRAIN.num_rows)
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i][0], rtol=1e-12)


def test_transformer_mid_expression():
    sc = StandardScaler()
    p = SQLProjection.marginalize(
        "SELECT sc_transform(sc_fit(age) OVER (PARTITION BY country), age).age"
        " * 10 + 1 AS z, name FROM __THIS__",
        captured={"sc": sc},
    ).fit(TRAIN)
    got = _by_name(p.transform(TRAIN), "z")
    feats = np.array([TRAIN.column("age").to_pylist()], dtype=float).T
    ref = _reference(sc, feats, [(c,) for c in TRAIN.column("country").to_pylist()])
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i][0] * 10 + 1, rtol=1e-12)


def test_pipeline_pca2_with_struct_bundle_two_field_reads():
    embed = Pipeline([("s", StandardScaler()), ("p", PCA(n_components=2))])
    p = SQLProjection.marginalize(
        "SELECT embed(struct_pack(a := age, f := fare)).pca0 AS e_pca0,"
        " embed(struct_pack(a := age, f := fare)).pca1 AS e_pca1, name"
        " FROM __THIS__",
        captured={"embed": embed},
    ).fit(TRAIN)
    out = p.transform(TRAIN)
    assert out.column_names == ["e_pca0", "e_pca1", "name"]
    got = {
        n: (a, b)
        for n, a, b in zip(
            out.column("name").to_pylist(),
            out.column("e_pca0").to_pylist(),
            out.column("e_pca1").to_pylist(),
            strict=True,
        )
    }
    feats = np.array(
        [TRAIN.column("age").to_pylist(), TRAIN.column("fare").to_pylist()],
        dtype=float,
    ).T
    ref = _reference(embed, feats, [()] * TRAIN.num_rows)
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i], rtol=1e-9)


def test_per_country_pipeline_pca1_with_struct_bundle():
    embed = Pipeline([("s", StandardScaler()), ("p", PCA(n_components=1))])
    p = SQLProjection.marginalize(
        "SELECT embed_transform(embed_fit(struct_pack(a := age, f := fare))"
        " OVER (PARTITION BY country), struct_pack(a := age, f := fare)).pca0"
        " AS e, name FROM __THIS__",
        captured={"embed": embed},
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
        "WITH p AS (SELECT country, sc_fit(age) AS iid FROM __FIT__ GROUP BY country)"
        " SELECT sc_transform(p.iid, t.age).age AS z FROM __THIS__ t"
        " LEFT JOIN p ON t.country IS NOT DISTINCT FROM p.country",
        captured={"sc": sc},
    ).fit(TRAIN)
    (params,) = p.params.values()
    assert params.schema.field("iid").type == pa.int64()
    countries = params.column("country").to_pylist()
    assert None in countries
    ids = params.column("iid").to_pylist()
    assert set(ids) == set(p.instances)
    assert len(ids) == len(set(countries))


def test_unseen_group_gets_null():
    sc = StandardScaler()
    p = SQLProjection.marginalize(
        "SELECT sc_transform(sc_fit(age) OVER (PARTITION BY country), age).age"
        " AS z, name FROM __THIS__",
        captured={"sc": sc},
    ).fit(TRAIN)
    new = pa.table({"country": ["JP"], "age": [1.0], "fare": [1.0], "name": ["q"]})
    assert p.transform(new).column("z").to_pylist() == [None]


def test_mixed_sql_and_transformer_columns():
    sc = StandardScaler()
    p = SQLProjection.marginalize(
        "SELECT age - avg(age) OVER () AS c, sc(fare).fare AS z, name FROM __THIS__",
        captured={"sc": sc},
    ).fit(TRAIN)
    out = p.transform(TRAIN)
    mean = np.mean(TRAIN.column("age").to_pylist())
    got = _by_name(out, "c")
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], TRAIN.column("age")[i].as_py() - mean)
    assert out.column_names == ["c", "z", "name"]


def test_serving_artifact_calls_in_place():
    sc = StandardScaler()
    p = SQLProjection.marginalize(
        "SELECT sc_transform(sc_fit(age) OVER (PARTITION BY country), age).age"
        " + 1 AS z FROM __THIS__",
        captured={"sc": sc},
    ).fit(TRAIN)
    expected = p.transform(TRAIN).to_pylist()
    compiled = p.compile()
    assert compiled.infer_rows(TRAIN.to_pylist()) == expected
    assert compiled.infer_arrow(TRAIN).to_pylist() == expected


def test_transformer_at_final_level_of_chain_with_renamed_feature():
    sc = StandardScaler()
    # Migrated chain: retain its original projected fit source, not serving joins.
    p = SQLProjection(
        "WITH p AS (SELECT country, sc_fit(age2) AS iid"
        " FROM (SELECT age * 2 AS age2, country, name FROM __FIT__) a"
        " GROUP BY country)"
        " SELECT sc_transform(p.iid, struct_pack(age2 := t.age * 2)).age2 - 1 AS z,"
        " t.name FROM __THIS__ t"
        " LEFT JOIN p ON t.country IS NOT DISTINCT FROM p.country",
        captured={"sc": sc},
    ).fit(TRAIN)
    got = _by_name(p.transform(TRAIN), "z")
    feats = np.array([TRAIN.column("age").to_pylist()], dtype=float).T * 2
    ref = _reference(sc, feats, [(c,) for c in TRAIN.column("country").to_pylist()])
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i][0] - 1, rtol=1e-12)


# --- author UDFs ---------------------------------------------------------------


def test_author_udf_round_trips_exactly():
    halve = PythonUDF(
        "halve",
        lambda x: None if x is None else x / 2.0,
        pa.schema([("x", pa.float64())]),
    )
    p = SQLProjection.marginalize(
        "SELECT halve(age) - avg(halve(age)) OVER () AS d, name FROM __THIS__",
        captured={"halve": halve},
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
        "shout",
        lambda s: None if s is None else s.upper(),
        pa.schema([("s", pa.string())]),
        pa.string(),
    )
    assert shout is not None  # resolved from this scope by name
    p = SQLProjection.marginalize("SELECT shout(name) AS n FROM __THIS__").fit(TRAIN)
    assert p.udfs["shout"] is shout
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
        "halve",
        lambda x: None if x is None else x / 2.0,
        pa.schema([("x", pa.float64())]),
    )
    sc = StandardScaler()
    p = SQLProjection.marginalize(
        "SELECT sc(struct_pack(h := halve(age))).h AS z, name FROM __THIS__",
        captured={"halve": halve, "sc": sc},
    ).fit(TRAIN)
    assert p.udfs["halve"] is halve
    got = _by_name(p.transform(TRAIN), "z")
    feats = np.array([TRAIN.column("age").to_pylist()], dtype=float).T / 2
    ref = _reference(sc, feats, [()] * TRAIN.num_rows)
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i][0], rtol=1e-12)


def test_captured_beats_scope():
    sc = StandardScaler()  # noqa: F841 — the scope decoy
    pipeline = Pipeline([("s", StandardScaler())])
    projection = SQLProjection.marginalize(
        "SELECT sc(age).age AS z FROM __THIS__",
        captured={"sc": pipeline},
    )
    assert projection.captured["sc"] is pipeline
    fitted = projection.fit(TRAIN)
    assert isinstance(next(iter(fitted.instances.values())), Pipeline)


def test_python_transform_missing_id_raises():
    from sql_transform import PythonTransform

    t = PythonTransform("t", instances={}, takes=pa.schema([("x", pa.float64())]))
    assert t(None, 1.0) is None
    with pytest.raises(UDFError, match="different fits"):
        t(0, 1.0)


def test_udf_declaration_violation_traps():
    liar = PythonUDF("liar", lambda x: "not a float", pa.schema([("x", pa.float64())]))
    p = SQLProjection.marginalize(
        "SELECT liar(age) AS z FROM __THIS__", captured={"liar": liar}
    ).fit(TRAIN)
    with pytest.raises(Exception, match="(?i)convert|conversion|cast"):
        p.transform(TRAIN)


@pytest.mark.parametrize(
    "sql,match",
    [
        ("SELECT nope(age) OVER () FROM __THIS__", "not an aggregate"),
        (
            "SELECT sc_transform(sc_fit(age) OVER (ORDER BY age), age).age"
            " AS z FROM __THIS__",
            "running fit",
        ),
        ("SELECT sc(*).a AS z FROM __THIS__", r"\*|bundle"),
        ("SELECT sc(age, fare).a AS z FROM __THIS__", "one.*bundle"),
        ("SELECT sc(struct_pack(age)).a AS z FROM __THIS__", "named"),
        (
            "WITH a AS (SELECT sc(age).age AS z FROM __THIS__) SELECT z FROM a",
            "chain",
        ),
        ("SELECT sk.sc(age).age AS z FROM __THIS__", "namespaced"),
        (
            "SELECT avg(sc(age).age) OVER () FROM __THIS__",
            "nests.*estimator",
        ),
        ("SELECT mystery(age) FROM __THIS__", "mystery"),
        (
            "SELECT (SELECT max(mystery(age)) FROM __THIS__) FROM __THIS__",
            "mystery",
        ),
    ],
)
def test_refusals(sql, match):
    sc = StandardScaler()  # noqa: F841 — resolved from scope where relevant
    with pytest.raises(TransformError, match=match):
        SQLProjection.marginalize(sql).fit(TRAIN)


def test_udf_arity_mismatch_refuses():
    one = PythonUDF("one", lambda x: x, pa.schema([("x", pa.float64())]))
    with pytest.raises(TransformError, match="one.*1.*2"):
        SQLProjection.marginalize(
            "SELECT one(age, fare) FROM __THIS__", captured={"one": one}
        )


def test_udf_name_mismatch_refuses():
    other = PythonUDF("something_else", lambda x: x, pa.schema([("x", pa.float64())]))
    with pytest.raises(TransformError, match="named|name"):
        SQLProjection.marginalize(
            "SELECT one(age) FROM __THIS__", captured={"one": other}
        )


def test_a_udf_named_after_a_duckdb_function_refuses(recwarn):  # noqa: ARG001
    """`_known_functions()` gated UDF resolution, so a declared UDF named
    after any DuckDB function was never resolved, never recorded, and the
    call planned as the BUILTIN — the user declared a callable and got
    someone else's function, silently. confit refuses the same collision one
    layer down, but the dropped UDF never reaches it, so that guard was
    unreachable from here."""
    shout = PythonUDF(
        "abs",
        lambda x: 111.0,
        pa.schema([("x", pa.float64())]),
        pa.float64(),
    )
    with pytest.raises(TransformError, match="abs.*builtin|builtin.*abs"):
        SQLProjection.marginalize(
            "SELECT abs(age) AS a FROM __THIS__", captured={"abs": shout}
        ).fit(TRAIN)


def test_a_builtin_call_without_a_colliding_udf_still_plans():
    """Control: the refusal must not fire on ordinary builtin calls."""
    p = SQLProjection.marginalize("SELECT abs(age) AS a FROM __THIS__").fit(TRAIN)
    assert p.udfs == {}
    assert p.transform(TRAIN).column("a").to_pylist() == TRAIN.column("age").to_pylist()


@pytest.mark.parametrize(
    "sql,column,expected",
    [
        (
            "SELECT (age * 2 + 1) * 3 AS out, name FROM __THIS__",
            "out",
            [(age * 2 + 1) * 3 for age in TRAIN.column("age").to_pylist()],
        ),
        (
            "SELECT age * 2 + 1 AS out, name FROM __THIS__",
            "out",
            [age * 2 + 1 for age in TRAIN.column("age").to_pylist()],
        ),
        (
            "SELECT struct_extract(struct_pack(f := age), 'f') + 1 AS z,"
            " name FROM __THIS__",
            "z",
            [age + 1 for age in TRAIN.column("age").to_pylist()],
        ),
        (
            "SELECT name AS age, age AS out FROM __THIS__",
            "out",
            TRAIN.column("age").to_pylist(),
        ),
    ],
)
def test_migrated_private_value_computations(sql, column, expected):
    # Hidden aliases are removed; expressions or ordinary visible aliases replace them.
    fitted = SQLProjection(sql).fit(TRAIN)
    assert fitted.transform(TRAIN).column(column).to_pylist() == expected
    assert [
        row[column] for row in fitted.compile().infer_rows(TRAIN.to_pylist())
    ] == expected


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT struct_pack(a := age) AS s, name FROM __THIS__",
        "SELECT age AS _x, struct_pack(a := _x) AS s, name FROM __THIS__",
    ],
)
def test_ordinary_lateral_struct_preserves_named_fields(sql):
    fitted = SQLProjection(sql).fit(TRAIN)
    expected = [{"a": age} for age in TRAIN.column("age").to_pylist()]
    assert fitted.transform(TRAIN).column("s").to_pylist() == expected
    assert [
        row["s"] for row in fitted.compile().infer_rows(TRAIN.to_pylist())
    ] == expected


def test_ordinary_alias_reprojection_preserves_names_and_values():
    fitted = SQLProjection("SELECT age + 1 AS b, b AS b2, name FROM __THIS__").fit(
        TRAIN
    )
    out = fitted.transform(TRAIN)
    assert out.column_names == ["b", "b2", "name"]
    expected = [age + 1 for age in TRAIN.column("age").to_pylist()]
    assert out.column("b").to_pylist() == expected
    assert out.column("b2").to_pylist() == expected


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT age - avg(age) OVER(PARTITION BY country) AS c, name FROM __THIS__",
        "SELECT age >= 30 AS o, avg(fare) OVER(PARTITION BY age >= 30) AS m,"
        " name FROM __THIS__",
        "SELECT min(struct_pack(a := age)) OVER() AS m1,"
        " min(struct_pack(b := age)) OVER() AS m2, name FROM __THIS__",
        "SELECT (SELECT struct_pack(a := max(age)) FROM __THIS__) AS s1,"
        " (SELECT struct_pack(b := max(age)) FROM __THIS__) AS s2, name FROM __THIS__",
    ],
)
def test_migrated_private_windows_and_named_identity(sql):
    from sql_transform._marginal_projection_test import gate

    gate(sql, TRAIN)


def test_migrated_private_estimator_field_uses_explicit_extract():
    fitted = SQLProjection.marginalize(
        "SELECT struct_extract(sc(struct_pack(v := age)), 'v') AS z,"
        " name FROM __THIS__",
        captured={"sc": StandardScaler()},
    ).fit(TRAIN)
    feats = np.array(TRAIN.column("age").to_pylist(), dtype=float).reshape(-1, 1)
    expected = _reference(StandardScaler(), feats, [()] * TRAIN.num_rows)
    got = fitted.transform(TRAIN).column("z").to_pylist()
    for value, reference in zip(got, expected, strict=True):
        np.testing.assert_allclose(value, reference[0], rtol=1e-12)


def test_underscore_input_columns_are_ordinary_public_values():
    data = pa.table({"age": [1.0, 2.0], "_meta": ["a", "b"]})
    fitted = SQLProjection("SELECT * FROM __THIS__").fit(data)
    assert fitted.transform(data).equals(data)
    assert fitted.compile().infer_rows(data.to_pylist()) == data.to_pylist()
    explicit = SQLProjection("SELECT age FROM __THIS__").fit(data)
    assert explicit.transform(data).equals(data.select(["age"]))


def test_migrated_centered_renamed_feature_uses_original_fit_window_prefix():
    # The final raw fit reads one original inline prefix over FIT. Its request
    # stage reads a DISTINCT pick of that window's mean, never a grouped mean.
    fit_prefix = (
        "SELECT country, age * 2 - avg(age * 2) OVER(PARTITION BY country) AS age2,"
        " name FROM __FIT__"
    )
    projection = SQLProjection(
        "WITH carrier AS (SELECT country, age * 2 AS age2,"
        " avg(age * 2) OVER(PARTITION BY country) AS m, name FROM __FIT__),"
        " means AS (SELECT DISTINCT c.country, c.m FROM carrier c),"
        f" p AS (SELECT country, sc_fit(age2) AS iid "
        f"FROM ({fit_prefix}) a GROUP BY country)"
        " SELECT sc_transform(p.iid,"
        " struct_pack(age2 := t.age * 2 - means.m)).age2 - 1 AS z,"
        " t.name FROM __THIS__ t"
        " LEFT JOIN means ON t.country IS NOT DISTINCT FROM means.country"
        " LEFT JOIN p ON t.country IS NOT DISTINCT FROM p.country",
        captured={"sc": StandardScaler()},
    )
    fitted = projection.fit(TRAIN)
    con = duckdb.connect()
    try:
        con.execute("SET threads = 1")
        con.execute("PRAGMA disable_optimizer")
        con.register("__FIT__", TRAIN)
        original = con.execute(fit_prefix).to_arrow_table()
    finally:
        con.close()
    feats = np.array(original.column("age2").to_pylist(), dtype=float).reshape(-1, 1)
    expected = _reference(
        StandardScaler(), feats, [(g,) for g in original.column("country").to_pylist()]
    )
    got = _by_name(fitted.transform(TRAIN), "z")
    for i, name in enumerate(original.column("name").to_pylist()):
        np.testing.assert_allclose(got[name], expected[i][0] - 1, rtol=1e-12)
    compiled = fitted.compile()
    assert compiled.infer_rows(TRAIN.to_pylist()) == fitted.transform(TRAIN).to_pylist()
    replay = SQLProjection(projection.source, captured=projection.captured).fit(TRAIN)
    assert replay.transform(TRAIN).equals(fitted.transform(TRAIN))


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT avg(age) OVER() AS _m, sum(_m) OVER() AS s FROM __THIS__",
        "SELECT avg(age) OVER() AS _m,"
        " avg(fare) OVER(PARTITION BY _m) AS s FROM __THIS__",
        "SELECT age >= 30 AS o, avg(fare) OVER(PARTITION BY o) AS m FROM __THIS__",
        "SELECT age / 2 AS _h, sc(struct_pack(v := _h)).v AS z FROM __THIS__",
    ],
)
def test_alias_expansion_into_fit_expressions_is_removed(sql):
    with pytest.raises(TransformError, match="sibling.*alias|inline the expression"):
        SQLProjection.marginalize(sql, captured={"sc": StandardScaler()})
