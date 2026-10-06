"""Projection-only typed Python fits and their public serving artifacts."""

import math

import duckdb
import numpy as np
import pyarrow as pa
import pytest
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from sql_transform import (
    UDF,
    Named,
    OrderSensitive,
    PythonTransform,
    PythonUDF,
    SQLProjection,
    SQLTransform,
    TransformError,
)


class Codes:
    def fit(self, rows):
        self.codes = {value: i for i, value in enumerate(dict.fromkeys(rows[:, 0]))}
        return self

    def transform(self, rows):
        return np.asarray([[float(self.codes.get(row[0], -1))] for row in rows])


class Mixed:
    def fit(self, rows):
        self.codes = {value: i for i, value in enumerate(dict.fromkeys(rows[:, 1]))}
        self.seen = rows.copy()
        return self

    def transform(self, rows):
        return np.asarray(
            [
                [
                    float(row[0]) if not math.isnan(row[0]) else -1.0,
                    float(self.codes.get(row[1], -1)),
                ]
                for row in rows
            ]
        )


class Counting:
    fits = 0

    def fit(self, rows):
        type(self).fits += 1
        self.mean = float(np.mean(rows[:, 0]))
        return self

    def transform(self, rows):
        return np.asarray([[float(row[0]) - self.mean] for row in rows])


class IntegerIdentity(UDF):
    name = "ident"
    takes = pa.schema([("v", pa.int64())])
    returns = pa.int64()

    def __call__(self, value):
        return (value,)


FIT = pa.table({"g": ["a", "a", None], "v": [10.0, 20.0, 7.0]})
REQUEST = pa.table({"g": ["a", "NEW", None], "v": [20.0, 4.0, 14.0]})
EXPLICIT = """
    WITH p AS (SELECT g, sc_fit(v) AS iid FROM __FIT__ GROUP BY g)
    SELECT t.g, sc_transform(p.iid, t.v).v AS z
    FROM __THIS__ t LEFT JOIN p ON t.g IS NOT DISTINCT FROM p.g
"""


def public_paths(fitted, request, expected):
    from confit import DuckDBInferFn

    public = DuckDBInferFn(
        sql=fitted.sql,
        row_tables={"__THIS__": fitted.schema},
        static_tables=fitted.params,
        udfs=list(fitted.udfs.values()),
        shape="map",
    )
    assert fitted.transform(request).to_pylist() == expected
    assert fitted.compile().infer_rows(request.to_pylist()) == expected
    assert public.infer_rows(request.to_pylist()) == expected
    assert public.infer_arrow(request).to_pylist() == expected


@pytest.mark.parametrize("marginal", [False, True])
def test_scaler_canonical_cte_and_window_have_the_same_public_artifact(marginal):
    sc = StandardScaler()
    if marginal:
        projection = SQLProjection.marginalize(
            "SELECT g, sc_transform(sc_fit(v) OVER(PARTITION BY g), v).v AS z "
            "FROM __THIS__",
            captured={"sc": sc},
        )
    else:
        projection = SQLProjection(EXPLICIT, captured={"sc": sc})
    fitted = projection.fit(FIT)
    public_paths(
        fitted,
        REQUEST,
        [{"g": "a", "z": 1.0}, {"g": "NEW", "z": None}, {"g": None, "z": 7.0}],
    )
    for udf in fitted.udfs.values():
        assert isinstance(udf, PythonTransform)
        assert all(fitted.instances[iid] is est for iid, est in udf.instances.items())
    assert all(
        pa.types.is_int64(table.schema.field(-1).type)
        for table in fitted.params.values()
    )
    replay = SQLProjection(projection.source, captured=projection.captured).fit(FIT)
    assert replay.transform(REQUEST).equals(fitted.transform(REQUEST))


@pytest.mark.parametrize("partitioned", [False, True])
def test_filtered_numeric_looking_strings_remain_strings(partitioned):
    data = pa.table(
        {"g": ["a", "a", "a"], "s": ["1", "2", "x"], "keep": [True, True, False]}
    )
    code = Named(Codes(), returns=("code",))
    partition = "PARTITION BY g" if partitioned else ""
    fitted = SQLProjection.marginalize(
        "SELECT code_transform(code_fit(s) FILTER(WHERE keep) "
        f"OVER({partition}), s).code AS c FROM __THIS__",
        captured={"code": code},
    ).fit(data)
    public_paths(fitted, data, [{"c": 0.0}, {"c": 1.0}, {"c": -1.0}])
    assert [udf.takes for udf in fitted.udfs.values()] == [
        pa.schema([("s", pa.string())])
    ]


def test_mixed_arrow_schema_controls_fit_and_serving_conversion():
    estimator = Named(Mixed(), returns=("numeric", "code"))
    fit = pa.table({"v": pa.array([2**53 + 1, None], pa.int64()), "s": ["1", "2"]})
    request = pa.table(
        {"v": pa.array([2**53 + 1, None, 7], pa.int64()), "s": ["1", "2", "x"]}
    )
    fitted = SQLProjection(
        "WITH p AS (SELECT md_fit(struct_pack(v := v, s := s)) AS iid FROM __FIT__) "
        'SELECT md_transform(p.iid, struct_pack(v := t.v, s := t.s)) AS "out" '
        "FROM __THIS__ t LEFT JOIN p ON 1 = 1",
        captured={"md": estimator},
    ).fit(fit)
    public_paths(
        fitted,
        request,
        [
            {"out": {"numeric": float(2**53 + 1), "code": 0.0}},
            {"out": {"numeric": -1.0, "code": 1.0}},
            {"out": {"numeric": 7.0, "code": -1.0}},
        ],
    )
    (udf,) = fitted.udfs.values()
    assert udf.takes == pa.schema([("v", pa.int64()), ("s", pa.string())])
    (instance,) = fitted.instances.values()
    assert instance.estimator.seen[0, 0] == float(2**53 + 1)
    assert math.isnan(instance.estimator.seen[1, 0])
    assert instance.estimator.seen[:, 1].tolist() == ["1", "2"]


@pytest.mark.parametrize("type", [pa.int32(), pa.float32()])
def test_narrow_features_publish_normalized_declarations(type):
    data = pa.table(
        {
            "g": ["a", "a"],
            "v": pa.array([10, 20], type),
            "value": pa.array([10, 20], pa.float64()),
        }
    )
    sql = EXPLICIT.replace(
        "sc_transform(p.iid, t.v)", "sc_transform(p.iid, struct_pack(v := t.value))"
    )
    fitted = SQLProjection(sql, captured={"sc": StandardScaler()}).fit(data)
    request = data.select(["g", "value"])
    public_paths(fitted, request, [{"g": "a", "z": -1.0}, {"g": "a", "z": 1.0}])
    (udf,) = fitted.udfs.values()
    assert udf.takes.field("v").type == (
        pa.int64() if pa.types.is_integer(type) else pa.float64()
    )


@pytest.mark.parametrize(
    "cast, expected_type",
    [
        ("INTEGER", pa.int64()),
        ("FLOAT", pa.float64()),
    ],
)
def test_sql_cast_fit_features_publish_normalized_declarations(cast, expected_type):
    sql = EXPLICIT.replace("sc_fit(v)", f"sc_fit(struct_pack(v := CAST(v AS {cast})))")
    fitted = SQLProjection(sql, captured={"sc": StandardScaler()}).fit(FIT)
    public_paths(
        fitted,
        FIT,
        [
            {"g": "a", "z": -1.0},
            {"g": "a", "z": 1.0},
            {"g": None, "z": 0.0},
        ],
    )
    (udf,) = fitted.udfs.values()
    assert udf.takes.field("v").type == expected_type


def test_inline_scope_sharing_does_not_merge_separately_authored_fits():
    Counting.fits = 0
    sc = Named(Counting(), returns=("v",))
    shared = SQLProjection.marginalize(
        "SELECT sc_transform(sc_fit(v) OVER(PARTITION BY g), v).v AS a, "
        "sc_transform(sc_fit(v) OVER(PARTITION BY g),"
        " struct_pack(v := v * 2)).v AS b FROM __THIS__",
        captured={"sc": sc},
    ).fit(FIT)
    assert Counting.fits == 2
    public_paths(
        shared,
        FIT,
        [{"a": -5.0, "b": 5.0}, {"a": 5.0, "b": 25.0}, {"a": 0.0, "b": 7.0}],
    )
    Counting.fits = 0
    separate = SQLProjection(
        "WITH p AS (SELECT g, sc_fit(v) AS iid FROM __FIT__ GROUP BY g), "
        "q AS (SELECT g, sc_fit(v) AS iid FROM __FIT__ GROUP BY g) "
        "SELECT sc_transform(p.iid,t.v).v AS a,"
        " sc_transform(q.iid,struct_pack(v := t.v * 2)).v AS b "
        "FROM __THIS__ t LEFT JOIN p ON t.g IS NOT DISTINCT FROM p.g "
        "LEFT JOIN q ON t.g IS NOT DISTINCT FROM q.g",
        captured={"sc": sc},
    ).fit(FIT)
    assert Counting.fits == 4
    assert separate.transform(FIT).equals(shared.transform(FIT))


def test_ordered_fits_keep_input_order_for_ties_through_nested_wrappers():
    data = pa.table({"s": ["b", "a", "c"], "ts": [1, 1, 0]})
    code = Named(OrderSensitive(Codes()), returns=("code",))
    fitted = SQLProjection.marginalize(
        "SELECT code_transform(code_fit(s ORDER BY ts) OVER(),"
        " s).code AS c FROM __THIS__",
        captured={"code": code},
    ).fit(data)
    public_paths(fitted, data, [{"c": 1.0}, {"c": 2.0}, {"c": 0.0}])
    with pytest.raises(TransformError, match="ORDER"):
        SQLProjection.marginalize(
            "SELECT code(s).code AS c FROM __THIS__", captured={"code": code}
        )


def test_filtered_empty_group_mints_no_instance_and_misses():
    data = pa.table(
        {"g": ["a", "a", "b"], "s": ["1", "2", "x"], "keep": [True, True, False]}
    )
    fitted = SQLProjection.marginalize(
        "SELECT g, code_transform(code_fit(s) FILTER(WHERE keep)"
        " OVER(PARTITION BY g), s).code AS c FROM __THIS__",
        captured={"code": Named(Codes(), returns=("code",))},
    ).fit(data)
    public_paths(
        fitted,
        data,
        [{"g": "a", "c": 0.0}, {"g": "a", "c": 1.0}, {"g": "b", "c": None}],
    )
    assert len(fitted.instances) == 1
    with pytest.raises(TransformError, match="empty|FILTER|filtered"):
        SQLProjection.marginalize(
            "SELECT code_transform(code_fit(s) FILTER(WHERE keep) OVER(),"
            " s).code AS c FROM __THIS__",
            captured={"code": Named(Codes(), returns=("code",))},
        ).fit(data.set_column(2, "keep", pa.array([False] * 3)))


@pytest.mark.parametrize(
    "udf",
    [
        IntegerIdentity(),
        PythonUDF("ident", lambda v: v, pa.schema([("v", pa.int64())]), pa.int64()),
    ],
)
def test_scalar_udfs_keep_exact_large_integers_and_nulls(udf):
    data = pa.table({"v": pa.array([2**53 + 1, None, -(2**53 + 1)], pa.int64())})
    fitted = SQLProjection(
        "SELECT ident(v) AS z FROM __THIS__", captured={"ident": udf}
    ).fit(data)
    public_paths(fitted, data, [{"z": 2**53 + 1}, {"z": None}, {"z": -(2**53 + 1)}])
    with pytest.raises(TransformError):
        SQLTransform("SELECT ident(v) AS z FROM __THIS__", captured={"ident": udf})


def test_raw_estimator_cannot_shadow_a_builtin_before_lowering():
    with pytest.raises(TransformError, match="builtin"):
        SQLProjection.marginalize(
            "SELECT abs(v).v AS z FROM __THIS__",
            captured={"abs": StandardScaler()},
        )


def test_source_replay_and_clones_do_not_add_generated_capture_keys():
    captures = {"sc": StandardScaler()}
    projection = SQLProjection(EXPLICIT, captured=captures)
    first = SQLProjection(projection.source, captured=projection.captured)
    second = SQLProjection(first.source, captured=first.captured)
    assert first.captured is captures and second.captured is captures
    assert set(captures) == {"sc"}
    generic = SQLTransform("SELECT v * 2 AS z FROM __THIS__")
    assert clone(clone(generic)).fit(FIT).transform(REQUEST).to_pylist() == [
        {"z": 40.0},
        {"z": 8.0},
        {"z": 28.0},
    ]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT sc_transform(0, v).v AS z FROM __THIS__",
        "WITH p AS (SELECT sc_fit(v) AS iid FROM __FIT__) "
        "SELECT sc_transform(p.iid + 1,v).v AS z FROM __THIS__ t,p",
        "WITH p AS (SELECT sc_fit(v) AS iid FROM __THIS__) "
        "SELECT sc_transform(p.iid,v).v AS z FROM __THIS__ t,p",
        "WITH f AS (SELECT v FROM __FIT__),"
        " p AS (SELECT sc_fit(v) AS iid FROM f) "
        "SELECT sc_transform(p.iid,v).v AS z FROM __THIS__ t,p",
        "WITH p AS (SELECT sc_fit(v) AS iid, avg(v) AS m FROM __FIT__) "
        "SELECT sc_transform(p.iid,v).v AS z FROM __THIS__ t,p",
        "WITH p AS (SELECT sc_fit(v) AS iid "
        "FROM (SELECT v FROM (SELECT v FROM __FIT__) f) q) "
        "SELECT sc_transform(p.iid,v).v AS z FROM __THIS__ t,p",
        "WITH p AS (SELECT sc_fit(v) AS iid FROM __FIT__),"
        " q AS (SELECT iid FROM p) "
        "SELECT sc_transform(q.iid,v).v AS z FROM __THIS__ t,q",
    ],
)
def test_raw_ids_and_sources_have_one_positive_grammar(sql):
    with pytest.raises(TransformError):
        SQLProjection(sql, captured={"sc": StandardScaler()})


def test_parked_raw_fit_window_names_the_inline_and_explicit_forms():
    with pytest.raises(TransformError, match="inline.*explicit CTE"):
        SQLProjection.marginalize(
            "SELECT sc_fit(v) OVER() AS _th FROM __THIS__",
            captured={"sc": StandardScaler()},
        )


def test_learned_missing_field_refuses_at_fit():
    with pytest.raises(TransformError, match="missing"):
        SQLProjection(
            EXPLICIT.replace(").v AS z", ").missing AS z"),
            captured={"sc": StandardScaler()},
        ).fit(FIT)


def test_fit_ordinal_collisions_refuse_before_numbering():
    for name in ("__cf_fit_row", "__CF_ROW"):
        data = FIT.append_column(name, pa.array([999] * FIT.num_rows))
        with pytest.raises(TransformError, match="reserved"):
            SQLProjection(EXPLICIT, captured={"sc": StandardScaler()}).fit(data)


def test_raw_eager_artifacts_do_not_change_outstanding_generic_callbacks():
    import pyarrow.compute as pc

    from sql_transform import Transform

    callbacks = Transform(
        fit=lambda f: pc.mean(f["v"]).as_py(),
        transform=lambda mean, t: pa.table({"v": pc.subtract(t["v"], mean)}),
        takes=("v",),
        returns=("v",),
    )
    with duckdb.connect() as con:
        con.execute("SET threads = 3")
        generic = SQLTransform(
            "SELECT sc_transform(p.theta, struct_pack(v := t.v)).v AS z "
            "FROM __THIS__ t, (SELECT sc_fit(struct_pack(v := v)) AS theta "
            "FROM __FIT__) p",
            connection=con,
            captured={"sc": callbacks},
        ).fit(FIT)
        request = pa.table({"v": [30.0]})
        outstanding = generic.relation(request)
        first = SQLProjection(
            EXPLICIT, connection=con, captured={"sc": StandardScaler()}
        ).fit(FIT)
        second = SQLProjection(
            EXPLICIT,
            connection=con,
            captured={"sc": StandardScaler(with_mean=False)},
        ).fit(FIT)
        assert first.transform(REQUEST).to_pylist() == [
            {"g": "a", "z": 1.0},
            {"g": "NEW", "z": None},
            {"g": None, "z": 7.0},
        ]
        assert second.transform(REQUEST).to_pylist() == [
            {"g": "a", "z": 4.0},
            {"g": "NEW", "z": None},
            {"g": None, "z": 14.0},
        ]
        assert outstanding.to_arrow_table().to_pylist() == [
            {"z": 30.0 - (10.0 + 20.0 + 7.0) / 3}
        ]
        assert con.execute("SELECT current_setting('threads')").fetchone() == (3,)
        generic.release()


def test_partial_function_registration_preserves_error_and_restores_threads():
    class Broken(IntegerIdentity):
        name = "broken"

        fail_registration = False

        def _duck_signature(self):
            if self.fail_registration:
                raise RuntimeError("broken registration")
            return super()._duck_signature()

    with duckdb.connect() as con:
        con.execute("SET threads = 3")
        before = con.execute(
            "SELECT function_name FROM duckdb_functions() WHERE function_type='scalar'"
        ).fetchall()
        broken = Broken()
        projection = SQLProjection(
            "SELECT ident(v) AS a, broken(v) AS b FROM __THIS__",
            connection=con,
            captured={"ident": IntegerIdentity(), "broken": broken},
        )
        broken.fail_registration = True
        with pytest.raises(RuntimeError, match="broken registration"):
            projection.fit(FIT)
        assert con.execute("SELECT current_setting('threads')").fetchone() == (3,)
        assert (
            con.execute(
                "SELECT function_name FROM duckdb_functions() "
                "WHERE function_type='scalar'"
            ).fetchall()
            == before
        )
