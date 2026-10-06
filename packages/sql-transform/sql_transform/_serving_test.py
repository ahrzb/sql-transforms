"""The serving gate: compiled Confit functions equal fitted DuckDB transforms.

Both bindings run the same artifact — SQL, params tables, and UDF objects —
so their outputs must agree value-for-value on every admitted query.
``transform`` is the DuckDB batch side; ``infer_rows`` and ``infer_arrow``
exercise Confit's row-local serving path.
"""

import math

import pyarrow as pa
import pytest
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sql_transform import (
    FittedProjection,
    PythonTransform,
    PythonUDF,
    SQLProjection,
    Transform,
    TransformError,
)

TRAIN = pa.table(
    {
        "country": ["US", "US", None, "DE", None, "DE", "FR"],
        "age": [40.0, 30.0, 20.0, 25.0, 50.0, 45.0, 35.0],
        "fare": [7, 8, 6, 9, 10, 11, 5],
        "name": ["x", "y", "z", "w", "v", "u", "t"],
    }
)


def serve_gate(projection: SQLProjection, table: pa.Table = TRAIN) -> FittedProjection:
    """Fit, then compare both Confit interfaces with the DuckDB batch path."""
    fitted = projection.fit(table)
    want = fitted.transform(table).to_pylist()
    fn = fitted.compile()
    got = fn.infer_rows(table.to_pylist())
    assert got == want, f"\n{got}\n!=\n{want}"
    assert fn.infer_arrow(table).to_pylist() == want
    return fitted


def test_aggregates_only():
    p = serve_gate(
        SQLProjection.marginalize(
            "SELECT age - avg(age) OVER (PARTITION BY country) AS d, name FROM __THIS__"
        )
    )
    fn = p.compile()
    assert fn.backend in ("cranelift", "interpreter")
    assert fn.boundary == "marshaller"
    assert p.compile() is not fn


def test_keyless_aggregate_and_int_column():
    serve_gate(
        SQLProjection.marginalize(
            "SELECT fare - avg(fare) OVER () AS d, name FROM __THIS__"
        )
    )


def test_transformer_partitioned_width1():
    sc = StandardScaler()
    p = serve_gate(
        SQLProjection.marginalize(
            "SELECT sc_transform(sc_fit(age) OVER (PARTITION BY country), age).age"
            " * 10 + 1 AS z, name FROM __THIS__",
            captured={"sc": sc},
        )
    )
    assert all(isinstance(udf, PythonTransform) for udf in p.udfs.values())
    out = p.compile().infer_rows(
        [{"country": "US", "age": 33.0, "fare": 1, "name": "q"}]
    )[0]
    assert isinstance(out["z"], float)


def test_transformer_global_width2_two_field_reads():
    embed = Pipeline([("s", StandardScaler()), ("p", PCA(n_components=2))])
    p = serve_gate(
        SQLProjection.marginalize(
            "SELECT embed(struct_pack(a := age, f := fare)).pca0 AS e_pca0,"
            " embed(struct_pack(a := age, f := fare)).pca1 AS e_pca1, name"
            " FROM __THIS__",
            captured={"embed": embed},
        )
    )
    out = p.compile().infer_rows(
        [{"country": "US", "age": 33.0, "fare": 4, "name": "q"}]
    )[0]
    # Two field reads of one call (struct-valued calls; counted in
    # _single_eval_test.py).
    assert isinstance(out["e_pca0"], float) and isinstance(out["e_pca1"], float)


def test_unseen_group_is_null_row_at_a_time():
    sc = StandardScaler()
    p = serve_gate(
        SQLProjection.marginalize(
            "SELECT sc_transform(sc_fit(age) OVER (PARTITION BY country), age).age"
            " AS z, name FROM __THIS__",
            captured={"sc": sc},
        )
    )
    out = p.compile().infer_rows(
        [{"country": "JP", "age": 1.0, "fare": 1, "name": "q"}]
    )[0]
    assert out["z"] is None


def test_author_udf_serves():
    halve = PythonUDF(
        "halve",
        lambda x: None if x is None else x / 2.0,
        pa.schema([("x", pa.float64())]),
    )
    serve_gate(
        SQLProjection.marginalize(
            "SELECT halve(age) - avg(halve(age)) OVER () AS d, name FROM __THIS__",
            captured={"halve": halve},
        )
    )


def test_author_udf_preserves_declared_integer_values_and_nulls():
    identity = PythonUDF(
        "int_identity",
        lambda value: value,
        pa.schema([("v", pa.int64())]),
        pa.int64(),
    )
    data = pa.table({"v": pa.array([2**53 + 1, None, -(2**53 + 1)], pa.int64())})
    fitted = serve_gate(
        SQLProjection(
            "SELECT int_identity(v) AS z FROM __THIS__",
            captured={"int_identity": identity},
        ),
        data,
    )
    expected = pa.table({"z": data["v"]})
    assert fitted.transform(data).equals(expected)
    assert fitted.compile().infer_arrow(data).equals(expected)


def test_migrated_chain_with_transformer():
    # The original chain fits age * 2 by country, then subtracts one.
    # The projected fit source and request expression state both stages explicitly.
    sc = StandardScaler()
    p = serve_gate(
        SQLProjection(
            "WITH p AS (SELECT country, sc_fit(age2) AS iid "
            "FROM (SELECT age * 2 AS age2, country, name FROM __FIT__) f "
            "GROUP BY country) "
            "SELECT sc_transform(p.iid, struct_pack(age2 := t.age * 2)).age2"
            " - 1 AS z, t.name FROM __THIS__ t "
            "LEFT JOIN p ON t.country IS NOT DISTINCT FROM p.country",
            captured={"sc": sc},
        )
    )
    rows = TRAIN.to_pylist()
    models = {
        group: clone(sc).fit(
            [[row["age"] * 2] for row in rows if row["country"] == group]
        )
        for group in dict.fromkeys(TRAIN["country"].to_pylist())
    }
    expected = [
        {
            "z": float(models[row["country"]].transform([[row["age"] * 2]])[0, 0]) - 1,
            "name": row["name"],
        }
        for row in rows
    ]
    assert p.transform(TRAIN).to_pylist() == expected
    assert p.compile().infer_rows(rows) == expected


def test_dict_rows_and_object_rows_agree():
    # Confit accepts dict rows and attribute-bearing objects through the same
    # marshalling contract.
    from types import SimpleNamespace

    p = serve_gate(
        SQLProjection.marginalize("SELECT age - avg(age) OVER () AS d FROM __THIS__")
    )
    row = {"country": "US", "age": 33.0, "fare": 1, "name": "q"}
    obj_row = SimpleNamespace(**row)
    fn = p.compile()
    assert fn.infer_rows([row]) == fn.infer_rows([obj_row])


def test_transform_preserves_input_row_order_with_unseen_groups():
    """DuckDB's params LEFT JOIN can emit unmatched rows after matched ones.
    Batch output must restore input order to preserve positional alignment."""
    p = serve_gate(
        SQLProjection.marginalize(
            "SELECT sc_transform(sc_fit(age) OVER (PARTITION BY country), age).age"
            " AS z, name FROM __THIS__",
            captured={"sc": StandardScaler()},
        )
    )
    frame = pa.table(
        {
            "country": ["JP", "US", "JP", "US"],
            "age": [1.0, 33.0, 2.0, 41.0],
            "fare": [0.0, 0.0, 0.0, 0.0],
            "name": ["a", "b", "c", "d"],
        }
    )
    out = p.transform(frame)
    assert out.column("name").to_pylist() == ["a", "b", "c", "d"]
    z = out.column("z").to_pylist()
    assert z[0] is None and z[2] is None  # unseen group -> NULL, in place
    assert z[1] is not None and z[3] is not None


def test_null_feature_serves_nan_on_both_paths():
    """Native Python-UDF return conversion can map NaN to NULL. Arrow-typed
    registration must preserve sklearn's NaN convention in both serving paths."""
    p = serve_gate(
        SQLProjection.marginalize(
            "SELECT sc_transform(sc_fit(struct_pack(v := age)) OVER (),"
            " struct_pack(v := fare)).v AS z, name FROM __THIS__",
            captured={"sc": StandardScaler()},
        )
    )
    frame = pa.table(
        {
            "country": ["US"],
            "age": [10.0],
            "fare": pa.array([None], pa.float64()),
            "name": ["p"],
        }
    )
    batch = p.transform(frame).column("z").to_pylist()[0]
    fn = p.compile()
    row = fn.infer_rows([{"country": "US", "age": 10.0, "fare": None, "name": "p"}])[0][
        "z"
    ]
    arrow = (
        fn.infer_arrow(frame.select(p.schema.names).cast(p.schema))
        .column("z")
        .to_pylist()[0]
    )
    assert batch is not None and math.isnan(batch)
    assert row is not None and math.isnan(row)
    assert arrow is not None and math.isnan(arrow)


def test_generic_python_callbacks_remain_batch_only():
    import pyarrow.compute as pc

    center = Transform(
        fit=lambda data: pc.mean(data["v"]).as_py(),
        transform=lambda mean, data: pa.table({"v": pc.subtract(data["v"], mean)}),
        takes=("v",),
        returns=("v",),
    )
    fitted = SQLProjection(
        "SELECT center_transform(p.theta, struct_pack(v := t.age)).v AS z "
        "FROM __THIS__ t, "
        "(SELECT center_fit(struct_pack(v := age)) AS theta FROM __FIT__) p",
        captured={"center": center},
    ).fit(TRAIN)
    mean = pc.mean(TRAIN["age"]).as_py()
    expected = pc.subtract(TRAIN["age"], mean).to_pylist()
    assert fitted.transform(TRAIN)["z"].to_pylist() == expected
    with pytest.raises(TransformError, match="row path"):
        fitted.compile()
