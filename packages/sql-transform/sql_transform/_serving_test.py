"""The serving gate: infer/infer_batch (Confit) == transform (DuckDB).

Both bindings run the same artifact — serving_sql, params tables, UDF
objects — so their outputs must agree value-for-value on every admitted
query. ``transform`` is the DuckDB oracle side; ``infer_batch`` is the
row-at-a-time path this project exists for.
"""

import math

import pyarrow as pa
import pytest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sql_transform import MarginalizeError, PythonUDF, SQLProjection

TRAIN = pa.table(
    {
        "country": ["US", "US", None, "DE", None, "DE", "FR"],
        "age": [40.0, 30.0, 20.0, 25.0, 50.0, 45.0, 35.0],
        "fare": [7, 8, 6, 9, 10, 11, 5],
        "name": ["x", "y", "z", "w", "v", "u", "t"],
    }
)


def serve_gate(sql: str, table: pa.Table = TRAIN, **kwargs) -> SQLProjection:
    """Fit, then assert the Confit row path equals the DuckDB batch path."""
    p = SQLProjection(sql, **kwargs).fit(table)
    want = p.transform(table).to_pylist()
    got = p.infer_batch(table.to_pylist())
    assert got == want, f"\n{got}\n!=\n{want}"
    return p


def test_aggregates_only():
    p = serve_gate(
        "SELECT age - avg(age) OVER (PARTITION BY country) AS d, name FROM __THIS__"
    )
    assert p.backend in ("cranelift", "interpreter")
    assert p.boundary == "marshaller"


def test_keyless_aggregate_and_int_column():
    serve_gate("SELECT fare - avg(fare) OVER () AS d, name FROM __THIS__")


def test_transformer_partitioned_width1():
    sc = StandardScaler()
    p = serve_gate(
        "SELECT sc_transform(sc_fit(age) OVER (PARTITION BY country), age).age"
        " * 10 + 1 AS z, name FROM __THIS__",
        transformers={"sc": sc},
    )
    out = p.infer({"country": "US", "age": 33.0, "fare": 1, "name": "q"})
    assert isinstance(out["z"], float)


def test_transformer_global_width2_two_field_reads():
    embed = Pipeline([("s", StandardScaler()), ("p", PCA(n_components=2))])
    p = serve_gate(
        "SELECT embed(struct_pack(a := age, f := fare)).pca0 AS e_pca0,"
        " embed(struct_pack(a := age, f := fare)).pca1 AS e_pca1, name"
        " FROM __THIS__",
        transformers={"embed": embed},
    )
    out = p.infer({"country": "US", "age": 33.0, "fare": 4, "name": "q"})
    # Two field reads of one call (struct-valued calls; counted in
    # _single_eval_test.py).
    assert isinstance(out["e_pca0"], float) and isinstance(out["e_pca1"], float)


def test_unseen_group_is_null_row_at_a_time():
    sc = StandardScaler()
    p = serve_gate(
        "SELECT sc_transform(sc_fit(age) OVER (PARTITION BY country), age).age"
        " AS z, name FROM __THIS__",
        transformers={"sc": sc},
    )
    out = p.infer({"country": "JP", "age": 1.0, "fare": 1, "name": "q"})
    assert out["z"] is None


def test_author_udf_serves():
    halve = PythonUDF(
        "halve",
        lambda x: None if x is None else x / 2.0,
        pa.schema([("x", pa.float64())]),
    )
    serve_gate(
        "SELECT halve(age) - avg(halve(age)) OVER () AS d, name FROM __THIS__",
        transformers={"halve": halve},
    )


def test_chain_with_transformer():
    sc = StandardScaler()
    serve_gate(
        "WITH a AS (SELECT age * 2 AS age2, country, name FROM __THIS__)"
        " SELECT sc_transform(sc_fit(age2) OVER (PARTITION BY country), age2).age2"
        " - 1 AS z, name FROM a",
        transformers={"sc": sc},
    )


def test_dict_rows_and_object_rows_agree():
    # MIGRATION-NOTE: the old pydantic ``_row_model`` synthesized-model
    # constructor is gone (no output/row model to build). The equivalent
    # property in the arrow surface is dict-or-object marshalling
    # (packages/confit/docs/specs/2026-08-13-arrow-schema-api-design.md):
    # infer() accepts a dict or an attribute-bearing object and must serve
    # the same row either way.
    from types import SimpleNamespace

    p = serve_gate("SELECT age - avg(age) OVER () AS d FROM __THIS__")
    row = {"country": "US", "age": 33.0, "fare": 1, "name": "q"}
    obj_row = SimpleNamespace(**row)
    assert p.infer(row) == p.infer(obj_row)


def test_not_fitted_refuses():
    p = SQLProjection("SELECT age - avg(age) OVER () AS d FROM __THIS__")
    with pytest.raises(MarginalizeError, match="not fitted"):
        p.infer({"age": 1.0})
    with pytest.raises(MarginalizeError, match="not fitted"):
        _ = p.backend


def test_transform_preserves_input_row_order_with_unseen_groups():
    """Review round (2026-08-05): DuckDB's params LEFT JOIN emits unmatched
    probe rows after matched ones, so a frame with unseen-group rows came
    back reordered — silently misaligning positional consumers. transform()
    must restore input row order explicitly."""
    p = serve_gate(
        "SELECT sc_transform(sc_fit(age) OVER (PARTITION BY country), age).age"
        " AS z, name FROM __THIS__",
        transformers={"sc": StandardScaler()},
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
    """Review round (2026-08-05): DuckDB's native Python-UDF return
    conversion mapped a returned NaN to NULL, so a NULL-fed transformer
    call diverged between batch (NULL) and row (NaN) serving. Registration
    is arrow-typed now — sklearn's NaN convention is the value on BOTH
    paths, bit-equal."""
    p = serve_gate(
        "SELECT sc_transform(sc_fit(struct_pack(v := age)) OVER (),"
        " struct_pack(v := fare)).v AS z, name FROM __THIS__",
        transformers={"sc": StandardScaler()},
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
    row = p.infer({"country": "US", "age": 10.0, "fare": None, "name": "p"})["z"]
    assert batch is not None and math.isnan(batch)
    assert row is not None and math.isnan(row)
