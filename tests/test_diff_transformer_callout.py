"""Differential parity: transform (DataFusion UDF) == infer (Rust Expr::Transform)."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from datafusion import SessionContext
from differential import _rows_equal
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer, StandardScaler

from sql_transform._interpreter import InferFn
from sql_transform._schema import synthesize_this_model
from sql_transform._transformer_udf import _transformer_udf

# See test_transformer_udf.py: the nameless-input warning is a known false
# positive; both the oracle UDF and the Rust infer path emit it.
pytestmark = pytest.mark.filterwarnings(
    "ignore:X does not have valid feature names:UserWarning"
)


def _parity(sql, table, obj, in_schema, out_schema, name="__tfm_0__"):
    # Oracle: DataFusion with the transformer registered as a UDF.
    ctx = SessionContext()
    ctx.from_arrow(table, name="__THIS__")
    ctx.register_udf(_transformer_udf(obj, in_schema, out_schema, name))
    df = ctx.sql(sql)
    oracle = pa.Table.from_batches(df.collect(), schema=df.schema()).to_pylist()

    # Rust infer with the same object in the transformers registry.
    model = synthesize_this_model(table.schema)
    rows = [model(**r) for r in table.to_pylist()]
    fn = InferFn(
        sql,
        row_tables={"__THIS__": model},
        static_tables={},
        transformers={name: (obj, out_schema)},
    )
    actual = [r.model_dump() for r in fn.infer({"__THIS__": rows})]

    assert _rows_equal(actual, oracle), (sql, actual, oracle)
    return oracle


def test_standard_scaler_parity():
    train = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]})
    sc = StandardScaler().fit(train)
    schema = pa.schema([("age", pa.float64()), ("income", pa.float64())])
    sql = "SELECT __tfm_0__(named_struct('age', age, 'income', income)) AS s FROM __THIS__"
    _parity(sql, pa.Table.from_pandas(train), sc, schema, schema)


def test_struct_field_order_independence_parity():
    # The named_struct is authored in the OPPOSITE order to feature_names_in_
    # (income, age vs age, income). Both engines must align by NAME, not
    # position -- a positional bug in either would diverge here.
    train = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]})
    sc = StandardScaler().fit(train)  # feature_names_in_ == [age, income]
    in_schema = pa.schema([("income", pa.float64()), ("age", pa.float64())])
    out_schema = pa.schema([("age", pa.float64()), ("income", pa.float64())])
    sql = "SELECT __tfm_0__(named_struct('income', income, 'age', age)) AS s FROM __THIS__"
    _parity(sql, pa.Table.from_pandas(train), sc, in_schema, out_schema)


def test_whole_pipeline_parity():
    train = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]})
    pipe = Pipeline([("sc", StandardScaler()), ("pt", PowerTransformer())]).fit(train)
    schema = pa.schema([("age", pa.float64()), ("income", pa.float64())])
    sql = "SELECT __tfm_0__(named_struct('age', age, 'income', income)) AS s FROM __THIS__"
    _parity(sql, pa.Table.from_pandas(train), pipe, schema, schema)


def test_ordinal_encoder_non_float_in_and_out_parity():
    train = pd.DataFrame(
        {"color": ["red", "green", "blue", "red"], "size": ["S", "M", "L", "M"]}
    )
    enc = OrdinalEncoder(dtype=np.int64).fit(train)  # string in, int out
    in_schema = pa.schema([("color", pa.string()), ("size", pa.string())])
    out_schema = pa.schema([("color", pa.int64()), ("size", pa.int64())])
    # pa.Table.from_pandas defaults object-dtype columns to large_string,
    # which doesn't match the Utf8 in_schema the UDF is registered with; cast
    # to align the physical table type with the declared schema.
    table = pa.Table.from_pandas(train).cast(in_schema)
    sql = "SELECT __tfm_0__(named_struct('color', color, 'size', size)) AS s FROM __THIS__"
    _parity(sql, table, enc, in_schema, out_schema)
