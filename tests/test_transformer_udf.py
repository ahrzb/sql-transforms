"""Standalone oracle test: the DataFusion transformer UDF must match sklearn."""

import pandas as pd
import pyarrow as pa
import pytest
from datafusion import SessionContext
from sklearn.preprocessing import StandardScaler

from sql_transform._transformer_udf import _transformer_udf

# Both engines deliberately feed sklearn positionally-aligned nameless arrays
# (we reorder to feature_names_in_ ourselves), so sklearn's redundant
# "X does not have valid feature names" warning is a known false positive here.
pytestmark = pytest.mark.filterwarnings(
    "ignore:X does not have valid feature names:UserWarning"
)


def _collect(df) -> list[dict]:
    return pa.Table.from_batches(df.collect(), schema=df.schema()).to_pylist()


def test_standard_scaler_udf_matches_sklearn():
    train_df = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    sc = StandardScaler().fit(train_df)  # fit on NAMED data -> feature_names_in_
    in_schema = pa.schema([("age", pa.float64()), ("income", pa.float64())])
    out_schema = pa.schema([("age", pa.float64()), ("income", pa.float64())])

    table = pa.Table.from_pandas(train_df)
    ctx = SessionContext()
    ctx.from_arrow(table, name="__THIS__")
    ctx.register_udf(_transformer_udf(sc, in_schema, out_schema, "__tfm_0__"))

    q = (
        "SELECT __tfm_0__(named_struct('age', age, 'income', income)) "
        "AS s FROM __THIS__"
    )
    got = _collect(ctx.sql(q))

    expected = sc.transform(train_df)
    assert len(got) == 4
    for i, r in enumerate(got):
        assert abs(r["s"]["age"] - expected[i][0]) < 1e-9
        assert abs(r["s"]["income"] - expected[i][1]) < 1e-9
