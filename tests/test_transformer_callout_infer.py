"""Direct `infer` tests for the Expr::Transform callout (no oracle).

Parity with DataFusion is proven separately in test_diff_transformer_callout.py;
these pin the Rust marshalling to hand-computed values and the build-time errors.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from sklearn.preprocessing import StandardScaler

from sql_transform._interpreter import InferFn
from sql_transform._schema import synthesize_this_model

# See test_transformer_udf.py: the nameless-input warning is a known false positive.
pytestmark = pytest.mark.filterwarnings(
    "ignore:X does not have valid feature names:UserWarning"
)

_THIS = pa.schema([("age", pa.float64()), ("income", pa.float64())])
_OUT = pa.schema([("age", pa.float64()), ("income", pa.float64())])
_SQL = "SELECT __tfm_0__(named_struct('age', age, 'income', income)) AS s FROM __THIS__"


def _fitted_scaler():
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    return StandardScaler().fit(train), train


def _infer(sql, transformers, rows):
    model = synthesize_this_model(_THIS)
    fn = InferFn(
        sql,
        row_tables={"__THIS__": model},
        static_tables={},
        transformers=transformers,
    )
    return [r.model_dump() for r in fn.infer({"__THIS__": [model(**r) for r in rows]})]


def test_standard_scaler_infer_hand_computed():
    sc, _ = _fitted_scaler()
    out = _infer(_SQL, {"__tfm_0__": (sc, _OUT)}, [{"age": 10.0, "income": 1.0}])
    # population std (ddof=0): age mean 25, std 11.18034; income mean 2.5, std 1.11803
    assert abs(out[0]["s"]["age"] - ((10.0 - 25.0) / 11.180339887498949)) < 1e-9
    assert abs(out[0]["s"]["income"] - ((1.0 - 2.5) / 1.1180339887498949)) < 1e-9


def test_missing_feature_names_in_is_build_error():
    sc = StandardScaler().fit(np.array([[10.0, 1.0], [20.0, 2.0]]))  # bare array
    with pytest.raises(ValueError, match="feature_names_in_"):
        _infer(_SQL, {"__tfm_0__": (sc, _OUT)}, [{"age": 10.0, "income": 1.0}])


def test_non_struct_argument_is_build_error():
    sc, _ = _fitted_scaler()
    sql = "SELECT __tfm_0__(age) AS s FROM __THIS__"
    with pytest.raises(ValueError, match="must be a struct"):
        _infer(sql, {"__tfm_0__": (sc, _OUT)}, [{"age": 10.0, "income": 1.0}])


def test_field_name_mismatch_is_build_error():
    sc, _ = _fitted_scaler()
    sql = (
        "SELECT __tfm_0__(named_struct('age', age, 'wrong', income)) AS s FROM __THIS__"
    )
    with pytest.raises(ValueError, match="feature_names_in_"):
        _infer(sql, {"__tfm_0__": (sc, _OUT)}, [{"age": 10.0, "income": 1.0}])
