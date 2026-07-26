"""Tests for SpecializedTransform — the specializer-served authoring surface."""

import pyarrow as pa
import pytest
from pydantic import BaseModel

from sql_transform import SpecializedTransform, SQLTransform

TRAIN = pa.table({"age": [25, 30, 35], "city": ["a", "b", "a"]})


def test_infer_before_fit_raises_runtime_error():
    t = SpecializedTransform("SELECT age FROM __THIS__")
    with pytest.raises(RuntimeError):
        t.infer({"age": 1})


def test_plain_projection_dict_and_model_rows():
    t = SpecializedTransform(
        "SELECT age * 2 AS a2, upper(city) AS c FROM __THIS__"
    ).fit(TRAIN)
    assert t.backend == "cranelift"
    assert t.boundary == "marshaller"

    class Row(BaseModel):
        age: int
        city: str

    got_dict = t.infer({"age": 21, "city": "xy"})
    got_model = t.infer(Row(age=21, city="xy"))
    for got in (got_dict, got_model):
        assert got.a2 == 42
        assert got.c == "XY"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__",
        "SELECT age - MEAN(age) OVER (PARTITION BY city) AS d, city FROM __THIS__",
    ],
)
def test_window_aggregates_agree_with_sqltransform(sql):
    spec = SpecializedTransform(sql).fit(TRAIN)
    ref = SQLTransform(sql).fit(TRAIN)
    assert spec.backend == "cranelift"
    rows = [{"age": 40, "city": "a"}, {"age": 25, "city": "b"}]
    got = spec.infer_batch(rows)
    want = ref.infer_batch(rows)
    assert [m.model_dump() for m in got] == [m.model_dump() for m in want]


def test_transformer_ref_rejected_at_construction():
    class FakeFitted:
        n_features_in_ = 1

        def transform(self, x):
            return x

    scaler = FakeFitted()
    with pytest.raises(ValueError, match="transformer refs"):
        SpecializedTransform(t"SELECT {scaler}(age) AS s FROM __THIS__")
