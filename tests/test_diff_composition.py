"""Differential parity for SQLTransform composition ({transform}(col))."""

import pyarrow as pa
from differential import _rows_equal

from sql_transform import SQLTransform


def _fit_scaler():
    train = pa.table({"age": [10.0, 20.0, 30.0, 40.0]})
    scaler = SQLTransform(
        "SELECT (age - AVG(age) OVER ()) / STDDEV(age) OVER () AS s FROM __THIS__"
    ).fit(train)
    return scaler, train


def _parity(composite, batch):
    rows = batch.to_pylist()
    batch_out = composite.transform(batch).to_pylist()
    infer_out = [r.model_dump() for r in composite.infer_batch(rows)]
    assert _rows_equal(batch_out, infer_out), (batch_out, infer_out)
    return batch_out


def test_single_reference_parity():
    scaler, train = _fit_scaler()
    composite = SQLTransform(
        t"SELECT {scaler.transform}(age) AS age_scaled FROM __THIS__"
    ).fit(train)
    out = _parity(composite, train)
    # (age - mean=25) / stddev(sample)=12.909944...
    assert abs(out[0]["age_scaled"] - ((10.0 - 25.0) / 12.909944487358056)) < 1e-9


def test_column_remap_parity():
    scaler, train = _fit_scaler()
    data = pa.table({"income": [10.0, 20.0, 30.0, 40.0]})
    composite = SQLTransform(
        t"SELECT {scaler.transform}(income) AS scaled FROM __THIS__"
    ).fit(train.append_column("income", train.column("age")))
    _parity(composite, data.append_column("age", data.column("income")))


def test_zero_state_inner_parity():
    train = pa.table({"age": [1.0, 2.0, 3.0]})
    doubler = SQLTransform("SELECT age * 2 AS d FROM __THIS__").fit(train)
    composite = SQLTransform(
        t"SELECT {doubler.transform}(age) AS d2 FROM __THIS__"
    ).fit(train)
    out = _parity(composite, train)
    assert out[0]["d2"] == 2.0


def test_repeated_and_multiple_references_parity():
    scaler, train = _fit_scaler()
    doubler = SQLTransform("SELECT age * 2 AS d FROM __THIS__").fit(train)
    composite = SQLTransform(
        t"SELECT {scaler.transform}(age) AS a, {scaler.transform}(age) AS b, "
        t"{doubler.transform}(age) AS c FROM __THIS__"
    ).fit(train)
    out = _parity(composite, train)
    assert out[0]["a"] == out[0]["b"]
    assert out[0]["c"] == 20.0  # doubler(age)=age*2 on train age[0]=10.0
