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


def test_referenced_transform_not_mutated():
    scaler, train = _fit_scaler()
    before = scaler.transform(train).to_pylist()
    SQLTransform(t"SELECT {scaler.transform}(age) AS s FROM __THIS__").fit(train)
    after = scaler.transform(train).to_pylist()
    assert before == after  # scaler still fitted + unchanged


def test_outer_aggregate_over_inlined_column_parity():
    scaler, train = _fit_scaler()
    composite = SQLTransform(
        t"SELECT {scaler.transform}(age) "
        t"/ AVG({scaler.transform}(age)) OVER () AS z FROM __THIS__"
    ).fit(train)
    _parity(composite, train)


def test_outer_aggregate_over_inlined_column_finite_parity():
    scaler, train = _fit_scaler()
    # scaled=(age-25)/std; MAX(scaled)=15/std; z=scaled/MAX(scaled)=(age-25)/15
    composite = SQLTransform(
        t"SELECT {scaler.transform}(age) "
        t"/ MAX({scaler.transform}(age)) OVER () AS z FROM __THIS__"
    ).fit(train)
    out = _parity(composite, train)  # asserts transform() vs infer() agree
    # train age = [10,20,30,40] -> z = [-1, -1/3, 1/3, 1] (std cancels)
    zs = sorted(r["z"] for r in out)
    for got, exp in zip(zs, [-1.0, -1 / 3, 1 / 3, 1.0], strict=True):
        assert abs(got - exp) < 1e-9, (zs,)


def test_composition_over_quoted_capitalized_column_parity():
    # Regression: a fitted transform composed over a quoted case-sensitive column
    # must keep the quoting when inlined; else the rewrite emits __THIS__.Age
    # (unquoted -> DataFusion folds to `age`) and fit() fails "No field named age".
    train = pa.table({"Age": [10.0, 20.0, 30.0, 40.0]})
    scaler = SQLTransform(
        'SELECT ("Age" - AVG("Age") OVER ()) / STDDEV("Age") OVER () AS s FROM __THIS__'
    ).fit(train)
    composite = SQLTransform(
        t'SELECT {scaler.transform}("Age") AS s2 FROM __THIS__'
    ).fit(train)
    out = _parity(composite, train)
    assert abs(out[0]["s2"] - ((10.0 - 25.0) / 12.909944487358056)) < 1e-9
