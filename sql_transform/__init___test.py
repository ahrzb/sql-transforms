"""Tests for SQLTransform."""

import pyarrow as pa
import pytest
from pydantic import BaseModel


def assert_approx_equal(actual: list, expected: list) -> None:
    for a, e in zip(actual, expected, strict=True):
        assert abs(a - e) < 0.001


def test_transform_before_fit_raises_runtime_error():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age FROM __THIS__")
    with pytest.raises(RuntimeError):
        t.transform(pa.table({"age": [1]}))


def test_infer_before_fit_raises_runtime_error():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age FROM __THIS__")
    with pytest.raises(RuntimeError):
        t._infer({"age": 1})


def test_fit_and_transform_batch_no_agg():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age FROM __THIS__")
    data = pa.table({"age": [1, 2, 3]})
    result = t.fit(data).transform(data)
    assert result.column("age").to_pylist() == [1, 2, 3]


def test_fit_and_transform_constant_agg():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
    data = pa.table({"age": [25, 30, 35]})
    result = t.fit(data).transform(data)
    assert_approx_equal(
        result.column("age_norm").to_pylist(),
        [25 / 30, 30 / 30, 35 / 30],
    )


def test_transform_on_unseen_data():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
    train = pa.table({"age": [25, 30, 35]})
    test_data = pa.table({"age": [40, 50]})
    result = t.fit(train).transform(test_data)
    assert_approx_equal(
        result.column("age_norm").to_pylist(),
        [40 / 30, 50 / 30],
    )


def test_single_row_inference():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
    t.fit(pa.table({"age": [25, 30, 35]}))
    result = t._infer({"age": 40})
    assert abs(result["age_norm"] - 40 / 30) < 0.001


def test_multiple_columns():
    from sql_transform import SQLTransform

    sql = (
        "SELECT age / MEAN(age) OVER () AS age_norm, "
        "score / SUM(score) OVER () AS score_norm FROM __THIS__"
    )
    t = SQLTransform(sql)
    data = pa.table({"age": [25, 30, 35], "score": [10, 20, 30]})
    result = t.fit(data).transform(data)
    assert "age_norm" in result.schema.names
    assert "score_norm" in result.schema.names


def test_fit_returns_self():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age FROM __THIS__")
    result = t.fit(pa.table({"age": [1]}))
    assert result is t


def test_from_file(tmp_path):
    from sql_transform import SQLTransform

    sql_file = tmp_path / "features.sql"
    sql_file.write_text("SELECT age / MEAN(age) OVER () AS x FROM __THIS__")

    t = SQLTransform.from_file(str(sql_file))
    t.fit(pa.table({"age": [1, 2, 3]}))
    result = t._infer({"age": 10})
    assert "x" in result


def test_e2e_two_transforms_and_dedup():
    """End-to-end: fit on training, transform batch, infer single row,
    with a repeated aggregate deduped across two projections."""
    from sql_transform import SQLTransform

    sql = """
    SELECT
        age / MEAN(age) OVER () AS age_norm,
        income / SUM(income) OVER () AS income_share
    FROM __THIS__
    """

    t = SQLTransform(sql)
    train = pa.table(
        {
            "age": [25, 30, 35, 40],
            "income": [50_000, 60_000, 70_000, 80_000],
        }
    )

    t.fit(train)

    out = t.transform(train)
    assert out.schema.names == ["age_norm", "income_share"]
    assert len(out) == 4

    row = {"age": 50, "income": 100_000}
    result = t._infer(row)

    mean_age = 32.5
    assert abs(result["age_norm"] - 50 / mean_age) < 0.001

    total_income = 260_000.0
    assert abs(result["income_share"] - 100_000 / total_income) < 0.001


def test_partitioned_agg_raises_not_implemented():
    from sql_transform import SQLTransform

    sql = "SELECT MEAN(target) OVER (PARTITION BY city) AS city_enc FROM __THIS__"
    t = SQLTransform(sql)
    data = pa.table({"city": ["a", "b"], "target": [1.0, 2.0]})
    with pytest.raises(NotImplementedError):
        t.fit(data)


def test_this_model_omitted_synthesizes_from_table_schema():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age FROM __THIS__")
    t.fit(pa.table({"age": [1, 2, 3]}))
    result = t._infer({"age": 5})
    assert result["age"] == 5


def test_this_model_supplied_compatible():
    from sql_transform import SQLTransform

    class Row(BaseModel):
        age: int

    t = SQLTransform("SELECT age FROM __THIS__")
    t.fit(pa.table({"age": [1, 2, 3]}), this_model=Row)
    result = t._infer({"age": 7})
    assert result["age"] == 7


def test_this_model_supplied_missing_referenced_column_raises():
    from sql_transform import SQLTransform

    class IncompleteRow(BaseModel):
        other: int  # doesn't declare "age", which the query references

    t = SQLTransform("SELECT age FROM __THIS__")
    with pytest.raises(ValueError):
        t.fit(pa.table({"age": [1, 2, 3]}), this_model=IncompleteRow)


def test_state_is_typed_pydantic_instance():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
    t.fit(pa.table({"age": [25, 30, 35]}))
    # DataFusion normalizes MEAN to avg internally, so the field is avg_age.
    assert isinstance(t._state.avg_age, float)
    assert t._state.avg_age == 30.0
