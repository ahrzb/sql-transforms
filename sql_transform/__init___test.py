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
        t.infer({"age": 1})


def test_fit_rejects_where_clause():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age FROM __THIS__ WHERE age > 1")
    with pytest.raises(ValueError, match="WHERE"):
        t.fit(pa.table({"age": [1, 2, 3]}))


def test_fit_rejects_wrong_from_table():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age FROM data")
    with pytest.raises(ValueError, match="__THIS__"):
        t.fit(pa.table({"age": [1, 2, 3]}))


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
    result = t.infer({"age": 40})
    assert abs(result.age_norm - 40 / 30) < 0.001


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
    result = t.infer({"age": 10})
    assert hasattr(result, "x")


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
            # Floats: SUM(income) now preserves its real Arrow type (no more
            # float-coercing state model), so an int column would make this
            # division integer division.
            "income": [50_000.0, 60_000.0, 70_000.0, 80_000.0],
        }
    )

    t.fit(train)

    out = t.transform(train)
    assert out.schema.names == ["age_norm", "income_share"]
    assert len(out) == 4

    row = {"age": 50, "income": 100_000.0}
    result = t.infer(row)

    mean_age = 32.5
    assert abs(result.age_norm - 50 / mean_age) < 0.001

    total_income = 260_000.0
    assert abs(result.income_share - 100_000 / total_income) < 0.001


def test_this_model_omitted_synthesizes_from_table_schema():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age FROM __THIS__")
    t.fit(pa.table({"age": [1, 2, 3]}))
    result = t.infer({"age": 5})
    assert result.age == 5


def test_this_model_supplied_compatible():
    from sql_transform import SQLTransform

    class Row(BaseModel):
        age: int

    t = SQLTransform("SELECT age FROM __THIS__")
    t.fit(pa.table({"age": [1, 2, 3]}), this_model=Row)
    result = t.infer({"age": 7})
    assert result.age == 7


def test_this_model_supplied_missing_referenced_column_raises():
    from sql_transform import SQLTransform

    class IncompleteRow(BaseModel):
        other: int  # doesn't declare "age", which the query references

    t = SQLTransform("SELECT age FROM __THIS__")
    with pytest.raises(ValueError):
        t.fit(pa.table({"age": [1, 2, 3]}), this_model=IncompleteRow)


def test_state_tables_hold_typed_values():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
    t.fit(pa.table({"age": [25, 30, 35]}))
    # State is a dict of pyarrow tables, not a Pydantic model.
    assert t._state_tables["__STATE__"].column("avg_age").to_pylist() == [30.0]


def test_infer_accepts_pydantic_model():
    from sql_transform import SQLTransform

    class Row(BaseModel):
        age: int

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
    t.fit(pa.table({"age": [25, 30, 35]}), this_model=Row)
    result = t.infer(Row(age=40))
    assert abs(result.age_norm - 40 / 30) < 0.001


def test_infer_batch_returns_typed_models():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
    t.fit(pa.table({"age": [25, 30, 35]}))
    out = t.infer_batch([{"age": 40}, {"age": 50}])
    assert len(out) == 2
    assert all(isinstance(o, BaseModel) for o in out)
    assert abs(out[0].age_norm - 40 / 30) < 0.001
    assert abs(out[1].age_norm - 50 / 30) < 0.001


def test_transform_and_infer_batch_agree():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
    t.fit(pa.table({"age": [25, 30, 35]}))

    test = pa.table({"age": [40, 50, 60]})
    batch = t.transform(test)
    rows = t.infer_batch([{"age": 40}, {"age": 50}, {"age": 60}])

    assert_approx_equal(
        batch.column("age_norm").to_pylist(), [r.age_norm for r in rows]
    )


@pytest.mark.xfail(
    reason="batch (DataFusion) surfaces its own error type, not the clean "
    "ValueError the Rust inference path raises -- see docs/BACKLOG.md",
    strict=True,
)
def test_transform_raises_clean_valueerror_on_div_by_zero():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT a / b AS x FROM __THIS__")
    t.fit(pa.table({"a": [1], "b": [1]}))
    with pytest.raises(ValueError):
        t.transform(pa.table({"a": [1], "b": [0]}))


def test_partition_by_target_encoding_seen_and_unseen():
    from sql_transform import SQLTransform

    t = SQLTransform(
        "SELECT MEAN(target) OVER (PARTITION BY city) AS enc FROM __THIS__"
    )
    t.fit(pa.table({"city": ["a", "b", "a", "b"], "target": [1.0, 3.0, 2.0, 4.0]}))

    seen = t.infer({"city": "a", "target": 0.0})
    assert seen.enc == 1.5

    unseen = t.infer({"city": "zzz", "target": 0.0})
    assert unseen.enc is None  # unseen partition -> NULL


def test_partition_by_count_encoding_is_integer():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT COUNT(target) OVER (PARTITION BY city) AS n FROM __THIS__")
    t.fit(pa.table({"city": ["a", "a", "b"], "target": [1, 2, 3]}))
    out = t.infer({"city": "a", "target": 0})
    assert out.n == 2
    assert isinstance(out.n, int)  # count encoding stays an int, not 2.0


def test_partition_by_transform_is_one_to_one_and_matches_infer():
    from sql_transform import SQLTransform

    t = SQLTransform(
        "SELECT MEAN(target) OVER (PARTITION BY city) AS enc FROM __THIS__"
    )
    t.fit(pa.table({"city": ["a", "b", "a", "b"], "target": [1.0, 3.0, 2.0, 4.0]}))

    batch = pa.table({"city": ["a", "b", "zzz"], "target": [0.0, 0.0, 0.0]})
    out = t.transform(batch)
    assert out.num_rows == 3  # strictly 1-to-1, unseen row preserved

    rows = t.infer_batch(
        [
            {"city": "a", "target": 0.0},
            {"city": "b", "target": 0.0},
            {"city": "zzz", "target": 0.0},
        ]
    )
    assert out.column("enc").to_pylist() == [r.enc for r in rows]
    assert out.column("enc").to_pylist()[2] is None  # unseen -> NULL, both engines
