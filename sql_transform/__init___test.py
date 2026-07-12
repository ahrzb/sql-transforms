"""Tests for SQLTransform."""

import pyarrow as pa


def assert_approx_equal(actual: list, expected: list) -> None:
    for a, e in zip(actual, expected, strict=True):
        assert abs(a - e) < 0.001


def test_fit_and_transform_batch_no_agg():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age FROM data")
    data = pa.table({"age": [1, 2, 3]})
    result = t.fit(data).transform(data)
    assert result.column("age").to_pylist() == [1, 2, 3]


def test_fit_and_transform_constant_agg():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM data")
    data = pa.table({"age": [25, 30, 35]})
    result = t.fit(data).transform(data)
    assert_approx_equal(
        result.column("age_norm").to_pylist(),
        [25 / 30, 30 / 30, 35 / 30],
    )


def test_transform_on_unseen_data():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM data")
    train = pa.table({"age": [25, 30, 35]})
    test_data = pa.table({"age": [40, 50]})
    result = t.fit(train).transform(test_data)
    assert_approx_equal(
        result.column("age_norm").to_pylist(),
        [40 / 30, 50 / 30],
    )


def test_single_row_inference():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM data")
    t.fit(pa.table({"age": [25, 30, 35]}))
    result = t._infer({"age": 40})
    assert abs(result["age_norm"] - 40 / 30) < 0.001


def test_partitioned_agg_transform():
    from sql_transform import SQLTransform

    sql = "SELECT MEAN(target) OVER (PARTITION BY city) AS city_enc FROM data"
    t = SQLTransform(sql)
    data = pa.table({"city": ["a", "b", "a", "b"], "target": [1.0, 2.0, 3.0, 4.0]})
    result = t.fit(data).transform(data)
    assert_approx_equal(
        result.column("city_enc").to_pylist(),
        [2.0, 3.0, 2.0, 3.0],
    )


def test_partitioned_single_row_inference():
    from sql_transform import SQLTransform

    sql = "SELECT MEAN(target) OVER (PARTITION BY city) AS city_enc FROM data"
    t = SQLTransform(sql)
    t.fit(pa.table({"city": ["a", "b", "a", "b"], "target": [1.0, 2.0, 3.0, 4.0]}))

    result = t._infer({"city": "a", "target": 10.0})
    assert abs(result["city_enc"] - 2.0) < 0.001


def test_multiple_columns():
    from sql_transform import SQLTransform

    sql = (
        "SELECT age / MEAN(age) OVER () AS age_norm, "
        "score / SUM(score) OVER () AS score_norm FROM data"
    )
    t = SQLTransform(sql)
    data = pa.table({"age": [25, 30, 35], "score": [10, 20, 30]})
    result = t.fit(data).transform(data)
    assert "age_norm" in result.schema.names
    assert "score_norm" in result.schema.names


def test_fit_returns_self():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age FROM data")
    result = t.fit(pa.table({"age": [1]}))
    assert result is t


def test_from_file(tmp_path):
    from sql_transform import SQLTransform

    sql_file = tmp_path / "features.sql"
    sql_file.write_text("SELECT age / MEAN(age) OVER () AS x FROM data")

    t = SQLTransform.from_file(str(sql_file))
    t.fit(pa.table({"age": [1, 2, 3]}))
    result = t._infer({"age": 10})
    assert "x" in result


def test_e2e_three_transforms():
    """End-to-end: fit on training, transform batch, infer single row."""
    from sql_transform import SQLTransform

    sql = """
    SELECT
        age / MEAN(age) OVER () AS age_norm,
        income / SUM(income) OVER () AS income_share,
        MEAN(target) OVER (PARTITION BY city) AS city_enc
    FROM data
    """

    t = SQLTransform(sql)
    train = pa.table(
        {
            "age": [25, 30, 35, 40],
            "income": [50_000, 60_000, 70_000, 80_000],
            "city": ["paris", "paris", "tehran", "tehran"],
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )

    t.fit(train)

    # Batch transform
    out = t.transform(train)
    assert out.schema.names == ["age_norm", "income_share", "city_enc"]
    assert len(out) == 4

    # Single-row inference
    row = {"age": 50, "income": 100_000, "city": "tehran", "target": 5.0}
    result = t._infer(row)

    mean_age = 32.5
    assert abs(result["age_norm"] - 50 / mean_age) < 0.001

    total_income = 260_000.0
    assert abs(result["income_share"] - 100_000 / total_income) < 0.001

    assert abs(result["city_enc"] - 3.5) < 0.001
