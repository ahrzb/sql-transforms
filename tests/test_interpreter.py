"""Tests for the Rust SQL interpreter (sql_transform._interpreter).

Every behavioral test compares InferFn output against real DataFusion
batch output for the same SQL + data, per the Phase 2 spec's testing
strategy.
"""

import datafusion
import pytest
from sql_transform._interpreter import InferFn


def test_module_imports_and_constructs():
    fn = InferFn("SELECT age FROM data", row_tables=["data"], static_tables={})
    assert fn is not None


def _expected(sql: str, data: dict) -> list[dict]:
    ctx = datafusion.SessionContext()
    ctx.from_pydict(data, name="data")
    return ctx.sql(sql).collect()[0].to_pylist()


def test_column_pass_through():
    sql = "SELECT age FROM data"
    data = {"age": [30]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"age": 30}]})
    assert actual == _expected(sql, data)


def test_multiple_columns():
    sql = "SELECT a, b FROM data"
    data = {"a": [1], "b": ["x"]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"a": 1, "b": "x"}]})
    assert actual == _expected(sql, data)


def test_literal():
    sql = "SELECT 42 AS x FROM data"
    data = {"age": [1]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"age": 1}]})
    assert actual == _expected(sql, data)


def test_arithmetic():
    sql = "SELECT age / 2 AS half FROM data"
    data = {"age": [30]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"age": 30}]})
    assert actual == _expected(sql, data)


def test_arithmetic_precedence():
    sql = "SELECT (a + b) * c AS x FROM data"
    data = {"a": [2], "b": [3], "c": [4]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"a": 2, "b": 3, "c": 4}]})
    assert actual == _expected(sql, data)


def test_negative_integer_division_truncates():
    sql = "SELECT c / b AS x, c % b AS y FROM data"
    data = {"c": [-7], "b": [2]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"c": -7, "b": 2}]})
    assert actual == _expected(sql, data)


def test_mixed_int_float_division():
    sql = "SELECT a / f AS x FROM data"
    data = {"a": [7], "f": [2.5]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"a": 7, "f": 2.5}]})
    assert actual == _expected(sql, data)


def test_null_comparison_is_null():
    sql = "SELECT age > 100 AS gt, age > NULL AS gtnull FROM data"
    data = {"age": [30]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"age": 30}]})
    assert actual == _expected(sql, data)


def test_and_or_three_valued_logic():
    # age > 100 evaluates to FALSE for age=30, giving a real (non-literal)
    # FALSE operand alongside literal NULL/TRUE to exercise SQL's
    # three-valued AND/OR truth tables, including the asymmetric cases
    # where a FALSE (for AND) or TRUE (for OR) operand short-circuits a
    # NULL operand to a definite result instead of propagating NULL.
    sql = (
        "SELECT "
        "age > 100 AND NULL AS false_and_null, "
        "NULL AND age > 100 AS null_and_false, "
        "NULL AND TRUE AS null_and_true, "
        "NULL OR TRUE AS null_or_true, "
        "TRUE OR NULL AS true_or_null, "
        "NULL OR FALSE AS null_or_false "
        "FROM data"
    )
    data = {"age": [30]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"age": 30}]})
    assert actual == _expected(sql, data)


def test_where_filter():
    sql = "SELECT x FROM data WHERE x > 5"
    data = {"x": [3, 7, 10]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"x": 3}, {"x": 7}, {"x": 10}]})
    assert actual == _expected(sql, data)


def test_multi_row():
    sql = "SELECT age FROM data"
    data = {"age": [1, 2, 3]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"age": 1}, {"age": 2}, {"age": 3}]})
    assert actual == _expected(sql, data)


def test_builtin_upper():
    sql = "SELECT UPPER(name) AS up FROM data"
    data = {"name": ["hello"]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"name": "hello"}]})
    assert actual == _expected(sql, data)


def test_builtin_concat():
    sql = "SELECT CONCAT(a, '-', b) AS combo FROM data"
    data = {"a": ["x"], "b": ["y"]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"a": "x", "b": "y"}]})
    assert actual == _expected(sql, data)


def test_builtin_substr_trim_abs_round():
    sql = (
        "SELECT SUBSTR(s, 2, 3) AS sub, TRIM(pad) AS t, "
        "ABS(neg) AS ab, ROUND(pi) AS ro FROM data"
    )
    data = {"s": ["hello"], "pad": ["  x  "], "neg": [-5], "pi": [3.6]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"s": "hello", "pad": "  x  ", "neg": -5, "pi": 3.6}]})
    assert actual == _expected(sql, data)


def test_builtin_coalesce_nullif():
    sql = "SELECT COALESCE(a, b, 5) AS co, NULLIF(x, x) AS ni FROM data"
    data = {"a": [None], "b": [None], "x": [3]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"a": None, "b": None, "x": 3}]})
    assert actual == _expected(sql, data)


def test_cast():
    sql = "SELECT CAST(age AS VARCHAR) AS s FROM data"
    data = {"age": [42]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"age": 42}]})
    assert actual == _expected(sql, data)


def test_substring_for_only_takes_first_n_chars():
    # SQL-92: SUBSTRING(expr FOR n) with no FROM means "the first n
    # characters", equivalent to SUBSTRING(expr FROM 1 FOR n) — not
    # "from position n to the end".
    sql = "SELECT SUBSTRING(s FOR 3) AS sub FROM data"
    data = {"s": ["hello"]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"s": "hello"}]})
    assert actual == _expected(sql, data)


def test_substring_from_only():
    sql = "SELECT SUBSTRING(s FROM 2) AS sub FROM data"
    data = {"s": ["hello"]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"s": "hello"}]})
    assert actual == _expected(sql, data)


def test_substring_bare_form_rejected_at_build_time():
    with pytest.raises(ValueError):
        InferFn(
            "SELECT SUBSTRING(s) AS x FROM data",
            row_tables=["data"],
            static_tables={},
        )


def test_cross_join():
    sql = "SELECT a.x, b.y FROM a, b"
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"x": [1]}, name="a")
    ctx.from_pydict({"y": [2]}, name="b")
    expected = ctx.sql(sql).collect()[0].to_pylist()

    fn = InferFn(sql, row_tables=["a", "b"], static_tables={})
    actual = fn.infer({"a": [{"x": 1}], "b": [{"y": 2}]})
    assert actual == expected


def test_inner_join_two_row_tables():
    sql = "SELECT a.x, b.y FROM a JOIN b ON a.id = b.id"
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"id": [1, 2], "x": [10, 20]}, name="a")
    ctx.from_pydict({"id": [1, 2], "y": [100, 200]}, name="b")
    expected = ctx.sql(sql).collect()[0].to_pylist()

    fn = InferFn(sql, row_tables=["a", "b"], static_tables={})
    actual = fn.infer(
        {
            "a": [{"id": 1, "x": 10}, {"id": 2, "x": 20}],
            "b": [{"id": 1, "y": 100}, {"id": 2, "y": 200}],
        }
    )
    assert actual == expected


def test_inner_join_multi_key():
    sql = "SELECT a.x, b.y FROM a JOIN b ON a.k1 = b.k1 AND a.k2 = b.k2"
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"k1": [1], "k2": ["p"], "x": [10]}, name="a")
    ctx.from_pydict({"k1": [1], "k2": ["p"], "y": [100]}, name="b")
    expected = ctx.sql(sql).collect()[0].to_pylist()

    fn = InferFn(sql, row_tables=["a", "b"], static_tables={})
    actual = fn.infer(
        {
            "a": [{"k1": 1, "k2": "p", "x": 10}],
            "b": [{"k1": 1, "k2": "p", "y": 100}],
        }
    )
    assert actual == expected


def test_error_left_join():
    sql = "SELECT a.x FROM a LEFT JOIN b ON a.id = b.id"
    with pytest.raises(ValueError):
        InferFn(sql, row_tables=["a", "b"], static_tables={})


def test_error_non_equality_on():
    sql = "SELECT a.x FROM a JOIN b ON a.id > b.id"
    with pytest.raises(ValueError):
        InferFn(sql, row_tables=["a", "b"], static_tables={})


def test_error_self_join():
    sql = "SELECT a.x FROM a JOIN a ON a.id = a.id"
    with pytest.raises(ValueError):
        InferFn(sql, row_tables=["a"], static_tables={})
