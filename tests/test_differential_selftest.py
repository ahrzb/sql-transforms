"""Tests OF the differential harness itself."""

import pytest
from differential import (
    _rows_equal,
    check,
    check_both_raise,
    row,
    rows,
    static,
)


def test_check_passes_on_agreement():
    check("SELECT a + b AS s FROM t", {"t": row(a=6, b=4)}, expect=[{"s": 10}])


def test_check_expect_mismatch_raises():
    with pytest.raises(AssertionError):
        check("SELECT a + b AS s FROM t", {"t": row(a=6, b=4)}, expect=[{"s": 999}])


def test_rows_equal_order_insensitive():
    assert _rows_equal([{"x": 1}, {"x": 2}], [{"x": 2}, {"x": 1}])
    assert not _rows_equal([{"x": 1}], [{"x": 1}, {"x": 2}])


def test_rows_equal_float_tolerance_and_null():
    assert _rows_equal([{"x": 1.0}], [{"x": 1.0 + 1e-12}])
    assert _rows_equal([{"x": None}], [{"x": None}])
    assert not _rows_equal([{"x": None}], [{"x": 1}])


def test_row_infers_types_and_check_multirow():
    check(
        "SELECT age FROM t WHERE age > 18",
        {"t": rows({"age": "int"}, [{"age": 15}, {"age": 25}, {"age": 40}])},
        expect=[{"age": 25}, {"age": 40}],
    )


def test_static_join_agrees():
    check(
        "SELECT data.x, ref.y FROM data JOIN ref ON data.id = ref.id",
        {
            "data": rows({"id": "int", "x": "int"}, [{"id": 1, "x": 10}]),
            "ref": static({"id": "int", "y": "str"}, [{"id": 1, "y": "a"}]),
        },
    )


def test_nullable_column_omitted_value():
    check(
        "SELECT UPPER(name) AS n FROM t",
        {"t": rows({"name": "str?"}, [{"name": "alice"}, {}])},  # 2nd row: name NULL
        expect=[{"n": "ALICE"}, {"n": None}],
    )


def test_check_both_raise_on_unknown_column():
    check_both_raise("SELECT nope FROM t", {"t": row(a=1)})


def test_rows_equal_distinguishes_bool_from_int():
    # bool is an int subclass in Python; the comparator must not conflate them.
    assert not _rows_equal([{"x": True}], [{"x": 1}])
    assert not _rows_equal([{"x": False}], [{"x": 0}])
    assert _rows_equal([{"x": True}], [{"x": True}])
