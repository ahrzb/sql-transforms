"""Cases where both engines reject a query (build or execution errors)."""

from differential import check_both_raise, rows


def test_unknown_column_rejected_by_both():
    check_both_raise("SELECT nope FROM t", {"t": rows({"age": "int"}, [{"age": 1}])})


def test_self_join_rejected_by_both():
    check_both_raise(
        "SELECT a.x FROM a JOIN a ON a.id = a.id",
        {"a": rows({"id": "int", "x": "int"}, [{"id": 1, "x": 1}])},
    )


def test_int_div_by_zero_rejected_by_both():
    # Both engines raise on integer division by zero (error TYPES differ -- Rust
    # raises its own error, DataFusion an Arrow DivideByZero -- but both reject,
    # which is what check_both_raise asserts).
    check_both_raise(
        "SELECT a / b AS q FROM t",
        {"t": rows({"a": "int", "b": "int"}, [{"a": 1, "b": 0}])},
    )


def test_int_mod_by_zero_rejected_by_both():
    check_both_raise(
        "SELECT a % b AS r FROM t",
        {"t": rows({"a": "int", "b": "int"}, [{"a": 5, "b": 0}])},
    )


def test_unknown_function_rejected_by_both():
    # foobar() is invalid SQL -- DataFusion errors -- and is NOT a `__tfm_`
    # transformer placeholder, so codegen must raise a hard error too (not
    # UnsupportedInCodegen, which the harness would skip as a deferred surface).
    check_both_raise(
        "SELECT foobar(x) AS y FROM t",
        {"t": rows({"x": "int"}, [{"x": 1}])},
    )
