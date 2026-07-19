"""Differential coverage of SQL CASE (searched + simple forms).

CASE is codegen-only until the native engine gains it (TASK-27), so these run
against the DataFusion oracle on the codegen backend and skip on native via
codegen_only=True. Emission must short-circuit (only the taken branch evaluates),
which the short-circuit test below pins.
"""

from differential import check, rows


def test_case_searched_with_else():
    check(
        "SELECT CASE WHEN x > 0 THEN 'pos' WHEN x < 0 THEN 'neg' ELSE 'zero' END "
        "AS c FROM t",
        {"t": rows({"x": "int"}, [{"x": 5}, {"x": -3}, {"x": 0}])},
        expect=[{"c": "pos"}, {"c": "neg"}, {"c": "zero"}],
        codegen_only=True,
    )


def test_case_searched_no_else_unmatched_is_null():
    check(
        "SELECT CASE WHEN x > 0 THEN 1 END AS c FROM t",
        {"t": rows({"x": "int"}, [{"x": 5}, {"x": -1}])},
        expect=[{"c": 1}, {"c": None}],
        codegen_only=True,
    )


def test_case_simple_form():
    check(
        "SELECT CASE g WHEN 1 THEN 'a' WHEN 2 THEN 'b' ELSE 'z' END AS c FROM t",
        {"t": rows({"g": "int"}, [{"g": 1}, {"g": 2}, {"g": 9}])},
        expect=[{"c": "a"}, {"c": "b"}, {"c": "z"}],
        codegen_only=True,
    )


def test_case_simple_null_operand_falls_through():
    # NULL operand matches no WHEN value (NULL = v is NULL, not true) -> ELSE.
    check(
        "SELECT CASE g WHEN 1 THEN 'a' ELSE 'z' END AS c FROM t",
        {"t": rows({"g": "int?"}, [{"g": None}])},
        expect=[{"c": "z"}],
        codegen_only=True,
    )


def test_case_result_int_float_coerces_to_float():
    # Mixed int/float branches unify to float, matching the oracle (COALESCE rule).
    check(
        "SELECT CASE WHEN x > 0 THEN 1 ELSE 2.5 END AS c FROM t",
        {"t": rows({"x": "int"}, [{"x": 5}, {"x": -1}])},
        expect=[{"c": 1.0}, {"c": 2.5}],
        codegen_only=True,
    )


def test_case_short_circuits_avoiding_error():
    # The non-taken THEN (1 / x with x=0) must NOT be evaluated -- CASE is lazy.
    # If emission were eager this would raise "division by zero".
    check(
        "SELECT CASE WHEN x > 0 THEN 1 / x ELSE 0 END AS c FROM t",
        {"t": rows({"x": "int"}, [{"x": 0}])},
        expect=[{"c": 0}],
        codegen_only=True,
    )


def test_case_null_condition_skips_arm():
    # A NULL WHEN condition doesn't match (three-valued) -> next arm / ELSE.
    check(
        "SELECT CASE WHEN b THEN 'yes' ELSE 'no' END AS c FROM t",
        {"t": rows({"b": "bool?"}, [{"b": None}, {"b": True}])},
        expect=[{"c": "no"}, {"c": "yes"}],
        codegen_only=True,
    )


def test_case_nested():
    check(
        "SELECT CASE WHEN x > 0 THEN "
        "CASE WHEN x > 10 THEN 'big' ELSE 'small' END "
        "ELSE 'neg' END AS c FROM t",
        {"t": rows({"x": "int"}, [{"x": 20}, {"x": 5}, {"x": -1}])},
        expect=[{"c": "big"}, {"c": "small"}, {"c": "neg"}],
        codegen_only=True,
    )


def test_case_no_else_result_stays_int_in_arithmetic():
    # A no-ELSE CASE's THEN type must survive into outer arithmetic: the result
    # is int, not silently widened to float. Guards infer_type's no-ELSE typing.
    check(
        "SELECT (CASE WHEN x > 0 THEN 1 END) + 1 AS c FROM t",
        {"t": rows({"x": "int"}, [{"x": 5}])},
        expect=[{"c": 2}],
        codegen_only=True,
    )
