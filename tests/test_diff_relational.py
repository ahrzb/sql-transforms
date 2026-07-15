"""Differential coverage of the Rust engine's relational surface."""

import pytest
from differential import check, rows, static


def test_where_filters_rows():
    check(
        "SELECT age FROM t WHERE age >= 18",
        {"t": rows({"age": "int"}, [{"age": 10}, {"age": 18}, {"age": 40}])},
        expect=[{"age": 18}, {"age": 40}],
    )


def test_cross_join():
    check(
        "SELECT a.x, b.y FROM a, b",
        {
            "a": rows({"id": "int", "x": "int"}, [{"id": 1, "x": 10}]),
            "b": rows({"id": "int", "y": "int"}, [{"id": 1, "y": 20}]),
        },
    )


def test_inner_join_multiple_rows():
    check(
        "SELECT a.x, b.y FROM a JOIN b ON a.id = b.id",
        {
            "a": rows(
                {"id": "int", "x": "int"}, [{"id": 1, "x": 10}, {"id": 2, "x": 20}]
            ),
            "b": rows(
                {"id": "int", "y": "int"}, [{"id": 1, "y": 100}, {"id": 2, "y": 200}]
            ),
        },
    )


def test_row_static_lookup_join():
    check(
        "SELECT data.x, ref.y FROM data JOIN ref ON data.id = ref.id",
        {
            "data": rows(
                {"id": "int", "x": "int"}, [{"id": 1, "x": 5}, {"id": 2, "x": 6}]
            ),
            "ref": static(
                {"id": "int", "y": "int"}, [{"id": 1, "y": 10}, {"id": 2, "y": 20}]
            ),
        },
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Confirmed Rust-engine bug, not a harness mistake: InferFn's output-"
        "schema synthesis infers nullability from each table's own declared "
        "schema only (src/types.rs resolve_column_type, fed by "
        "src/plan.rs::validate_columns' effective_schemas), which never "
        "threads the `outer: bool` flag on RelNode::Join/LookupJoin. So the "
        "synthesized OutputRow model declares `ref.y` as a required "
        "(non-nullable) int even though this is a LEFT JOIN. At runtime the "
        "unmatched id=99 row legitimately produces y=None, and constructing "
        "OutputRow(**row) in _run_infer raises "
        "pydantic_core.ValidationError (int_type on None) -- an InferFn-"
        "internal crash, before DataFusion's output is even compared. "
        "DataFusion has no such issue (LEFT JOIN columns are nullable "
        "there). Fix belongs in src/types.rs/plan.rs: an outer join's "
        "right-side (or LookupJoin's lookup-side) columns must be widened "
        "to nullable when computing effective_schemas."
    ),
)
def test_left_lookup_join_hit_and_miss():
    # id=99 has no match -> LEFT JOIN yields NULL for ref.y on BOTH engines.
    check(
        "SELECT data.x, ref.y FROM data LEFT JOIN ref ON data.id = ref.id",
        {
            "data": rows(
                {"id": "int", "x": "int"}, [{"id": 1, "x": 5}, {"id": 99, "x": 6}]
            ),
            "ref": static({"id": "int", "y": "int"}, [{"id": 1, "y": 10}]),
        },
        expect=[{"x": 5, "y": 10}, {"x": 6, "y": None}],
    )


def test_multi_row_projection_all_rows_present():
    check(
        "SELECT age, age * 2 AS d FROM t",
        {"t": rows({"age": "int"}, [{"age": 1}, {"age": 2}, {"age": 3}])},
    )


def test_composite_key_lookup():
    check(
        "SELECT d.v, r.z FROM d JOIN r ON d.a = r.a AND d.b = r.b",
        {
            "d": rows({"a": "int", "b": "int", "v": "int"}, [{"a": 1, "b": 2, "v": 7}]),
            "r": static(
                {"a": "int", "b": "int", "z": "int"}, [{"a": 1, "b": 2, "z": 9}]
            ),
        },
    )
