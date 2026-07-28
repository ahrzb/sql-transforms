"""Differential coverage of the Rust engine's relational surface."""

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


def test_left_lookup_join_hit_and_miss():
    # id=99 has no match -> LEFT JOIN yields NULL for ref.y on BOTH engines.
    # Regression: the synthesized output model must type ref.y nullable (outer
    # side of the LEFT join), so the unmatched y=None validates instead of raising.
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
