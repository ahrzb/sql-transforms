from differential import check, rows


def test_struct_construct():
    check(
        "SELECT named_struct('x', a, 'y', b) AS s FROM t",
        {"t": rows({"a": "int", "b": "int"}, [{"a": 1, "b": 2}])},
    )


def test_list_construct():
    check(
        "SELECT [a, b, a] AS l FROM t",
        {"t": rows({"a": "int", "b": "int"}, [{"a": 1, "b": 2}])},
    )


def test_struct_input_roundtrip():
    check(
        "SELECT s FROM t",
        {"t": rows({"s": "struct{x:int,y:int}"}, [{"s": {"x": 1, "y": 2}}])},
    )


def test_list_input_roundtrip():
    check(
        "SELECT l FROM t",
        {"t": rows({"l": "list[int]"}, [{"l": [1, 2, 3]}])},
    )
