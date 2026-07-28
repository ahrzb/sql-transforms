"""End-to-end tests for the codegen engine.

Broad semantic parity is proven by the differential harness (tests/); these
cover the engine's own seams -- compilation, marshalling, output typing.
"""

import typing

import pyarrow as pa
import pytest
from pydantic import BaseModel

from sql_transform._codegen import CodegenFn


class Row(BaseModel):
    a: int
    b: float | None = None
    s: str = "x"


def test_projection_and_arithmetic():
    fn = CodegenFn("SELECT a + 1 AS x FROM t", {"t": Row}, {})
    assert [r.x for r in fn.infer({"t": [Row(a=1), Row(a=2)]})] == [2, 3]


def test_output_model_is_synthesized_with_inferred_types():
    fn = CodegenFn("SELECT a AS i, a / 2 AS q, s AS name FROM t", {"t": Row}, {})
    fields = fn.output_model.model_fields
    assert fields["i"].annotation is int
    assert fields["q"].annotation is int  # int / int stays int
    assert fields["name"].annotation is str


def test_nullable_column_yields_an_optional_output_field():
    fn = CodegenFn("SELECT b AS x FROM t", {"t": Row}, {})
    assert fn.output_model.model_fields["x"].annotation == typing.Optional[float]  # noqa: UP045
    assert fn.infer({"t": [Row(a=1)]})[0].x is None


def test_where_filters_rows():
    fn = CodegenFn("SELECT a AS x FROM t WHERE a > 1", {"t": Row}, {})
    assert [r.x for r in fn.infer({"t": [Row(a=1), Row(a=2), Row(a=3)]})] == [2, 3]


def test_where_drops_null_and_non_true_predicates():
    fn = CodegenFn("SELECT a AS x FROM t WHERE b > 1.0", {"t": Row}, {})
    assert fn.infer({"t": [Row(a=1)]}) == []  # b is None -> predicate NULL -> dropped


def test_table_alias():
    fn = CodegenFn("SELECT z.a AS x FROM t AS z", {"t": Row}, {})
    assert [r.x for r in fn.infer({"t": [Row(a=7)]})] == [7]


def test_infer_accepts_kwargs_as_well_as_a_tables_dict():
    fn = CodegenFn("SELECT a AS x FROM t", {"t": Row}, {})
    assert [r.x for r in fn.infer(t=[Row(a=5)])] == [5]


def test_only_referenced_columns_are_read_from_the_row():
    fn = CodegenFn("SELECT a AS x FROM t", {"t": Row}, {})

    class Partial:
        a = 3  # no b/s at all; the engine must not touch them

    assert [r.x for r in fn.infer({"t": [Partial()]})] == [3]


def test_missing_attribute_is_a_clear_error():
    fn = CodegenFn("SELECT a AS x FROM t", {"t": Row}, {})

    class Empty:
        pass

    with pytest.raises(ValueError, match="missing attribute 'a'"):
        fn.infer({"t": [Empty()]})


def test_unknown_table_in_from_is_rejected():
    fn = CodegenFn("SELECT a AS x FROM t", {"t": Row}, {})
    with pytest.raises(ValueError, match="Unknown table"):
        fn.infer({"other": [Row(a=1)]})


def test_supplied_output_model_is_validated():
    class Good(BaseModel):
        x: float  # int is compatible with a declared float

    class MissingField(BaseModel):
        nope: int

    class Extra(BaseModel):
        x: int
        surplus: int

    CodegenFn("SELECT a AS x FROM t", {"t": Row}, {}, output_model=Good)
    with pytest.raises(ValueError, match="missing field 'x'"):
        CodegenFn("SELECT a AS x FROM t", {"t": Row}, {}, output_model=MissingField)
    with pytest.raises(ValueError, match="not produced by the query"):
        CodegenFn("SELECT a AS x FROM t", {"t": Row}, {}, output_model=Extra)


def test_incompatible_output_model_is_rejected():
    class Bad(BaseModel):
        x: str  # int is not compatible with a declared str

    with pytest.raises(ValueError, match="incompatible"):
        CodegenFn("SELECT a AS x FROM t", {"t": Row}, {}, output_model=Bad)


def test_int_division_by_zero_raises_value_error():
    fn = CodegenFn("SELECT a / 0 AS x FROM t", {"t": Row}, {})
    with pytest.raises(ValueError, match="division by zero"):
        fn.infer({"t": [Row(a=1)]})


def test_container_columns_struct_passthrough():
    class Inner(BaseModel):
        x: int

    class WithStruct(BaseModel):
        s: Inner

    # Struct columns now pass through (no longer deferred)
    fn = CodegenFn("SELECT s AS out FROM t", {"t": WithStruct}, {})
    result = fn.infer(t=[WithStruct(s=Inner(x=42))])
    assert len(result) == 1
    assert result[0].out.x == 42


class TwoLists(BaseModel):
    a: list[int]
    b: list[int]


def test_multi_unnest_rejected():
    # Two unnest(list) calls is a cross-product cardinality change we don't
    # support (mirrors native). The differential harness can't cover this --
    # its multi-unnest test builds a native InferFn directly on both backends.
    with pytest.raises(ValueError, match="Only one unnest"):
        CodegenFn("SELECT unnest(a) AS x, unnest(b) AS y FROM t", {"t": TwoLists}, {})


def test_unnest_of_a_scalar_is_rejected():
    with pytest.raises(ValueError, match="struct or list"):
        CodegenFn("SELECT unnest(a) AS x FROM t", {"t": Row}, {})


def test_generated_source_is_available_for_debugging():
    fn = CodegenFn("SELECT a AS x FROM t", {"t": Row}, {})
    assert "def _run(" in fn.source


class Key(BaseModel):
    k: int
    a: int = 1


def _lookup_table():
    return pa.table(
        {"k": pa.array([1, 2], type=pa.int64()), "v": pa.array([10.0, 20.0])}
    )


def test_cross_join_is_a_cartesian_product():
    class L(BaseModel):
        a: int

    class R(BaseModel):
        b: int

    fn = CodegenFn("SELECT a AS x, b AS y FROM l CROSS JOIN r", {"l": L, "r": R}, {})
    out = fn.infer({"l": [L(a=1), L(a=2)], "r": [R(b=9)]})
    assert [(r.x, r.y) for r in out] == [(1, 9), (2, 9)]


def test_inner_join_matches_on_keys():
    class L(BaseModel):
        k: int
        a: int

    class R(BaseModel):
        k: int
        b: int

    fn = CodegenFn(
        "SELECT a AS x, b AS y FROM l JOIN r ON l.k = r.k", {"l": L, "r": R}, {}
    )
    out = fn.infer({"l": [L(k=1, a=1), L(k=2, a=2)], "r": [R(k=2, b=9)]})
    assert [(r.x, r.y) for r in out] == [(2, 9)]


def test_inner_join_never_matches_null_keys():
    class L(BaseModel):
        k: int | None
        a: int

    class R(BaseModel):
        k: int | None
        b: int

    fn = CodegenFn(
        "SELECT a AS x, b AS y FROM l JOIN r ON l.k = r.k", {"l": L, "r": R}, {}
    )
    assert fn.infer({"l": [L(k=None, a=1)], "r": [R(k=None, b=9)]}) == []


def test_lookup_join_binds_the_matching_static_row():
    fn = CodegenFn(
        "SELECT v AS x FROM t JOIN s ON t.k = s.k", {"t": Key}, {"s": _lookup_table()}
    )
    assert [r.x for r in fn.infer({"t": [Key(k=2)]})] == [20.0]


def test_inner_lookup_join_miss_raises_key_error():
    fn = CodegenFn(
        "SELECT v AS x FROM t JOIN s ON t.k = s.k", {"t": Key}, {"s": _lookup_table()}
    )
    with pytest.raises(KeyError, match="No row in static table 's'"):
        fn.infer({"t": [Key(k=99)]})


def test_left_lookup_join_miss_yields_nulls_and_a_nullable_output():
    fn = CodegenFn(
        "SELECT v AS x FROM t LEFT JOIN s ON t.k = s.k",
        {"t": Key},
        {"s": _lookup_table()},
    )
    assert fn.output_model.model_fields["x"].annotation == typing.Optional[float]  # noqa: UP045
    assert [r.x for r in fn.infer({"t": [Key(k=99)]})] == [None]


def test_lookup_join_keys_are_type_strict():
    # Value::Int(1) and Value::Float(1.0) hash differently, so a float key must
    # not match an int key row -- Python's 1 == 1.0 would wrongly match.
    class FloatKey(BaseModel):
        k: float

    fn = CodegenFn(
        "SELECT v AS x FROM t LEFT JOIN s ON t.k = s.k",
        {"t": FloatKey},
        {"s": _lookup_table()},
    )
    assert [r.x for r in fn.infer({"t": [FloatKey(k=1.0)]})] == [None]
