"""End-to-end tests for the codegen engine.

Broad semantic parity is proven by the differential harness (tests/); these
cover the engine's own seams -- compilation, marshalling, output typing.
"""

import typing

import pytest
from pydantic import BaseModel

from sql_transform._codegen import CodegenFn, UnsupportedInCodegen


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


def test_container_columns_are_deferred_not_silently_wrong():
    class Inner(BaseModel):
        x: int

    class WithStruct(BaseModel):
        s: Inner

    with pytest.raises(UnsupportedInCodegen):
        CodegenFn("SELECT s AS out FROM t", {"t": WithStruct}, {})


def test_generated_source_is_available_for_debugging():
    fn = CodegenFn("SELECT a AS x FROM t", {"t": Row}, {})
    assert "def _run(" in fn.source
