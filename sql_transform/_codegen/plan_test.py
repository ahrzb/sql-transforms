"""Unit tests for the codegen front-end (types, parse, optimize, validate)."""

import typing

import pyarrow as pa
import pytest
from pydantic import BaseModel

from sql_transform._codegen import plan as cp


def test_schema_from_pydantic_reads_types_and_nullability():
    class Row(BaseModel):
        a: int
        b: float | None
        c: str
        d: bool

    schema = cp.schema_from_pydantic(Row)
    assert schema["a"] == cp.FieldType(cp.INT, False)
    assert schema["b"] == cp.FieldType(cp.FLOAT, True)
    assert schema["c"] == cp.FieldType(cp.STR, False)
    assert schema["d"] == cp.FieldType(cp.BOOL, False)


def test_schema_from_pydantic_optional_and_any():
    class Row(BaseModel):
        a: int | None
        b: typing.Any

    schema = cp.schema_from_pydantic(Row)
    assert schema["a"] == cp.FieldType(cp.INT, True)
    assert schema["b"].base == cp.OTHER


def test_schema_from_pydantic_nested_model_is_a_struct():
    class Inner(BaseModel):
        x: int

    class Row(BaseModel):
        s: Inner

    schema = cp.schema_from_pydantic(Row)
    assert schema["s"].base == cp.StructBase((("x", cp.FieldType(cp.INT, False)),))
    assert cp.is_container(schema["s"].base)


def test_schema_from_arrow_reads_types_and_nullability():
    table = pa.table(
        {"a": pa.array([1], type=pa.int64()), "b": pa.array([1.0], type=pa.float64())},
        schema=pa.schema(
            [pa.field("a", pa.int64(), nullable=False), pa.field("b", pa.float64())]
        ),
    )
    schema = cp.schema_from_arrow(table)
    assert schema["a"] == cp.FieldType(cp.INT, False)
    assert schema["b"] == cp.FieldType(cp.FLOAT, True)


def test_field_type_to_python_round_trips():
    assert cp.field_type_to_python(cp.FieldType(cp.INT, False)) is int
    assert cp.field_type_to_python(cp.FieldType(cp.INT, True)) == (int | None)
    assert cp.field_type_to_python(cp.FieldType(cp.OTHER, False)) is typing.Any


def test_compatible_allows_int_into_float_and_unknown_into_anything():
    assert cp.compatible(cp.INT, cp.FLOAT)
    assert cp.compatible(cp.OTHER, cp.STR)
    assert cp.compatible(cp.INT, cp.INT)
    assert not cp.compatible(cp.STR, cp.INT)
    assert not cp.compatible(cp.FLOAT, cp.INT)
    assert cp.is_container(cp.INT) is False


def test_compatible_struct_and_list():
    x_int = ("x", cp.FieldType(cp.INT, False))
    y_str = ("y", cp.FieldType(cp.STR, False))

    # reordered same-name-same-type structs are compatible (order-independent).
    inferred = cp.StructBase((x_int, y_str))
    declared_reordered = cp.StructBase((y_str, x_int))
    assert cp.compatible(inferred, declared_reordered)
    assert cp.is_container(inferred)

    # widening a field INT->FLOAT inside a struct is compatible.
    declared_widened = cp.StructBase((("x", cp.FieldType(cp.FLOAT, False)), y_str))
    assert cp.compatible(inferred, declared_widened)

    # different field names -> not compatible.
    declared_diff_name = cp.StructBase(
        (("x", cp.FieldType(cp.INT, False)), ("z", cp.FieldType(cp.STR, False)))
    )
    assert not cp.compatible(inferred, declared_diff_name)

    # different field count -> not compatible.
    declared_diff_count = cp.StructBase((x_int,))
    assert not cp.compatible(inferred, declared_diff_count)

    # incompatible field type -> not compatible.
    declared_diff_type = cp.StructBase((("x", cp.FieldType(cp.STR, False)), y_str))
    assert not cp.compatible(inferred, declared_diff_type)

    # list element base compatibility mirrors scalar rules.
    list_int = cp.ListBase(cp.FieldType(cp.INT, False))
    list_float = cp.ListBase(cp.FieldType(cp.FLOAT, False))
    list_str = cp.ListBase(cp.FieldType(cp.STR, False))
    assert cp.compatible(list_int, list_float)
    assert not cp.compatible(list_str, list_int)
    assert cp.is_container(list_int)

    # struct vs list -> not compatible.
    assert not cp.compatible(inferred, list_int)


def test_build_plan_simple_projection():
    plan = cp.build_plan("SELECT a AS x FROM t")
    assert plan.input == cp.TableScan("t")
    assert plan.projection == [("x", cp.Column(None, "a"))]


def test_build_plan_unaliased_column_keeps_its_name():
    plan = cp.build_plan("SELECT a FROM t")
    assert plan.projection == [("a", cp.Column(None, "a"))]


def test_build_plan_qualified_column():
    plan = cp.build_plan("SELECT t.a AS x FROM t")
    assert plan.projection == [("x", cp.Column("t", "a"))]


def test_build_plan_requires_an_alias_for_expressions():
    with pytest.raises(ValueError, match="alias"):
        cp.build_plan("SELECT a + 1 FROM t")


def test_build_plan_where_becomes_a_filter():
    plan = cp.build_plan("SELECT a AS x FROM t WHERE a > 1")
    assert isinstance(plan.input, cp.Filter)
    assert plan.input.predicate == cp.BinaryOp(
        "gt", cp.Column(None, "a"), cp.Literal(1)
    )


def test_build_plan_binary_ops_and_literals():
    plan = cp.build_plan(
        "SELECT a + 1 AS x, b / 2.0 AS y, c AS z FROM t WHERE c = 'hi'"
    )
    assert plan.projection[0][1] == cp.BinaryOp(
        "add", cp.Column(None, "a"), cp.Literal(1)
    )
    assert plan.projection[1][1] == cp.BinaryOp(
        "div", cp.Column(None, "b"), cp.Literal(2.0)
    )
    assert plan.input.predicate == cp.BinaryOp(
        "eq", cp.Column(None, "c"), cp.Literal("hi")
    )


def test_build_plan_null_and_boolean_literals():
    plan = cp.build_plan("SELECT NULL AS x, TRUE AS y FROM t")
    assert plan.projection == [("x", cp.Literal(None)), ("y", cp.Literal(True))]


def test_build_plan_alias():
    plan = cp.build_plan("SELECT s.a AS x FROM t AS s")
    assert plan.input == cp.SubqueryAlias(cp.TableScan("t"), "s")


def test_build_plan_rejects_duplicate_relations():
    with pytest.raises(ValueError, match="more than once"):
        cp.build_plan("SELECT a AS x FROM t JOIN t ON t.a = t.a")


def test_build_plan_cross_join():
    plan = cp.build_plan("SELECT a AS x FROM t CROSS JOIN u")
    assert plan.input == cp.CrossJoin(cp.TableScan("t"), cp.TableScan("u"))


def test_build_plan_comma_join_is_a_cross_join():
    plan = cp.build_plan("SELECT a AS x FROM t, u")
    assert plan.input == cp.CrossJoin(cp.TableScan("t"), cp.TableScan("u"))


def test_build_plan_inner_join_extracts_equality_keys():
    plan = cp.build_plan("SELECT a AS x FROM t JOIN u ON t.k = u.k")
    assert plan.input == cp.Join(
        cp.TableScan("t"),
        cp.TableScan("u"),
        [(cp.Column("t", "k"), cp.Column("u", "k"))],
        False,
    )


def test_build_plan_left_join_is_outer():
    plan = cp.build_plan("SELECT a AS x FROM t LEFT JOIN u ON t.k = u.k")
    assert plan.input.outer is True


def test_build_plan_join_on_and_of_equalities():
    plan = cp.build_plan("SELECT a AS x FROM t JOIN u ON t.k = u.k AND t.j = u.j")
    assert len(plan.input.on) == 2


def test_build_plan_rejects_non_equality_join_on():
    with pytest.raises(ValueError, match="equalit"):
        cp.build_plan("SELECT a AS x FROM t JOIN u ON t.k > u.k")


def test_build_plan_functions_and_cast():
    plan = cp.build_plan(
        "SELECT UPPER(a) AS u, SUBSTR(a, 2, 3) AS s, COALESCE(a, b) AS c, "
        "CAST(a AS VARCHAR) AS v FROM t"
    )
    assert plan.projection[0][1] == cp.Func("upper", [cp.Column(None, "a")])
    assert plan.projection[1][1] == cp.Func(
        "substr", [cp.Column(None, "a"), cp.Literal(2), cp.Literal(3)]
    )
    assert plan.projection[2][1] == cp.Func(
        "coalesce", [cp.Column(None, "a"), cp.Column(None, "b")]
    )
    assert plan.projection[3][1] == cp.Cast(cp.Column(None, "a"), cp.STR)


def test_build_plan_cast_targets():
    plan = cp.build_plan(
        "SELECT CAST(a AS BIGINT) AS i, CAST(a AS DOUBLE) AS f, "
        "CAST(a AS BOOLEAN) AS b FROM t"
    )
    assert [e.target for _, e in plan.projection] == [cp.INT, cp.FLOAT, cp.BOOL]


def test_build_plan_not_and_logic():
    plan = cp.build_plan("SELECT a AS x FROM t WHERE NOT (a AND b)")
    assert plan.input.predicate == cp.Not(
        cp.BinaryOp("and", cp.Column(None, "a"), cp.Column(None, "b"))
    )


def test_build_plan_defers_containers():
    with pytest.raises(cp.UnsupportedInCodegen):
        cp.build_plan("SELECT unnest(a) AS x FROM t")
    with pytest.raises(cp.UnsupportedInCodegen):
        cp.build_plan("SELECT named_struct('k', a) AS x FROM t")


def _schemas():
    return (
        {"t": {"a": cp.FieldType(cp.INT, False), "k": cp.FieldType(cp.INT, False)}},
        {"s": {"k": cp.FieldType(cp.INT, False), "v": cp.FieldType(cp.FLOAT, False)}},
    )


def test_optimize_rewrites_a_static_join_into_a_lookup_join():
    plan = cp.build_plan("SELECT v AS x FROM t JOIN s ON t.k = s.k")
    plan, specs = cp.optimize(plan, {"s"})
    assert isinstance(plan.input, cp.LookupJoin)
    assert plan.input.table == "s"
    assert plan.input.keys == [cp.Column("t", "k")]
    assert specs[0].static_table == "s"
    assert specs[0].key_columns == ["k"]


def test_optimize_finds_the_static_side_regardless_of_on_order():
    plan = cp.build_plan("SELECT v AS x FROM t JOIN s ON s.k = t.k")
    plan, _ = cp.optimize(plan, {"s"})
    assert plan.input.keys == [cp.Column("t", "k")]


def test_optimize_rejects_a_static_to_static_join():
    plan = cp.build_plan("SELECT v AS x FROM s JOIN s2 ON s.k = s2.k")
    with pytest.raises(ValueError, match="two static tables"):
        cp.optimize(plan, {"s", "s2"})


def test_optimize_rejects_a_row_to_row_left_join():
    plan = cp.build_plan("SELECT a AS x FROM t LEFT JOIN u ON t.k = u.k")
    with pytest.raises(ValueError, match="only supported against a static"):
        cp.optimize(plan, set())


def test_validate_resolves_unqualified_columns_and_collects_used():
    row, static = _schemas()
    plan = cp.build_plan("SELECT a AS x FROM t")
    v = cp.validate_columns(plan, {"t"}, row, static)
    assert plan.projection[0][1] == cp.Column("t", "a")  # rewritten in place
    assert v.row_table_columns == {"t": ["a"]}


def test_validate_rejects_unknown_and_ambiguous_columns():
    row, static = _schemas()
    plan = cp.build_plan("SELECT nope AS x FROM t")
    with pytest.raises(ValueError, match="Unknown column"):
        cp.validate_columns(plan, {"t"}, row, static)

    row2 = {
        "t": {"a": cp.FieldType(cp.INT, False)},
        "u": {"a": cp.FieldType(cp.INT, False)},
    }
    plan = cp.build_plan("SELECT a AS x FROM t CROSS JOIN u")
    with pytest.raises(ValueError, match="Ambiguous"):
        cp.validate_columns(plan, {"t", "u"}, row2, {})


def test_validate_widens_the_outer_side_of_a_left_lookup_join_to_nullable():
    row, static = _schemas()
    plan = cp.build_plan("SELECT v AS x FROM t LEFT JOIN s ON t.k = s.k")
    plan, _ = cp.optimize(plan, {"s"})
    v = cp.validate_columns(plan, {"t"}, row, static)
    assert v.effective_schemas["s"]["v"].nullable is True


def test_validate_resolves_through_an_alias():
    row, static = _schemas()
    plan = cp.build_plan("SELECT z.a AS x FROM t AS z")
    v = cp.validate_columns(plan, {"t"}, row, static)
    assert v.effective_schemas["z"]["a"] == cp.FieldType(cp.INT, False)
    assert v.row_table_columns == {"t": ["a"]}


def test_infer_type_arithmetic_and_nullability():
    schemas = {
        "t": {
            "i": cp.FieldType(cp.INT, False),
            "f": cp.FieldType(cp.FLOAT, False),
            "n": cp.FieldType(cp.INT, True),
        }
    }
    add_i_lit = cp.BinaryOp("add", cp.Column("t", "i"), cp.Literal(1))
    assert cp.infer_type(add_i_lit, schemas) == cp.FieldType(cp.INT, False)

    add_i_f = cp.BinaryOp("add", cp.Column("t", "i"), cp.Column("t", "f"))
    assert cp.infer_type(add_i_f, schemas) == cp.FieldType(cp.FLOAT, False)

    add_i_n = cp.BinaryOp("add", cp.Column("t", "i"), cp.Column("t", "n"))
    assert cp.infer_type(add_i_n, schemas) == cp.FieldType(cp.INT, True)

    gt_i_lit = cp.BinaryOp("gt", cp.Column("t", "i"), cp.Literal(1))
    assert cp.infer_type(gt_i_lit, schemas) == cp.FieldType(cp.BOOL, False)


def test_infer_type_functions_and_casts():
    schemas = {
        "t": {"s": cp.FieldType(cp.STR, False), "i": cp.FieldType(cp.INT, True)}
    }
    upper_s = cp.Func("upper", [cp.Column("t", "s")])
    assert cp.infer_type(upper_s, schemas).base == cp.STR

    concat_i = cp.Func("concat", [cp.Column("t", "i")])
    assert cp.infer_type(concat_i, schemas) == cp.FieldType(cp.STR, False)  # never null

    abs_i = cp.Func("abs", [cp.Column("t", "i")])
    assert cp.infer_type(abs_i, schemas) == cp.FieldType(cp.INT, True)  # keeps base

    # ROUND always types as float, even on an int arg (DataFusion; Rust bug types
    # it as the arg base, which pydantic then coerces the float result back to).
    round_i = cp.Func("round", [cp.Column("t", "i")])
    assert cp.infer_type(round_i, schemas) == cp.FieldType(cp.FLOAT, True)

    coalesce_i = cp.Func("coalesce", [cp.Column("t", "i")])
    assert cp.infer_type(coalesce_i, schemas).nullable is True
    assert cp.infer_type(cp.Cast(cp.Column("t", "i"), cp.STR), schemas) == (
        cp.FieldType(cp.STR, True)
    )
