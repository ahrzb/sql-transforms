import pyarrow as pa
import pytest
from differential import check, check_both_raise, rows
from pydantic import BaseModel

from sql_transform import SQLTransform
from sql_transform._interpreter import InferFn


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


# The three construction shapes below (positional struct, named struct(...),
# make_array) are supported on codegen and match the DataFusion oracle, but
# native has no dispatch for them at all -- so they xfail_on_native, the same
# parity-gap pattern as test_list_construct_mixed_numeric_widens. Codegen
# correctness is the assertion; the native gaps want their own tickets.
def test_struct_construct_positional(xfail_on_native):
    # struct(a, b) names fields positionally c0, c1 (matches DataFusion).
    xfail_on_native("native has no struct(...) construction dispatch (expr_build.rs)")
    check(
        "SELECT struct(a, b) AS s FROM t",
        {"t": rows({"a": "int", "b": "int"}, [{"a": 1, "b": 2}])},
        expect=[{"s": {"c0": 1, "c1": 2}}],
    )


def test_struct_construct_named(xfail_on_native):
    # struct(a AS x, b AS y) parses as exp.PropertyEQ -> explicit field names.
    xfail_on_native("native has no struct(...) construction dispatch (expr_build.rs)")
    check(
        "SELECT struct(a AS x, b AS y) AS s FROM t",
        {"t": rows({"a": "int", "b": "int"}, [{"a": 1, "b": 2}])},
        expect=[{"s": {"x": 1, "y": 2}}],
    )


def test_make_array_construct(xfail_on_native):
    # make_array(a, b) is a DataFusion builtin; native doesn't recognize it as a
    # function (only the bracket literal [a, b] reaches native's list path).
    xfail_on_native("native has no make_array function dispatch (expr_build.rs)")
    check(
        "SELECT make_array(a, b) AS l FROM t",
        {"t": rows({"a": "int", "b": "int"}, [{"a": 1, "b": 2}])},
        expect=[{"l": [1, 2]}],
    )


def test_list_construct_mixed_numeric_widens(xfail_on_native):
    # DataFusion widens the int element to match the float element (list<double>).
    # Locks in infer_type's ListExpr arm unifying element bases via _common_base
    # (like COALESCE), so the output model types the list as list[float] and
    # pydantic coerces the runtime int element to float.
    #
    # Bracket literal `[x, y]` (not make_array(...)): native's function dispatch
    # doesn't recognize "make_array"/"array" as a builtin at all (a separate,
    # pre-existing gap -- expr_build.rs's convert_function has no such case,
    # unlike Python's ANONYMOUS-function branch), so it isn't a usable surface
    # to compare both engines against the oracle. The bracket form reaches the
    # same Expr::List / ListExpr construction path in both engines.
    #
    # xfail_on_native: measured -- native's unify_list_element_types
    # (src/types.rs) is exact-equality-only, the same bug class just fixed here
    # in codegen's infer_type. Native still emits [1, 2.5] (un-widened) instead
    # of DataFusion's [1.0, 2.5]. A native-side fix is out of scope here; flag
    # as a parity bug for its own ticket rather than fixing inline.
    xfail_on_native(
        "native does not widen mixed int/float list elements "
        "(types.rs unify_list_element_types is exact-equality-only) -- "
        "separate parity bug, own ticket"
    )
    check(
        "SELECT [x, y] AS l FROM t",
        {"t": rows({"x": "int", "y": "float"}, [{"x": 1, "y": 2.5}])},
        expect=[{"l": [1.0, 2.5]}],
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


def test_malformed_struct_input_raises():
    # A scalar where a struct is declared must error, not silently marshal
    # into an all-null struct. Direct infer() test: the differential check()
    # harness validates rows via model(**r) before they reach the Rust side,
    # so it can't exercise this path.
    schema = pa.schema(
        [pa.field("s", pa.struct([("x", pa.int64()), ("y", pa.int64())]))]
    )
    table = pa.Table.from_pylist([{"s": {"x": 1, "y": 2}}], schema=schema)
    t = SQLTransform("SELECT s FROM __THIS__").fit(table)
    with pytest.raises(ValueError):
        t.infer({"s": 5})


def test_deep_nesting_roundtrip():
    check(
        "SELECT s FROM t",
        {
            "t": rows(
                {"s": "struct{a:int,inner:list[int]}"},
                [{"s": {"a": 1, "inner": [1, 2, 3]}}],
            )
        },
    )


def test_struct_field_access():
    check(
        "SELECT s.x AS fx FROM t",
        {"t": rows({"s": "struct{x:int,y:int}"}, [{"s": {"x": 5, "y": 9}}])},
    )


def test_nested_struct_field_access():
    check(
        "SELECT s.a.b AS v FROM t",
        {"t": rows({"s": "struct{a:struct{b:int}}"}, [{"s": {"a": {"b": 7}}}])},
    )


def test_qualified_struct_field_access():
    check(
        "SELECT t.s.x AS v FROM t",
        {"t": rows({"s": "struct{x:int,y:int}"}, [{"s": {"x": 5, "y": 9}}])},
    )


def test_uppercase_qualifier_field_access(xfail_on_native):
    # An unquoted struct-column qualifier folds like any column: `S.x` -> `s.x`,
    # matching DataFusion. Native doesn't fold the qualifier (raises "Unknown
    # column: S") -- parity gap, wants its own ticket.
    xfail_on_native("native does not fold an unquoted struct-column qualifier (S.x)")
    check(
        "SELECT S.x AS v FROM t",
        {"t": rows({"s": "struct{x:int}"}, [{"s": {"x": 7}}])},
    )


def test_struct_equality_true():
    check(
        "SELECT (s = s) AS eq FROM t",
        {"t": rows({"s": "struct{x:int,y:int}"}, [{"s": {"x": 1, "y": 2}}])},
    )


def test_struct_equality_false():
    # Deep, type-tagged: differs in one field -> not equal.
    check(
        "SELECT (s = named_struct('x', 1, 'y', 99)) AS eq FROM t",
        {"t": rows({"s": "struct{x:int,y:int}"}, [{"s": {"x": 1, "y": 2}}])},
    )


def test_list_equality():
    check(
        "SELECT (l = l) AS eq FROM t",
        {"t": rows({"l": "list[int]"}, [{"l": [1, 2, 3]}])},
    )


def test_list_inequality():
    check(
        "SELECT (l = [9, 9, 9]) AS eq FROM t",
        {"t": rows({"l": "list[int]"}, [{"l": [1, 2, 3]}])},
    )


def test_unnest_struct_expands_columns():
    check(
        "SELECT unnest(named_struct('x', a, 'y', b)) FROM t",
        {
            "t": __import__("differential").rows(
                {"a": "int", "b": "int"}, [{"a": 1, "b": 2}]
            )
        },
    )


def test_unnest_struct_column_expands_columns():
    check(
        "SELECT unnest(s) FROM t",
        {"t": rows({"s": "struct{x:int,y:int}"}, [{"s": {"x": 5, "y": 9}}])},
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "CONFIRMED bug on BOTH engines (measured 2026-07-24): DataFusion names "
        "unnest() of a struct-typed FIELD ACCESS using bracket notation for the "
        "intermediate field, e.g. 't.s[inner].x' -- not the dot-chained "
        "'t.s.inner.x' that both native's unnest_display_name (src/plan.rs) and "
        "codegen's _unnest_display_name (sql_transform/_codegen/plan.py) "
        "produce. codegen's FieldAccess branch was written to mirror native's, "
        "so it faithfully reproduces the same wrong answer rather than "
        "matching the oracle. Needs a ticket; not fixed here (tests/-only "
        "scope for this task, and native bugs are never fixed inline)."
    ),
)
def test_unnest_struct_field_access_expands_columns():
    # unnest() of a struct-typed FIELD of a struct column -- exercises
    # _unnest_display_name's recursive FieldAccess branch, not just the bare
    # Column/StructExpr branches covered above.
    check(
        "SELECT unnest(s.inner) FROM t",
        {
            "t": rows(
                {"s": "struct{inner:struct{x:int,y:int}}"},
                [{"s": {"inner": {"x": 5, "y": 9}}}],
            )
        },
    )


def test_unnest_struct_alias_is_discarded():
    # DataFusion names unnest(struct)'s expanded columns from the argument's
    # display, ignoring any SELECT-list AS alias -- pin that behavior.
    check(
        "SELECT unnest(s) AS renamed FROM t",
        {"t": rows({"s": "struct{x:int,y:int}"}, [{"s": {"x": 5, "y": 9}}])},
        expect=[{"t.s.x": 5, "t.s.y": 9}],
    )


def test_unnest_list_expands_rows():
    check(
        "SELECT id, unnest(vals) AS v FROM t",
        {
            "t": rows(
                {"id": "int", "vals": "list[int]?"},
                [
                    {"id": 1, "vals": [10, 20, 30]},
                    {"id": 2, "vals": []},
                    {"id": 3, "vals": None},
                ],
            )
        },
        expect=[{"id": 1, "v": 10}, {"id": 1, "v": 20}, {"id": 1, "v": 30}],
    )
    # empty list (id=2) and NULL list (id=3) both -> zero rows


def test_unnest_list_all_dropped_yields_no_rows():
    # Every input row's list is empty or NULL -> zero output rows.
    check(
        "SELECT id, unnest(vals) AS v FROM t",
        {
            "t": rows(
                {"id": "int", "vals": "list[int]?"},
                [{"id": 1, "vals": []}, {"id": 2, "vals": None}],
            )
        },
        expect=[],
    )


def test_struct_equality():
    # s1 = s2 / s1 != s2 must use deep structural equality (DataFusion parity),
    # not fall into the scalar-only arithmetic comparison path.
    check(
        "SELECT (s1 = s2) AS eq FROM t",
        {
            "t": rows(
                {"s1": "struct{x:int,y:int}", "s2": "struct{x:int,y:int}"},
                [
                    {"s1": {"x": 1, "y": 2}, "s2": {"x": 1, "y": 2}},
                    {"s1": {"x": 1, "y": 2}, "s2": {"x": 1, "y": 3}},
                ],
            )
        },
        expect=[{"eq": True}, {"eq": False}],
    )


def test_struct_as_join_key():
    # A struct column used as a row x row JOIN key. Works today via `Value`'s
    # structural PartialEq in execute_rel's Join arm; locked in here.
    check(
        "SELECT a.v, b.w FROM a JOIN b ON a.s = b.s",
        {
            "a": rows(
                {"s": "struct{x:int,y:int}", "v": "int"},
                [
                    {"s": {"x": 1, "y": 2}, "v": 10},
                    {"s": {"x": 3, "y": 4}, "v": 20},
                ],
            ),
            "b": rows(
                {"s": "struct{x:int,y:int}", "w": "int"},
                [{"s": {"x": 1, "y": 2}, "w": 100}],
            ),
        },
        expect=[{"v": 10, "w": 100}],
    )


class _TwoLists(BaseModel):
    a: list[int]
    b: list[int]


def test_multi_unnest_rejected():
    # Two unnest(list) calls in one query is a cross-product cardinality
    # change we don't support (Task 6). DataFusion accepts it (cross product),
    # so this diverges by design and can't go through check() -- direct
    # InferFn construction instead.
    sql = "SELECT unnest(a) AS ea, unnest(b) AS eb FROM t"
    with pytest.raises(ValueError):
        InferFn(sql, row_tables={"t": _TwoLists}, static_tables={})


def test_unnest_list_preserves_other_columns():
    # Multiple non-unnest columns ride along on each emitted row.
    check(
        "SELECT id, label, unnest(vals) AS v FROM t",
        {
            "t": rows(
                {"id": "int", "label": "str", "vals": "list[int]?"},
                [{"id": 1, "label": "a", "vals": [7, 8]}],
            )
        },
        expect=[
            {"id": 1, "label": "a", "v": 7},
            {"id": 1, "label": "a", "v": 8},
        ],
    )


def test_unnest_extra_argument_silently_dropped(xfail_on_codegen):
    # CONFIRMED bug on codegen only (measured 2026-07-24): sqlglot's default
    # parse dialect accepts only one unnest() argument at the grammar level --
    # `unnest(a, b)` parses to Unnest(expressions=[Column(a)]) with `b` gone
    # from the tree entirely (no trace in .args, comments, or meta), so
    # codegen has no way to detect the dropped argument and rejects nothing.
    # Both the oracle ("unnest() requires exactly one argument") and native
    # ("Unknown function: unnest") reject the query outright; codegen instead
    # silently drops `b` and answers using only `a`. See the comment on the
    # exp.Unnest arm of _convert_expr (sql_transform/_codegen/plan.py) for why
    # no guard is possible here.
    xfail_on_codegen(
        "codegen cannot detect a dropped unnest() argument -- sqlglot's "
        "default dialect discards it at parse time before codegen ever sees "
        "it (measured: Unnest(expressions=[Column(a)]), no trace of `b` "
        "anywhere in the tree); oracle and native both reject unnest(a, b), "
        "codegen silently answers using only `a`"
    )
    check_both_raise(
        "SELECT unnest(a, b) AS x FROM t",
        {"t": rows({"a": "list[int]", "b": "int"}, [{"a": [1, 2], "b": 10}])},
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "CONFIRMED bug on BOTH engines (measured 2026-07-24): a bare "
        "unnest(<list>) with no AS names its output column 'unnest' (the "
        "placeholder in codegen's _build_projection exp.Unnest arm and "
        "native's equivalent column_name in src/plan.rs), but the oracle "
        "names it from the DataFusion logical-plan display of the argument "
        "-- 'UNNEST(t.l)' here. codegen faithfully mirrors native's naming "
        "bug (correct per project rules), but nothing pinned the divergence "
        "against the oracle until now. Needs a ticket; not fixed here "
        "(tests/-only scope for this task, and native bugs are never fixed "
        "inline)."
    ),
)
def test_unnest_bare_list_column_name_diverges():
    check(
        "SELECT unnest(l) FROM t",
        {"t": rows({"id": "int", "l": "list[int]"}, [{"id": 9, "l": [1, 2]}])},
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "CONFIRMED bug on BOTH engines (measured 2026-07-24): the same "
        "'unnest' placeholder name (see test_unnest_bare_list_column_name_"
        "diverges) collides with a real `AS unnest` alias elsewhere in the "
        "SELECT list. The oracle keeps both columns distinct ('unnest' from "
        "the alias, 'UNNEST(t.l)' from the bare unnest); both engines key "
        "their output row by that placeholder name, so the second entry "
        "silently overwrites the first and the `id AS unnest` column "
        "vanishes from the output entirely. Needs a ticket; not fixed here "
        "(tests/-only scope for this task, and native bugs are never fixed "
        "inline)."
    ),
)
def test_unnest_bare_list_alias_collision_drops_column():
    check(
        "SELECT id AS unnest, unnest(l) FROM t",
        {"t": rows({"id": "int", "l": "list[int]"}, [{"id": 9, "l": [1, 2]}])},
    )
