"""DuckDBInferFn vs duckdb-python: the specializer's differential oracle.

`duck_check` runs the same SQL on the same data through both engines and
asserts the outputs agree row-for-row. Stretch-3 surface: the projection /
WHERE spine plus equi-joins to static tables (INNER and LEFT), which lower
to map probes.
"""

from __future__ import annotations

from typing import Any

import duckdb
import pyarrow as pa
import pytest
from pydantic import create_model

from sql_transform._interpreter import DuckDBInferFn

_PY = {"int": int, "float": float, "str": str, "bool": bool}
_ARROW = {
    "int": pa.int64(),
    "float": pa.float64(),
    "str": pa.string(),
    "bool": pa.bool_(),
}


def _row_model(schema: dict[str, str]):
    fields: dict[str, Any] = {}
    for name, spec in schema.items():
        if spec.endswith("?"):
            fields[name] = (_PY[spec[:-1]] | None, None)
        else:
            fields[name] = (_PY[spec], ...)
    return create_model("Row", **fields)


def static(schema: dict[str, str], rows: list[dict[str, Any]]) -> pa.Table:
    arrow = pa.schema(
        pa.field(n, _ARROW[s.rstrip("?")], nullable=s.endswith("?"))
        for n, s in schema.items()
    )
    return pa.Table.from_pylist(rows, schema=arrow)


def duck_check(
    sql: str,
    row_schema: dict[str, str],
    row_rows: list[dict[str, Any]],
    statics: dict[str, pa.Table] | None = None,
) -> None:
    statics = statics or {}
    model = _row_model(row_schema)
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": model}, static_tables=statics)
    inputs = [model(**r) for r in row_rows]
    got = [r.model_dump() for r in fn.infer({"__THIS__": inputs})]

    con = duckdb.connect()
    # Materialize NATIVE tables: duckdb pushes constant filters into
    # registered-arrow scans with IEEE NaN semantics, which disagrees with
    # its own native-table comparison order (adversarial probe, 2026-07-26).
    # The engine follows native-table semantics — the corpus's world.
    for name, table in statics.items():
        con.register(f"__arrow_{name}", table)
        con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM "__arrow_{name}"')
    con.register("__arrow_this", static(row_schema, row_rows))
    con.execute("CREATE TABLE __THIS__ AS SELECT * FROM __arrow_this")
    want = con.execute(sql).to_arrow_table().to_pylist()

    # Row order is not part of the contract (a join may reorder); compare as
    # multisets of repr'd rows — repr keeps value types apart (1 vs '1' vs
    # 1.0) and makes NaN compare equal to itself.
    key = lambda r: sorted((k, repr(v)) for k, v in r.items())  # noqa: E731
    assert sorted(map(key, got)) == sorted(map(key, want)), f"{got} != {want}"


DIM = static(
    {"id": "int", "name": "str"},
    [{"id": 1, "name": "one"}, {"id": 3, "name": "three"}],
)


def test_projection_and_where_differential():
    duck_check(
        "SELECT a + 1 AS x, a / 2 AS h FROM __THIS__ WHERE a > 0",
        {"a": "int"},
        [{"a": 4}, {"a": -1}, {"a": 7}],
    )


def test_inner_join_hits_and_misses():
    duck_check(
        "SELECT k, name FROM __THIS__ JOIN dim ON k = dim.id",
        {"k": "int"},
        [{"k": 1}, {"k": 2}, {"k": 3}],
        {"dim": DIM},
    )


def test_left_join_miss_and_null_key():
    duck_check(
        "SELECT k, name FROM __THIS__ LEFT JOIN dim ON k = dim.id",
        {"k": "int?"},
        [{"k": 1}, {"k": 2}, {"k": None}],
        {"dim": DIM},
    )


def test_inner_join_null_key_drops():
    duck_check(
        "SELECT name FROM __THIS__ JOIN dim ON k = dim.id",
        {"k": "int?"},
        [{"k": 1}, {"k": None}],
        {"dim": DIM},
    )


def test_join_key_promotion_int_row_against_float_col():
    duck_check(
        "SELECT v FROM __THIS__ JOIN dim ON k = dim.id",
        {"k": "int"},
        [{"k": 1}, {"k": 2}],
        {"dim": static({"id": "float", "v": "int"}, [{"id": 1.0, "v": 10}])},
    )


def test_join_key_expression_and_where_interaction():
    duck_check(
        "SELECT k, name FROM __THIS__ JOIN dim ON k + 1 = dim.id WHERE k >= 0",
        {"k": "int"},
        [{"k": 0}, {"k": 2}, {"k": -5}],
        {"dim": DIM},
    )


def test_duplicate_build_keys_error():
    dup = static({"id": "int", "v": "int"}, [{"id": 1, "v": 10}, {"id": 1, "v": 11}])
    with pytest.raises(ValueError, match="duplicate map key"):
        DuckDBInferFn(
            "SELECT v FROM __THIS__ JOIN dim ON k = dim.id",
            row_tables={"__THIS__": _row_model({"k": "int"})},
            static_tables={"dim": dup},
        )


def test_null_in_value_column_errors():
    holed = static({"id": "int", "v": "int?"}, [{"id": 1, "v": None}])
    with pytest.raises(ValueError, match="NULL in value column"):
        DuckDBInferFn(
            "SELECT v FROM __THIS__ JOIN dim ON k = dim.id",
            row_tables={"__THIS__": _row_model({"k": "int"})},
            static_tables={"dim": holed},
        )


def test_null_key_build_rows_are_dropped():
    # A NULL build key never equi-matches, so the row is dropped rather than
    # rejected — probing k=1 still hits the valid entry.
    holed = static(
        {"id": "int?", "v": "int"}, [{"id": 1, "v": 10}, {"id": None, "v": 99}]
    )
    duck_check(
        "SELECT v FROM __THIS__ JOIN dim ON k = dim.id",
        {"k": "int"},
        [{"k": 1}],
        {"dim": holed},
    )


def test_unsupported_is_a_clean_value_error():
    with pytest.raises(ValueError, match="unsupported.*GROUP BY"):
        DuckDBInferFn(
            "SELECT a FROM __THIS__ GROUP BY a",
            row_tables={"__THIS__": _row_model({"a": "int"})},
            static_tables={},
        )


def test_bad_sql_is_a_build_error():
    with pytest.raises(ValueError, match="parse error"):
        DuckDBInferFn(
            "SELECT FROM",
            row_tables={"__THIS__": _row_model({"a": "int"})},
            static_tables={},
        )


def test_unknown_infer_table_is_rejected():
    model = _row_model({"a": "int"})
    fn = DuckDBInferFn(
        "SELECT a FROM __THIS__", row_tables={"__THIS__": model}, static_tables={}
    )
    with pytest.raises(ValueError, match="unknown table"):
        fn.infer({"wrong": [model(a=1)]})


def test_output_model_is_synthesized():
    fn = DuckDBInferFn(
        "SELECT a + 1 AS x, b AS y FROM __THIS__",
        row_tables={"__THIS__": _row_model({"a": "int", "b": "float?"})},
        static_tables={},
    )
    fields = fn.output_model.model_fields
    assert list(fields) == ["x", "y"]
    assert fields["x"].annotation is int
    assert fields["y"].annotation == float | None


# ------------------------------------------------------------- stretch 4:
# builtin catalogue, differential vs duckdb per the measured pins.


def test_string_builtins_differential():
    duck_check(
        "SELECT upper(s) AS u, lower(s) AS l, trim(s) AS t, ltrim(s, 'a') AS lt, "
        "rtrim(s) AS rt, substr(s, 2, 3) AS sub FROM __THIS__",
        {"s": "str?"},
        [{"s": "  aBc  "}, {"s": "abcdef"}, {"s": ""}, {"s": None}],
    )


def test_substr_edges_differential():
    duck_check(
        "SELECT substr(s, 0, 3) AS a, substr(s, -2) AS b, substr(s, -10, 8) AS c, "
        "substr(s, 1, 0) AS d, substr(s, 9) AS e FROM __THIS__",
        {"s": "str"},
        [{"s": "hello"}, {"s": "x"}],
    )


def test_concat_and_pipes_differential():
    duck_check(
        "SELECT n || '!' AS a, 'v=' || n AS b, concat(s, n, 'z') AS c, "
        "concat(s) AS d FROM __THIS__",
        {"n": "int?", "s": "str?"},
        [{"n": 1, "s": "a"}, {"n": None, "s": None}, {"n": -3, "s": ""}],
    )


def test_abs_round_differential():
    duck_check(
        "SELECT abs(n) AS an, abs(x) AS ax, round(x) AS rx, round(n) AS rn "
        "FROM __THIS__",
        {"n": "int?", "x": "float?"},
        [
            {"n": -5, "x": -2.5},
            {"n": 3, "x": 2.5},
            {"n": None, "x": None},
            {"n": 0, "x": -0.4},
        ],
    )


def test_rem_by_zero_differential():
    duck_check(
        "SELECT a % b AS r FROM __THIS__",
        {"a": "int", "b": "int"},
        [{"a": 5, "b": 0}, {"a": 5, "b": 3}, {"a": -7, "b": 2}],
    )


def test_float_rem_differential():
    duck_check(
        "SELECT x % y AS r FROM __THIS__",
        {"x": "float", "y": "float"},
        [{"x": -5.5, "y": 2.5}, {"x": 7.0, "y": 4.0}],
    )


def test_coalesce_nullif_differential():
    duck_check(
        "SELECT coalesce(n, 9) AS a, coalesce(NULL, n, 9) AS b, "
        "nullif(n, 3) AS c, nullif(s, 'x') AS d FROM __THIS__",
        {"n": "int?", "s": "str?"},
        [{"n": 3, "s": "x"}, {"n": None, "s": "y"}, {"n": 7, "s": None}],
    )


def test_nan_comparison_differential():
    # DuckDB DOUBLE order: NaN = NaN keeps the row.
    duck_check(
        "SELECT x FROM __THIS__ WHERE x = x",
        {"x": "float"},
        [{"x": float("nan")}, {"x": 1.5}],
    )


def test_simple_case_mapping_matches_duckdb():
    # Formerly a strict xfail: DuckDB uses utf8proc SIMPLE case maps, Rust
    # std only has full maps. src/specializer/exec/casemap.rs now carries the
    # measured exception table (see scripts/gen_casemap.py for why it exists
    # and why it is dependency-free).
    duck_check(
        "SELECT upper(s) AS u, lower(s) AS l FROM __THIS__",
        {"s": "str"},
        [{"s": "ß"}, {"s": "İ"}, {"s": "ᾀ"}, {"s": "ƛ"}],
    )


def test_simple_case_mapping_full_codepoint_census():
    # THE authority on casemap.rs: every Unicode scalar value, chunked into
    # long strings (per-codepoint mapping makes one string test them all),
    # through both engines. If a duckdb bump shifts utf8proc's tables, this
    # fails and scripts/gen_casemap.py regenerates the exception table.
    step = 0x8000
    rows = []
    for lo in range(1, 0x110000, step):
        s = "".join(
            chr(c)
            for c in range(lo, min(lo + step, 0x110000))
            if not (0xD800 <= c <= 0xDFFF)
        )
        if s:
            rows.append({"s": s})
    duck_check("SELECT upper(s) AS u, lower(s) AS l FROM __THIS__", {"s": "str"}, rows)


# --------------------------------------------- adversarial-fleet fixes:
# each case below was a measured divergence, now pinned differentially.


def test_trim_zs_set_differential():
    duck_check(
        "SELECT trim(s) AS t, ltrim(s) AS l, rtrim(s) AS r FROM __THIS__",
        {"s": "str"},
        [{"s": "\u00a0a\u3000"}, {"s": "\ta\n"}, {"s": " a "}, {"s": "\u2003a"}],
    )


def test_substr_negative_length_differential():
    duck_check(
        "SELECT substr(s, st, ln) AS r FROM __THIS__",
        {"s": "str", "st": "int", "ln": "int"},
        [
            {"s": "hello", "st": 3, "ln": -2},
            {"s": "hello", "st": 6, "ln": -5},
            {"s": "h\u00e9llo", "st": 4, "ln": -10},
            {"s": "hello", "st": 1, "ln": -1},
        ],
    )


def test_substr_negative_start_column_path_differential():
    # The TASK-45-review "parity bug" triples, through the path real queries
    # take (column input -> DuckDB's vectorized substr). Both engines agree:
    # negative start clamps to 1 BEFORE the length window (builtin-pins \u00a74).
    duck_check(
        "SELECT substr(s, st, ln) AS r FROM __THIS__",
        {"s": "str", "st": "int", "ln": "int"},
        [
            {"s": "ab", "st": -4, "ln": 2},
            {"s": "ab", "st": -3, "ln": 2},
            {"s": "ab", "st": -5, "ln": 3},
        ],
    )


@pytest.mark.xfail(
    strict=True,
    reason="DuckDB's constant-fold substr disagrees with its own vectorized "
    "path on negative starts (const substr('ab',-4,2)='', vectorized='ab'; "
    "measured 1.5.5). The engine pins the vectorized path, so pure-literal "
    "negative-start substr diverges \u2014 known residual, builtin-pins spec \u00a74",
)
def test_substr_constant_fold_divergence():
    duck_check(
        "SELECT substr('ab', -4, 2) AS a, substr('ab', -3, 2) AS b, "
        "substr('ab', -5, 3) AS c FROM __THIS__",
        {"s": "str"},
        [{"s": "x"}],
    )


def test_float_rendering_differential():
    duck_check(
        "SELECT x || '' AS s FROM __THIS__",
        {"x": "float"},
        [
            {"x": 1e300},
            {"x": 1e-05},
            {"x": float("nan")},
            {"x": 2.5},
            {"x": 1e16},
            {"x": -1e300},
        ],
    )


def test_null_divisor_differential():
    duck_check(
        "SELECT a % b AS r FROM __THIS__",
        {"a": "int?", "b": "int?"},
        [{"a": 7, "b": None}, {"a": None, "b": 0}, {"a": 7, "b": 3}],
    )


def test_numeric_where_differential():
    duck_check(
        "SELECT a FROM __THIS__ WHERE a % 2",
        {"a": "int"},
        [{"a": 1}, {"a": 2}, {"a": 3}],
    )


def test_nan_filter_differential_on_native_tables():
    duck_check(
        "SELECT x FROM __THIS__ WHERE x > 0",
        {"x": "float?"},
        [{"x": float("nan")}, {"x": 1.0}, {"x": None}, {"x": float("inf")}],
    )


# ------------------------------------------------- static-only queries:
# AC #2 — evaluated once at build time by DuckDB, nothing dynamic remains.


def test_static_only_query_is_a_constant_emitter():
    dim = static(
        {"id": "int", "name": "str"},
        [
            {"id": 1, "name": "one"},
            {"id": 2, "name": "two"},
            {"id": 3, "name": "three"},
        ],
    )
    model = _row_model({"a": "int"})
    fn = DuckDBInferFn(
        "SELECT name, id * 10 AS x FROM dim WHERE id <> 2 ORDER BY id DESC",
        row_tables={"__THIS__": model},
        static_tables={"dim": dim},
    )
    # Input rows are irrelevant; the result is fixed at build time —
    # and constructs like ORDER BY work because DuckDB itself evaluated it.
    for rows_in in ([], [model(a=1)], [model(a=1), model(a=2)]):
        got = [r.model_dump() for r in fn.infer({"__THIS__": rows_in})]
        assert got == [
            {"name": "three", "x": 30},
            {"name": "one", "x": 10},
        ]


def test_static_only_aggregation_works_via_duckdb():
    dim = static({"v": "int"}, [{"v": 1}, {"v": 2}, {"v": 3}])
    fn = DuckDBInferFn(
        "SELECT sum(v) AS s FROM dim",
        row_tables={"__THIS__": _row_model({"a": "int"})},
        static_tables={"dim": dim},
    )
    assert [r.model_dump() for r in fn.infer({"__THIS__": []})] == [{"s": 6}]


def test_unknown_driving_table_stays_clean_unsupported():
    # Not a static table either -> the original clean unsupported surfaces.
    with pytest.raises(ValueError, match="driving relation"):
        DuckDBInferFn(
            "SELECT x FROM nope",
            row_tables={"__THIS__": _row_model({"a": "int"})},
            static_tables={},
        )


# --------------------------------------------------------- M-cranelift:
# the codegen backend runs the same suite; these pin the backend choice.


def test_v0_queries_run_on_cranelift():
    fn = DuckDBInferFn(
        "SELECT k + 1 AS x, upper(s) AS u FROM __THIS__ JOIN dim ON k = dim.id",
        row_tables={"__THIS__": _row_model({"k": "int", "s": "str"})},
        static_tables={"dim": DIM},
    )
    assert fn.backend == "cranelift"


def test_static_only_backend_is_constant():
    fn = DuckDBInferFn(
        "SELECT sum(v) AS s FROM dim",
        row_tables={"__THIS__": _row_model({"a": "int"})},
        static_tables={"dim": static({"v": "int"}, [{"v": 1}, {"v": 2}])},
    )
    assert fn.backend == "constant"


# --------------------------------------------------------- M-boundary:
# the generated row marshaller. These pin the adversarial-review fixes
# (2026-07-26): supplied output models keep full pydantic semantics, the
# generic baseline accepts the same inputs, and reentrancy degrades to the
# generic path instead of erroring.


def test_supplied_output_model_keeps_validate_semantics():
    from pydantic import BaseModel, field_validator

    class Out(BaseModel):
        x: float  # engine emits int; validate coerces
        note: str = "default"  # not in the projection; validate fills

        @field_validator("x")
        @classmethod
        def clamp(cls, v):
            return min(v, 10.0)

    fn = DuckDBInferFn(
        "SELECT k + 1 AS x FROM __THIS__",
        row_tables={"__THIS__": _row_model({"k": "int"})},
        static_tables={},
        output_model=Out,
    )
    assert fn.boundary == "marshaller"
    (m,) = fn.infer_rows([{"k": 41}])
    assert m.x == 10.0  # validator ran AND coerced to float
    assert m.note == "default"  # default applied
    assert m.model_dump() == {"x": 10.0, "note": "default"}


def test_generic_boundary_accepts_dict_rows(monkeypatch):
    monkeypatch.setenv("SPECIALIZER_GENERIC_BOUNDARY", "1")
    fn = DuckDBInferFn(
        "SELECT k * 2 AS d FROM __THIS__",
        row_tables={"__THIS__": _row_model({"k": "int"})},
        static_tables={},
    )
    assert fn.boundary == "generic"
    assert fn.infer_rows([{"k": 21}])[0].d == 42


def test_reentrant_infer_falls_back_instead_of_erroring():
    fn = DuckDBInferFn(
        "SELECT k * 2 AS d FROM __THIS__",
        row_tables={"__THIS__": _row_model({"k": "int"})},
        static_tables={},
    )
    inner: list = []

    class Row:
        @property
        def k(self):
            if not inner:
                inner.append(fn.infer_rows([{"k": 5}])[0].d)
            return 7

    (m,) = fn.infer_rows([Row()])
    assert m.d == 14
    assert inner == [10]  # the nested call completed via the generic path
